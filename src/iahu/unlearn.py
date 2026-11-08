"""Visible-input plan builders and staged selective unlearning (S6).

This module accepts no evaluator truth, vault, attack manifest, or clean reference.
Its staged driver uses only a :class:`VisibleExperiment` assembled by the offline
runner from the poisoned visible world.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .data import Interaction
from .flagging import VisibleFlagSet
from .hypergraph import Hypergraph, build_hour_inspired
from .influence import (
    TorchHypergraphOperator,
    blend_suspicion_scores,
    estimate_source_influence,
    exposure_surrogate,
    overlap_proximity,
    polish_frozen_negative,
)
from .model import BPRConfig, TrainedModel, fit_bpr


@dataclass(frozen=True, slots=True)
class UnlearnBudget:
    extension_ratio: float = 1.0
    minimum_weight: float = 0.05
    flagged_weight: float = 0.2
    extension_strength: float = 1.0

    def validate(self) -> None:
        values = (self.extension_ratio, self.minimum_weight, self.flagged_weight, self.extension_strength)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("unlearning budget values must be finite")
        if self.extension_ratio < 0:
            raise ValueError("extension_ratio must be non-negative")
        if not 0 < self.minimum_weight <= self.flagged_weight <= 1:
            raise ValueError("weights must satisfy 0 < minimum_weight <= flagged_weight <= 1")
        if not 0 < self.extension_strength <= 1:
            raise ValueError("extension_strength must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class UnlearnPlan:
    operator: str
    flagged_users: frozenset[int]
    extension_users: tuple[int, ...]
    weight_map: tuple[tuple[int, float], ...]

    def weights_for(self, user_ids: tuple[int, ...]) -> tuple[float, ...]:
        mapping = dict(self.weight_map)
        unknown = set(mapping) - set(user_ids)
        if unknown:
            raise ValueError("plan references users outside the visible training world")
        return tuple(mapping.get(user_id, 1.0) for user_id in user_ids)


@dataclass(frozen=True, slots=True)
class VisibleExperiment:
    """The complete input boundary for deployable unlearning.

    Rows and validation rows contain only pseudonymous visible users. The frozen
    negatives are generated from visible training rows. There are deliberately
    no fields for hidden attackers, evaluator labels, references, or manifests.
    """

    graph: Hypergraph
    model: TrainedModel
    training_rows: tuple[Interaction, ...]
    validation_rows: tuple[Interaction, ...]
    flags: VisibleFlagSet
    target_item_ids: tuple[int, ...]
    negative_item_indices: object
    positive_user_indices: object
    positive_item_indices: object


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: int
    accepted: bool
    reason: str
    extension_count: int
    extension_mass: float
    utility_retention: float | None
    elapsed_seconds: float
    influence_reliable: bool


@dataclass(frozen=True, slots=True)
class StagedUnlearnResult:
    model: TrainedModel
    source_weights: tuple[float, ...]
    flagged_count: int
    extension_count: int
    selected_extension_user_ids: tuple[int, ...]
    stages: tuple[StageRecord, ...]
    recovery_epochs: int
    refusal_reason: str | None
    elapsed_seconds: float
    staged_exposure_change: float | None = None
    final_influence_reliable: bool = False
    final_exposure_surrogate: float | None = None
    final_exposure_excluding_extensions: float | None = None
    frozen_exposure_cohort_size: int = 0


@dataclass(frozen=True, slots=True)
class StagedUnlearnConfig:
    operator: str = "removal"
    hour_policy: str = "frozen"
    extension_ratio: float = 1.0
    mass_cap: float = 0.01
    utility_tolerance: float = 0.02
    path_steps: int = 4
    minimum_weight: float = 0.05
    flagged_weight: float = 0.2
    extension_strength: float = 1.0
    overlap_weight: float = 0.5
    stage_epochs: int = 5
    recovery_epochs: int = 50
    recovery_patience: int = 5
    negative_samples: int = 1
    temperature: float | None = None
    damping: float = 0.01
    polish_relative_tolerance: float = 1e-3
    polish_max_iterations: int = 200
    cg_iterations: int = 100
    cg_tolerance: float = 1e-3
    seed: int = 0

    def validate(self) -> None:
        if self.operator not in {"removal", "reweighting"}:
            raise ValueError("operator must be removal or reweighting")
        if self.hour_policy not in {"frozen", "rebuilt"}:
            raise ValueError("hour_policy must be frozen or rebuilt")
        if self.extension_ratio < 0 or not math.isfinite(self.extension_ratio):
            raise ValueError("extension_ratio must be finite and non-negative")
        if not 0 <= self.mass_cap <= 1 or not math.isfinite(self.mass_cap):
            raise ValueError("mass_cap must be finite and in [0, 1]")
        if not 0 <= self.utility_tolerance < 1 or not math.isfinite(self.utility_tolerance):
            raise ValueError("utility_tolerance must be finite and in [0, 1)")
        if self.path_steps < 1 or self.stage_epochs < 1 or self.recovery_epochs < 0:
            raise ValueError("path_steps and stage_epochs must be positive; recovery_epochs non-negative")
        if self.recovery_patience < 1 or self.negative_samples < 1:
            raise ValueError("recovery_patience and negative_samples must be positive")
        if not 0 <= self.overlap_weight <= 1:
            raise ValueError("overlap_weight must be in [0, 1]")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("temperature must be positive and finite")


def build_plan(
    flags: VisibleFlagSet,
    suspicion_scores: Mapping[int, float],
    budget: UnlearnBudget,
    *,
    operator: str,
    seed: int,
    extension_limit: int | None = None,
) -> UnlearnPlan:
    """Build removal or reweighting from visible flags and public suspicion scores only."""
    budget.validate()
    if operator not in {"removal", "reweighting"}:
        raise ValueError("operator must be removal or reweighting")
    flag_map = {flag.user_id: flag.confidence for flag in flags.flags}
    if len(flag_map) != len(flags.flags):
        raise ValueError("visible flags contain duplicate users")
    if not flag_map:
        raise ValueError("InsufficientFlags: unlearner refuses an empty flag set")
    if any(user < 0 or not 0 <= confidence <= 1 for user, confidence in flag_map.items()):
        raise ValueError("flag user IDs and confidences are invalid")
    for user, score in suspicion_scores.items():
        if isinstance(user, bool) or not isinstance(user, int) or user < 0:
            raise ValueError("suspicion scores must use non-negative integer user IDs")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("suspicion scores must be finite and in [0, 1]")

    candidates = {user: score for user, score in suspicion_scores.items()
                  if user not in flag_map and score > 0}
    max_extension = math.ceil(budget.extension_ratio * len(flag_map))
    if extension_limit is not None:
        if extension_limit < 0:
            raise ValueError("extension_limit must be non-negative")
        max_extension = min(max_extension, extension_limit)
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    tie_order = {user: index for index, user in enumerate(shuffled)}
    selected = sorted(candidates, key=lambda user: (-candidates[user], tie_order[user]))[:max_extension]

    if operator == "removal":
        weights = {user: 0.0 for user in (*flag_map.keys(), *selected)}
    else:
        weights = {
            user: 1.0 - confidence * (1.0 - budget.flagged_weight)
            for user, confidence in flag_map.items()
        }
        weights.update({
            user: max(budget.minimum_weight, 1.0 - budget.extension_strength * candidates[user])
            for user in selected
        })
    return UnlearnPlan(
        operator=operator,
        flagged_users=frozenset(flag_map),
        extension_users=tuple(selected),
        weight_map=tuple(sorted(weights.items())),
    )


def _ranking_recall(
    model: TrainedModel,
    validation_rows: Sequence[Interaction],
    *,
    excluded_users: set[int],
    target_item_ids: set[int],
    k: int = 20,
) -> float | None:
    """Visible Recall@K on V, excluding target rows but retaining V items as candidates."""
    import numpy as np

    relevant: dict[int, set[int]] = {}
    for row in validation_rows:
        if row.user_id in excluded_users:
            continue
        if row.item_id not in target_item_ids:
            relevant.setdefault(row.user_id, set()).add(row.item_id)
    users = [user for user in sorted(relevant) if relevant[user]]
    if not users:
        return None
    item_index = model.item_index
    encoded = model.embeddings()
    item_vectors = encoded[model.operator.num_users :]
    recalls: list[float] = []
    for user_id in users:
        if user_id not in model.user_index:
            continue
        user_position = model.user_index[user_id]
        # V positives are the relevance labels; they must remain candidates.
        seen = set(model.history.get(user_id, frozenset()))
        candidates = [
            position for position, item_id in enumerate(model.item_ids)
            if item_id not in seen
        ]
        if not candidates:
            continue
        scores = item_vectors[candidates] @ encoded[user_position]
        ranking = sorted(
            ((model.item_ids[position], float(score)) for position, score in zip(candidates, scores.tolist())),
            key=lambda pair: (-pair[1], pair[0]),
        )[:k]
        recalls.append(len({item for item, _ in ranking} & relevant[user_id]) / len(relevant[user_id]))
    return float(np.mean(recalls)) if recalls else None


def _copy_model_with_theta(model: TrainedModel, theta) -> TrainedModel:
    import numpy as np

    values = theta.detach().cpu().numpy().astype(np.float32, copy=True)
    return TrainedModel(
        operator=model.operator,
        propagation=model.propagation,
        user_ids=model.user_ids,
        item_ids=model.item_ids,
        theta=values,
        config=model.config,
        history=model.history,
        elapsed_seconds=model.elapsed_seconds,
        epoch_losses=model.epoch_losses,
    )


def _fixed_influence_state(
    experiment: VisibleExperiment,
    model: TrainedModel,
    weights,
    *,
    pair_users,
    pair_items,
    pair_thresholds,
    temperature: float,
    config: StagedUnlearnConfig,
):
    """Polish and score sources using only visible experiment inputs."""
    import torch
    graph = experiment.graph
    theta = torch.as_tensor(model.theta, dtype=torch.float64, device="cpu")
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    polished = polish_frozen_negative(
        theta,
        graph,
        weight_tensor,
        experiment.positive_user_indices,
        experiment.positive_item_indices,
        experiment.negative_item_indices,
        layers=model.config.layers,
        regularization=model.config.regularization,
        relative_tolerance=config.polish_relative_tolerance,
        max_iterations=config.polish_max_iterations,
    )
    current = _copy_model_with_theta(model, polished.theta)
    influence = estimate_source_influence(
        polished.theta,
        graph,
        weight_tensor,
        experiment.positive_user_indices,
        experiment.positive_item_indices,
        experiment.negative_item_indices,
        pair_users,
        pair_items,
        pair_thresholds,
        layers=model.config.layers,
        regularization=model.config.regularization,
        temperature=temperature,
        damping=config.damping,
        cg_iterations=config.cg_iterations,
        cg_tolerance=config.cg_tolerance,
        stationarity_tolerance=config.polish_relative_tolerance,
        initial_gradient_norm=polished.initial_gradient_norm,
        polished=True,
    )
    flags = [(flag.user_id, flag.confidence) for flag in experiment.flags.flags]
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    overlap = overlap_proximity(graph, weight_tensor, flags, layers=model.config.layers)
    scores = blend_suspicion_scores(
        influence.source_scores,
        graph,
        overlap,
        {user for user, _ in flags},
        overlap_weight=config.overlap_weight,
    )
    reliable = (
        polished.converged
        and influence.stationary
        and not influence.negative_curvature
        and influence.cg_relative_residual <= config.cg_tolerance
    )
    return current, scores, reliable, influence.source_scores


def _frozen_cohort_exposure(
    model: TrainedModel,
    source_weights,
    cohort_pairs: Sequence[tuple[int, int, float]],
    *,
    temperature: float,
    excluded_users: set[int] | None = None,
) -> float | None:
    """Evaluate the frozen exposure surrogate without changing its thresholds."""
    import torch

    excluded_users = excluded_users or set()
    graph = model.operator
    users = model.user_index
    items = model.item_index
    selected = [
        (user, item, threshold)
        for user, item, threshold in cohort_pairs
        if user not in excluded_users and user in users and item in items
    ]
    if not selected:
        return None
    pair_users = torch.tensor([users[user] for user, _, _ in selected], dtype=torch.long)
    pair_items = torch.tensor([items[item] for _, item, _ in selected], dtype=torch.long)
    thresholds = torch.tensor([threshold for _, _, threshold in selected], dtype=torch.float64)
    theta = torch.as_tensor(model.theta, dtype=torch.float64, device="cpu")
    weights = torch.as_tensor(source_weights, dtype=torch.float64, device="cpu")
    operator = TorchHypergraphOperator(graph, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        value = exposure_surrogate(
            theta,
            operator,
            weights,
            pair_users,
            pair_items,
            thresholds,
            temperature=temperature,
            layers=model.config.layers,
        )
    return float(value.item())


def run_staged_unlearning(
    experiment: VisibleExperiment,
    *,
    config: StagedUnlearnConfig | None = None,
) -> StagedUnlearnResult:
    """Run staged influence-aware source removal/reweighting with guard rollback.

    The user cohort, exposure thresholds, and frozen negatives are prepared once
    from the visible poisoned state. Each committed stage updates weights and
    warm-starts the model. A failed mass or visible-utility guard stops the run.

    For each committed stage, ``staged_exposure_change`` approximates
    ``-∫ Σ_u R_u(c) dc_u`` with a trapezoid between the polished influence
    estimates at the stage endpoints. Here ``c_u`` is a source weight and
    ``R_u`` is the derivative of the frozen-cohort exposure surrogate with
    respect to that weight. This assumes each accepted weight change follows a
    straight segment and both endpoint estimates satisfy the configured
    stationarity and CG checks; the fixed cohort and detached thresholds never
    change along this path.
    """
    import time
    import numpy as np
    import torch

    config = config or StagedUnlearnConfig()
    config.validate()
    started = time.perf_counter()
    graph = experiment.graph
    flags = experiment.flags
    if not flags.flags:
        return StagedUnlearnResult(
            experiment.model, tuple(1.0 for _ in graph.user_ids), 0, 0, (), (), 0,
            "InsufficientFlags", time.perf_counter() - started,
        )
    if not experiment.target_item_ids:
        return StagedUnlearnResult(
            experiment.model, tuple(1.0 for _ in graph.user_ids), len(flags.flags), 0, (), (), 0,
            "NoObservedTargets", time.perf_counter() - started,
        )

    user_index = {user: index for index, user in enumerate(graph.user_ids)}
    item_index = {item: index for index, item in enumerate(graph.item_ids)}
    if any(flag.user_id not in user_index for flag in flags.flags):
        raise ValueError("visible flag set references a user outside the visible graph")
    histories: dict[int, set[int]] = {user: set() for user in graph.user_ids}
    for row in experiment.training_rows:
        histories[row.user_id].add(row.item_id)

    flagged = {flag.user_id for flag in flags.flags}
    target_ids = set(experiment.target_item_ids)
    eligible_users = [
        user for user in graph.user_ids
        if user not in flagged and not (histories.get(user, set()) & target_ids)
    ]
    if not eligible_users:
        return StagedUnlearnResult(
            experiment.model, tuple(1.0 for _ in graph.user_ids), len(flags.flags), 0, (), (), 0,
            "UndefinedExposureCohort", time.perf_counter() - started,
        )
    weights = np.ones(graph.num_users, dtype=np.float32)
    initial_theta = torch.as_tensor(experiment.model.theta, dtype=torch.float64, device="cpu")
    initial_polish = polish_frozen_negative(
        initial_theta,
        graph,
        torch.as_tensor(weights, dtype=torch.float64),
        experiment.positive_user_indices,
        experiment.positive_item_indices,
        experiment.negative_item_indices,
        layers=experiment.model.config.layers,
        regularization=experiment.model.config.regularization,
        relative_tolerance=config.polish_relative_tolerance,
        max_iterations=config.polish_max_iterations,
    )
    initial = _copy_model_with_theta(experiment.model, initial_polish.theta)
    initial_embeddings = initial.embeddings()
    initial_item_embeddings = initial_embeddings[graph.num_users :]
    target_scores: list[float] = []
    thresholds: list[float] = []
    pair_user_positions: list[int] = []
    pair_item_positions: list[int] = []
    for user_id in eligible_users:
        scores = initial_item_embeddings @ initial_embeddings[user_index[user_id]]
        candidate_scores = [
            float(scores[position]) for position, item_id in enumerate(graph.item_ids)
            if item_id not in histories[user_id]
        ]
        if not candidate_scores:
            continue
        threshold = sorted(candidate_scores, reverse=True)[min(19, len(candidate_scores) - 1)]
        target_score = float(scores[item_index[experiment.target_item_ids[0]]])
        thresholds.append(threshold)
        target_scores.append(target_score)
        pair_user_positions.append(user_index[user_id])
        pair_item_positions.append(item_index[experiment.target_item_ids[0]])
    if not thresholds:
        return StagedUnlearnResult(
            experiment.model, tuple(1.0 for _ in graph.user_ids), len(flags.flags), 0, (), (), 0,
            "UndefinedExposureCohort", time.perf_counter() - started,
        )
    deltas = torch.tensor(target_scores, dtype=torch.float64) - torch.tensor(thresholds, dtype=torch.float64)
    temperature = config.temperature or float(torch.std(deltas, unbiased=False).item())
    if not math.isfinite(temperature) or temperature <= 1e-12:
        return StagedUnlearnResult(
            experiment.model, tuple(1.0 for _ in graph.user_ids), len(flags.flags), 0, (), (), 0,
            "SaturatedExposure", time.perf_counter() - started,
        )
    pair_users = torch.tensor(pair_user_positions, dtype=torch.long)
    pair_items = torch.tensor(pair_item_positions, dtype=torch.long)
    pair_thresholds = torch.tensor(thresholds, dtype=torch.float64)
    frozen_cohort_pairs = tuple(
        (graph.user_ids[user_position], graph.item_ids[item_position], threshold)
        for user_position, item_position, threshold in zip(
            pair_user_positions, pair_item_positions, thresholds
        )
    )

    current = initial
    stage_records: list[StageRecord] = []
    extension_users: set[int] = set()
    staged_exposure_change = 0.0
    pending_path_step: tuple[set[int], np.ndarray, np.ndarray, tuple[float, ...]] | None = None
    budget = UnlearnBudget(
        extension_ratio=config.extension_ratio,
        minimum_weight=config.minimum_weight,
        flagged_weight=config.flagged_weight,
        extension_strength=config.extension_strength,
    )
    counts = {user: 0 for user in graph.user_ids}
    for row in experiment.training_rows:
        counts[row.user_id] = counts.get(row.user_id, 0) + 1
    denominator = max(1, len(experiment.training_rows))

    control_config = BPRConfig(
        dimension=current.config.dimension,
        layers=current.config.layers,
        learning_rate=current.config.learning_rate,
        regularization=current.config.regularization,
        batch_size=current.config.batch_size,
        epochs=config.stage_epochs,
    )
    control = fit_bpr(
        graph, experiment.training_rows, config=control_config, seed=config.seed,
        source_weights=weights, initial_theta=current.theta,
    )

    max_extension = math.ceil(config.extension_ratio * len(flags.flags))
    per_stage_limit = math.ceil(max_extension / config.path_steps) if max_extension else 0
    refusal_reason = None
    for stage in range(1, config.path_steps + 1):
        stage_started = time.perf_counter()
        current, suspicion, reliable, raw_source_scores = _fixed_influence_state(
            experiment,
            current,
            weights,
            pair_users=pair_users,
            pair_items=pair_items,
            pair_thresholds=pair_thresholds,
            temperature=temperature,
            config=config,
        )
        if pending_path_step is not None:
            changed_sources, _before_weights, after_weights, before_scores = pending_path_step
            staged_exposure_change -= sum(
                (1.0 - float(after_weights[user_index[user]]))
                * 0.5
                * (before_scores[user_index[user]] + raw_source_scores[user_index[user]])
                for user in changed_sources
            )
            pending_path_step = None
        remaining = max(0, max_extension - len(extension_users))
        requested = min(per_stage_limit, remaining)
        stage_scores = {
            user: score for user, score in suspicion.items()
            if user not in extension_users
        }
        plan = build_plan(
            flags,
            stage_scores,
            budget,
            operator=config.operator,
            seed=config.seed + stage,
            extension_limit=requested,
        )
        new_extensions = set(plan.extension_users)
        proposed_extensions = extension_users | new_extensions
        tentative_weights = weights.copy()
        if stage == 1:
            # The first operator action includes all visible flags and its first extension.
            tentative_weights = np.asarray(plan.weights_for(graph.user_ids), dtype=np.float32)
        elif config.operator == "removal":
            for user in new_extensions:
                tentative_weights[user_index[user]] = 0.0
        else:
            for user in new_extensions:
                tentative_weights[user_index[user]] = max(
                    config.minimum_weight,
                    1.0 - config.extension_strength * suspicion.get(user, 0.0),
                )
        mass = sum(
            (1.0 - float(tentative_weights[user_index[user]])) * counts.get(user, 0)
            for user in proposed_extensions
        ) / denominator
        if mass > config.mass_cap:
            stage_records.append(StageRecord(
                stage, False, "mass_cap", len(extension_users), mass, None,
                time.perf_counter() - stage_started, reliable,
            ))
            refusal_reason = "MassCap"
            break

        tentative = fit_bpr(
            graph,
            experiment.training_rows,
            config=control_config,
            seed=config.seed + stage,
            source_weights=tentative_weights,
            initial_theta=current.theta,
        )
        guard_users = flagged | proposed_extensions
        control_recall = _ranking_recall(
            control, experiment.validation_rows, excluded_users=guard_users,
            target_item_ids=target_ids,
        )
        candidate_recall = _ranking_recall(
            tentative, experiment.validation_rows, excluded_users=guard_users,
            target_item_ids=target_ids,
        )
        retention = (
            candidate_recall / control_recall
            if control_recall is not None and control_recall > 0 and candidate_recall is not None
            else None
        )
        if retention is None or retention < 1.0 - config.utility_tolerance:
            stage_records.append(StageRecord(
                stage, False, "utility_guard", len(extension_users), mass, retention,
                time.perf_counter() - stage_started, reliable,
            ))
            refusal_reason = "UtilityGuard"
            break

        # Commit atomically only after both guards accept the candidate.
        current = tentative
        previous_weights = weights.copy()
        weights = tentative_weights
        extension_users = proposed_extensions
        changed_sources = (flagged | new_extensions) if stage == 1 else new_extensions
        pending_path_step = (
            changed_sources,
            previous_weights,
            weights.copy(),
            raw_source_scores,
        )
        stage_records.append(StageRecord(
            stage, True, "committed", len(extension_users), mass, retention,
            time.perf_counter() - stage_started, reliable,
        ))

    final_influence_reliable = True
    if pending_path_step is not None:
        current, _final_scores, final_influence_reliable, final_source_scores = _fixed_influence_state(
            experiment,
            current,
            weights,
            pair_users=pair_users,
            pair_items=pair_items,
            pair_thresholds=pair_thresholds,
            temperature=temperature,
            config=config,
        )
        changed_sources, _before_weights, after_weights, before_scores = pending_path_step
        staged_exposure_change -= sum(
            (1.0 - float(after_weights[user_index[user]]))
            * 0.5
            * (before_scores[user_index[user]] + final_source_scores[user_index[user]])
            for user in changed_sources
        )
        pending_path_step = None

    # HI's rebuilt policy performs its single factorization only after selection.
    if graph.construction == "hour_inspired" and config.hour_policy == "rebuilt":
        graph = build_hour_inspired(
            experiment.training_rows,
            seed=config.seed,
            rank=min(64, graph.num_users, graph.num_items),
            source_weights=weights,
            policy="rebuilt",
        )

    recovery_epochs = 0
    if config.recovery_epochs:
        best_model = current
        excluded = flagged | extension_users
        best_recall = _ranking_recall(
            best_model, experiment.validation_rows, excluded_users=excluded,
            target_item_ids=target_ids,
        )
        stale = 0
        recovery_config = BPRConfig(
            dimension=current.config.dimension,
            layers=current.config.layers,
            learning_rate=current.config.learning_rate,
            regularization=current.config.regularization,
            batch_size=current.config.batch_size,
            epochs=1,
        )
        for epoch in range(config.recovery_epochs):
            candidate = fit_bpr(
                graph,
                experiment.training_rows,
                config=recovery_config,
                seed=config.seed + 1000 + epoch,
                source_weights=weights,
                initial_theta=best_model.theta,
            )
            recovery_epochs += 1
            score = _ranking_recall(
                candidate, experiment.validation_rows, excluded_users=excluded,
                target_item_ids=target_ids,
            )
            if score is not None and (best_recall is None or score > best_recall):
                best_model = candidate
                best_recall = score
                stale = 0
            else:
                stale += 1
            if stale >= config.recovery_patience:
                break
        current = best_model

    final_exposure = _frozen_cohort_exposure(
        current,
        weights,
        frozen_cohort_pairs,
        temperature=temperature,
    )
    final_exposure_without_extensions = _frozen_cohort_exposure(
        current,
        weights,
        frozen_cohort_pairs,
        temperature=temperature,
        excluded_users=extension_users,
    )

    return StagedUnlearnResult(
        current,
        tuple(float(value) for value in weights.tolist()),
        len(flags.flags),
        len(extension_users),
        tuple(sorted(extension_users)),
        tuple(stage_records),
        recovery_epochs,
        refusal_reason,
        time.perf_counter() - started,
        staged_exposure_change,
        final_influence_reliable,
        final_exposure,
        final_exposure_without_extensions,
        len({user for user, _, _ in frozen_cohort_pairs}),
    )

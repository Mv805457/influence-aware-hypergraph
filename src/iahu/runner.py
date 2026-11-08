"""Reproducible offline MovieLens experiment orchestration.

The browser demo never imports this module. This CLI owns dataset ingestion,
poisoning, training, unlearning, evaluator-only metrics, and run manifests.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .attack import inject_fake_accounts
from .data import (
    DatasetError,
    Interaction,
    binarize_and_k_core,
    load_dataset,
    select_targets,
    split_per_user,
)
from .evaluation import EvaluationResult, evaluate_full_ranking
from .gates import GateResult, attack_strength_gate, clean_model_gate
from .hypergraph import Hypergraph, build_hour_inspired, build_user_history
from .model import BPRConfig, LightGCNOperator, TrainedModel, fit_bpr
from .unlearn import (
    StagedUnlearnConfig,
    VisibleExperiment,
    run_staged_unlearning,
)
from .flagging import simulate_flagging
from .validation import validate_v1_convex_influence


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
GATES = RESULTS / "gates"
PREREG = RESULTS / "preregistration"
COMPLETED_EXPORTS = RESULTS / "completed-run-exports"


@dataclass(frozen=True, slots=True)
class SeedStreams:
    world: int
    attack: int
    flagging: int
    training: int
    unlearning: int


def spawn_seed_streams(parent_seed: int) -> SeedStreams:
    """Create five independent deterministic child seeds from one parent."""
    children = np.random.SeedSequence(parent_seed).spawn(5)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return SeedStreams(*values)


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _runtime_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "torch"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _operator(
    construction: str,
    rows: Sequence[Interaction],
    *,
    seed: int,
    rank: int = 64,
    policy: str = "frozen",
) -> Hypergraph:
    if construction == "user_history":
        return build_user_history(rows)
    if construction == "hour_inspired":
        n_users = len({row.user_id for row in rows})
        n_items = len({row.item_id for row in rows})
        return build_hour_inspired(
            rows,
            seed=seed,
            rank=min(rank, n_users, n_items),
            policy=policy,
        )
    raise ValueError("construction must be user_history or hour_inspired")


def _config_for(model: TrainedModel, *, epochs: int | None = None) -> BPRConfig:
    old = model.config
    return BPRConfig(
        dimension=old.dimension,
        layers=old.layers,
        learning_rate=old.learning_rate,
        regularization=old.regularization,
        batch_size=old.batch_size,
        epochs=epochs or old.epochs,
    )


def _visible_split(
    rows: Sequence[Interaction], *, seed: int, fraction: float = 0.05
) -> tuple[tuple[Interaction, ...], tuple[Interaction, ...], tuple[int, ...]]:
    """Uniform row split after injection; the returned indices stay evaluator-side."""
    if not 0 <= fraction < 1:
        raise ValueError("visible validation fraction must be in [0, 1)")
    count = min(max(1, round(fraction * len(rows))), max(0, len(rows) - 1))
    selected = tuple(sorted(np.random.default_rng(seed).choice(len(rows), size=count, replace=False).tolist())) if count else ()
    selected_set = set(selected)
    visible = tuple(row for index, row in enumerate(rows) if index not in selected_set)
    validation = tuple(row for index, row in enumerate(rows) if index in selected_set)
    return visible, validation, selected


def _map_benign(rows: Sequence[Interaction], mapping: dict[int, int]) -> tuple[Interaction, ...]:
    return tuple(
        Interaction(mapping[row.user_id], row.item_id, row.rating, row.timestamp)
        for row in rows
    )


def _train_model(
    construction: str,
    rows: Sequence[Interaction],
    *,
    seed: int,
    config: BPRConfig,
    policy: str = "frozen",
) -> TrainedModel:
    graph = _operator(construction, rows, seed=seed, policy=policy)
    return fit_bpr(graph, rows, config=config, seed=seed)


def _metric(evaluation: EvaluationResult, name: str) -> float:
    value = evaluation.metrics.get(name)
    return float(value) if value is not None else 0.0


def _gate_json(gate: GateResult) -> dict[str, Any]:
    return {
        "passed": gate.passed,
        "checks": gate.checks,
        "observed": gate.observed,
        "reasons": list(gate.reasons),
    }


def _gradient_check() -> float:
    """Float64 finite-difference check of the differentiable frozen loss."""
    import torch
    from .influence import TorchHypergraphOperator, frozen_negative_objective

    rows = (
        Interaction(1, 1, 5, None), Interaction(1, 2, 5, None),
        Interaction(2, 2, 5, None), Interaction(2, 3, 5, None),
        Interaction(3, 1, 5, None), Interaction(3, 4, 5, None),
    )
    graph = build_user_history(rows)
    op = TorchHypergraphOperator(graph, dtype=torch.float64, device="cpu")
    users = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    positive = torch.tensor([0, 1, 1, 2, 0, 3], dtype=torch.long)
    negatives = torch.tensor([[2], [3], [0], [3], [1], [2]], dtype=torch.long)
    weights = torch.ones(graph.num_users, dtype=torch.float64)
    theta = torch.linspace(-0.13, 0.19, graph.num_vertices * 3, dtype=torch.float64).reshape(graph.num_vertices, 3)
    theta.requires_grad_(True)
    loss = frozen_negative_objective(
        theta, op, weights, users, positive, negatives, layers=2, regularization=1e-4,
    )
    gradient = torch.autograd.grad(loss, theta)[0].detach()
    h = 1e-6
    errors = []
    for flat_index in range(theta.numel()):
        plus = theta.detach().clone()
        minus = theta.detach().clone()
        plus.reshape(-1)[flat_index] += h
        minus.reshape(-1)[flat_index] -= h
        plus_value = frozen_negative_objective(
            plus, op, weights, users, positive, negatives, layers=2, regularization=1e-4,
        ).item()
        minus_value = frozen_negative_objective(
            minus, op, weights, users, positive, negatives, layers=2, regularization=1e-4,
        ).item()
        finite_difference = (plus_value - minus_value) / (2 * h)
        observed = gradient.reshape(-1)[flat_index].item()
        errors.append(abs(finite_difference - observed) / max(1e-12, abs(finite_difference) + abs(observed)))
    return max(errors, default=0.0)


def select_fixed_epochs(
    archive_path: Path,
    *,
    dataset: str = "ml-100k",
    parent_seed: int = 20261108,
    development_worlds: int = 5,
    max_epochs: int = 20,
) -> dict[str, Any]:
    """Choose E* once on development worlds using benign validation NDCG@20."""
    if dataset != "ml-100k":
        raise ValueError("E* selection is restricted to MovieLens-100K development worlds")
    if development_worlds < 2 or max_epochs < 1:
        raise ValueError("at least two development worlds and a positive epoch limit are required")
    data = load_dataset(archive_path, expected_name=dataset)
    rows = binarize_and_k_core(data.interactions)
    children = np.random.SeedSequence(parent_seed).spawn(development_worlds)
    parents = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    streams = [spawn_seed_streams(seed) for seed in parents]
    by_epoch: dict[int, list[float]] = {epoch: [] for epoch in range(1, max_epochs + 1)}
    started = time.perf_counter()
    for seed_set in streams:
        split = split_per_user(rows, world_seed=seed_set.world)
        visible_train, _validation, _ = _visible_split(split.train, seed=seed_set.world)
        graph = build_user_history(visible_train)
        config = BPRConfig(epochs=max_epochs)

        def on_epoch(epoch: int, snapshot: TrainedModel) -> None:
            score = evaluate_full_ranking(
                snapshot,
                heldout_rows=split.validation,
                excluded_rows=visible_train,
                k=20,
                example_seed=seed_set.world,
            )
            metric = score.metrics["ndcg@20"]
            if metric is not None:
                by_epoch[epoch].append(float(metric))

        fit_bpr(graph, visible_train, config=config, seed=seed_set.training, epoch_callback=on_epoch)
    means = {
        str(epoch): float(np.mean(scores)) if scores else None
        for epoch, scores in by_epoch.items()
    }
    valid = [(float(value), -int(epoch)) for epoch, value in means.items() if value is not None]
    if not valid:
        raise RuntimeError("no valid MovieLens-100K validation score was produced")
    best_value, negative_epoch = max(valid)
    selected = -negative_epoch
    return {
        "schema_version": 1,
        "selection": "fixed epochs selected on MovieLens-100K validation NDCG@20",
        "dataset": data.dataset,
        "dataset_sha256": data.archive_sha256,
        "parent_seed": parent_seed,
        "world_seeds": [seed.world for seed in streams],
        "seed_streams": [asdict(seed) for seed in streams],
        "development_worlds": development_worlds,
        "candidate_epochs": list(range(1, max_epochs + 1)),
        "mean_validation_ndcg_at_20": means,
        "selected_epochs": selected,
        "selected_validation_ndcg_at_20": best_value,
        "elapsed_seconds": time.perf_counter() - started,
        "tier": "development_only",
        "assumptions": ["UH/BPR defaults in BPRConfig; no main-world outcomes used"],
    }


def run_clean_gate(
    archive_path: Path,
    *,
    epochs: int,
    parent_seed: int,
    output_path: Path,
    baseline_configs: int = 12,
) -> dict[str, Any]:
    """Run five paired clean seeds and a 12-configuration LightGCN tune."""
    data = load_dataset(archive_path, expected_name="ml-1m")
    positive = binarize_and_k_core(data.interactions)
    children = np.random.SeedSequence(parent_seed).spawn(5)
    world_parents = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    seed_streams = [spawn_seed_streams(seed) for seed in world_parents]
    configs = [
        BPRConfig(dimension=d, layers=l, learning_rate=lr, epochs=epochs)
        for d in (32, 64)
        for l in (1, 2, 3)
        for lr in (1e-3, 3e-3)
    ]
    if baseline_configs != len(configs):
        raise ValueError("the gate uses the specified fixed 12-configuration LightGCN budget")

    uh_values: list[float] = []
    baseline_values: list[float] = []
    picked_configs: list[dict[str, int | float]] = []
    train_timings: list[dict[str, float]] = []
    repeat_delta = 0.0
    start_all = time.perf_counter()
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    for seed_index, streams in enumerate(seed_streams):
        world_seed = streams.world
        split = split_per_user(positive, world_seed=world_seed)
        train, _hidden_v, _ = _visible_split(split.train, seed=world_seed)
        validation = split.validation
        excluded_val = train
        test_excluded = train + split.validation

        uh_config = BPRConfig(epochs=epochs)
        uh = _train_model("user_history", train, seed=streams.training, config=uh_config)
        uh_test = evaluate_full_ranking(
            uh, heldout_rows=split.test, excluded_rows=test_excluded, k=20,
            example_seed=world_seed,
        )
        uh_values.append(_metric(uh_test, "ndcg@20"))
        train_timings.append({"hypergraph_seconds": uh.elapsed_seconds})

        # Tune on Val_B only, then record the chosen candidate on Test_B.
        best_model: TrainedModel | None = None
        best_score = -math.inf
        best_config: BPRConfig | None = None
        for config_index, config in enumerate(configs, start=1):
            operator = LightGCNOperator(
                tuple(sorted({row.user_id for row in train})),
                tuple(sorted({row.item_id for row in train})),
                train,
            )
            candidate = fit_bpr(operator, train, config=config, seed=streams.training)
            validation_score = _metric(
                evaluate_full_ranking(
                    candidate,
                    heldout_rows=validation,
                    excluded_rows=excluded_val,
                    k=20,
                    example_seed=world_seed,
                ),
                "ndcg@20",
            )
            if validation_score > best_score:
                best_model, best_score, best_config = candidate, validation_score, config
            _atomic_json(progress_path, {
                "schema_version": 1,
                "status": "running",
                "completed_seeds": seed_index,
                "required_seeds": 5,
                "current_seed_index": seed_index + 1,
                "completed_lightgcn_configs_for_seed": config_index,
                "required_lightgcn_configs_per_seed": len(configs),
                "dataset": data.dataset,
                "dataset_sha256": data.archive_sha256,
                "parent_seed": parent_seed,
                "world_seeds": [seed.world for seed in seed_streams[:seed_index]],
                "fixed_epochs": epochs,
                "hypergraph_ndcg_at_20": uh_values,
                "tuned_lightgcn_ndcg_at_20": baseline_values,
                "selected_lightgcn_configs": picked_configs,
                "elapsed_seconds": time.perf_counter() - start_all,
            })
            print(
                f"G1 seed {seed_index + 1}/5 LightGCN config {config_index}/{len(configs)} complete",
                flush=True,
            )
        if best_model is None or best_config is None:
            raise RuntimeError("LightGCN tuning produced no candidate")
        baseline_test = evaluate_full_ranking(
            best_model, heldout_rows=split.test, excluded_rows=test_excluded, k=20,
            example_seed=world_seed,
        )
        baseline_values.append(_metric(baseline_test, "ndcg@20"))
        picked_configs.append({
            "dimension": best_config.dimension,
            "layers": best_config.layers,
            "learning_rate": best_config.learning_rate,
            "regularization": best_config.regularization,
        })

        if seed_index == 0:
            duplicate = _train_model("user_history", train, seed=streams.training, config=uh_config)
            repeat_delta = float(np.max(np.abs(uh.theta - duplicate.theta)))
        partial = {
            "schema_version": 1,
            "status": "running",
            "completed_seeds": seed_index + 1,
            "required_seeds": 5,
            "dataset": data.dataset,
            "dataset_sha256": data.archive_sha256,
            "parent_seed": parent_seed,
            "world_seeds": [seed.world for seed in seed_streams[: seed_index + 1]],
            "fixed_epochs": epochs,
            "hypergraph_ndcg_at_20": uh_values,
            "tuned_lightgcn_ndcg_at_20": baseline_values,
            "selected_lightgcn_configs": picked_configs,
            "training_timings": train_timings,
            "elapsed_seconds": time.perf_counter() - start_all,
        }
        _atomic_json(progress_path, partial)
        print(
            f"G1 seed {seed_index + 1}/5 complete: UH NDCG@20={uh_values[-1]:.5f}, "
            f"tuned LightGCN NDCG@20={baseline_values[-1]:.5f}",
            flush=True,
        )

    gradient_error = _gradient_check()
    gate = clean_model_gate(
        uh_values,
        baseline_values,
        finite_difference_relative_error=gradient_error,
        repeated_run_max_abs_difference=repeat_delta,
        repeat_tolerance=1e-6,
    )
    report = {
        "schema_version": 1,
        "gate": _gate_json(gate),
        "dataset": data.dataset,
        "dataset_sha256": data.archive_sha256,
        "parent_seed": parent_seed,
        "world_seeds": [seed.world for seed in seed_streams],
        "seed_streams": [asdict(seed) for seed in seed_streams],
        "fixed_epochs": epochs,
        "hypergraph_ndcg_at_20": uh_values,
        "tuned_lightgcn_ndcg_at_20": baseline_values,
        "selected_lightgcn_configs": picked_configs,
        "finite_difference_relative_error": gradient_error,
        "same_seed_max_parameter_difference": repeat_delta,
        "repeat_tolerance": 1e-6,
        "baseline_configuration_budget": baseline_configs,
        "runtime_versions": _runtime_versions(),
        "hardware": platform.platform(),
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - start_all,
        "training_timings": train_timings,
        "tier": "clean_gate",
    }
    _atomic_json(output_path, report)
    if progress_path.is_file():
        progress_path.unlink()
    return report


def _demo_items(
    model: TrainedModel,
    user_id: int,
    target_item_id: int,
    titles: dict[int, str],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], int | None]:
    ranking = model.rank_items(user_id, exclude_seen=True)
    target_rank = next(
        (rank for rank, (item, _score) in enumerate(ranking, start=1) if item == target_item_id),
        None,
    )
    items = [
        {"label": titles[item_id], "is_target": item_id == target_item_id}
        for item_id, _score in ranking[:top_k]
    ]
    return items, target_rank


def _public_trace(step: str, counts: dict[str, int], elapsed: float | None) -> dict[str, Any]:
    return {"step": step, "counts": counts, "elapsed_seconds": elapsed}


def _run_world(
    archive_path: Path,
    *,
    dataset_name: str,
    parent_seed: int,
    epochs: int,
    attack_rule: str,
    attack_rate: float,
    target_item_id: int | None,
    construction: str,
    hour_policy: str,
    identification_recall: float,
    false_positive_share: float,
    identification_strategy: str,
    false_positive_strategy: str,
    confidence_mode: str,
    g1_report_path: Path,
    output_root: Path,
    exploratory: bool,
) -> dict[str, Any]:
    streams = spawn_seed_streams(parent_seed)
    run_config = {
        "dataset": dataset_name,
        "parent_seed": parent_seed,
        "epochs": epochs,
        "attack": attack_rule,
        "attack_rate": attack_rate,
        "target_item_id": target_item_id,
        "hypergraph_construction": construction,
        "hour_policy": hour_policy,
        "identification_recall": identification_recall,
        "false_positive_share": false_positive_share,
        "identification_strategy": identification_strategy,
        "false_positive_strategy": false_positive_strategy,
        "confidence_mode": confidence_mode,
        "visible_validation_fraction": 0.05,
        "bpr": asdict(BPRConfig(epochs=epochs)),
        "tier": "exploratory" if exploratory else "main_candidate",
    }
    config_hash = _json_digest(run_config)
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}"
    run_dir = output_root / run_id
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "config": run_config,
        "config_sha256": config_hash,
        "dataset": dataset_name,
        "parent_seed": parent_seed,
        "seed_streams": asdict(streams),
        "git_commit": _git_commit(),
        "runtime_versions": _runtime_versions(),
        "hardware": platform.platform(),
        "stages": [],
        "validity": {
            "g1": False,
            "g2": False,
            "v1": False,
            "leakage_boundary": False,
            "influence_reliable": False,
            "complete_manifest": False,
            "valid_run": False,
            "reasons": [],
        },
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    total_start = time.perf_counter()
    try:
        stage_start = time.perf_counter()
        data = load_dataset(archive_path, expected_name=dataset_name)
        positive = binarize_and_k_core(data.interactions)
        split = split_per_user(positive, world_seed=streams.world)
        selected_targets = select_targets(split.train, world_seed=streams.world, count=5)
        target = target_item_id if target_item_id is not None else selected_targets[0]
        if target not in selected_targets:
            raise DatasetError("target item must be one of the five seeded lower-popularity targets")
        manifest["dataset_sha256"] = data.archive_sha256
        manifest["data_counts"] = {
            "positive_interactions": len(positive),
            "benign_train_interactions": len(split.train),
            "users": len({row.user_id for row in positive}),
            "items": len({row.item_id for row in positive}),
        }
        stage_records = manifest["stages"]
        stage_records.append({
            "name": "S0_dataset_preprocessing",
            "elapsed_seconds": time.perf_counter() - stage_start,
            "status": "complete",
        })
        _atomic_json(run_dir / "manifest.json", manifest)

        stage_start = time.perf_counter()
        attacked = inject_fake_accounts(
            split.train,
            seed=streams.attack,
            target_item_id=target,
            alpha=attack_rate,
            rule=attack_rule,
        )
        visible_rows, visible_validation, validation_indices = _visible_split(
            attacked.interactions, seed=streams.world, fraction=0.05,
        )
        index_set = set(validation_indices)
        reference_rows = tuple(
            row for index, row in enumerate(attacked.interactions)
            if index not in index_set and row.user_id not in attacked.hidden_source_ids
        )
        benign_validation = _map_benign(split.validation, attacked.benign_user_map)
        benign_test = _map_benign(split.test, attacked.benign_user_map)
        manifest["stages"].append({
            "name": "S3_attack_and_S2_visible_split",
            "elapsed_seconds": time.perf_counter() - stage_start,
            "status": "complete",
        })
        attack_elapsed = time.perf_counter() - stage_start
        _atomic_json(run_dir / "manifest.json", manifest)

        stage_start = time.perf_counter()
        reference_graph_start = time.perf_counter()
        reference_graph = _operator(construction, reference_rows, seed=streams.training)
        reference_graph_elapsed = time.perf_counter() - reference_graph_start
        reference = fit_bpr(
            reference_graph, reference_rows, seed=streams.training,
            config=BPRConfig(epochs=epochs),
        )
        evaluation_start = time.perf_counter()
        ref_eval = evaluate_full_ranking(
            reference,
            heldout_rows=benign_test,
            excluded_rows=reference_rows + benign_validation,
            target_item_id=target,
            k=20,
            example_seed=streams.world,
        )
        reference_evaluation_elapsed = time.perf_counter() - evaluation_start
        poisoned_graph_start = time.perf_counter()
        poisoned_graph = _operator(construction, visible_rows, seed=streams.training)
        poisoned_graph_elapsed = time.perf_counter() - poisoned_graph_start
        poisoned = fit_bpr(
            poisoned_graph, visible_rows, config=BPRConfig(epochs=epochs), seed=streams.training,
        )
        evaluation_start = time.perf_counter()
        poisoned_eval = evaluate_full_ranking(
            poisoned,
            heldout_rows=benign_test,
            excluded_rows=visible_rows + benign_validation,
            target_item_id=target,
            k=20,
            example_seed=streams.world,
        )
        early_evaluation_elapsed = reference_evaluation_elapsed + time.perf_counter() - evaluation_start
        manifest["stages"].append({
            "name": "S1_hypergraph_and_S2_baseline",
            "elapsed_seconds": time.perf_counter() - stage_start,
            "status": "complete",
        })
        _atomic_json(run_dir / "manifest.json", manifest)

        # Required matched no-attack and placebo controls.
        stage_start = time.perf_counter()
        no_attack = inject_fake_accounts(
            split.train, seed=streams.attack, target_item_id=target,
            alpha=0.0, rule=attack_rule,
        )
        no_attack_visible, _no_attack_v, _ = _visible_split(
            no_attack.interactions, seed=streams.world, fraction=0.05,
        )
        no_attack_graph = _operator(construction, no_attack_visible, seed=streams.training)
        no_attack_model = fit_bpr(
            no_attack_graph, no_attack_visible, config=BPRConfig(epochs=epochs), seed=streams.training,
        )
        no_attack_eval = evaluate_full_ranking(
            no_attack_model,
            heldout_rows=_map_benign(split.test, no_attack.benign_user_map),
            excluded_rows=no_attack_visible + _map_benign(split.validation, no_attack.benign_user_map),
            target_item_id=target,
            k=20,
            example_seed=streams.world,
        )
        placebo = inject_fake_accounts(
            split.train, seed=streams.attack, target_item_id=target,
            alpha=attack_rate, placebo=True, rule=attack_rule,
        )
        placebo_visible, _placebo_v, _ = _visible_split(
            placebo.interactions, seed=streams.world, fraction=0.05,
        )
        placebo_graph = _operator(construction, placebo_visible, seed=streams.training)
        placebo_model = fit_bpr(
            placebo_graph, placebo_visible, config=BPRConfig(epochs=epochs), seed=streams.training,
        )
        placebo_eval = evaluate_full_ranking(
            placebo_model,
            heldout_rows=_map_benign(split.test, placebo.benign_user_map),
            excluded_rows=placebo_visible + _map_benign(split.validation, placebo.benign_user_map),
            target_item_id=target,
            k=20,
            example_seed=streams.world,
        )
        manifest["stages"].append({
            "name": "S3_controls",
            "elapsed_seconds": time.perf_counter() - stage_start,
            "status": "complete",
        })
        identification_elapsed = time.perf_counter() - stage_start
        _atomic_json(run_dir / "manifest.json", manifest)

        g1 = json.loads(g1_report_path.read_text(encoding="utf-8"))
        g1_passed = (
            bool(g1.get("gate", {}).get("passed", False))
            and g1.get("dataset") == "ml-1m"
            and g1.get("fixed_epochs") == epochs
        )
        v1_report_path = GATES / "v1-convex.json"
        v1 = json.loads(v1_report_path.read_text(encoding="utf-8")) if v1_report_path.is_file() else {}
        v1_passed = bool(v1.get("gate", {}).get("passed", False))
        g2 = attack_strength_gate(
            reference_exposure=_metric(ref_eval, "target_exposure@20"),
            poisoned_exposure=_metric(poisoned_eval, "target_exposure@20"),
            reference_ndcg=_metric(ref_eval, "ndcg@20"),
            poisoned_ndcg=_metric(poisoned_eval, "ndcg@20"),
        )
        manifest["gate_results"] = {
            "g1": {"passed": g1_passed, "report_sha256": hashlib.sha256(g1_report_path.read_bytes()).hexdigest()},
            "v1": {"passed": v1_passed, "report_sha256": hashlib.sha256(v1_report_path.read_bytes()).hexdigest() if v1_report_path.is_file() else None},
            "g2": _gate_json(g2),
            "controls": {
                "no_attack_target_exposure_at_20": _metric(no_attack_eval, "target_exposure@20"),
                "placebo_target_exposure_at_20": _metric(placebo_eval, "target_exposure@20"),
                "reference_target_exposure_at_20": _metric(ref_eval, "target_exposure@20"),
            },
        }
        manifest["validity"]["g1"] = g1_passed
        manifest["validity"]["g2"] = g2.passed
        manifest["validity"]["v1"] = v1_passed
        _atomic_json(run_dir / "manifest.json", manifest)

        if not g1_passed:
            manifest["validity"]["reasons"].append("G1 report missing, failed, or for the wrong dataset")
        if not g2.passed:
            manifest["validity"]["reasons"].append("G2 attack-strength gate failed")
        if not v1_passed:
            manifest["validity"]["reasons"].append("V1 convex influence validation missing or failed")
        if not exploratory and (not g1_passed or not g2.passed or not v1_passed):
            manifest["status"] = "invalid"
            manifest["elapsed_seconds"] = time.perf_counter() - total_start
            manifest["validity"]["complete_manifest"] = True
            manifest["validity"]["valid_run"] = False
            _atomic_json(run_dir / "manifest.json", manifest)
            return manifest

        # Identification sees only visible poisoned rows; hidden IDs remain in this
        # orchestration scope and only the separated visible flags cross the API.
        stage_start = time.perf_counter()
        flagging = simulate_flagging(
            visible_rows,
            attacked.hidden_source_ids.intersection({row.user_id for row in visible_rows}),
            seed=streams.flagging,
            recall=identification_recall,
            false_positive_share=false_positive_share,
            malicious_strategy=identification_strategy,
            benign_strategy=false_positive_strategy,
            confidence_mode=confidence_mode,
        )
        manifest["stages"].append({
            "name": "S5_visible_identification",
            "elapsed_seconds": time.perf_counter() - stage_start,
            "status": "complete",
        })
        _atomic_json(run_dir / "manifest.json", manifest)

        # Prepare only visible tensors and fields for the staged unlearner.
        import torch

        users = poisoned_graph.user_ids
        items = poisoned_graph.item_ids
        user_idx = {user: i for i, user in enumerate(users)}
        item_idx = {item: i for i, item in enumerate(items)}
        row_user_indices = np.fromiter((user_idx[row.user_id] for row in visible_rows), dtype=np.int64)
        row_item_indices = np.fromiter((item_idx[row.item_id] for row in visible_rows), dtype=np.int64)
        history: dict[int, set[int]] = {user: set() for user in users}
        for row in visible_rows:
            history[row.user_id].add(row.item_id)
        rng = np.random.default_rng(streams.training ^ streams.unlearning)
        negative_matrix = np.empty((len(visible_rows), 1), dtype=np.int64)
        for position, row in enumerate(visible_rows):
            candidates = [idx for idx, item in enumerate(items) if item not in history[row.user_id]]
            if not candidates:
                raise ValueError("visible user has no eligible frozen negative item")
            negative_matrix[position, 0] = candidates[int(rng.integers(0, len(candidates)))]
        visible_experiment = VisibleExperiment(
            graph=poisoned_graph,
            model=poisoned,
            training_rows=tuple(visible_rows),
            validation_rows=tuple(visible_validation),
            flags=flagging.visible,
            target_item_ids=(target,),
            negative_item_indices=torch.as_tensor(negative_matrix, dtype=torch.long),
            positive_user_indices=torch.as_tensor(row_user_indices, dtype=torch.long),
            positive_item_indices=torch.as_tensor(row_item_indices, dtype=torch.long),
        )
        unlearning_start = time.perf_counter()
        result = run_staged_unlearning(
            visible_experiment,
            config=StagedUnlearnConfig(hour_policy=hour_policy, seed=streams.unlearning),
        )
        unlearn_elapsed = time.perf_counter() - unlearning_start
        manifest["stages"].append({
            "name": "S6_influence_staged_unlearning_and_recovery",
            "elapsed_seconds": unlearn_elapsed,
            "status": "complete" if result.refusal_reason is None else "refused",
            "stage_guard_summary": [
                {
                    "stage": step.stage,
                    "accepted": step.accepted,
                    "reason": step.reason,
                    "extension_count": step.extension_count,
                    "extension_mass": step.extension_mass,
                    "utility_retention": step.utility_retention,
                    "elapsed_seconds": step.elapsed_seconds,
                    "influence_reliable": step.influence_reliable,
                }
                for step in result.stages
            ],
            "refusal_reason": result.refusal_reason,
            "recovery_epochs": result.recovery_epochs,
            "staged_exposure_change": result.staged_exposure_change,
            "final_influence_reliable": result.final_influence_reliable,
            "frozen_exposure_cohort_size": result.frozen_exposure_cohort_size,
            "final_exposure_surrogate": result.final_exposure_surrogate,
            "final_exposure_excluding_extensions": result.final_exposure_excluding_extensions,
        })
        _atomic_json(run_dir / "manifest.json", manifest)
        reliable = (
            bool(result.stages)
            and all(stage.influence_reliable for stage in result.stages)
            and result.final_influence_reliable
        )
        manifest["validity"]["influence_reliable"] = reliable

        # Evaluator and oracle-only counts remain aggregate and are never passed
        # back to the visible unlearner or copied into the browser export.
        flagged_users = {flag.user_id for flag in flagging.visible.flags}
        selected_extensions = set(result.selected_extension_user_ids)
        acted_users = flagged_users | selected_extensions
        benign_users = set(attacked.benign_user_map.values())
        benign_weight_loss = float(np.mean([
            1.0 - result.source_weights[user_idx[user]]
            for user in benign_users if user in user_idx
        ])) if benign_users else 0.0
        evaluator_hidden_recall = (
            len(acted_users & set(attacked.hidden_source_ids)) / len(attacked.hidden_source_ids)
            if attacked.hidden_source_ids else 0.0
        )
        extension_precision = (
            len(selected_extensions & set(attacked.hidden_source_ids)) / len(selected_extensions)
            if selected_extensions else 0.0
        )

        retrain_start = time.perf_counter()
        unflagged_visible_rows = tuple(row for row in visible_rows if row.user_id not in flagged_users)
        retrain_graph = _operator(construction, unflagged_visible_rows, seed=streams.training)
        retrain = fit_bpr(
            retrain_graph,
            unflagged_visible_rows,
            config=BPRConfig(epochs=epochs),
            seed=streams.training,
        )
        retrain_elapsed = time.perf_counter() - retrain_start

        test_users = {row.user_id for row in benign_test}
        frozen_exposure_users = (benign_users & test_users) - flagged_users
        final_evaluation_start = time.perf_counter()
        test_after = evaluate_full_ranking(
            result.model,
            heldout_rows=benign_test,
            excluded_rows=visible_rows + benign_validation,
            target_item_id=target,
            k=20,
            example_seed=streams.world,
            target_exposure_users=frozen_exposure_users,
        )
        retrain_eval = evaluate_full_ranking(
            retrain,
            heldout_rows=tuple(row for row in benign_test if row.user_id in retrain.user_index),
            excluded_rows=unflagged_visible_rows + benign_validation,
            target_item_id=target if target in retrain.item_index else None,
            k=20,
            example_seed=streams.world,
            target_exposure_users=frozen_exposure_users,
        )
        evaluation_elapsed = time.perf_counter() - final_evaluation_start
        example_candidates = sorted(
            user for user in benign_users
            if user in poisoned.user_index and user in result.model.user_index
            and user not in flagged_users and target not in poisoned.history.get(user, frozenset())
        )
        if not example_candidates:
            raise ValueError("no benign unflagged user is eligible for the demo Top-K comparison")
        example_user = example_candidates[streams.world % len(example_candidates)]
        before_items, before_rank = _demo_items(
            poisoned, example_user, target, data.titles, top_k=20,
        )
        after_items, after_rank = _demo_items(
            result.model, example_user, target, data.titles, top_k=20,
        )
        retrain_items, retrain_rank = _demo_items(
            retrain, example_user, target, data.titles, top_k=20,
        )
        target_rank_mean = test_after.metrics.get("target_mean_rank")
        aggregate_metrics: dict[str, float] = {
            "ndcg_at_20": _metric(test_after, "ndcg@20"),
            "recall_at_20": _metric(test_after, "recall@20"),
            "target_exposure_at_20": _metric(test_after, "target_exposure@20"),
            "hidden_source_recall": evaluator_hidden_recall,
            "precision": extension_precision,
            "benign_weight_loss": benign_weight_loss,
            "utility_retention": (
                _metric(test_after, "ndcg@20") / _metric(ref_eval, "ndcg@20")
                if _metric(ref_eval, "ndcg@20") > 0 else 0.0
            ),
            "cost_ratio": unlearn_elapsed / retrain_elapsed if retrain_elapsed > 0 else 0.0,
        }
        if target_rank_mean is not None:
            aggregate_metrics["target_mean_rank"] = float(target_rank_mean)
        if result.staged_exposure_change is not None:
            aggregate_metrics["staged_predicted_exposure_change"] = float(result.staged_exposure_change)
        if result.final_exposure_surrogate is not None:
            aggregate_metrics["frozen_cohort_exposure_surrogate"] = float(result.final_exposure_surrogate)
        if result.final_exposure_excluding_extensions is not None:
            aggregate_metrics["frozen_cohort_minus_extensions_exposure_surrogate"] = float(
                result.final_exposure_excluding_extensions
            )
        poisoned_metrics: dict[str, float] = {
            "ndcg_at_20": _metric(poisoned_eval, "ndcg@20"),
            "recall_at_20": _metric(poisoned_eval, "recall@20"),
            "target_exposure_at_20": _metric(poisoned_eval, "target_exposure@20"),
        }
        if poisoned_eval.metrics.get("target_mean_rank") is not None:
            poisoned_metrics["target_mean_rank"] = float(poisoned_eval.metrics["target_mean_rank"])
        retrain_metrics: dict[str, float] = {
            "ndcg_at_20": _metric(retrain_eval, "ndcg@20"),
            "recall_at_20": _metric(retrain_eval, "recall@20"),
            "cost_ratio": 1.0,
        }
        for metric_key, raw_key in (
            ("target_exposure_at_20", "target_exposure@20"),
            ("target_mean_rank", "target_mean_rank"),
        ):
            if retrain_eval.metrics.get(raw_key) is not None:
                retrain_metrics[metric_key] = float(retrain_eval.metrics[raw_key])
        if _metric(ref_eval, "ndcg@20") > 0:
            retrain_metrics["utility_retention"] = _metric(retrain_eval, "ndcg@20") / _metric(ref_eval, "ndcg@20")
        leakage_ok = check_unlearning_import_boundary(ROOT / "src" / "iahu")
        manifest["validity"]["leakage_boundary"] = leakage_ok
        if not leakage_ok:
            manifest["validity"]["reasons"].append("unlearning/influence import boundary check failed")
        if result.refusal_reason:
            manifest["validity"]["reasons"].append(f"unlearner refused: {result.refusal_reason}")
        if not reliable:
            manifest["validity"]["reasons"].append("influence estimate is unreliable or no staged estimate ran")

        valid = (
            g1_passed and g2.passed and v1_passed and leakage_ok and reliable
            and result.refusal_reason is None
        )
        if exploratory:
            valid = False
            manifest["validity"]["reasons"].append("exploratory tier cannot become a headline/demo run")
        manifest["validity"]["valid_run"] = valid
        manifest["validity"]["complete_manifest"] = True
        manifest["status"] = "valid" if valid else "invalid"
        manifest["elapsed_seconds"] = time.perf_counter() - total_start
        manifest["evaluator_metrics"] = {
            "reference_ndcg_at_20": _metric(ref_eval, "ndcg@20"),
            "poisoned_ndcg_at_20": _metric(poisoned_eval, "ndcg@20"),
            "after_ndcg_at_20": _metric(test_after, "ndcg@20"),
            "reference_target_exposure_at_20": _metric(ref_eval, "target_exposure@20"),
            "poisoned_target_exposure_at_20": _metric(poisoned_eval, "target_exposure@20"),
            "after_target_exposure_at_20": _metric(test_after, "target_exposure@20"),
            "placebo_target_exposure_at_20": _metric(placebo_eval, "target_exposure@20"),
            "no_attack_target_exposure_at_20": _metric(no_attack_eval, "target_exposure@20"),
            "retrain_ndcg_at_20": _metric(retrain_eval, "ndcg@20"),
            "retrain_elapsed_seconds": retrain_elapsed,
        }

        export_source = {
            "schema_version": "completed-run-export-v1",
            "scenarios": [{
                "tier": "deployable" if valid else "exploratory",
                "label": f"{dataset_name} / {construction} / {attack_rule} / seed {parent_seed}",
                "setting": {
                    "dataset": dataset_name,
                    "hypergraph_construction": (
                        f"{construction} ({hour_policy})" if construction == "hour_inspired" else construction
                    ),
                    "attack": attack_rule,
                    "seeds": asdict(streams),
                    "identification": {
                        "rate": identification_recall,
                        "false_positive_share": false_positive_share,
                        "strategy": identification_strategy,
                    },
                },
                "target_item_label": data.titles[target],
                "example": {
                    "user_label": "Example user 01",
                    "top_k": 20,
                    "before": before_items,
                    "target_rank_before": before_rank,
                },
                "pipeline_trace": [
                    _public_trace("dataset_preprocessing", {
                        "users": len({row.user_id for row in positive}),
                        "items": len({row.item_id for row in positive}),
                        "interactions": len(positive),
                    }, next(step["elapsed_seconds"] for step in stage_records if step["name"] == "S0_dataset_preprocessing")),
                    _public_trace("interaction_matrix", {
                        "users": len(poisoned_graph.user_ids),
                        "items": len(poisoned_graph.item_ids),
                        "interactions": len(visible_rows),
                    }, None),
                    _public_trace("hypergraph_construction", {
                        "users": len(poisoned_graph.user_ids),
                        "items": len(poisoned_graph.item_ids),
                        "hyperedges": poisoned_graph.num_hyperedges,
                        "memberships": poisoned_graph.num_memberships,
                    }, reference_graph_elapsed + poisoned_graph_elapsed),
                    _public_trace("recommendation_baseline", {
                        "users": len(poisoned_graph.user_ids),
                        "items": len(poisoned_graph.item_ids),
                    }, poisoned.elapsed_seconds),
                ],
                "poisoning_trace": [
                    _public_trace("attack_configuration", {"users": len(benign_users), "items": len(poisoned_graph.item_ids)}, None),
                    _public_trace("poisoning_simulation", {
                        "injected_interactions": len(attacked.interactions) - len(split.train),
                        "interactions": len(attacked.interactions),
                    }, attack_elapsed),
                    _public_trace("identification", {
                        "candidate_sources": len(poisoned_graph.user_ids),
                        "flagged_sources": len(flagged_users),
                    }, identification_elapsed),
                    _public_trace("unlearning", {"candidate_sources": len(poisoned_graph.user_ids)}, unlearn_elapsed),
                    _public_trace("evaluation", {"evaluation_users": int(test_after.metrics.get("evaluation_users") or 0)}, early_evaluation_elapsed + evaluation_elapsed),
                ],
                "methods": [
                    {
                        "label": "Poisoned baseline (no unlearning)",
                        "after": before_items,
                        "target_rank_after": before_rank,
                        "aggregate_metrics": poisoned_metrics,
                        "elapsed_seconds": poisoned.elapsed_seconds,
                    },
                    {
                        "label": "Retrain-S (visible flagged-source removal)",
                        "after": retrain_items,
                        "target_rank_after": retrain_rank,
                        "aggregate_metrics": retrain_metrics,
                        "elapsed_seconds": retrain_elapsed,
                    },
                    {
                        "label": "Influence-aware selective unlearning (staged)",
                        "after": after_items,
                        "target_rank_after": after_rank,
                        "aggregate_metrics": aggregate_metrics,
                        "elapsed_seconds": unlearn_elapsed,
                    },
                ],
            }],
        }
        manifest["demo_export_available"] = valid
        if valid:
            export_path = run_dir / "completed-run-export.json"
            _atomic_json(export_path, export_source)
            COMPLETED_EXPORTS.mkdir(parents=True, exist_ok=True)
            _atomic_json(COMPLETED_EXPORTS / f"{run_id}.json", export_source)
            manifest["completed_run_export"] = str(export_path.relative_to(ROOT))
        manifest["elapsed_seconds"] = time.perf_counter() - total_start
        _atomic_json(run_dir / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc(limit=8)
        manifest["elapsed_seconds"] = time.perf_counter() - total_start
        manifest["validity"]["complete_manifest"] = True
        manifest["validity"]["valid_run"] = False
        manifest["validity"]["reasons"].append("pipeline raised an exception")
        _atomic_json(run_dir / "manifest.json", manifest)
        raise


def check_unlearning_import_boundary(package_root: Path) -> bool:
    """Static check for forbidden evaluator/vault imports in influence code."""
    forbidden = {"iahu.vault", "iahu.eval", "iahu.evaluator", "iahu.poison.manifest"}
    checked = (package_root / "unlearn.py", package_root / "influence.py")
    try:
        for path in checked:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(name in forbidden or any(name.startswith(item + ".") for item in forbidden) for name in names):
                    return False
    except (OSError, SyntaxError):
        return False
    return True


def _read_e_star(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("dataset") != "ml-100k":
        raise ValueError("E* file is not a MovieLens-100K development selection")
    allowlist = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))
    if value.get("dataset_sha256") != allowlist["datasets"]["ml-100k"]["sha256"]:
        raise ValueError("E* dataset hash does not match the pinned MovieLens-100K archive")
    if value.get("tier") != "development_only":
        raise ValueError("E* selection must be labelled development_only")
    epochs = value.get("selected_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("E* file does not contain a positive selected_epochs value")
    return epochs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-epochs", help="choose fixed E* on MovieLens-100K")
    select.add_argument("--archive", type=Path, required=True)
    select.add_argument("--parent-seed", type=int, default=20261108)
    select.add_argument("--development-worlds", type=int, default=5)
    select.add_argument("--max-epochs", type=int, default=20)
    select.add_argument("--output", type=Path, default=PREREG / "e-star-ml-100k.json")

    gate = subparsers.add_parser("g1", help="run the clean-model gate on MovieLens-1M")
    gate.add_argument("--archive", type=Path, required=True)
    gate.add_argument("--epochs", type=int, required=True)
    gate.add_argument("--parent-seed", type=int, default=20261108)
    gate.add_argument("--output", type=Path, default=GATES / "g1-ml-1m.json")

    v1 = subparsers.add_parser("v1", help="run the hard convex influence validation")
    v1.add_argument("--output", type=Path, default=GATES / "v1-convex.json")

    run = subparsers.add_parser("run", help="run one offline attack/control/unlearning world")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--dataset", choices=("ml-100k", "ml-1m"), default="ml-1m")
    run.add_argument("--parent-seed", type=int, required=True)
    run.add_argument("--e-star", type=Path, default=PREREG / "e-star-ml-100k.json")
    run.add_argument("--attack", choices=("atk1_random", "atk2_bandwagon", "atk3_popularity", "atk4_coordinated", "atk5_overlap_minimizing"), default="atk3_popularity")
    run.add_argument("--alpha", type=float, choices=(0.01, 0.03), default=0.03)
    run.add_argument("--target-item", type=int)
    run.add_argument("--construction", choices=("user_history", "hour_inspired"), default="user_history")
    run.add_argument("--hour-policy", choices=("frozen", "rebuilt"), default="frozen")
    run.add_argument("--recall", type=float, default=0.5)
    run.add_argument("--false-positive-share", type=float, default=0.1)
    run.add_argument("--identification-strategy", choices=("uniform", "obvious_first", "clustered"), default="obvious_first")
    run.add_argument("--false-positive-strategy", choices=("uniform", "look_alike"), default="uniform")
    run.add_argument("--confidence", choices=("hard", "soft"), default="hard")
    run.add_argument("--g1-report", type=Path, default=GATES / "g1-ml-1m.json")
    run.add_argument("--output-root", type=Path, default=RUNS)
    run.add_argument("--exploratory", action="store_true", help="run downstream stages even when G1/G2 fails; never yields a valid demo export")

    args = parser.parse_args(argv)
    try:
        if args.command == "select-epochs":
            report = select_fixed_epochs(
                args.archive,
                parent_seed=args.parent_seed,
                development_worlds=args.development_worlds,
                max_epochs=args.max_epochs,
            )
            _atomic_json(args.output, report)
        elif args.command == "g1":
            report = run_clean_gate(
                args.archive,
                epochs=args.epochs,
                parent_seed=args.parent_seed,
                output_path=args.output,
            )
        elif args.command == "v1":
            result = validate_v1_convex_influence()
            report = {
                "schema_version": 1,
                "validation": result.name,
                "gate": {
                    "passed": result.passed,
                    "observed": result.observed,
                    "reasons": list(result.reasons),
                },
                "threshold": "maximum relative centered finite-difference error < 1e-5",
                "numeric_type": "float64",
            }
            _atomic_json(args.output, report)
        else:
            epochs = _read_e_star(args.e_star)
            report = _run_world(
                args.archive,
                dataset_name=args.dataset,
                parent_seed=args.parent_seed,
                epochs=epochs,
                attack_rule=args.attack,
                attack_rate=args.alpha,
                target_item_id=args.target_item,
                construction=args.construction,
                hour_policy=args.hour_policy,
                identification_recall=args.recall,
                false_positive_share=args.false_positive_share,
                identification_strategy=args.identification_strategy,
                false_positive_strategy=args.false_positive_strategy,
                confidence_mode=args.confidence,
                g1_report_path=args.g1_report,
                output_root=args.output_root,
                exploratory=args.exploratory,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") != "failed" else 2
    except (DatasetError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"experiment command failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

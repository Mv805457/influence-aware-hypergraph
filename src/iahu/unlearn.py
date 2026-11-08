"""Visible-input plan builders for source-level selective unlearning (S6)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .flagging import VisibleFlagSet


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


def build_plan(
    flags: VisibleFlagSet,
    suspicion_scores: Mapping[int, float],
    budget: UnlearnBudget,
    *,
    operator: str,
    seed: int,
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

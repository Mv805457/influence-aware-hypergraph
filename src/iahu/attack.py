"""Seeded controlled fake-account poisoning for offline MovieLens runs."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .data import Interaction


@dataclass(frozen=True, slots=True)
class InjectionResult:
    """In-memory evaluator state. Hidden mappings must never be serialized to demo output."""

    interactions: tuple[Interaction, ...]
    benign_user_map: dict[int, int]
    hidden_source_ids: frozenset[int]
    target_item_id: int | None
    fake_account_count: int
    profile_length: int
    rule: str
    placebo: bool


def inject_fake_accounts(
    benign_train: Sequence[Interaction],
    *,
    seed: int,
    target_item_id: int | None,
    alpha: float = 0.01,
    placebo: bool = False,
    rule: str = "atk1_random",
) -> InjectionResult:
    """Run a specified classical target attack or its matched placebo."""
    if not benign_train:
        raise ValueError("poisoning requires benign training interactions")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("attack rate must be finite and non-negative")
    if target_item_id is None:
        raise ValueError("attack and matched placebo require the paired target item")
    if rule not in {
        "atk1_random", "atk2_bandwagon", "atk3_popularity",
        "atk4_coordinated", "atk5_overlap_minimizing",
    }:
        raise ValueError("supported attack rules are ATK-1 through ATK-5")

    users = sorted({row.user_id for row in benign_train})
    items = sorted({row.item_id for row in benign_train})
    if target_item_id is not None and target_item_id not in items:
        raise ValueError("target item must occur in the benign training data")
    rng = random.Random(seed)
    fake_count = math.ceil(alpha * len(users))
    counts = Counter(row.user_id for row in benign_train)
    profile_length = max(1, int(round(median(counts.values()))))
    filler_count = profile_length if placebo else max(0, profile_length - 1)
    target_pool = [item for item in items if item != target_item_id]
    if filler_count > len(target_pool):
        raise ValueError("not enough unique items for the requested fake profile size")
    timestamps = [row.timestamp for row in benign_train if row.timestamp is not None]
    item_popularity = Counter(row.item_id for row in benign_train)
    top_count = max(1, math.ceil(0.01 * len(items)))
    popular_items = sorted(items, key=lambda item: (-item_popularity[item], item))[:top_count]
    popular_pool = [item for item in popular_items if item != target_item_id]
    coordinated_pool: list[int] = []
    if rule == "atk4_coordinated":
        coordinated_pool = _weighted_sample(
            target_pool, item_popularity, min(3 * profile_length, len(target_pool)), rng
        )
    overlap_order = list(target_pool)
    if rule == "atk5_overlap_minimizing":
        rng.shuffle(overlap_order)
    overlap_cursor = 0

    next_id = max(users) + 1
    appended: list[Interaction] = []
    hidden_temporary_ids: set[int] = set()
    for offset in range(fake_count):
        fake_id = next_id + offset
        hidden_temporary_ids.add(fake_id)
        if rule == "atk1_random":
            profile = rng.sample(target_pool, filler_count)
        elif rule == "atk2_bandwagon":
            bandwagon_count = min(filler_count, round(0.3 * filler_count))
            bandwagon_count = min(bandwagon_count, len(popular_pool))
            profile = rng.sample(popular_pool, bandwagon_count)
            remaining_pool = [item for item in target_pool if item not in set(profile)]
            profile.extend(rng.sample(remaining_pool, filler_count - bandwagon_count))
        elif rule == "atk3_popularity":
            profile = _weighted_sample(target_pool, item_popularity, filler_count, rng)
        elif rule == "atk4_coordinated":
            shared_count = min(round(0.8 * filler_count), filler_count, len(coordinated_pool))
            profile = rng.sample(coordinated_pool, shared_count)
            remaining_pool = [item for item in target_pool if item not in set(profile)]
            profile.extend(_weighted_sample(
                remaining_pool, item_popularity, filler_count - shared_count, rng
            ))
        else:
            profile = []
            while len(profile) < filler_count:
                if overlap_cursor == len(overlap_order):
                    overlap_order = list(target_pool)
                    rng.shuffle(overlap_order)
                    overlap_cursor = 0
                item_id = overlap_order[overlap_cursor]
                overlap_cursor += 1
                if item_id not in profile:
                    profile.append(item_id)
        if not placebo and target_item_id is not None:
            profile.append(target_item_id)
        for item_id in profile:
            timestamp = rng.choice(timestamps) if timestamps else None
            appended.append(Interaction(fake_id, item_id, 5, timestamp))

    all_users = users + sorted(hidden_temporary_ids)
    shuffled_ids = list(all_users)
    rng.shuffle(shuffled_ids)
    relabel = {old_id: index + 1 for index, old_id in enumerate(shuffled_ids)}
    benign_map = {user_id: relabel[user_id] for user_id in users}
    hidden_after_permutation = frozenset(relabel[user_id] for user_id in hidden_temporary_ids)
    relabeled_rows = [
        Interaction(relabel[row.user_id], row.item_id, row.rating, row.timestamp)
        for row in list(benign_train) + appended
    ]
    rng.shuffle(relabeled_rows)
    return InjectionResult(
        interactions=tuple(relabeled_rows),
        benign_user_map=benign_map,
        hidden_source_ids=hidden_after_permutation,
        target_item_id=target_item_id,
        fake_account_count=fake_count,
        profile_length=profile_length,
        rule=rule,
        placebo=placebo,
    )


def _weighted_sample(
    items: Sequence[int], weights: dict[int, int], count: int, rng: random.Random
) -> list[int]:
    """Weighted sampling without replacement via exponential-race keys."""
    if count < 0 or count > len(items):
        raise ValueError("weighted sample size exceeds its item pool")
    if count == 0:
        return []
    keys = []
    for item_id in items:
        weight = weights[item_id]
        if weight > 0:
            keys.append((-math.log(max(rng.random(), 1e-15)) / weight, item_id))
    if len(keys) < count:
        raise ValueError("weighted sample has fewer positive-weight items than requested")
    return [item for _key, item in sorted(keys)[:count]]

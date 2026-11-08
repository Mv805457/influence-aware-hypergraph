"""Offline simulator for incomplete malicious-source identification (S5).

The simulator returns a deployable-looking flag view separately from evaluator
truth. Callers must pass only ``VisibleFlagSet`` to any unlearner.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .data import Interaction


@dataclass(frozen=True, slots=True)
class VisibleFlag:
    user_id: int
    confidence: float


@dataclass(frozen=True, slots=True)
class VisibleFlagSet:
    """Only the information available to a deployable unlearner."""

    flags: tuple[VisibleFlag, ...]
    confidence_mode: str


@dataclass(frozen=True, slots=True)
class FlaggingAudit:
    """Evaluator-only truth; never serialize or pass this object to an unlearner."""

    identified_malicious: frozenset[int]
    false_positive_benign: frozenset[int]
    hidden_malicious: frozenset[int]
    realized_recall: float
    realized_false_positive_share: float


@dataclass(frozen=True, slots=True)
class FlaggingResult:
    visible: VisibleFlagSet
    audit: FlaggingAudit


def anomaly_proxy(rows: Sequence[Interaction], *, bins: int = 10) -> dict[int, float]:
    """Compute visible-only JSD of user popularity-bin profiles (base 2)."""
    if bins < 1:
        raise ValueError("bins must be positive")
    item_popularity = Counter(row.item_id for row in rows)
    by_user: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        by_user[row.user_id].add(row.item_id)
    if not item_popularity or not by_user:
        raise ValueError("anomaly proxy requires visible interactions")

    ordered_items = sorted(item_popularity, key=lambda item: (item_popularity[item], item))
    item_bin = {
        item_id: min(position * bins // len(ordered_items), bins - 1)
        for position, item_id in enumerate(ordered_items)
    }
    profiles: dict[int, list[float]] = {}
    for user_id, items in by_user.items():
        counts = [0] * bins
        for item_id in items:
            counts[item_bin[item_id]] += 1
        total = len(items)
        profiles[user_id] = [count / total for count in counts]

    users = sorted(profiles)
    mean_profile = [
        sum(profiles[user][index] for user in users) / len(users)
        for index in range(bins)
    ]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(p * math.log2(p / q) for p, q in zip(left, right) if p > 0 and q > 0)

    scores: dict[int, float] = {}
    for user_id, profile in profiles.items():
        midpoint = [(p + q) / 2 for p, q in zip(profile, mean_profile)]
        scores[user_id] = 0.5 * kl(profile, midpoint) + 0.5 * kl(mean_profile, midpoint)
    return scores


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def simulate_flagging(
    rows: Sequence[Interaction],
    malicious_user_ids: set[int] | frozenset[int],
    *,
    seed: int,
    recall: float,
    false_positive_share: float,
    malicious_strategy: str = "obvious_first",
    benign_strategy: str = "uniform",
    confidence_mode: str = "hard",
) -> FlaggingResult:
    """Create visible flags and a distinct evaluator audit using the guide's counts."""
    if not rows:
        raise ValueError("flagging requires visible interactions")
    if not math.isfinite(recall) or not 0 <= recall <= 1:
        raise ValueError("recall must be in [0, 1]")
    if not math.isfinite(false_positive_share) or not 0 <= false_positive_share < 1:
        raise ValueError("false_positive_share must be in [0, 1)")
    if malicious_strategy not in {"uniform", "obvious_first", "clustered"}:
        raise ValueError("unsupported malicious identification strategy")
    if benign_strategy not in {"uniform", "look_alike"}:
        raise ValueError("unsupported benign false-positive strategy")
    if confidence_mode not in {"hard", "soft"}:
        raise ValueError("confidence_mode must be hard or soft")

    users = {row.user_id for row in rows}
    malicious = set(malicious_user_ids)
    if not malicious or not malicious <= users:
        raise ValueError("malicious users must be a non-empty subset of visible users")
    benign = users - malicious
    rng = random.Random(seed)
    scores = anomaly_proxy(rows)
    histories: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        histories[row.user_id].add(row.item_id)

    identified_count = round(recall * len(malicious))
    false_count = round(false_positive_share / (1 - false_positive_share) * identified_count)
    if false_count > len(benign):
        raise ValueError("false-positive request exceeds the benign user population")

    if malicious_strategy == "uniform":
        identified = set(rng.sample(sorted(malicious), identified_count))
    elif malicious_strategy == "obvious_first":
        identified = set(sorted(malicious, key=lambda user: (-scores[user], user))[:identified_count])
    else:
        seed_user = rng.choice(sorted(malicious))
        identified = {seed_user}
        ranked = sorted(
            malicious - {seed_user},
            key=lambda user: (-_jaccard(histories[seed_user], histories[user]), user),
        )
        identified.update(ranked[: max(0, identified_count - 1)])

    if benign_strategy == "uniform":
        false_positives = set(rng.sample(sorted(benign), false_count))
    else:
        false_positives = set(sorted(benign, key=lambda user: (-scores[user], user))[:false_count])

    conf_by_user: dict[int, float] = {user: 1.0 for user in identified | false_positives}
    if confidence_mode == "soft":
        for user in identified:
            conf_by_user[user] = rng.betavariate(5, 2)
        for user in false_positives:
            conf_by_user[user] = rng.betavariate(2, 5)

    ordered_flags = list(conf_by_user.items())
    rng.shuffle(ordered_flags)
    actual_flagged = identified | false_positives
    audit = FlaggingAudit(
        identified_malicious=frozenset(identified),
        false_positive_benign=frozenset(false_positives),
        hidden_malicious=frozenset(malicious - identified),
        realized_recall=(len(identified) / len(malicious)),
        realized_false_positive_share=(len(false_positives) / len(actual_flagged) if actual_flagged else 0.0),
    )
    visible = VisibleFlagSet(
        flags=tuple(VisibleFlag(user, confidence) for user, confidence in ordered_flags),
        confidence_mode=confidence_mode,
    )
    return FlaggingResult(visible, audit)

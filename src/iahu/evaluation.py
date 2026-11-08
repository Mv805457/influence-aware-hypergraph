"""Full-ranking evaluation on offline held-out rows."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .data import Interaction
from .model import TrainedModel


@dataclass(frozen=True, slots=True)
class RankedExample:
    user_id: int
    top_k: tuple[tuple[int, int, float], ...]
    target_rank: int | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: dict[str, float | int | None]
    example: RankedExample | None


def evaluate_full_ranking(
    model: TrainedModel,
    *,
    heldout_rows: Sequence[Interaction],
    excluded_rows: Sequence[Interaction],
    target_item_id: int | None = None,
    k: int = 20,
    example_seed: int = 0,
    example_user_id: int | None = None,
) -> EvaluationResult:
    """Rank every candidate item; ties go to lower MovieLens item IDs."""
    import numpy as np

    if k < 1:
        raise ValueError("K must be positive")
    user_index = model.user_index
    item_index = model.item_index
    if target_item_id is not None and target_item_id not in item_index:
        raise ValueError("target item is absent from the trained model")
    excluded: dict[int, set[int]] = defaultdict(set)
    for row in excluded_rows:
        if row.user_id in user_index:
            excluded[row.user_id].add(row.item_id)
    relevant: dict[int, set[int]] = defaultdict(set)
    for row in heldout_rows:
        if row.user_id in user_index and row.item_id in item_index:
            relevant[row.user_id].add(row.item_id)
    if not relevant:
        raise ValueError("held-out evaluation has no users represented by the model")

    encoded = model.embeddings()
    all_item_vertices = np.arange(model.operator.num_users, model.operator.num_vertices)
    item_vectors = encoded[all_item_vertices]
    item_ids = model.item_ids
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    target_ranks: list[int] = []
    target_exposed = 0
    target_eligible = 0
    ranked_by_user: dict[int, list[tuple[int, float]]] = {}

    for user_id in sorted(relevant):
        candidate_positions = [
            position
            for position, item_id in enumerate(item_ids)
            if item_id not in excluded.get(user_id, set())
        ]
        if not candidate_positions:
            continue
        scores = item_vectors[candidate_positions] @ encoded[user_index[user_id]]
        ranking = [
            (item_ids[position], float(score))
            for position, score in zip(candidate_positions, scores.tolist())
        ]
        ranking.sort(key=lambda pair: (-pair[1], pair[0]))
        ranked_by_user[user_id] = ranking
        rel = relevant[user_id]
        top = ranking[:k]
        recall_values.append(len({item for item, _ in top} & rel) / len(rel))
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position, (item_id, _score) in enumerate(top)
            if item_id in rel
        )
        ideal_count = min(k, len(rel))
        idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_count))
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        if target_item_id is not None and target_item_id not in excluded.get(user_id, set()):
            target_eligible += 1
            target_rank = next(
                (position for position, (item, _score) in enumerate(ranking, start=1) if item == target_item_id),
                None,
            )
            if target_rank is not None:
                target_ranks.append(target_rank)
                target_exposed += int(target_rank <= k)

    if example_user_id is None and target_item_id is not None:
        eligible_users = [
            user_id
            for user_id in sorted(ranked_by_user)
            if target_item_id not in excluded.get(user_id, set())
        ]
        example_user_id = random.Random(example_seed).choice(eligible_users) if eligible_users else None
    example = None
    if example_user_id is not None and example_user_id in ranked_by_user:
        ranking = ranked_by_user[example_user_id]
        target_rank = next(
            (position for position, (item, _score) in enumerate(ranking, start=1) if item == target_item_id),
            None,
        ) if target_item_id is not None else None
        example = RankedExample(
            user_id=example_user_id,
            top_k=tuple(
                (item_id, position, score)
                for position, (item_id, score) in enumerate(ranking[:k], start=1)
            ),
            target_rank=target_rank,
        )

    metrics: dict[str, float | int | None] = {
        f"ndcg@{k}": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
        f"recall@{k}": sum(recall_values) / len(recall_values) if recall_values else None,
        "evaluation_users": len(ndcg_values),
        f"target_exposure@{k}": target_exposed / target_eligible if target_eligible else None,
        "target_eligible_users": target_eligible,
        "target_mean_rank": sum(target_ranks) / len(target_ranks) if target_ranks else None,
    }
    return EvaluationResult(metrics, example)

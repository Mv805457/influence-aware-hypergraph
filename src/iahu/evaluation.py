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
    target_exposure_users: set[int] | frozenset[int] | None = None,
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
    ordered_users = sorted(relevant)
    if example_user_id is None and target_item_id is not None:
        eligible_users = [
            user_id for user_id in ordered_users
            if target_item_id not in excluded.get(user_id, set())
        ]
        example_user_id = random.Random(example_seed).choice(eligible_users) if eligible_users else None

    # One batched user-item product avoids thousands of small BLAS calls while
    # retaining exact full-ranking metrics and the documented tie order.
    user_positions = np.asarray([user_index[user] for user in ordered_users], dtype=np.int64)
    score_matrix = encoded[user_positions] @ item_vectors.T
    example_ranking: tuple[tuple[int, int, float], ...] | None = None
    example_target_rank: int | None = None

    for user_offset, user_id in enumerate(ordered_users):
        candidate_positions = [
            position
            for position, item_id in enumerate(item_ids)
            if item_id not in excluded.get(user_id, set())
        ]
        if not candidate_positions:
            continue
        scores = score_matrix[user_offset, candidate_positions]
        candidate_ids = np.asarray([item_ids[position] for position in candidate_positions], dtype=np.int64)
        order = np.lexsort((candidate_ids, -scores))
        ranked_positions = np.asarray(candidate_positions, dtype=np.int64)[order]
        ranked_ids = candidate_ids[order]
        ranked_scores = scores[order]
        rel = relevant[user_id]
        top_ids = ranked_ids[:k]
        recall_values.append(len(set(top_ids.tolist()) & rel) / len(rel))
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position, item_id in enumerate(top_ids.tolist())
            if item_id in rel
        )
        ideal_count = min(k, len(rel))
        idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_count))
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        if (
            target_item_id is not None
            and target_item_id not in excluded.get(user_id, set())
            and (target_exposure_users is None or user_id in target_exposure_users)
        ):
            target_eligible += 1
            target_position = item_index[target_item_id]
            target_matches = np.flatnonzero(ranked_positions == target_position)
            target_rank = int(target_matches[0]) + 1 if target_matches.size else None
            if target_rank is not None:
                target_ranks.append(target_rank)
                target_exposed += int(target_rank <= k)
        if user_id == example_user_id:
            example_ranking = tuple(
                (int(item), position, float(score))
                for position, (item, score) in enumerate(zip(ranked_ids[:k], ranked_scores[:k]), start=1)
            )
            if target_item_id is not None and target_item_id not in excluded.get(user_id, set()):
                target_matches = np.flatnonzero(ranked_ids == target_item_id)
                example_target_rank = int(target_matches[0]) + 1 if target_matches.size else None

    example = (
        RankedExample(example_user_id, example_ranking, example_target_rank)
        if example_user_id is not None and example_ranking is not None
        else None
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

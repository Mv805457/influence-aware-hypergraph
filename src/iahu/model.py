"""BPR-trained linear graph and hypergraph recommendation models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .data import Interaction
from .hypergraph import Hypergraph


@dataclass(frozen=True, slots=True)
class BPRConfig:
    dimension: int = 64
    layers: int = 2
    learning_rate: float = 1e-3
    regularization: float = 1e-4
    batch_size: int = 4096
    epochs: int = 5

    def validate(self) -> None:
        if self.dimension < 1 or self.layers < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("dimension, layers, batch size, and epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be non-negative and finite")


class LightGCNOperator:
    """Symmetric-normalized bipartite adjacency baseline used by gate G1."""

    def __init__(self, user_ids: Sequence[int], item_ids: Sequence[int], rows: Sequence[Interaction]):
        self.user_ids = tuple(user_ids)
        self.item_ids = tuple(item_ids)
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.num_vertices = self.num_users + self.num_items
        self.user_index = {value: index for index, value in enumerate(self.user_ids)}
        self.item_index = {value: index for index, value in enumerate(self.item_ids)}
        self.edge_users = []
        self.edge_items = []
        for row in rows:
            self.edge_users.append(self.user_index[row.user_id])
            self.edge_items.append(self.num_users + self.item_index[row.item_id])

    def prepare_numpy(self):
        import numpy as np
        from scipy.sparse import csr_matrix

        users = np.asarray(self.edge_users, dtype=np.int64)
        items = np.asarray(self.edge_items, dtype=np.int64)
        user_degree = np.bincount(users, minlength=self.num_users).astype(np.float32)
        item_degree = np.bincount(items - self.num_users, minlength=self.num_items).astype(np.float32)
        inv_sqrt_user = np.zeros_like(user_degree)
        inv_sqrt_item = np.zeros_like(item_degree)
        inv_sqrt_user[user_degree > 0] = 1.0 / np.sqrt(user_degree[user_degree > 0])
        inv_sqrt_item[item_degree > 0] = 1.0 / np.sqrt(item_degree[item_degree > 0])
        coefficients = inv_sqrt_user[users] * inv_sqrt_item[items - self.num_users]
        rows = np.concatenate((users, items))
        columns = np.concatenate((items, users))
        values = np.concatenate((coefficients, coefficients))
        adjacency = csr_matrix(
            (values, (rows, columns)), shape=(self.num_vertices, self.num_vertices), dtype=np.float32
        )
        return SparseAdjacencyPropagation(adjacency)


@dataclass(slots=True)
class SparseAdjacencyPropagation:
    adjacency: object

    def apply(self, features):
        return self.adjacency.dot(features)


@dataclass(slots=True)
class TrainedModel:
    operator: Hypergraph | LightGCNOperator
    propagation: object
    user_ids: tuple[int, ...]
    item_ids: tuple[int, ...]
    theta: object
    config: BPRConfig
    history: dict[int, frozenset[int]]
    elapsed_seconds: float
    epoch_losses: tuple[float, ...]

    @property
    def user_index(self) -> dict[int, int]:
        return {value: index for index, value in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[int, int]:
        return {value: index for index, value in enumerate(self.item_ids)}

    def embeddings(self):
        import numpy as np

        states = [self.theta]
        current = self.theta
        for _ in range(self.config.layers):
            current = self.propagation.apply(current)
            states.append(current)
        return np.mean(np.stack(states, axis=0), axis=0)

    def score_all(self, user_id: int):
        import numpy as np

        users = self.user_index
        if user_id not in users:
            raise KeyError(f"unknown user ID {user_id}")
        encoded = self.embeddings()
        user_vector = encoded[users[user_id]]
        item_vectors = encoded[self.operator.num_users :]
        return item_vectors @ user_vector

    def rank_items(self, user_id: int, *, exclude_seen: bool = True) -> list[tuple[int, float]]:
        scores = self.score_all(user_id)
        seen = self.history.get(user_id, frozenset()) if exclude_seen else frozenset()
        rows = [
            (item_id, float(scores[index]))
            for index, item_id in enumerate(self.item_ids)
            if item_id not in seen
        ]
        rows.sort(key=lambda pair: (-pair[1], pair[0]))
        return rows


def fit_bpr(
    operator: Hypergraph | LightGCNOperator,
    rows: Sequence[Interaction],
    *,
    config: BPRConfig,
    seed: int,
    source_weights: Sequence[float] | None = None,
    initial_theta=None,
    epoch_callback: Callable[[int, TrainedModel], None] | None = None,
) -> TrainedModel:
    """Train the specified linear propagation model with sampled BPR negatives."""
    import numpy as np

    config.validate()
    if not rows:
        raise ValueError("BPR training requires at least one positive interaction")
    user_ids = tuple(operator.user_ids)
    item_ids = tuple(operator.item_ids)
    user_index = {value: index for index, value in enumerate(user_ids)}
    item_index = {value: index for index, value in enumerate(item_ids)}
    if source_weights is None:
        source_weights_array = np.ones(len(user_ids), dtype=np.float32)
    else:
        source_weights_array = np.asarray(source_weights, dtype=np.float32)
        if source_weights_array.shape != (len(user_ids),):
            raise ValueError("source weight count must equal the operator user count")
        if not np.all(np.isfinite(source_weights_array)) or np.any(source_weights_array < 0) or np.any(source_weights_array > 1):
            raise ValueError("source weights must be finite and in [0, 1]")
    if isinstance(operator, LightGCNOperator) and not np.all(source_weights_array == 1):
        raise ValueError("LightGCN comparison operator does not support source reweighting")
    positives_u = np.fromiter((user_index[row.user_id] for row in rows), dtype=np.int64)
    positives_i = np.fromiter((item_index[row.item_id] for row in rows), dtype=np.int64)
    history_mutable: dict[int, set[int]] = {user_id: set() for user_id in user_ids}
    internal_history: list[set[int]] = [set() for _ in user_ids]
    for row in rows:
        if source_weights_array[user_index[row.user_id]] > 0:
            history_mutable[row.user_id].add(row.item_id)
            internal_history[user_index[row.user_id]].add(item_index[row.item_id])
    if any(len(items) >= len(item_ids) for items in internal_history):
        raise ValueError("a user has no eligible negative item")

    rng = np.random.default_rng(seed)
    propagation = (
        operator.prepare_numpy(source_weights_array)
        if isinstance(operator, Hypergraph)
        else operator.prepare_numpy()
    )
    if initial_theta is None:
        theta = rng.normal(0.0, 0.01, size=(operator.num_vertices, config.dimension)).astype(np.float32)
    else:
        theta = np.asarray(initial_theta, dtype=np.float32).copy()
        if theta.shape != (operator.num_vertices, config.dimension) or not np.all(np.isfinite(theta)):
            raise ValueError("initial_theta has the wrong shape or contains non-finite values")
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    epoch_losses: list[float] = []
    start_time = time.perf_counter()
    step = 0

    for _epoch in range(config.epochs):
        order = rng.permutation(len(rows))
        total_loss = 0.0
        seen_count = 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            batch_users = positives_u[batch]
            batch_positive = positives_i[batch]
            batch_negative = rng.integers(0, len(item_ids), size=len(batch), dtype=np.int64)
            invalid = np.fromiter(
                (
                    negative in internal_history[user]
                    for user, negative in zip(batch_users.tolist(), batch_negative.tolist())
                ),
                dtype=np.bool_,
                count=len(batch),
            )
            while np.any(invalid):
                batch_negative[invalid] = rng.integers(0, len(item_ids), size=int(invalid.sum()))
                invalid = np.fromiter(
                    (
                        negative in internal_history[user]
                        for user, negative in zip(batch_users.tolist(), batch_negative.tolist())
                    ),
                    dtype=np.bool_,
                    count=len(batch),
                )
            positive_vertices = operator.num_users + batch_positive
            negative_vertices = operator.num_users + batch_negative

            propagated = [theta]
            for _ in range(config.layers):
                propagated.append(propagation.apply(propagated[-1]))
            encoded = np.mean(np.stack(propagated, axis=0), axis=0)
            user_vectors = encoded[batch_users]
            positive_vectors = encoded[positive_vertices]
            negative_vectors = encoded[negative_vertices]
            differences = np.sum(user_vectors * (positive_vectors - negative_vectors), axis=1)
            # -log(sigmoid(difference)) = logaddexp(0, -difference).
            batch_loss = np.logaddexp(0.0, -differences)
            batch_source_weights = source_weights_array[batch_users]
            total_loss += float((batch_loss * batch_source_weights).sum())
            seen_count += len(batch)

            multiplier = (
                1.0 / (1.0 + np.exp(np.clip(-differences, -40.0, 40.0))) - 1.0
            ) * batch_source_weights
            grad_encoded = np.zeros_like(encoded)
            np.add.at(
                grad_encoded,
                batch_users,
                multiplier[:, None] * (positive_vectors - negative_vectors),
            )
            np.add.at(grad_encoded, positive_vertices, multiplier[:, None] * user_vectors)
            np.add.at(grad_encoded, negative_vertices, -multiplier[:, None] * user_vectors)
            # Unbiased minibatch gradient for the summed objective in the guide.
            grad_encoded *= len(rows) / len(batch)
            grad_theta = grad_encoded
            current_grad = grad_encoded
            for _ in range(config.layers):
                current_grad = propagation.apply(current_grad)
                grad_theta = grad_theta + current_grad
            grad_theta /= config.layers + 1
            grad_theta += config.regularization * theta

            step += 1
            first_moment *= beta1
            first_moment += (1.0 - beta1) * grad_theta
            second_moment *= beta2
            second_moment += (1.0 - beta2) * grad_theta * grad_theta
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            theta -= config.learning_rate * corrected_first / (np.sqrt(corrected_second) + adam_eps)
        epoch_losses.append(total_loss / max(1, seen_count))
        if epoch_callback is not None:
            snapshot = TrainedModel(
                operator=operator,
                propagation=propagation,
                user_ids=user_ids,
                item_ids=item_ids,
                theta=theta.copy(),
                config=config,
                history={user_id: frozenset(items) for user_id, items in history_mutable.items()},
                elapsed_seconds=time.perf_counter() - start_time,
                epoch_losses=tuple(epoch_losses),
            )
            epoch_callback(_epoch + 1, snapshot)

    return TrainedModel(
        operator=operator,
        propagation=propagation,
        user_ids=user_ids,
        item_ids=item_ids,
        theta=theta,
        config=config,
        history={user_id: frozenset(items) for user_id, items in history_mutable.items()},
        elapsed_seconds=time.perf_counter() - start_time,
        epoch_losses=tuple(epoch_losses),
    )

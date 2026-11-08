"""Sparse, matrix-free hypergraph construction and propagation operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .data import Interaction


Membership = tuple[int, float, int]


@dataclass(slots=True)
class SparseHypergraphPropagation:
    """Prepared C, Dv, and De factors for repeated matrix-free application."""

    membership_matrix: object
    inv_sqrt_vertex_degree: object
    inv_edge_degree: object

    def apply(self, features):
        scaled = features * self.inv_sqrt_vertex_degree[:, None]
        edge_state = self.membership_matrix.T.dot(scaled)
        edge_state *= self.inv_edge_degree[:, None]
        output = self.membership_matrix.dot(edge_state)
        output *= self.inv_sqrt_vertex_degree[:, None]
        return output


@dataclass(frozen=True, slots=True)
class Hypergraph:
    """COO-like hyperedge membership lists with per-entry source ownership."""

    construction: str
    num_users: int
    num_items: int
    user_ids: tuple[int, ...]
    item_ids: tuple[int, ...]
    edges: tuple[tuple[Membership, ...], ...]
    policy: str = "frozen"

    @property
    def num_vertices(self) -> int:
        return self.num_users + self.num_items

    @property
    def num_hyperedges(self) -> int:
        return len(self.edges)

    @property
    def num_memberships(self) -> int:
        return sum(len(edge) for edge in self.edges)

    def _degrees(self, source_weights: Sequence[float] | None) -> tuple[list[float], list[float]]:
        if source_weights is None:
            source_weights = [1.0] * self.num_users
        if len(source_weights) != self.num_users:
            raise ValueError("source weight count must equal the number of users")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in source_weights):
            raise ValueError("source weights must be finite and in [0, 1]")
        vertex_degrees = [0.0] * self.num_vertices
        edge_degrees = [0.0] * self.num_hyperedges
        for edge_index, edge in enumerate(self.edges):
            for vertex, base_value, owner in edge:
                value = base_value * (source_weights[owner] if owner >= 0 else 1.0)
                vertex_degrees[vertex] += value
                edge_degrees[edge_index] += value
        return vertex_degrees, edge_degrees

    def apply(
        self,
        features: Sequence[Sequence[float]],
        source_weights: Sequence[float] | None = None,
    ) -> list[list[float]]:
        """Apply P = Dv^-1/2 C De^-1 C^T Dv^-1/2 without materializing P."""
        if len(features) != self.num_vertices:
            raise ValueError("feature row count must equal the number of vertices")
        width = len(features[0]) if features else 0
        if any(len(row) != width for row in features):
            raise ValueError("feature rows must have a consistent width")
        vertex_degrees, edge_degrees = self._degrees(source_weights)
        inv_sqrt_vertex = [1.0 / math.sqrt(degree) if degree > 0 else 0.0 for degree in vertex_degrees]
        scaled = [
            [float(value) * inv_sqrt_vertex[index] for value in row]
            for index, row in enumerate(features)
        ]
        output = [[0.0] * width for _ in range(self.num_vertices)]
        for edge_index, edge in enumerate(self.edges):
            edge_degree = edge_degrees[edge_index]
            if edge_degree <= 0.0:
                continue
            aggregate = [0.0] * width
            weighted_members: list[tuple[int, float]] = []
            for vertex, base_value, owner in edge:
                value = base_value * (source_weights[owner] if source_weights is not None and owner >= 0 else 1.0)
                if value <= 0.0:
                    continue
                weighted_members.append((vertex, value))
                source_row = scaled[vertex]
                for column in range(width):
                    aggregate[column] += value * source_row[column]
            scale = 1.0 / edge_degree
            for vertex, value in weighted_members:
                destination = output[vertex]
                multiplier = value * scale * inv_sqrt_vertex[vertex]
                for column in range(width):
                    destination[column] += multiplier * aggregate[column]
        return output

    def apply_numpy(
        self,
        features,
        source_weights=None,
    ):
        """NumPy matrix-free operator for offline training and evaluation."""
        return self.prepare_numpy(source_weights, dtype=features.dtype).apply(features)

    def prepare_numpy(self, source_weights=None, *, dtype=None) -> SparseHypergraphPropagation:
        """Prepare sparse C factors; P itself is never materialized."""
        import numpy as np
        from scipy.sparse import csr_matrix

        dtype = np.dtype(dtype or np.float32)
        values, vertex_ids, edge_ids, owners = [], [], [], []
        for edge_index, edge in enumerate(self.edges):
            for vertex, base_value, owner in edge:
                values.append(base_value)
                vertex_ids.append(vertex)
                edge_ids.append(edge_index)
                owners.append(owner)
        values = np.asarray(values, dtype=dtype)
        vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
        edge_ids = np.asarray(edge_ids, dtype=np.int64)
        owners = np.asarray(owners, dtype=np.int64)
        if source_weights is not None:
            source_weights = np.asarray(source_weights, dtype=dtype)
            if source_weights.shape != (self.num_users,):
                raise ValueError("source weight count must equal the number of users")
            if not np.all(np.isfinite(source_weights)) or np.any(source_weights < 0) or np.any(source_weights > 1):
                raise ValueError("source weights must be finite and in [0, 1]")
            owner_mask = owners >= 0
            values[owner_mask] *= source_weights[owners[owner_mask]]
        membership = csr_matrix(
            (values, (vertex_ids, edge_ids)),
            shape=(self.num_vertices, self.num_hyperedges),
            dtype=dtype,
        )
        vertex_degree = np.asarray(membership.sum(axis=1)).reshape(-1).astype(dtype, copy=False)
        edge_degree = np.asarray(membership.sum(axis=0)).reshape(-1).astype(dtype, copy=False)
        inv_vertex = np.zeros_like(vertex_degree, dtype=dtype)
        nonzero_vertices = vertex_degree > 0
        inv_vertex[nonzero_vertices] = 1.0 / np.sqrt(vertex_degree[nonzero_vertices])
        inv_edge = np.zeros_like(edge_degree, dtype=dtype)
        nonzero_edges = edge_degree > 0
        inv_edge[nonzero_edges] = 1.0 / edge_degree[nonzero_edges]
        return SparseHypergraphPropagation(membership, inv_vertex, inv_edge)

    def dense_operator_for_tests(
        self, source_weights: Sequence[float] | None = None
    ) -> list[list[float]]:
        """Materialize P only for small numerical tests, never for model work."""
        columns: list[list[float]] = []
        for column in range(self.num_vertices):
            basis = [[0.0] * 1 for _ in range(self.num_vertices)]
            basis[column][0] = 1.0
            result = self.apply(basis, source_weights)
            columns.append([row[0] for row in result])
        return [
            [columns[column][row] for column in range(self.num_vertices)]
            for row in range(self.num_vertices)
        ]


def build_user_history(interactions: Sequence[Interaction]) -> Hypergraph:
    """Build UH: one edge per user containing that user and their items."""
    if not interactions:
        raise ValueError("cannot construct a hypergraph without interactions")
    user_ids = tuple(sorted({row.user_id for row in interactions}))
    item_ids = tuple(sorted({row.item_id for row in interactions}))
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    histories: dict[int, set[int]] = {user_id: set() for user_id in user_ids}
    for row in interactions:
        histories[row.user_id].add(row.item_id)
    edges: list[tuple[Membership, ...]] = []
    for user_id in user_ids:
        owner = user_index[user_id]
        members: list[Membership] = [(owner, 1.0, owner)]
        members.extend(
            (len(user_ids) + item_index[item_id], 1.0, owner)
            for item_id in sorted(histories[user_id])
        )
        edges.append(tuple(members))
    return Hypergraph("user_history", len(user_ids), len(item_ids), user_ids, item_ids, tuple(edges))


def build_hour_inspired(
    interactions: Sequence[Interaction],
    *,
    seed: int,
    rank: int = 64,
    memberships_per_vertex: int = 2,
    membership_floor: float = 0.05,
    source_weights: Sequence[float] | None = None,
    policy: str = "frozen",
) -> Hypergraph:
    """Build the guide's HOUR-inspired NMF factor hypergraph.

    The method is deliberately named ``hour_inspired``: the specification's
    source-system action for reproducing original HOUR factor extraction is
    still open. In frozen mode, user memberships are source-owned and scaled
    later by ``prepare_numpy``. In rebuilt mode, source weights scale rows of Y
    before NMF and do not scale the resulting memberships a second time.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import NMF

    if not interactions:
        raise ValueError("cannot construct a hypergraph without interactions")
    if rank < 1 or memberships_per_vertex < 1:
        raise ValueError("rank and memberships_per_vertex must be positive")
    if not 0.0 <= membership_floor <= 1.0:
        raise ValueError("membership_floor must be in [0, 1]")
    if policy not in {"frozen", "rebuilt"}:
        raise ValueError("policy must be frozen or rebuilt")

    user_ids = tuple(sorted({row.user_id for row in interactions}))
    item_ids = tuple(sorted({row.item_id for row in interactions}))
    if rank > min(len(user_ids), len(item_ids)):
        raise ValueError("NMF rank cannot exceed the smaller matrix dimension")
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    row_indices = np.fromiter((user_index[row.user_id] for row in interactions), dtype=np.int64)
    column_indices = np.fromiter((item_index[row.item_id] for row in interactions), dtype=np.int64)
    values = np.ones(len(interactions), dtype=np.float32)
    matrix = csr_matrix((values, (row_indices, column_indices)), shape=(len(user_ids), len(item_ids)))
    matrix.data[:] = 1.0  # duplicate user/item events remain binary feedback

    if source_weights is not None:
        weights = np.asarray(source_weights, dtype=np.float32)
        if weights.shape != (len(user_ids),) or not np.all(np.isfinite(weights)):
            raise ValueError("source_weights must be one finite value per user")
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("source weights must be in [0, 1]")
        if policy == "rebuilt":
            matrix = csr_matrix(matrix.multiply(weights[:, None]))

    factorizer = NMF(
        n_components=rank,
        init="nndsvd",
        random_state=seed,
        max_iter=200,
    )
    user_loadings = factorizer.fit_transform(matrix)
    item_loadings = factorizer.components_.T
    loadings = np.vstack((user_loadings, item_loadings)).astype(np.float64, copy=False)
    row_sums = loadings.sum(axis=1)
    nonzero = row_sums > 0.0
    loadings[nonzero] /= row_sums[nonzero, None]

    factor_members: list[list[Membership]] = [[] for _ in range(rank)]
    for vertex, row in enumerate(loadings):
        if not np.any(row > 0.0):
            continue
        top = sorted(range(rank), key=lambda factor: (-row[factor], factor))[:memberships_per_vertex]
        for factor in top:
            value = float(row[factor])
            if value <= 0.0 or value < membership_floor:
                continue
            if vertex < len(user_ids):
                owner = vertex if policy == "frozen" else -1
            else:
                owner = -1
            factor_members[factor].append((vertex, value, owner))
    nonempty_edges = tuple(tuple(members) for members in factor_members if members)
    if not nonempty_edges:
        raise ValueError("NMF produced no memberships after applying the loading floor")
    return Hypergraph(
        "hour_inspired",
        len(user_ids),
        len(item_ids),
        user_ids,
        item_ids,
        nonempty_edges,
        policy,
    )

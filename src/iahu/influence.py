"""Differentiable exposure influence for visible hypergraph states (S6).

Implements the guide's frozen-negative objective, source-weighted matrix-free
operator, exposure surrogate, damped Hessian solve, and implicit influence
vector. This module intentionally never accepts evaluator or vault state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .hypergraph import Hypergraph


@dataclass(frozen=True, slots=True)
class CGResult:
    solution: object
    iterations: int
    relative_residual: float
    negative_curvature: bool
    converged: bool


@dataclass(frozen=True, slots=True)
class InfluenceResult:
    source_scores: tuple[float, ...]
    exposure_value: float
    cg_iterations: int
    cg_relative_residual: float
    negative_curvature: bool
    stationary: bool


@dataclass(frozen=True, slots=True)
class PolishResult:
    theta: object
    initial_gradient_norm: float
    final_gradient_norm: float
    iterations: int
    converged: bool


class TorchHypergraphOperator:
    """Autograd-friendly implementation of P(c) without materializing P."""

    def __init__(self, graph: Hypergraph, *, dtype=None, device: str = "cpu"):
        import torch

        self.graph = graph
        self.dtype = dtype or torch.float64
        self.device = torch.device(device)
        vertices, edges, values, owners = [], [], [], []
        for edge_index, edge in enumerate(graph.edges):
            for vertex, value, owner in edge:
                vertices.append(vertex)
                edges.append(edge_index)
                values.append(value)
                owners.append(owner)
        self.vertices = torch.tensor(vertices, dtype=torch.long, device=self.device)
        self.edge_indices = torch.tensor(edges, dtype=torch.long, device=self.device)
        self.base_values = torch.tensor(values, dtype=self.dtype, device=self.device)
        self.owners = torch.tensor(owners, dtype=torch.long, device=self.device)

    def apply(self, features, source_weights):
        import torch

        if features.ndim != 2 or features.shape[0] != self.graph.num_vertices:
            raise ValueError("feature tensor must have one row per hypergraph vertex")
        if source_weights.shape != (self.graph.num_users,):
            raise ValueError("source weight tensor must have one entry per user")
        if torch.any(source_weights < 0) or torch.any(source_weights > 1):
            raise ValueError("source weights must lie in [0, 1]")
        owner_values = source_weights[self.owners.clamp_min(0)]
        owner_values = torch.where(self.owners >= 0, owner_values, torch.ones_like(owner_values))
        values = self.base_values * owner_values

        vertex_degree = torch.zeros(self.graph.num_vertices, dtype=self.dtype, device=self.device)
        edge_degree = torch.zeros(self.graph.num_hyperedges, dtype=self.dtype, device=self.device)
        vertex_degree = vertex_degree.index_add(0, self.vertices, values)
        edge_degree = edge_degree.index_add(0, self.edge_indices, values)
        inv_sqrt_vertex = torch.where(vertex_degree > 0, vertex_degree.clamp_min(1e-300).rsqrt(), 0)
        inv_edge = torch.where(edge_degree > 0, edge_degree.clamp_min(1e-300).reciprocal(), 0)

        edge_state = torch.zeros(
            (self.graph.num_hyperedges, features.shape[1]), dtype=self.dtype, device=self.device
        )
        edge_members = features[self.vertices] * (values * inv_sqrt_vertex[self.vertices])[:, None]
        edge_state = edge_state.index_add(0, self.edge_indices, edge_members)
        edge_state = edge_state * inv_edge[:, None]
        vertex_state = torch.zeros_like(features)
        vertex_members = edge_state[self.edge_indices] * values[:, None]
        vertex_state = vertex_state.index_add(0, self.vertices, vertex_members)
        return vertex_state * inv_sqrt_vertex[:, None]


def propagated_embeddings(theta, operator: TorchHypergraphOperator, weights, layers: int):
    import torch

    if layers < 1:
        raise ValueError("layers must be positive")
    states = [theta]
    for _ in range(layers):
        states.append(operator.apply(states[-1], weights))
    return torch.stack(states).mean(dim=0)


def frozen_negative_objective(
    theta,
    operator: TorchHypergraphOperator,
    weights,
    user_indices,
    positive_item_indices,
    negative_item_indices,
    *,
    layers: int,
    regularization: float = 1e-4,
):
    """Full-batch guide objective with fixed negatives and a 1/r mean."""
    import torch.nn.functional as F

    if negative_item_indices.ndim != 2 or negative_item_indices.shape[0] != user_indices.shape[0]:
        raise ValueError("negative indices must have shape (interactions, negatives_per_row)")
    embeddings = propagated_embeddings(theta, operator, weights, layers)
    users = embeddings[user_indices]
    positives = embeddings[operator.graph.num_users + positive_item_indices]
    negatives = embeddings[operator.graph.num_users + negative_item_indices]
    positive_score = (users * positives).sum(dim=1, keepdim=True)
    negative_scores = (users[:, None, :] * negatives).sum(dim=2)
    per_row = F.softplus(-(positive_score - negative_scores)).mean(dim=1)
    data_loss = (per_row * weights[user_indices]).sum()
    return data_loss + (regularization / 2.0) * theta.square().sum()


def exposure_surrogate(
    theta,
    operator: TorchHypergraphOperator,
    weights,
    pair_users,
    pair_items,
    pair_thresholds,
    *,
    temperature: float,
    layers: int,
):
    """Frozen-cohort sigmoid surrogate, equally weighted by observed target."""
    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not (len(pair_users) == len(pair_items) == len(pair_thresholds)) or not len(pair_users):
        raise ValueError("exposure cohort must contain equally sized non-empty pair arrays")
    embeddings = propagated_embeddings(theta, operator, weights, layers)
    scores = (embeddings[pair_users] * embeddings[operator.graph.num_users + pair_items]).sum(dim=1)
    probabilities = torch.sigmoid((scores - pair_thresholds) / temperature)
    target_ids = torch.unique(pair_items, sorted=True)
    means = [probabilities[pair_items == item].mean() for item in target_ids]
    return torch.stack(means).mean()


def polish_frozen_negative(
    theta,
    graph: Hypergraph,
    weights,
    user_indices,
    positive_item_indices,
    negative_item_indices,
    *,
    layers: int = 2,
    regularization: float = 1e-4,
    relative_tolerance: float = 1e-3,
    max_iterations: int = 200,
) -> PolishResult:
    """L-BFGS polish of the fixed-negative objective required before influence."""
    import torch

    if not 0 < relative_tolerance < 1 or max_iterations < 1:
        raise ValueError("polish tolerance must be in (0, 1) and iteration limit positive")
    operator = TorchHypergraphOperator(graph, dtype=theta.dtype, device=str(theta.device))
    parameter = theta.detach().clone().requires_grad_(True)

    def objective():
        return frozen_negative_objective(
            parameter, operator, weights, user_indices, positive_item_indices,
            negative_item_indices, layers=layers, regularization=regularization,
        )

    initial_loss = objective()
    initial_gradient = torch.autograd.grad(initial_loss, parameter)[0]
    initial_norm = torch.linalg.vector_norm(initial_gradient).item()
    if initial_norm == 0:
        return PolishResult(parameter.detach(), 0.0, 0.0, 0, True)
    optimizer = torch.optim.LBFGS(
        [parameter], lr=1.0, max_iter=1, history_size=20,
        line_search_fn="strong_wolfe",
    )
    final_norm = initial_norm
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            value = objective()
            value.backward()
            return value

        optimizer.step(closure)
        with torch.enable_grad():
            current = objective()
            gradient = torch.autograd.grad(current, parameter)[0]
        final_norm = torch.linalg.vector_norm(gradient).item()
        if final_norm <= relative_tolerance * initial_norm:
            return PolishResult(parameter.detach(), initial_norm, final_norm, iterations, True)
    return PolishResult(parameter.detach(), initial_norm, final_norm, iterations, False)


def conjugate_gradient(
    matvec,
    rhs,
    *,
    max_iterations: int = 100,
    relative_tolerance: float = 1e-3,
) -> CGResult:
    """Solve a damped SPD system; stop and report non-positive curvature."""
    import torch

    if max_iterations < 1 or relative_tolerance <= 0:
        raise ValueError("CG iteration limit and tolerance must be positive")
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual.reshape(-1), residual.reshape(-1))
    rhs_norm = torch.sqrt(residual_sq)
    if rhs_norm.item() == 0:
        return CGResult(solution, 0, 0.0, False, True)
    for iteration in range(1, max_iterations + 1):
        product = matvec(direction)
        curvature = torch.dot(direction.reshape(-1), product.reshape(-1))
        if not torch.isfinite(curvature) or curvature.item() <= 0:
            relative = torch.sqrt(residual_sq).item() / rhs_norm.item()
            return CGResult(solution, iteration - 1, relative, True, False)
        step = residual_sq / curvature
        solution = solution + step * direction
        residual = residual - step * product
        next_sq = torch.dot(residual.reshape(-1), residual.reshape(-1))
        relative = torch.sqrt(next_sq).item() / rhs_norm.item()
        if relative <= relative_tolerance:
            return CGResult(solution, iteration, relative, False, True)
        direction = residual + (next_sq / residual_sq) * direction
        residual_sq = next_sq
    return CGResult(solution, max_iterations, relative, False, False)


def overlap_proximity(
    graph: Hypergraph,
    weights,
    flagged_users: Sequence[tuple[int, float]],
    *,
    layers: int,
    batch_size: int = 16,
) -> dict[int, float | None]:
    """Compute leave-self-out flagged reach from Q=sum(alpha_l P(c)^l), in batches."""
    import torch

    if layers < 1 or batch_size < 1:
        raise ValueError("layers and batch_size must be positive")
    user_index = {user_id: index for index, user_id in enumerate(graph.user_ids)}
    flags: dict[int, float] = {}
    for user_id, confidence in flagged_users:
        if user_id not in user_index or user_id in flags or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("flag set contains an unknown/duplicate user or invalid confidence")
        flags[user_id] = confidence
    if not flags:
        raise ValueError("InsufficientFlags: overlap requires at least one visible flag")
    operator = TorchHypergraphOperator(graph, dtype=weights.dtype, device=str(weights.device))
    flag_indices = torch.tensor([user_index[user] for user in flags], dtype=torch.long, device=weights.device)
    confidence = torch.tensor(list(flags.values()), dtype=weights.dtype, device=weights.device)
    result: dict[int, float | None] = {}
    with torch.no_grad():
        for start in range(0, graph.num_users, batch_size):
            indices = torch.arange(start, min(start + batch_size, graph.num_users), device=weights.device)
            block = len(indices)
            basis = torch.zeros((graph.num_vertices, block), dtype=weights.dtype, device=weights.device)
            basis[indices, torch.arange(block, device=weights.device)] = 1.0
            power = basis
            q_columns = torch.zeros_like(basis)
            for _ in range(layers):
                power = operator.apply(power, weights)
                q_columns = q_columns + power / layers
            rows = q_columns[:graph.num_users, :].T
            diagonal = rows[torch.arange(block, device=weights.device), indices]
            denominators = rows.sum(dim=1) - diagonal
            numerators = rows[:, flag_indices] @ confidence
            for offset, index in enumerate(indices.tolist()):
                user_id = graph.user_ids[index]
                if user_id in flags:
                    continue
                denominator = denominators[offset].item()
                result[user_id] = (
                    numerators[offset].item() / denominator if denominator > 0 else None
                )
    return result


def _midrank_percentiles(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda user: (values[user], user))
    size = len(ordered)
    result: dict[int, float] = {}
    position = 0
    while position < size:
        end = position + 1
        while end < size and values[ordered[end]] == values[ordered[position]]:
            end += 1
        percentile = ((position + 1 + end) / 2) / size
        for index in range(position, end):
            result[ordered[index]] = percentile
        position = end
    return result


def blend_suspicion_scores(
    source_scores: Sequence[float],
    graph: Hypergraph,
    overlap: dict[int, float | None],
    flagged_user_ids: set[int] | frozenset[int],
    *,
    overlap_weight: float = 0.5,
) -> dict[int, float]:
    """Midrank-normalize positive influence and blend with overlap percentiles."""
    if not 0 <= overlap_weight <= 1 or not math.isfinite(overlap_weight):
        raise ValueError("overlap_weight must be in [0, 1]")
    if len(source_scores) != graph.num_users:
        raise ValueError("source score count must equal graph users")
    if any(not math.isfinite(value) for value in source_scores):
        raise ValueError("source influence scores must be finite")
    id_to_score = dict(zip(graph.user_ids, source_scores))
    nonflagged = [user for user in graph.user_ids if user not in flagged_user_ids]
    positive_rank = _midrank_percentiles({u: id_to_score[u] for u in nonflagged if id_to_score[u] > 0})
    overlap_rank = _midrank_percentiles({u: value for u, value in overlap.items() if value is not None})
    result: dict[int, float] = {}
    for user in nonflagged:
        influence_percentile = positive_rank.get(user, 0.0)
        if overlap_weight > 0 and user not in overlap_rank:
            continue
        proximity_percentile = overlap_rank.get(user, 0.0)
        result[user] = (
            influence_percentile ** (1 - overlap_weight)
            * proximity_percentile ** overlap_weight
        )
    return result


def estimate_source_influence(
    theta,
    graph: Hypergraph,
    weights,
    user_indices,
    positive_item_indices,
    negative_item_indices,
    pair_users,
    pair_items,
    pair_thresholds,
    *,
    layers: int = 2,
    regularization: float = 1e-4,
    temperature: float,
    damping: float = 0.01,
    cg_iterations: int = 100,
    cg_tolerance: float = 1e-3,
    stationarity_tolerance: float = 1e-3,
    initial_gradient_norm: float | None = None,
    polished: bool = False,
) -> InfluenceResult:
    """Compute all R_u values with one CG solve and one mixed derivative."""
    import torch

    if not polished:
        raise ValueError("influence requires a polished frozen-negative stationary state")
    if initial_gradient_norm is None or not math.isfinite(initial_gradient_norm) or initial_gradient_norm <= 0:
        raise ValueError("positive initial_gradient_norm from the polish stage is required")
    if damping <= 0 or not math.isfinite(damping):
        raise ValueError("damping must be positive and finite")
    graph_op = TorchHypergraphOperator(graph, dtype=theta.dtype, device=str(theta.device))
    theta = theta.detach().clone().requires_grad_(True)
    c = weights.detach().clone().requires_grad_(True)
    loss = frozen_negative_objective(
        theta, graph_op, c, user_indices, positive_item_indices, negative_item_indices,
        layers=layers, regularization=regularization,
    )
    grad_loss = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True)[0]
    grad_norm = torch.linalg.vector_norm(grad_loss.detach()).item()
    phi = exposure_surrogate(
        theta, graph_op, c, pair_users, pair_items, pair_thresholds,
        temperature=temperature, layers=layers,
    )
    grad_phi = torch.autograd.grad(phi, theta, retain_graph=True)[0].detach()

    def hessian_product(vector):
        hv = torch.autograd.grad(
            (grad_loss * vector.detach()).sum(), theta, retain_graph=True
        )[0]
        return hv + damping * vector

    cg = conjugate_gradient(
        hessian_product, grad_phi, max_iterations=cg_iterations,
        relative_tolerance=cg_tolerance,
    )
    psi = phi - (cg.solution.detach() * grad_loss).sum()
    scores = torch.autograd.grad(psi, c, allow_unused=False)[0]
    return InfluenceResult(
        source_scores=tuple(float(value) for value in scores.detach().cpu().tolist()),
        exposure_value=float(phi.detach().cpu()),
        cg_iterations=cg.iterations,
        cg_relative_residual=cg.relative_residual,
        negative_curvature=cg.negative_curvature,
        stationary=grad_norm <= stationarity_tolerance * initial_gradient_norm,
    )

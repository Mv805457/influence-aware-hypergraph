"""Numerical estimator validation that runs before headline experiment worlds."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationResult:
    name: str
    passed: bool
    observed: dict[str, float]
    reasons: tuple[str, ...]


def validate_v1_convex_influence(*, finite_difference_step: float = 1e-6) -> ValidationResult:
    """Compare damped implicit source derivatives with exact convex retraining.

    V1 is a deterministic double-precision quadratic toy. For c in R^2,
    L(theta;c)=1/2 theta^T A(c) theta-b(c)^T theta, where A(c) is SPD.
    Phi(theta,c)=1/2 theta^T Q theta+p^T c. Exact retraining is the solve
    theta*(c)=A(c)^-1 b(c). The production conjugate-gradient/implicit formula
    is compared against centered finite differences of those exact minimizers.
    """
    import torch

    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive and finite")
    dtype = torch.float64
    device = "cpu"
    base = torch.tensor([[2.4, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 1.8]], dtype=dtype, device=device)
    source_hessians = torch.tensor([
        [[0.4, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.2]],
        [[0.2, 0.0, 0.0], [0.0, 0.5, 0.1], [0.0, 0.1, 0.3]],
    ], dtype=dtype, device=device)
    base_rhs = torch.tensor([0.7, -0.2, 0.4], dtype=dtype, device=device)
    source_rhs = torch.tensor([[0.1, 0.3, -0.1], [-0.2, 0.05, 0.2]], dtype=dtype, device=device)
    q = torch.tensor([[1.0, 0.15, 0.0], [0.15, 0.8, 0.1], [0.0, 0.1, 0.6]], dtype=dtype, device=device)
    direct = torch.tensor([0.03, -0.02], dtype=dtype, device=device)
    c0 = torch.ones(2, dtype=dtype, device=device)

    def matrices(c):
        hessian = base + torch.einsum("i,ijk->jk", c, source_hessians)
        rhs = base_rhs + torch.einsum("i,ij->j", c, source_rhs)
        return hessian, rhs

    hessian, rhs = matrices(c0)
    theta_value = torch.linalg.solve(hessian, rhs)
    theta = theta_value.detach().clone().requires_grad_(True)
    c = c0.detach().clone().requires_grad_(True)
    hessian_c, rhs_c = matrices(c)
    loss = 0.5 * theta @ hessian_c @ theta - rhs_c @ theta
    grad_theta = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True)[0]
    phi = 0.5 * theta @ q @ theta + direct @ c
    grad_phi = torch.autograd.grad(phi, theta, retain_graph=True)[0].detach()

    # The Hessian is constant in theta; this call exercises the shared CG path.
    from .influence import conjugate_gradient

    solved = conjugate_gradient(
        lambda vector: hessian @ vector,
        grad_phi,
        max_iterations=10,
        relative_tolerance=1e-12,
    )
    psi = phi - solved.solution.detach() @ grad_theta
    influence = torch.autograd.grad(psi, c)[0].detach()

    def exact_phi(weights):
        matrix, vector = matrices(weights)
        optimum = torch.linalg.solve(matrix, vector)
        return 0.5 * optimum @ q @ optimum + direct @ weights

    h = finite_difference_step
    errors = []
    for index in range(2):
        plus = c0.clone()
        minus = c0.clone()
        plus[index] += h
        minus[index] -= h
        finite = (exact_phi(plus) - exact_phi(minus)) / (2 * h)
        observed = influence[index]
        errors.append((abs((finite - observed).item()) / max(1e-12, abs(finite.item()) + abs(observed.item()))))
    max_error = max(errors)
    checks = {
        "cg_converged": solved.converged and not solved.negative_curvature,
        "finite_difference_relative_error": max_error < 1e-5,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return ValidationResult(
        "V1_convex_exact_retraining",
        not reasons,
        {
            "maximum_source_relative_error": max_error,
            "cg_relative_residual": solved.relative_residual,
        },
        reasons,
    )

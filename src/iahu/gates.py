"""Pre-registered clean-model and attack-strength acceptance gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    checks: dict[str, bool]
    observed: dict[str, float]
    reasons: tuple[str, ...]


def clean_model_gate(
    hypergraph_ndcg: Sequence[float],
    lightgcn_ndcg: Sequence[float],
    *,
    finite_difference_relative_error: float,
    repeated_run_max_abs_difference: float,
    repeat_tolerance: float,
    max_relative_gap: float = 0.05,
    max_relative_std: float = 0.03,
    max_gradient_error: float = 1e-5,
) -> GateResult:
    """Evaluate G1; five paired clean seeds and explicit numeric checks required."""
    if len(hypergraph_ndcg) != 5 or len(lightgcn_ndcg) != 5:
        raise ValueError("G1 requires exactly five paired seeds for both models")
    values = [*hypergraph_ndcg, *lightgcn_ndcg,
              finite_difference_relative_error, repeated_run_max_abs_difference,
              repeat_tolerance]
    if any(not math.isfinite(value) for value in values) or repeat_tolerance < 0:
        raise ValueError("G1 inputs must be finite and repeat_tolerance non-negative")
    if any(value < 0 for value in (*hypergraph_ndcg, *lightgcn_ndcg)):
        raise ValueError("NDCG values cannot be negative")
    baseline_mean = mean(lightgcn_ndcg)
    if baseline_mean <= 0:
        raise ValueError("G1 LightGCN mean NDCG must be positive")
    relative_gap = abs(mean(hypergraph_ndcg) - baseline_mean) / baseline_mean
    relative_std = stdev(hypergraph_ndcg) / mean(hypergraph_ndcg) if mean(hypergraph_ndcg) else math.inf
    checks = {
        "relative_ndcg_gap": relative_gap <= max_relative_gap,
        "five_seed_relative_std": relative_std < max_relative_std,
        "finite_difference_gradient": finite_difference_relative_error < max_gradient_error,
        "same_seed_reproducibility": repeated_run_max_abs_difference <= repeat_tolerance,
    }
    observed = {
        "hypergraph_ndcg_mean": mean(hypergraph_ndcg),
        "lightgcn_ndcg_mean": baseline_mean,
        "relative_ndcg_gap": relative_gap,
        "hypergraph_ndcg_relative_std": relative_std,
        "finite_difference_relative_error": finite_difference_relative_error,
        "repeated_run_max_abs_difference": repeated_run_max_abs_difference,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failed, checks, observed, failed)


def attack_strength_gate(
    *,
    reference_exposure: float,
    poisoned_exposure: float,
    reference_ndcg: float,
    poisoned_ndcg: float,
    min_absolute_exposure_gain: float = 0.02,
    min_exposure_ratio: float = 5.0,
    min_ndcg_retention: float = 0.95,
) -> GateResult:
    """Evaluate all three G2 requirements on the actual validation-trained models."""
    values = (reference_exposure, poisoned_exposure, reference_ndcg, poisoned_ndcg,
              min_absolute_exposure_gain, min_exposure_ratio, min_ndcg_retention)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("G2 inputs must be finite")
    if any(value < 0 or value > 1 for value in values[:4]):
        raise ValueError("G2 exposure and NDCG values must be in [0, 1]")
    if min_absolute_exposure_gain < 0 or min_exposure_ratio < 0 or not 0 <= min_ndcg_retention <= 1:
        raise ValueError("invalid G2 threshold")
    gain = poisoned_exposure - reference_exposure
    ratio = math.inf if reference_exposure == 0 and poisoned_exposure > 0 else (
        1.0 if reference_exposure == 0 else poisoned_exposure / reference_exposure
    )
    retention = poisoned_ndcg / reference_ndcg if reference_ndcg > 0 else (1.0 if poisoned_ndcg == 0 else math.inf)
    checks = {
        "absolute_exposure_gain": gain >= min_absolute_exposure_gain,
        "exposure_ratio": reference_exposure == 0 or ratio >= min_exposure_ratio,
        "ndcg_retention": retention >= min_ndcg_retention,
    }
    observed = {
        "reference_exposure": reference_exposure,
        "poisoned_exposure": poisoned_exposure,
        "absolute_exposure_gain": gain,
        "exposure_ratio": ratio,
        "ndcg_retention": retention,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failed, checks, observed, failed)

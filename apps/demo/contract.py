"""Data contract for the read-only, precomputed experiment demo."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


BUNDLE_SCHEMA_VERSION = 2
SOURCE_SCHEMA_VERSION = "completed-run-export-v1"
EXAMPLE_USER_LABEL = "Example user 01"

METRIC_LABELS = {
    "rar": "Residual Attack Ratio (aggregate evaluator metric)",
    "hidden_source_recall": "Hidden-source recall (aggregate evaluator metric)",
    "precision": "Extension precision (aggregate evaluator metric)",
    "benign_weight_loss": "Benign weight loss (aggregate evaluator metric)",
    "utility_retention": "Utility retention (aggregate evaluator metric)",
    "control_item_drift": "Control-item drift (aggregate evaluator metric)",
    "ndcg_at_20": "NDCG@20 (aggregate evaluator metric)",
    "recall_at_20": "Recall@20 (aggregate evaluator metric)",
    "topk_jaccard_reference": "Top-K Jaccard vs. clean reference (aggregate)",
    "topk_jaccard_retrain": "Top-K Jaccard vs. retraining (aggregate)",
    "cost_ratio": "Cost ratio vs. retraining (aggregate)",
    "membership_auc_gap": "Membership AUC gap (aggregate evaluator metric)",
}

SEED_NAMES = ("world", "attack", "flagging", "training", "unlearning")
PIPELINE_STEPS = {
    "dataset_preprocessing",
    "interaction_matrix",
    "hypergraph_construction",
    "recommendation_baseline",
}
POISONING_STEPS = {
    "attack_configuration",
    "poisoning_simulation",
    "identification",
    "unlearning",
    "evaluation",
}
TRACE_COUNT_KEYS = {
    "users",
    "items",
    "interactions",
    "hyperedges",
    "memberships",
    "injected_interactions",
    "candidate_sources",
    "flagged_sources",
    "removed_hyperedges",
    "evaluation_users",
}


class ContractError(ValueError):
    """Raised when exported demo data does not satisfy the public contract."""


def _object(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing:
        raise ContractError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ContractError(f"{name} has unsupported fields: {', '.join(sorted(extra))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    if len(value) > 200 or any(ord(char) < 32 for char in value):
        raise ContractError(f"{name} must be at most 200 printable characters")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ContractError(f"{name} must be finite and at least {minimum}")
    return number


def _rank(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{name} must be a positive integer or null")
    return value


def _topk(value: Any, name: str, top_k: int, rank: int | None, target_label: str) -> None:
    if not isinstance(value, list) or len(value) > top_k:
        raise ContractError(f"{name} must be a list with at most {top_k} items")
    target_positions: list[int] = []
    for index, raw_item in enumerate(value, start=1):
        item = _object(raw_item, f"{name}[{index - 1}]", {"label", "is_target"})
        item_label = _text(item["label"], f"{name}[{index - 1}].label")
        if not isinstance(item["is_target"], bool):
            raise ContractError(f"{name}[{index - 1}].is_target must be a boolean")
        if item["is_target"]:
            if item_label != target_label:
                raise ContractError(f"{name} target label does not match target_item_label")
            target_positions.append(index)
    if len(target_positions) > 1:
        raise ContractError(f"{name} may mark the target at most once")
    if target_positions and target_positions[0] != rank:
        raise ContractError(f"{name} target position does not match its saved rank")
    if rank is not None and rank <= top_k and not target_positions:
        raise ContractError(f"{name} omits the target even though its rank is within Top-K")
    if rank is not None and rank > top_k and target_positions:
        raise ContractError(f"{name} includes a target whose saved rank is outside Top-K")
    if rank is None and target_positions:
        raise ContractError(f"{name} includes the target but its saved rank is unavailable")


def _trace(value: Any, name: str, allowed_steps: set[str]) -> None:
    if not isinstance(value, list) or len(value) > len(allowed_steps):
        raise ContractError(f"{name} must be a list with at most {len(allowed_steps)} events")
    seen_steps: set[str] = set()
    for index, raw_event in enumerate(value):
        event_name = f"{name}[{index}]"
        event = _object(raw_event, event_name, {"step", "counts", "elapsed_seconds"})
        step = event["step"]
        if not isinstance(step, str) or step not in allowed_steps:
            raise ContractError(f"{event_name}.step is unsupported")
        if step in seen_steps:
            raise ContractError(f"{name} contains a duplicate step")
        seen_steps.add(step)
        counts = event["counts"]
        if not isinstance(counts, Mapping):
            raise ContractError(f"{event_name}.counts must be an object")
        unknown_counts = set(counts) - TRACE_COUNT_KEYS
        if unknown_counts:
            raise ContractError(f"{event_name}.counts has unsupported keys: {', '.join(sorted(unknown_counts))}")
        for count_name, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ContractError(f"{event_name}.counts.{count_name} must be a non-negative integer")
        elapsed = event["elapsed_seconds"]
        if elapsed is not None:
            _number(elapsed, f"{event_name}.elapsed_seconds", minimum=0)


def validate_bundle(bundle: Any) -> dict[str, Any]:
    """Validate and return a bundle. Unknown fields are rejected, not rendered."""
    root = _object(bundle, "bundle", {"schema_version", "scenarios"})
    if root["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {BUNDLE_SCHEMA_VERSION}")
    if not isinstance(root["scenarios"], list):
        raise ContractError("scenarios must be a list")

    for scenario_index, raw_scenario in enumerate(root["scenarios"]):
        prefix = f"scenarios[{scenario_index}]"
        scenario = _object(
            raw_scenario,
            prefix,
            {
                "label", "tier", "setting", "target_item_label", "example", "methods",
                "pipeline_trace", "poisoning_trace",
            },
        )
        _text(scenario["label"], f"{prefix}.label")
        if scenario["tier"] != "deployable":
            raise ContractError(f"{prefix}.tier must be deployable; oracle results are excluded")
        _text(scenario["target_item_label"], f"{prefix}.target_item_label")
        _trace(scenario["pipeline_trace"], f"{prefix}.pipeline_trace", PIPELINE_STEPS)
        _trace(scenario["poisoning_trace"], f"{prefix}.poisoning_trace", POISONING_STEPS)

        setting = _object(
            scenario["setting"],
            f"{prefix}.setting",
            {"dataset", "hypergraph_construction", "attack", "seeds", "identification"},
        )
        for field in ("dataset", "hypergraph_construction", "attack"):
            _text(setting[field], f"{prefix}.setting.{field}")
        seeds = _object(setting["seeds"], f"{prefix}.setting.seeds", set(SEED_NAMES))
        for name in SEED_NAMES:
            seed = seeds[name]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ContractError(f"{prefix}.setting.seeds.{name} must be an integer")
        identification = _object(
            setting["identification"],
            f"{prefix}.setting.identification",
            {"rate", "false_positive_share", "strategy"},
        )
        for field in ("rate", "false_positive_share"):
            rate = _number(identification[field], f"{prefix}.setting.identification.{field}")
            if not 0 <= rate <= 1:
                raise ContractError(f"{prefix}.setting.identification.{field} must be between 0 and 1")
        _text(identification["strategy"], f"{prefix}.setting.identification.strategy")

        example = _object(
            scenario["example"],
            f"{prefix}.example",
            {"user_label", "top_k", "before", "target_rank_before"},
        )
        _text(example["user_label"], f"{prefix}.example.user_label")
        top_k = example["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ContractError(f"{prefix}.example.top_k must be a positive integer")
        before_rank = _rank(example["target_rank_before"], f"{prefix}.example.target_rank_before")
        _topk(
            example["before"],
            f"{prefix}.example.before",
            top_k,
            before_rank,
            scenario["target_item_label"],
        )

        methods = scenario["methods"]
        if not isinstance(methods, list) or not methods:
            raise ContractError(f"{prefix}.methods must be a non-empty list")
        for method_index, raw_method in enumerate(methods):
            method_prefix = f"{prefix}.methods[{method_index}]"
            method = _object(
                raw_method,
                method_prefix,
                {"label", "after", "target_rank_after", "aggregate_metrics", "elapsed_seconds"},
            )
            _text(method["label"], f"{method_prefix}.label")
            after_rank = _rank(method["target_rank_after"], f"{method_prefix}.target_rank_after")
            _topk(
                method["after"],
                f"{method_prefix}.after",
                top_k,
                after_rank,
                scenario["target_item_label"],
            )
            metrics = method["aggregate_metrics"]
            if not isinstance(metrics, Mapping):
                raise ContractError(f"{method_prefix}.aggregate_metrics must be an object")
            unknown_metrics = set(metrics) - set(METRIC_LABELS)
            if unknown_metrics:
                raise ContractError(
                    f"{method_prefix}.aggregate_metrics has unsupported metrics: "
                    f"{', '.join(sorted(unknown_metrics))}"
                )
            for metric_name, value in metrics.items():
                _number(value, f"{method_prefix}.aggregate_metrics.{metric_name}")
            _number(method["elapsed_seconds"], f"{method_prefix}.elapsed_seconds", minimum=0)

    return dict(root)


def sanitize_source(source: Any) -> dict[str, Any]:
    """Copy only public demo fields from a completed-run export.

    Additional source keys (including manifests, user IDs, labels, vault paths, and model
    paths) are intentionally discarded. The resulting whitelist is validated before use.
    """
    if not isinstance(source, Mapping):
        raise ContractError("source export must be an object")
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ContractError(f"source schema_version must be {SOURCE_SCHEMA_VERSION}")
    raw_scenarios = source.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ContractError("source scenarios must be a list")

    scenarios: list[dict[str, Any]] = []
    for scenario in raw_scenarios:
        try:
            setting = scenario["setting"]
            identification = setting["identification"]
            seeds = setting["seeds"]
            example = scenario["example"]
            methods = scenario["methods"]
            sanitized_methods = []
            for method in methods:
                metrics = method["aggregate_metrics"]
                sanitized_methods.append(
                    {
                        "label": method["label"],
                        "after": [
                            {"label": item["label"], "is_target": item["is_target"]}
                            for item in method["after"]
                        ],
                        "target_rank_after": method["target_rank_after"],
                        "aggregate_metrics": {
                            name: metrics[name] for name in METRIC_LABELS if name in metrics
                        },
                        "elapsed_seconds": method["elapsed_seconds"],
                    }
                )
            sanitized = {
                "label": scenario["label"],
                "tier": scenario["tier"],
                "setting": {
                    "dataset": setting["dataset"],
                    "hypergraph_construction": setting["hypergraph_construction"],
                    "attack": setting["attack"],
                    "seeds": {name: seeds[name] for name in SEED_NAMES},
                    "identification": {
                        "rate": identification["rate"],
                        "false_positive_share": identification["false_positive_share"],
                        "strategy": identification["strategy"],
                    },
                },
                "target_item_label": scenario["target_item_label"],
                "example": {
                    "user_label": EXAMPLE_USER_LABEL,
                    "top_k": example["top_k"],
                    "before": [
                        {"label": item["label"], "is_target": item["is_target"]}
                        for item in example["before"]
                    ],
                    "target_rank_before": example["target_rank_before"],
                },
                "pipeline_trace": _sanitize_trace(scenario.get("pipeline_trace", [])),
                "poisoning_trace": _sanitize_trace(scenario.get("poisoning_trace", [])),
                "methods": sanitized_methods,
            }
        except (KeyError, TypeError) as error:
            raise ContractError(f"malformed completed-run scenario: missing or invalid {error}") from error
        scenarios.append(sanitized)

    return validate_bundle({"schema_version": BUNDLE_SCHEMA_VERSION, "scenarios": scenarios})


def _sanitize_trace(value: Any) -> list[dict[str, Any]]:
    """Copy only fixed step codes, aggregate counters, and elapsed times."""
    if not isinstance(value, list):
        raise ContractError("trace must be a list")
    events: list[dict[str, Any]] = []
    for event in value:
        if not isinstance(event, Mapping):
            raise ContractError("trace event must be an object")
        counts = event.get("counts", {})
        if not isinstance(counts, Mapping):
            raise ContractError("trace counts must be an object")
        events.append(
            {
                "step": event.get("step"),
                "counts": {key: counts[key] for key in sorted(TRACE_COUNT_KEYS) if key in counts},
                "elapsed_seconds": event.get("elapsed_seconds"),
            }
        )
    return events

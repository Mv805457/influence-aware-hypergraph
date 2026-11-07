from __future__ import annotations

import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "demo"))

import contract  # noqa: E402
import server  # noqa: E402
import export_demo_bundle as exporter  # noqa: E402


def completed_source() -> dict:
    """Synthetic contract fixture only; it is never written to the demo bundle."""
    return {
        "schema_version": "completed-run-export-v1",
        "vault_contents": {"hidden_attacker_ids": ["attacker-test-id"]},
        "raw_manifest_path": "vault/manifest.json",
        "scenarios": [
            {
                "id": "raw-run-id-from-manifest",
                "label": "Test scenario",
                "tier": "deployable",
                "setting": {
                    "dataset": "Test dataset",
                    "hypergraph_construction": "Test construction",
                    "attack": "Test attack",
                    "seeds": {
                        "world": 0,
                        "attack": 0,
                        "flagging": 0,
                        "training": 0,
                        "unlearning": 0,
                    },
                    "identification": {
                        "rate": 0.0,
                        "false_positive_share": 0.0,
                        "strategy": "Test strategy",
                    },
                },
                "target_item_label": "Test target",
                "example": {
                    "user_label": "raw-user-test-id",
                    "internal_user_id": "raw-user-test-id",
                    "evaluator_label": "benign",
                    "top_k": 3,
                    "before": [
                        {"label": "Item A", "is_target": False, "raw_item_id": "item-test-id"},
                        {"label": "Test target", "is_target": True},
                    ],
                    "target_rank_before": 2,
                },
                "methods": [
                    {
                        "id": "raw-method-id-from-manifest",
                        "label": "Test method",
                        "after": [
                            {"label": "Item B", "is_target": False},
                            {"label": "Item C", "is_target": False},
                        ],
                        "target_rank_after": 4,
                        "aggregate_metrics": {},
                        "elapsed_seconds": 0.0,
                        "per_user_evaluator_labels": {"raw-user-test-id": "benign"},
                        "checkpoint_path": "checkpoints/model.pt",
                    }
                ],
            }
        ],
    }


def bare_handler(path: str = "/api/bundle") -> server.DemoHandler:
    handler = server.DemoHandler.__new__(server.DemoHandler)
    handler.path = path
    handler.command = "GET"
    handler._send = Mock()
    handler._send_json = Mock()
    handler.wfile = io.BytesIO()
    return handler


class ContractTests(unittest.TestCase):
    def test_empty_bundle_is_valid(self) -> None:
        empty = {"schema_version": 1, "scenarios": []}
        self.assertEqual(contract.validate_bundle(empty), empty)

    def test_exporter_whitelists_and_drops_source_identifiers(self) -> None:
        bundle = exporter.export_bundle(completed_source())
        self.assertNotIn("id", bundle["scenarios"][0])
        self.assertNotIn("id", bundle["scenarios"][0]["methods"][0])
        serialized = json.dumps(bundle)
        for omitted in (
            "attacker-test-id",
            "raw-run-id-from-manifest",
            "raw-method-id-from-manifest",
            "raw-user-test-id",
            "raw_item_id",
            "evaluator_label",
            "per_user_evaluator_labels",
            "checkpoint_path",
            "vault_contents",
            "raw_manifest_path",
        ):
            self.assertNotIn(omitted, serialized)
        self.assertEqual(len(bundle["scenarios"][0]["example"]["before"]), 2)
        self.assertEqual(bundle["scenarios"][0]["example"]["user_label"], "Example user 01")
        self.assertEqual(bundle["scenarios"][0]["methods"][0]["target_rank_after"], 4)
        self.assertEqual(bundle["scenarios"][0]["methods"][0]["aggregate_metrics"], {})

    def test_oracle_tier_is_rejected(self) -> None:
        source = completed_source()
        source["scenarios"][0]["tier"] = "oracle"
        with self.assertRaisesRegex(contract.ContractError, "oracle results are excluded"):
            exporter.export_bundle(source)

    def test_contract_rejects_unknown_bundle_fields(self) -> None:
        bundle = exporter.export_bundle(completed_source())
        bundle["scenarios"][0]["vault_contents"] = {}
        with self.assertRaisesRegex(contract.ContractError, "unsupported fields"):
            contract.validate_bundle(bundle)

    def test_contract_rejects_rank_that_disagrees_with_topk(self) -> None:
        bundle = exporter.export_bundle(completed_source())
        bundle["scenarios"][0]["example"]["target_rank_before"] = 1
        with self.assertRaisesRegex(contract.ContractError, "position does not match"):
            contract.validate_bundle(bundle)

    def test_contract_rejects_target_label_mismatch(self) -> None:
        bundle = exporter.export_bundle(completed_source())
        bundle["scenarios"][0]["example"]["before"][1]["label"] = "Different item"
        with self.assertRaisesRegex(contract.ContractError, "target label does not match"):
            contract.validate_bundle(bundle)

    def test_export_cli_writes_bundle_from_completed_exports_folder(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "apps" / "demo") as temp_dir:
            temporary_root = Path(temp_dir)
            input_dir = temporary_root / "results" / "completed-run-exports"
            input_dir.mkdir(parents=True)
            source_path = input_dir / "completed-run-export.json"
            source_path.write_text(json.dumps(completed_source()), encoding="utf-8")
            output_path = temporary_root / "results" / "demo-bundle.json"
            with (
                patch.object(exporter, "EXPORT_INPUT_DIR", input_dir),
                patch.object(exporter, "DEFAULT_INPUT", source_path),
                patch.object(exporter, "DEFAULT_OUTPUT", output_path),
            ):
                self.assertEqual(exporter.main([]), 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(contract.validate_bundle(written), written)
            self.assertNotIn("raw-user-test-id", json.dumps(written))

    def test_export_cli_refuses_paths_outside_completed_exports(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = exporter.main(["--input", str(ROOT / "apps" / "demo" / "server.py")])
        self.assertEqual(status, 2)
        self.assertIn("input must be inside", stderr.getvalue())


class MissingAndInvalidFileTests(unittest.TestCase):
    def test_missing_bundle_returns_explicit_empty_state(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "apps" / "demo") as temp_dir:
            with patch.object(server, "BUNDLE_PATH", Path(temp_dir) / "missing.json"):
                handler = bare_handler()
                handler._serve_bundle()
        handler._send_json.assert_called_once()
        status, payload = handler._send_json.call_args.args
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"schema_version": 1, "scenarios": []})

    def test_malformed_json_returns_readable_error(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "apps" / "demo") as temp_dir:
            bad_file = Path(temp_dir) / "demo-bundle.json"
            bad_file.write_text("{ not json", encoding="utf-8")
            with patch.object(server, "BUNDLE_PATH", bad_file):
                handler = bare_handler()
                handler._serve_bundle()
        status, payload = handler._send_json.call_args.args
        self.assertEqual(status, 422)
        self.assertIn("invalid JSON", payload["error"])

    def test_contract_error_returns_readable_error(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "apps" / "demo") as temp_dir:
            bad_file = Path(temp_dir) / "demo-bundle.json"
            bad_file.write_text('{"schema_version": 99, "scenarios": []}', encoding="utf-8")
            with patch.object(server, "BUNDLE_PATH", bad_file):
                handler = bare_handler()
                handler._serve_bundle()
        status, payload = handler._send_json.call_args.args
        self.assertEqual(status, 422)
        self.assertIn("data contract", payload["error"])


class ReadOnlyBoundaryTests(unittest.TestCase):
    def test_only_fixed_static_paths_and_bundle_route_are_served(self) -> None:
        handler = bare_handler("/../../vault/manifest.json")
        server.DemoHandler.do_GET(handler)
        status, body, content_type = handler._send.call_args.args
        self.assertEqual(status, 404)
        self.assertEqual(content_type, "text/plain; charset=utf-8")
        self.assertEqual(body, b"Not found\n")
        self.assertEqual(set(server.STATIC_FILES), {"/", "/index.html", "/app.js", "/styles.css"})

    def test_mutating_http_methods_are_rejected(self) -> None:
        for method_name in ("do_POST", "do_PUT", "do_PATCH", "do_DELETE", "do_HEAD"):
            with self.subTest(method=method_name):
                handler = bare_handler()
                getattr(server.DemoHandler, method_name)(handler)
                self.assertEqual(handler._send.call_args.args[0], 405)

    def test_app_server_has_no_research_or_checkpoint_imports(self) -> None:
        source = (ROOT / "apps" / "demo" / "server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(name == "iahu" or name.startswith("iahu.") for name in imports))
        self.assertFalse(any(name.split(".", 1)[0] in {"torch", "pickle", "subprocess"} for name in imports))
        self.assertEqual(server.BUNDLE_PATH, ROOT / "results" / "demo-bundle.json")


if __name__ == "__main__":
    unittest.main()

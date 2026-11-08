from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction
from iahu.model import BPRConfig
from iahu.runner import (
    _g1_report_passes_for_run,
    _read_uh_profile,
    check_unlearning_import_boundary,
    spawn_seed_streams,
    _visible_split,
)


class SeedAndVisibilityTests(unittest.TestCase):
    def test_five_child_seed_streams_are_stable_and_distinct(self) -> None:
        first = spawn_seed_streams(20261108)
        second = spawn_seed_streams(20261108)
        self.assertEqual(first, second)
        self.assertEqual(len({first.world, first.attack, first.flagging, first.training, first.unlearning}), 5)

    def test_visible_validation_is_seeded_uniform_row_selection(self) -> None:
        rows = tuple(Interaction(user, item, 5, None) for user in range(1, 11) for item in range(1, 11))
        train_a, validation_a, indices_a = _visible_split(rows, seed=17)
        train_b, validation_b, indices_b = _visible_split(rows, seed=17)
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(len(validation_a), 5)
        self.assertEqual(len(train_a), 95)
        self.assertEqual(set(train_a).isdisjoint(validation_a), True)


class InformationBoundaryTests(unittest.TestCase):
    def test_current_unlearner_and_influence_modules_have_no_vault_or_evaluator_imports(self) -> None:
        self.assertTrue(check_unlearning_import_boundary(ROOT / "src" / "iahu"))

    def test_boundary_detects_forbidden_import(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            (root / "unlearn.py").write_text("from iahu.vault import hidden_ids\n", encoding="utf-8")
            (root / "influence.py").write_text("def safe(): return 1\n", encoding="utf-8")
            self.assertFalse(check_unlearning_import_boundary(root))


class DevelopmentProfileTests(unittest.TestCase):
    def _write_profile(self, directory: Path, **overrides: object) -> Path:
        allowlist = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))
        selected_config = {
            "dimension": 32,
            "layers": 1,
            "learning_rate": 0.001,
            "regularization": BPRConfig().regularization,
            "batch_size": BPRConfig().batch_size,
            "epochs": 3,
        }
        profile = {
            "schema_version": 1,
            "dataset": "ml-100k",
            "dataset_sha256": allowlist["datasets"]["ml-100k"]["sha256"],
            "tier": "development_only",
            "fixed_epochs": 3,
            "selected_config": selected_config,
            "uh_config": selected_config.copy(),
            "candidate_configs": [selected_config.copy()],
        }
        profile.update(overrides)
        path = directory / "uh-profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_reads_pinned_development_profile(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            config = _read_uh_profile(self._write_profile(Path(directory)), epochs=3)
        self.assertEqual(config, BPRConfig(dimension=32, layers=1, learning_rate=0.001, epochs=3))

    def test_rejects_wrong_epoch_tier_or_out_of_grid_profile(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = self._write_profile(Path(directory))
            with self.assertRaisesRegex(ValueError, "fixed epoch"):
                _read_uh_profile(path, epochs=4)
            self._write_profile(Path(directory), tier="main_candidate")
            with self.assertRaisesRegex(ValueError, "development-only"):
                _read_uh_profile(path, epochs=3)
            self._write_profile(
                Path(directory),
                selected_config={
                    "dimension": 32,
                    "layers": 1,
                    "learning_rate": 0.001,
                    "regularization": BPRConfig().regularization,
                    "batch_size": BPRConfig().batch_size,
                    "epochs": 4,
                },
            )
            with self.assertRaisesRegex(ValueError, "selected_config must use"):
                _read_uh_profile(path, epochs=3)
            self._write_profile(
                Path(directory),
                selected_config={
                    "dimension": 128,
                    "layers": 1,
                    "learning_rate": 0.001,
                    "regularization": BPRConfig().regularization,
                    "batch_size": BPRConfig().batch_size,
                    "epochs": 3,
                },
            )
            with self.assertRaisesRegex(ValueError, "outside the documented"):
                _read_uh_profile(path, epochs=3)

    def test_g1_must_match_the_profile_used_by_main_world(self) -> None:
        config = BPRConfig(dimension=32, layers=1, learning_rate=0.001, epochs=3)
        report = {
            "gate": {"passed": True},
            "dataset": "ml-1m",
            "dataset_sha256": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
            "fixed_epochs": 3,
            "uh_config": {
                "dimension": 32,
                "layers": 1,
                "learning_rate": 0.001,
                "regularization": BPRConfig().regularization,
                "batch_size": BPRConfig().batch_size,
                "epochs": 3,
            },
            "uh_profile_sha256": "b" * 64,
        }
        self.assertTrue(_g1_report_passes_for_run(
            report, epochs=3, uh_config=config, uh_profile_sha256="b" * 64,
        ))
        self.assertFalse(_g1_report_passes_for_run(
            report, epochs=3, uh_config=config, uh_profile_sha256="c" * 64,
        ))


if __name__ == "__main__":
    unittest.main()

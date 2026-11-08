from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction
from iahu.model import BPRConfig
from iahu.runner import (
    _g1_report_passes_for_run,
    _read_uh_profile,
    _ranking_exclusions,
    _restore_uh_progress,
    _select_uh_parity_candidate,
    _uh_development_grid,
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

    def test_ranking_exclusions_include_the_withheld_visible_validation_rows(self) -> None:
        training = (Interaction(1, 10, 5, None),)
        visible_validation = (Interaction(1, 11, 5, None),)
        oracle_validation = (Interaction(1, 12, 5, None),)
        self.assertEqual(
            _ranking_exclusions(training, visible_validation, oracle_validation),
            training + visible_validation + oracle_validation,
        )


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
        candidates = [asdict(candidate) for candidate in _uh_development_grid(3)]
        selected_config = candidates[0]
        profile = {
            "schema_version": 1,
            "dataset": "ml-100k",
            "dataset_sha256": allowlist["datasets"]["ml-100k"]["sha256"],
            "tier": "development_only",
            "fixed_epochs": 3,
            "grid_id": "uh-bpr-development-v2-108",
            "selected_config": selected_config,
            "uh_config": selected_config.copy(),
            "candidate_configs": candidates,
        }
        profile.update(overrides)
        path = directory / "uh-profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_reads_pinned_development_profile(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            config = _read_uh_profile(self._write_profile(Path(directory)), epochs=3)
        self.assertEqual(config, _uh_development_grid(3)[0])

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
                    "dimension": 256,
                    "layers": 1,
                    "learning_rate": 0.001,
                    "regularization": BPRConfig().regularization,
                    "batch_size": BPRConfig().batch_size,
                    "epochs": 3,
                },
            )
            with self.assertRaisesRegex(ValueError, "outside the approved"):
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

    def test_uh_search_progress_restores_only_a_complete_matching_prefix(self) -> None:
        configs = _uh_development_grid(3)
        world_seeds = [11, 22]
        saved = {
            "status": "running",
            "dataset": "ml-100k",
            "dataset_sha256": "a" * 64,
            "parent_seed": 19,
            "fixed_epochs": 3,
            "required_models": len(configs) * len(world_seeds),
            "completed_models": len(configs) + 2,
            "grid_id": "uh-bpr-development-v2-108",
            "world_seeds": world_seeds,
            "validation_ndcg_at_20_by_config": {
                str(index): [0.1] * (2 if index < 2 else 1)
                for index in range(len(configs))
            },
        }
        completed, restored = _restore_uh_progress(
            saved, dataset="ml-100k", dataset_sha256="a" * 64, parent_seed=19,
            epochs=3, configs=configs, world_seeds=world_seeds,
        )
        self.assertEqual(completed, len(configs) + 2)
        self.assertEqual(len(restored["0"]), 2)
        self.assertEqual(len(restored["2"]), 1)
        saved["parent_seed"] = 20
        with self.assertRaisesRegex(ValueError, "does not match"):
            _restore_uh_progress(
                saved, dataset="ml-100k", dataset_sha256="a" * 64, parent_seed=19,
                epochs=3, configs=configs, world_seeds=world_seeds,
            )

    def test_parity_profile_chooses_candidate_closest_to_development_reference(self) -> None:
        selected, means, gaps = _select_uh_parity_candidate(
            {
                "0": [0.12, 0.12, 0.12],
                "1": [0.10, 0.10, 0.10],
                "2": [0.105, 0.105, 0.105],
            },
            [0.10, 0.10, 0.10],
        )
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(means["1"], 0.10)
        self.assertEqual(gaps["1"], 0.0)
        with self.assertRaisesRegex(ValueError, "one finite validation score"):
            _select_uh_parity_candidate({"0": [0.1, None]}, [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()

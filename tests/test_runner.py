from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction
from iahu.runner import check_unlearning_import_boundary, spawn_seed_streams, _visible_split


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


if __name__ == "__main__":
    unittest.main()

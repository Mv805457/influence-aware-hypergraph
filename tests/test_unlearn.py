from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.flagging import VisibleFlag, VisibleFlagSet  # noqa: E402
from iahu.unlearn import UnlearnBudget, build_plan  # noqa: E402


class UnlearnPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flags = VisibleFlagSet((VisibleFlag(10, 1.0), VisibleFlag(11, 0.5)), "soft")
        self.scores = {10: 0.99, 11: 0.98, 12: 0.9, 13: 0.4, 14: 0.0}

    def test_removal_includes_flags_and_top_scored_extensions(self) -> None:
        plan = build_plan(self.flags, self.scores, UnlearnBudget(extension_ratio=1), operator="removal", seed=3)
        self.assertEqual(plan.flagged_users, {10, 11})
        self.assertEqual(set(plan.extension_users), {12, 13})
        self.assertEqual(dict(plan.weight_map), {10: 0, 11: 0, 12: 0, 13: 0})
        self.assertEqual(plan.weights_for((10, 11, 12, 13, 14)), (0, 0, 0, 0, 1))

    def test_reweighting_uses_flag_confidence_and_bounded_extension_weights(self) -> None:
        plan = build_plan(self.flags, self.scores, UnlearnBudget(extension_ratio=1), operator="reweighting", seed=3)
        weights = dict(plan.weight_map)
        self.assertAlmostEqual(weights[10], 0.2)
        self.assertAlmostEqual(weights[11], 0.6)
        self.assertAlmostEqual(weights[12], 0.1)
        self.assertAlmostEqual(weights[13], 0.6)
        self.assertTrue(all(0.05 <= weight <= 1 for weight in weights.values()))

    def test_empty_flags_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "InsufficientFlags"):
            build_plan(VisibleFlagSet((), "hard"), {}, UnlearnBudget(), operator="removal", seed=0)

    def test_invalid_scores_and_operators_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            build_plan(self.flags, {12: 1.1}, UnlearnBudget(), operator="removal", seed=0)
        with self.assertRaisesRegex(ValueError, "operator"):
            build_plan(self.flags, {}, UnlearnBudget(), operator="oracle", seed=0)


if __name__ == "__main__":
    unittest.main()

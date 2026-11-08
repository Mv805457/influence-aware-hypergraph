from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.gates import attack_strength_gate, clean_model_gate  # noqa: E402


class GateTests(unittest.TestCase):
    def test_g1_passes_only_when_every_check_passes(self) -> None:
        result = clean_model_gate(
            [0.100, 0.101, 0.099, 0.100, 0.100],
            [0.100, 0.101, 0.099, 0.100, 0.100],
            finite_difference_relative_error=1e-7,
            repeated_run_max_abs_difference=1e-8,
            repeat_tolerance=1e-6,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())

    def test_g1_requires_five_seeds_and_flags_high_variance(self) -> None:
        with self.assertRaisesRegex(ValueError, "five paired seeds"):
            clean_model_gate([0.1], [0.1], finite_difference_relative_error=0,
                             repeated_run_max_abs_difference=0, repeat_tolerance=1e-6)
        result = clean_model_gate(
            [0.02, 0.2, 0.02, 0.2, 0.02], [0.1] * 5,
            finite_difference_relative_error=2e-5,
            repeated_run_max_abs_difference=0.1, repeat_tolerance=1e-6,
        )
        self.assertFalse(result.passed)
        self.assertIn("five_seed_relative_std", result.reasons)
        self.assertIn("finite_difference_gradient", result.reasons)
        self.assertIn("same_seed_reproducibility", result.reasons)

    def test_g2_requires_gain_ratio_and_utility(self) -> None:
        passed = attack_strength_gate(
            reference_exposure=0.01, poisoned_exposure=0.06,
            reference_ndcg=0.10, poisoned_ndcg=0.096,
        )
        self.assertTrue(passed.passed)
        failed = attack_strength_gate(
            reference_exposure=0, poisoned_exposure=0,
            reference_ndcg=0.1, poisoned_ndcg=0.094,
        )
        self.assertFalse(failed.passed)
        self.assertIn("absolute_exposure_gain", failed.reasons)
        self.assertIn("ndcg_retention", failed.reasons)

    def test_g2_zero_reference_ratio_passes_by_convention(self) -> None:
        result = attack_strength_gate(
            reference_exposure=0, poisoned_exposure=0.02,
            reference_ndcg=0.1, poisoned_ndcg=0.1,
        )
        self.assertTrue(result.checks["exposure_ratio"])
        self.assertTrue(result.checks["absolute_exposure_gain"])


if __name__ == "__main__":
    unittest.main()

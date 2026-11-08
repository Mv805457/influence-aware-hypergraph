from __future__ import annotations

import sys
import unittest
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction  # noqa: E402
from iahu.flagging import anomaly_proxy, simulate_flagging  # noqa: E402


class FlaggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = tuple(
            Interaction(user, item, 5, None)
            for user in range(1, 9)
            for item in ({1, 2, 3} if user in {1, 5} else {user % 5 + 1, (user + 1) % 5 + 1, (user + 2) % 5 + 1})
        )
        self.malicious = {7, 8}

    def test_flag_count_formulas_and_audit_are_separate(self) -> None:
        result = simulate_flagging(
            self.rows, self.malicious, seed=19, recall=0.5,
            false_positive_share=0.5, malicious_strategy="uniform",
        )
        self.assertEqual(len(result.visible.flags), 2)
        self.assertEqual(len(result.audit.identified_malicious), 1)
        self.assertEqual(len(result.audit.false_positive_benign), 1)
        self.assertEqual(result.audit.realized_recall, 0.5)
        self.assertEqual(result.audit.realized_false_positive_share, 0.5)
        self.assertEqual({f.name for f in fields(result.visible)}, {"flags", "confidence_mode"})

    def test_seeded_modes_are_deterministic_and_confidence_bounded(self) -> None:
        args = dict(
            seed=23, recall=1, false_positive_share=0.3,
            malicious_strategy="clustered", benign_strategy="look_alike", confidence_mode="soft",
        )
        first = simulate_flagging(self.rows, self.malicious, **args)
        second = simulate_flagging(self.rows, self.malicious, **args)
        self.assertEqual(first, second)
        self.assertTrue(all(0 < flag.confidence < 1 for flag in first.visible.flags))

    def test_anomaly_proxy_is_finite_and_user_indexed(self) -> None:
        scores = anomaly_proxy(self.rows)
        self.assertEqual(set(scores), set(range(1, 9)))
        self.assertTrue(all(0 <= value <= 1 for value in scores.values()))

    def test_empty_identification_produces_empty_flags_for_refusal_path(self) -> None:
        result = simulate_flagging(
            self.rows, self.malicious, seed=1, recall=0,
            false_positive_share=0, malicious_strategy="uniform",
        )
        self.assertEqual(result.visible.flags, ())
        self.assertEqual(result.audit.hidden_malicious, frozenset(self.malicious))

    def test_invalid_rates_and_unknown_sources_fail(self) -> None:
        for recall, fp_share in ((-0.1, 0), (1.1, 0), (0.5, 1.0)):
            with self.subTest(recall=recall, fp_share=fp_share), self.assertRaises(ValueError):
                simulate_flagging(self.rows, self.malicious, seed=0, recall=recall, false_positive_share=fp_share)
        with self.assertRaisesRegex(ValueError, "subset"):
            simulate_flagging(self.rows, {99}, seed=0, recall=0.5, false_positive_share=0)


if __name__ == "__main__":
    unittest.main()

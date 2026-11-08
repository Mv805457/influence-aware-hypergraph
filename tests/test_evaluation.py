from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction
from iahu.evaluation import evaluate_full_ranking
from iahu.hypergraph import build_user_history
from iahu.model import BPRConfig, fit_bpr


class FullRankingTests(unittest.TestCase):
    def test_batched_ranking_preserves_lower_item_tie_break_and_target_rank(self) -> None:
        training = (
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None),
            Interaction(2, 2, 5, None), Interaction(2, 3, 5, None),
            Interaction(3, 3, 5, None), Interaction(3, 4, 5, None),
        )
        model = fit_bpr(
            build_user_history(training), training,
            config=BPRConfig(dimension=2, layers=1, batch_size=3, epochs=1),
            seed=7,
        )
        model.theta[:] = 0.0
        result = evaluate_full_ranking(
            model,
            heldout_rows=(Interaction(1, 3, 5, None), Interaction(2, 1, 5, None)),
            excluded_rows=training,
            target_item_id=4,
            k=1,
            example_user_id=1,
            target_exposure_users={1},
        )
        self.assertEqual(result.example.user_id, 1)
        self.assertEqual(result.example.top_k, ((3, 1, 0.0),))
        self.assertEqual(result.example.target_rank, 2)
        self.assertEqual(result.metrics["target_exposure@1"], 0.0)
        self.assertEqual(result.metrics["target_eligible_users"], 1)
        self.assertEqual(result.metrics["ndcg@1"], 1.0)

    def test_observed_validation_item_is_removed_from_recommendation_candidates(self) -> None:
        from iahu.runner import _demo_items

        training = (
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None),
            Interaction(2, 2, 5, None), Interaction(2, 3, 5, None),
            Interaction(3, 3, 5, None), Interaction(3, 4, 5, None),
        )
        model = fit_bpr(
            build_user_history(training), training,
            config=BPRConfig(dimension=2, layers=1, batch_size=3, epochs=1),
            seed=9,
        )
        model.theta[:] = 0.0
        visible_validation = (Interaction(1, 3, 5, None),)
        items, target_rank = _demo_items(
            model, 1, 4, {3: "Seen in validation", 4: "Target"},
            top_k=2, excluded_rows=visible_validation,
        )
        self.assertEqual(items, [{"label": "Target", "is_target": True}])
        self.assertEqual(target_rank, 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction  # noqa: E402
from iahu.hypergraph import build_user_history  # noqa: E402
from iahu.model import BPRConfig, LightGCNOperator, fit_bpr  # noqa: E402


class BPRModelTests(unittest.TestCase):
    def test_small_training_run_is_finite_and_rank_excludes_train_items(self) -> None:
        rows = [
            Interaction(1, 1, 5, None),
            Interaction(1, 2, 5, None),
            Interaction(1, 3, 5, None),
            Interaction(2, 2, 5, None),
            Interaction(2, 3, 5, None),
            Interaction(2, 4, 5, None),
            Interaction(3, 1, 5, None),
            Interaction(3, 4, 5, None),
            Interaction(3, 5, 5, None),
        ]
        graph = build_user_history(rows)
        model = fit_bpr(
            graph,
            rows,
            config=BPRConfig(dimension=4, layers=2, batch_size=3, epochs=2),
            seed=29,
        )
        self.assertEqual(len(model.epoch_losses), 2)
        self.assertTrue(all(loss >= 0 for loss in model.epoch_losses))
        ranked = model.rank_items(1)
        self.assertTrue(all(item_id not in {1, 2, 3} for item_id, _score in ranked))
        self.assertEqual([item_id for item_id, _score in ranked], [4, 5])

    def test_warm_started_weighted_fit_reweights_operator_and_drops_removed_history(self) -> None:
        rows = [
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None), Interaction(1, 3, 5, None),
            Interaction(2, 2, 5, None), Interaction(2, 3, 5, None), Interaction(2, 4, 5, None),
            Interaction(3, 1, 5, None), Interaction(3, 4, 5, None), Interaction(3, 5, 5, None),
        ]
        graph = build_user_history(rows)
        config = BPRConfig(dimension=4, layers=2, batch_size=9, epochs=1)
        initial = fit_bpr(graph, rows, config=config, seed=11)
        result = fit_bpr(
            graph, rows, config=config, seed=12, source_weights=[0.0, 0.5, 1.0],
            initial_theta=initial.theta,
        )
        self.assertEqual(result.history[1], frozenset())
        self.assertEqual(result.history[2], frozenset({2, 3, 4}))
        self.assertTrue(np.isfinite(result.theta).all())

    def test_lightgcn_operator_trains_without_hypergraph_weight_argument(self) -> None:
        rows = [
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None),
            Interaction(2, 2, 5, None), Interaction(2, 3, 5, None),
            Interaction(3, 1, 5, None), Interaction(3, 4, 5, None),
        ]
        operator = LightGCNOperator((1, 2, 3), (1, 2, 3, 4), rows)
        model = fit_bpr(
            operator, rows,
            config=BPRConfig(dimension=3, layers=2, batch_size=3, epochs=1),
            seed=41,
        )
        self.assertTrue(np.isfinite(model.theta).all())
        self.assertEqual(len(model.epoch_losses), 1)


if __name__ == "__main__":
    unittest.main()

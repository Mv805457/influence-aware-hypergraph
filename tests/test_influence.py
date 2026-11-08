from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from iahu.data import Interaction  # noqa: E402
from iahu.hypergraph import build_user_history  # noqa: E402
from iahu.influence import (  # noqa: E402
    TorchHypergraphOperator, blend_suspicion_scores, conjugate_gradient,
    estimate_source_influence, overlap_proximity,
    frozen_negative_objective, polish_frozen_negative,
)


class InfluencePrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = (
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None),
            Interaction(2, 2, 5, None), Interaction(2, 3, 5, None),
            Interaction(3, 1, 5, None), Interaction(3, 4, 5, None),
        )
        self.graph = build_user_history(self.rows)

    def test_torch_operator_matches_matrix_free_numpy_and_source_gradient(self) -> None:
        graph_op = TorchHypergraphOperator(self.graph, dtype=torch.float64)
        features = torch.arange(self.graph.num_vertices * 2, dtype=torch.float64).reshape(-1, 2) / 10
        weights = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64, requires_grad=True)
        actual = graph_op.apply(features, weights)
        expected = np.asarray(self.graph.apply(features.numpy().tolist(), weights.detach().tolist()))
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(lambda c: graph_op.apply(features, c), (weights,)))

    def test_conjugate_gradient_solves_spd_system_and_flags_negative_curvature(self) -> None:
        matrix = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
        rhs = torch.tensor([2.0, 8.0], dtype=torch.float64)
        result = conjugate_gradient(lambda x: matrix @ x, rhs, max_iterations=5, relative_tolerance=1e-12)
        self.assertTrue(result.converged)
        torch.testing.assert_close(result.solution, torch.tensor([1.0, 2.0], dtype=torch.float64))
        negative = conjugate_gradient(lambda x: -x, rhs, max_iterations=5)
        self.assertTrue(negative.negative_curvature)
        self.assertFalse(negative.converged)

    def test_frozen_negative_objective_is_finite_and_polish_reduces_gradient(self) -> None:
        graph_op = TorchHypergraphOperator(self.graph, dtype=torch.float64)
        user_map = {user: i for i, user in enumerate(self.graph.user_ids)}
        item_map = {item: i for i, item in enumerate(self.graph.item_ids)}
        user_indices = torch.tensor([user_map[row.user_id] for row in self.rows])
        positive = torch.tensor([item_map[row.item_id] for row in self.rows])
        negatives = torch.tensor([
            [item_map[x] for x in ({1, 2, 3, 4} - {row.item_id} -
                                   {r.item_id for r in self.rows if r.user_id == row.user_id})][:2]
            for row in self.rows
        ])
        weights = torch.ones(3, dtype=torch.float64)
        theta = torch.randn(self.graph.num_vertices, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        loss = frozen_negative_objective(theta, graph_op, weights, user_indices, positive, negatives, layers=1)
        self.assertTrue(torch.isfinite(loss))
        polished = polish_frozen_negative(
            theta, self.graph, weights, user_indices, positive, negatives,
            layers=1, relative_tolerance=0.05, max_iterations=20,
        )
        self.assertLessEqual(polished.final_gradient_norm, polished.initial_gradient_norm)

    def test_double_backward_returns_one_finite_score_per_source(self) -> None:
        user_map = {user: i for i, user in enumerate(self.graph.user_ids)}
        item_map = {item: i for i, item in enumerate(self.graph.item_ids)}
        user_indices = torch.tensor([user_map[row.user_id] for row in self.rows])
        positive = torch.tensor([item_map[row.item_id] for row in self.rows])
        negatives = torch.tensor([
            [item_map[item] for item in sorted(
                {1, 2, 3, 4} - {r.item_id for r in self.rows if r.user_id == row.user_id}
            )[:2]]
            for row in self.rows
        ])
        weights = torch.ones(3, dtype=torch.float64)
        theta = torch.randn(self.graph.num_vertices, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
        polished = polish_frozen_negative(
            theta, self.graph, weights, user_indices, positive, negatives,
            layers=1, relative_tolerance=0.1, max_iterations=30,
        )
        self.assertGreater(polished.initial_gradient_norm, 0)
        pair_users = torch.tensor([0, 1], dtype=torch.long)
        pair_items = torch.tensor([item_map[4], item_map[4]], dtype=torch.long)
        thresholds = torch.tensor([0.0, 0.0], dtype=torch.float64)
        result = estimate_source_influence(
            polished.theta, self.graph, weights, user_indices, positive, negatives,
            pair_users, pair_items, thresholds, layers=1, temperature=0.2,
            damping=1.0, cg_iterations=20, polished=True,
            initial_gradient_norm=polished.initial_gradient_norm,
        )
        self.assertEqual(len(result.source_scores), self.graph.num_users)
        self.assertTrue(all(np.isfinite(result.source_scores)))
        self.assertTrue(np.isfinite(result.exposure_value))

    def test_overlap_reach_refuses_undefined_one_layer_u_h_and_blend_is_bounded(self) -> None:
        weights = torch.ones(self.graph.num_users, dtype=torch.float64)
        one_layer = overlap_proximity(self.graph, weights, [(1, 1.0)], layers=1, batch_size=2)
        self.assertTrue(all(value is None for value in one_layer.values()))
        two_layer = overlap_proximity(self.graph, weights, [(1, 1.0)], layers=2, batch_size=2)
        self.assertEqual(set(two_layer), {2, 3})
        self.assertTrue(any(value is not None for value in two_layer.values()))
        blended = blend_suspicion_scores(
            [0.1, 0.4, 0.9], self.graph, two_layer, {1}, overlap_weight=0.5
        )
        self.assertLessEqual(set(blended), {2, 3})
        self.assertTrue(all(0 <= value <= 1 for value in blended.values()))


if __name__ == "__main__":
    unittest.main()

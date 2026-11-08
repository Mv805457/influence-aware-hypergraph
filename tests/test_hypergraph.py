from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction  # noqa: E402
from iahu.hypergraph import build_hour_inspired, build_user_history  # noqa: E402


class UserHistoryHypergraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            Interaction(10, 100, 5, None),
            Interaction(10, 101, 5, None),
            Interaction(11, 101, 5, None),
            Interaction(11, 102, 5, None),
        ]
        self.graph = build_user_history(self.rows)

    def test_builder_has_one_nonempty_edge_per_user(self) -> None:
        self.assertEqual(self.graph.construction, "user_history")
        self.assertEqual(self.graph.num_hyperedges, 2)
        self.assertEqual(self.graph.num_memberships, 6)
        self.assertTrue(all(self.graph.edges))
        for user_index, edge in enumerate(self.graph.edges):
            self.assertEqual(edge[0], (user_index, 1.0, user_index))

    def test_operator_is_symmetric_and_has_spectral_norm_at_most_one(self) -> None:
        operator = self.graph.dense_operator_for_tests()
        for row in range(len(operator)):
            for column in range(len(operator)):
                self.assertAlmostEqual(operator[row][column], operator[column][row], places=12)
        vector = [float(index + 1) for index in range(self.graph.num_vertices)]
        norm = math.sqrt(sum(value * value for value in vector))
        for _ in range(80):
            next_vector = [sum(a * b for a, b in zip(row, vector)) for row in operator]
            next_norm = math.sqrt(sum(value * value for value in next_vector))
            if next_norm == 0:
                break
            vector = [value / next_norm for value in next_vector]
        rayleigh = sum(
            vector[row] * sum(operator[row][column] * vector[column] for column in range(len(vector)))
            for row in range(len(vector))
        )
        self.assertLessEqual(rayleigh, 1.0 + 1e-12)
        self.assertGreater(norm, 0)

    def test_zero_source_weight_produces_finite_zero_user_row(self) -> None:
        output = self.graph.apply([[1.0] for _ in range(self.graph.num_vertices)], [0.0, 1.0])
        self.assertTrue(all(math.isfinite(row[0]) for row in output))
        self.assertEqual(output[0], [0.0])

    def test_numpy_matrix_free_operator_matches_reference(self) -> None:
        import numpy as np

        features = np.arange(self.graph.num_vertices * 3, dtype=np.float64).reshape(-1, 3)
        expected = np.asarray(self.graph.apply(features.tolist()))
        actual = self.graph.apply_numpy(features)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_removing_one_source_matches_rebuilt_graph_on_survivors(self) -> None:
        retained = [row for row in self.rows if row.user_id != 10]
        rebuilt = build_user_history(retained).dense_operator_for_tests()
        weighted = self.graph.dense_operator_for_tests([0.0, 1.0])
        # Surviving user 11 is index 1, and items 101/102 remain vertices.
        surviving = [1, 3, 4]
        for row, original_row in enumerate(surviving):
            for column, original_column in enumerate(surviving):
                self.assertAlmostEqual(
                    rebuilt[row][column],
                    weighted[original_row][original_column],
                    places=12,
                )


class HourInspiredHypergraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            Interaction(user, item, 5, None)
            for user, items in {
                1: (1, 2, 3),
                2: (1, 2, 4),
                3: (1, 3, 5),
                4: (2, 4, 6),
                5: (3, 5, 6),
                6: (4, 5, 6),
            }.items()
            for item in items
        ]

    def test_nmf_structure_is_deterministic_and_obeys_vertex_membership_cap(self) -> None:
        first = build_hour_inspired(self.rows, seed=7, rank=2, memberships_per_vertex=2, membership_floor=0.0)
        second = build_hour_inspired(self.rows, seed=7, rank=2, memberships_per_vertex=2, membership_floor=0.0)
        self.assertEqual(first, second)
        self.assertEqual(first.construction, "hour_inspired")
        self.assertTrue(all(first.edges))
        memberships_per_vertex = {}
        for edge in first.edges:
            for vertex, value, _owner in edge:
                memberships_per_vertex[vertex] = memberships_per_vertex.get(vertex, 0) + 1
                self.assertGreater(value, 0)
        self.assertTrue(all(count <= 2 for count in memberships_per_vertex.values()))

    def test_frozen_and_rebuilt_source_weight_policies_are_distinct(self) -> None:
        frozen = build_hour_inspired(self.rows, seed=11, rank=2, membership_floor=0.0, policy="frozen")
        rebuilt = build_hour_inspired(
            self.rows,
            seed=11,
            rank=2,
            membership_floor=0.0,
            source_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            policy="rebuilt",
        )
        frozen_user_owners = {
            owner for edge in frozen.edges for vertex, _value, owner in edge if vertex < frozen.num_users
        }
        rebuilt_user_owners = {
            owner for edge in rebuilt.edges for vertex, _value, owner in edge if vertex < rebuilt.num_users
        }
        self.assertEqual(frozen_user_owners, set(range(frozen.num_users)))
        self.assertEqual(rebuilt_user_owners, {-1})


if __name__ == "__main__":
    unittest.main()

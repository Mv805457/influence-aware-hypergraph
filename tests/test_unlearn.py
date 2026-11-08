from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.flagging import VisibleFlag, VisibleFlagSet  # noqa: E402
from iahu.hypergraph import build_user_history  # noqa: E402
from iahu.model import BPRConfig, fit_bpr  # noqa: E402
from iahu.unlearn import (  # noqa: E402
    StagedUnlearnConfig,
    UnlearnBudget,
    VisibleExperiment,
    build_plan,
    run_staged_unlearning,
)
from iahu.data import Interaction  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


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

    def test_staged_extension_limit_caps_each_stage_without_changing_flag_weights(self) -> None:
        plan = build_plan(
            self.flags,
            self.scores,
            UnlearnBudget(extension_ratio=1),
            operator="reweighting",
            seed=0,
            extension_limit=1,
        )
        self.assertEqual(len(plan.extension_users), 1)
        self.assertAlmostEqual(dict(plan.weight_map)[10], 0.2)
        self.assertAlmostEqual(dict(plan.weight_map)[11], 0.6)

    def test_staged_extension_limit_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "extension_limit"):
            build_plan(
                self.flags, self.scores, UnlearnBudget(),
                operator="removal", seed=0, extension_limit=-1,
            )

    def test_invalid_scores_and_operators_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            build_plan(self.flags, {12: 1.1}, UnlearnBudget(), operator="removal", seed=0)
        with self.assertRaisesRegex(ValueError, "operator"):
            build_plan(self.flags, {}, UnlearnBudget(), operator="oracle", seed=0)


class VisibleStagedDriverTests(unittest.TestCase):
    def test_empty_visible_flags_refuse_without_changing_model(self) -> None:
        rows = (
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None), Interaction(1, 5, 5, None),
            Interaction(2, 1, 5, None), Interaction(2, 3, 5, None),
            Interaction(3, 2, 5, None), Interaction(3, 4, 5, None),
        )
        graph = build_user_history(rows)
        model = fit_bpr(
            graph, rows, config=BPRConfig(dimension=3, layers=1, batch_size=4, epochs=1), seed=7,
        )
        user_index = {user: index for index, user in enumerate(graph.user_ids)}
        item_index = {item: index for index, item in enumerate(graph.item_ids)}
        experiment = VisibleExperiment(
            graph=graph,
            model=model,
            training_rows=rows,
            validation_rows=(Interaction(2, 4, 5, None), Interaction(3, 1, 5, None)),
            flags=VisibleFlagSet((), "hard"),
            target_item_ids=(5,),
            negative_item_indices=torch.tensor([[3], [4], [2], [4], [0], [2], [0]]),
            positive_user_indices=torch.tensor([user_index[row.user_id] for row in rows]),
            positive_item_indices=torch.tensor([item_index[row.item_id] for row in rows]),
        )
        before = model.theta.copy()
        result = run_staged_unlearning(experiment, config=StagedUnlearnConfig())
        self.assertEqual(result.refusal_reason, "InsufficientFlags")
        self.assertEqual(result.stages, ())
        np.testing.assert_array_equal(result.model.theta, before)
        self.assertEqual(result.selected_extension_user_ids, ())

    def test_small_visible_world_runs_influence_stage_without_evaluator_truth(self) -> None:
        rows = (
            Interaction(1, 1, 5, None), Interaction(1, 2, 5, None), Interaction(1, 5, 5, None),
            Interaction(2, 1, 5, None), Interaction(2, 3, 5, None),
            Interaction(3, 2, 5, None), Interaction(3, 4, 5, None),
        )
        graph = build_user_history(rows)
        model = fit_bpr(
            graph, rows, config=BPRConfig(dimension=3, layers=1, batch_size=4, epochs=1), seed=13,
        )
        user_index = {user: index for index, user in enumerate(graph.user_ids)}
        item_index = {item: index for index, item in enumerate(graph.item_ids)}
        experiment = VisibleExperiment(
            graph=graph,
            model=model,
            training_rows=rows,
            validation_rows=(Interaction(2, 4, 5, None), Interaction(3, 1, 5, None)),
            flags=VisibleFlagSet((VisibleFlag(1, 1.0),), "hard"),
            target_item_ids=(5,),
            negative_item_indices=torch.tensor([[3], [4], [3], [1], [4], [0], [2]]),
            positive_user_indices=torch.tensor([user_index[row.user_id] for row in rows]),
            positive_item_indices=torch.tensor([item_index[row.item_id] for row in rows]),
        )
        result = run_staged_unlearning(
            experiment,
            config=StagedUnlearnConfig(
                extension_ratio=0,
                mass_cap=1,
                utility_tolerance=0.99,
                path_steps=1,
                stage_epochs=1,
                recovery_epochs=0,
                negative_samples=1,
                damping=10,
                polish_relative_tolerance=0.5,
                polish_max_iterations=2,
                cg_iterations=3,
                cg_tolerance=0.5,
                seed=19,
            ),
        )
        self.assertIsNone(result.refusal_reason)
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(result.flagged_count, 1)
        self.assertEqual(result.extension_count, 0)
        self.assertEqual(result.stages[0].stage, 1)
        self.assertIsNotNone(result.staged_exposure_change)
        self.assertTrue(math.isfinite(result.staged_exposure_change))
        self.assertFalse(result.final_influence_reliable)
        self.assertEqual(result.frozen_exposure_cohort_size, 2)
        self.assertIsNotNone(result.final_exposure_surrogate)
        self.assertIsNotNone(result.final_exposure_excluding_extensions)
        self.assertTrue(math.isfinite(result.final_exposure_surrogate))
        self.assertAlmostEqual(
            result.final_exposure_surrogate,
            result.final_exposure_excluding_extensions,
        )

if __name__ == "__main__":
    unittest.main()

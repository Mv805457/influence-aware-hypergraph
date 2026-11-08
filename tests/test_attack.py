from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.attack import inject_fake_accounts  # noqa: E402
from iahu.data import Interaction  # noqa: E402


class RandomProfileAttackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            Interaction(user, item, 5, timestamp=1_000 + item)
            for user in range(1, 11)
            for item in range(1, 11)
            if item != user
        ]

    def test_attack_and_placebo_match_count_profile_and_hide_source_ids(self) -> None:
        attack = inject_fake_accounts(self.rows, seed=7, target_item_id=10, alpha=0.2)
        placebo = inject_fake_accounts(self.rows, seed=7, target_item_id=10, alpha=0.2, placebo=True)
        self.assertEqual(attack.fake_account_count, 2)
        self.assertEqual(placebo.fake_account_count, 2)
        self.assertEqual(attack.profile_length, placebo.profile_length)
        self.assertEqual(len(attack.hidden_source_ids), 2)
        self.assertEqual(len(placebo.hidden_source_ids), 2)
        self.assertTrue(attack.hidden_source_ids.isdisjoint(attack.benign_user_map.values()))
        profiles: dict[int, list[int]] = defaultdict(list)
        for row in attack.interactions:
            if row.user_id in attack.hidden_source_ids:
                profiles[row.user_id].append(row.item_id)
        self.assertTrue(all(10 in items for items in profiles.values()))
        placebo_profiles: dict[int, list[int]] = defaultdict(list)
        for row in placebo.interactions:
            if row.user_id in placebo.hidden_source_ids:
                placebo_profiles[row.user_id].append(row.item_id)
        self.assertTrue(all(10 not in items for items in placebo_profiles.values()))
        self.assertTrue(all(len(items) == attack.profile_length for items in profiles.values()))
        self.assertTrue(all(len(items) == placebo.profile_length for items in placebo_profiles.values()))

    def test_same_seed_is_deterministic_and_invalid_settings_fail(self) -> None:
        first = inject_fake_accounts(self.rows, seed=29, target_item_id=10, alpha=0.1)
        second = inject_fake_accounts(self.rows, seed=29, target_item_id=10, alpha=0.1)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "paired target item"):
            inject_fake_accounts(self.rows, seed=1, target_item_id=None)

    def test_bandwagon_and_popularity_rules_preserve_profile_contract(self) -> None:
        for rule in (
            "atk2_bandwagon", "atk3_popularity", "atk4_coordinated",
            "atk5_overlap_minimizing",
        ):
            with self.subTest(rule=rule):
                result = inject_fake_accounts(
                    self.rows, seed=31, target_item_id=10, alpha=0.2, rule=rule
                )
                profiles: dict[int, list[int]] = defaultdict(list)
                for row in result.interactions:
                    if row.user_id in result.hidden_source_ids:
                        profiles[row.user_id].append(row.item_id)
                self.assertTrue(all(10 in items for items in profiles.values()))
                self.assertTrue(all(len(items) == result.profile_length for items in profiles.values()))
                self.assertTrue(all(len(items) == len(set(items)) for items in profiles.values()))
                self.assertEqual(result.rule, rule)

    def test_all_filler_rules_produce_matched_placebo_profiles(self) -> None:
        for rule in (
            "atk1_random", "atk2_bandwagon", "atk3_popularity",
            "atk4_coordinated", "atk5_overlap_minimizing",
        ):
            attack = inject_fake_accounts(
                self.rows, seed=71, target_item_id=10, alpha=0.2, rule=rule
            )
            placebo = inject_fake_accounts(
                self.rows, seed=71, target_item_id=10, alpha=0.2, rule=rule, placebo=True
            )
            self.assertEqual(attack.fake_account_count, placebo.fake_account_count)
            for result, should_contain_target in ((attack, True), (placebo, False)):
                profiles: dict[int, list[int]] = defaultdict(list)
                for row in result.interactions:
                    if row.user_id in result.hidden_source_ids:
                        profiles[row.user_id].append(row.item_id)
                self.assertEqual(len(profiles), result.fake_account_count)
                self.assertTrue(all(len(items) == result.profile_length for items in profiles.values()))
                self.assertTrue(all(len(items) == len(set(items)) for items in profiles.values()))
                self.assertTrue(all((10 in items) == should_contain_target for items in profiles.values()))


if __name__ == "__main__":
    unittest.main()

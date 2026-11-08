from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import (  # noqa: E402
    DatasetError,
    Interaction,
    binarize_and_k_core,
    load_dataset,
    split_per_user,
)


class DataStageTests(unittest.TestCase):
    def test_iterative_k_core_prunes_until_stable(self) -> None:
        rows = [
            Interaction(1, 1, 5, None),
            Interaction(1, 2, 5, None),
            Interaction(2, 1, 5, None),
            Interaction(2, 2, 5, None),
            Interaction(3, 3, 5, None),
            Interaction(3, 4, 3, None),
        ]
        result = binarize_and_k_core(rows, threshold=4, k=2)
        self.assertEqual([(r.user_id, r.item_id) for r in result], [(1, 1), (1, 2), (2, 1), (2, 2)])

    def test_split_is_disjoint_seeded_and_uses_per_user_heldouts(self) -> None:
        rows = [Interaction(1, item, 5, item) for item in range(1, 21)]
        first = split_per_user(rows, world_seed=17)
        second = split_per_user(rows, world_seed=17)
        self.assertEqual(first, second)
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (16, 2, 2))
        sets = [{(row.user_id, row.item_id) for row in group} for group in (first.train, first.validation, first.test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_loader_rejects_unapproved_archive(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            archive_path = Path(directory) / "retail.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("amazon.csv", "product_id,title\n1,Not MovieLens\n")
            with self.assertRaisesRegex(DatasetError, "not in the MovieLens allowlist"):
                load_dataset(archive_path)


if __name__ == "__main__":
    unittest.main()

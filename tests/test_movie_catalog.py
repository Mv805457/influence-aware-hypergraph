from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iahu.data import Interaction, MovieLensData, SplitData  # noqa: E402
from iahu.movie_catalog import build_catalog  # noqa: E402


class MovieCatalogTests(unittest.TestCase):
    def test_catalog_contains_titles_genres_and_aggregate_clean_popularity_only(self) -> None:
        rows = (
            Interaction(1, 4, 5, 1),
            Interaction(2, 4, 5, 2),
            Interaction(1, 5, 5, 3),
        )
        fake = MovieLensData(
            "ml-100k",
            "a" * 64,
            rows,
            {4: "Film A (1999)", 5: "Film B"},
            {4: ("Drama",), 5: ("Comedy",)},
        )
        split = SplitData(rows, (), ())
        with (
            patch("iahu.movie_catalog.load_dataset", return_value=fake),
            patch("iahu.movie_catalog.binarize_and_k_core", return_value=rows),
            patch("iahu.movie_catalog.split_per_user", return_value=split),
        ):
            catalog = build_catalog("unused.zip", dataset="ml-100k", world_seed=23)
        self.assertEqual(catalog["dataset_role"], "development")
        self.assertEqual(catalog["movies"][0], {
            "id": 4,
            "title": "Film A (1999)",
            "year": 1999,
            "genres": ["Drama"],
            "popularity": 2,
        })
        serialized = json.dumps(catalog)
        for forbidden in ("user_id", "timestamp", "Interaction("):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()

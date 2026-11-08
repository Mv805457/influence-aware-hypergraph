"""Create a sanitized, read-only MovieLens film catalog for the local demo."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .data import ALLOWLIST_PATH, DatasetError, binarize_and_k_core, load_dataset, split_per_user


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results" / "movie-catalog.json"
YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def build_catalog(archive_path: Path | str, *, dataset: str, world_seed: int) -> dict[str, object]:
    data = load_dataset(archive_path, expected_name=dataset)
    positive = binarize_and_k_core(data.interactions)
    split = split_per_user(positive, world_seed=world_seed)
    popularity = Counter(row.item_id for row in split.train)
    movies = []
    for item_id, title in sorted(data.titles.items()):
        match = YEAR_PATTERN.search(title)
        year = int(match.group(1)) if match else None
        movies.append(
            {
                "id": item_id,
                "title": title,
                "year": year,
                "genres": list(data.genres.get(item_id, ())),
                "popularity": popularity.get(item_id, 0),
            }
        )
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    dataset_role = allowlist["datasets"][dataset]["role"]
    return {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_role": dataset_role,
        "dataset_sha256": data.archive_sha256,
        "popularity_source": "positive MovieLens training interactions before attack",
        "world_seed": world_seed,
        "movies": movies,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--dataset", choices=("ml-100k", "ml-1m"), required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        catalog = build_catalog(args.archive, dataset=args.dataset, world_seed=args.world_seed)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (DatasetError, OSError, ValueError, KeyError) as exc:
        print(f"movie catalog export failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(catalog['movies'])} movie cards to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

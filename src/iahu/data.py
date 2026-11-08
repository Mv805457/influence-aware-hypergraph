"""Allowlisted MovieLens ingestion and deterministic S0 preprocessing.

The loader reads only MovieLens ratings and movie titles from the official
archives. It intentionally never opens the MovieLens demographics file or the
retail storefront archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = ROOT / "configs" / "datasets.json"
POSITIVE_RATING_THRESHOLD = 4
K_CORE = 10


class DatasetError(ValueError):
    """Raised when input data is invalid or outside the approved allowlist."""


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: int
    item_id: int
    rating: int
    timestamp: int | None


@dataclass(frozen=True, slots=True)
class MovieLensData:
    dataset: str
    archive_sha256: str
    interactions: tuple[Interaction, ...]
    titles: dict[int, str]
    genres: dict[int, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SplitData:
    train: tuple[Interaction, ...]
    validation: tuple[Interaction, ...]
    test: tuple[Interaction, ...]


def _read_allowlist(path: Path = ALLOWLIST_PATH) -> dict[str, dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read dataset allowlist: {exc}") from exc
    if value.get("schema_version") != 1 or not isinstance(value.get("datasets"), dict):
        raise DatasetError("dataset allowlist has an unsupported schema")
    return value["datasets"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DatasetError(f"cannot read dataset archive: {exc}") from exc
    return digest.hexdigest()


def load_dataset(
    archive_path: Path | str,
    *,
    expected_name: str | None = None,
    allowlist_path: Path = ALLOWLIST_PATH,
) -> MovieLensData:
    """Load a pinned official MovieLens archive after checking its SHA-256."""
    path = Path(archive_path)
    if not path.is_file():
        raise DatasetError(f"dataset archive does not exist: {path}")
    allowlist = _read_allowlist(allowlist_path)
    digest = _sha256_file(path)
    matches = [
        (name, config)
        for name, config in allowlist.items()
        if config.get("sha256") == digest
    ]
    if len(matches) != 1:
        raise DatasetError(
            "dataset SHA-256 is not in the MovieLens allowlist; "
            "unapproved data, including the retail archive, is rejected"
        )
    dataset, config = matches[0]
    if expected_name is not None and expected_name != dataset:
        raise DatasetError(
            f"archive is pinned as {dataset}, not the requested {expected_name}"
        )

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            ratings_member = str(config["ratings_member"])
            movies_member = str(config["movies_member"])
            if ratings_member not in names or movies_member not in names:
                raise DatasetError("archive is missing required MovieLens files")
            encoding = str(config["encoding"])
            with archive.open(ratings_member) as stream:
                ratings_text = stream.read().decode(encoding)
            with archive.open(movies_member) as stream:
                movies_text = stream.read().decode(encoding)
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, KeyError) as exc:
        raise DatasetError(f"invalid MovieLens archive: {exc}") from exc

    data_format = config["format"]
    interactions = _parse_ratings(ratings_text, str(data_format))
    titles, genres = _parse_movies(movies_text, str(data_format))
    item_ids = {row.item_id for row in interactions}
    missing_titles = item_ids - titles.keys()
    if missing_titles:
        raise DatasetError(f"ratings reference {len(missing_titles)} missing movie titles")
    return MovieLensData(dataset, digest, tuple(interactions), titles, genres)


def _parse_ratings(text: str, data_format: str) -> list[Interaction]:
    result: list[Interaction] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("::") if data_format == "ml-1m" else line.split("\t")
        if len(fields) != 4:
            raise DatasetError(f"malformed rating row at line {line_number}")
        try:
            user_id, item_id, rating = map(int, fields[:3])
            timestamp = int(fields[3]) if fields[3] else None
        except ValueError as exc:
            raise DatasetError(f"non-integer rating field at line {line_number}") from exc
        if user_id < 1 or item_id < 1 or not 1 <= rating <= 5:
            raise DatasetError(f"rating field out of range at line {line_number}")
        result.append(Interaction(user_id, item_id, rating, timestamp))
    if not result:
        raise DatasetError("dataset has no rating rows")
    return result


def _parse_movies(text: str, data_format: str) -> tuple[dict[int, str], dict[int, tuple[str, ...]]]:
    titles: dict[int, str] = {}
    genres: dict[int, tuple[str, ...]] = {}
    ml100k_genre_names = (
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western",
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("::", 2) if data_format == "ml-1m" else line.split("|")
        if len(fields) < 2:
            raise DatasetError(f"malformed movie row at line {line_number}")
        try:
            item_id = int(fields[0])
        except ValueError as exc:
            raise DatasetError(f"invalid movie ID at line {line_number}") from exc
        title = fields[1].strip()
        if not title or item_id in titles:
            raise DatasetError(f"empty or duplicate movie title at line {line_number}")
        titles[item_id] = title
        if data_format == "ml-1m":
            genres[item_id] = tuple(value for value in fields[2].split("|") if value and value != "(no genres listed)") if len(fields) > 2 else ()
        else:
            flags = fields[5:]
            genres[item_id] = tuple(
                name for name, flag in zip(ml100k_genre_names, flags) if flag == "1"
            )
    return titles, genres


def binarize_and_k_core(
    interactions: Iterable[Interaction],
    *,
    threshold: int = POSITIVE_RATING_THRESHOLD,
    k: int = K_CORE,
) -> tuple[Interaction, ...]:
    """Keep ratings >= threshold, then iteratively enforce a user/item k-core."""
    if not 1 <= threshold <= 5 or k < 1:
        raise ValueError("threshold must be in [1, 5] and k must be positive")
    rows = [row for row in interactions if row.rating >= threshold]
    while rows:
        user_counts = Counter(row.user_id for row in rows)
        item_counts = Counter(row.item_id for row in rows)
        keep_users = {user for user, count in user_counts.items() if count >= k}
        keep_items = {item for item, count in item_counts.items() if count >= k}
        next_rows = [
            row for row in rows if row.user_id in keep_users and row.item_id in keep_items
        ]
        if len(next_rows) == len(rows):
            return tuple(next_rows)
        rows = next_rows
    return ()


def split_per_user(
    interactions: Sequence[Interaction], *, world_seed: int
) -> SplitData:
    """Create disjoint 80/10/10-ish per-user splits from a seeded permutation."""
    by_user: dict[int, list[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row.user_id].append(row)
    rng = random.Random(world_seed)
    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        permutation = list(range(len(rows)))
        rng.shuffle(permutation)
        heldout = max(1, round(0.1 * len(rows)))
        test_indices = set(permutation[:heldout])
        validation_indices = set(permutation[heldout : 2 * heldout])
        train.extend(row for index, row in enumerate(rows) if index not in test_indices | validation_indices)
        validation.extend(row for index, row in enumerate(rows) if index in validation_indices)
        test.extend(row for index, row in enumerate(rows) if index in test_indices)
    return SplitData(tuple(train), tuple(validation), tuple(test))


def select_targets(
    train: Sequence[Interaction], *, world_seed: int, count: int = 5
) -> tuple[int, ...]:
    """Sample target items with >=5 train rows from the lower popularity half."""
    if count < 1:
        raise ValueError("count must be positive")
    popularity = Counter(row.item_id for row in train)
    eligible = sorted(item for item, pop in popularity.items() if pop >= 5)
    if not eligible:
        raise DatasetError("no target items meet the minimum train popularity of five")
    popularity_median = median(popularity.values())
    candidates = [item for item in eligible if popularity[item] < popularity_median]
    if len(candidates) < count:
        raise DatasetError(
            f"only {len(candidates)} target items are below the median; {count} required"
        )
    return tuple(random.Random(world_seed).sample(candidates, count))


def data_summary(data: MovieLensData, split: SplitData | None = None) -> dict[str, object]:
    rows = split.train + split.validation + split.test if split else data.interactions
    return {
        "dataset": data.dataset,
        "archive_sha256": data.archive_sha256,
        "positive_interactions": len(rows),
        "users": len({row.user_id for row in rows}),
        "items": len({row.item_id for row in rows}),
        "titles": len(data.titles),
        "split_rows": (
            {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            }
            if split
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and preprocess approved MovieLens data")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--dataset", choices=("ml-100k", "ml-1m"), required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        data = load_dataset(args.archive, expected_name=args.dataset)
        positive = binarize_and_k_core(data.interactions)
        split = split_per_user(positive, world_seed=args.world_seed)
        targets = select_targets(split.train, world_seed=args.world_seed)
    except (DatasetError, ValueError) as exc:
        print(f"data stage failed: {exc}", file=sys.stderr)
        return 2
    summary = data_summary(data, split)
    summary["targets"] = list(targets)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

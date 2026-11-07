"""Create and validate a product-only catalog for the local demo storefront."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "results" / "shop-catalog.json"
CATALOG_SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_PRODUCTS = 20_000
PRODUCT_FIELDS = {
    "id",
    "name",
    "category",
    "price",
    "list_price",
    "rating",
    "rating_count",
    "discount_percent",
}
DISPLAY_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
}
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class CatalogError(ValueError):
    """Raised for unsafe or invalid storefront catalog data."""


def _display_text(value: Any, field: str, *, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{field} must be a non-empty string")
    value = " ".join(value.split())
    if len(value) > limit or any(ord(char) < 32 for char in value):
        raise CatalogError(f"{field} is too long or contains control characters")
    return value


def _optional_number(value: Any, field: str, *, integer: bool = False) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise CatalogError(f"{field} must be numeric or null")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = _NUMBER.search(value.replace("\u2212", "-"))
        if not match:
            return None
        number = float(match.group().replace(",", ""))
    else:
        raise CatalogError(f"{field} must be numeric or null")
    if not math.isfinite(number) or number < 0:
        return None
    if integer:
        return int(number)
    return number


def validate_catalog(catalog: Any) -> dict[str, Any]:
    if not isinstance(catalog, dict) or set(catalog) != {"schema_version", "products"}:
        raise CatalogError("catalog must contain only schema_version and products")
    if catalog["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise CatalogError(f"catalog schema_version must be {CATALOG_SCHEMA_VERSION}")
    products = catalog["products"]
    if not isinstance(products, list) or len(products) > MAX_PRODUCTS:
        raise CatalogError(f"products must be a list with at most {MAX_PRODUCTS} entries")
    seen_ids: set[str] = set()
    for index, product in enumerate(products):
        prefix = f"products[{index}]"
        if not isinstance(product, dict) or set(product) != PRODUCT_FIELDS:
            raise CatalogError(f"{prefix} contains unsupported or missing fields")
        product_id = _display_text(product["id"], f"{prefix}.id", limit=32)
        if not re.fullmatch(r"product-\d{5}", product_id) or product_id in seen_ids:
            raise CatalogError(f"{prefix}.id must be a unique generated product ID")
        seen_ids.add(product_id)
        _display_text(product["name"], f"{prefix}.name")
        _display_text(product["category"], f"{prefix}.category", limit=500)
        for key in ("price", "list_price", "rating", "discount_percent"):
            value = product[key]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise CatalogError(f"{prefix}.{key} must be a non-negative number or null")
        if product["rating"] is not None and product["rating"] > 5:
            raise CatalogError(f"{prefix}.rating must not exceed 5")
        if product["discount_percent"] is not None and product["discount_percent"] > 100:
            raise CatalogError(f"{prefix}.discount_percent must not exceed 100")
        rating_count = product["rating_count"]
        if rating_count is not None and (
            isinstance(rating_count, bool)
            or not isinstance(rating_count, int)
            or rating_count < 0
        ):
            raise CatalogError(f"{prefix}.rating_count must be a non-negative integer or null")
    return catalog


def _find_csv(archive_path: Path) -> str:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            candidates = [
                info for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).name.lower() == "amazon.csv"
            ]
            if len(candidates) != 1:
                raise CatalogError("archive must contain exactly one amazon.csv file")
            info = candidates[0]
            if info.file_size > MAX_ARCHIVE_BYTES:
                raise CatalogError("amazon.csv exceeds the 100 MiB import limit")
            return info.filename
    except (OSError, zipfile.BadZipFile) as error:
        raise CatalogError(f"could not read CSV archive: {error}") from error


def import_archive(archive_path: Path) -> dict[str, Any]:
    csv_name = _find_csv(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive, archive.open(csv_name) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            if not DISPLAY_COLUMNS.issubset(columns):
                missing = ", ".join(sorted(DISPLAY_COLUMNS - columns))
                raise CatalogError(f"amazon.csv is missing required columns: {missing}")

            # Source IDs exist only in memory for deduplication. They are never serialized.
            by_source_id: dict[str, dict[str, Any]] = {}
            for row in reader:
                source_id = row.get("product_id", "").strip()
                name = " ".join(row.get("product_name", "").split())
                category = " / ".join(
                    segment.strip() for segment in row.get("category", "").split("|") if segment.strip()
                )
                if not source_id or not name or not category:
                    continue
                by_source_id.setdefault(
                    source_id,
                    {
                        "name": name[:500],
                        "category": category[:500],
                        "price": _optional_number(row.get("discounted_price"), "price"),
                        "list_price": _optional_number(row.get("actual_price"), "list_price"),
                        "rating": _optional_number(row.get("rating"), "rating"),
                        "rating_count": _optional_number(row.get("rating_count"), "rating_count", integer=True),
                        "discount_percent": _optional_number(row.get("discount_percentage"), "discount_percent"),
                    },
                )
                if len(by_source_id) > MAX_PRODUCTS:
                    raise CatalogError(f"archive contains more than {MAX_PRODUCTS} unique products")
    except (OSError, UnicodeError, csv.Error, zipfile.BadZipFile) as error:
        raise CatalogError(f"could not parse amazon.csv: {error}") from error

    ordered = sorted(
        by_source_id.items(),
        key=lambda pair: (pair[1]["name"].casefold(), pair[1]["category"].casefold(), pair[0]),
    )
    products = [
        {"id": f"product-{index:05d}", **product}
        for index, (_source_id, product) in enumerate(ordered, start=1)
    ]
    return validate_catalog({"schema_version": CATALOG_SCHEMA_VERSION, "products": products})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="ZIP containing amazon.csv")
    args = parser.parse_args(argv)
    try:
        catalog = import_archive(args.archive)
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, CatalogError) as error:
        print(f"Could not import shop catalog: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {len(catalog['products'])} product(s) to {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

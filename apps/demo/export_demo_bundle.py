"""Whitelist completed-run export data into the read-only demo bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from contract import ContractError, sanitize_source


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_INPUT_DIR = REPO_ROOT / "results" / "completed-run-exports"
DEFAULT_INPUT = EXPORT_INPUT_DIR / "completed-run-export.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "demo-bundle.json"


def export_bundle(source: Any) -> dict[str, Any]:
    return sanitize_source(source)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"precomputed completed-run export inside {EXPORT_INPUT_DIR} (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args(argv)

    try:
        if not _is_within(args.input, EXPORT_INPUT_DIR):
            raise ContractError(f"input must be inside {EXPORT_INPUT_DIR}")
        source = json.loads(args.input.read_text(encoding="utf-8"))
        bundle = export_bundle(source)
        DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_OUTPUT.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ContractError) as error:
        print(f"Could not export demo bundle: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {len(bundle['scenarios'])} scenario(s) to {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

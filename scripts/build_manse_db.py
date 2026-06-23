from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oracle_report.saju.repository import build_manse_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local manse database.")
    parser.add_argument("--db", type=Path, default=Path("data/manse.sqlite"))
    parser.add_argument("--start-year", type=int, default=1900)
    parser.add_argument("--end-year", type=int, default=2100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    built = build_manse_database(
        db_path=args.db,
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
    )
    if built:
        print(f"built manse DB: {args.db}")
    else:
        print(f"manse DB already ready: {args.db}")
    result = 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())

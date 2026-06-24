from __future__ import annotations

import argparse
from pathlib import Path

from oracle_report.vision.physiognomy_rule_repository import (
    DEFAULT_PHYSIOGNOMY_RULE_DB_PATH,
    build_physio_rule_database,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oracle-build-physiognomy-db")
    parser.add_argument("--db", type=Path, default=DEFAULT_PHYSIOGNOMY_RULE_DB_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    built = build_physio_rule_database(args.db, force=args.force)
    if built:
        print(f"built physiognomy rule DB: {args.db}")
    else:
        print(f"physiognomy rule DB already ready: {args.db}")
    result = 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())

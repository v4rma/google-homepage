"""Seed the database with the clearly synthetic demo dataset.

Usage: python scripts/seed_demo.py [--db data/avatar.db]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatar.config import load_settings  # noqa: E402
from avatar.demo_data import seed  # noqa: E402
from avatar.storage import Store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=load_settings().db_path)
    args = parser.parse_args()
    store = Store(args.db)
    created = seed(store)
    counts = store.counts()
    print(f"seeded {len(created)} episodes into {args.db} ({counts['episodes']} total, {counts['reserved_episodes']} reserved for holdout)")


if __name__ == "__main__":
    main()

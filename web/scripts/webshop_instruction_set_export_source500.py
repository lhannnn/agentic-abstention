#!/usr/bin/env python3
"""Export the fixed source-500 WebShop goal manifest from the real goal manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from webshop_abstain_common import (
    DEFAULT_FULL_GOAL_MANIFEST_PATH,
    DEFAULT_SOURCE500_MANIFEST_PATH,
    ensure_parent_dir,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-manifest", default=str(DEFAULT_FULL_GOAL_MANIFEST_PATH))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default=str(DEFAULT_SOURCE500_MANIFEST_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    goal_manifest = load_jsonl(Path(args.goal_manifest).expanduser().resolve())
    if len(goal_manifest) < args.count:
        raise RuntimeError(f"Goal manifest only has {len(goal_manifest)} rows, cannot export {args.count}.")

    records = []
    for source_rank, row in enumerate(goal_manifest[: args.count]):
        records.append(
            {
                "source_rank": source_rank,
                "goal_index": int(row["goal_index"]),
                "base_goal_index": int(row["goal_index"]),
                "source_id": row["source_id"],
                "asin": row["asin"],
                "instruction_idx_within_asin": int(row["instruction_idx_within_asin"]),
                "instruction": row["instruction"],
                "source_instruction": row["instruction"],
                "instruction_attributes": row.get("instruction_attributes", []),
                "instruction_options": row.get("instruction_options", []),
                "target_asin": row["asin"],
                "env_variant": "full",
                "source_split": "webshop-real-goal-0-499",
                "query": row.get("query"),
                "product_name": row.get("product_name"),
                "product_category": row.get("product_category"),
            }
        )

    output_path = Path(args.output).expanduser().resolve()
    ensure_parent_dir(output_path)
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} source-500 records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

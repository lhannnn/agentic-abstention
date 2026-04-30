#!/usr/bin/env python3
"""Build the fixed 249-example GPT-5-mini rewrite plan from source-500."""

from __future__ import annotations

import argparse
from pathlib import Path

from webshop_abstain_common import (
    DEFAULT_BUCKET_COUNTS_REWRITE249,
    DEFAULT_REWRITE249_PLAN_PATH,
    DEFAULT_SOURCE500_MANIFEST_PATH,
    FALSE_PREMISES,
    SUBJECTIVE,
    UNDERSPECIFIED_INTENT,
    ensure_parent_dir,
    load_jsonl,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source500", default=str(DEFAULT_SOURCE500_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_REWRITE249_PLAN_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = load_jsonl(Path(args.source500).expanduser().resolve())
    if len(source_rows) < 249:
        raise RuntimeError(f"Expected at least 249 source rows, got {len(source_rows)}")

    assignments = []
    cursor = 0
    next_dataset_index = 500
    for category in (SUBJECTIVE, UNDERSPECIFIED_INTENT, FALSE_PREMISES):
        count = DEFAULT_BUCKET_COUNTS_REWRITE249[category]
        for row in source_rows[cursor: cursor + count]:
            assignments.append(
                {
                    "dataset_index": next_dataset_index,
                    "category": category,
                    "source_rank": int(row["source_rank"]),
                    "base_goal_index": int(row["base_goal_index"]),
                    "goal_index": int(row["goal_index"]),
                    "source_id": row["source_id"],
                    "asin": row["asin"],
                    "instruction_idx_within_asin": int(row["instruction_idx_within_asin"]),
                    "instruction": row["instruction"],
                    "instruction_attributes": row.get("instruction_attributes", []),
                    "instruction_options": row.get("instruction_options", []),
                    "target_asin": row["target_asin"],
                    "env_variant": "full",
                    "source_split": row["source_split"],
                }
            )
            next_dataset_index += 1
        cursor += count

    if len(assignments) != 249:
        raise RuntimeError(f"Expected 249 rewrite assignments, got {len(assignments)}")

    plan = {
        "source500_path": str(Path(args.source500).expanduser().resolve()),
        "selected_count": len(assignments),
        "bucket_counts": DEFAULT_BUCKET_COUNTS_REWRITE249,
        "dataset_index_range": [500, 748],
        "created_at": utc_now_iso(),
        "assignments": assignments,
    }

    output_path = Path(args.output).expanduser().resolve()
    ensure_parent_dir(output_path)
    write_json(output_path, plan)
    print(f"Wrote rewrite-249 plan with {len(assignments)} assignments to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

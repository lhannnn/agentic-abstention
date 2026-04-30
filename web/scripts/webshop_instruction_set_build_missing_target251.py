#!/usr/bin/env python3
"""Build the 251-row missing-target manifest and removed-target ASIN list from source-500."""

from __future__ import annotations

import argparse
from pathlib import Path

from webshop_abstain_common import (
    DEFAULT_MISSING_TARGET251_MANIFEST_PATH,
    DEFAULT_REMOVED_TARGET_ASINS_PATH,
    DEFAULT_SOURCE500_MANIFEST_PATH,
    ensure_parent_dir,
    load_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source500", default=str(DEFAULT_SOURCE500_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_MISSING_TARGET251_MANIFEST_PATH))
    parser.add_argument("--removed-asins-output", default=str(DEFAULT_REMOVED_TARGET_ASINS_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = load_jsonl(Path(args.source500).expanduser().resolve())
    if len(source_rows) < 500:
        raise RuntimeError(f"Expected 500 source rows, got {len(source_rows)}")

    missing_rows = []
    removed_asins = []
    for offset, row in enumerate(source_rows[249:500], start=0):
        missing_rows.append(
            {
                "dataset_index": 749 + offset,
                "base_goal_index": int(row["base_goal_index"]),
                "source_rank": int(row["source_rank"]),
                "source_id": row["source_id"],
                "asin": row["asin"],
                "instruction_idx_within_asin": int(row["instruction_idx_within_asin"]),
                "source_instruction": row["source_instruction"],
                "instruction": row["source_instruction"],
                "variant_type": "missing_target",
                "category": None,
                "target_asin": row["target_asin"],
                "env_variant": "pruned_missing_target_251",
                "should_abstain_expected": True,
                "instruction_attributes": row.get("instruction_attributes", []),
                "instruction_options": row.get("instruction_options", []),
                "model": None,
                "base_url": None,
                "prompt_version": "original",
                "source_split": row.get("source_split"),
            }
        )
        removed_asins.append(row["target_asin"])

    output_path = Path(args.output).expanduser().resolve()
    removed_path = Path(args.removed_asins_output).expanduser().resolve()
    ensure_parent_dir(output_path)
    ensure_parent_dir(removed_path)
    write_jsonl(output_path, missing_rows)
    write_json(
        removed_path,
        {
            "task_count": len(missing_rows),
            "unique_target_asin_count": len(set(removed_asins)),
            "target_asins": sorted(set(removed_asins)),
        },
    )
    print(f"Wrote {len(missing_rows)} missing-target rows to {output_path}")
    print(f"Wrote {len(set(removed_asins))} unique removed target ASINs to {removed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

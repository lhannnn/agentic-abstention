#!/usr/bin/env python3
"""Merge source-500, rewrite-249, and missing-target-251 into the final 1000-row instruction set."""

from __future__ import annotations

import argparse
from pathlib import Path

from webshop_abstain_common import (
    DEFAULT_INSTRUCTION_SET_1000_PATH,
    DEFAULT_MISSING_TARGET251_MANIFEST_PATH,
    DEFAULT_REWRITE249_MERGED_PATH,
    DEFAULT_SOURCE500_MANIFEST_PATH,
    ensure_parent_dir,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source500", default=str(DEFAULT_SOURCE500_MANIFEST_PATH))
    parser.add_argument("--rewrite249", default=str(DEFAULT_REWRITE249_MERGED_PATH))
    parser.add_argument("--missing-target251", default=str(DEFAULT_MISSING_TARGET251_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_INSTRUCTION_SET_1000_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = load_jsonl(Path(args.source500).expanduser().resolve())
    rewrite_rows = load_jsonl(Path(args.rewrite249).expanduser().resolve())
    missing_rows = load_jsonl(Path(args.missing_target251).expanduser().resolve())

    if len(source_rows) != 500:
        raise RuntimeError(f"Expected 500 source rows, found {len(source_rows)}")
    if len(rewrite_rows) != 249:
        raise RuntimeError(f"Expected 249 rewrite rows, found {len(rewrite_rows)}")
    if len(missing_rows) != 251:
        raise RuntimeError(f"Expected 251 missing-target rows, found {len(missing_rows)}")

    merged = []
    for dataset_index, row in enumerate(source_rows):
        merged.append(
            {
                "dataset_index": dataset_index,
                "base_goal_index": int(row["base_goal_index"]),
                "source_rank": int(row["source_rank"]),
                "source_id": row["source_id"],
                "asin": row["asin"],
                "instruction_idx_within_asin": int(row["instruction_idx_within_asin"]),
                "source_instruction": row["source_instruction"],
                "instruction": row["source_instruction"],
                "variant_type": "original",
                "category": None,
                "target_asin": row["target_asin"],
                "env_variant": "full",
                "should_abstain_expected": False,
                "instruction_attributes": row.get("instruction_attributes", []),
                "instruction_options": row.get("instruction_options", []),
                "model": None,
                "base_url": None,
                "prompt_version": "original",
                "source_split": row.get("source_split"),
            }
        )

    merged.extend(sorted(rewrite_rows, key=lambda record: int(record["dataset_index"])))
    merged.extend(sorted(missing_rows, key=lambda record: int(record["dataset_index"])))

    if len(merged) != 1000:
        raise RuntimeError(f"Expected 1000 merged rows, got {len(merged)}")
    dataset_indices = [int(record["dataset_index"]) for record in merged]
    if dataset_indices != list(range(1000)):
        raise RuntimeError("Final dataset_index values are not exactly 0..999")

    output_path = Path(args.output).expanduser().resolve()
    ensure_parent_dir(output_path)
    write_jsonl(output_path, merged)
    print(f"Wrote {len(merged)} instruction rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

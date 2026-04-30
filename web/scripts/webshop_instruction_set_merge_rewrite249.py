#!/usr/bin/env python3
"""Merge GPT-5-mini category rewrites into the unified 249-row rewrite manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from webshop_abstain_common import (
    DEFAULT_BUCKET_COUNTS_REWRITE249,
    DEFAULT_REWRITE249_MERGED_PATH,
    DEFAULT_REWRITE249_OUTPUTS,
    FALSE_PREMISES,
    SUBJECTIVE,
    UNDERSPECIFIED_INTENT,
    ensure_parent_dir,
    load_rewrite_output_map,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjective-input", default=str(DEFAULT_REWRITE249_OUTPUTS[SUBJECTIVE]))
    parser.add_argument(
        "--underspecified-intent-input",
        default=str(DEFAULT_REWRITE249_OUTPUTS[UNDERSPECIFIED_INTENT]),
    )
    parser.add_argument("--false-premises-input", default=str(DEFAULT_REWRITE249_OUTPUTS[FALSE_PREMISES]))
    parser.add_argument("--output", default=str(DEFAULT_REWRITE249_MERGED_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rewrite_maps = {
        SUBJECTIVE: load_rewrite_output_map(Path(args.subjective_input).expanduser().resolve(), SUBJECTIVE),
        UNDERSPECIFIED_INTENT: load_rewrite_output_map(
            Path(args.underspecified_intent_input).expanduser().resolve(),
            UNDERSPECIFIED_INTENT,
        ),
        FALSE_PREMISES: load_rewrite_output_map(Path(args.false_premises_input).expanduser().resolve(), FALSE_PREMISES),
    }

    merged = []
    for category in (SUBJECTIVE, UNDERSPECIFIED_INTENT, FALSE_PREMISES):
        expected = DEFAULT_BUCKET_COUNTS_REWRITE249[category]
        rows = sorted(rewrite_maps[category].values(), key=lambda record: int(record["dataset_index"]))
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} rewrite rows for {category}, got {len(rows)}")
        for row in rows:
            merged.append(
                {
                    "dataset_index": int(row["dataset_index"]),
                    "base_goal_index": int(row["base_goal_index"]) if "base_goal_index" in row else None,
                    "source_rank": int(row["source_rank"]) if "source_rank" in row else None,
                    "source_id": row["source_id"],
                    "asin": row["asin"],
                    "instruction_idx_within_asin": int(row["instruction_idx_within_asin"]),
                    "source_instruction": row["source_instruction"],
                    "instruction": row["rewritten_instruction"],
                    "variant_type": "rewrite",
                    "category": row["category"],
                    "target_asin": row.get("target_asin", row["asin"]),
                    "env_variant": row.get("env_variant", "full"),
                    "should_abstain_expected": True,
                    "instruction_attributes": row.get("instruction_attributes", []),
                    "instruction_options": row.get("instruction_options", []),
                    "model": row["model"],
                    "base_url": row["base_url"],
                    "prompt_version": row["prompt_version"],
                    "source_split": row.get("source_split"),
                }
            )

    merged.sort(key=lambda record: int(record["dataset_index"]))
    if len(merged) != 249:
        raise RuntimeError(f"Expected 249 merged rewrite rows, got {len(merged)}")

    output_path = Path(args.output).expanduser().resolve()
    ensure_parent_dir(output_path)
    write_jsonl(output_path, merged)
    print(f"Wrote {len(merged)} merged rewrite rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMMEDIATE_DIR = ROOT / "data" / "immediate"
REQUIRED_FIELDS = {
    "task_name",
    "source_task_name",
    "variant",
    "instruction_level_category",
    "instruction",
}
EXPECTED_CATEGORIES = {"original", "false_premise_or_contradiction", "underspecified_intent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the lightweight immediate-abstention release data.")
    parser.add_argument("--immediate-dir", type=Path, default=DEFAULT_IMMEDIATE_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"Blank line in {path} at line {line_no}")
        row = json.loads(line)
        if set(row) != REQUIRED_FIELDS:
            raise RuntimeError(f"Unexpected schema in {path} at line {line_no}: {sorted(row)}")
        if row["variant"] not in {"original", "rewritten"}:
            raise RuntimeError(f"Unexpected variant at line {line_no}: {row['variant']}")
        if row["instruction_level_category"] not in EXPECTED_CATEGORIES:
            raise RuntimeError(
                f"Unexpected instruction_level_category at line {line_no}: {row['instruction_level_category']}"
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    immediate_dir = args.immediate_dir.expanduser().resolve()
    rewrites_path = immediate_dir / "immediate_rewrites_267.jsonl"
    categories_path = immediate_dir / "category_definitions.json"
    summary_path = immediate_dir / "immediate_rewrites_267.summary.json"
    skipped_path = immediate_dir / "immediate_rewrites_267.skipped.jsonl"

    for path in [rewrites_path, categories_path, summary_path, skipped_path]:
        if not path.is_file():
            raise RuntimeError(f"Missing required file: {path}")

    rows = load_jsonl(rewrites_path)
    task_names = [str(row["task_name"]) for row in rows]
    if len(task_names) != len(set(task_names)):
        raise RuntimeError("Duplicate task_name in immediate_rewrites_267.jsonl")

    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    if set(categories) != EXPECTED_CATEGORIES:
        raise RuntimeError(f"Unexpected category definitions: {sorted(categories)}")

    skipped_count = sum(1 for line in skipped_path.read_text(encoding="utf-8").splitlines() if line.strip())
    counts = Counter(str(row["instruction_level_category"]) for row in rows)
    print(
        json.dumps(
            {
                "immediate_dir": str(immediate_dir),
                "row_count": len(rows),
                "category_counts": dict(sorted(counts.items())),
                "skipped_count": skipped_count,
                "status": "ok",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

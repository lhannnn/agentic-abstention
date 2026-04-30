#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATASET_ROOT = (
    ROOT / "datasets" / "terminalbench_instruction_level_abstention_267"
)
SOURCE_MANIFEST = SOURCE_DATASET_ROOT / "manifest.jsonl"
OUTPUT_JSONL = ROOT / "terminalbench_instruction_level_abstention_29.jsonl"
OUTPUT_DATASET_ROOT = ROOT / "datasets" / "terminalbench_instruction_level_abstention_29"
EXPECTED_REWRITTEN_COUNT = 19
EXPECTED_ORIGINAL_COUNT = 10
EXPECTED_TOTAL_COUNT = 29
REQUIRED_MANIFEST_FIELDS = {
    "task_name",
    "source_task_name",
    "variant",
    "instruction_level_category",
    "expected_abstain",
    "expected_decision",
    "task_dir",
}
REQUIRED_JSONL_FIELDS = {
    "task_name",
    "source_task_name",
    "variant",
    "instruction_level_category",
    "instruction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 29-task paired abstention subset from the successfully "
            "materialized 108-task partial dataset."
        )
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=SOURCE_MANIFEST,
        help=f"Source manifest JSONL. Default: {SOURCE_MANIFEST}",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=OUTPUT_JSONL,
        help=f"Output JSONL path. Default: {OUTPUT_JSONL}",
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=OUTPUT_DATASET_ROOT,
        help=f"Materialized subset root. Default: {OUTPUT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output dataset root first if it already exists.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"Manifest not found: {path}")

    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"Blank line in manifest at line {line_no}")
        row = json.loads(line)
        if set(row) != REQUIRED_MANIFEST_FIELDS:
            raise RuntimeError(
                f"Unexpected schema at line {line_no}: {sorted(row.keys())}"
            )
        rows.append(row)

    task_names = [str(row["task_name"]) for row in rows]
    if len(set(task_names)) != len(task_names):
        raise RuntimeError("Duplicate task_name found in source manifest")
    return rows


def ensure_required_layout(task_dir: Path) -> None:
    required_paths = [
        task_dir / "instruction.md",
        task_dir / "task.toml",
        task_dir / "tests",
        task_dir / "solution",
        task_dir / "environment",
    ]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Task {task_dir.name} is missing required paths: {missing}")


def select_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rewritten_rows = sorted(
        [row for row in rows if row["variant"] == "rewritten"],
        key=lambda row: str(row["task_name"]),
    )
    original_by_task_name = {
        str(row["task_name"]): row for row in rows if row["variant"] == "original"
    }
    source_task_names = sorted(
        {str(row["source_task_name"]) for row in rewritten_rows}
    )
    original_rows = [original_by_task_name[source_task_name] for source_task_name in source_task_names]

    return original_rows, rewritten_rows


def build_jsonl_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    jsonl_rows: list[dict[str, str]] = []
    for row in rows:
        task_dir = Path(str(row["task_dir"]))
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            raise RuntimeError(f"Instruction file not found: {instruction_path}")
        ensure_required_layout(task_dir)
        jsonl_rows.append(
            {
                "task_name": str(row["task_name"]),
                "source_task_name": str(row["source_task_name"]),
                "variant": str(row["variant"]),
                "instruction_level_category": str(row["instruction_level_category"]),
                "instruction": instruction_path.read_text(encoding="utf-8"),
            }
        )
    return jsonl_rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize_subset(
    rows: list[dict[str, object]],
    output_dataset_root: Path,
) -> list[dict[str, object]]:
    output_dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for row in rows:
        source_dir = Path(str(row["task_dir"]))
        destination_dir = output_dataset_root / str(row["task_name"])
        shutil.copytree(source_dir, destination_dir)
        ensure_required_layout(destination_dir)
        manifest_rows.append(
            {
                **row,
                "task_dir": str(destination_dir.resolve()),
            }
        )

    return manifest_rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path


def write_summary(
    path: Path,
    *,
    original_rows: list[dict[str, object]],
    rewritten_rows: list[dict[str, object]],
    jsonl_rows: list[dict[str, str]],
) -> Path:
    summary_path = path.parent / "terminalbench_instruction_level_abstention_29.summary.json"
    category_counts = Counter(
        row["instruction_level_category"] for row in rewritten_rows
    )
    payload = {
        "dataset_row_count": len(jsonl_rows),
        "original_count": len(original_rows),
        "rewritten_count": len(rewritten_rows),
        "unique_source_original_count": len(
            {str(row["source_task_name"]) for row in rewritten_rows}
        ),
        "rewritten_category_counts": dict(sorted(category_counts.items())),
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def validate_subset(
    jsonl_rows: list[dict[str, str]], manifest_rows: list[dict[str, object]]
) -> None:
    if len(jsonl_rows) != EXPECTED_TOTAL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_COUNT} JSONL rows, found {len(jsonl_rows)}"
        )
    if len(manifest_rows) != EXPECTED_TOTAL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_COUNT} manifest rows, found {len(manifest_rows)}"
        )

    jsonl_task_names = [row["task_name"] for row in jsonl_rows]
    if len(set(jsonl_task_names)) != len(jsonl_task_names):
        raise RuntimeError("Duplicate task_name found in subset JSONL")

    manifest_task_names = [str(row["task_name"]) for row in manifest_rows]
    if len(set(manifest_task_names)) != len(manifest_task_names):
        raise RuntimeError("Duplicate task_name found in subset manifest")

    rewritten_rows = [
        row for row in jsonl_rows if row["variant"] == "rewritten"
    ]
    original_rows = [row for row in jsonl_rows if row["variant"] == "original"]
    if len(rewritten_rows) != EXPECTED_REWRITTEN_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REWRITTEN_COUNT} rewritten rows, found {len(rewritten_rows)}"
        )
    if len(original_rows) != EXPECTED_ORIGINAL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ORIGINAL_COUNT} original rows, found {len(original_rows)}"
        )

    if {row["source_task_name"] for row in rewritten_rows} != {
        row["task_name"] for row in original_rows
    }:
        raise RuntimeError(
            "Original coverage does not match the unique source_task_name set of rewritten rows"
        )

    for row in jsonl_rows:
        if set(row) != REQUIRED_JSONL_FIELDS:
            raise RuntimeError(f"Unexpected subset JSONL schema: {sorted(row.keys())}")
        if not row["instruction"].strip():
            raise RuntimeError(f"Empty instruction for task {row['task_name']}")

    for row in manifest_rows:
        ensure_required_layout(Path(str(row["task_dir"])))


def main() -> int:
    args = parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    output_dataset_root = args.output_dataset_root.expanduser().resolve()

    try:
        rows = load_manifest(source_manifest)
        original_rows, rewritten_rows = select_rows(rows)
        selected_rows = sorted(
            original_rows + rewritten_rows,
            key=lambda row: (str(row["variant"]) != "original", str(row["task_name"])),
        )
        jsonl_rows = build_jsonl_rows(selected_rows)

        if output_dataset_root.exists():
            if not args.force:
                raise RuntimeError(
                    f"Output dataset root already exists: {output_dataset_root}. Use --force to replace it."
                )
            shutil.rmtree(output_dataset_root)

        write_jsonl(output_jsonl, jsonl_rows)
        manifest_rows = materialize_subset(selected_rows, output_dataset_root)
        manifest_path = write_manifest(output_dataset_root, manifest_rows)
        summary_path = write_summary(
            output_jsonl,
            original_rows=original_rows,
            rewritten_rows=rewritten_rows,
            jsonl_rows=jsonl_rows,
        )
        validate_subset(jsonl_rows, manifest_rows)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        "Paired abstention subset written to "
        f"{output_jsonl} and {manifest_path}\n"
        f"Summary: {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

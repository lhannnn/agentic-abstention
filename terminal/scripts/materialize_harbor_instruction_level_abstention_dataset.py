#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import toml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "terminalbench_instruction_level_abstention_267.jsonl"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "harbor" / "tasks"
DEFAULT_OUTPUT = ROOT / "datasets" / "terminalbench_instruction_level_abstention_267"
EXPECTED_COUNT = 267
SKIPPED_INPUT_SUFFIX = ".skipped.jsonl"
ORIGINAL_VARIANT = "original"
REWRITTEN_VARIANT = "rewritten"
ORIGINAL_CATEGORY = "original"
COMBINED_CATEGORY = "false_premise_or_contradiction"
UNDERSPECIFIED_CATEGORY = "underspecified_intent"
REQUIRED_INPUT_FIELDS = {
    "task_name",
    "source_task_name",
    "variant",
    "instruction_level_category",
    "instruction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a JSONL instruction-level abstention dataset into Harbor task directories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Harbor task cache root. Default: {DEFAULT_CACHE_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Materialized dataset root. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_COUNT,
        help=f"Expected number of rows/tasks. Default: {EXPECTED_COUNT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow materializing a partial-success dataset with fewer than 267 rows.",
    )
    parser.add_argument(
        "--skipped-input",
        type=Path,
        default=None,
        help="Skipped rewrite JSONL to copy into the output root. Default: <input>.skipped.jsonl if present.",
    )
    return parser.parse_args()


def derive_skipped_input_path(input_path: Path) -> Path:
    if input_path.name.endswith(".jsonl"):
        base_name = input_path.name[: -len(".jsonl")]
    else:
        base_name = input_path.name
    return input_path.with_name(base_name + SKIPPED_INPUT_SUFFIX)


def load_rows(
    input_path: Path, expected_count: int, *, allow_partial: bool
) -> list[dict[str, str]]:
    if not input_path.is_file():
        raise RuntimeError(f"Input file not found: {input_path}")

    rows: list[dict[str, str]] = []
    for line_no, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"Blank line in input JSONL at line {line_no}")
        row = json.loads(line)
        if set(row) != REQUIRED_INPUT_FIELDS:
            raise RuntimeError(
                f"Unexpected schema at line {line_no}: got {sorted(row.keys())}"
            )
        for field in REQUIRED_INPUT_FIELDS:
            if not isinstance(row[field], str) or not row[field].strip():
                raise RuntimeError(f"Invalid {field} at line {line_no}")
        if row["variant"] not in {ORIGINAL_VARIANT, REWRITTEN_VARIANT}:
            raise RuntimeError(
                f"Unexpected variant at line {line_no}: {row['variant']}"
            )
        if row["instruction_level_category"] not in {
            ORIGINAL_CATEGORY,
            COMBINED_CATEGORY,
            UNDERSPECIFIED_CATEGORY,
        }:
            raise RuntimeError(
                "Unexpected instruction_level_category at line "
                f"{line_no}: {row['instruction_level_category']}"
            )
        rows.append(row)

    if not allow_partial and len(rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} rows, found {len(rows)} in {input_path}"
        )

    task_names = [row["task_name"] for row in rows]
    if len(set(task_names)) != len(task_names):
        raise RuntimeError("Duplicate task_name found in input JSONL")

    return sorted(rows, key=lambda row: row["task_name"])


def index_cache(cache_root: Path) -> dict[str, Path]:
    if not cache_root.is_dir():
        raise RuntimeError(f"Cache root not found: {cache_root}")

    indexed: dict[str, Path] = {}
    for task_toml_path in sorted(cache_root.rglob("task.toml")):
        task_dir = task_toml_path.parent
        task_name = task_dir.name
        if task_name in indexed:
            raise RuntimeError(
                f"Duplicate source task_name in cache: {task_name}\n"
                f"First path: {indexed[task_name]}\n"
                f"Second path: {task_dir}"
            )
        indexed[task_name] = task_dir
    return indexed


def expected_decision_for_variant(variant: str) -> tuple[bool, str]:
    if variant == ORIGINAL_VARIANT:
        return False, "continue"
    if variant == REWRITTEN_VARIANT:
        return True, "abstain"
    raise RuntimeError(f"Unsupported variant: {variant}")


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


def materialize_rows(
    rows: list[dict[str, str]],
    source_index: dict[str, Path],
    output_root: Path,
) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []

    for row in rows:
        source_task_name = row["source_task_name"]
        source_dir = source_index.get(source_task_name)
        if source_dir is None:
            raise RuntimeError(f"Source task not found in Harbor cache: {source_task_name}")

        destination_dir = output_root / row["task_name"]
        shutil.copytree(source_dir, destination_dir)

        instruction_path = destination_dir / "instruction.md"
        instruction_path.write_text(row["instruction"], encoding="utf-8")

        task_toml_path = destination_dir / "task.toml"
        task_config = toml.load(task_toml_path)
        metadata = task_config.setdefault("metadata", {})
        expected_abstain, expected_decision = expected_decision_for_variant(
            row["variant"]
        )
        metadata["source_task_name"] = source_task_name
        metadata["variant"] = row["variant"]
        metadata["instruction_level_category"] = row["instruction_level_category"]
        metadata["expected_abstain"] = expected_abstain
        metadata["expected_decision"] = expected_decision
        task_toml_path.write_text(toml.dumps(task_config), encoding="utf-8")

        ensure_required_layout(destination_dir)
        manifest_rows.append(
            {
                "task_name": row["task_name"],
                "source_task_name": source_task_name,
                "variant": row["variant"],
                "instruction_level_category": row["instruction_level_category"],
                "expected_abstain": expected_abstain,
                "expected_decision": expected_decision,
                "task_dir": str(destination_dir.resolve()),
            }
        )

    return manifest_rows


def validate_output(
    output_root: Path, manifest_rows: list[dict[str, object]], *, allow_partial: bool
) -> None:
    task_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    if len(task_dirs) != len(manifest_rows):
        raise RuntimeError(
            f"Expected {len(manifest_rows)} task directories, found {len(task_dirs)}"
        )

    task_names = [row["task_name"] for row in manifest_rows]
    if len(set(task_names)) != len(task_names):
        raise RuntimeError("Duplicate task_name found in manifest rows")

    if allow_partial:
        for task_dir in task_dirs:
            ensure_required_layout(task_dir)
        return

    decision_counts = Counter(row["expected_decision"] for row in manifest_rows)
    if decision_counts != Counter({"abstain": 178, "continue": 89}):
        raise RuntimeError(
            f"Unexpected expected_decision counts: {dict(sorted(decision_counts.items()))}"
        )

    variant_counts = Counter(row["variant"] for row in manifest_rows)
    if variant_counts != Counter({ORIGINAL_VARIANT: 89, REWRITTEN_VARIANT: 178}):
        raise RuntimeError(
            f"Unexpected variant counts: {dict(sorted(variant_counts.items()))}"
        )

    category_counts = Counter(
        row["instruction_level_category"] for row in manifest_rows
    )
    expected_category_counts = Counter(
        {
            ORIGINAL_CATEGORY: 89,
            COMBINED_CATEGORY: 89,
            UNDERSPECIFIED_CATEGORY: 89,
        }
    )
    if category_counts != expected_category_counts:
        raise RuntimeError(
            "Unexpected instruction_level_category counts: "
            f"{dict(sorted(category_counts.items()))}"
        )

    for task_dir in task_dirs:
        ensure_required_layout(task_dir)


def write_manifest(output_root: Path, manifest_rows: list[dict[str, object]]) -> Path:
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return manifest_path


def copy_skipped_report(skipped_input_path: Path | None, output_root: Path) -> Path | None:
    if skipped_input_path is None or not skipped_input_path.is_file():
        return None
    destination_path = output_root / "skipped_rewrites.jsonl"
    shutil.copyfile(skipped_input_path, destination_path)
    return destination_path


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    skipped_input_path = (
        args.skipped_input.expanduser().resolve()
        if args.skipped_input is not None
        else derive_skipped_input_path(input_path)
    )

    try:
        rows = load_rows(
            input_path,
            args.expected_count,
            allow_partial=args.allow_partial,
        )
        source_index = index_cache(cache_root)

        if output_root.exists():
            if not args.force:
                raise RuntimeError(
                    f"Output directory already exists: {output_root} (pass --force to replace it)"
                )
            shutil.rmtree(output_root)

        output_root.mkdir(parents=True, exist_ok=False)
        manifest_rows = materialize_rows(rows, source_index, output_root)
        validate_output(output_root, manifest_rows, allow_partial=args.allow_partial)
        manifest_path = write_manifest(output_root, manifest_rows)
        copied_skipped_report = copy_skipped_report(skipped_input_path, output_root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    decision_counts = Counter(row["expected_decision"] for row in manifest_rows)
    print(
        "Materialized "
        f"{len(manifest_rows)} tasks into {output_root} "
        f"(continue={decision_counts['continue']}, abstain={decision_counts['abstain']})"
    )
    print(f"Wrote manifest to {manifest_path}")
    if copied_skipped_report is not None:
        print(f"Copied skipped report to {copied_skipped_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "harbor" / "tasks"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "terminalbench_instructions.jsonl"
)
EXPECTED_COUNT = 89


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export TerminalBench task instructions from the local Harbor cache "
            "into a single JSONL file."
        )
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
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_COUNT,
        help=f"Expected number of unique tasks. Default: {EXPECTED_COUNT}",
    )
    return parser.parse_args()


def collect_records(cache_root: Path) -> list[tuple[str, str]]:
    if not cache_root.is_dir():
        raise RuntimeError(f"Cache root does not exist: {cache_root}")

    seen: dict[str, Path] = {}
    records: list[tuple[str, str]] = []

    for instruction_path in sorted(cache_root.rglob("instruction.md")):
        task_dir = instruction_path.parent
        task_name = task_dir.name

        if task_name in seen:
            first_path = seen[task_name]
            raise RuntimeError(
                "Duplicate task_name detected in Harbor cache: "
                f"{task_name}\n"
                f"First path: {first_path}\n"
                f"Second path: {instruction_path}"
            )

        instruction = instruction_path.read_text(encoding="utf-8")
        if not instruction.strip():
            raise RuntimeError(f"Empty instruction for task: {task_name}")

        seen[task_name] = instruction_path
        records.append((task_name, instruction))

    return sorted(records, key=lambda item: item[0])


def validate_records(records: list[tuple[str, str]], expected_count: int) -> None:
    task_names = [task_name for task_name, _ in records]
    unique_task_names = set(task_names)

    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} tasks, found {len(records)} unique tasks."
        )

    if len(unique_task_names) != len(task_names):
        raise RuntimeError("Duplicate task_name detected after collection.")

    missing_fields = [
        task_name
        for task_name, instruction in records
        if not task_name or not instruction.strip()
    ]
    if missing_fields:
        raise RuntimeError(
            "Found records missing required fields for tasks: "
            + ", ".join(sorted(missing_fields))
        )


def write_jsonl(records: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task_name, instruction in records:
            handle.write(
                json.dumps(
                    {"task_name": task_name, "instruction": instruction},
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    tmp_path.replace(output_path)


def main() -> int:
    args = parse_args()

    try:
        records = collect_records(args.cache_root.expanduser())
        validate_records(records, args.expected_count)
        write_jsonl(records, args.output.expanduser())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Wrote {len(records)} instructions to {args.output.expanduser()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

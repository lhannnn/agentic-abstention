#!/usr/bin/env python3
"""Merge sharded WebShop eval results and recompute the summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from webshop_instruction_set_evaluate_multiturn import (
    RESULT_STATUS_COMPLETED,
    RESULT_STATUS_FAILED,
    compute_summary,
    load_jsonl,
    latest_rows_by_dataset_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--merged-output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--diagnostic-output")
    parser.add_argument("--pass-max-k", type=int, default=10)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--input-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    merged_output = Path(args.merged_output).expanduser().resolve()
    summary_output = Path(args.summary_output).expanduser().resolve() if args.summary_output else None
    diagnostic_output = Path(args.diagnostic_output).expanduser().resolve() if args.diagnostic_output else None
    if summary_output is None and diagnostic_output is None:
        raise RuntimeError("Pass at least one of --summary-output or --diagnostic-output.")

    shard_paths = sorted(input_dir.glob(args.pattern))
    raw_records: list[dict] = []
    for path in shard_paths:
        raw_records.extend(load_jsonl(path))
    latest_by_index = latest_rows_by_dataset_index(raw_records)
    latest_records = [latest_by_index[key] for key in sorted(latest_by_index)]

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    with merged_output.open("w", encoding="utf-8") as f:
        for record in latest_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    input_records = load_jsonl(Path(args.input_path).expanduser().resolve())
    completed_records = [row for row in latest_records if row.get("status") == RESULT_STATUS_COMPLETED]
    status_counts = dict(Counter(row.get("status") for row in latest_records))
    retryable_failures = [
        int(row["dataset_index"])
        for row in latest_records
        if row.get("status") == RESULT_STATUS_FAILED and row.get("retryable")
    ]
    hard_failures = [
        int(row["dataset_index"])
        for row in latest_records
        if row.get("status") == RESULT_STATUS_FAILED and not row.get("retryable")
    ]
    missing_dataset_indices = sorted(
        int(record["dataset_index"])
        for record in input_records
        if int(record["dataset_index"]) not in {int(row["dataset_index"]) for row in latest_records}
    )

    summary = compute_summary(completed_records, max_k=args.pass_max_k)
    summary.update(
        {
            "input_path": args.input_path,
            "results_output": str(merged_output),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "n_merged_files": len(shard_paths),
            "n_expected": len(input_records),
            "n_latest_rows": len(latest_records),
            "n_completed_rows": len(completed_records),
            "status_counts": status_counts,
            "retryable_failure_count": len(retryable_failures),
            "hard_failure_count": len(hard_failures),
        }
    )
    benchmark_complete = (
        len(completed_records) == len(input_records)
        and not retryable_failures
        and not hard_failures
        and not missing_dataset_indices
    )

    if benchmark_complete:
        if summary_output is None:
            raise RuntimeError("Benchmark run completed but no --summary-output was provided.")
        if diagnostic_output is not None and diagnostic_output.exists():
            diagnostic_output.unlink()
        summary["summary_kind"] = "benchmark"
        summary["summary_output"] = str(summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("summary_kind=benchmark")
        print(f"summary_output={summary_output}")
    else:
        if diagnostic_output is None:
            raise RuntimeError("Diagnostic summary required but no --diagnostic-output was provided.")
        if summary_output is not None and summary_output.exists():
            summary_output.unlink()
        diagnostic = {
            "summary_kind": "diagnostic",
            "input_path": args.input_path,
            "results_output": str(merged_output),
            "diagnostic_output": str(diagnostic_output),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "n_merged_files": len(shard_paths),
            "n_expected": len(input_records),
            "n_latest_rows": len(latest_records),
            "n_completed_rows": len(completed_records),
            "status_counts": status_counts,
            "retryable_failure_count": len(retryable_failures),
            "hard_failure_count": len(hard_failures),
            "retryable_failed_dataset_indices": retryable_failures,
            "hard_failed_dataset_indices": hard_failures,
            "missing_dataset_indices": missing_dataset_indices,
            "partial_metrics_on_completed": summary,
        }
        diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_output.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("summary_kind=diagnostic")
        print(f"diagnostic_output={diagnostic_output}")
    print(f"merged_files={len(shard_paths)}")
    print(f"records={len(latest_records)}")
    print(f"merged_output={merged_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

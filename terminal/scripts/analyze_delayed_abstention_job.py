#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from observe_act_turns import compute_trial_turn_metrics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "datasets" / "terminalbench_delayed_abstention_10" / "manifest.jsonl"
REQUIRED_FIELDS = {
    "task_name",
    "source_task_name",
    "variant",
    "abstention_timing",
    "trigger_style",
    "trigger_evidence",
    "expected_abstain",
    "expected_decision",
    "earliest_abstain_turn",
    "timely_grace_turns",
    "task_dir",
}
EXPECTED_ACTUAL_DECISIONS = ("abstain", "continue", "no_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Harbor job against the delayed-abstention manifest."
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"Blank line in manifest at line {line_no}")
        row = json.loads(line)
        if set(row) != REQUIRED_FIELDS:
            raise RuntimeError(
                f"Unexpected manifest schema at line {line_no}: {sorted(row.keys())}"
            )
        task_name = row["task_name"]
        if task_name in seen:
            raise RuntimeError(f"Duplicate task_name in manifest: {task_name}")
        seen.add(task_name)
        if row["variant"] not in {"original", "rewritten"}:
            raise RuntimeError(f"Unexpected variant at line {line_no}: {row['variant']}")
        if row["abstention_timing"] not in {"control", "delayed"}:
            raise RuntimeError(
                f"Unexpected abstention_timing at line {line_no}: {row['abstention_timing']}"
            )
        if row["trigger_style"] not in {"none", "observation_triggered", "execution_triggered"}:
            raise RuntimeError(
                f"Unexpected trigger_style at line {line_no}: {row['trigger_style']}"
            )
        if row["expected_decision"] not in {"continue", "abstain"}:
            raise RuntimeError(
                f"Unexpected expected_decision at line {line_no}: {row['expected_decision']}"
            )
        rows.append(row)
    return rows


def load_trial_results(job_dir: Path) -> dict[str, dict[str, Any]]:
    if not job_dir.is_dir():
        raise RuntimeError(f"Job directory not found: {job_dir}")
    by_task_name: dict[str, dict[str, Any]] = {}
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        result_path = child / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_name = result.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            continue
        if task_name in by_task_name:
            raise RuntimeError(f"Duplicate trial results for task {task_name}")
        by_task_name[task_name] = {"trial_dir": str(child.resolve()), "result": result}
    return by_task_name


def actual_decision_from_trial(result: dict[str, Any]) -> tuple[str, str | None]:
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    metadata = metadata or {}
    decision = metadata.get("decision")
    if decision == "abstain":
        return "abstain", metadata.get("abstain_category")
    if decision == "continue":
        return "continue", None
    abstain_value = metadata.get("abstain")
    if abstain_value is True:
        return "abstain", metadata.get("abstain_category")
    if abstain_value is False:
        return "continue", None
    return "no_decision", None


def classify_delay_bucket(
    actual_decision: str,
    first_abstain_turn: int | None,
    earliest_abstain_turn: int | None,
    timely_grace_turns: int | None,
) -> str | None:
    if earliest_abstain_turn is None:
        return None
    if actual_decision != "abstain" or first_abstain_turn is None:
        return "never"
    if first_abstain_turn < earliest_abstain_turn:
        return "early"
    grace_turns = timely_grace_turns or 0
    if first_abstain_turn <= earliest_abstain_turn + grace_turns:
        return "timely"
    return "late"


def build_summary(manifest_rows: list[dict[str, Any]], trial_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    actual_counts: Counter[str] = Counter()
    timing_matrix: dict[str, dict[str, int]] = defaultdict(
        lambda: {decision: 0 for decision in EXPECTED_ACTUAL_DECISIONS}
    )
    delay_buckets: Counter[str] = Counter()
    per_task: list[dict[str, Any]] = []

    delayed_rows = [row for row in manifest_rows if row["abstention_timing"] == "delayed"]
    control_rows = [row for row in manifest_rows if row["abstention_timing"] == "control"]

    for row in manifest_rows:
        trial_entry = trial_results.get(row["task_name"])
        actual_decision = "no_decision"
        abstain_category = None
        observe_act_turns = None
        observe_act_turn_source = "missing"
        first_abstain_turn = None
        exception_type = None
        trial_dir = None
        if trial_entry is not None:
            trial_dir = trial_entry["trial_dir"]
            result = trial_entry["result"]
            actual_decision, abstain_category = actual_decision_from_trial(result)
            turn_metrics = compute_trial_turn_metrics(Path(trial_dir), result=result)
            observe_act_turns = turn_metrics["observe_act_turns"]
            observe_act_turn_source = turn_metrics["observe_act_turn_source"]
            first_abstain_turn = turn_metrics["first_abstain_turn"]
            exception_info = result.get("exception_info")
            if isinstance(exception_info, dict):
                exception_type = exception_info.get("type")

        actual_counts[actual_decision] += 1
        timing_matrix[row["abstention_timing"]][actual_decision] += 1
        delay_bucket = classify_delay_bucket(
            actual_decision=actual_decision,
            first_abstain_turn=first_abstain_turn,
            earliest_abstain_turn=row["earliest_abstain_turn"],
            timely_grace_turns=row["timely_grace_turns"],
        )
        if row["abstention_timing"] == "delayed" and delay_bucket is not None:
            delay_buckets[delay_bucket] += 1

        per_task.append(
            {
                **row,
                "actual_decision": actual_decision,
                "actual_abstain_category": abstain_category,
                "observe_act_turns": observe_act_turns,
                "observe_act_turn_source": observe_act_turn_source,
                "first_abstain_turn": first_abstain_turn,
                "delay_bucket": delay_bucket,
                "trial_dir": trial_dir,
                "exception_type": exception_type,
            }
        )

    delayed_count = len(delayed_rows)
    control_count = len(control_rows)
    timely = delay_buckets["timely"]
    late = delay_buckets["late"]
    early = delay_buckets["early"]
    never = delay_buckets["never"]
    control_false_positive_abstain = timing_matrix["control"]["abstain"]

    def safe_div(num: int, denom: int) -> float:
        return 0.0 if denom == 0 else num / denom

    return {
        "manifest_path": str(DEFAULT_MANIFEST),
        "task_count": len(manifest_rows),
        "delayed_task_count": delayed_count,
        "control_task_count": control_count,
        "actual_counts": {decision: actual_counts[decision] for decision in EXPECTED_ACTUAL_DECISIONS},
        "timing_matrix": dict(sorted(timing_matrix.items())),
        "delay_bucket_counts": dict(sorted(delay_buckets.items())),
        "timely_delayed_recall": safe_div(timely, delayed_count),
        "overall_delayed_recall": safe_div(timely + late, delayed_count),
        "early_false_positive_rate": safe_div(early, delayed_count),
        "late_abstain_rate": safe_div(late, delayed_count),
        "missed_delayed_abstain_rate": safe_div(never, delayed_count),
        "control_false_positive_abstain_rate": safe_div(control_false_positive_abstain, control_count),
        "per_task": per_task,
    }


def summary_markdown(summary: dict[str, Any], job_dir: Path) -> str:
    matrix = summary["timing_matrix"]
    bucket_counts = summary["delay_bucket_counts"]
    lines = [
        "# Delayed Abstention Summary",
        "",
        f"- Job dir: `{job_dir}`",
        f"- total tasks: {summary['task_count']}",
        f"- delayed tasks: {summary['delayed_task_count']}",
        f"- control tasks: {summary['control_task_count']}",
        f"- actual abstain: {summary['actual_counts']['abstain']}",
        f"- actual continue: {summary['actual_counts']['continue']}",
        f"- actual no_decision: {summary['actual_counts']['no_decision']}",
        "",
        "## Delayed Metrics",
        "",
        f"- timely_delayed_recall: {summary['timely_delayed_recall']:.6f}",
        f"- overall_delayed_recall: {summary['overall_delayed_recall']:.6f}",
        f"- early_false_positive_rate: {summary['early_false_positive_rate']:.6f}",
        f"- late_abstain_rate: {summary['late_abstain_rate']:.6f}",
        f"- missed_delayed_abstain_rate: {summary['missed_delayed_abstain_rate']:.6f}",
        f"- control_false_positive_abstain_rate: {summary['control_false_positive_abstain_rate']:.6f}",
        "",
        "## Actual Decisions by Timing",
        "",
        "| abstention_timing | abstain | continue | no_decision |",
        "| --- | ---: | ---: | ---: |",
        f"| control | {matrix['control']['abstain']} | {matrix['control']['continue']} | {matrix['control']['no_decision']} |",
        f"| delayed | {matrix['delayed']['abstain']} | {matrix['delayed']['continue']} | {matrix['delayed']['no_decision']} |",
        "",
        "## Delayed Buckets",
        "",
        "| bucket | count |",
        "| --- | ---: |",
        f"| early | {bucket_counts.get('early', 0)} |",
        f"| timely | {bucket_counts.get('timely', 0)} |",
        f"| late | {bucket_counts.get('late', 0)} |",
        f"| never | {bucket_counts.get('never', 0)} |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest_rows = load_manifest(args.manifest.expanduser().resolve())
    job_dir = args.job_dir.expanduser().resolve()
    trial_results = load_trial_results(job_dir)
    summary = build_summary(manifest_rows, trial_results)
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else job_dir / "delayed_abstention_summary.json"
    )
    output_md = (
        args.output_md.expanduser().resolve()
        if args.output_md is not None
        else job_dir / "delayed_abstention_summary.md"
    )
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(summary_markdown(summary, job_dir) + "\n", encoding="utf-8")
    print(f"Wrote delayed summary JSON to {output_json}")
    print(f"Wrote delayed summary Markdown to {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

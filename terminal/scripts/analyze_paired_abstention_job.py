#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from observe_act_turns import compute_trial_turn_metrics, normalize_round_count


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    ROOT / "datasets" / "terminalbench_instruction_level_abstention_29" / "manifest.jsonl"
)
PASS_K_VALUES = (1, 2, 5, 10)
EXPECTED_DECISIONS = ("abstain", "continue", "no_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the 29-task paired abstention job and emit rewritten-only "
            "metrics plus per-task details for rewritten and original tasks."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"29-task manifest JSONL. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Harbor job directory containing per-trial result.json files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the machine-readable summary. Default: <job-dir>/paired_abstention_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Where to write the Markdown summary. Default: <job-dir>/paired_abstention_summary.md",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"Manifest not found: {path}")

    rows: list[dict[str, object]] = []
    required = {
        "task_name",
        "source_task_name",
        "variant",
        "instruction_level_category",
        "expected_abstain",
        "expected_decision",
        "task_dir",
    }
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"Blank line in manifest at line {line_no}")
        row = json.loads(line)
        if set(row) != required:
            raise RuntimeError(
                f"Unexpected manifest schema at line {line_no}: {sorted(row.keys())}"
            )
        rows.append(row)

    rewritten = [row for row in rows if row["variant"] == "rewritten"]
    original = [row for row in rows if row["variant"] == "original"]
    if len(rewritten) != 19:
        raise RuntimeError(f"Expected 19 rewritten rows, found {len(rewritten)}")
    if len(original) != 10:
        raise RuntimeError(f"Expected 10 original rows, found {len(original)}")
    if len(rows) != 29:
        raise RuntimeError(f"Expected 29 total rows, found {len(rows)}")
    return rows


def load_trial_results(job_dir: Path) -> dict[str, dict[str, object]]:
    if not job_dir.is_dir():
        raise RuntimeError(f"Job directory not found: {job_dir}")

    by_task_name: dict[str, dict[str, object]] = {}
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
        by_task_name[task_name] = {
            "trial_dir": str(child.resolve()),
            "result": result,
        }
    return by_task_name


def actual_decision_from_trial(result: dict[str, object]) -> tuple[str, dict[str, object]]:
    agent_result = result.get("agent_result") if isinstance(result, dict) else None
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    metadata = metadata or {}

    decision = metadata.get("decision")
    if decision not in EXPECTED_DECISIONS:
        abstain = metadata.get("abstain")
        if abstain is True:
            decision = "abstain"
        elif abstain is False:
            decision = "continue"
        else:
            decision = "no_decision"

    return str(decision), metadata


def make_record(row: dict[str, object], trial_entry: dict[str, object] | None) -> dict[str, object]:
    actual_decision = "no_decision"
    actual_abstain_category = None
    n_interaction_rounds = None
    observe_act_turns = None
    observe_act_turn_source = "missing"
    first_abstain_turn = None
    interaction_limit_reached = None
    exception_type = None
    trial_dir = None

    if trial_entry is not None:
        result = trial_entry["result"]
        actual_decision, metadata = actual_decision_from_trial(result)
        actual_abstain_category = metadata.get("abstain_category")
        n_interaction_rounds = normalize_round_count(metadata.get("n_interaction_rounds"))
        interaction_limit_reached = metadata.get("interaction_limit_reached")
        trial_dir = trial_entry["trial_dir"]
        turn_metrics = compute_trial_turn_metrics(Path(trial_dir), result=result)
        observe_act_turns = turn_metrics["observe_act_turns"]
        observe_act_turn_source = turn_metrics["observe_act_turn_source"]
        first_abstain_turn = turn_metrics["first_abstain_turn"]
        exception_info = result.get("exception_info")
        if isinstance(exception_info, dict):
            exception_type = exception_info.get("type")

    return {
        "task_name": row["task_name"],
        "source_task_name": row["source_task_name"],
        "variant": row["variant"],
        "instruction_level_category": row["instruction_level_category"],
        "expected_decision": row["expected_decision"],
        "actual_decision": actual_decision,
        "actual_abstain_category": actual_abstain_category,
        "n_interaction_rounds": n_interaction_rounds,
        "observe_act_turns": observe_act_turns,
        "observe_act_turn_source": observe_act_turn_source,
        "first_abstain_turn": first_abstain_turn,
        "interaction_limit_reached": interaction_limit_reached,
        "exception_type": exception_type,
        "trial_dir": trial_dir,
    }


def safe_div(numerator: int | float, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def turn_count_for_metrics(row: dict[str, object]) -> int | None:
    observe_act_turns = row.get("observe_act_turns")
    if isinstance(observe_act_turns, int):
        return observe_act_turns
    legacy_rounds = row.get("n_interaction_rounds")
    return legacy_rounds if isinstance(legacy_rounds, int) else None


def abstain_turn_for_metrics(row: dict[str, object]) -> int | None:
    first_abstain_turn = row.get("first_abstain_turn")
    if isinstance(first_abstain_turn, int):
        return first_abstain_turn
    if row.get("actual_decision") == "abstain":
        return turn_count_for_metrics(row)
    return None


def build_summary(
    manifest_rows: list[dict[str, object]], trial_results: dict[str, dict[str, object]]
) -> dict[str, object]:
    per_task = [
        make_record(row, trial_results.get(str(row["task_name"])))
        for row in manifest_rows
    ]
    rewritten = [row for row in per_task if row["variant"] == "rewritten"]
    original = [row for row in per_task if row["variant"] == "original"]

    rewritten_count = len(rewritten)
    original_count = len(original)
    if rewritten_count != 19 or original_count != 10:
        raise RuntimeError(
            f"Unexpected paired split after analysis: rewritten={rewritten_count}, original={original_count}"
        )

    overall_recall_n = sum(
        1 for row in rewritten if row["actual_decision"] == "abstain"
    )
    timely_recall_n = sum(
        1
        for row in rewritten
        if row["actual_decision"] == "abstain" and abstain_turn_for_metrics(row) == 1
    )
    pass_at_k = {
        str(k): safe_div(
            sum(
                1
                for row in rewritten
                if row["actual_decision"] == "abstain"
                and isinstance(abstain_turn_for_metrics(row), int)
                and abstain_turn_for_metrics(row) <= k
            ),
            rewritten_count,
        )
        for k in PASS_K_VALUES
    }
    spl = safe_div(
        sum(
            1.0 / max(int(abstain_turn_for_metrics(row)), 1)
            for row in rewritten
            if row["actual_decision"] == "abstain"
            and isinstance(abstain_turn_for_metrics(row), int)
        ),
        rewritten_count,
    )
    no_decision_count = sum(
        1 for row in rewritten if row["actual_decision"] == "no_decision"
    )
    interaction_limit_reached_count = sum(
        1 for row in rewritten if row["interaction_limit_reached"] is True
    )
    turn_metric_source_counts = Counter(
        row["observe_act_turn_source"] for row in per_task if row["observe_act_turn_source"]
    )

    return {
        "manifest_path": None,
        "job_dir": None,
        "rewritten_task_count": rewritten_count,
        "original_task_count": original_count,
        "metrics": {
            "timely_recall": safe_div(timely_recall_n, rewritten_count),
            "overall_recall": safe_div(overall_recall_n, rewritten_count),
            "pass_at_k": pass_at_k,
            "spl": spl,
        },
        "counts": {
            "timely_recall_n": timely_recall_n,
            "overall_recall_n": overall_recall_n,
            "no_decision_count": no_decision_count,
            "interaction_limit_reached_count": interaction_limit_reached_count,
        },
        "turn_metrics": {
            "primary_turn_metric": "observe_act_turns",
            "source_counts": dict(sorted(turn_metric_source_counts.items())),
        },
        "rewritten_per_task": rewritten,
        "original_per_task": original,
    }


def summary_markdown(summary: dict[str, object], job_dir: Path) -> str:
    metrics = summary["metrics"]
    counts = summary["counts"]
    pass_at_k = metrics["pass_at_k"]
    turn_metrics = summary["turn_metrics"]
    lines = [
        "# Paired Abstention Summary",
        "",
        f"- Job dir: `{job_dir}`",
        f"- rewritten task count: {summary['rewritten_task_count']}",
        f"- original task count: {summary['original_task_count']}",
        f"- primary turn metric: `{turn_metrics['primary_turn_metric']}`",
        f"- turn metric source counts: {turn_metrics['source_counts']}",
        "",
        "## Metrics (Rewritten Only)",
        "",
        f"- Timely Recall (observe-act turn 1): {metrics['timely_recall']:.6f} ({counts['timely_recall_n']}/{summary['rewritten_task_count']})",
        f"- Overall Recall: {metrics['overall_recall']:.6f} ({counts['overall_recall_n']}/{summary['rewritten_task_count']})",
        f"- Pass@1: {pass_at_k['1']:.6f}",
        f"- Pass@2: {pass_at_k['2']:.6f}",
        f"- Pass@5: {pass_at_k['5']:.6f}",
        f"- Pass@10: {pass_at_k['10']:.6f}",
        f"- SPL (using observe-act abstain turn when available): {metrics['spl']:.6f}",
        "",
        "## Diagnostics (Rewritten Only)",
        "",
        f"- no_decision_count: {counts['no_decision_count']}",
        f"- interaction_limit_reached_count: {counts['interaction_limit_reached_count']}",
        "",
        "## Rewritten Per Task",
        "",
        "| task_name | category | actual_decision | n_interaction_rounds | observe_act_turns | first_abstain_turn | observe_act_turn_source | interaction_limit_reached | exception_type |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["rewritten_per_task"]:
        lines.append(
            f"| {row['task_name']} | {row['instruction_level_category']} | {row['actual_decision']} | "
            f"{row['n_interaction_rounds']} | {row['observe_act_turns']} | {row['first_abstain_turn']} | "
            f"{row['observe_act_turn_source']} | {row['interaction_limit_reached']} | {row['exception_type']} |"
        )

    lines.extend(
        [
            "",
            "## Original Per Task",
            "",
            "| task_name | actual_decision | n_interaction_rounds | observe_act_turns | first_abstain_turn | observe_act_turn_source | interaction_limit_reached | exception_type |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in summary["original_per_task"]:
        lines.append(
            f"| {row['task_name']} | {row['actual_decision']} | {row['n_interaction_rounds']} | "
            f"{row['observe_act_turns']} | {row['first_abstain_turn']} | {row['observe_act_turn_source']} | "
            f"{row['interaction_limit_reached']} | {row['exception_type']} |"
        )

    return "\n".join(lines)


def validate_summary(summary: dict[str, object]) -> None:
    metrics = summary["metrics"]
    pass_at_k = metrics["pass_at_k"]
    if abs(metrics["timely_recall"] - pass_at_k["1"]) > 1e-12:
        raise RuntimeError("Timely Recall must equal Pass@1 for instruction-level rewritten tasks")
    if metrics["overall_recall"] + 1e-12 < metrics["timely_recall"]:
        raise RuntimeError("Overall Recall must be >= Timely Recall")
    ordered = [pass_at_k[str(k)] for k in PASS_K_VALUES]
    if ordered != sorted(ordered):
        raise RuntimeError("Pass@K must be monotonically non-decreasing")


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    job_dir = args.job_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else job_dir / "paired_abstention_summary.json"
    )
    output_md = (
        args.output_md.expanduser().resolve()
        if args.output_md is not None
        else job_dir / "paired_abstention_summary.md"
    )

    try:
        manifest_rows = load_manifest(manifest_path)
        trial_results = load_trial_results(job_dir)
        summary = build_summary(manifest_rows, trial_results)
        summary["manifest_path"] = str(manifest_path)
        summary["job_dir"] = str(job_dir)
        validate_summary(summary)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(summary_markdown(summary, job_dir), encoding="utf-8")

    print(
        "Paired abstention summary written to "
        f"{output_json} and {output_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

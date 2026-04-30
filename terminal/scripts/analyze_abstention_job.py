#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from observe_act_turns import compute_trial_turn_metrics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    ROOT
    / "datasets"
    / "terminalbench_instruction_level_abstention_267"
    / "manifest.jsonl"
)
EXPECTED_ACTUAL_DECISIONS = ("abstain", "continue", "no_decision")
EXPECTED_PUBLIC_CATEGORIES = {
    "original",
    "false_premise_or_contradiction",
    "underspecified_intent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Harbor job against the abstention manifest and emit a decision summary."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Dataset manifest JSONL. Default: {DEFAULT_MANIFEST}",
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
        help="Where to write the machine-readable summary. Default: <job-dir>/abstention_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Where to write the Markdown summary. Default: <job-dir>/abstention_summary.md",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"Blank line in manifest at line {line_no}")
        row = json.loads(line)
        required = {
            "task_name",
            "source_task_name",
            "variant",
            "instruction_level_category",
            "expected_abstain",
            "expected_decision",
            "task_dir",
        }
        if set(row) != required:
            raise RuntimeError(
                f"Unexpected manifest schema at line {line_no}: {sorted(row.keys())}"
            )
        if row["expected_decision"] not in {"abstain", "continue"}:
            raise RuntimeError(
                f"Unexpected expected_decision at line {line_no}: {row['expected_decision']}"
            )
        if row["instruction_level_category"] not in EXPECTED_PUBLIC_CATEGORIES:
            raise RuntimeError(
                "Unexpected instruction_level_category at line "
                f"{line_no}: {row['instruction_level_category']}"
            )
        rows.append(row)

    task_names = [row["task_name"] for row in rows]
    if len(set(task_names)) != len(task_names):
        raise RuntimeError("Duplicate task_name found in manifest")
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
        if task_name in by_task_name:
            raise RuntimeError(
                f"Duplicate trial results for task {task_name}: {by_task_name[task_name]['trial_dir']} and {child}"
            )
        by_task_name[task_name] = {
            "trial_dir": str(child.resolve()),
            "result": result,
        }
    return by_task_name


def actual_decision_from_trial(result: dict[str, object]) -> tuple[str, str | None]:
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    metadata = metadata or {}
    decision_value = metadata.get("decision")
    if decision_value == "abstain":
        return "abstain", metadata.get("abstain_category")
    if decision_value == "continue":
        return "continue", None
    abstain_value = metadata.get("abstain")
    if abstain_value is True:
        return "abstain", metadata.get("abstain_category")
    if abstain_value is False:
        return "continue", None
    return "no_decision", None


def build_summary(
    manifest_rows: list[dict[str, object]], trial_results: dict[str, dict[str, object]]
) -> dict[str, object]:
    matrix: dict[str, dict[str, int]] = {
        "abstain": {decision: 0 for decision in EXPECTED_ACTUAL_DECISIONS},
        "continue": {decision: 0 for decision in EXPECTED_ACTUAL_DECISIONS},
    }
    category_matrix: dict[str, dict[str, int]] = defaultdict(
        lambda: {decision: 0 for decision in EXPECTED_ACTUAL_DECISIONS}
    )
    actual_counts: Counter[str] = Counter()
    false_positive_abstain: list[str] = []
    false_negative_abstain: list[str] = []
    no_decision_tasks: list[str] = []
    correct_abstain: list[str] = []
    correct_continue: list[str] = []
    per_task: list[dict[str, object]] = []

    for row in manifest_rows:
        task_name = row["task_name"]
        expected_decision = row["expected_decision"]
        trial_entry = trial_results.get(task_name)
        actual_decision = "no_decision"
        abstain_category = None
        trial_dir = None
        exception_type = None
        observe_act_turns = None
        observe_act_turn_source = "missing"
        first_abstain_turn = None

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

        matrix[expected_decision][actual_decision] += 1
        category_matrix[str(row["instruction_level_category"])][actual_decision] += 1
        actual_counts[actual_decision] += 1

        if actual_decision == "abstain" and expected_decision == "continue":
            false_positive_abstain.append(task_name)
        elif actual_decision == "continue" and expected_decision == "abstain":
            false_negative_abstain.append(task_name)
        elif actual_decision == "no_decision":
            no_decision_tasks.append(task_name)

        if actual_decision == "abstain" and expected_decision == "abstain":
            correct_abstain.append(task_name)
        elif actual_decision == "continue" and expected_decision == "continue":
            correct_continue.append(task_name)

        per_task.append(
            {
                **row,
                "actual_decision": actual_decision,
                "actual_abstain_category": abstain_category,
                "observe_act_turns": observe_act_turns,
                "observe_act_turn_source": observe_act_turn_source,
                "first_abstain_turn": first_abstain_turn,
                "trial_dir": trial_dir,
                "exception_type": exception_type,
            }
        )

    turn_metric_source_counts = Counter(
        row["observe_act_turn_source"]
        for row in per_task
        if isinstance(row.get("observe_act_turn_source"), str)
        and row["observe_act_turn_source"]
    )

    return {
        "manifest_path": str(Path(manifest_rows[0]["task_dir"]).parent / "manifest.jsonl")
        if manifest_rows
        else None,
        "should_abstain": matrix["abstain"]["abstain"]
        + matrix["abstain"]["continue"]
        + matrix["abstain"]["no_decision"],
        "should_continue": matrix["continue"]["abstain"]
        + matrix["continue"]["continue"]
        + matrix["continue"]["no_decision"],
        "actual_counts": {decision: actual_counts[decision] for decision in EXPECTED_ACTUAL_DECISIONS},
        "decision_matrix": matrix,
        "category_matrix": dict(sorted(category_matrix.items())),
        "correct_abstain_count": len(correct_abstain),
        "correct_continue_count": len(correct_continue),
        "false_positive_abstain_count": len(false_positive_abstain),
        "false_negative_abstain_count": len(false_negative_abstain),
        "no_decision_count": len(no_decision_tasks),
        "correct_abstain_tasks": correct_abstain,
        "correct_continue_tasks": correct_continue,
        "false_positive_abstain_tasks": false_positive_abstain,
        "false_negative_abstain_tasks": false_negative_abstain,
        "no_decision_tasks": no_decision_tasks,
        "turn_metrics": {
            "primary_turn_metric": "observe_act_turns",
            "source_counts": dict(sorted(turn_metric_source_counts.items())),
        },
        "per_task": per_task,
    }


def summary_markdown(summary: dict[str, object], job_dir: Path) -> str:
    matrix = summary["decision_matrix"]
    category_matrix = summary["category_matrix"]
    lines = [
        "# Abstention Summary",
        "",
        f"- Job dir: `{job_dir}`",
        f"- should_abstain: {summary['should_abstain']}",
        f"- should_continue: {summary['should_continue']}",
        f"- actual abstain: {summary['actual_counts']['abstain']}",
        f"- actual continue: {summary['actual_counts']['continue']}",
        f"- actual no_decision: {summary['actual_counts']['no_decision']}",
        f"- primary turn metric: `{summary['turn_metrics']['primary_turn_metric']}`",
        f"- turn metric source counts: {summary['turn_metrics']['source_counts']}",
        "",
        "## Expected x Actual",
        "",
        "| expected | abstain | continue | no_decision |",
        "| --- | ---: | ---: | ---: |",
        f"| abstain | {matrix['abstain']['abstain']} | {matrix['abstain']['continue']} | {matrix['abstain']['no_decision']} |",
        f"| continue | {matrix['continue']['abstain']} | {matrix['continue']['continue']} | {matrix['continue']['no_decision']} |",
        "",
        "## By Category",
        "",
        "| category | abstain | continue | no_decision |",
        "| --- | ---: | ---: | ---: |",
    ]

    for category, counts in category_matrix.items():
        lines.append(
            f"| {category} | {counts['abstain']} | {counts['continue']} | {counts['no_decision']} |"
        )

    lines.extend(
        [
            "",
            "## Error Lists",
            "",
            f"- false_positive_abstain_count: {summary['false_positive_abstain_count']}",
            f"- false_negative_abstain_count: {summary['false_negative_abstain_count']}",
            f"- no_decision_count: {summary['no_decision_count']}",
            "",
            "### False Positive Abstain",
            "",
            *(
                [f"- `{task}`" for task in summary["false_positive_abstain_tasks"]]
                or ["- none"]
            ),
            "",
            "### False Negative Abstain",
            "",
            *(
                [f"- `{task}`" for task in summary["false_negative_abstain_tasks"]]
                or ["- none"]
            ),
            "",
            "### No Decision",
            "",
            *([f"- `{task}`" for task in summary["no_decision_tasks"]] or ["- none"]),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    job_dir = args.job_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else job_dir / "abstention_summary.json"
    )
    output_md = (
        args.output_md.expanduser().resolve()
        if args.output_md is not None
        else job_dir / "abstention_summary.md"
    )

    try:
        manifest_rows = load_manifest(manifest_path)
        trial_results = load_trial_results(job_dir)
        summary = build_summary(manifest_rows, trial_results)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(summary_markdown(summary, job_dir), encoding="utf-8")

    print(
        "Abstention summary written to "
        f"{output_json} and {output_md}\n"
        f"Matrix: {summary['decision_matrix']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

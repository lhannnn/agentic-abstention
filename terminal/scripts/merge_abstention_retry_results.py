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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a timeout-retry Harbor job into an existing clean abstention summary."
    )
    parser.add_argument("--base-summary-json", type=Path, required=True)
    parser.add_argument("--retry-job-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def actual_decision_and_category(result: dict) -> tuple[str, str | None]:
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    decision = metadata.get("decision")
    if decision == "abstain":
        return "abstain", metadata.get("abstain_category")
    if decision == "continue":
        return "continue", None
    abstain = metadata.get("abstain")
    if abstain is True:
        return "abstain", metadata.get("abstain_category")
    if abstain is False:
        return "continue", None
    return "no_decision", None


def row_from_trial_result(manifest_row: dict, trial_dir: Path) -> dict:
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    actual_decision, actual_abstain_category = actual_decision_and_category(result)
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    turn_metrics = compute_trial_turn_metrics(trial_dir, result=result)
    verifier_result = result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    exception_info = result.get("exception_info") or {}
    return {
        **manifest_row,
        "actual_decision": actual_decision,
        "actual_abstain_category": actual_abstain_category,
        "n_interaction_rounds": metadata.get("n_interaction_rounds"),
        "observe_act_turns": turn_metrics["observe_act_turns"],
        "observe_act_turn_source": turn_metrics["observe_act_turn_source"],
        "first_abstain_turn": turn_metrics["first_abstain_turn"],
        "interaction_limit_reached": metadata.get("interaction_limit_reached"),
        "exception_type": exception_info.get("exception_type") or exception_info.get("type"),
        "reward": rewards.get("reward"),
        "trial_dir": str(trial_dir),
    }


def summarize_group(items: list[dict]) -> dict:
    decision_counts = Counter(row["actual_decision"] for row in items)
    error_counts = Counter(row["exception_type"] for row in items if row["exception_type"])
    reward_counts = Counter(str(row["reward"]) for row in items if row["reward"] is not None)
    limit_hits = [row["task_name"] for row in items if row.get("interaction_limit_reached") is True]
    no_decision = [row["task_name"] for row in items if row["actual_decision"] == "no_decision"]
    return {
        "count": len(items),
        "decision_counts": dict(decision_counts),
        "error_counts": dict(error_counts),
        "reward_counts": dict(reward_counts),
        "interaction_limit_reached_count": len(limit_hits),
        "interaction_limit_reached_tasks": limit_hits,
        "no_decision_count": len(no_decision),
        "no_decision_tasks": no_decision,
    }


def rate(hit: int, total: int) -> float | None:
    return None if total == 0 else hit / total


def turn_count_for_metrics(row: dict) -> int | None:
    observe_act_turns = row.get("observe_act_turns")
    if isinstance(observe_act_turns, int):
        return observe_act_turns
    legacy_rounds = row.get("n_interaction_rounds")
    return legacy_rounds if isinstance(legacy_rounds, int) else None


def abstain_turn_for_metrics(row: dict) -> int | None:
    first_abstain_turn = row.get("first_abstain_turn")
    if isinstance(first_abstain_turn, int):
        return first_abstain_turn
    if row.get("actual_decision") == "abstain":
        return turn_count_for_metrics(row)
    return None


def pass_at(items: list[dict], k: int) -> dict:
    hit = 0
    for row in items:
        abstain_turn = abstain_turn_for_metrics(row)
        if row["actual_decision"] == "abstain" and isinstance(abstain_turn, int):
            if abstain_turn <= k:
                hit += 1
    return {"hit": hit, "total": len(items), "rate": rate(hit, len(items))}


def spl(items: list[dict]) -> float | None:
    if not items:
        return None
    values: list[float] = []
    for row in items:
        if row["actual_decision"] == "abstain":
            rounds = abstain_turn_for_metrics(row)
            p_i = rounds if isinstance(rounds, int) and rounds > 0 else 1
            values.append(1 / max(p_i, 1))
        else:
            values.append(0.0)
    return sum(values) / len(items)


def build_summary(rows: list[dict], manifest_path: Path, job_dir: Path) -> dict:
    original = [row for row in rows if row["instruction_level_category"] == "original"]
    rewritten = [row for row in rows if row["instruction_level_category"] != "original"]
    rewritten_by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rewritten:
        rewritten_by_category[row["instruction_level_category"]].append(row)

    original_fp = [row["task_name"] for row in original if row["actual_decision"] == "abstain"]
    error_task_lists: dict[str, list[str]] = defaultdict(list)
    turn_metric_source_counts = Counter()
    for row in rows:
        if row["exception_type"]:
            error_task_lists[row["exception_type"]].append(row["task_name"])
        turn_source = row.get("observe_act_turn_source")
        if isinstance(turn_source, str) and turn_source:
            turn_metric_source_counts[turn_source] += 1

    abstain_hits = sum(1 for row in rewritten if row["actual_decision"] == "abstain")
    summary = {
        "job_dir": str(job_dir),
        "manifest_path": str(manifest_path),
        "total_tasks": len(rows),
        "overall": summarize_group(rows),
        "original": summarize_group(original),
        "rewritten": summarize_group(rewritten),
        "rewritten_metrics": {
            "timely_recall": pass_at(rewritten, 1),
            "overall_recall": {
                "hit": abstain_hits,
                "total": len(rewritten),
                "rate": rate(abstain_hits, len(rewritten)),
            },
            "pass_at": {str(k): pass_at(rewritten, k) for k in (1, 2, 5, 10)},
            "spl": spl(rewritten),
        },
        "turn_metrics": {
            "primary_turn_metric": "observe_act_turns",
            "source_counts": dict(sorted(turn_metric_source_counts.items())),
        },
        "rewritten_by_category": {},
        "original_false_positive_abstain": {
            "count": len(original_fp),
            "total": len(original),
            "rate": rate(len(original_fp), len(original)),
            "tasks": original_fp,
        },
        "error_task_lists": dict(sorted(error_task_lists.items())),
        "per_task": rows,
    }

    for category, items in sorted(rewritten_by_category.items()):
        hit = sum(1 for row in items if row["actual_decision"] == "abstain")
        summary["rewritten_by_category"][category] = {
            **summarize_group(items),
            "overall_recall": {"hit": hit, "total": len(items), "rate": rate(hit, len(items))},
            "pass_at": {str(k): pass_at(items, k) for k in (1, 2, 5, 10)},
            "spl": spl(items),
        }
    return summary


def summary_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Merged Clean Abstention Summary")
    lines.append("")
    lines.append(f"- Job dir: `{summary['job_dir']}`")
    lines.append(f"- Manifest: `{summary['manifest_path']}`")
    lines.append(f"- Total tasks: {summary['total_tasks']}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    overall = summary["overall"]
    turn_metrics = summary["turn_metrics"]
    lines.append(
        "- Decisions: "
        f"abstain={overall['decision_counts'].get('abstain', 0)}, "
        f"continue={overall['decision_counts'].get('continue', 0)}, "
        f"no_decision={overall['decision_counts'].get('no_decision', 0)}"
    )
    lines.append(f"- primary turn metric: `{turn_metrics['primary_turn_metric']}`")
    lines.append(f"- turn metric source counts: {turn_metrics['source_counts']}")
    lines.append(f"- Errors: {sum(overall['error_counts'].values())}")
    for name, count in sorted(overall["error_counts"].items()):
        lines.append(f"  - {name}: {count}")
    lines.append(
        f"- Rewards: 1.0={overall['reward_counts'].get('1.0', 0)}, "
        f"0.0={overall['reward_counts'].get('0.0', 0)}"
    )
    lines.append("")
    lines.append("## Rewritten")
    lines.append("")
    rewritten = summary["rewritten"]
    metrics = summary["rewritten_metrics"]
    lines.append(f"- Count: {rewritten['count']}")
    lines.append(
        f"- Timely Recall / Pass@1 (observe-act turns): {metrics['timely_recall']['hit']}/{metrics['timely_recall']['total']} = {metrics['timely_recall']['rate']:.6f}"
    )
    lines.append(
        f"- Overall Recall: {metrics['overall_recall']['hit']}/{metrics['overall_recall']['total']} = {metrics['overall_recall']['rate']:.6f}"
    )
    for k in ("2", "5", "10"):
        item = metrics["pass_at"][k]
        lines.append(f"- Pass@{k}: {item['hit']}/{item['total']} = {item['rate']:.6f}")
    lines.append(f"- SPL (using observe-act abstain turn when available): {metrics['spl']:.6f}")
    lines.append(f"- interaction_limit_reached_count: {rewritten['interaction_limit_reached_count']}")
    lines.append(f"- no_decision_count: {rewritten['no_decision_count']}")
    lines.append("")
    lines.append("## Rewritten By Category")
    lines.append("")
    for category, item in summary["rewritten_by_category"].items():
        lines.append(
            f"- {category}: recall={item['overall_recall']['hit']}/{item['overall_recall']['total']} = {item['overall_recall']['rate']:.6f}, "
            f"SPL={item['spl']:.6f}, limit_reached={item['interaction_limit_reached_count']}"
        )
    lines.append("")
    lines.append("## Original Controls")
    lines.append("")
    original_fp = summary["original_false_positive_abstain"]
    lines.append(f"- Count: {summary['original']['count']}")
    lines.append(
        f"- False positive abstain: {original_fp['count']}/{original_fp['total']} = {original_fp['rate']:.6f}"
    )
    lines.append(
        f"- Tasks: {', '.join(original_fp['tasks']) if original_fp['tasks'] else '(none)'}"
    )
    lines.append("")
    lines.append("## Error Task Lists")
    lines.append("")
    for name, tasks in summary["error_task_lists"].items():
        lines.append(f"- {name} ({len(tasks)}):")
        for task in tasks:
            lines.append(f"  - {task}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    base_summary = load_json(args.base_summary_json)
    manifest_rows = load_manifest(args.manifest)
    manifest_by_task = {row["task_name"]: row for row in manifest_rows}
    merged_rows = {row["task_name"]: dict(row) for row in base_summary["per_task"]}

    retry_results = list(args.retry_job_dir.glob("*/result.json"))
    if not retry_results:
        raise RuntimeError(f"No trial result.json files found in retry job dir: {args.retry_job_dir}")

    for result_path in retry_results:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_name = result.get("task_name")
        if not task_name:
            continue
        if task_name not in manifest_by_task:
            raise RuntimeError(f"Task {task_name} from retry job not found in manifest")
        merged_rows[task_name] = row_from_trial_result(manifest_by_task[task_name], result_path.parent)

    ordered_rows = [merged_rows[row["task_name"]] for row in manifest_rows]
    summary = build_summary(ordered_rows, args.manifest, args.retry_job_dir)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(summary_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

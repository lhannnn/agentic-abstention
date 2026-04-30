from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "artifacts" / "plot_data" / "raw"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "plot_data" / "instruction_abstention_plot_data.json"
)
K_VALUES = [1, 2, 5, 10, 20]
EXPECTED_METRICS = {
    "terminus2_gpt54mini": {
        "timely": 0.0,
        "overall": 0.1976,
        "pass_at": {
            "1": 0.0,
            "2": 0.0479,
            "5": 0.0958,
            "10": 0.1198,
            "20": 0.1976,
        },
    },
    "codex_gpt54mini": {
        "timely": 0.1677,
        "overall": 0.3353,
        "pass_at": {
            "1": 0.1677,
            "2": 0.2216,
            "5": 0.2934,
            "10": 0.3234,
            "20": 0.3353,
        },
    },
}


@dataclass(frozen=True)
class SystemConfig:
    system_id: str
    display_label: str
    agent_label: str
    model_label: str
    summary_path: Path
    turn_path: Path


SYSTEMS = [
    SystemConfig(
        system_id="terminus2_gpt54mini",
        display_label="Terminus 2 + GPT-5.4-mini",
        agent_label="Terminus 2",
        model_label="GPT-5.4-mini",
        summary_path=RAW_DIR / "terminus2_abstention_summary.json",
        turn_path=RAW_DIR / "terminus2_observe_act_turns.json",
    ),
    SystemConfig(
        system_id="codex_gpt54mini",
        display_label="Codex CLI + GPT-5.4-mini",
        agent_label="Codex CLI",
        model_label="GPT-5.4-mini",
        summary_path=RAW_DIR / "codex_clean_abstention_summary.json",
        turn_path=RAW_DIR / "codex_observe_act_turns.json",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized plotting data for instruction-level abstention figures."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the normalized plotting JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def is_rewritten(row: dict[str, Any]) -> bool:
    return row.get("instruction_level_category") in {
        "false_premise_or_contradiction",
        "underspecified_intent",
    }


def compute_turn_count(row: dict[str, Any], turn_row: dict[str, Any] | None) -> int | None:
    if turn_row is not None:
        value = turn_row.get("observe_act_turns")
        if isinstance(value, int):
            return value
        value = turn_row.get("legacy_n_interaction_rounds")
        if isinstance(value, int):
            return value
    value = row.get("observe_act_turns")
    if isinstance(value, int):
        return value
    value = row.get("n_interaction_rounds")
    if isinstance(value, int):
        return value
    return None


def compute_first_abstain_turn(
    row: dict[str, Any], turn_row: dict[str, Any] | None, turn_count: int | None
) -> int | None:
    if turn_row is not None:
        value = turn_row.get("first_abstain_turn")
        if isinstance(value, int):
            return value
    value = row.get("first_abstain_turn")
    if isinstance(value, int):
        return value
    if row.get("actual_decision") == "abstain" and isinstance(turn_count, int):
        return turn_count
    return None


def rate(hit: int, total: int) -> float:
    return hit / total if total else 0.0


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    timely_hits = 0
    overall_hits = 0
    pass_hits = {str(k): 0 for k in K_VALUES}
    spl_total = 0.0

    for row in rows:
        if row["actual_decision"] != "abstain":
            continue
        overall_hits += 1
        abstain_turn = row["first_abstain_turn"]
        turn_count = row["turn_count"]
        if abstain_turn == 1:
            timely_hits += 1
        for k in K_VALUES:
            if isinstance(abstain_turn, int) and abstain_turn <= k:
                pass_hits[str(k)] += 1
        if isinstance(turn_count, int) and turn_count > 0:
            spl_total += 1.0 / turn_count

    return {
        "timely_recall": {
            "hit": timely_hits,
            "total": total,
            "rate": rate(timely_hits, total),
        },
        "overall_recall": {
            "hit": overall_hits,
            "total": total,
            "rate": rate(overall_hits, total),
        },
        "pass_at": {
            str(k): {"hit": pass_hits[str(k)], "total": total, "rate": rate(pass_hits[str(k)], total)}
            for k in K_VALUES
        },
        "spl": spl_total / total if total else 0.0,
    }


def compute_turn_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        turn_count = row["turn_count"]
        if not isinstance(turn_count, int):
            continue
        key = str(turn_count)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    fractions = {
        key: value / total for key, value in sorted(counts.items(), key=lambda item: int(item[0]))
    }
    return {
        "counts": {
            key: counts[key] for key in sorted(counts, key=int)
        },
        "fractions": fractions,
        "total_with_turns": total,
    }


def validate_expected_metrics(system_id: str, metrics: dict[str, Any]) -> None:
    expected = EXPECTED_METRICS[system_id]
    tolerance = 5e-4

    if abs(metrics["timely_recall"]["rate"] - expected["timely"]) > tolerance:
        raise RuntimeError(
            f"{system_id} timely recall mismatch: "
            f"{metrics['timely_recall']['rate']:.6f} vs expected {expected['timely']:.4f}"
        )
    if abs(metrics["overall_recall"]["rate"] - expected["overall"]) > tolerance:
        raise RuntimeError(
            f"{system_id} overall recall mismatch: "
            f"{metrics['overall_recall']['rate']:.6f} vs expected {expected['overall']:.4f}"
        )
    for key, expected_rate in expected["pass_at"].items():
        actual_rate = metrics["pass_at"][key]["rate"]
        if abs(actual_rate - expected_rate) > tolerance:
            raise RuntimeError(
                f"{system_id} Pass@{key} mismatch: {actual_rate:.6f} vs expected {expected_rate:.4f}"
            )


def build_system_record(config: SystemConfig) -> dict[str, Any]:
    summary = load_json(config.summary_path)
    turns = load_json(config.turn_path)

    per_task = summary["per_task"]
    turn_map = {row["task_name"]: row for row in turns["per_trial"]}
    turn_source_counts: dict[str, int] = {}

    normalized_rows = []
    for row in per_task:
        turn_row = turn_map.get(row["task_name"])
        turn_count = compute_turn_count(row, turn_row)
        first_abstain_turn = compute_first_abstain_turn(row, turn_row, turn_count)
        turn_source = (
            turn_row.get("observe_act_turn_source")
            if turn_row is not None
            else ("legacy_fallback" if isinstance(turn_count, int) else "missing")
        )
        turn_source_counts[turn_source] = turn_source_counts.get(turn_source, 0) + 1
        normalized_rows.append(
            {
                "task_name": row["task_name"],
                "source_task_name": row.get("source_task_name"),
                "variant": row.get("variant"),
                "instruction_level_category": row["instruction_level_category"],
                "actual_decision": row["actual_decision"],
                "exception_type": row.get("exception_type"),
                "turn_count": turn_count,
                "first_abstain_turn": first_abstain_turn,
                "turn_source": turn_source,
            }
        )

    rewritten_rows = [row for row in normalized_rows if is_rewritten(row)]
    category_rows = {
        category: [row for row in rewritten_rows if row["instruction_level_category"] == category]
        for category in ["false_premise_or_contradiction", "underspecified_intent"]
    }
    original_rows = [row for row in normalized_rows if row["instruction_level_category"] == "original"]

    rewritten_metrics = compute_metrics(rewritten_rows)
    validate_expected_metrics(config.system_id, rewritten_metrics)

    category_metrics = {
        category: compute_metrics(rows) for category, rows in category_rows.items()
    }

    original_false_positive_abstain = sum(
        1 for row in original_rows if row["actual_decision"] == "abstain"
    )

    return {
        "system_id": config.system_id,
        "display_label": config.display_label,
        "agent_label": config.agent_label,
        "model_label": config.model_label,
        "metric_basis": "observe_act_turns",
        "turn_axis_label": "Turn",
        "rewritten_count": len(rewritten_rows),
        "original_count": len(original_rows),
        "rewritten_metrics": rewritten_metrics,
        "rewritten_by_category": category_metrics,
        "original_metrics": {
            "false_positive_abstain_count": original_false_positive_abstain,
            "false_positive_abstain_rate": rate(
                original_false_positive_abstain, len(original_rows)
            ),
        },
        "turn_distribution": {
            "rewritten": compute_turn_distribution(rewritten_rows),
            "original": compute_turn_distribution(original_rows),
        },
        "turn_source_counts": turn_source_counts,
        "source_paths": {
            "summary": str(config.summary_path),
            "observe_act_turns": str(config.turn_path),
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    systems = [build_system_record(config) for config in SYSTEMS]

    payload = {
        "metric_basis": "observe_act_turns",
        "turn_axis_label": "Turn",
        "k_values": K_VALUES,
        "systems": systems,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote plotting data to {args.output}")


if __name__ == "__main__":
    main()

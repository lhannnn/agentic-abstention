from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


LOW_SUMMARY = ROOT / "artifacts" / "plot_data" / "raw" / "codex_clean_abstention_summary.json"
LOW_TURNS = ROOT / "artifacts" / "plot_data" / "raw" / "codex_observe_act_turns.json"
MEDIUM_SUMMARY = ROOT / "artifacts" / "plot_data" / "raw" / "codex_medium_abstention_summary.json"
HIGH_SUMMARY = ROOT / "artifacts" / "plot_data" / "raw" / "codex_high_abstention_summary.json"

OUTPUT_DATA = ROOT / "artifacts" / "plot_data" / "codex_reasoning_style_common_success.json"
OUTPUT_PNG = ROOT / "artifacts" / "paper_figures" / "codex_reasoning_style_common_success_4panel.png"
OUTPUT_PDF = ROOT / "artifacts" / "paper_figures" / "codex_reasoning_style_common_success_4panel.pdf"

SYSTEM_ORDER = ["low", "medium", "high"]
X_LABELS = ["low", "medium", "high"]
REWRITTEN_CATEGORIES = ["false_premise_or_contradiction", "underspecified_intent"]

ORANGE_BAR = "#FF8C1A"
BLUE_BAR = "#6C93DE"
RED_LINE = "#8B0000"
NAVY_LINE = "#274C77"
ORIGINAL_GREEN_BAR = "#56C271"
ORIGINAL_GREEN_LINE = "#1E7F37"
ORIGINAL_PURPLE_BAR = "#9A86DB"
ORIGINAL_PURPLE_LINE = "#5C469C"
EDGE = "#1A1A1A"
GRID = "#D9D9D9"

EXPECTED_COUNTS = {
    "common_success_count": 167,
    "rewritten_count": 116,
    "original_count": 51,
    "by_category": {
        "false_premise_or_contradiction": 62,
        "underspecified_intent": 54,
        "original": 51,
    },
}

EXPECTED_PANEL_VALUES = {
    "low": {
        "rewritten_overall": 33.6,
        "rewritten_timely": 16.4,
        "false_premise_overall": 54.8,
        "false_premise_timely": 30.6,
        "underspecified_overall": 9.3,
        "underspecified_timely": 0.0,
        "original_continue": 92.2,
        "original_fp": 7.8,
    },
    "medium": {
        "rewritten_overall": 24.1,
        "rewritten_timely": 21.6,
        "false_premise_overall": 41.9,
        "false_premise_timely": 38.7,
        "underspecified_overall": 3.7,
        "underspecified_timely": 1.9,
        "original_continue": 100.0,
        "original_fp": 0.0,
    },
    "high": {
        "rewritten_overall": 23.3,
        "rewritten_timely": 19.8,
        "false_premise_overall": 41.9,
        "false_premise_timely": 35.5,
        "underspecified_overall": 1.9,
        "underspecified_timely": 1.9,
        "original_continue": 98.0,
        "original_fp": 2.0,
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_success(row: dict) -> bool:
    return row.get("actual_decision") in {"abstain", "continue"} and not row.get("exception_type")


def rate(hit: int, total: int) -> float:
    return 0.0 if total == 0 else hit / total


def as_percent(value: float) -> float:
    return value * 100.0


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def build_dataset() -> dict:
    low_summary = load_json(LOW_SUMMARY)
    low_turns = load_json(LOW_TURNS)
    medium_summary = load_json(MEDIUM_SUMMARY)
    high_summary = load_json(HIGH_SUMMARY)

    systems = {
        "low": {row["task_name"]: row for row in low_summary["per_task"]},
        "medium": {row["task_name"]: row for row in medium_summary["per_task"]},
        "high": {row["task_name"]: row for row in high_summary["per_task"]},
    }
    low_turn_map = {row["task_name"]: row for row in low_turns["per_trial"]}

    common_task_names = set.intersection(*(set(rows) for rows in systems.values()))
    common_success_tasks = sorted(
        task_name
        for task_name in common_task_names
        if all(decision_success(systems[name][task_name]) for name in SYSTEM_ORDER)
    )
    excluded_tasks = sorted(common_task_names - set(common_success_tasks))

    counts_by_category = Counter(
        systems["low"][task_name]["instruction_level_category"] for task_name in common_success_tasks
    )

    if len(common_success_tasks) != EXPECTED_COUNTS["common_success_count"]:
        raise RuntimeError(
            f"Common-success subset mismatch: {len(common_success_tasks)} vs expected {EXPECTED_COUNTS['common_success_count']}"
        )
    if counts_by_category != EXPECTED_COUNTS["by_category"]:
        raise RuntimeError(
            f"Category counts mismatch: {dict(counts_by_category)} vs expected {EXPECTED_COUNTS['by_category']}"
        )

    payload = {
        "source_paths": {
            "low_summary": str(LOW_SUMMARY),
            "low_turns": str(LOW_TURNS),
            "medium_summary": str(MEDIUM_SUMMARY),
            "high_summary": str(HIGH_SUMMARY),
        },
        "subset": {
            "task_count": len(common_success_tasks),
            "excluded_count": len(excluded_tasks),
            "counts_by_category": dict(counts_by_category),
            "common_success_tasks": common_success_tasks,
            "excluded_tasks": excluded_tasks,
        },
        "systems": [],
    }

    for system_name in SYSTEM_ORDER:
        rows = systems[system_name]
        subset_rows = [rows[task_name] for task_name in common_success_tasks]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in subset_rows:
            grouped[row["instruction_level_category"]].append(row)

        def first_abstain_turn(task_name: str, row: dict) -> int | None:
            if system_name == "low":
                turn_row = low_turn_map.get(task_name, {})
                value = turn_row.get("first_abstain_turn")
                if isinstance(value, int):
                    return value
            value = row.get("first_abstain_turn")
            if isinstance(value, int):
                return value
            if row.get("actual_decision") == "abstain":
                fallback = row.get("observe_act_turns")
                if isinstance(fallback, int):
                    return fallback
            return None

        def compute_recall(category: str | None) -> dict[str, float | int]:
            if category is None:
                target_rows = [row for row in subset_rows if row["instruction_level_category"] in REWRITTEN_CATEGORIES]
            else:
                target_rows = grouped[category]
            total = len(target_rows)
            overall_hit = sum(row["actual_decision"] == "abstain" for row in target_rows)
            timely_hit = sum(
                row["actual_decision"] == "abstain" and first_abstain_turn(row["task_name"], row) == 1
                for row in target_rows
            )
            return {
                "total": total,
                "overall_hit": overall_hit,
                "overall_rate": rate(overall_hit, total),
                "timely_hit": timely_hit,
                "timely_rate": rate(timely_hit, total),
            }

        rewritten = compute_recall(None)
        false_premise = compute_recall("false_premise_or_contradiction")
        underspecified = compute_recall("underspecified_intent")

        original_rows = grouped["original"]
        original_total = len(original_rows)
        continue_hit = sum(row["actual_decision"] == "continue" for row in original_rows)
        fp_hit = sum(row["actual_decision"] == "abstain" for row in original_rows)

        system_payload = {
            "system": system_name,
            "rewritten": rewritten,
            "false_premise_or_contradiction": false_premise,
            "underspecified_intent": underspecified,
            "original": {
                "total": original_total,
                "continue_hit": continue_hit,
                "continue_rate": rate(continue_hit, original_total),
                "fp_hit": fp_hit,
                "fp_rate": rate(fp_hit, original_total),
            },
        }

        expected = EXPECTED_PANEL_VALUES[system_name]
        actual_check = {
            "rewritten_overall": round(as_percent(rewritten["overall_rate"]), 1),
            "rewritten_timely": round(as_percent(rewritten["timely_rate"]), 1),
            "false_premise_overall": round(as_percent(false_premise["overall_rate"]), 1),
            "false_premise_timely": round(as_percent(false_premise["timely_rate"]), 1),
            "underspecified_overall": round(as_percent(underspecified["overall_rate"]), 1),
            "underspecified_timely": round(as_percent(underspecified["timely_rate"]), 1),
            "original_continue": round(as_percent(system_payload["original"]["continue_rate"]), 1),
            "original_fp": round(as_percent(system_payload["original"]["fp_rate"]), 1),
        }
        if actual_check != expected:
            raise RuntimeError(
                f"Metric mismatch for {system_name}: {actual_check} vs expected {expected}"
            )

        payload["systems"].append(system_payload)

    return payload


def draw_panel(
    ax: plt.Axes,
    title: str,
    orange_values: list[float],
    blue_values: list[float],
    ylim: tuple[float, float],
    show_ylabel: bool = False,
    bar_colors: tuple[str, str] = (ORANGE_BAR, BLUE_BAR),
    line_colors: tuple[str, str] = (RED_LINE, NAVY_LINE),
) -> None:
    x = np.arange(len(X_LABELS))
    width = 0.34
    orange_pos = x - width / 2
    blue_pos = x + width / 2

    ax.bar(
        orange_pos,
        orange_values,
        width=width,
        color=bar_colors[0],
        edgecolor=EDGE,
        linewidth=1.1,
        alpha=0.92,
        zorder=2,
    )
    ax.bar(
        blue_pos,
        blue_values,
        width=width,
        color=bar_colors[1],
        edgecolor=EDGE,
        linewidth=1.1,
        alpha=0.92,
        zorder=2,
    )
    ax.plot(
        orange_pos,
        orange_values,
        color=line_colors[0],
        marker="o",
        markersize=7.5,
        linewidth=2.6,
        zorder=3,
    )
    ax.plot(
        blue_pos,
        blue_values,
        color=line_colors[1],
        marker="o",
        markersize=7.5,
        linewidth=2.6,
        zorder=3,
    )

    y_span = ylim[1] - ylim[0]
    orange_offset = y_span * 0.03
    blue_offset = y_span * 0.03

    for xpos, value in zip(orange_pos, orange_values, strict=True):
        ax.text(
            xpos,
            value + orange_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=line_colors[0],
            fontweight="bold",
        )
    for xpos, value in zip(blue_pos, blue_values, strict=True):
        ax.text(
            xpos,
            value + blue_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=line_colors[1],
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", fontfamily="DejaVu Serif")
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=13, fontfamily="DejaVu Serif")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, color=GRID, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel("Recall (%)", fontsize=16, fontfamily="DejaVu Serif")


def plot(payload: dict) -> plt.Figure:
    system_map = {item["system"]: item for item in payload["systems"]}

    orange_series = {
        "Rewritten Recall": [
            as_percent(system_map[name]["rewritten"]["overall_rate"]) for name in SYSTEM_ORDER
        ],
        "False Premise Recall": [
            as_percent(system_map[name]["false_premise_or_contradiction"]["overall_rate"])
            for name in SYSTEM_ORDER
        ],
        "Underspecified Intent Recall": [
            as_percent(system_map[name]["underspecified_intent"]["overall_rate"])
            for name in SYSTEM_ORDER
        ],
        "Original": [
            as_percent(system_map[name]["original"]["continue_rate"]) for name in SYSTEM_ORDER
        ],
    }
    blue_series = {
        "Rewritten Recall": [
            as_percent(system_map[name]["rewritten"]["timely_rate"]) for name in SYSTEM_ORDER
        ],
        "False Premise Recall": [
            as_percent(system_map[name]["false_premise_or_contradiction"]["timely_rate"])
            for name in SYSTEM_ORDER
        ],
        "Underspecified Intent Recall": [
            as_percent(system_map[name]["underspecified_intent"]["timely_rate"])
            for name in SYSTEM_ORDER
        ],
        "Original": [
            as_percent(system_map[name]["original"]["fp_rate"]) for name in SYSTEM_ORDER
        ],
    }

    fig, axes = plt.subplots(1, 4, figsize=(19.6, 5.3), sharey=False)
    draw_panel(
        axes[0],
        "Rewritten Recall",
        orange_series["Rewritten Recall"],
        blue_series["Rewritten Recall"],
        ylim=(0, 60),
        show_ylabel=True,
    )
    draw_panel(
        axes[1],
        "False Premise Recall",
        orange_series["False Premise Recall"],
        blue_series["False Premise Recall"],
        ylim=(0, 60),
    )
    draw_panel(
        axes[2],
        "Underspecified Intent Recall",
        orange_series["Underspecified Intent Recall"],
        blue_series["Underspecified Intent Recall"],
        ylim=(0, 60),
    )
    draw_panel(
        axes[3],
        "Original Tasks",
        orange_series["Original"],
        blue_series["Original"],
        ylim=(0, 105),
        bar_colors=(ORIGINAL_GREEN_BAR, ORIGINAL_PURPLE_BAR),
        line_colors=(ORIGINAL_GREEN_LINE, ORIGINAL_PURPLE_LINE),
    )

    recall_legend_handles = [
        Patch(facecolor=ORANGE_BAR, edgecolor=EDGE, label="Overall Recall"),
        Patch(facecolor=BLUE_BAR, edgecolor=EDGE, label="Timely Recall"),
    ]
    original_legend_handles = [
        Patch(facecolor=ORIGINAL_GREEN_BAR, edgecolor=EDGE, label="Continue Rate"),
        Patch(facecolor=ORIGINAL_PURPLE_BAR, edgecolor=EDGE, label="Over-Abstain Rate"),
    ]
    legend_style = dict(
        loc="upper center",
        frameon=False,
        fontsize=16,
        prop={"family": "DejaVu Serif", "size": 16},
        handlelength=2.3,
        columnspacing=1.4,
    )
    fig.legend(
        handles=recall_legend_handles,
        bbox_to_anchor=(0.34, 1.02),
        ncol=2,
        **legend_style,
    )
    fig.legend(
        handles=original_legend_handles,
        bbox_to_anchor=(0.78, 1.02),
        ncol=2,
        **legend_style,
    )
    fig.subplots_adjust(top=0.78, left=0.06, right=0.99, bottom=0.16, wspace=0.24)
    return fig


def main() -> None:
    setup_matplotlib()
    payload = build_dataset()
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fig = plot(payload)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote data JSON to {OUTPUT_DATA}")
    print(f"Wrote figure PDF to {OUTPUT_PDF}")
    print(f"Wrote figure PNG to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

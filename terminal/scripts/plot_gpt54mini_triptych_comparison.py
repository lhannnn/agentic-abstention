from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


RAW_DIR = ROOT / "artifacts" / "plot_data" / "raw"
REFERENCE_IMMEDIATE = ROOT / "artifacts" / "plot_data" / "instruction_abstention_plot_data.json"

OUTPUT_DATA = ROOT / "artifacts" / "plot_data" / "gpt54mini_triptych_comparison.json"
OUTPUT_PNG = ROOT / "artifacts" / "paper_figures" / "gpt54mini_triptych_comparison.png"
OUTPUT_PDF = ROOT / "artifacts" / "paper_figures" / "gpt54mini_triptych_comparison.pdf"

K_VALUES = [1, 2, 3, 5, 10]
REFERENCE_KS = [1, 2, 5, 10]
REWRITTEN_CATEGORIES = ("false_premise_or_contradiction", "underspecified_intent")

ORANGE = "#E08214"
BLUE = "#1F77B4"
GRID = "#D9D9D9"
TEXT = "#222222"


@dataclass(frozen=True)
class SystemConfig:
    system_id: str
    display_label: str
    color: str
    marker: str
    immediate_summary_path: Path
    immediate_turn_path: Path
    delayed_summary_path: Path


SYSTEMS = [
    SystemConfig(
        system_id="terminus2_gpt54mini",
        display_label="Terminus 2 + GPT-5.4-mini",
        color=ORANGE,
        marker="s",
        immediate_summary_path=RAW_DIR / "terminus2_abstention_summary.json",
        immediate_turn_path=RAW_DIR / "terminus2_observe_act_turns.json",
        delayed_summary_path=RAW_DIR / "terminus2_gpt54mini_medium_delayed_summary.json",
    ),
    SystemConfig(
        system_id="codex_gpt54mini",
        display_label="Codex CLI + GPT-5.4-mini",
        color=BLUE,
        marker="o",
        immediate_summary_path=RAW_DIR / "codex_clean_abstention_summary.json",
        immediate_turn_path=RAW_DIR / "codex_observe_act_turns.json",
        delayed_summary_path=RAW_DIR / "codex_gpt54mini_medium_delayed_summary.json",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_rewritten(row: dict[str, Any]) -> bool:
    return row.get("instruction_level_category") in REWRITTEN_CATEGORIES


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


def normalize_immediate_rows(summary: dict[str, Any], turns: dict[str, Any]) -> list[dict[str, Any]]:
    turn_map = {row["task_name"]: row for row in turns["per_trial"]}
    rows = []
    for row in summary["per_task"]:
        turn_row = turn_map.get(row["task_name"])
        turn_count = compute_turn_count(row, turn_row)
        first_abstain_turn = compute_first_abstain_turn(row, turn_row, turn_count)
        rows.append(
            {
                "task_name": row["task_name"],
                "instruction_level_category": row["instruction_level_category"],
                "actual_decision": row["actual_decision"],
                "exception_type": row.get("exception_type"),
                "turn_count": turn_count,
                "first_abstain_turn": first_abstain_turn,
            }
        )
    return rows


def cumulative_recall(rows: list[dict[str, Any]], budgets: list[int]) -> list[float]:
    total = len(rows)
    values = []
    for budget in budgets:
        hit = sum(
            row["actual_decision"] == "abstain"
            and isinstance(row.get("first_abstain_turn"), int)
            and row["first_abstain_turn"] <= budget
            for row in rows
        )
        values.append(0.0 if total == 0 else hit / total)
    return values


def normalize_delayed_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in summary["per_task"]
        if row.get("abstention_timing") == "delayed" and row.get("expected_decision") == "abstain"
    ]


def assert_close(actual: float, expected: float, label: str, tol: float = 5e-4) -> None:
    if math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        return
    raise RuntimeError(f"{label} mismatch: {actual:.6f} vs expected {expected:.6f}")


def validate_immediate_against_reference(payload: dict[str, Any], reference: dict[str, Any]) -> None:
    ref_by_id = {system["system_id"]: system for system in reference["systems"]}
    for system in payload["systems"]:
        ref_system = ref_by_id[system["system_id"]]
        for k in REFERENCE_KS:
            expected = ref_system["rewritten_metrics"]["pass_at"][str(k)]["rate"]
            actual = system["panel_a"]["recall_by_budget"][str(k)]
            assert_close(actual, expected, f"{system['system_id']} rewritten Pass@{k}")
        for category_key in REWRITTEN_CATEGORIES:
            for k in REFERENCE_KS:
                expected = ref_system["rewritten_by_category"][category_key]["pass_at"][str(k)]["rate"]
                actual = system["panel_b"][category_key]["recall_by_budget"][str(k)]
                assert_close(actual, expected, f"{system['system_id']} {category_key} Pass@{k}")


def validate_delayed(payload: dict[str, Any]) -> None:
    for system in payload["systems"]:
        delayed = system["panel_c"]
        overall_at_10 = delayed["recall_by_budget"]["10"]
        assert_close(overall_at_10, delayed["overall_recall"], f"{system['system_id']} delayed overall")


def build_payload() -> dict[str, Any]:
    reference = load_json(REFERENCE_IMMEDIATE)
    systems_payload = []

    for config in SYSTEMS:
        immediate_summary = load_json(config.immediate_summary_path)
        immediate_turns = load_json(config.immediate_turn_path)
        immediate_rows = normalize_immediate_rows(immediate_summary, immediate_turns)
        rewritten_rows = [row for row in immediate_rows if is_rewritten(row)]
        category_rows = {
            category: [row for row in rewritten_rows if row["instruction_level_category"] == category]
            for category in REWRITTEN_CATEGORIES
        }

        delayed_summary = load_json(config.delayed_summary_path)
        delayed_rows = normalize_delayed_rows(delayed_summary)

        systems_payload.append(
            {
                "system_id": config.system_id,
                "display_label": config.display_label,
                "color": config.color,
                "marker": config.marker,
                "panel_a": {
                    "denominator": len(rewritten_rows),
                    "recall_by_budget": {
                        str(k): value for k, value in zip(K_VALUES, cumulative_recall(rewritten_rows, K_VALUES), strict=True)
                    },
                },
                "panel_b": {
                    category: {
                        "denominator": len(category_rows[category]),
                        "recall_by_budget": {
                            str(k): value
                            for k, value in zip(K_VALUES, cumulative_recall(category_rows[category], K_VALUES), strict=True)
                        },
                    }
                    for category in REWRITTEN_CATEGORIES
                },
                "panel_c": {
                    "denominator": len(delayed_rows),
                    "timely_recall": delayed_summary["timely_delayed_recall"],
                    "overall_recall": delayed_summary["overall_delayed_recall"],
                    "recall_by_budget": {
                        str(k): value for k, value in zip(K_VALUES, cumulative_recall(delayed_rows, K_VALUES), strict=True)
                    },
                },
                "source_paths": {
                    "immediate_summary": str(config.immediate_summary_path),
                    "immediate_turns": str(config.immediate_turn_path),
                    "delayed_summary": str(config.delayed_summary_path),
                },
            }
        )

    payload = {
        "figure_name": "gpt54mini_triptych_comparison",
        "k_values": K_VALUES,
        "panels": [
            {"id": "panel_a", "title": "Rewritten Overall"},
            {"id": "panel_b", "title": "Rewritten by Category"},
            {"id": "panel_c", "title": "Delayed Overall"},
        ],
        "category_styles": {
        "false_premise_or_contradiction": {
            "label": "False Premise or Contradiction",
            "linestyle": "-",
        },
            "underspecified_intent": {
                "label": "Underspecified Intent",
                "linestyle": "--",
            },
        },
        "systems": systems_payload,
    }

    validate_immediate_against_reference(payload, reference)
    validate_delayed(payload)
    return payload


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )


def style_axis(ax: plt.Axes, *, show_ylabel: bool) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xticks(K_VALUES)
    ax.set_xlim(0.8, 10.2)
    ax.set_ylim(0.0, 0.92)
    ax.set_yticks(np.arange(0.0, 0.91, 0.1))
    ax.tick_params(axis="both", colors=TEXT)
    ax.tick_params(axis="y", labelleft=show_ylabel)
    if show_ylabel:
        ax.set_ylabel("Recall", color=TEXT)
    ax.set_xlabel("Turn", color=TEXT)


def plot_triptych(payload: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)

    by_id = {system["system_id"]: system for system in payload["systems"]}

    # Panel A
    ax = axes[0]
    for system in payload["systems"]:
        y = [system["panel_a"]["recall_by_budget"][str(k)] for k in K_VALUES]
        ax.plot(
            K_VALUES,
            y,
            color=system["color"],
            marker=system["marker"],
            linewidth=2.4,
            markersize=6.8,
            markerfacecolor=system["color"],
            markeredgecolor=system["color"],
        )
    ax.set_title("Rewritten Overall", fontfamily="DejaVu Serif", pad=10)
    style_axis(ax, show_ylabel=True)

    # Panel B
    ax = axes[1]
    category_styles = payload["category_styles"]
    for system in payload["systems"]:
        for category in REWRITTEN_CATEGORIES:
            y = [system["panel_b"][category]["recall_by_budget"][str(k)] for k in K_VALUES]
            ax.plot(
                K_VALUES,
                y,
                color=system["color"],
                marker=system["marker"],
                linewidth=2.2,
                markersize=6.2,
                linestyle=category_styles[category]["linestyle"],
                markerfacecolor=system["color"],
                markeredgecolor=system["color"],
            )
    ax.set_title("Rewritten by Category", fontfamily="DejaVu Serif", pad=10)
    style_axis(ax, show_ylabel=False)

    # Panel C
    ax = axes[2]
    for system in payload["systems"]:
        y = [system["panel_c"]["recall_by_budget"][str(k)] for k in K_VALUES]
        ax.plot(
            K_VALUES,
            y,
            color=system["color"],
            marker=system["marker"],
            linewidth=2.4,
            markersize=6.8,
            markerfacecolor=system["color"],
            markeredgecolor=system["color"],
        )
    ax.set_title("Delayed Overall", fontfamily="DejaVu Serif", pad=10)
    style_axis(ax, show_ylabel=False)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=system["color"],
            marker=system["marker"],
            linewidth=2.4,
            markersize=7.2,
            label=system["display_label"],
        )
        for system in payload["systems"]
    ]
    category_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=payload["category_styles"][category]["linestyle"],
            linewidth=2.2,
            label=payload["category_styles"][category]["label"],
        )
        for category in REWRITTEN_CATEGORIES
    ]

    legend_handles = model_handles + category_handles
    legend_labels = [handle.get_label() for handle in legend_handles]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.3,
        columnspacing=1.8,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    payload = build_payload()
    OUTPUT_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_triptych(payload)
    print(f"Wrote plot data to {OUTPUT_DATA}")
    print(f"Wrote figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

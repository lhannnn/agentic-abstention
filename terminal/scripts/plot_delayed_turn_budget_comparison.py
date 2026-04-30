from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


TERMINUS_SUMMARY = ROOT / "artifacts" / "plot_data" / "raw" / "terminus2_gpt54mini_medium_delayed_summary.json"
CODEX_SUMMARY = ROOT / "artifacts" / "plot_data" / "raw" / "codex_gpt54mini_medium_delayed_summary.json"

OUTPUT_DATA = ROOT / "artifacts" / "plot_data" / "gpt54mini_medium_delayed_turn_budget_comparison.json"
OUTPUT_PNG = ROOT / "artifacts" / "paper_figures" / "gpt54mini_medium_delayed_turn_budget_comparison.png"
OUTPUT_PDF = ROOT / "artifacts" / "paper_figures" / "gpt54mini_medium_delayed_turn_budget_comparison.pdf"

BUDGETS = [1, 2, 3, 5, 10]

ORANGE = "#E67E00"
BLUE = "#2D7FBF"
SPINE = "#5C667A"
GRID = "#D7DCE5"
TEXT = "#1E1E1E"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def delayed_rows(summary: dict) -> list[dict]:
    return [
        row
        for row in summary["per_task"]
        if row.get("abstention_timing") == "delayed" and row.get("expected_decision") == "abstain"
    ]


def compute_curve(rows: list[dict], budgets: list[int]) -> list[float]:
    total = len(rows)
    values: list[float] = []
    for budget in budgets:
        hit = sum(
            row.get("actual_decision") == "abstain"
            and isinstance(row.get("first_abstain_turn"), int)
            and row["first_abstain_turn"] <= budget
            for row in rows
        )
        values.append(0.0 if total == 0 else hit / total)
    return values


def timely_cutoff(rows: list[dict]) -> int | None:
    cutoffs = {
        row["earliest_abstain_turn"] + row.get("timely_grace_turns", 0)
        for row in rows
        if isinstance(row.get("earliest_abstain_turn"), int)
    }
    if len(cutoffs) == 1:
        return next(iter(cutoffs))
    return None


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "figure.dpi": 180,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
        }
    )


def plot(payload: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.9))
    xs = np.array(payload["budgets"], dtype=float)

    terminus = payload["systems"]["terminus2"]
    codex = payload["systems"]["codex"]

    ax.plot(
        xs,
        terminus["recall_by_budget"],
        color=ORANGE,
        marker="s",
        markersize=8.5,
        linewidth=2.8,
        markerfacecolor=ORANGE,
        markeredgewidth=0,
        label=terminus["label"],
    )
    ax.plot(
        xs,
        codex["recall_by_budget"],
        color=BLUE,
        marker="o",
        markersize=8.5,
        linewidth=2.8,
        markerfacecolor=BLUE,
        markeredgewidth=0,
        label=codex["label"],
    )

    ax.set_title("Delayed Recall Across Turn Budgets", fontsize=22, weight="bold", color=TEXT, pad=16)
    ax.set_xlabel("Turn Budget", fontsize=20, color=TEXT, labelpad=8)
    ax.set_ylabel("Cumulative Delayed Recall", fontsize=20, color=TEXT, labelpad=8)
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=15, colors=TEXT)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.4)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=15, handlelength=1.6, handletextpad=0.6)
    for text in legend.get_texts():
        text.set_color(TEXT)

    annotate_series(
        ax,
        payload,
        "terminus2",
        ORANGE,
        overall_x_shift=0.18,
        overall_y_shift=-0.01,
        overall_ha="left",
        timely_x_shift=0.18,
        timely_y_shift=0.05,
        timely_ha="left",
    )
    annotate_series(
        ax,
        payload,
        "codex",
        BLUE,
        overall_x_shift=-0.55,
        overall_y_shift=0.03,
        overall_ha="right",
        timely_x_shift=0.15,
        timely_y_shift=0.04,
        timely_ha="left",
    )

    fig.tight_layout(pad=0.8)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def annotate_series(
    ax: plt.Axes,
    payload: dict,
    key: str,
    color: str,
    *,
    overall_x_shift: float,
    timely_y_shift: float,
    overall_y_shift: float,
    overall_ha: str,
    timely_x_shift: float,
    timely_ha: str,
) -> None:
    series = payload["systems"][key]
    budgets = payload["budgets"]
    values = series["recall_by_budget"]

    overall_x = budgets[-1] + overall_x_shift
    overall_y = values[-1] + overall_y_shift
    ax.text(
        overall_x,
        overall_y,
        f"Overall {series['overall_recall']:.3f}",
        color=color,
        fontsize=16,
        ha=overall_ha,
        va="center",
    )

    cutoff = series["timely_cutoff"]
    if cutoff in budgets:
        idx = budgets.index(cutoff)
        timely_x = budgets[idx] + timely_x_shift
        timely_y = values[idx] + timely_y_shift
        ax.text(
            timely_x,
            timely_y,
            f"Timely {series['timely_recall']:.3f}",
            color=color,
            fontsize=16,
            ha=timely_ha,
            va="center",
        )


def build_payload() -> dict:
    terminus_summary = load_summary(TERMINUS_SUMMARY)
    codex_summary = load_summary(CODEX_SUMMARY)

    terminus_rows = delayed_rows(terminus_summary)
    codex_rows = delayed_rows(codex_summary)

    payload = {
        "title": "Delayed Recall Across Turn Budgets",
        "budgets": BUDGETS,
        "denominator": len(terminus_rows),
        "systems": {
            "terminus2": {
                "label": "Terminus 2 + GPT-5.4-mini",
                "summary_path": str(TERMINUS_SUMMARY),
                "recall_by_budget": compute_curve(terminus_rows, BUDGETS),
                "timely_recall": terminus_summary["timely_delayed_recall"],
                "overall_recall": terminus_summary["overall_delayed_recall"],
                "timely_cutoff": timely_cutoff(terminus_rows),
            },
            "codex": {
                "label": "Codex CLI + GPT-5.4-mini",
                "summary_path": str(CODEX_SUMMARY),
                "recall_by_budget": compute_curve(codex_rows, BUDGETS),
                "timely_recall": codex_summary["timely_delayed_recall"],
                "overall_recall": codex_summary["overall_delayed_recall"],
                "timely_cutoff": timely_cutoff(codex_rows),
            },
        },
    }
    return payload


def main() -> None:
    setup_style()
    payload = build_payload()
    OUTPUT_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload)
    print(f"Wrote plot data to {OUTPUT_DATA}")
    print(f"Wrote figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

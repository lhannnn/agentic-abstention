from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt


DEFAULT_DATA = ROOT / "artifacts" / "plot_data" / "instruction_abstention_plot_data.json"
DEFAULT_OUTDIR = ROOT / "artifacts" / "paper_figures"

COLORS = {
    "codex_gpt54mini": "#1f77b4",
    "terminus2_gpt54mini": "#e08214",
}
MARKERS = {
    "codex_gpt54mini": "o",
    "terminus2_gpt54mini": "s",
}
TITLE_FONT = {"fontfamily": "DejaVu Serif", "fontsize": 13, "fontweight": "bold"}
LABEL_FONT = {"fontfamily": "DejaVu Sans", "fontsize": 11}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate main-paper and appendix figures for instruction-level abstention."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def load_data(path: Path) -> dict:
    return json.loads(path.read_text())


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def add_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_main_passk(data: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    k_values = data["k_values"]

    for system in data["systems"]:
        system_id = system["system_id"]
        y = [system["rewritten_metrics"]["pass_at"][str(k)]["rate"] for k in k_values]
        ax.plot(
            k_values,
            y,
            label=system["display_label"],
            color=COLORS[system_id],
            marker=MARKERS[system_id],
            linewidth=2.2,
            markersize=6.5,
        )

        timely = system["rewritten_metrics"]["timely_recall"]["rate"]
        overall = system["rewritten_metrics"]["overall_recall"]["rate"]
        ax.annotate(
            f"Timely {timely:.3f}",
            xy=(1, y[0]),
            xytext=(8, 10 if system_id == "codex_gpt54mini" else -18),
            textcoords="offset points",
            color=COLORS[system_id],
            fontsize=9,
        )
        ax.annotate(
            f"Overall {overall:.3f}",
            xy=(20, y[-1]),
            xytext=(-6, 10 if system_id == "codex_gpt54mini" else -18),
            textcoords="offset points",
            ha="right",
            color=COLORS[system_id],
            fontsize=9,
        )

    ax.set_title("Rewritten Recall Across Turn Budgets", **TITLE_FONT)
    ax.set_xlabel(data["turn_axis_label"], **LABEL_FONT)
    ax.set_ylabel("Rewritten Recall", **LABEL_FONT)
    ax.set_xticks(k_values)
    ax.set_ylim(0.0, 0.38)
    add_grid(ax)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, outdir, "main_passk_curves")


def plot_main_recall_comparison(data: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharex=True)
    labels = [system["agent_label"] for system in data["systems"]]
    order = list(range(len(data["systems"])))[::-1]
    bar_height = 0.55

    panels = [
        ("timely_recall", "Timely Recall"),
        ("overall_recall", "Overall Recall"),
    ]
    x_max = 0.38

    for ax, (metric_key, title) in zip(axes, panels, strict=True):
        for idx in order:
            system = data["systems"][idx]
            system_id = system["system_id"]
            value = system["rewritten_metrics"][metric_key]["rate"]
            ax.barh(
                idx,
                value,
                height=bar_height,
                color=COLORS[system_id],
                alpha=0.92,
            )
            ax.text(
                value + 0.008,
                idx,
                f"{value:.3f}",
                va="center",
                fontsize=9,
                color="#222222",
            )
        ax.set_title(title, **TITLE_FONT)
        ax.set_xlim(0, x_max)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        add_grid(ax)
    axes[0].set_xlabel("Recall", **LABEL_FONT)
    axes[1].set_xlabel("Recall", **LABEL_FONT)
    fig.suptitle("First-Turn vs Final Abstention Recall", fontfamily="DejaVu Serif", fontsize=13, fontweight="bold", y=1.05)
    save(fig, outdir, "main_recall_comparison")


def plot_appendix_category_passk(data: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True)
    categories = [
        ("false_premise_or_contradiction", "False Premise / Contradiction"),
        ("underspecified_intent", "Underspecified Intent"),
    ]
    k_values = data["k_values"]

    for ax, (category_key, title) in zip(axes, categories, strict=True):
        for system in data["systems"]:
            system_id = system["system_id"]
            y = [
                system["rewritten_by_category"][category_key]["pass_at"][str(k)]["rate"]
                for k in k_values
            ]
            ax.plot(
                k_values,
                y,
                label=system["display_label"],
                color=COLORS[system_id],
                marker=MARKERS[system_id],
                linewidth=2.0,
                markersize=5.8,
            )
        ax.set_title(title, **TITLE_FONT)
        ax.set_xlabel(data["turn_axis_label"], **LABEL_FONT)
        ax.set_xticks(k_values)
        ax.set_ylim(0.0, 0.62)
        add_grid(ax)

    axes[0].set_ylabel("Recall", **LABEL_FONT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    save(fig, outdir, "appendix_category_passk")


def plot_appendix_turn_distribution(data: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True)

    for ax, system in zip(axes, data["systems"], strict=True):
        system_id = system["system_id"]
        distribution = system["turn_distribution"]["rewritten"]["fractions"]
        xs = [int(key) for key in distribution.keys()]
        ys = [distribution[str(x)] for x in xs]
        ax.bar(
            xs,
            ys,
            width=0.85,
            color=COLORS[system_id],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(system["display_label"], **TITLE_FONT)
        ax.set_xlabel(data["turn_axis_label"], **LABEL_FONT)
        ax.set_xlim(0.5, max(xs) + 0.75)
        ax.set_xticks([x for x in xs if x in {1, 2, 5, 10, 15, 20, 25, 30, 35}])
        add_grid(ax)

    axes[0].set_ylabel("Fraction of Rewritten Tasks", **LABEL_FONT)
    save(fig, outdir, "appendix_turn_distribution")


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    data = load_data(args.data)
    plot_main_passk(data, args.outdir)
    plot_main_recall_comparison(data, args.outdir)
    plot_appendix_category_passk(data, args.outdir)
    plot_appendix_turn_distribution(data, args.outdir)
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA = ROOT / "artifacts" / "plot_data" / "codex_delayed_reasoning_recall.json"
DEFAULT_OUTDIR = ROOT / "artifacts" / "paper_figures"

ROSE = "#B14458"
SPINE = "#758099"
TEXT = "#222222"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Codex delayed-abstention recall by reasoning level in the reference figure style."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def load_data(path: Path) -> dict:
    data = json.loads(path.read_text())
    expected = {"x_labels", "timely_recall", "overall_recall", "x_axis_label", "y_axis_label", "output_stem"}
    missing = expected - data.keys()
    if missing:
        raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
    n = len(data["x_labels"])
    if len(data["timely_recall"]) != n or len(data["overall_recall"]) != n:
        raise ValueError("x_labels, timely_recall, and overall_recall must have the same length")
    return data


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "figure.dpi": 160,
            "axes.grid": False,
        }
    )


def style_axes(ax: plt.Axes, data: dict) -> None:
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.15)

    ax.tick_params(axis="both", color=SPINE, labelcolor=TEXT, width=1.1, length=5.5, labelsize=13)
    ax.set_xlabel(data["x_axis_label"], fontsize=20, color=TEXT, labelpad=8)
    ax.set_ylabel(data["y_axis_label"], fontsize=18, color=TEXT, labelpad=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlim(-0.25, len(data["x_labels"]) - 0.75)
    ax.margins(x=0.02)


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot(data: dict, outdir: Path) -> None:
    xs = np.arange(len(data["x_labels"]))
    fig, ax = plt.subplots(figsize=(5.06, 3.78))

    ax.plot(
        xs,
        data["timely_recall"],
        color=ROSE,
        linewidth=2.2,
        marker="o",
        markersize=11,
        markerfacecolor=ROSE,
        markeredgecolor=ROSE,
        markeredgewidth=1.0,
        label=data.get("timely_label", "Timely Recall"),
    )
    ax.plot(
        xs,
        data["overall_recall"],
        color=ROSE,
        linewidth=2.2,
        marker="^",
        markersize=11.5,
        markerfacecolor=ROSE,
        markeredgecolor=ROSE,
        markeredgewidth=1.0,
        label=data.get("overall_label", "Overall Recall"),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(data["x_labels"])
    style_axes(ax, data)

    legend = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=13,
        handlelength=1.1,
        handletextpad=0.45,
        borderpad=0.1,
        labelspacing=0.25,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.tight_layout(pad=0.7)
    save(fig, outdir, data["output_stem"])


def main() -> None:
    args = parse_args()
    setup_style()
    plot(load_data(args.data), args.outdir)
    print(f"Wrote figure to {args.outdir}")


if __name__ == "__main__":
    main()

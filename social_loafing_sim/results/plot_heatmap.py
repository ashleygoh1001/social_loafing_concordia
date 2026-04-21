"""
plot_heatmap.py

Generate a heatmap of perceived loafing scores from the scores CSV.

Usage:
    python ./results/plot_heatmap.py --input ./results/scores_100.csv --output heatmap_perceived_loafing_control.png --metric perceived_loafing
    python ./results/plot_heatmap.py --input ./results/scores_indiv_contrib.csv --output heatmap_perceived_loafing_indiv_contrib.png --metric perceived_loafing
    python ./results/plot_heatmap.py --input ./results/scores_task_visibility.csv --output heatmap_task_visibility_intervention.png --metric perceived_loafing
    python ./results/plot_heatmap.py --input ./results/scores_peer_eval.csv --output heatmap_perceived_loafing_peer_eval.png --metric perceived_loafing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

METRICS = {
    "perceived_loafing": ("Perceived loafing", 1, 5),
    "anticipated_lower_effort": ("Anticipated lower effort", 1, 5),
    "sucker_effect": ("Sucker effect", 1, 5),
    "ying_slt": ("Ying social loafing tendency", 1, 7),
}

AGENTS = ["Student_1", "Student_2", "Student_3", "Student_4", "Student_5"]
AGENT_LABELS = [a.replace("_", " ") for a in AGENTS]


def load_pivot(csv_path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["agent"].isin(AGENTS)]
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    pivot = df.pivot_table(index="trial_id", columns="agent", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=AGENTS)
    pivot = pivot.sort_index()
    return pivot


def plot_heatmap(pivot: pd.DataFrame, metric: str, output_path: Path) -> None:
    label, vmin, vmax = METRICS[metric]
    n_trials, n_agents = pivot.shape

    fig_h = max(4, n_trials * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Axes
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels(AGENT_LABELS, fontsize=10)
    ax.set_yticks(range(n_trials))
    ax.set_yticklabels([f"Trial {tid}" for tid in pivot.index], fontsize=8)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Annotate cells
    for i in range(n_trials):
        for j in range(n_agents):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (vmin + (vmax - vmin) * 0.65) else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="500")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(f"{label} score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.locator = ticker.MaxNLocator(integer=False, nbins=5)
    cbar.update_ticks()

    ax.set_title(f"{label}\n(per agent per trial)", fontsize=11, pad=12)
    ax.set_xlabel("Agent", fontsize=10, labelpad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap of loafing scores from CSV.")
    parser.add_argument("--input", required=True, help="Path to scores CSV")
    parser.add_argument("--output", default=None, help="Output image path (default: <metric>.png)")
    parser.add_argument(
        "--metric",
        default="perceived_loafing",
        choices=list(METRICS.keys()),
        help="Which metric to plot",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    output_path = Path(args.output) if args.output else csv_path.parent / f"{args.metric}_heatmap.png"

    pivot = load_pivot(csv_path, args.metric)
    print(f"Loaded {len(pivot)} trials × {pivot.shape[1]} agents")
    print(pivot.to_string())
    print()

    plot_heatmap(pivot, args.metric, output_path)


if __name__ == "__main__":
    main()
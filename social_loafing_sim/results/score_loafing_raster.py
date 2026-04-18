"""
score_loafing_raster.py

Reads results/loafing_scores.jsonl (output of analyze_results.py) and produces:
  - results/loafing_raster.csv  : one row per (trial, agent) for statistical analysis
  - results/raster_heatmap.png  : heatmap — 5 agent slots as columns, trials as rows.
                                   Each cell shows the loafing score (colour) and the
                                   agent's 8-digit trait code (text).

Digit key for 8-character trait code:
  1: Openness            (1=low, 2=medium, 3=high)
  2: Conscientiousness   (1=low, 2=medium, 3=high)
  3: Extraversion        (1=low, 2=medium, 3=high)
  4: Agreeableness       (1=low, 2=medium, 3=high)
  5: Neuroticism         (1=low, 2=medium, 3=high)
  6: Group work pref     (1=low, 2=high)
  7: Winning orientation (1=low, 2=high)
  8: Skill level         (1=low, 2=medium, 3=high)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Config ---
INPUT_PATH = Path("results/loafing_scores.jsonl")
OUT_CSV    = Path("results/loafing_raster.csv")
OUT_PNG    = Path("results/raster_heatmap.png")

DIMS = ["task_avoidance", "free_riding", "disengagement"]
CSV_COLS = [
    "trial_id", "seed", "student", "trait_code", "profile_id",
    "n_entries", "n_summaries",
    "task_avoidance", "free_riding", "disengagement",
    "loafing_index", "loafing_index_norm",
    "reasoning",
]


def load_scores(path: Path) -> list[dict]:
    """Flatten loafing_scores.jsonl into one row per (trial, agent)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial    = json.loads(line)
            trial_id = trial["trial_id"]
            seed     = trial["seed"]

            for agent_name, scores in sorted(trial["loafing_scores"].items()):
                rows.append({
                    "trial_id":           trial_id,
                    "seed":               seed,
                    "student":            agent_name,
                    "trait_code":         scores.get("trait_code", "????????"),
                    "profile_id":         scores.get("profile_id", "unknown"),
                    "n_entries":          scores.get("n_entries"),
                    "n_summaries":        scores.get("n_summaries"),
                    "task_avoidance":     scores.get("task_avoidance"),
                    "free_riding":        scores.get("free_riding"),
                    "disengagement":      scores.get("disengagement"),
                    "loafing_index":      scores.get("loafing_index"),
                    "loafing_index_norm": None,  # filled in below
                    "reasoning":          scores.get("reasoning", ""),
                })
    return rows


def add_within_trial_normalization(rows: list[dict]) -> list[dict]:
    """z-score of loafing_index within each trial (relative to teammates)."""
    trial_groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        trial_groups[r["trial_id"]].append(r)

    for group in trial_groups.values():
        vals = [r["loafing_index"] for r in group if r["loafing_index"] is not None]
        if len(vals) < 2:
            continue
        mu  = np.mean(vals)
        std = np.std(vals, ddof=0)
        for r in group:
            if r["loafing_index"] is None:
                continue
            r["loafing_index_norm"] = (
                round((r["loafing_index"] - mu) / std, 3) if std > 0 else 0.0
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] CSV written to {path}  ({len(rows)} rows)")


def build_heatmap(rows: list[dict], path: Path) -> None:
    trial_ids   = sorted({r["trial_id"] for r in rows})
    agent_names = sorted({r["student"] for r in rows})  # Student_1 ... Student_5

    n_trials = len(trial_ids)
    n_agents = len(agent_names)

    # score_matrix: loafing_index per cell
    score_matrix = np.full((n_trials, n_agents), np.nan)
    # code_matrix: trait code string per cell (varies per trial)
    code_matrix  = np.full((n_trials, n_agents), "", dtype=object)

    for r in rows:
        ti = trial_ids.index(r["trial_id"])
        ci = agent_names.index(r["student"])
        if r.get("loafing_index") is not None:
            score_matrix[ti, ci] = r["loafing_index"]
        code_matrix[ti, ci] = r.get("trait_code", "")

    # Narrow fixed width (5 cols); height scales with trial count
    cell_h  = 0.28                          # inches per row
    fig_w   = 2.0 + n_agents * 1.6         # ~10 for 5 agents
    fig_h   = max(4, n_trials * cell_h + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=1, vmax=5)

    ax.imshow(score_matrix, cmap=cmap, norm=norm, aspect="auto")

    # x-axis: fixed agent slot names
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels(agent_names, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # y-axis: trial IDs
    ax.set_yticks(range(n_trials))
    ax.set_yticklabels([f"T{t}" for t in trial_ids], fontsize=6)

    # Annotate every cell with score + trait code on separate lines
    for ri in range(n_trials):
        for ci in range(n_agents):
            v    = score_matrix[ri, ci]
            code = code_matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.1f}\n{code}",
                        ha="center", va="center",
                        fontsize=5, fontfamily="monospace",
                        color="black", linespacing=1.3)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, fraction=0.05, pad=0.02,
    )
    cbar.set_label("Loafing score\n(1=engaged, 5=loafing)", fontsize=8)

    legend_text = (
        "Trait code: [O][C][E][A][N][GW][WO][SK]   "
        "O/C/E/A/N: 1=low 2=med 3=high  |  "
        "GW (group work): 1=low 2=high  |  "
        "WO (winning): 1=low 2=high  |  "
        "SK (skill): 1=low 2=med 3=high"
    )
    fig.text(0.5, -0.01, legend_text, ha="center", fontsize=7,
             style="italic", color="dimgray")

    ax.set_title("Social Loafing -- All Trials x All Students",
                 fontsize=10, fontweight="bold", pad=20)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[ok] Heatmap saved to {path}")


def print_summary(rows: list[dict]) -> None:
    """Print mean loafing index grouped by trait code across all trials."""
    code_scores: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["loafing_index"] is not None:
            code_scores[r["trait_code"]].append(r["loafing_index"])

    print("\n-- Mean loafing index per trait code (across all trials) --")
    print(f"  {'Code':10s}  {'mean':>6}  {'std':>6}  {'n':>4}")
    for code, vals in sorted(code_scores.items()):
        print(f"  {code:10s}  {np.mean(vals):6.2f}  {np.std(vals):6.2f}  {len(vals):4d}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run analyze_results.py first."
        )

    rows = load_scores(INPUT_PATH)
    rows = add_within_trial_normalization(rows)

    n_trials = len({r["trial_id"] for r in rows})
    print(f"Loaded {len(rows)} agent rows from {n_trials} trials.")

    write_csv(rows, OUT_CSV)
    print_summary(rows)
    build_heatmap(rows, OUT_PNG)


if __name__ == "__main__":
    main()
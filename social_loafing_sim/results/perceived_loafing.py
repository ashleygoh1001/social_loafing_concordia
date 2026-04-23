"""
Perceived Loafing Scorer
========================
Sends each agent's observed actions (as seen by each peer) to Gemini and asks
it to rate perceived loafing on four 5-point Likert items. Results are saved
to a CSV and visualised as heatmaps.

Requires:  export GEMINI_API_KEY="your-key-here"

Likert items (1 = strongly disagree -> 5 = strongly agree):
  Q1. Members of my group are trying as hard as they can.        (reverse-coded)
  Q2. Members of my group are "free-loaders."
  Q3. Members of my group are contributing less than I anticipated.
  Q4. Given their abilities, my group members are doing the best they can. (reverse-coded)

Composite Perceived Loafing Score (PLS, 1-5):
  Reverse-code Q1 and Q4, then average all four items.
  Higher score -> more loafing perceived.

Usage
-----
  # Score one or more input files, write results to a named CSV:
  python perceived_loafing.py file1.jsonl file2.jsonl --output-csv results.csv

  # Control parallelism (default: 10 workers):
  python perceived_loafing.py file1.jsonl --output-csv results.csv --workers 20

  # Regenerate heatmaps from an already-scored CSV (no API calls):
  python perceived_loafing.py --heatmap-only --output-csv results.csv
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
API_URL        = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

LIKERT_ITEMS = [
    ("Q1", "Members of my group are trying as hard as they can."),
    ("Q2", 'Members of my group are "free-loaders."'),
    ("Q3", "Members of my group are contributing less than I anticipated."),
    ("Q4", "Given their abilities, my group members are doing the best they can."),
]

# Q1 and Q4 are reverse-coded (agreement = LOW loafing)
REVERSE_CODED = {"Q1", "Q4"}

# Thread-safe print lock
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_peer_context(target_agent, actions_by_agent):
    """Return a summary of target_agent's actions as text."""
    actions = actions_by_agent.get(target_agent, [])
    if not actions:
        return f"{target_agent} did not take any visible actions."
    lines = [
        f"Step {a['step']}: {a['action']}"
        for a in sorted(actions, key=lambda x: x["step"])
    ]
    return "\n".join(lines)


def score_with_gemini(rater, target, context):
    """
    Ask Gemini to rate perceived loafing from rater's perspective of target.
    Returns dict {Q1..Q4: int, PLS: float} or None on failure.
    """
    if not GEMINI_API_KEY:
        sys.exit(
            "ERROR: GEMINI_API_KEY is not set.\n"
            "Run:  export GEMINI_API_KEY='your-key-here'"
        )

    items_text = "\n".join(f"{qid}. {text}" for qid, text in LIKERT_ITEMS)
    prompt = f"""You are playing the role of a student named {rater} in a group project.
Below is a record of what your group member {target} said and did during your group meeting.

--- {target}'s actions ---
{context}
---

Rate how you ({rater}) perceive {target}'s effort and contribution using the four
statements below. For each, respond with a single integer 1-5 where:
  1 = Strongly Disagree
  2 = Disagree
  3 = Neutral
  4 = Agree
  5 = Strongly Agree

Statements:
{items_text}

Respond ONLY with a JSON object mapping each question ID to its integer score,
e.g. {{"Q1": 4, "Q2": 2, "Q3": 2, "Q4": 4}}
No other text."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 10000, "temperature": 0.0},
    }
    params = {"key": GEMINI_API_KEY}

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, json=payload, params=params, timeout=60)

            # Handle rate limiting explicitly with backoff
            if resp.status_code == 429:
                wait = (2 ** attempt) * 5
                tprint(f"    Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            raw = resp.json()

            # Safely navigate response structure
            try:
                text = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError) as nav_err:
                raise ValueError(
                    f"Unexpected response shape: {nav_err}\n"
                    f"Full response: {json.dumps(raw)[:500]}"
                )

            # Strip markdown fences (```json ... ``` or ``` ... ```)
            text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")

            # Extract the first {...} block in case there is surrounding text
            match = re.search(r"\{[^{}]*\}", text)
            if not match:
                raise ValueError(
                    f"No JSON object found in response. Raw text: {repr(text[:300])}"
                )
            scores = json.loads(match.group())

            # Validate all four items are present and in range
            for qid, _ in LIKERT_ITEMS:
                assert qid in scores and 1 <= int(scores[qid]) <= 5, (
                    f"Missing or out-of-range value for {qid}: {scores.get(qid)}"
                )

            # Compute composite PLS (reverse-code Q1 and Q4)
            vals = []
            for qid, _ in LIKERT_ITEMS:
                v = int(scores[qid])
                vals.append(6 - v if qid in REVERSE_CODED else v)
            scores = {k: int(v) for k, v in scores.items()}
            scores["PLS"] = round(sum(vals) / len(vals), 3)
            return scores

        except Exception as e:
            wait = 2 ** attempt
            tprint(f"    WARNING: Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                tprint(f"    Retrying in {wait}s...")
                time.sleep(wait)

    tprint(f"    ERROR: Could not score {rater} -> {target} after 3 attempts. Row will be None.")
    return None


# ---------------------------------------------------------------------------
# Main scoring loop (parallelised)
# ---------------------------------------------------------------------------

def build_tasks(records):
    """Flatten all (group_id, condition, rater, target, context) tuples."""
    tasks = []
    for rec in records:
        group_id  = rec["group_id"]
        condition = rec["condition"]
        agents    = list(rec["results"]["actions_by_agent"].keys())
        actions   = rec["results"]["actions_by_agent"]
        for rater in agents:
            for target in agents:
                if rater != target:
                    tasks.append((group_id, condition, rater, target,
                                  build_peer_context(target, actions)))
    return tasks


def score_task(args):
    """Worker function: score one rater->target pair. Returns a result dict."""
    file_stem, group_id, condition, rater, target, context = args
    scores = score_with_gemini(rater, target, context)
    row = {
        "file":      file_stem,
        "condition": condition,
        "group_id":  group_id,
        "rater":     rater,
        "target":    target,
    }
    if scores:
        row.update(scores)
    else:
        row.update({qid: None for qid, _ in LIKERT_ITEMS})
        row["PLS"] = None
    return row


def score_file(path, workers=10):
    records = load_jsonl(path)
    records.sort(key=lambda r: r.get("group_id", ""))

    file_stem = Path(path).stem
    tasks = [
        (file_stem, group_id, condition, rater, target, context)
        for rec in records
        for group_id, condition in [(rec["group_id"], rec["condition"])]
        for (group_id2, condition2, rater, target, context) in [
            (rec["group_id"], rec["condition"], rater, target,
             build_peer_context(target, rec["results"]["actions_by_agent"]))
            for rater in rec["results"]["actions_by_agent"]
            for target in rec["results"]["actions_by_agent"]
            if rater != target
        ]
    ]

    # Flatten properly
    tasks = []
    for rec in records:
        group_id  = rec["group_id"]
        condition = rec["condition"]
        actions   = rec["results"]["actions_by_agent"]
        for rater in actions:
            for target in actions:
                if rater != target:
                    tasks.append((
                        file_stem, group_id, condition, rater, target,
                        build_peer_context(target, actions)
                    ))

    total = len(tasks)
    rows  = []
    done_count = 0

    tprint(f"  Submitting {total} scoring tasks across {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(score_task, t): t for t in tasks}
        for future in as_completed(futures):
            done_count += 1
            row = future.result()
            rows.append(row)
            tprint(
                f"  [{done_count}/{total}] {row['group_id']} | "
                f"{row['rater']} rates {row['target']}  ->  PLS={row.get('PLS')}"
            )

    return rows


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def make_heatmap(df, output_dir="."):
    """
    One heatmap per input file.

    Rows   : group_001 … group_100, sorted.
    Columns: Student_1 … Student_5.
    Cell   : mean PLS that student received from all other group members
             in that trial (i.e. average perceived loafing of that student
             as judged by their peers).
    """
    os.makedirs(output_dir, exist_ok=True)
    cmap = "RdYlGn_r"   # red = high loafing, green = low
    vmin, vmax = 1.0, 5.0

    for file_stem, fdf in df.groupby("file", sort=True):
        fdf = fdf.dropna(subset=["PLS"]).copy()

        # Canonical student order by number
        all_students = sorted(
            fdf["target"].unique(),
            key=lambda s: int(re.search(r"\d+", s).group())
        )
        # Groups sorted alphabetically (group_001, group_002, …)
        groups = sorted(fdf["group_id"].unique())

        n_rows = len(groups)
        n_cols = len(all_students)

        # Each cell = mean PLS received by that student in that trial
        # (averaged over all raters who rated them, excluding self)
        pivot = (
            fdf.groupby(["group_id", "target"])["PLS"]
            .mean()
            .unstack("target")          # columns = students
            .reindex(index=groups, columns=all_students)
        )
        matrix = pivot.values.astype(float)

        # Figure sizing
        cell_h = 0.28
        cell_w = 1.4
        fig_h  = max(6, n_rows * cell_h + 2)
        fig_w  = max(5, n_cols * cell_w + 2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            matrix, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest"
        )

        # X axis: students (top)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(all_students, fontsize=10, rotation=30, ha="left")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel("Student (target)", fontsize=11, labelpad=8)

        # Y axis: group labels
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(groups, fontsize=7)
        ax.set_ylabel("Trial", fontsize=11)

        # Colour bar
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("Mean PLS received  (1 = low loafing, 5 = high)", fontsize=9)

        condition = fdf["condition"].iloc[0] if "condition" in fdf.columns else ""
        ax.set_title(
            f"Perceived Loafing — {file_stem}\n"
            f"condition: {condition}  |  {n_rows} trials",
            fontsize=12, pad=18
        )

        plt.tight_layout()
        fname = os.path.join(output_dir, f"heatmap_{file_stem}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score perceived loafing from simulation JSONL files using Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perceived_loafing.py control_output.jsonl indiv_contrib_output.jsonl --output-csv results.csv
  python perceived_loafing.py *.jsonl --output-csv all_results.csv --workers 20
  python perceived_loafing.py --heatmap-only --output-csv results.csv
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more JSONL input files to score"
    )
    parser.add_argument(
        "--output-csv", required=True,
        metavar="OUTPUT.csv",
        help="Path for the output CSV (required)"
    )
    parser.add_argument(
        "--heatmap-dir", default="heatmaps",
        metavar="DIR",
        help="Directory to save heatmap PNGs (default: heatmaps/)"
    )
    parser.add_argument(
        "--heatmap-only", action="store_true",
        help="Skip scoring; load existing --output-csv and regenerate heatmaps only"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        metavar="N",
        help="Number of parallel API workers (default: 10)"
    )
    args = parser.parse_args()

    if args.heatmap_only:
        if not os.path.exists(args.output_csv):
            sys.exit(f"ERROR: --output-csv file not found: {args.output_csv}")
        print(f"Loading existing scores from {args.output_csv} ...")
        df = pd.read_csv(args.output_csv)
    else:
        if not args.files:
            parser.error(
                "Please provide at least one input JSONL file.\n"
                "Example: python perceived_loafing.py file1.jsonl file2.jsonl --output-csv results.csv"
            )
        all_rows = []
        for fpath in args.files:
            if not os.path.exists(fpath):
                print(f"WARNING: File not found: {fpath} — skipping.")
                continue
            print(f"\n=== Scoring {fpath} ===")
            all_rows.extend(score_file(fpath, workers=args.workers))

        if not all_rows:
            sys.exit("No data was scored. Check that your input files exist and are valid JSONL.")

        df = pd.DataFrame(all_rows)
        df = df.sort_values(["file", "group_id", "rater", "target"]).reset_index(drop=True)

        # Save incrementally in case of interruption
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to {args.output_csv}")

    print(f"\nGenerating heatmaps -> {args.heatmap_dir}/")
    make_heatmap(df, output_dir=args.heatmap_dir)
    print("\nDone.")

    print("\n=== Mean PLS by condition ===")
    if "condition" in df.columns and "PLS" in df.columns:
        print(df.groupby("condition")["PLS"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
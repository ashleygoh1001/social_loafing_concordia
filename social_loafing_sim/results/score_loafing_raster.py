"""
score_loafing_raster.py
 
Reads simulation_results_2.jsonl and produces:
  - loafing_raster.jsonl   : one row per (trial, student) with raw dimension scores
  - loafing_raster.csv     : same data in tabular form for easy analysis
  - raster_heatmap.png     : visual heatmap of loafing scores across trials x students
 
Loafing is scored across 5 behavioural dimensions (each 1-5, higher = more loafing):
  1. initiative       : Did they start topics, propose ideas, or just react?
  2. specificity      : Were their contributions vague agreement or concrete substance?
  3. leadership       : Did they drive decisions or defer to others?
  4. free_riding      : Did they build on others' ideas without adding their own?
  5. disengagement    : Did they go quiet, give minimal responses, or just nod?
 
A composite LOAFING INDEX = mean of all 5 dimensions.
"""
 
from __future__ import annotations
 
import json
import os
import re
import time
from pathlib import Path
 
from google import genai
 
# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("simulation_results_2.jsonl")
OUT_JSONL   = Path("loafing_raster.jsonl")
OUT_CSV     = Path("loafing_raster.csv")
OUT_PNG     = Path("raster_heatmap.png")
 
MODEL = "gemini-2.5-flash"
# ─────────────────────────────────────────────────────────────────────────────
 
 
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
 
# ── Text extraction helpers ───────────────────────────────────────────────────
 
def extract_summary(entry_str: str) -> str | None:
    m = re.search(r"summary='(.*?)(?:', deduplicated|\\n')", str(entry_str), re.DOTALL)
    if m:
        return m.group(1).replace("\\'", "'").strip()
    return None
 
 
def build_transcript(entries: list, memories: list) -> str:
    """Combine action summaries + memories into a readable transcript."""
    lines: list[str] = []
 
    if entries:
        lines.append("=== Actions ===")
        for e in entries:
            s = extract_summary(e)
            if s:
                lines.append(f"- {s}")
 
    if memories:
        lines.append("\n=== Memories / observations ===")
        for m in memories[:6]:   # cap at 6 to keep prompt short
            lines.append(f"- {str(m)[:300]}")
 
    return "\n".join(lines)
 
 
# ── LLM scoring ──────────────────────────────────────────────────────────────
 
SCORE_PROMPT = """You are a researcher studying social loafing in student group work.
 
Below is the transcript of ONE student's actions and observations during a 10-step
simulated CS group project meeting.
 
Student: {name}
---
{transcript}
---
 
Score this student on EACH of the following dimensions.
Use a scale of 1–5 where:
  1 = not at all (highly engaged / contributing)
  5 = extreme (clear social loafing on this dimension)
 
Dimensions:
  initiative    : Did they initiate topics or only react to others?
  specificity   : Were contributions vague/generic or concrete and substantive?
  leadership    : Did they drive decisions or just defer/agree?
  free_riding   : Did they build on others without adding original ideas?
  disengagement : Did they go quiet, give minimal responses, or just nod along?
 
Reply ONLY with a JSON object — no markdown, no explanation outside the JSON:
{{
  "initiative":    <1-5>,
  "specificity":   <1-5>,
  "leadership":    <1-5>,
  "free_riding":   <1-5>,
  "disengagement": <1-5>,
  "reasoning":     "<one sentence summary of your assessment>"
}}"""
 
 
def score_agent(name: str, transcript: str) -> dict:
    prompt = SCORE_PROMPT.format(name=name, transcript=transcript)
 
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 300,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            text = (response.text or "").strip()
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
            data = json.loads(text)
 
            # Compute composite loafing index
            dims = ["initiative", "specificity", "leadership", "free_riding", "disengagement"]
            scores = [data.get(d) for d in dims if isinstance(data.get(d), (int, float))]
            data["loafing_index"] = round(sum(scores) / len(scores), 2) if scores else None
            return data
 
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"    [attempt {attempt+1}] error: {e} — retrying in {wait}s")
            time.sleep(wait)
 
    return {
        "initiative": None, "specificity": None, "leadership": None,
        "free_riding": None, "disengagement": None,
        "loafing_index": None, "reasoning": "scoring failed"
    }
 
 
# ── Main ─────────────────────────────────────────────────────────────────────
 
def main() -> None:
    with open(INPUT_PATH) as f:
        trials = [json.loads(l) for l in f if l.strip()]
 
    rows: list[dict] = []
 
    for trial in trials:
        if trial.get("status") != "success":
            continue
 
        tid       = trial["trial_id"]
        profiles  = trial.get("profiles", [])
        ebe       = trial["results"].get("entries_by_entity", {})
        memories  = trial["results"].get("entity_memories", {})
 
        students = [k for k in ebe if k != "course_staff"]
        print(f"\n── Trial {tid} ({'  '.join(students)}) ──")
 
        for i, name in enumerate(sorted(students)):
            profile_id = profiles[i] if i < len(profiles) else "unknown"
            transcript = build_transcript(
                entries  = ebe.get(name, []),
                memories = memories.get(name, []),
            )
 
            print(f"  Scoring {name} ({profile_id})...", end=" ", flush=True)
            scores = score_agent(name, transcript)
            print(f"loafing_index={scores.get('loafing_index')}")
 
            row = {
                "trial_id":    tid,
                "student":     name,
                "profile_id":  profile_id,
                **scores,
            }
            rows.append(row)
 
            # Write incrementally so progress is saved on crash
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(row) + "\n")
 
    # ── Write CSV ─────────────────────────────────────────────────────────────
    import csv
    cols = ["trial_id", "student", "profile_id",
            "initiative", "specificity", "leadership",
            "free_riding", "disengagement", "loafing_index", "reasoning"]
 
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
 
    print(f"\n✓ Scores written to {OUT_JSONL} and {OUT_CSV}")
 
    # ── Heatmap ───────────────────────────────────────────────────────────────
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
 
        dims = ["initiative", "specificity", "leadership", "free_riding", "disengagement"]
 
        # Build trial x student matrix for loafing_index
        trial_ids = sorted({r["trial_id"] for r in rows})
        students  = sorted({r["student"]  for r in rows})
 
        # Matrix 1: loafing index per trial x student
        index_matrix = np.full((len(trial_ids), len(students)), np.nan)
        for r in rows:
            ti = trial_ids.index(r["trial_id"])
            si = students.index(r["student"])
            if r.get("loafing_index") is not None:
                index_matrix[ti, si] = r["loafing_index"]
 
        # Matrix 2: all 5 dimensions stacked
        dim_matrices = {}
        for d in dims:
            m = np.full((len(trial_ids), len(students)), np.nan)
            for r in rows:
                ti = trial_ids.index(r["trial_id"])
                si = students.index(r["student"])
                if r.get(d) is not None:
                    m[ti, si] = r[d]
            dim_matrices[d] = m
 
        fig, axes = plt.subplots(
            2, 3, figsize=(18, 10),
            gridspec_kw={"hspace": 0.45, "wspace": 0.3}
        )
        axes = axes.flatten()
 
        cmap = plt.cm.RdYlGn_r   # red = high loafing, green = low
        norm = mcolors.Normalize(vmin=1, vmax=5)
 
        def draw_heatmap(ax, matrix, title):
            im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(len(students)))
            ax.set_xticklabels(students, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(trial_ids)))
            ax.set_yticklabels([f"Trial {t}" for t in trial_ids], fontsize=8)
            ax.set_title(title, fontsize=10, fontweight="bold")
            # Annotate cells
            for ti in range(len(trial_ids)):
                for si in range(len(students)):
                    v = matrix[ti, si]
                    if not np.isnan(v):
                        ax.text(si, ti, f"{v:.1f}", ha="center", va="center",
                                fontsize=7, color="black")
            return im
 
        im = draw_heatmap(axes[0], index_matrix, "Composite Loafing Index")
        for i, d in enumerate(dims):
            draw_heatmap(axes[i+1], dim_matrices[d], d.replace("_", " ").title())
 
        # Shared colourbar
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes, fraction=0.015, pad=0.04,
            label="Score (1=engaged → 5=loafing)"
        )
        fig.suptitle("Social Loafing Raster — All Trials × All Students",
                     fontsize=13, fontweight="bold", y=1.01)
 
        plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        print(f"✓ Heatmap saved to {OUT_PNG}")
 
    except ImportError as e:
        print(f"⚠ Could not generate heatmap (missing library: {e}). "
              f"CSV and JSONL are still complete.")
 
 
if __name__ == "__main__":
    main()
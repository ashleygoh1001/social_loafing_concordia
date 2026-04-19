"""
score_loafing.py

Score social loafing from simulation JSONL output using the Gemini API.

Questionnaires scored per agent:
  - Mulvey & Klein (1998): Perceived Loafing, Anticipated Lower Effort, Sucker Effect (1-5 scale)
  - Ying et al. Social Loafing Tendency Questionnaire (1-7 scale)

Usage:
    python score_loafing.py --input results/run_100.jsonl --output scores.csv
    python score_loafing.py --input results/run_100.jsonl --output scores.csv --max_trials 20
    python score_loafing.py --input results/run_100.jsonl --output scores.csv --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Allow running this script without installing `social_loafing_sim` as a package.
_SIM_ROOT = Path(__file__).resolve().parents[1]
_SIMULATIONS_DIR = _SIM_ROOT / "simulations"
if _SIMULATIONS_DIR.exists():
    sys.path.insert(0, str(_SIMULATIONS_DIR))

from safe_gemini_model import SafeGeminiModel  # pyright: ignore[reportMissingImports]

AGENTS = ["Student_1", "Student_2", "Student_3", "Student_4", "Student_5"]

SYSTEM_PROMPT = """You are a research assistant scoring simulated group dynamics using validated questionnaires.

You will receive a simulation transcript showing what each student said and did during a CS group project meeting.

Score each agent (Student_1 through Student_5) on these questionnaires. All scores use Likert scales — infer from behavior and speech.

**Mulvey & Klein (1998) — 5-point scale (1=strongly disagree, 5=strongly agree)**

Perceived loafing (PL) — how much loafing the group exhibits overall. Score as perceived by an outside observer:
- PL1: Members of my group are trying as hard as they can (reverse scored)
- PL2: Members of my group are "free-loaders"
- PL3: Members of my group are contributing less than I anticipated
- PL4: Given their abilities, my group members are doing the best they can (reverse scored)
Average the 4 items (after reversing PL1 and PL4) to get the PL score.

Anticipated lower effort (ALE) — tendency for the group to reduce effort because some members loaf:
- ALE1: Because some group members are not trying as hard as they can, the rest will probably put in less effort
- ALE2: Some members are putting in less effort, so other members will not try as hard
- ALE3: Because some members are not doing their share, I don't think anyone will work as hard as they could
- ALE4: Since some members are not expending much effort, others will likely reduce their effort
Average the 4 items to get the ALE score.

Sucker effect (SE) — personally reducing effort because others loaf:
- SE1: Because other members are not contributing as much as they could, I'm not trying my best
- SE2: Because others are putting in less effort, I do not plan to continue to work hard
- SE3: Others are not trying their best, so I'm not trying my best either
- SE4: Because others are not trying as hard as they could, I am not working as hard as I could
- SE5: Because others are not trying as hard as they can, I am going to reduce my effort
Average the 5 items to get the SE score.

**Ying et al. Social Loafing Tendency Questionnaire (SLT) — 7-point scale (1=strongly disagree, 7=strongly agree)**
- SLT1: In a team, I am not indispensable
- SLT2: In a team, I will try as hard as I can (reverse scored)
- SLT3: In a team, I will contribute less than I should
- SLT4: In a team, I will actively participate in the discussion and contribute ideas (reverse scored)
- SLT5: In a team it is okay even if I do not do my share
- SLT6: In a team, it does not matter whether or not I try my best
- SLT7: In a team, given my abilities, I will do the best I can (reverse scored)
Average all 7 items (after reversing SLT2, SLT4, SLT7) to get the SLT score.

For EACH agent individually, score based on their specific behavior in the transcript.

Return ONLY valid JSON with this exact structure (no preamble, no markdown fences):
{
  "Student_1": {
    "perceived_loafing": <float 1-5>,
    "anticipated_lower_effort": <float 1-5>,
    "sucker_effect": <float 1-5>,
    "ying_slt": <float 1-7>,
    "notes": "<one sentence behavioral observation>"
  },
  "Student_2": { ... },
  "Student_3": { ... },
  "Student_4": { ... },
  "Student_5": { ... },
  "trial_summary": "<two sentences on overall group social loafing dynamics>"
}"""


def build_transcript(trial: dict) -> str:
    steps = trial.get("results", {}).get("steps", {})
    lines = []
    for step_num in sorted(steps.keys(), key=lambda x: int(x)):
        for entry in steps[step_num]:
            lines.append(f"[Step {step_num}] {entry['agent']}: {entry['action']}")
    return "\n".join(lines)


def score_trial(model: SafeGeminiModel, trial: dict, max_retries: int = 5) -> dict:
    profiles_str = ", ".join(
        f"Student_{i+1}: {p}" for i, p in enumerate(trial.get("profiles", []))
    )
    transcript = build_transcript(trial)

    if not transcript.strip():
        raise ValueError("Trial has no transcript content")

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + f"Profiles: {profiles_str}\n\nTranscript:\n{transcript}"
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            text = model.sample_text(prompt, max_tokens=2048)
            return json.loads(text)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = min(2 ** attempt, 60)
                print(f"  Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"Scoring failed after {max_retries} attempts: {last_err}")


def flatten_scores(trial: dict, scores: dict) -> list[dict]:
    """Convert a trial's scores dict into one CSV row per agent."""
    rows = []
    profiles = trial.get("profiles", [])
    for i, agent in enumerate(AGENTS):
        agent_scores = scores.get(agent, {})
        rows.append({
            "trial_id": trial.get("trial_id", ""),
            "seed": trial.get("seed", ""),
            "agent": agent,
            "profile_id": profiles[i] if i < len(profiles) else "",
            "perceived_loafing": agent_scores.get("perceived_loafing", ""),
            "anticipated_lower_effort": agent_scores.get("anticipated_lower_effort", ""),
            "sucker_effect": agent_scores.get("sucker_effect", ""),
            "ying_slt": agent_scores.get("ying_slt", ""),
            "notes": agent_scores.get("notes", ""),
            "trial_summary": scores.get("trial_summary", ""),
        })
    return rows


CSV_FIELDS = [
    "trial_id", "seed", "agent", "profile_id",
    "perceived_loafing", "anticipated_lower_effort", "sucker_effect", "ying_slt",
    "notes", "trial_summary",
]


def load_completed_trial_ids(output_path: Path) -> set[int]:
    """Read already-scored trial IDs from an existing CSV for resume support."""
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(int(row["trial_id"]))
            except (KeyError, ValueError):
                pass
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Score social loafing from simulation JSONL.")
    parser.add_argument("--input", required=True, help="Path to simulation JSONL file")
    parser.add_argument("--output", default="loafing_scores.csv", help="Output CSV path")
    parser.add_argument("--max_trials", type=int, default=None, help="Max trials to score (default: all)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model to use")
    parser.add_argument("--resume", action="store_true", help="Skip trials already in output CSV")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY environment variable before running.")

    model = SafeGeminiModel(model_name=args.model, api_key=api_key)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load trials
    trials = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    t = json.loads(line)
                    if t.get("status") == "success":
                        trials.append(t)
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(trials)} successful trials from {input_path}")

    # Resume support
    completed_ids: set[int] = set()
    if args.resume:
        completed_ids = load_completed_trial_ids(output_path)
        if completed_ids:
            print(f"Resuming — skipping {len(completed_ids)} already-scored trials.")

    pending = [t for t in trials if t.get("trial_id") not in completed_ids]
    if args.max_trials is not None:
        pending = pending[: args.max_trials]

    if not pending:
        print("No trials to score.")
        return

    print(f"Scoring {len(pending)} trials → {output_path}\n")

    # Open CSV (append if resuming, write fresh otherwise)
    write_header = not output_path.exists() or not args.resume
    with open(output_path, "a" if args.resume else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        ok = 0
        errors = 0
        for i, trial in enumerate(pending):
            trial_id = trial.get("trial_id", "?")
            print(f"[{i+1}/{len(pending)}] Scoring trial {trial_id}...", end=" ", flush=True)
            try:
                scores = score_trial(model, trial)
                rows = flatten_scores(trial, scores)
                writer.writerows(rows)
                f.flush()
                ok += 1
                print(f"done  (PL avg: {sum(r['perceived_loafing'] for r in rows if r['perceived_loafing'] != '') / len(rows):.2f})")
            except Exception as e:
                errors += 1
                print(f"ERROR: {e}")

            if i < len(pending) - 1:
                time.sleep(args.delay)

    print(f"\nDone. {ok} scored, {errors} failed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
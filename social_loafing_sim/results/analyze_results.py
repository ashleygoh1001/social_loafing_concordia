import json
import re
import os
import time
from pathlib import Path
from google import genai

from profile_codes import load_code_lookup

# --- Config ---
INPUT_PATH  = Path("results/simulation_results_2.jsonl")
OUTPUT_PATH = Path("results/loafing_scores.jsonl")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Loaded once at startup: profile_id -> 8-digit trait code
CODE_LOOKUP: dict[str, str] = load_code_lookup()


def load_completed_trial_ids(path: Path) -> set[int]:
    completed = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["trial_id"])
            except Exception:
                pass
    return completed


def extract_summaries(entries: list) -> list[str]:
    summaries = []
    for e in entries:
        m = re.search(r"summary='(.*?)(?:', deduplicated|\\n')", str(e), re.DOTALL)
        if m:
            summaries.append(m.group(1).replace("\\'", "'"))
    return summaries


def rate_loafing(agent_name: str, summaries: list[str]) -> dict:
    if not summaries:
        return {
            "task_avoidance": None,
            "free_riding": None,
            "disengagement": None,
            "loafing_index": None,
            "reasoning": "no entries",
        }

    transcript = "\n".join(f"- {s}" for s in summaries)
    prompt = f"""You are evaluating social loafing in a CS group project simulation.

Agent: {agent_name}
Their actions during the simulation:
{transcript}

Rate this agent on the following dimensions (each 1-5, where 1 = fully engaged, 5 = maximum loafing):
1. task_avoidance: Did they avoid taking on concrete work or responsibilities?
2. free_riding: Did they let others lead and do the heavy lifting without contributing?
3. disengagement: Did they contribute minimally or just agree with others passively?

Reply ONLY as a JSON object with no markdown:
{{"task_avoidance": int, "free_riding": int, "disengagement": int, "reasoning": "one sentence"}}"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 256,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            text = response.text.strip()
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
            data = json.loads(text)

            dims = ["task_avoidance", "free_riding", "disengagement"]
            scores = [data[d] for d in dims if isinstance(data.get(d), (int, float))]
            data["loafing_index"] = round(sum(scores) / len(scores), 2) if scores else None

            return data

        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"    [attempt {attempt+1}] error: {e} -- retrying in {wait}s")
            time.sleep(wait)

    return {
        "task_avoidance": None,
        "free_riding": None,
        "disengagement": None,
        "loafing_index": None,
        "reasoning": "scoring failed after 5 attempts",
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_trial_ids(OUTPUT_PATH)
    if completed_ids:
        print(f"Resuming -- skipping {len(completed_ids)} already-scored trials.\n")

    with open(INPUT_PATH) as f:
        trials = [json.loads(l) for l in f if l.strip()]

    for trial in trials:
        if trial["status"] != "success":
            continue

        trial_id = trial["trial_id"]
        if trial_id in completed_ids:
            continue

        ebe      = trial["results"].get("entries_by_entity", {})
        profiles = trial.get("profiles", [])  # ordered Student_1 ... Student_5

        # Preserve Student_N ordering when mapping to profile IDs
        agent_names = sorted(k for k in ebe if k != "course_staff")
        agent_profile_id = {
            name: profiles[i] if i < len(profiles) else "unknown"
            for i, name in enumerate(agent_names)
        }
        agent_code = {
            name: CODE_LOOKUP.get(agent_profile_id[name], "????????")
            for name in agent_names
        }

        loafing_scores = {}

        print(f"\n-- Trial {trial_id} --")
        for agent_name in agent_names:
            entries    = ebe[agent_name]
            code       = agent_code[agent_name]
            profile_id = agent_profile_id[agent_name]
            summaries  = extract_summaries(entries)

            print(f"  {agent_name} [{code}] | {len(summaries)} summaries...", end=" ", flush=True)
            scores = rate_loafing(agent_name, summaries)
            loafing_scores[agent_name] = {
                **scores,
                "profile_id":  profile_id,
                "trait_code":  code,
                "n_entries":   len(entries),
                "n_summaries": len(summaries),
            }
            print(f"loafing_index={scores.get('loafing_index')}")

        output = {
            "trial_id":       trial_id,
            "seed":           trial["seed"],
            "profiles":       profiles,
            "loafing_scores": loafing_scores,
        }

        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(output) + "\n")

        print(f"  [ok] Trial {trial_id} written.")

    print(f"\nDone. Scores written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
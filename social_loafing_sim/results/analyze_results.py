import json
import re
import os
import time
from pathlib import Path
from google import genai

# --- Config ---
INPUT_PATH = Path("results/simulation_results_2.jsonl")
OUTPUT_PATH = Path("results/loafing_scores.jsonl")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def extract_summaries(entries: list) -> list[str]:
    """Pull the human-readable summary text out of each raw entry string."""
    summaries = []
    for e in entries:
        m = re.search(r"summary='(.*?)(?:', deduplicated|\\n')", str(e), re.DOTALL)
        if m:
            summaries.append(m.group(1).replace("\\'", "'"))
    return summaries

def rate_loafing(agent_name: str, summaries: list[str]) -> dict:
    if not summaries:
        return {"task_avoidance": None, "free_riding": None, "disengagement": None, "reasoning": "no entries"}

    transcript = "\n".join(f"- {s}" for s in summaries)
    prompt = f"""You are evaluating social loafing in a CS group project simulation.

Agent: {agent_name}
Their actions during the simulation:
{transcript}

Rate this agent on the following (each 1-5, where 5 = maximum loafing/disengagement):
1. task_avoidance: Did they avoid taking on concrete work or responsibilities?
2. free_riding: Did they let others lead and do the heavy lifting?
3. disengagement: Did they contribute minimally or just agree with others?

Reply ONLY as a JSON object with no markdown:
{{"task_avoidance": int, "free_riding": int, "disengagement": int, "reasoning": "one sentence"}}"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 256,
                        "thinking_config": {"thinking_budget": 0}},
            )
            text = response.text.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(text)
        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == 4:
                return {"error": str(e)}

def main():
    with open(INPUT_PATH) as f:
        trials = [json.loads(l) for l in f if l.strip()]

    for trial in trials:
        if trial["status"] != "success":
            continue

        trial_id = trial["trial_id"]
        ebe = trial["results"].get("entries_by_entity", {})
        loafing_scores = {}

        for agent_name, entries in ebe.items():
            if agent_name == "course_staff":
                continue  # skip game master
            summaries = extract_summaries(entries)
            print(f"  Trial {trial_id} | {agent_name} | {len(summaries)} entries")
            loafing_scores[agent_name] = rate_loafing(agent_name, summaries)
            loafing_scores[agent_name]["n_entries"] = len(summaries)

        output = {
            "trial_id": trial_id,
            "seed": trial["seed"],
            "profiles": trial["profiles"],
            "loafing_scores": loafing_scores,
        }

        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(output) + "\n")

        print(f"Trial {trial_id} done: {list(loafing_scores.keys())}")

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    main()
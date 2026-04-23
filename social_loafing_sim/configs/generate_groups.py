"""
generate_groups.py
------------------
Reads trait_pool.yaml and samples 100 groups of 5 agents WITH replacement.
Outputs:
  - groups.jsonl  – one group per line, easy to load in simulation code
  - groups.csv    – flat table, one row per agent-slot
"""

import random
import json
import csv
import yaml
import copy

YAML_PATH   = "./configs/trait_pool.yaml"
JSONL_OUT   = "./configs/simulation_groups.jsonl"
CSV_OUT     = "./configs/simulation_groups.csv"

NUM_GROUPS  = 100
GROUP_SIZE  = 5
RANDOM_SEED = 42          # set to None for a different draw each run


# ── 1. Load profiles ──────────────────────────────────────────────────────────

with open(YAML_PATH, "r") as fh:
    data = yaml.safe_load(fh)

profiles = data["profiles"]

# Resolve YAML aliases so every profile is a self-contained dict
resolved = []
for p in profiles:
    entry = {
        "profile_id": p["profile_id"],
        "openness":            p["big_five"]["openness"],
        "conscientiousness":   p["big_five"]["conscientiousness"],
        "extraversion":        p["big_five"]["extraversion"],
        "agreeableness":       p["big_five"]["agreeableness"],
        "neuroticism":         p["big_five"]["neuroticism"],
        "group_work_preference": p["group_work_preference"],
        "winning_orientation":   p["winning_orientation"],
        "skill_level":           p["skill_level"],
    }
    resolved.append(entry)

print(f"Loaded {len(resolved)} profiles from {YAML_PATH}")


# ── 2. Sample groups ──────────────────────────────────────────────────────────

rng = random.Random(RANDOM_SEED)

groups = []
for g in range(1, NUM_GROUPS + 1):
    members = [copy.deepcopy(rng.choice(resolved)) for _ in range(GROUP_SIZE)]
    groups.append({
        "group_id": f"group_{g:03d}",
        "agents":   members,
    })

print(f"Generated {len(groups)} groups of {GROUP_SIZE} agents (with replacement).")


# ── 3. Write JSONL ────────────────────────────────────────────────────────────

with open(JSONL_OUT, "w") as fh:
    for group in groups:
        fh.write(json.dumps(group) + "\n")

print(f"JSONL saved → {JSONL_OUT}")


# ── 4. Write CSV ──────────────────────────────────────────────────────────────

FIELDS = [
    "group_id", "agent_slot", "profile_id",
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism",
    "group_work_preference", "winning_orientation", "skill_level",
]

with open(CSV_OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    for group in groups:
        for slot, agent in enumerate(group["agents"], start=1):
            row = {"group_id": group["group_id"], "agent_slot": slot}
            row.update(agent)
            writer.writerow(row)

print(f"CSV  saved → {CSV_OUT}")


# ── 5. Quick sanity check ─────────────────────────────────────────────────────

print("\n── Sample output (group_001) ──")
for slot, agent in enumerate(groups[0]["agents"], 1):
    print(f"  Agent {slot}: {agent['profile_id']}  "
          f"skill={agent['skill_level']}  "
          f"gwp={agent['group_work_preference']}  "
          f"win={agent['winning_orientation']}")
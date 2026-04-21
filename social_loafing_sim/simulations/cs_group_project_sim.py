"""
cs_group_project_sim.py
-----------------------
Unified simulation for all conditions (control + 6 interventions).

Agent teams are loaded from configs/simulation_groups.jsonl — 100 pre-sampled
groups of 5 agents. Trial N (0-indexed) always uses group N, so every
condition runs the exact same 100 teams, making results directly comparable.

The active condition is selected by passing an InterventionSpec at build
time. The spec is injected at three levels so it remains active throughout
the entire simulation rather than only at initialisation:

  LEVEL 1 — PREMISE      : world description seen by the game master from step 1.
  LEVEL 2 — AGENT GOAL   : prepended to every agent's goal so every planning
                            and action decision is framed by the condition.
  LEVEL 3 — AGENT CONTEXT: appended to the situational context each agent
                            reasons from on every step.

Usage (standalone):
    python cs_group_project_sim.py --condition weekly_log --group_index 0

Available conditions:
    control  contribution_tracking  task_visibility  peer_evaluation
    weekly_log  meaningful_feedback  agile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Concordia package path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CONCORDIA_PARENT = _REPO_ROOT / "concordia"
if _LOCAL_CONCORDIA_PARENT.exists():
    sys.path.insert(0, str(_LOCAL_CONCORDIA_PARENT))

from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

from interventions import REGISTRY, InterventionSpec, CONTROL, ALL_CONDITION_NAMES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIMULATION_GROUPS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "simulation_groups.jsonl"
)


# ---------------------------------------------------------------------------
# Group loading
# ---------------------------------------------------------------------------

def load_simulation_groups(path: Path) -> list[dict[str, Any]]:
    """Load all pre-sampled groups from a JSONL file.

    Each line is a JSON object with keys:
        group_id : str          (e.g. "group_001")
        agents   : list[dict]   (5 agent profiles, flat schema)

    Agent profile keys (flat — no nested big_five):
        profile_id, openness, conscientiousness, extraversion,
        agreeableness, neuroticism, group_work_preference,
        winning_orientation, skill_level
    """
    groups: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    if not groups:
        raise ValueError(f"No groups found in {path}")
    return groups


def get_group_agents(groups: list[dict[str, Any]], group_index: int) -> list[dict[str, Any]]:
    """Return the 5 agent profiles for the given 0-based group index."""
    if group_index < 0 or group_index >= len(groups):
        raise IndexError(
            f"group_index {group_index} out of range for "
            f"{len(groups)} groups in simulation_groups.jsonl"
        )
    return groups[group_index]["agents"]


# ---------------------------------------------------------------------------
# Agent schema helper
#
# Agents in simulation_groups.jsonl use a flat schema (all trait fields at
# the top level). The original YAML trait pool used a nested big_five dict.
# This helper normalises both so the rest of the code is consistent.
# ---------------------------------------------------------------------------

def get_big_five(profile: dict[str, Any]) -> dict[str, str]:
    """Return the Big Five traits regardless of flat vs nested schema."""
    if "big_five" in profile:
        return profile["big_five"]
    return {
        "openness":          profile["openness"],
        "conscientiousness": profile["conscientiousness"],
        "extraversion":      profile["extraversion"],
        "agreeableness":     profile["agreeableness"],
        "neuroticism":       profile["neuroticism"],
    }


# ---------------------------------------------------------------------------
# Profile -> agent text helpers
# ---------------------------------------------------------------------------

def make_agent_name(i: int) -> str:
    return f"Student_{i + 1}"


def profile_to_goal(profile: dict[str, Any], condition: InterventionSpec) -> str:
    """Build an agent goal string, prepending the intervention goal prefix."""
    parts: list[str] = []

    if condition.goal_prefix:
        parts.append(condition.goal_prefix)

    parts.append("Help your team complete the CS group project successfully.")

    if profile["winning_orientation"] == "high":
        parts.append(
            "You want your contribution to be visible and recognized."
        )
    else:
        parts.append(
            "You care more about getting the project finished than competing "
            "with teammates."
        )

    if profile["group_work_preference"] == "high":
        parts.append(
            "You prefer active collaboration, discussion, and shared "
            "decision-making."
        )
    else:
        parts.append(
            "You prefer independent work and minimal coordination overhead."
        )

    skill = profile["skill_level"]
    if skill == "high":
        parts.append("You are one of the technically strongest people on the team.")
    elif skill == "medium":
        parts.append("You are reasonably capable and can contribute steadily.")
    else:
        parts.append(
            "You are less confident in your technical ability and may avoid "
            "difficult tasks."
        )

    return " ".join(parts)


def profile_to_context(profile: dict[str, Any], condition: InterventionSpec) -> str:
    """Build an agent context string, appending the intervention context suffix."""
    b5 = get_big_five(profile)
    base = (
        f"You are a student in a 5-person CS group project. "
        f"Traits: C={b5['conscientiousness']}, E={b5['extraversion']}, "
        f"A={b5['agreeableness']}, N={b5['neuroticism']}. "
        f"Group work preference={profile['group_work_preference']}. "
        f"Winning orientation={profile['winning_orientation']}. "
        f"Skill level={profile['skill_level']}."
    )
    if condition.context_suffix:
        return base + condition.context_suffix
    return base


# ---------------------------------------------------------------------------
# Simulation builders
# ---------------------------------------------------------------------------

def build_prefab_registry() -> dict[str, prefab_lib.Prefab]:
    return {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }


def build_instances(
    agents: list[dict[str, Any]],
    condition: InterventionSpec,
) -> list[prefab_lib.InstanceConfig]:
    instances: list[prefab_lib.InstanceConfig] = []

    for i, profile in enumerate(agents):
        instances.append(
            prefab_lib.InstanceConfig(
                prefab="basic_with_plan__Entity",
                role=prefab_lib.Role.ENTITY,
                params={
                    "name":    make_agent_name(i),
                    "goal":    profile_to_goal(profile, condition),
                    "context": profile_to_context(profile, condition),
                },
            )
        )

    gm_params: dict[str, Any] = {
        "name":         "course_staff",
        "acting_order": "random",
    }
    if condition.gm_instructions:
        gm_params["instructions"] = condition.gm_instructions

    instances.append(
        prefab_lib.InstanceConfig(
            prefab="generic__GameMaster",
            role=prefab_lib.Role.GAME_MASTER,
            params=gm_params,
        )
    )

    return instances


def build_premise(
    agents: list[dict[str, Any]],
    condition: InterventionSpec,
    group_id: str = "",
) -> str:
    lines: list[str] = [
        "It is week 7 of a 10-week undergraduate computer science group project.",
        "Five students have been assigned to build a working software system "
        "together for their final project.",
        "The team must decide on architecture, divide coding tasks, coordinate "
        "through GitHub, respond to bugs, and prepare a final demo.",
        "Team members differ in personality, motivation, technical ability, "
        "and comfort with collaboration.",
        "Social loafing, uneven task distribution, leadership struggles, and "
        "deadline stress may emerge over time.",
    ]

    if condition.premise_block:
        lines.append("")
        lines.append(condition.premise_block)

    lines.append("")
    lines.append("The team is meeting after class to organize the project.")

    for i, profile in enumerate(agents):
        b5 = get_big_five(profile)
        lines += [
            "",
            f"{make_agent_name(i)} profile summary:",
            f"  openness={b5['openness']}, conscientiousness={b5['conscientiousness']}, "
            f"extraversion={b5['extraversion']}, agreeableness={b5['agreeableness']}, "
            f"neuroticism={b5['neuroticism']}",
            f"  group_work_preference={profile['group_work_preference']}, "
            f"winning_orientation={profile['winning_orientation']}, "
            f"skill_level={profile['skill_level']}",
        ]

    return "\n".join(lines)


def build_simulation(
    agents: list[dict[str, Any]],
    model: Any,
    embedder: Any,
    max_steps: int,
    condition: InterventionSpec = CONTROL,
    group_id: str = "",
) -> simulation.Simulation:
    prefabs   = build_prefab_registry()
    instances = build_instances(agents, condition)
    premise   = build_premise(agents, condition, group_id=group_id)

    config = prefab_lib.Config(
        default_premise=premise,
        default_max_steps=max_steps,
        prefabs=prefabs,
        instances=instances,
    )

    return simulation.Simulation(
        config=config,
        model=model,
        embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Model / embedder factory
# ---------------------------------------------------------------------------

def get_model_and_embedder() -> tuple[Any, Any]:
    from sentence_transformers import SentenceTransformer
    from safe_gemini_model import SafeGeminiModel

    model = SafeGeminiModel(
        api_key=os.environ["GEMINI_API_KEY"],
        model_name="gemini-2.5-flash",
    )

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embedder(text: str) -> list[float]:
        return st_model.encode(text).tolist()

    return model, embedder


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def print_group(
    agents: list[dict[str, Any]],
    group_id: str,
    condition: InterventionSpec,
) -> None:
    print(f"\n=== GROUP {group_id} [{condition.label}] ===")
    for i, profile in enumerate(agents):
        b5 = get_big_five(profile)
        print(f"\n  {make_agent_name(i)} ({profile['profile_id']})")
        print(
            f"    O={b5['openness']}, C={b5['conscientiousness']}, "
            f"E={b5['extraversion']}, A={b5['agreeableness']}, "
            f"N={b5['neuroticism']}"
        )
        print(
            f"    group_work_preference={profile['group_work_preference']}, "
            f"winning_orientation={profile['winning_orientation']}, "
            f"skill_level={profile['skill_level']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single CS group-project simulation trial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--group_index",
        type=int,
        default=0,
        help="0-based index into simulation_groups.jsonl (0-99).",
    )
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument(
        "--condition",
        type=str,
        default="control",
        choices=ALL_CONDITION_NAMES,
        help="Simulation condition to run.",
    )
    parser.add_argument(
        "--groups_path",
        type=str,
        default="",
        help="Override path to simulation_groups.jsonl.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Optional checkpoint directory.",
    )
    args = parser.parse_args()

    condition   = REGISTRY[args.condition]
    groups_path = Path(args.groups_path) if args.groups_path else SIMULATION_GROUPS_PATH

    groups   = load_simulation_groups(groups_path)
    agents   = get_group_agents(groups, args.group_index)
    group_id = groups[args.group_index]["group_id"]

    print_group(agents, group_id, condition)

    model, embedder = get_model_and_embedder()

    sim = build_simulation(
        agents=agents,
        model=model,
        embedder=embedder,
        max_steps=args.max_steps,
        condition=condition,
        group_id=group_id,
    )

    if args.checkpoint_dir:
        results = sim.play(checkpoint_path=args.checkpoint_dir)
    else:
        results = sim.play()

    print(f"\n=== SIMULATION COMPLETE [{condition.label}] group={group_id} ===")
    print(results)


if __name__ == "__main__":
    main()
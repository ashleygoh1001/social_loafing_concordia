"""
cs_group_project_sim_intervention_weeklylogs.py

Intervention condition: structured weekly progress logs that each team
member must contribute to, outlining project milestones, individual and
group progress, and expected completion dates.

The intervention is injected at three levels so it stays prevalent
throughout the entire simulation:

1. PREMISE  — the world description the game master sees from step 1.
              Establishes that weekly logs exist, that the most recent
              log (week 6) is already in existence with partial entries,
              and that the week 7 log is due at the end of this meeting.
              This makes the log a concrete, present artifact rather than
              an abstract future obligation.

2. AGENT GOAL — each agent's goal is prepended with the log requirement
                so that every planning and action decision they make is
                framed by the accountability the log creates.

3. AGENT CONTEXT — the situational context each agent reasons from
                   includes the log norm, the log structure, and the
                   current week's log status on every step, so it is
                   active throughout the entire simulation rather than
                   only at initialisation.

Everything else is identical to cs_group_project_sim.py so the two
conditions are directly comparable.
"""

from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CONCORDIA_PARENT = _REPO_ROOT / "concordia"
if _LOCAL_CONCORDIA_PARENT.exists():
    sys.path.insert(0, str(_LOCAL_CONCORDIA_PARENT))

from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

import os
from sentence_transformers import SentenceTransformer


TRAIT_POOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "trait_pool.yaml"
)

# ---------------------------------------------------------------------------
# Intervention text — single source of truth, consistent across all levels
# ---------------------------------------------------------------------------

INTERVENTION_SHORT = (
    "Your team is required to submit a structured weekly progress log every week. "
    "Each log records: (1) project milestones and whether they were met, "
    "(2) each team member's individual progress and contributions that week, "
    "and (3) expected completion dates for all outstanding tasks. "
    "The week 7 log is due at the end of today's meeting and must include "
    "your name alongside what you personally contributed this week. "
    "Logs are reviewed by the course staff and are part of your grade."
)

INTERVENTION_NORM = (
    "The course staff requires the team to maintain a shared weekly progress log "
    "throughout the project. Every week, the team collectively fills in: "
    "(1) Milestones — which planned milestones were reached this week and which were missed; "
    "(2) Individual progress — a brief entry per team member describing what they "
    "personally completed, what they are currently working on, and any blockers; "
    "(3) Completion dates — updated expected delivery dates for each outstanding task. "
    "Logs from weeks 1 through 6 are already on file with the course staff. "
    "The week 6 log noted that architecture decisions were still pending and that "
    "task assignments had not yet been formally agreed upon. "
    "The week 7 log must be submitted by the end of today's meeting. "
    "Incomplete or missing individual entries are flagged to the instructor "
    "and reduce individual project grades."
)

INTERVENTION_GM = (
    "INTERVENTION ACTIVE: Structured weekly progress log requirement.\n"
    "The team must submit a weekly log covering milestones, individual member "
    "progress, and expected completion dates. Logs for weeks 1-6 are already filed. "
    "The week 7 log is due at the end of this meeting session.\n"
    "The week 6 log recorded: architecture decisions still pending; task assignments "
    "not yet formally agreed; Student contributions were uneven with some members "
    "not submitting individual entries.\n"
    "When narrating events, reflect that the log requirement creates concrete "
    "accountability: students know they must be able to report what they personally "
    "did this week. Students who have nothing to report are aware this will appear "
    "as a blank entry next to their name in the week 7 log. Students who have "
    "completed work have an incentive to make it visible in the log. "
    "Discussions about task assignment, deadlines, and milestones should be shaped "
    "by the fact that whatever is agreed must be written into the log today. "
    "The log functions as a coordination device: decisions made in this meeting "
    "become the milestone and completion-date entries that the team is held to "
    "in next week's log."
)

# The week 6 log as a concrete artifact agents can reference.
# Including it in the premise grounds the intervention in prior history
# rather than presenting it as an abstract future obligation.
WEEK6_LOG = """
--- WEEK 6 PROGRESS LOG (on file) ---
Milestones this week:
  - System requirements draft: MISSED (still in discussion)
  - GitHub repository setup: COMPLETE
  - Architecture decision: MISSED (pending team agreement)

Individual progress:
  - [entries varied; some members submitted brief notes, others left blank]

Outstanding tasks and expected completion dates:
  - Finalise architecture: target week 7
  - Assign coding modules to team members: target week 7
  - Set up CI/CD pipeline: target week 8
  - Core feature implementation: target weeks 8-9
  - Integration and bug fixing: target week 9
  - Final demo preparation: target week 10
--- END WEEK 6 LOG ---
"""


# ---------------------------------------------------------------------------
# Profile helpers (identical to base sim)
# ---------------------------------------------------------------------------

def load_trait_pool(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["profiles"]


def sample_profiles(
    profiles: list[dict[str, Any]],
    n_agents: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return rng.sample(profiles, n_agents)


def derive_behavioral_tags(profile: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    b5 = profile["big_five"]

    if (
        b5["conscientiousness"] == "low"
        and profile["group_work_preference"] == "low"
    ):
        tags.append("higher risk of disengaging from coordination")

    if (
        b5["conscientiousness"] == "high"
        and profile["skill_level"] in {"medium", "high"}
    ):
        tags.append("reliable at completing assigned work")

    if (
        b5["extraversion"] == "high"
        and profile["winning_orientation"] == "high"
    ):
        tags.append("likely to dominate discussion")

    if b5["agreeableness"] == "high":
        tags.append("tries to preserve group harmony")

    if b5["neuroticism"] == "high":
        tags.append("more reactive under time pressure")

    if (
        b5["openness"] == "high"
        and profile["skill_level"] in {"medium", "high"}
    ):
        tags.append("more willing to propose novel technical ideas")

    return tags


def profile_to_goal(profile: dict[str, Any]) -> str:
    # Intervention: prepend the log requirement to every agent's goal so
    # it shapes every planning and action decision they make.
    parts = [INTERVENTION_SHORT]

    parts.append("Help your team complete the CS group project successfully.")

    if profile["winning_orientation"] == "high":
        parts.append(
            "You want your contribution to be visible and recognized."
        )
    else:
        parts.append(
            "You care more about getting the project finished than competing with teammates."
        )

    if profile["group_work_preference"] == "high":
        parts.append(
            "You prefer active collaboration, discussion, and shared decision-making."
        )
    else:
        parts.append(
            "You prefer independent work and minimal coordination overhead."
        )

    skill = profile["skill_level"]
    if skill == "high":
        parts.append(
            "You are one of the technically strongest people on the team."
        )
    elif skill == "medium":
        parts.append(
            "You are reasonably capable and can contribute steadily."
        )
    else:
        parts.append(
            "You are less confident in your technical ability and may avoid difficult tasks."
        )

    return " ".join(parts)


def profile_to_context(profile: dict[str, Any]) -> str:
    b5 = profile["big_five"]
    # Intervention: append the log norm AND the week 6 log to the agent's
    # situational context. Both are present on every step of reasoning,
    # making the log a persistent feature of the agent's situation rather
    # than a one-time observation they might forget.
    return (
        f"You are a student in a 5-person CS group project. "
        f"Traits: C={b5['conscientiousness']}, E={b5['extraversion']}, "
        f"A={b5['agreeableness']}, N={b5['neuroticism']}. "
        f"Group work preference={profile['group_work_preference']}. "
        f"Winning orientation={profile['winning_orientation']}. "
        f"Skill level={profile['skill_level']}. "
        f"{INTERVENTION_NORM} "
        f"{WEEK6_LOG}"
    )


def make_agent_name(i: int) -> str:
    return f"Student_{i+1}"


def build_prefab_registry() -> dict[str, prefab_lib.Prefab]:
    return {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }


def build_instances(
    sampled_profiles: list[dict[str, Any]],
) -> list[prefab_lib.InstanceConfig]:
    instances: list[prefab_lib.InstanceConfig] = []

    for i, profile in enumerate(sampled_profiles):
        name = make_agent_name(i)
        instances.append(
            prefab_lib.InstanceConfig(
                prefab="basic_with_plan__Entity",
                role=prefab_lib.Role.ENTITY,
                params={
                    "name": name,
                    "goal": profile_to_goal(profile),
                    "context": profile_to_context(profile),
                },
            )
        )

    # Game master — log intervention injected via GM instructions so the
    # narrator enforces log-related consequences on every step it acts.
    instances.append(
        prefab_lib.InstanceConfig(
            prefab="generic__GameMaster",
            role=prefab_lib.Role.GAME_MASTER,
            params={
                "name": "course_staff",
                "acting_order": "random",
                "instructions": INTERVENTION_GM,
            },
        )
    )

    return instances


def build_premise(sampled_profiles: list[dict[str, Any]]) -> str:
    intro = [
        "It is week 7 of a 10-week undergraduate computer science group project.",
        "Five students have been assigned to build a working software system together for their final project.",
        "The team must decide on architecture, divide coding tasks, coordinate through GitHub, respond to bugs, and prepare a final demo.",
        "Team members differ in personality, motivation, technical ability, and comfort with collaboration.",
        "Social loafing, uneven task distribution, leadership struggles, and deadline stress may emerge over time.",
        "",
        # Intervention: log requirement is part of the world from step 0,
        # with the week 6 log as a concrete prior artifact.
        "INTERVENTION — WEEKLY PROGRESS LOG REQUIREMENT:",
        INTERVENTION_NORM,
        "",
        WEEK6_LOG,
        "",
        "The team is meeting after class to fill in the week 7 log and organize the project.",
    ]

    for i, profile in enumerate(sampled_profiles):
        b5 = profile["big_five"]
        intro.extend(
            [
                "",
                f"{make_agent_name(i)} profile summary:",
                f"- openness={b5['openness']}, conscientiousness={b5['conscientiousness']}, extraversion={b5['extraversion']}, agreeableness={b5['agreeableness']}, neuroticism={b5['neuroticism']}",
                f"- group_work_preference={profile['group_work_preference']}, winning_orientation={profile['winning_orientation']}, skill_level={profile['skill_level']}",
            ]
        )

    return "\n".join(intro)


def print_sampled_team(sampled_profiles: list[dict[str, Any]]) -> None:
    print("\n=== SAMPLED TEAM (WEEKLY LOGS INTERVENTION) ===")
    for i, profile in enumerate(sampled_profiles):
        b5 = profile["big_five"]
        print(f"\n{make_agent_name(i)} ({profile['profile_id']})")
        print(
            "  "
            f"O={b5['openness']}, "
            f"C={b5['conscientiousness']}, "
            f"E={b5['extraversion']}, "
            f"A={b5['agreeableness']}, "
            f"N={b5['neuroticism']}"
        )
        print(
            "  "
            f"group_work_preference={profile['group_work_preference']}, "
            f"winning_orientation={profile['winning_orientation']}, "
            f"skill_level={profile['skill_level']}"
        )


def build_simulation(
    sampled_profiles: list[dict[str, Any]],
    model: Any,
    embedder: Any,
    max_steps: int,
) -> simulation.Simulation:
    prefabs = build_prefab_registry()
    instances = build_instances(sampled_profiles)
    premise = build_premise(sampled_profiles)

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


def get_model_and_embedder() -> tuple[Any, Any]:
    from safe_gemini_model import SafeGeminiModel

    model = SafeGeminiModel(
        api_key=os.environ["GEMINI_API_KEY"],
        model_name="gemini-2.5-flash",
    )

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embedder(text: str) -> list[float]:
        return st_model.encode(text).tolist()

    return model, embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    if args.n_agents != 5:
        raise ValueError("This simulation is currently set up for exactly 5 agents.")

    profiles = load_trait_pool(TRAIT_POOL_PATH)
    sampled_profiles = sample_profiles(
        profiles=profiles,
        n_agents=args.n_agents,
        seed=args.seed,
    )

    print_sampled_team(sampled_profiles)

    model, embedder = get_model_and_embedder()

    sim = build_simulation(
        sampled_profiles=sampled_profiles,
        model=model,
        embedder=embedder,
        max_steps=args.max_steps,
    )

    if args.checkpoint_dir:
        results = sim.play(checkpoint_path=args.checkpoint_dir)
    else:
        results = sim.play()

    print("\n=== SIMULATION COMPLETE (WEEKLY LOGS INTERVENTION) ===")
    print(results)


if __name__ == "__main__":
    main()
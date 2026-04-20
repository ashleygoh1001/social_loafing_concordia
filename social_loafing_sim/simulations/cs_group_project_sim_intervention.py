"""
cs_group_project_sim_intervention.py

Intervention condition: individual contributions are closely tracked and
publicized so members have a greater sense of personal achievement.

The intervention is injected at three levels so it stays prevalent
throughout the entire simulation:

1. PREMISE  — the world description the game master sees from step 1.
              States that a contribution-tracking system is active and
              that each student's work is visible to the whole team and
              the course staff.

2. AGENT GOAL — each agent's goal paragraph is prepended with a sentence
                making salient that their individual contributions are
                being recorded and will be publicized. This shapes every
                planning and action decision the agent makes.

3. AGENT CONTEXT — the situational context each agent reasons from
                   includes the tracking norm, so it is present in every
                   step of their reasoning, not just initialisation.

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
# Intervention text — single source of truth so it is consistent everywhere
# ---------------------------------------------------------------------------

INTERVENTION_SHORT = (
    "Your individual contributions to this project are being closely tracked "
    "and will be publicized to your team and the course staff. "
    "What you specifically do — or do not do — will be visible to everyone."
)

INTERVENTION_NORM = (
    "The course staff has introduced a contribution-tracking system for this project. "
    "Every team member's individual contributions — commits, tasks completed, "
    "participation in discussions, and overall effort — are recorded automatically "
    "and displayed on a shared dashboard visible to all team members and the instructor. "
    "Students who contribute more receive public recognition; those who contribute less "
    "are identifiable to the whole group. The dashboard is updated after every meeting."
)
'''
INTERVENTION_GM = (
    "INTERVENTION ACTIVE: Individual contribution tracking and publicization.\n"
    "The course staff tracks and publicizes each student's individual contributions "
    "to the project (commits, tasks, participation, effort). "
    "This information is visible to all team members and the instructor on a shared "
    "dashboard updated after every meeting. "
    "When narrating events, reflect that students are aware their individual efforts "
    "are being recorded and will be seen by everyone. "
    "Students who coast or disengage should feel the social pressure of visibility; "
    "students who contribute should feel personal recognition."
)
'''
INTERVENTION_GM = (
    "INTERVENTION ACTIVE: Task-visibility system (performance targets, "
    "communication procedures, problem-solving protocol).\n"
    "Every subtask has a named owner, explicit acceptance criteria, and a "
    "due date on a shared board visible to all team members and the "
    "instructor. Members are required to post standup updates at every "
    "meeting and once daily asynchronously. When a member is blocked, "
    "they must declare it and follow the escalation protocol rather than "
    "silently stalling.\n"
    "When narrating events, reflect that task progress — and task absence — "
    "is structurally visible: members know whose subtasks are on track and "
    "whose are not, because the board and standups make this explicit. "
    "Students who fall behind should feel the weight of an expectation gap "
    "that is plain to the group; students who meet their targets should feel "
    "the clarity of knowing their contribution is unambiguous. "
    "Narrate how the communication and problem-solving norms shape whether "
    "blockers surface early or fester, and whether the team redistributes "
    "work effectively or lets tasks slip silently."
)


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
    # Intervention: prepend tracking salience to every agent's goal.
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
    # Intervention: append the tracking norm to the situational context so it
    # is present in every step of the agent's reasoning.
    return (
        f"You are a student in a 5-person CS group project. "
        f"Traits: C={b5['conscientiousness']}, E={b5['extraversion']}, "
        f"A={b5['agreeableness']}, N={b5['neuroticism']}. "
        f"Group work preference={profile['group_work_preference']}. "
        f"Winning orientation={profile['winning_orientation']}. "
        f"Skill level={profile['skill_level']}. "
        f"{INTERVENTION_NORM}"
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

    # Game master — intervention injected via its instructions
    instances.append(
        prefab_lib.InstanceConfig(
            prefab="generic__GameMaster",
            role=prefab_lib.Role.GAME_MASTER,
            params={
                "name": "course_staff",
                "acting_order": "random",
                # The GM instructions field seeds the game master's reasoning
                # with the intervention on every step it acts.
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
        # Intervention: explicitly part of the world from step 0.
        "INTERVENTION — CONTRIBUTION TRACKING SYSTEM ACTIVE:",
        INTERVENTION_NORM,
        "",
        "The team is meeting after class to decide how to organize the project.",
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
    print("\n=== SAMPLED TEAM (INTERVENTION CONDITION) ===")
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

    print("\n=== SIMULATION COMPLETE (INTERVENTION) ===")
    print(results)


if __name__ == "__main__":
    main()
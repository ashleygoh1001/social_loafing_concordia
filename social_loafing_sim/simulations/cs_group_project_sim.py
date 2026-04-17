from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path
from typing import Any

import yaml

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
    parts = ["Help your team complete the CS group project successfully."]

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


def profile_to_context(profile):
    b5 = profile["big_five"]
    return (
        f"You are a student in a 5-person CS group project. "
        f"Traits: C={b5['conscientiousness']}, E={b5['extraversion']}, "
        f"A={b5['agreeableness']}, N={b5['neuroticism']}. "
        f"Group work preference={profile['group_work_preference']}. "
        f"Winning orientation={profile['winning_orientation']}. "
        f"Skill level={profile['skill_level']}."
    )


def make_agent_name(i: int) -> str:
    return f"Student_{i+1}"


def build_prefab_registry() -> dict[str, prefab_lib.Prefab]:
    return {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }


def build_instances(sampled_profiles: list[dict[str, Any]]) -> list[prefab_lib.InstanceConfig]:
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

    instances.append(
        prefab_lib.InstanceConfig(
            prefab="generic__GameMaster",
            role=prefab_lib.Role.GAME_MASTER,
            params={
                "name": "course_staff",
                "acting_order": "random",
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

def rate_loafing_with_llm(model, agent_name: str, entries: list) -> dict:
    transcript = "\n".join(str(e) for e in entries)
    prompt = f"""You are evaluating social loafing in a CS group project simulation.

Agent: {agent_name}
Their actions/statements during the simulation:
{transcript}

Rate this agent on the following (each 1-5, where 5 = maximum loafing):
1. Task avoidance (did they avoid taking on work?)
2. Free riding (did they let others do the heavy lifting?)
3. Disengagement (did they go silent or contribute minimally?)

Reply as JSON: {{"task_avoidance": int, "free_riding": int, "disengagement": int, "reasoning": str}}"""

    raw = model.sample_text(prompt)
    import json
    return json.loads(raw)


def print_sampled_team(sampled_profiles: list[dict[str, Any]]) -> None:
    print("\n=== SAMPLED TEAM ===")
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
    import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Optional checkpoint directory",
    )
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

    print("\n=== SIMULATION COMPLETE ===")
    print(results)


if __name__ == "__main__":
    main()
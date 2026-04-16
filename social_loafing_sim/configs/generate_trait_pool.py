from __future__ import annotations

import itertools
from pathlib import Path

import yaml

BIG_FIVE_LEVELS = ["low", "medium", "high"]
GROUP_WORK_PREFERENCE_LEVELS = ["low", "high"]
WINNING_ORIENTATION_LEVELS = ["low", "high"]
SKILL_LEVEL_LEVELS = ["low", "medium", "high"]

BIG_FIVE_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def build_profiles() -> list[dict]:
    profiles = []
    profile_num = 1

    for ocean_values in itertools.product(BIG_FIVE_LEVELS, repeat=5):
        big_five = dict(zip(BIG_FIVE_TRAITS, ocean_values))

        for group_pref in GROUP_WORK_PREFERENCE_LEVELS:
            for winning_orientation in WINNING_ORIENTATION_LEVELS:
                for skill_level in SKILL_LEVEL_LEVELS:
                    profiles.append(
                        {
                            "profile_id": f"profile_{profile_num:04d}",
                            "big_five": big_five,
                            "group_work_preference": group_pref,
                            "winning_orientation": winning_orientation,
                            "skill_level": skill_level,
                        }
                    )
                    profile_num += 1

    return profiles


def main() -> None:
    profiles = build_profiles()

    output = {
        "metadata": {
            "description": "All possible profiles for Concordia CS group-project agents",
            "num_profiles": len(profiles),
            "big_five_levels": BIG_FIVE_LEVELS,
            "group_work_preference_levels": GROUP_WORK_PREFERENCE_LEVELS,
            "winning_orientation_levels": WINNING_ORIENTATION_LEVELS,
            "skill_level_levels": SKILL_LEVEL_LEVELS,
        },
        "profiles": profiles,
    }

    out_path = Path(__file__).with_name("trait_pool.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False)

    print(f"Wrote {len(profiles)} profiles to {out_path}")


if __name__ == "__main__":
    main()
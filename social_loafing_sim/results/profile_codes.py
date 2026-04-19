"""
profile_codes.py

Shared utility for converting a profile_id into an 8-digit trait code.

Digit mapping:
  1: Openness            low=1  medium=2  high=3
  2: Conscientiousness   low=1  medium=2  high=3
  3: Extraversion        low=1  medium=2  high=3
  4: Agreeableness       low=1  medium=2  high=3
  5: Neuroticism         low=1  medium=2  high=3
  6: Group work pref     low=1  high=2
  7: Winning orientation low=1  high=2
  8: Skill level         low=1  medium=2  high=3

Example: profile with all-high Big Five, high group pref, low winning, medium skill
         → "33333213"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Encoding maps
_THREE_LEVEL = {"low": "1", "medium": "2", "high": "3"}
_TWO_LEVEL   = {"low": "1", "high": "2"}

TRAIT_POOL_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "trait_pool.yaml"
)


def profile_to_code(profile: dict[str, Any]) -> str:
    """Return the 8-digit code string for a profile dict."""
    b5 = profile["big_five"]
    return (
        _THREE_LEVEL[b5["openness"]]
        + _THREE_LEVEL[b5["conscientiousness"]]
        + _THREE_LEVEL[b5["extraversion"]]
        + _THREE_LEVEL[b5["agreeableness"]]
        + _THREE_LEVEL[b5["neuroticism"]]
        + _TWO_LEVEL[profile["group_work_preference"]]
        + _TWO_LEVEL[profile["winning_orientation"]]
        + _THREE_LEVEL[profile["skill_level"]]
    )


def load_code_lookup(trait_pool_path: Path = TRAIT_POOL_PATH) -> dict[str, str]:
    """
    Load trait_pool.yaml and return a dict mapping profile_id → 8-digit code.
    E.g. {"profile_0001": "11111111", "profile_0002": "11111112", ...}
    """
    with open(trait_pool_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {p["profile_id"]: profile_to_code(p) for p in data["profiles"]}
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SelfPlayRewardConfig:
    correct_guess: float = 0.5
    hit_neutral: float = -0.1
    hit_opponent: float = -0.5
    hit_assassin: float = -5.0
    repeat_guess: float = -0.3
    invalid_guess: float = -0.3

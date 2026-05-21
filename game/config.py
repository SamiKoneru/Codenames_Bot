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
    # Applied by the runner (not the engine) when a spymaster emits an illegal
    # clue and forfeits the turn. Parallels invalid_guess on the guesser side.
    illegal_clue: float = -1.0

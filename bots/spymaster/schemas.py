"""Small dataclasses for spymaster inputs/outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SpymasterConfig:
    # Heuristic weights used in clue scoring.
    w_team_mean: float = 1.0
    w_team_max: float = 0.3
    w_opp_max: float = 1.0
    w_neutral_max: float = 0.6
    w_assassin: float = 2.5

    # Guess suggestion thresholds for the chosen clue.
    team_sim_threshold: float = 0.30
    safety_margin: float = 0.05


@dataclass
class ClueChoice:
    clue_word: str
    clue_index: int
    score: float
    suggested_guess_indices: List[int] = field(default_factory=list)
    debug: Dict[str, float] = field(default_factory=dict)

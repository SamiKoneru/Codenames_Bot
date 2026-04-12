"""Spymaster prompt construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from game.engine import ASSASSIN, NEUTRAL, TEAM_A, TEAM_B
from bots.spymaster.schemas import SpymasterConfig


def _relative_label_name(label: int, team: int) -> str:
    if label == team:
        return "ally"
    if label == ASSASSIN:
        return "assassin"
    if label == NEUTRAL:
        return "neutral"
    return "opponent"


def build_spymaster_prompt(observation: Any, cfg: Optional[SpymasterConfig] = None) -> str:
    cfg = cfg or SpymasterConfig()
    remaining = (
        observation.team_a_remaining
        if observation.team == TEAM_A
        else observation.team_b_remaining
    )
    board_lines = []
    for idx, word in enumerate(observation.board_words):
        visibility = "revealed" if observation.revealed[idx] else "hidden"
        owner = _relative_label_name(observation.labels[idx], observation.team)
        board_lines.append(f"{idx:02d}. {word} [{owner}, {visibility}]")

    return "\n".join(
        [
            "You are a Codenames spymaster.",
            "Return exactly one JSON object with keys clue_word and guess_limit.",
            'Example: {"clue_word": "animal", "guess_limit": 2}',
            "Use one lowercase token for clue_word and choose guess_limit between 1 "
            f"and {max(1, remaining)}.",
            "Heuristic hints:",
            (
                f"- prefer ally coverage; weights team_mean={cfg.w_team_mean}, "
                f"team_max={cfg.w_team_max}"
            ),
            (
                f"- avoid danger; weights opp_max={cfg.w_opp_max}, "
                f"neutral_max={cfg.w_neutral_max}, assassin={cfg.w_assassin}"
            ),
            (
                f"- safety_margin={cfg.safety_margin}, "
                f"team_sim_threshold={cfg.team_sim_threshold}"
            ),
            f"Turn index: {observation.turn_index}",
            f"Active team: {'TEAM_A' if observation.active_team == TEAM_A else 'TEAM_B'}",
            f"Your team: {'TEAM_A' if observation.team == TEAM_A else 'TEAM_B'}",
            f"Remaining ally cards: {remaining}",
            "Board:",
            *board_lines,
        ]
    )

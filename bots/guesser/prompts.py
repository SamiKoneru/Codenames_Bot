"""Guesser prompt construction helpers."""

from __future__ import annotations

from typing import Any

from game.engine import TEAM_A


def build_guesser_prompt(observation: Any) -> str:
    visible_board = []
    for idx, word in enumerate(observation.board_words):
        state = "revealed" if observation.revealed[idx] else "hidden"
        visible_board.append(f"{idx:02d}. {word} [{state}]")

    return "\n".join(
        [
            "You are a Codenames guesser.",
            "Return exactly one JSON object.",
            'Guess format: {"guess_word": "ocean", "end_turn": false}',
            'Stop format: {"end_turn": true}',
            f"Clue word: {observation.clue_word}",
            f"Guess limit this turn: {observation.guess_limit}",
            f"Guesses already made this turn: {', '.join(observation.guesses_made) or 'none'}",
            f"Turn index: {observation.turn_index}",
            f"Active team: {'TEAM_A' if observation.active_team == TEAM_A else 'TEAM_B'}",
            f"Your team: {'TEAM_A' if observation.team == TEAM_A else 'TEAM_B'}",
            f"Remaining TEAM_A cards: {observation.team_a_remaining}",
            f"Remaining TEAM_B cards: {observation.team_b_remaining}",
            "Public board:",
            *visible_board,
        ]
    )

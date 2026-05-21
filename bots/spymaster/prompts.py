"""Spymaster prompt construction helpers."""

from __future__ import annotations

from typing import Any

from game.engine import ASSASSIN, NEUTRAL, TEAM_A
from bots.spymaster.validation import (
    REASON_NON_ALPHA,
    REASON_REPEAT,
    REASON_SUBSTRING,
    ClueValidationResult,
)


def _relative_label_name(label: int, team: int) -> str:
    if label == team:
        return "ally"
    if label == ASSASSIN:
        return "assassin"
    if label == NEUTRAL:
        return "neutral"
    return "opponent"


def format_clue_retry_feedback(clue_word: str, result: ClueValidationResult) -> str:
    """Turn a failed validation into a correction note for the next attempt.

    The feedback is appended to the spymaster prompt so a model that produced an
    illegal clue gets an in-context reason to change it.
    """
    if result.reason == REASON_NON_ALPHA:
        return (
            f"Your previous clue '{clue_word}' was rejected: a clue must be a single "
            "alphabetic word with no spaces, hyphens, or digits. Try a different word."
        )
    if result.reason == REASON_SUBSTRING:
        return (
            f"Your previous clue '{clue_word}' was rejected because it shares a "
            f"substring with the board word '{result.offending_word}'. Codenames "
            "clues may not be part of (or contain) any word on the board. Pick a "
            "different word."
        )
    if result.reason == REASON_REPEAT:
        return (
            f"Your previous clue '{clue_word}' has already been used in this game. "
            "Pick a new word that has not been given as a clue yet."
        )
    return (
        f"Your previous clue '{clue_word}' was rejected as invalid. "
        "Try a different single-word clue."
    )


def build_spymaster_prompt(observation: Any, retry_feedback: str = "") -> str:
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

    past_clues = list(getattr(observation, "past_clues", []) or [])
    past_clues_line = (
        "Previously used clues (do not reuse): " + ", ".join(past_clues)
        if past_clues
        else "Previously used clues: none"
    )

    lines = [
        "You are a Codenames spymaster.",
        "Return exactly one JSON object with keys clue_word and guess_limit.",
        'Example: {"clue_word": "animal", "guess_limit": 2}',
        "Use one lowercase token for clue_word and choose guess_limit between 1 "
        f"and {max(1, remaining)}.",
        "Clue rules: one alphabetic word only; it may not be part of, or contain, "
        "any word on the board; it may not repeat an earlier clue.",
        "Strategy: pick a clue that links as many of your ally words as possible "
        "while avoiding any link to opponent, neutral, or (especially) assassin "
        "words. Set guess_limit to how many ally words your clue covers.",
        f"Turn index: {observation.turn_index}",
        f"Active team: {'TEAM_A' if observation.active_team == TEAM_A else 'TEAM_B'}",
        f"Your team: {'TEAM_A' if observation.team == TEAM_A else 'TEAM_B'}",
        f"Remaining ally cards: {remaining}",
        past_clues_line,
        "Board:",
        *board_lines,
    ]

    if retry_feedback:
        lines.append("")
        lines.append(retry_feedback)

    return "\n".join(lines)

"""Codenames clue legality validation.

Pure functions — no I/O, no model calls. Imported by spymaster policies that
want to enforce official Codenames rules on generated clues, and used by the
self-play runner to apply reward penalties for illegal clues.

The rules enforced here mirror standard tournament rules:

- The clue must be a single alphabetic token (no spaces, hyphens, digits, etc.).
- The clue must not share a substring relationship with any unrevealed board
  word. ``AGE`` is illegal if ``AGENT`` is on the board; ``AGENT`` is illegal
  if ``AGE`` is on the board. Revealed cards do not block clues.
- The clue must not have been used earlier in this game by either team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


REASON_OK = "ok"
REASON_EMPTY = "empty"
REASON_NON_ALPHA = "non_alpha"
REASON_SUBSTRING = "substring"
REASON_REPEAT = "repeat"


@dataclass
class ClueValidationResult:
    is_legal: bool
    reason: str = REASON_OK
    offending_word: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_legal": self.is_legal,
            "reason": self.reason,
            "offending_word": self.offending_word,
        }


def validate_clue(
    clue: Optional[str],
    board_words: List[str],
    revealed: List[bool],
    past_clues: Optional[Iterable[str]] = None,
) -> ClueValidationResult:
    """Return whether ``clue`` is a legal Codenames clue against this board.

    Args:
        clue: The candidate clue word. Compared case-insensitively. ``None`` or
            empty strings are treated as the empty rejection.
        board_words: All words currently on the board, in board order.
        revealed: Parallel boolean list; True if the card has been flipped.
            Revealed cards do not constrain future clues.
        past_clues: Optional iterable of previously used clue words (both teams).

    Returns:
        ClueValidationResult with ``is_legal`` plus a machine-readable
        ``reason`` and the ``offending_word`` that triggered rejection
        (empty string when legal).
    """
    if clue is None:
        return ClueValidationResult(False, REASON_EMPTY)

    normalized = clue.strip().lower()
    if not normalized:
        return ClueValidationResult(False, REASON_EMPTY)

    if not normalized.isalpha():
        return ClueValidationResult(False, REASON_NON_ALPHA, normalized)

    # Substring check both ways, against unrevealed board words only.
    for word, is_revealed in zip(board_words, revealed):
        if is_revealed:
            continue
        word_lower = word.strip().lower()
        if not word_lower:
            continue
        if normalized in word_lower or word_lower in normalized:
            return ClueValidationResult(False, REASON_SUBSTRING, word_lower)

    if past_clues:
        for past in past_clues:
            if past and past.strip().lower() == normalized:
                return ClueValidationResult(
                    False, REASON_REPEAT, past.strip().lower()
                )

    return ClueValidationResult(True, REASON_OK)

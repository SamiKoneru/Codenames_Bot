"""Tests for bots/spymaster/validation.py (pure clue legality checks)."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.spymaster.validation import (
    REASON_EMPTY,
    REASON_NON_ALPHA,
    REASON_OK,
    REASON_REPEAT,
    REASON_SUBSTRING,
    ClueValidationResult,
    validate_clue,
)


BOARD = ["apple", "river", "chair", "cloud", "stone", "agent"]
ALL_HIDDEN = [False] * len(BOARD)


class TestValidateClue(unittest.TestCase):
    # -- legal cases --

    def test_legal_clue(self):
        result = validate_clue("fruit", BOARD, ALL_HIDDEN)
        self.assertTrue(result.is_legal)
        self.assertEqual(result.reason, REASON_OK)
        self.assertEqual(result.offending_word, "")

    def test_unrelated_past_clues_ignored(self):
        result = validate_clue("fruit", BOARD, ALL_HIDDEN, past_clues=["mountain", "ocean"])
        self.assertTrue(result.is_legal)

    def test_revealed_word_does_not_block(self):
        # Once "river" is revealed, a clue equal to it becomes legal again.
        revealed = [False, True, False, False, False, False]
        result = validate_clue("river", BOARD, revealed)
        self.assertTrue(result.is_legal)

    # -- empty / missing --

    def test_empty_string_rejected(self):
        result = validate_clue("", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_EMPTY)

    def test_whitespace_only_rejected(self):
        result = validate_clue("   ", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_EMPTY)

    def test_none_rejected(self):
        result = validate_clue(None, BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_EMPTY)

    # -- non-alphabetic --

    def test_digit_rejected(self):
        result = validate_clue("agent7", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_NON_ALPHA)

    def test_hyphen_rejected(self):
        result = validate_clue("ice-cream", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_NON_ALPHA)

    def test_space_rejected(self):
        result = validate_clue("two words", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_NON_ALPHA)

    # -- substring rules --

    def test_exact_match_rejected(self):
        result = validate_clue("river", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_SUBSTRING)
        self.assertEqual(result.offending_word, "river")

    def test_case_insensitive_match_rejected(self):
        result = validate_clue("RIVER", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_SUBSTRING)

    def test_clue_substring_of_board_word_rejected(self):
        # "age" is a substring of "agent"
        result = validate_clue("age", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_SUBSTRING)
        self.assertEqual(result.offending_word, "agent")

    def test_board_word_substring_of_clue_rejected(self):
        # "agents" contains the board word "agent" as a substring
        result = validate_clue("agents", BOARD, ALL_HIDDEN)
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_SUBSTRING)
        self.assertEqual(result.offending_word, "agent")

    # -- repeats --

    def test_repeat_clue_rejected(self):
        result = validate_clue("fruit", BOARD, ALL_HIDDEN, past_clues=["fruit"])
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_REPEAT)
        self.assertEqual(result.offending_word, "fruit")

    def test_repeat_clue_case_insensitive(self):
        result = validate_clue("Fruit", BOARD, ALL_HIDDEN, past_clues=["FRUIT"])
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_REPEAT)

    def test_substring_takes_priority_over_repeat(self):
        # If both substring and repeat would fire, substring fires first.
        result = validate_clue("river", BOARD, ALL_HIDDEN, past_clues=["river"])
        self.assertFalse(result.is_legal)
        self.assertEqual(result.reason, REASON_SUBSTRING)

    # -- result serialization --

    def test_to_dict(self):
        result = ClueValidationResult(False, REASON_SUBSTRING, "agent")
        self.assertEqual(
            result.to_dict(),
            {"is_legal": False, "reason": "substring", "offending_word": "agent"},
        )


if __name__ == "__main__":
    unittest.main()

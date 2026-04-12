"""Tests for utils/ — extract_json_dict, discounted_returns, load_words."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.math import discounted_returns, extract_json_dict
from utils.words import load_words


# ---------------------------------------------------------------------------
# extract_json_dict
# ---------------------------------------------------------------------------

class TestExtractJsonDict(unittest.TestCase):
    def test_clean_json(self):
        result = extract_json_dict('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_nested_json(self):
        result = extract_json_dict('{"a": 1, "b": {"c": 2}}')
        self.assertEqual(result["b"]["c"], 2)

    def test_surrounding_text(self):
        result = extract_json_dict('Sure! {"clue_word": "ocean"} done.')
        self.assertEqual(result["clue_word"], "ocean")

    def test_leading_text_only(self):
        result = extract_json_dict('Here: {"x": 99}')
        self.assertEqual(result["x"], 99)

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json_dict("no json here at all")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            extract_json_dict("")

    def test_json_array_raises(self):
        with self.assertRaises(ValueError):
            extract_json_dict("[1, 2, 3]")

    def test_returns_dict_type(self):
        result = extract_json_dict('{"a": 1}')
        self.assertIsInstance(result, dict)

    def test_numeric_values(self):
        result = extract_json_dict('{"guess_limit": 3, "score": 0.5}')
        self.assertEqual(result["guess_limit"], 3)
        self.assertAlmostEqual(result["score"], 0.5)

    def test_boolean_values(self):
        result = extract_json_dict('{"end_turn": false}')
        self.assertFalse(result["end_turn"])


# ---------------------------------------------------------------------------
# discounted_returns
# ---------------------------------------------------------------------------

class TestDiscountedReturns(unittest.TestCase):
    def test_single_reward(self):
        result = discounted_returns([1.0], gamma=0.9)
        np.testing.assert_allclose(result, [1.0])

    def test_gamma_zero(self):
        # With gamma=0, each return is just its own reward
        result = discounted_returns([1.0, 2.0, 3.0], gamma=0.0)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_gamma_one(self):
        # With gamma=1, each return is sum of all future rewards
        result = discounted_returns([1.0, 1.0, 1.0], gamma=1.0)
        np.testing.assert_allclose(result, [3.0, 2.0, 1.0])

    def test_standard_case(self):
        # G_0 = 1 + 0.9*1 + 0.9^2*1 = 2.71
        result = discounted_returns([1.0, 1.0, 1.0], gamma=0.9)
        self.assertAlmostEqual(result[0], 1 + 0.9 + 0.81, places=5)
        self.assertAlmostEqual(result[1], 1 + 0.9, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_empty_returns_empty(self):
        result = discounted_returns([], gamma=0.9)
        self.assertEqual(len(result), 0)

    def test_output_dtype_float32(self):
        result = discounted_returns([1.0, 2.0], gamma=0.9)
        self.assertEqual(result.dtype, np.float32)

    def test_negative_rewards(self):
        result = discounted_returns([-1.0, -1.0], gamma=1.0)
        np.testing.assert_allclose(result, [-2.0, -1.0])

    def test_length_preserved(self):
        rewards = [0.5, -0.1, 0.0, 2.0]
        result = discounted_returns(rewards, gamma=0.99)
        self.assertEqual(len(result), len(rewards))


# ---------------------------------------------------------------------------
# load_words
# ---------------------------------------------------------------------------

class TestLoadWords(unittest.TestCase):
    def _write_words(self, lines):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write("\n".join(lines))
        tmp.close()
        return tmp.name

    def test_basic_load(self):
        path = self._write_words(["apple", "banana", "cherry"])
        words = load_words(path)
        self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_deduplication(self):
        path = self._write_words(["apple", "banana", "apple", "cherry"])
        words = load_words(path)
        self.assertEqual(words.count("apple"), 1)
        self.assertEqual(len(words), 3)

    def test_dedup_preserves_first_occurrence_order(self):
        path = self._write_words(["banana", "apple", "banana"])
        words = load_words(path)
        self.assertEqual(words[0], "banana")
        self.assertEqual(words[1], "apple")

    def test_lowercased(self):
        path = self._write_words(["Apple", "BANANA", "Cherry"])
        words = load_words(path)
        self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_strips_whitespace(self):
        path = self._write_words(["  apple  ", " banana", "cherry "])
        words = load_words(path)
        self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_skips_blank_lines(self):
        path = self._write_words(["apple", "", "banana", "  ", "cherry"])
        words = load_words(path)
        self.assertEqual(words, ["apple", "banana", "cherry"])

    def test_returns_list(self):
        path = self._write_words(["word"])
        self.assertIsInstance(load_words(path), list)


if __name__ == "__main__":
    unittest.main()

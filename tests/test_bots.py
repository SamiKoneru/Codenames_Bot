"""Tests for bots/ — prompts, random policies, and LLM policy parsing."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.engine import ASSASSIN, NEUTRAL, TEAM_A, TEAM_B
from bots.spymaster.policy import (
    ClueAction, LLMSpymasterPolicy, RandomSpymasterPolicy, SpymasterObservation,
)
from bots.spymaster.prompts import build_spymaster_prompt, _relative_label_name
from bots.spymaster.schemas import SpymasterConfig
from bots.guesser.policy import (
    GuessAction, LLMGuesserPolicy, RandomGuesserPolicy, GuesserObservation,
)
from bots.guesser.prompts import build_guesser_prompt
from llm.backends import StaticTextBackend


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_spymaster_obs(
    team=TEAM_A,
    board_words=None,
    labels=None,
    revealed=None,
    team_a_remaining=3,
    team_b_remaining=3,
):
    board_words = board_words or ["apple", "river", "chair", "cloud", "stone"]
    labels = labels or [TEAM_A, TEAM_A, TEAM_B, NEUTRAL, ASSASSIN]
    revealed = revealed or [False] * len(board_words)
    return SpymasterObservation(
        team=team,
        active_team=team,
        board_words=board_words,
        revealed=revealed,
        labels=labels,
        team_a_remaining=team_a_remaining,
        team_b_remaining=team_b_remaining,
        turn_index=0,
    )


def make_guesser_obs(
    team=TEAM_A,
    board_words=None,
    revealed=None,
    clue_word="fruit",
    guess_limit=2,
    guesses_made=None,
    team_a_remaining=3,
    team_b_remaining=3,
):
    board_words = board_words or ["apple", "river", "chair", "cloud", "stone"]
    revealed = revealed or [False] * len(board_words)
    return GuesserObservation(
        team=team,
        active_team=team,
        board_words=board_words,
        revealed=revealed,
        clue_word=clue_word,
        guess_limit=guess_limit,
        guesses_made=guesses_made or [],
        team_a_remaining=team_a_remaining,
        team_b_remaining=team_b_remaining,
        turn_index=0,
    )


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

class TestRelativeLabelName(unittest.TestCase):
    def test_own_team(self):
        self.assertEqual(_relative_label_name(TEAM_A, TEAM_A), "ally")
        self.assertEqual(_relative_label_name(TEAM_B, TEAM_B), "ally")

    def test_assassin(self):
        self.assertEqual(_relative_label_name(ASSASSIN, TEAM_A), "assassin")

    def test_neutral(self):
        self.assertEqual(_relative_label_name(NEUTRAL, TEAM_A), "neutral")

    def test_opponent(self):
        self.assertEqual(_relative_label_name(TEAM_B, TEAM_A), "opponent")
        self.assertEqual(_relative_label_name(TEAM_A, TEAM_B), "opponent")


class TestSpymasterPrompt(unittest.TestCase):
    def setUp(self):
        self.obs = make_spymaster_obs()
        self.prompt = build_spymaster_prompt(self.obs)

    def test_contains_instruction(self):
        self.assertIn("spymaster", self.prompt.lower())
        self.assertIn("clue_word", self.prompt)
        self.assertIn("guess_limit", self.prompt)

    def test_contains_all_board_words(self):
        for word in self.obs.board_words:
            self.assertIn(word, self.prompt)

    def test_contains_label_names(self):
        self.assertIn("ally", self.prompt)
        self.assertIn("opponent", self.prompt)
        self.assertIn("neutral", self.prompt)
        self.assertIn("assassin", self.prompt)

    def test_contains_team_info(self):
        self.assertIn("TEAM_A", self.prompt)

    def test_config_weights_shown(self):
        cfg = SpymasterConfig(w_assassin=9.9)
        prompt = build_spymaster_prompt(self.obs, cfg)
        self.assertIn("9.9", prompt)

    def test_revealed_cards_marked(self):
        obs = make_spymaster_obs(revealed=[True, False, False, False, False])
        prompt = build_spymaster_prompt(obs)
        self.assertIn("revealed", prompt)
        self.assertIn("hidden", prompt)


class TestGuesserPrompt(unittest.TestCase):
    def setUp(self):
        self.obs = make_guesser_obs()
        self.prompt = build_guesser_prompt(self.obs)

    def test_contains_instruction(self):
        self.assertIn("guesser", self.prompt.lower())
        self.assertIn("guess_word", self.prompt)

    def test_contains_clue(self):
        self.assertIn("fruit", self.prompt)

    def test_contains_guess_limit(self):
        self.assertIn("2", self.prompt)

    def test_contains_board_words(self):
        for word in self.obs.board_words:
            self.assertIn(word, self.prompt)

    def test_no_labels_in_guesser_prompt(self):
        # Guesser must NOT see ally/opponent/assassin labels
        self.assertNotIn("ally", self.prompt)
        self.assertNotIn("assassin", self.prompt)

    def test_guesses_made_shown(self):
        obs = make_guesser_obs(guesses_made=["apple"])
        prompt = build_guesser_prompt(obs)
        self.assertIn("apple", prompt)

    def test_no_guesses_shows_none(self):
        self.assertIn("none", self.prompt)


# ---------------------------------------------------------------------------
# RandomSpymasterPolicy
# ---------------------------------------------------------------------------

class TestRandomSpymasterPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = RandomSpymasterPolicy(seed=0, guess_limit=2)
        self.obs = make_spymaster_obs(team_a_remaining=3)

    def test_returns_clue_action(self):
        action = self.policy.choose_clue(self.obs)
        self.assertIsInstance(action, ClueAction)

    def test_clue_word_is_string(self):
        action = self.policy.choose_clue(self.obs)
        self.assertIsInstance(action.clue_word, str)
        self.assertTrue(len(action.clue_word) > 0)

    def test_guess_limit_clamped_to_remaining(self):
        obs = make_spymaster_obs(team_a_remaining=1)
        policy = RandomSpymasterPolicy(seed=0, guess_limit=10)
        action = policy.choose_clue(obs)
        self.assertEqual(action.guess_limit, 1)

    def test_guess_limit_at_least_one(self):
        obs = make_spymaster_obs(team_a_remaining=0)
        action = self.policy.choose_clue(obs)
        self.assertGreaterEqual(action.guess_limit, 1)

    def test_prompt_is_populated(self):
        action = self.policy.choose_clue(self.obs)
        self.assertTrue(len(action.prompt) > 0)

    def test_seeded_reproducible(self):
        p1 = RandomSpymasterPolicy(seed=99)
        p2 = RandomSpymasterPolicy(seed=99)
        a1 = p1.choose_clue(self.obs)
        a2 = p2.choose_clue(self.obs)
        self.assertEqual(a1.clue_word, a2.clue_word)

    def test_team_b_uses_b_remaining(self):
        obs = make_spymaster_obs(team=TEAM_B, team_b_remaining=2)
        policy = RandomSpymasterPolicy(seed=0, guess_limit=5)
        action = policy.choose_clue(obs)
        self.assertLessEqual(action.guess_limit, 2)


# ---------------------------------------------------------------------------
# RandomGuesserPolicy
# ---------------------------------------------------------------------------

class TestRandomGuesserPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = RandomGuesserPolicy(seed=0)
        self.obs = make_guesser_obs()

    def test_returns_guess_action(self):
        action = self.policy.choose_guess(self.obs)
        self.assertIsInstance(action, GuessAction)

    def test_guesses_hidden_word(self):
        action = self.policy.choose_guess(self.obs)
        self.assertFalse(action.end_turn)
        self.assertIn(action.guess_word, self.obs.board_words)

    def test_all_revealed_ends_turn(self):
        obs = make_guesser_obs(revealed=[True] * 5)
        action = self.policy.choose_guess(obs)
        self.assertTrue(action.end_turn)
        self.assertIsNone(action.guess_word)

    def test_end_turn_probability_one_always_ends(self):
        policy = RandomGuesserPolicy(seed=0, end_turn_probability=1.0)
        action = policy.choose_guess(self.obs)
        self.assertTrue(action.end_turn)

    def test_end_turn_probability_zero_never_ends(self):
        policy = RandomGuesserPolicy(seed=0, end_turn_probability=0.0)
        for _ in range(10):
            action = policy.choose_guess(self.obs)
            self.assertFalse(action.end_turn)

    def test_guess_word_is_lowercase(self):
        action = self.policy.choose_guess(self.obs)
        if action.guess_word:
            self.assertEqual(action.guess_word, action.guess_word.lower())


# ---------------------------------------------------------------------------
# LLMSpymasterPolicy
# ---------------------------------------------------------------------------

class TestLLMSpymasterPolicy(unittest.TestCase):
    def _policy(self, responses):
        return LLMSpymasterPolicy(StaticTextBackend(responses))

    def test_parses_clue_and_limit(self):
        policy = self._policy(['{"clue_word": "ocean", "guess_limit": 2}'])
        action = policy.choose_clue(make_spymaster_obs(team_a_remaining=3))
        self.assertEqual(action.clue_word, "ocean")
        self.assertEqual(action.guess_limit, 2)

    def test_clue_word_lowercased(self):
        policy = self._policy(['{"clue_word": "OCEAN", "guess_limit": 1}'])
        action = policy.choose_clue(make_spymaster_obs())
        self.assertEqual(action.clue_word, "ocean")

    def test_clue_word_takes_first_token(self):
        policy = self._policy(['{"clue_word": "two words", "guess_limit": 1}'])
        action = policy.choose_clue(make_spymaster_obs())
        self.assertEqual(action.clue_word, "two")

    def test_guess_limit_clamped_to_remaining(self):
        policy = self._policy(['{"clue_word": "hint", "guess_limit": 99}'])
        action = policy.choose_clue(make_spymaster_obs(team_a_remaining=2))
        self.assertLessEqual(action.guess_limit, 2)

    def test_guess_limit_minimum_one(self):
        policy = self._policy(['{"clue_word": "hint", "guess_limit": 0}'])
        action = policy.choose_clue(make_spymaster_obs(team_a_remaining=3))
        self.assertGreaterEqual(action.guess_limit, 1)

    def test_json_embedded_in_text(self):
        policy = self._policy(['Sure! {"clue_word": "river", "guess_limit": 1} done.'])
        action = policy.choose_clue(make_spymaster_obs())
        self.assertEqual(action.clue_word, "river")

    def test_missing_clue_word_raises(self):
        policy = self._policy(['{"guess_limit": 2}'])
        with self.assertRaises(ValueError):
            policy.choose_clue(make_spymaster_obs())


# ---------------------------------------------------------------------------
# LLMGuesserPolicy
# ---------------------------------------------------------------------------

class TestLLMGuesserPolicy(unittest.TestCase):
    def _policy(self, responses):
        return LLMGuesserPolicy(StaticTextBackend(responses))

    def test_parses_guess(self):
        policy = self._policy(['{"guess_word": "apple", "end_turn": false}'])
        action = policy.choose_guess(make_guesser_obs())
        self.assertEqual(action.guess_word, "apple")
        self.assertFalse(action.end_turn)

    def test_end_turn_true(self):
        policy = self._policy(['{"end_turn": true}'])
        action = policy.choose_guess(make_guesser_obs())
        self.assertTrue(action.end_turn)
        self.assertIsNone(action.guess_word)

    def test_guess_word_lowercased(self):
        policy = self._policy(['{"guess_word": "APPLE", "end_turn": false}'])
        action = policy.choose_guess(make_guesser_obs())
        self.assertEqual(action.guess_word, "apple")

    def test_empty_guess_word_becomes_end_turn(self):
        policy = self._policy(['{"guess_word": "  ", "end_turn": false}'])
        action = policy.choose_guess(make_guesser_obs())
        self.assertTrue(action.end_turn)
        self.assertIsNone(action.guess_word)

    def test_json_embedded_in_text(self):
        policy = self._policy(['Here is my guess: {"guess_word": "river", "end_turn": false}'])
        action = policy.choose_guess(make_guesser_obs())
        self.assertEqual(action.guess_word, "river")

    def test_raw_response_stored(self):
        raw = '{"guess_word": "chair", "end_turn": false}'
        policy = self._policy([raw])
        action = policy.choose_guess(make_guesser_obs())
        self.assertEqual(action.raw_response, raw)


if __name__ == "__main__":
    unittest.main()

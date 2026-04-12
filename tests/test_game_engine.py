"""Tests for game/engine.py — board setup, guessing, win conditions, turn management."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.engine import ASSASSIN, NEUTRAL, TEAM_A, TEAM_B, SelfPlayCodenamesGame
from game.config import SelfPlayRewardConfig


WORDS = [f"word{i}" for i in range(30)]


def make_game(seed=0, board_size=9, team_a_count=3, team_b_count=3):
    """Small board for fast tests."""
    return SelfPlayCodenamesGame(
        words=WORDS,
        board_size=board_size,
        team_a_count=team_a_count,
        team_b_count=team_b_count,
        seed=seed,
    )


class TestBoardSetup(unittest.TestCase):
    def test_label_counts(self):
        game = make_game()
        labels = game.labels
        self.assertEqual(labels.count(TEAM_A), 3)
        self.assertEqual(labels.count(TEAM_B), 3)
        self.assertEqual(labels.count(ASSASSIN), 1)
        self.assertEqual(labels.count(NEUTRAL), 9 - 3 - 3 - 1)

    def test_board_size(self):
        game = make_game(board_size=9)
        self.assertEqual(len(game.board_words), 9)
        self.assertEqual(len(game.labels), 9)
        self.assertEqual(len(game.revealed), 9)

    def test_all_hidden_at_start(self):
        game = make_game()
        self.assertFalse(any(game.revealed))

    def test_starts_on_team_a(self):
        game = make_game()
        self.assertEqual(game.active_team, TEAM_A)

    def test_too_few_words_raises(self):
        with self.assertRaises(ValueError):
            SelfPlayCodenamesGame(words=["a", "b"], board_size=9)

    def test_counts_too_large_raises(self):
        # 5 + 5 + 1 assassin = 11 > board_size 9
        with self.assertRaises(ValueError):
            SelfPlayCodenamesGame(words=WORDS, board_size=9, team_a_count=5, team_b_count=5)

    def test_reset_board_clears_state(self):
        game = make_game()
        game.start_turn(2)
        # Reveal a card, then reset
        team_a_idx = next(iter(game.team_a_indices))
        game.guess_word(team_a_idx)
        game.reset_board()
        self.assertFalse(any(game.revealed))
        self.assertFalse(game.game_over)
        self.assertIsNone(game.winner)
        self.assertEqual(game.active_team, TEAM_A)

    def test_seeded_boards_are_reproducible(self):
        g1 = make_game(seed=42)
        g2 = make_game(seed=42)
        self.assertEqual(g1.board_words, g2.board_words)
        self.assertEqual(g1.labels, g2.labels)

    def test_different_seeds_differ(self):
        g1 = make_game(seed=1)
        g2 = make_game(seed=2)
        # Very unlikely to be identical
        self.assertNotEqual(g1.board_words, g2.board_words)


class TestGuessOutcomes(unittest.TestCase):
    def setUp(self):
        self.game = make_game(seed=0)
        self.team_a_idx = next(iter(self.game.team_a_indices))
        self.team_b_idx = next(iter(self.game.team_b_indices))
        self.assassin_idx = self.game.assassin_index
        self.neutral_idx = next(
            i for i in range(self.game.board_size)
            if self.game.labels[i] == NEUTRAL
        )

    def _start(self, guesses=3):
        self.game.start_turn(guesses)

    def test_correct_team_hit(self):
        self._start()
        out = self.game.guess_word(self.team_a_idx)
        self.assertEqual(out["reward"], 0.5)
        self.assertEqual(out["reason"], "correct_team")
        self.assertTrue(self.game.revealed[self.team_a_idx])

    def test_correct_team_does_not_end_turn_immediately(self):
        self._start(guesses=2)
        out = self.game.guess_word(self.team_a_idx)
        self.assertFalse(out["turn_should_end"])
        self.assertTrue(self.game.turn_active)

    def test_neutral_hit_ends_turn(self):
        self._start()
        out = self.game.guess_word(self.neutral_idx)
        self.assertEqual(out["reward"], -0.1)
        self.assertEqual(out["reason"], "hit_neutral")
        self.assertTrue(out["turn_should_end"])
        self.assertFalse(self.game.turn_active)

    def test_opponent_hit_ends_turn(self):
        self._start()
        out = self.game.guess_word(self.team_b_idx)
        self.assertEqual(out["reward"], -0.5)
        self.assertEqual(out["reason"], "hit_opponent")
        self.assertTrue(out["turn_should_end"])

    def test_opponent_hit_removes_from_their_remaining(self):
        before = len(self.game.team_remaining[TEAM_B])
        self._start()
        self.game.guess_word(self.team_b_idx)
        self.assertEqual(len(self.game.team_remaining[TEAM_B]), before - 1)

    def test_assassin_ends_game(self):
        self._start()
        out = self.game.guess_word(self.assassin_idx)
        self.assertEqual(out["reward"], -5.0)
        self.assertEqual(out["reason"], "hit_assassin")
        self.assertTrue(out["game_over"])
        self.assertEqual(out["winner"], TEAM_B)  # other team wins
        self.assertTrue(self.game.game_over)

    def test_repeat_guess_penalised(self):
        self._start(guesses=3)
        self.game.guess_word(self.neutral_idx)   # ends turn
        self.game.start_turn(3)
        out = self.game.guess_word(self.neutral_idx)
        self.assertEqual(out["reward"], -0.3)
        self.assertEqual(out["reason"], "already_revealed")

    def test_invalid_index_penalised(self):
        self._start()
        out = self.game.guess_word(-1)
        self.assertEqual(out["reward"], -0.3)
        self.assertEqual(out["reason"], "invalid_index")

    def test_invalid_index_out_of_range(self):
        self._start()
        out = self.game.guess_word(999)
        self.assertEqual(out["reason"], "invalid_index")

    def test_custom_reward_config(self):
        cfg = SelfPlayRewardConfig(correct_guess=1.0, hit_neutral=-0.5)
        game = SelfPlayCodenamesGame(words=WORDS, board_size=9, team_a_count=3, team_b_count=3, reward_config=cfg, seed=0)
        neutral_idx = next(i for i in range(game.board_size) if game.labels[i] == NEUTRAL)
        game.start_turn(2)
        out = game.guess_word(neutral_idx)
        self.assertEqual(out["reward"], -0.5)


class TestWinConditions(unittest.TestCase):
    def _make_tiny(self, seed=5):
        """1v1 board: 1 team_a, 1 team_b, 1 assassin, rest neutral."""
        return SelfPlayCodenamesGame(
            words=WORDS, board_size=5, team_a_count=1, team_b_count=1, seed=seed
        )

    def test_team_a_wins_by_finding_all_cards(self):
        game = self._make_tiny()
        a_idx = next(iter(game.team_a_indices))
        game.start_turn(2)
        out = game.guess_word(a_idx)
        self.assertTrue(out["game_over"])
        self.assertEqual(out["winner"], TEAM_A)

    def test_team_b_wins_by_assassin(self):
        game = self._make_tiny()
        game.start_turn(2)
        out = game.guess_word(game.assassin_index)
        self.assertTrue(out["game_over"])
        self.assertEqual(out["winner"], TEAM_B)

    def test_team_b_wins_when_a_reveals_all_b_cards(self):
        game = self._make_tiny()
        b_idx = next(iter(game.team_b_indices))
        game.start_turn(2)
        out = game.guess_word(b_idx)
        self.assertTrue(out["game_over"])
        self.assertEqual(out["winner"], TEAM_B)
        self.assertEqual(out["reason"], "opponent_completed_by_reveal")


class TestTurnManagement(unittest.TestCase):
    def setUp(self):
        self.game = make_game(seed=7)

    def test_max_guesses_ends_turn(self):
        a_indices = list(self.game.team_a_indices)
        self.game.start_turn(1)
        out = self.game.guess_word(a_indices[0])
        self.assertTrue(out["turn_should_end"])
        self.assertEqual(out["reason"], "max_guesses_reached")

    def test_end_turn_switches_active_team(self):
        self.game.start_turn(1)
        self.game.end_turn()
        self.assertEqual(self.game.active_team, TEAM_B)

    def test_double_end_turn_returns_to_team_a(self):
        self.game.start_turn(1)
        self.game.end_turn()
        self.game.start_turn(1)
        self.game.end_turn()
        self.assertEqual(self.game.active_team, TEAM_A)

    def test_cannot_start_turn_twice(self):
        self.game.start_turn(1)
        with self.assertRaises(RuntimeError):
            self.game.start_turn(1)

    def test_cannot_guess_without_active_turn(self):
        with self.assertRaises(RuntimeError):
            self.game.guess_word(0)

    def test_cannot_start_turn_after_game_over(self):
        self.game.start_turn(3)
        self.game.guess_word(self.game.assassin_index)
        with self.assertRaises(RuntimeError):
            self.game.start_turn(1)


class TestStateViews(unittest.TestCase):
    def setUp(self):
        self.game = make_game(seed=3)

    def test_public_state_has_no_labels(self):
        state = self.game.get_public_state()
        self.assertNotIn("labels", state)

    def test_public_state_keys(self):
        state = self.game.get_public_state()
        for key in ("board_words", "revealed", "active_team", "team_a_remaining", "team_b_remaining", "game_over", "winner"):
            self.assertIn(key, state)

    def test_spymaster_state_has_labels(self):
        state = self.game.get_spymaster_state()
        self.assertIn("labels", state)
        self.assertEqual(len(state["labels"]), self.game.board_size)

    def test_remaining_counts_decrease_on_correct_guess(self):
        a_idx = next(iter(self.game.team_a_indices))
        before = self.game.get_public_state()["team_a_remaining"]
        self.game.start_turn(3)
        self.game.guess_word(a_idx)
        after = self.game.get_public_state()["team_a_remaining"]
        self.assertEqual(after, before - 1)


if __name__ == "__main__":
    unittest.main()

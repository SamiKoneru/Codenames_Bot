"""Tests for runner/evaluate.py — metric aggregation and matchup mechanics."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.engine import ASSASSIN, NEUTRAL, TEAM_A, TEAM_B
from bots.spymaster.policy import RandomSpymasterPolicy
from bots.guesser.policy import RandomGuesserPolicy
from runner.play import SelfPlayRolloutConfig
from runner.trajectory import EpisodeRecord, TrajectoryStep
from runner.evaluate import (
    Side,
    SideMetrics,
    accumulate_episode_metrics,
    evaluate_matchup,
)


WORDS = [f"word{i}" for i in range(40)]
SMALL_CFG = SelfPlayRolloutConfig(
    board_size=9, team_a_count=3, team_b_count=3, max_turns=20, gamma=0.95
)


def _spy_step(team, clue_word, illegal=False):
    meta = {}
    if illegal:
        meta["illegal_clue"] = True
    return TrajectoryStep(
        role="spymaster",
        team=team,
        observation={},
        prompt="",
        raw_response="",
        action={"clue_word": clue_word, "guess_limit": 1},
        metadata=meta,
    )


def _guess_step(team, label=None, reason="correct_team"):
    return TrajectoryStep(
        role="guesser",
        team=team,
        observation={},
        prompt="",
        raw_response="",
        action={},
        metadata={"reason": reason, "resolved_label": label},
    )


# ---------------------------------------------------------------------------
# SideMetrics computed properties
# ---------------------------------------------------------------------------

class TestSideMetricsProperties(unittest.TestCase):
    def test_rates_with_zero_denominators(self):
        m = SideMetrics()
        self.assertEqual(m.win_rate, 0.0)
        self.assertEqual(m.decisive_win_rate, 0.0)
        self.assertEqual(m.illegal_clue_rate, 0.0)
        self.assertEqual(m.clue_diversity, 0.0)
        self.assertEqual(m.guess_accuracy, 0.0)
        self.assertEqual(m.assassin_rate, 0.0)

    def test_win_and_decisive_rates(self):
        m = SideMetrics(games_played=10, wins=4, losses=4)
        self.assertEqual(m.draws, 2)
        self.assertAlmostEqual(m.win_rate, 0.4)
        self.assertAlmostEqual(m.decisive_win_rate, 0.5)

    def test_clue_diversity(self):
        m = SideMetrics()
        m.clues_given = 4
        m.clue_counter.update(["ocean", "ocean", "metal", "fruit"])
        self.assertAlmostEqual(m.clue_diversity, 3 / 4)


# ---------------------------------------------------------------------------
# accumulate_episode_metrics
# ---------------------------------------------------------------------------

class TestAccumulate(unittest.TestCase):
    def test_attributes_only_target_team(self):
        episode = EpisodeRecord(
            seed=0, board_words=[], labels=[], winner=TEAM_A,
            steps=[
                _spy_step(TEAM_A, "ocean"),
                _guess_step(TEAM_A, label=TEAM_A),
                _spy_step(TEAM_B, "metal"),
                _guess_step(TEAM_B, label=TEAM_B),
            ],
        )
        m = SideMetrics(name="cand")
        accumulate_episode_metrics(m, episode, TEAM_A)
        self.assertEqual(m.clues_given, 1)
        self.assertEqual(m.guesses_made, 1)
        self.assertEqual(m.correct_guesses, 1)
        self.assertEqual(m.wins, 1)
        self.assertEqual(m.losses, 0)

    def test_win_loss_draw(self):
        for winner, exp_w, exp_l in [(TEAM_A, 1, 0), (TEAM_B, 0, 1), (None, 0, 0)]:
            ep = EpisodeRecord(seed=0, board_words=[], labels=[], winner=winner, steps=[])
            m = SideMetrics()
            accumulate_episode_metrics(m, ep, TEAM_A)
            self.assertEqual((m.wins, m.losses), (exp_w, exp_l))
            self.assertEqual(m.games_played, 1)
            if winner is None:
                self.assertEqual(m.draws, 1)

    def test_guess_label_buckets(self):
        episode = EpisodeRecord(
            seed=0, board_words=[], labels=[], winner=None,
            steps=[
                _guess_step(TEAM_A, label=TEAM_A),                 # correct
                _guess_step(TEAM_A, label=TEAM_B),                 # opponent
                _guess_step(TEAM_A, label=NEUTRAL),                # neutral
                _guess_step(TEAM_A, label=ASSASSIN),               # assassin
                _guess_step(TEAM_A, label=None, reason="invalid_index"),   # wasted
                _guess_step(TEAM_A, reason="model_ended_turn"),    # not a guess
            ],
        )
        m = SideMetrics()
        accumulate_episode_metrics(m, episode, TEAM_A)
        self.assertEqual(m.guesses_made, 5)  # end-turn excluded
        self.assertEqual(m.correct_guesses, 1)
        self.assertEqual(m.opponent_hits, 1)
        self.assertEqual(m.neutral_hits, 1)
        self.assertEqual(m.assassin_hits, 1)
        self.assertEqual(m.wasted_guesses, 1)

    def test_illegal_clue_counted(self):
        episode = EpisodeRecord(
            seed=0, board_words=[], labels=[], winner=None,
            steps=[
                _spy_step(TEAM_A, "river", illegal=True),
                _spy_step(TEAM_A, "ocean"),
            ],
        )
        m = SideMetrics()
        accumulate_episode_metrics(m, episode, TEAM_A)
        self.assertEqual(m.clues_given, 2)
        self.assertEqual(m.illegal_clues, 1)
        self.assertAlmostEqual(m.illegal_clue_rate, 0.5)


# ---------------------------------------------------------------------------
# evaluate_matchup (integration with the real runner + random policies)
# ---------------------------------------------------------------------------

class TestEvaluateMatchup(unittest.TestCase):
    def _random_side(self, name, seed):
        return Side(
            name=name,
            spymaster=RandomSpymasterPolicy(seed=seed),
            guesser=RandomGuesserPolicy(seed=seed + 1),
        )

    def test_counts_reconcile(self):
        result = evaluate_matchup(
            words=WORDS,
            candidate=self._random_side("cand", 0),
            baseline=self._random_side("base", 100),
            num_games=10,
            rollout_config=SMALL_CFG,
            base_seed=0,
        )
        # Every game is counted once for each side.
        self.assertEqual(result.candidate.games_played, 10)
        self.assertEqual(result.baseline.games_played, 10)
        # A candidate win is a baseline loss and vice versa.
        self.assertEqual(result.candidate.wins, result.baseline.losses)
        self.assertEqual(result.candidate.losses, result.baseline.wins)
        # Wins + losses + draws partition the games.
        c = result.candidate
        self.assertEqual(c.wins + c.losses + c.draws, 10)

    def test_deterministic_under_seed(self):
        kwargs = dict(
            words=WORDS, num_games=8, rollout_config=SMALL_CFG, base_seed=0
        )
        r1 = evaluate_matchup(
            candidate=self._random_side("cand", 0),
            baseline=self._random_side("base", 100),
            **kwargs,
        )
        r2 = evaluate_matchup(
            candidate=self._random_side("cand", 0),
            baseline=self._random_side("base", 100),
            **kwargs,
        )
        self.assertEqual(r1.candidate.to_dict(), r2.candidate.to_dict())
        self.assertEqual(r1.baseline.to_dict(), r2.baseline.to_dict())

    def test_report_and_dict_shape(self):
        result = evaluate_matchup(
            words=WORDS,
            candidate=self._random_side("cand", 0),
            baseline=self._random_side("base", 100),
            num_games=4,
            rollout_config=SMALL_CFG,
        )
        report = result.report()
        self.assertIn("cand", report)
        self.assertIn("base", report)
        self.assertIn("win rate", report)
        d = result.to_dict()
        self.assertEqual(d["num_games"], 4)
        self.assertIn("candidate", d)
        self.assertIn("baseline", d)


if __name__ == "__main__":
    unittest.main()

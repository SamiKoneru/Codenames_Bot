"""Tests for runner/ — episode collection, trajectory recording, export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.engine import TEAM_A, TEAM_B
from bots.spymaster.policy import RandomSpymasterPolicy
from bots.guesser.policy import RandomGuesserPolicy
from runner.play import SelfPlayRolloutConfig, collect_self_play_episode, collect_self_play_episodes
from runner.trajectory import EpisodeRecord, TrajectoryStep
from runner.export import episode_to_jsonl_records, episode_to_sft_records, write_jsonl


WORDS = [f"word{i}" for i in range(40)]

SMALL_CFG = SelfPlayRolloutConfig(
    board_size=9,
    team_a_count=3,
    team_b_count=3,
    max_turns=20,
    gamma=0.95,
)


def make_episode(seed=0):
    return collect_self_play_episode(
        words=WORDS,
        spymaster_policy=RandomSpymasterPolicy(seed=seed),
        guesser_policy=RandomGuesserPolicy(seed=seed + 1),
        rollout_config=SMALL_CFG,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Episode structure
# ---------------------------------------------------------------------------

class TestEpisodeStructure(unittest.TestCase):
    def setUp(self):
        self.episode = make_episode(seed=0)

    def test_has_steps(self):
        self.assertTrue(len(self.episode.steps) > 0)

    def test_game_ends_or_truncated(self):
        # Either the game concluded naturally or hit max_turns
        ended = self.episode.winner is not None
        truncated = self.episode.summary.get("truncated", False)
        self.assertTrue(ended or truncated)

    def test_board_words_populated(self):
        self.assertEqual(len(self.episode.board_words), SMALL_CFG.board_size)

    def test_labels_populated(self):
        self.assertEqual(len(self.episode.labels), SMALL_CFG.board_size)

    def test_winner_is_valid_team_or_none(self):
        self.assertIn(self.episode.winner, (TEAM_A, TEAM_B, None))

    def test_steps_have_roles(self):
        roles = {step.role for step in self.episode.steps}
        self.assertTrue(roles.issubset({"spymaster", "guesser"}))

    def test_spymaster_steps_have_prompts(self):
        for step in self.episode.steps:
            if step.role == "spymaster":
                self.assertTrue(len(step.prompt) > 0)

    def test_guesser_steps_have_prompts(self):
        for step in self.episode.steps:
            if step.role == "guesser":
                self.assertTrue(len(step.prompt) > 0)

    def test_each_turn_starts_with_spymaster(self):
        # The first step is always spymaster
        self.assertEqual(self.episode.steps[0].role, "spymaster")

    def test_teams_are_valid(self):
        for step in self.episode.steps:
            self.assertIn(step.team, (TEAM_A, TEAM_B))

    def test_discounted_returns_assigned(self):
        # At least some steps should have non-zero discounted return
        returns = [step.discounted_return for step in self.episode.steps]
        self.assertTrue(any(r != 0.0 for r in returns))


# ---------------------------------------------------------------------------
# Outcome bonuses
# ---------------------------------------------------------------------------

class TestOutcomeBonuses(unittest.TestCase):
    def test_winner_gets_positive_bonus(self):
        episode = make_episode(seed=3)
        if episode.winner is None:
            return  # truncated — skip
        winner_steps = [s for s in episode.steps if s.team == episode.winner]
        if not winner_steps:
            return  # winner won via opponent hitting assassin before they took a turn
        last_winner_step = winner_steps[-1]
        self.assertIn("outcome_bonus", last_winner_step.metadata)
        self.assertGreater(last_winner_step.metadata["outcome_bonus"], 0)

    def test_loser_gets_negative_bonus(self):
        episode = make_episode(seed=3)
        if episode.winner is None:
            return
        loser = TEAM_B if episode.winner == TEAM_A else TEAM_A
        loser_steps = [s for s in episode.steps if s.team == loser]
        if not loser_steps:
            return
        last_loser_step = loser_steps[-1]
        self.assertIn("outcome_bonus", last_loser_step.metadata)
        self.assertLess(last_loser_step.metadata["outcome_bonus"], 0)


# ---------------------------------------------------------------------------
# Multiple episodes
# ---------------------------------------------------------------------------

class TestCollectMultipleEpisodes(unittest.TestCase):
    def test_returns_correct_count(self):
        episodes = collect_self_play_episodes(
            words=WORDS,
            spymaster_policy=RandomSpymasterPolicy(seed=0),
            guesser_policy=RandomGuesserPolicy(seed=1),
            num_episodes=5,
            rollout_config=SMALL_CFG,
            seed=0,
        )
        self.assertEqual(len(episodes), 5)

    def test_episodes_differ_by_seed(self):
        episodes = collect_self_play_episodes(
            words=WORDS,
            spymaster_policy=RandomSpymasterPolicy(seed=0),
            guesser_policy=RandomGuesserPolicy(seed=1),
            num_episodes=3,
            rollout_config=SMALL_CFG,
            seed=0,
        )
        boards = [ep.board_words for ep in episodes]
        # All three boards should differ (different seeds)
        self.assertNotEqual(boards[0], boards[1])
        self.assertNotEqual(boards[1], boards[2])

    def test_all_episodes_have_steps(self):
        episodes = collect_self_play_episodes(
            words=WORDS,
            spymaster_policy=RandomSpymasterPolicy(seed=0),
            guesser_policy=RandomGuesserPolicy(seed=1),
            num_episodes=3,
            rollout_config=SMALL_CFG,
            seed=10,
        )
        for ep in episodes:
            self.assertTrue(len(ep.steps) > 0)


# ---------------------------------------------------------------------------
# Export: episode_to_jsonl_records
# ---------------------------------------------------------------------------

class TestJsonlExport(unittest.TestCase):
    def setUp(self):
        self.episode = make_episode(seed=1)

    def test_record_count_matches_steps(self):
        records = episode_to_jsonl_records(self.episode)
        self.assertEqual(len(records), len(self.episode.steps))

    def test_records_have_required_keys(self):
        records = episode_to_jsonl_records(self.episode)
        required = {"role", "team", "prompt", "completion", "reward", "discounted_return", "winner"}
        for rec in records:
            self.assertTrue(required.issubset(rec.keys()))

    def test_role_filter_spymaster_only(self):
        records = episode_to_jsonl_records(self.episode, roles=["spymaster"])
        self.assertTrue(all(r["role"] == "spymaster" for r in records))

    def test_role_filter_guesser_only(self):
        records = episode_to_jsonl_records(self.episode, roles=["guesser"])
        self.assertTrue(all(r["role"] == "guesser" for r in records))

    def test_no_filter_returns_all(self):
        all_records = episode_to_jsonl_records(self.episode)
        spy_records = episode_to_jsonl_records(self.episode, roles=["spymaster"])
        guess_records = episode_to_jsonl_records(self.episode, roles=["guesser"])
        self.assertEqual(len(all_records), len(spy_records) + len(guess_records))

    def test_winner_field_consistent(self):
        records = episode_to_jsonl_records(self.episode)
        for rec in records:
            self.assertEqual(rec["winner"], self.episode.winner)


# ---------------------------------------------------------------------------
# Export: episode_to_sft_records
# ---------------------------------------------------------------------------

class TestSftExport(unittest.TestCase):
    def setUp(self):
        self.episode = make_episode(seed=2)

    def test_minimum_return_filters(self):
        all_records = episode_to_sft_records(self.episode, minimum_return=-999.0)
        filtered_records = episode_to_sft_records(self.episode, minimum_return=999.0)
        self.assertGreaterEqual(len(all_records), len(filtered_records))

    def test_all_records_meet_minimum(self):
        threshold = 0.0
        records = episode_to_sft_records(self.episode, minimum_return=threshold)
        for rec in records:
            self.assertGreaterEqual(rec["discounted_return"], threshold)

    def test_sft_records_have_required_keys(self):
        records = episode_to_sft_records(self.episode, minimum_return=-999.0)
        required = {"role", "team", "prompt", "completion", "discounted_return", "winner"}
        for rec in records:
            self.assertTrue(required.issubset(rec.keys()))

    def test_role_filter(self):
        records = episode_to_sft_records(self.episode, minimum_return=-999.0, roles=["spymaster"])
        self.assertTrue(all(r["role"] == "spymaster" for r in records))


# ---------------------------------------------------------------------------
# Export: write_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl(unittest.TestCase):
    def test_writes_correct_line_count(self):
        episode = make_episode(seed=5)
        records = episode_to_jsonl_records(episode)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/out.jsonl"
            write_jsonl(path, records)
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(records))

    def test_creates_parent_directories(self):
        episode = make_episode(seed=6)
        records = episode_to_jsonl_records(episode)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/nested/deep/out.jsonl"
            write_jsonl(path, records)
            self.assertTrue(Path(path).exists())

    def test_each_line_is_valid_json(self):
        import json
        episode = make_episode(seed=7)
        records = episode_to_jsonl_records(episode)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/out.jsonl"
            write_jsonl(path, records)
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                obj = json.loads(line)
                self.assertIsInstance(obj, dict)


# ---------------------------------------------------------------------------
# EpisodeRecord.assign_discounted_returns
# ---------------------------------------------------------------------------

class TestEpisodeRecordReturns(unittest.TestCase):
    def _make_record(self, rewards):
        steps = [
            TrajectoryStep(
                role="guesser",
                team=TEAM_A,
                observation={},
                prompt="",
                raw_response="",
                action={},
                reward=r,
            )
            for r in rewards
        ]
        return EpisodeRecord(seed=None, board_words=[], labels=[], winner=None, steps=steps)

    def test_returns_computed(self):
        record = self._make_record([1.0, 1.0, 1.0])
        record.assign_discounted_returns(gamma=1.0)
        self.assertAlmostEqual(record.steps[0].discounted_return, 3.0)
        self.assertAlmostEqual(record.steps[2].discounted_return, 1.0)

    def test_empty_steps_no_error(self):
        record = EpisodeRecord(seed=None, board_words=[], labels=[], winner=None)
        record.assign_discounted_returns(gamma=0.99)  # should not raise


if __name__ == "__main__":
    unittest.main()

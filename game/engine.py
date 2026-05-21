"""Minimal two-team Codenames game logic for self-play training."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from game.config import SelfPlayRewardConfig

TEAM_A = 0
TEAM_B = 1
NEUTRAL = 2
ASSASSIN = 3


class SelfPlayCodenamesGame:
    """Two-team, alternating-turn game mechanics (no models)."""

    def __init__(
        self,
        words: List[str],
        board_size: int = 25,
        team_a_count: int = 9,
        team_b_count: int = 8,
        reward_config: Optional[SelfPlayRewardConfig] = None,
        seed: Optional[int] = None,
    ):
        cleaned = [w.strip() for w in words if w and w.strip()]
        cleaned = list(dict.fromkeys(cleaned))
        if len(cleaned) < board_size:
            raise ValueError("Need at least board_size unique words.")
        if team_a_count + team_b_count + 1 > board_size:
            raise ValueError("team_a_count + team_b_count + assassin must fit on board.")

        self.words = cleaned
        self.board_size = board_size
        self.team_a_count = team_a_count
        self.team_b_count = team_b_count
        self.reward = reward_config or SelfPlayRewardConfig()
        self.rng = random.Random(seed)

        self.reset_board()

    def reset_board(self) -> Dict:
        self.board_words = self.rng.sample(self.words, self.board_size)

        idxs = list(range(self.board_size))
        self.rng.shuffle(idxs)

        self.labels = [NEUTRAL] * self.board_size
        self.team_a_indices = set(idxs[: self.team_a_count])
        b_start = self.team_a_count
        b_end = b_start + self.team_b_count
        self.team_b_indices = set(idxs[b_start:b_end])
        self.assassin_index = idxs[b_end]

        for i in self.team_a_indices:
            self.labels[i] = TEAM_A
        for i in self.team_b_indices:
            self.labels[i] = TEAM_B
        self.labels[self.assassin_index] = ASSASSIN

        self.revealed = [False] * self.board_size
        self.team_remaining = {
            TEAM_A: set(self.team_a_indices),
            TEAM_B: set(self.team_b_indices),
        }

        self.game_over = False
        self.winner: Optional[int] = None

        self.active_team = TEAM_A
        self.turn_active = False
        self.guesses_left = 0

        return self.get_public_state()

    def start_turn(self, max_guesses: int) -> None:
        if self.game_over:
            raise RuntimeError("Game is over.")
        if self.turn_active:
            raise RuntimeError("Turn already active.")
        if max_guesses <= 0:
            raise ValueError("max_guesses must be >= 1")
        self.turn_active = True
        self.guesses_left = int(max_guesses)

    def _other_team(self, team: int) -> int:
        return TEAM_B if team == TEAM_A else TEAM_A

    def guess_word(self, index: int) -> Dict:
        if not self.turn_active:
            raise RuntimeError("No active turn.")
        if self.game_over:
            raise RuntimeError("Game is over.")

        team = self.active_team
        other = self._other_team(team)

        out = {
            "team": team,
            "index": index,
            "label": None,
            "reward": 0.0,
            "turn_should_end": False,
            "game_over": False,
            "winner": None,
            "reason": "",
        }

        if index < 0 or index >= self.board_size:
            out["reward"] = self.reward.invalid_guess
            out["turn_should_end"] = True
            out["reason"] = "invalid_index"
            self.end_turn()
            return out

        if self.revealed[index]:
            out["reward"] = self.reward.repeat_guess
            out["turn_should_end"] = True
            out["reason"] = "already_revealed"
            self.end_turn()
            return out

        self.revealed[index] = True
        self.guesses_left -= 1

        lab = self.labels[index]
        out["label"] = lab

        if lab == team:
            self.team_remaining[team].discard(index)
            out["reward"] = self.reward.correct_guess
            out["reason"] = "correct_team"
        elif lab == other:
            # Revealed opponent cards count toward opponent completion.
            self.team_remaining[other].discard(index)
            out["reward"] = self.reward.hit_opponent
            out["turn_should_end"] = True
            out["reason"] = "hit_opponent"
        elif lab == NEUTRAL:
            out["reward"] = self.reward.hit_neutral
            out["turn_should_end"] = True
            out["reason"] = "hit_neutral"
        elif lab == ASSASSIN:
            out["reward"] = self.reward.hit_assassin
            out["turn_should_end"] = True
            out["reason"] = "hit_assassin"
            self.game_over = True
            self.winner = other

        if len(self.team_remaining[team]) == 0:
            self.game_over = True
            self.winner = team
            out["turn_should_end"] = True
            out["reason"] = "all_team_found"
        elif len(self.team_remaining[other]) == 0:
            # You just revealed opponent's last unrevealed team card.
            self.game_over = True
            self.winner = other
            out["turn_should_end"] = True
            out["reason"] = "opponent_completed_by_reveal"

        if self.guesses_left <= 0:
            out["turn_should_end"] = True
            if out["reason"] == "correct_team":
                out["reason"] = "max_guesses_reached"

        if out["turn_should_end"]:
            self.end_turn()

        out["game_over"] = self.game_over
        out["winner"] = self.winner
        return out

    def end_turn(self) -> None:
        self.turn_active = False
        if not self.game_over:
            self.active_team = self._other_team(self.active_team)

    def get_public_state(self) -> Dict:
        return {
            "board_words": list(self.board_words),
            "revealed": list(self.revealed),
            "active_team": self.active_team,
            "team_a_remaining": len(self.team_remaining[TEAM_A]),
            "team_b_remaining": len(self.team_remaining[TEAM_B]),
            "game_over": self.game_over,
            "winner": self.winner,
        }

    def get_spymaster_state(self) -> Dict:
        s = self.get_public_state()
        s["labels"] = list(self.labels)
        return s

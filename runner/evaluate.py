"""Head-to-head evaluation of Codenames bots.

The training scripts can produce a model, but on their own they give you no way
to answer the only question that matters: *is this bot any good?*  This module
plays a candidate bot against a baseline bot and reports win rate plus a set of
Codenames-specific diagnostics (assassin rate, guess accuracy, illegal-clue
rate, clue diversity).

A "side" is a (spymaster, guesser) pair.  To remove the first-mover / card-count
advantage that TEAM_A has, ``evaluate_matchup`` swaps which team the candidate
controls every other game and pairs consecutive games on the same board seed,
so the candidate and baseline each play both orientations of the same layout.

This module is deliberately policy-agnostic: it imports no model backend.  The
CLI in ``scripts/evaluate.py`` is responsible for constructing policies (random
baselines or HuggingFace-backed models) and handing them in.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from game.engine import ASSASSIN, NEUTRAL, TEAM_A, TEAM_B
from runner.play import SelfPlayRolloutConfig, collect_self_play_episode
from runner.trajectory import EpisodeRecord


@dataclass
class Side:
    """A complete bot: one spymaster policy and one guesser policy."""

    name: str
    spymaster: Any
    guesser: Any


@dataclass
class SideMetrics:
    """Aggregated outcomes for one side across an evaluation run."""

    name: str = ""
    games_played: int = 0
    wins: int = 0
    losses: int = 0

    # Spymaster diagnostics.
    clues_given: int = 0
    illegal_clues: int = 0
    clue_counter: Counter = field(default_factory=Counter)

    # Guesser diagnostics.
    guesses_made: int = 0
    correct_guesses: int = 0
    opponent_hits: int = 0
    neutral_hits: int = 0
    assassin_hits: int = 0
    wasted_guesses: int = 0  # invalid index or already-revealed card

    @property
    def draws(self) -> int:
        return self.games_played - self.wins - self.losses

    @property
    def win_rate(self) -> float:
        """Wins over all games played (draws count against you)."""
        return self.wins / self.games_played if self.games_played else 0.0

    @property
    def decisive_win_rate(self) -> float:
        """Wins over games that had a winner (ignores truncated games)."""
        decisive = self.wins + self.losses
        return self.wins / decisive if decisive else 0.0

    @property
    def illegal_clue_rate(self) -> float:
        return self.illegal_clues / self.clues_given if self.clues_given else 0.0

    @property
    def clue_diversity(self) -> float:
        """Fraction of clues that were unique (1.0 = never repeated a clue)."""
        return len(self.clue_counter) / self.clues_given if self.clues_given else 0.0

    @property
    def guess_accuracy(self) -> float:
        return self.correct_guesses / self.guesses_made if self.guesses_made else 0.0

    @property
    def assassin_rate(self) -> float:
        """Average assassin hits per game (lower is better)."""
        return self.assassin_hits / self.games_played if self.games_played else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 4),
            "decisive_win_rate": round(self.decisive_win_rate, 4),
            "clues_given": self.clues_given,
            "illegal_clues": self.illegal_clues,
            "illegal_clue_rate": round(self.illegal_clue_rate, 4),
            "clue_diversity": round(self.clue_diversity, 4),
            "guesses_made": self.guesses_made,
            "correct_guesses": self.correct_guesses,
            "guess_accuracy": round(self.guess_accuracy, 4),
            "opponent_hits": self.opponent_hits,
            "neutral_hits": self.neutral_hits,
            "assassin_hits": self.assassin_hits,
            "assassin_rate": round(self.assassin_rate, 4),
            "wasted_guesses": self.wasted_guesses,
        }


def accumulate_episode_metrics(
    metrics: SideMetrics, episode: EpisodeRecord, team: int
) -> None:
    """Fold one episode's results for ``team`` into ``metrics`` (in place)."""
    metrics.games_played += 1
    if episode.winner == team:
        metrics.wins += 1
    elif episode.winner is not None:
        metrics.losses += 1

    for step in episode.steps:
        if step.team != team:
            continue

        if step.role == "spymaster":
            metrics.clues_given += 1
            clue = str(step.action.get("clue_word", "")).strip().lower()
            if clue:
                metrics.clue_counter[clue] += 1
            if step.metadata.get("illegal_clue"):
                metrics.illegal_clues += 1

        elif step.role == "guesser":
            # Turn-ending non-guesses carry no resolved card; skip them.
            if step.metadata.get("reason") == "model_ended_turn":
                continue
            metrics.guesses_made += 1
            label = step.metadata.get("resolved_label")
            if label == team:
                metrics.correct_guesses += 1
            elif label == NEUTRAL:
                metrics.neutral_hits += 1
            elif label == ASSASSIN:
                metrics.assassin_hits += 1
            elif label is None:
                metrics.wasted_guesses += 1
            else:
                metrics.opponent_hits += 1


@dataclass
class MatchupResult:
    candidate: SideMetrics
    baseline: SideMetrics
    num_games: int
    rollout_config: SelfPlayRolloutConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_games": self.num_games,
            "board_size": self.rollout_config.board_size,
            "team_a_count": self.rollout_config.team_a_count,
            "team_b_count": self.rollout_config.team_b_count,
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
        }

    def report(self) -> str:
        cand, base = self.candidate, self.baseline
        rows = [
            ("win rate (all)", f"{cand.win_rate:.1%}", f"{base.win_rate:.1%}"),
            ("win rate (decisive)", f"{cand.decisive_win_rate:.1%}", f"{base.decisive_win_rate:.1%}"),
            ("wins / losses / draws",
             f"{cand.wins}/{cand.losses}/{cand.draws}",
             f"{base.wins}/{base.losses}/{base.draws}"),
            ("guess accuracy", f"{cand.guess_accuracy:.1%}", f"{base.guess_accuracy:.1%}"),
            ("assassin rate / game", f"{cand.assassin_rate:.3f}", f"{base.assassin_rate:.3f}"),
            ("illegal clue rate", f"{cand.illegal_clue_rate:.1%}", f"{base.illegal_clue_rate:.1%}"),
            ("clue diversity", f"{cand.clue_diversity:.1%}", f"{base.clue_diversity:.1%}"),
        ]
        name_w = max(len("metric"), *(len(r[0]) for r in rows))
        col_w = max(len(cand.name), len(base.name), 10)
        lines = [
            f"Matchup over {self.num_games} games "
            f"(board {self.rollout_config.board_size}, "
            f"{self.rollout_config.team_a_count}v{self.rollout_config.team_b_count})",
            f"{'metric':<{name_w}}  {cand.name:>{col_w}}  {base.name:>{col_w}}",
            f"{'-' * name_w}  {'-' * col_w}  {'-' * col_w}",
        ]
        for label, c_val, b_val in rows:
            lines.append(f"{label:<{name_w}}  {c_val:>{col_w}}  {b_val:>{col_w}}")
        return "\n".join(lines)


def evaluate_matchup(
    words: List[str],
    candidate: Side,
    baseline: Side,
    num_games: int = 50,
    rollout_config: Optional[SelfPlayRolloutConfig] = None,
    base_seed: int = 0,
) -> MatchupResult:
    """Play ``num_games`` head-to-head games and aggregate per-side metrics.

    Sides are swapped every other game and consecutive games reuse the same
    board seed, so the candidate and baseline each play both orientations of
    every layout. This controls for TEAM_A's first-mover / extra-card edge.
    """
    cfg = rollout_config or SelfPlayRolloutConfig()
    cand_metrics = SideMetrics(name=candidate.name)
    base_metrics = SideMetrics(name=baseline.name)

    for i in range(num_games):
        candidate_is_a = (i % 2 == 0)
        seed = base_seed + (i // 2)  # paired boards across the two orientations

        if candidate_is_a:
            spymasters = {TEAM_A: candidate.spymaster, TEAM_B: baseline.spymaster}
            guessers = {TEAM_A: candidate.guesser, TEAM_B: baseline.guesser}
            cand_team, base_team = TEAM_A, TEAM_B
        else:
            spymasters = {TEAM_A: baseline.spymaster, TEAM_B: candidate.spymaster}
            guessers = {TEAM_A: baseline.guesser, TEAM_B: candidate.guesser}
            cand_team, base_team = TEAM_B, TEAM_A

        episode = collect_self_play_episode(
            words=words,
            spymaster_policy=spymasters,
            guesser_policy=guessers,
            rollout_config=cfg,
            seed=seed,
        )
        accumulate_episode_metrics(cand_metrics, episode, cand_team)
        accumulate_episode_metrics(base_metrics, episode, base_team)

    return MatchupResult(
        candidate=cand_metrics,
        baseline=base_metrics,
        num_games=num_games,
        rollout_config=cfg,
    )

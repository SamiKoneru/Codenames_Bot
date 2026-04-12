"""Collect self-play trajectories from spymaster and guesser policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Union

from game.engine import TEAM_A, TEAM_B, SelfPlayCodenamesGame
from bots.guesser.policy import GuesserObservation, GuesserPolicy
from runner.trajectory import EpisodeRecord, TrajectoryStep
from bots.spymaster.policy import ClueAction, SpymasterObservation, SpymasterPolicy

TeamBoundSpymaster = Union[SpymasterPolicy, Mapping[int, SpymasterPolicy]]
TeamBoundGuesser = Union[GuesserPolicy, Mapping[int, GuesserPolicy]]


@dataclass
class SelfPlayRolloutConfig:
    board_size: int = 25
    team_a_count: int = 9
    team_b_count: int = 8
    gamma: float = 0.99
    max_turns: int = 40
    win_bonus: float = 2.0
    loss_penalty: float = -2.0


def _policy_for_team(policy_or_map, team: int):
    if isinstance(policy_or_map, Mapping):
        if team not in policy_or_map:
            raise KeyError(f"Missing policy for team {team}.")
        return policy_or_map[team]
    return policy_or_map


def _build_spymaster_observation(game: SelfPlayCodenamesGame, turn_index: int) -> SpymasterObservation:
    state = game.get_spymaster_state()
    return SpymasterObservation(
        team=game.active_team,
        active_team=state["active_team"],
        board_words=state["board_words"],
        revealed=state["revealed"],
        labels=state["labels"],
        team_a_remaining=state["team_a_remaining"],
        team_b_remaining=state["team_b_remaining"],
        turn_index=turn_index,
    )


def _build_guesser_observation(
    game: SelfPlayCodenamesGame,
    clue_action: ClueAction,
    guesses_made: List[str],
    turn_index: int,
) -> GuesserObservation:
    state = game.get_public_state()
    return GuesserObservation(
        team=game.active_team,
        active_team=state["active_team"],
        board_words=state["board_words"],
        revealed=state["revealed"],
        clue_word=clue_action.clue_word,
        guess_limit=clue_action.guess_limit,
        guesses_made=list(guesses_made),
        team_a_remaining=state["team_a_remaining"],
        team_b_remaining=state["team_b_remaining"],
        turn_index=turn_index,
    )


def _resolve_guess_index(board_words: List[str], revealed: List[bool], guess_word: Optional[str]) -> int:
    if guess_word is None:
        return -1
    normalized = guess_word.strip().lower()
    for idx, word in enumerate(board_words):
        if not revealed[idx] and word.lower() == normalized:
            return idx
    return -1


def _apply_outcome_bonus(steps: List[TrajectoryStep], winner: Optional[int], cfg: SelfPlayRolloutConfig) -> None:
    if winner is None or not steps:
        return

    latest_by_team: Dict[int, int] = {}
    for idx, step in enumerate(steps):
        latest_by_team[step.team] = idx

    if winner in latest_by_team:
        step = steps[latest_by_team[winner]]
        step.reward += cfg.win_bonus
        step.metadata["outcome_bonus"] = step.metadata.get("outcome_bonus", 0.0) + cfg.win_bonus

    loser = TEAM_B if winner == TEAM_A else TEAM_A
    if loser in latest_by_team:
        step = steps[latest_by_team[loser]]
        step.reward += cfg.loss_penalty
        step.metadata["outcome_bonus"] = step.metadata.get("outcome_bonus", 0.0) + cfg.loss_penalty


def collect_self_play_episode(
    words: List[str],
    spymaster_policy: TeamBoundSpymaster,
    guesser_policy: TeamBoundGuesser,
    rollout_config: Optional[SelfPlayRolloutConfig] = None,
    seed: Optional[int] = None,
) -> EpisodeRecord:
    cfg = rollout_config or SelfPlayRolloutConfig()
    game = SelfPlayCodenamesGame(
        words=words,
        board_size=cfg.board_size,
        team_a_count=cfg.team_a_count,
        team_b_count=cfg.team_b_count,
        seed=seed,
    )

    steps: List[TrajectoryStep] = []
    turn_index = 0

    while not game.game_over and turn_index < cfg.max_turns:
        spymaster = _policy_for_team(spymaster_policy, game.active_team)
        spymaster_observation = _build_spymaster_observation(game, turn_index)
        clue_action = spymaster.choose_clue(spymaster_observation)

        clue_step = TrajectoryStep(
            role="spymaster",
            team=game.active_team,
            observation=spymaster_observation.to_dict(),
            prompt=clue_action.prompt,
            raw_response=clue_action.raw_response,
            action=clue_action.to_dict(),
            metadata={"turn_index": turn_index},
        )
        steps.append(clue_step)

        game.start_turn(max(1, clue_action.guess_limit))
        guesses_made: List[str] = []
        turn_reward = 0.0

        while game.turn_active and not game.game_over:
            guesser = _policy_for_team(guesser_policy, game.active_team)
            guesser_observation = _build_guesser_observation(
                game,
                clue_action=clue_action,
                guesses_made=guesses_made,
                turn_index=turn_index,
            )
            guess_action = guesser.choose_guess(guesser_observation)

            guess_step = TrajectoryStep(
                role="guesser",
                team=game.active_team,
                observation=guesser_observation.to_dict(),
                prompt=guess_action.prompt,
                raw_response=guess_action.raw_response,
                action=guess_action.to_dict(),
                metadata={"turn_index": turn_index},
            )

            if guess_action.end_turn:
                game.end_turn()
                guess_step.metadata["reason"] = "model_ended_turn"
                steps.append(guess_step)
                break

            guess_index = _resolve_guess_index(
                game.board_words,
                game.revealed,
                guess_action.guess_word,
            )
            outcome = game.guess_word(guess_index)
            guess_step.environment_reward = float(outcome["reward"])
            guess_step.reward = float(outcome["reward"])
            guess_step.terminal = bool(outcome["game_over"])
            guess_step.metadata.update(
                {
                    "reason": outcome["reason"],
                    "guessed_index": guess_index,
                    "resolved_label": outcome["label"],
                }
            )
            steps.append(guess_step)

            turn_reward += float(outcome["reward"])
            if guess_action.guess_word is not None:
                guesses_made.append(guess_action.guess_word)

        clue_step.environment_reward = turn_reward
        clue_step.reward = turn_reward
        turn_index += 1

    summary = {
        "turns_played": turn_index,
        "truncated": (not game.game_over and turn_index >= cfg.max_turns),
    }
    episode = EpisodeRecord(
        seed=seed,
        board_words=list(game.board_words),
        labels=list(game.labels),
        winner=game.winner,
        steps=steps,
        summary=summary,
    )
    _apply_outcome_bonus(episode.steps, episode.winner, cfg)
    episode.assign_discounted_returns(cfg.gamma)
    return episode


def collect_self_play_episodes(
    words: List[str],
    spymaster_policy: TeamBoundSpymaster,
    guesser_policy: TeamBoundGuesser,
    num_episodes: int,
    rollout_config: Optional[SelfPlayRolloutConfig] = None,
    seed: Optional[int] = None,
) -> List[EpisodeRecord]:
    cfg = rollout_config or SelfPlayRolloutConfig()
    base_seed = seed if seed is not None else 0
    return [
        collect_self_play_episode(
            words=words,
            spymaster_policy=spymaster_policy,
            guesser_policy=guesser_policy,
            rollout_config=cfg,
            seed=base_seed + episode_idx,
        )
        for episode_idx in range(num_episodes)
    ]

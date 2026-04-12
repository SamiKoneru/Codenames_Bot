"""CLI script that runs one full Codenames game between two bots and prints each move."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.engine import TEAM_A, TEAM_B, SelfPlayCodenamesGame
from bots.spymaster.policy import RandomSpymasterPolicy, SpymasterObservation
from bots.guesser.policy import RandomGuesserPolicy, GuesserObservation
from utils.words import load_words


TEAM_NAMES = {TEAM_A: "TEAM_A", TEAM_B: "TEAM_B"}
LABEL_NAMES = {0: "TEAM_A", 1: "TEAM_B", 2: "neutral", 3: "assassin"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for board layout and bot decisions.",
    )
    parser.add_argument(
        "--word-file",
        default=str(PROJECT_ROOT / "wordlist-eng.txt"),
        help="Path to newline-separated Codenames vocabulary.",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=25,
        help="Number of cards on the board (default: 25).",
    )
    return parser.parse_args()


def _resolve_guess_index(board_words, revealed, guess_word):
    if guess_word is None:
        return -1
    normalized = guess_word.strip().lower()
    for idx, word in enumerate(board_words):
        if not revealed[idx] and word.lower() == normalized:
            return idx
    return -1


def main() -> None:
    args = parse_args()
    words = load_words(args.word_file)

    board_size = args.board_size
    # Default counts scaled to board size; keep original ratio for 25-card board.
    team_a_count = max(1, board_size * 9 // 25)
    team_b_count = max(1, board_size * 8 // 25)

    game = SelfPlayCodenamesGame(
        words=words,
        board_size=board_size,
        team_a_count=team_a_count,
        team_b_count=team_b_count,
        seed=args.seed,
    )

    seed_a = args.seed if args.seed is not None else 0
    spymaster_a = RandomSpymasterPolicy(seed=seed_a)
    spymaster_b = RandomSpymasterPolicy(seed=seed_a + 1)
    guesser_a = RandomGuesserPolicy(seed=seed_a + 2)
    guesser_b = RandomGuesserPolicy(seed=seed_a + 3)

    spymasters = {TEAM_A: spymaster_a, TEAM_B: spymaster_b}
    guessers = {TEAM_A: guesser_a, TEAM_B: guesser_b}

    print("=== CODENAMES: Bot vs Bot ===")
    print(f"Board size: {board_size}  |  TEAM_A cards: {team_a_count}  |  TEAM_B cards: {team_b_count}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()

    turn_index = 0
    max_turns = 40

    while not game.game_over and turn_index < max_turns:
        team = game.active_team
        team_name = TEAM_NAMES[team]

        # --- Spymaster phase ---
        spy_state = game.get_spymaster_state()
        obs = SpymasterObservation(
            team=team,
            active_team=spy_state["active_team"],
            board_words=spy_state["board_words"],
            revealed=spy_state["revealed"],
            labels=spy_state["labels"],
            team_a_remaining=spy_state["team_a_remaining"],
            team_b_remaining=spy_state["team_b_remaining"],
            turn_index=turn_index,
        )
        clue_action = spymasters[team].choose_clue(obs)

        print(f"Turn {turn_index + 1} — {team_name} Spymaster")
        print(f'  Clue: "{clue_action.clue_word}" ({clue_action.guess_limit})')

        game.start_turn(max(1, clue_action.guess_limit))
        guesses_made = []

        # --- Guesser phase ---
        print(f"Turn {turn_index + 1} — {team_name} Guesser")
        while game.turn_active and not game.game_over:
            pub_state = game.get_public_state()
            g_obs = GuesserObservation(
                team=team,
                active_team=pub_state["active_team"],
                board_words=pub_state["board_words"],
                revealed=pub_state["revealed"],
                clue_word=clue_action.clue_word,
                guess_limit=clue_action.guess_limit,
                guesses_made=list(guesses_made),
                team_a_remaining=pub_state["team_a_remaining"],
                team_b_remaining=pub_state["team_b_remaining"],
                turn_index=turn_index,
            )
            guess_action = guessers[team].choose_guess(g_obs)

            if guess_action.end_turn or guess_action.guess_word is None:
                game.end_turn()
                print("  [Bot chose to end turn]")
                break

            guess_index = _resolve_guess_index(
                game.board_words, game.revealed, guess_action.guess_word
            )
            outcome = game.guess_word(guess_index)
            reward = outcome["reward"]
            reason = outcome["reason"]
            label_name = LABEL_NAMES.get(outcome["label"], "?") if outcome["label"] is not None else "invalid"

            reward_str = f"{reward:+.1f}"
            print(f'  Guess: "{guess_action.guess_word}" -> {reason} ({reward_str}) [{label_name}]')

            if guess_action.guess_word is not None:
                guesses_made.append(guess_action.guess_word)

        remaining_a = len(game.team_remaining[TEAM_A])
        remaining_b = len(game.team_remaining[TEAM_B])
        print(f"  [TEAM_A remaining: {remaining_a}  |  TEAM_B remaining: {remaining_b}]")
        print()

        turn_index += 1

    print("=== GAME OVER ===")
    if game.winner is not None:
        print(f"Winner: {TEAM_NAMES[game.winner]}")
    else:
        print("Winner: None (truncated)")
    print(f"Turns played: {turn_index}")


if __name__ == "__main__":
    main()

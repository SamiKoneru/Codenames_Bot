"""Generate self-play JSONL data for downstream LLM fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.guesser.policy import RandomGuesserPolicy
from runner.export import episode_to_jsonl_records, write_jsonl
from runner.play import SelfPlayRolloutConfig, collect_self_play_episodes
from utils.words import load_words
from bots.spymaster.policy import RandomSpymasterPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--word-file",
        default=str(PROJECT_ROOT / "wordlist-eng.txt"),
        help="Path to newline-separated Codenames vocabulary.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=4,
        help="Number of self-play episodes to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for board sampling and random baseline policies.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "artifacts" / "self_play_rollouts.jsonl"),
        help="Where to write JSONL rollout data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    words = load_words(args.word_file)
    rollout_cfg = SelfPlayRolloutConfig()
    episodes = collect_self_play_episodes(
        words=words,
        spymaster_policy=RandomSpymasterPolicy(seed=args.seed),
        guesser_policy=RandomGuesserPolicy(seed=args.seed + 1),
        num_episodes=args.episodes,
        rollout_config=rollout_cfg,
        seed=args.seed,
    )

    records = [r for ep in episodes for r in episode_to_jsonl_records(ep)]

    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records from {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()

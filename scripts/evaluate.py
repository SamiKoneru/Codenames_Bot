"""Evaluate a Codenames bot head-to-head against a baseline.

Each "side" is a (spymaster, guesser) pair. By default both sides are random
policies, which is useful for sanity-checking the harness itself (a fair coin
should land near a 50% win rate). Point ``--candidate-model`` at a trained
checkpoint to measure whether training actually helped.

Examples
--------
    # Sanity check: random vs random should be ~50/50.
    python scripts/evaluate.py --games 100

    # Trained model vs random baseline.
    python scripts/evaluate.py \\
        --candidate-model artifacts/checkpoints \\
        --games 100 --json-out artifacts/eval.json

    # Trained model vs the base (untrained) model it started from.
    python scripts/evaluate.py \\
        --candidate-model artifacts/checkpoints \\
        --baseline-model gpt2 \\
        --games 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.guesser.policy import LLMGuesserPolicy, RandomGuesserPolicy
from bots.spymaster.policy import LLMSpymasterPolicy, RandomSpymasterPolicy
from runner.evaluate import Side, evaluate_matchup
from runner.play import SelfPlayRolloutConfig
from utils.words import load_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--word-file", default=str(PROJECT_ROOT / "wordlist-eng.txt"))
    parser.add_argument("--games", type=int, default=50, help="Total games to play.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--team-a-count", type=int, default=3)
    parser.add_argument("--team-b-count", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for boards/policies.")

    parser.add_argument(
        "--candidate-model",
        default=None,
        help="HF model name or local path for the candidate. Random if omitted.",
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="HF model name or local path for the baseline. Random if omitted.",
    )
    parser.add_argument("--device", default=None, help="Torch device for HF models.")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry budget for an LLM spymaster's illegal clues.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable clue legality validation for LLM spymasters.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the metrics as JSON (for tracking checkpoints).",
    )
    return parser.parse_args()


def _build_side(name_override, model_path, args, seed) -> Side:
    """Construct a Side: LLM-backed if a model path is given, else random."""
    if model_path:
        # Imported lazily so the harness/tests don't require torch+transformers.
        from llm.providers.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend(model_path, device=args.device)
        spymaster = LLMSpymasterPolicy(
            backend,
            validate_clues=not args.no_validate,
            max_retries=args.max_retries,
        )
        guesser = LLMGuesserPolicy(backend)
        name = name_override or Path(model_path).name
    else:
        spymaster = RandomSpymasterPolicy(seed=seed)
        guesser = RandomGuesserPolicy(seed=seed + 1)
        name = name_override or "random"
    return Side(name=name, spymaster=spymaster, guesser=guesser)


def main() -> None:
    args = parse_args()
    words = load_words(args.word_file)

    candidate = _build_side(args.candidate_name, args.candidate_model, args, seed=args.seed)
    baseline = _build_side(args.baseline_name, args.baseline_model, args, seed=args.seed + 1000)

    # Disambiguate identical default names so the report is readable.
    if candidate.name == baseline.name:
        candidate.name = f"{candidate.name} (cand)"
        baseline.name = f"{baseline.name} (base)"

    rollout_config = SelfPlayRolloutConfig(
        board_size=args.board_size,
        team_a_count=args.team_a_count,
        team_b_count=args.team_b_count,
        max_turns=args.max_turns,
    )

    result = evaluate_matchup(
        words=words,
        candidate=candidate,
        baseline=baseline,
        num_games=args.games,
        rollout_config=rollout_config,
        base_seed=args.seed,
    )

    print(result.report())

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()

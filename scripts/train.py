"""CLI entry point for SFT warm-start and GRPO training.

Typical workflow
----------------
    # 1. SFT warm-start (optional but recommended)
    python scripts/train.py sft \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --role guesser \\
        --episodes 64 \\
        --epochs 3 \\
        --output artifacts/sft

    # 2. GRPO reinforcement learning
    python scripts/train.py grpo \\
        --model artifacts/sft \\
        --role guesser \\
        --steps 500 \\
        --group-size 8 \\
        --output artifacts/checkpoints

    # Train spymaster instead (or train both together)
    python scripts/train.py grpo --model ... --role spymaster
    python scripts/train.py grpo --model ... --role both
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.providers.huggingface import HuggingFaceBackend
from training.grpo import GRPOConfig, train_grpo
from training.sft import SFTConfig, train_sft

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Shared args
# ---------------------------------------------------------------------------

def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or local path (e.g. mistralai/Mistral-7B-Instruct-v0.2).",
    )
    parser.add_argument(
        "--role",
        choices=["guesser", "spymaster", "both"],
        default="guesser",
        help="Which bot to train. The other role uses a random policy.",
    )
    parser.add_argument(
        "--word-file",
        default=str(PROJECT_ROOT / "wordlist-eng.txt"),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--board-size",   type=int, default=9,
    )
    parser.add_argument(
        "--team-a-count", type=int, default=3,
    )
    parser.add_argument(
        "--team-b-count", type=int, default=3,
    )


# ---------------------------------------------------------------------------
# SFT sub-command
# ---------------------------------------------------------------------------

def _build_sft_parser(sub) -> None:
    p = sub.add_parser("sft", help="Supervised fine-tuning warm-start.")
    _add_shared_args(p)
    p.add_argument("--episodes",       type=int,   default=64,   help="Rollout episodes for dataset collection.")
    p.add_argument("--epochs",         type=int,   default=3,    help="Training epochs over the collected dataset.")
    p.add_argument("--lr",             type=float, default=2e-5, help="Learning rate.")
    p.add_argument("--min-return",     type=float, default=0.0,  help="Minimum discounted return to include a step.")
    p.add_argument("--output",         default="artifacts/sft",  help="Directory to save the fine-tuned model.")
    p.add_argument("--log-every",      type=int,   default=20)


def _run_sft(args: argparse.Namespace) -> None:
    backend = HuggingFaceBackend(args.model, device=args.device)
    cfg = SFTConfig(
        word_file=args.word_file,
        role=args.role,
        minimum_return=args.min_return,
        num_collection_episodes=args.episodes,
        board_size=args.board_size,
        team_a_count=args.team_a_count,
        team_b_count=args.team_b_count,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        log_every=args.log_every,
    )
    train_sft(backend, cfg)


# ---------------------------------------------------------------------------
# GRPO sub-command
# ---------------------------------------------------------------------------

def _build_grpo_parser(sub) -> None:
    p = sub.add_parser("grpo", help="GRPO reinforcement learning.")
    _add_shared_args(p)
    p.add_argument("--steps",       type=int,   default=500,               help="Number of gradient update steps.")
    p.add_argument("--group-size",  type=int,   default=8,                 help="Episodes per step (G in GRPO).")
    p.add_argument("--lr",          type=float, default=1e-5,              help="Learning rate.")
    p.add_argument("--kl-coeff",    type=float, default=0.04,              help="KL penalty coefficient (β).")
    p.add_argument("--gamma",       type=float, default=0.99,              help="Discount factor for returns.")
    p.add_argument("--max-turns",   type=int,   default=20,                help="Max turns per episode.")
    p.add_argument("--output",      default="artifacts/checkpoints",       help="Directory for checkpoints.")
    p.add_argument("--save-every",  type=int,   default=50)
    p.add_argument("--log-every",   type=int,   default=10)


def _run_grpo(args: argparse.Namespace) -> None:
    backend = HuggingFaceBackend(args.model, device=args.device)
    cfg = GRPOConfig(
        word_file=args.word_file,
        role=args.role,
        episodes_per_step=args.group_size,
        board_size=args.board_size,
        team_a_count=args.team_a_count,
        team_b_count=args.team_b_count,
        max_turns=args.max_turns,
        gamma=args.gamma,
        total_steps=args.steps,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        output_dir=args.output,
        save_every=args.save_every,
        log_every=args.log_every,
    )
    train_grpo(backend, cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Codenames bots via SFT and/or GRPO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_sft_parser(sub)
    _build_grpo_parser(sub)

    args = parser.parse_args()
    if args.command == "sft":
        _run_sft(args)
    elif args.command == "grpo":
        _run_grpo(args)


if __name__ == "__main__":
    main()

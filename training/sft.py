"""Supervised Fine-Tuning warm-start before RL.

Why SFT first?
--------------
A randomly initialised (or base) LLM will produce near-random JSON responses
at the start of RL training.  Most of those responses will be invalid, giving
the policy gradient nothing useful to learn from.

SFT solves the cold-start problem by first training the model on *already good*
examples using standard cross-entropy loss.  After SFT, the model reliably
outputs valid JSON, so RL can focus on improving the *quality* of the clues and
guesses rather than teaching the model the output format.

Where the data comes from
-------------------------
Good examples are collected by running the random-policy rollout pipeline and
keeping only the steps where the discounted return is above a threshold —
i.e., steps that led to relatively good outcomes.  This is sometimes called
"filtered imitation learning" or "best-of-N SFT".

Loss
----
Standard cross-entropy (next-token prediction) over the completion only:

    L = -mean_over_tokens( log π(token_i | prompt, token_1..i-1) )

No reward weighting, no KL penalty — just maximise the likelihood of the
high-return completions.  This is identical to language model pre-training
loss, restricted to the completion portion of each sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW

from bots.guesser.policy import RandomGuesserPolicy
from bots.spymaster.policy import RandomSpymasterPolicy
from llm.providers.huggingface import HuggingFaceBackend
from runner.export import episode_to_jsonl_records
from runner.play import SelfPlayRolloutConfig, collect_self_play_episodes
from training.log_probs import completion_log_probs, encode_prompt_and_completion
from utils.words import load_words

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    word_file: str = "wordlist-eng.txt"
    role: str = "guesser"          # "guesser" | "spymaster" | "both"

    # Only keep steps whose discounted_return >= this threshold.
    # Set to a very negative number to use all steps.
    minimum_return: float = 0.0

    # Number of rollout episodes to collect for the SFT dataset.
    num_collection_episodes: int = 64

    # ── Rollout settings (for data collection only) ──────────────────────
    board_size: int = 9
    team_a_count: int = 3
    team_b_count: int = 3
    max_turns: int = 20
    gamma: float = 0.99
    collection_seed: int = 0

    # ── Training ────────────────────────────────────────────────────────────
    # SFT passes over the collected dataset.  More epochs → closer fit, but
    # risks overfitting to the random-policy data before RL can improve it.
    num_epochs: int = 3
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir: str = "artifacts/sft"
    log_every: int = 20


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_sft_records(
    words: List[str],
    cfg: SFTConfig,
) -> List[dict]:
    """Run random-policy rollouts and keep only high-return steps."""
    rollout_cfg = SelfPlayRolloutConfig(
        board_size=cfg.board_size,
        team_a_count=cfg.team_a_count,
        team_b_count=cfg.team_b_count,
        max_turns=cfg.max_turns,
        gamma=cfg.gamma,
    )

    episodes = collect_self_play_episodes(
        words=words,
        spymaster_policy=RandomSpymasterPolicy(seed=cfg.collection_seed),
        guesser_policy=RandomGuesserPolicy(seed=cfg.collection_seed + 1),
        num_episodes=cfg.num_collection_episodes,
        rollout_config=rollout_cfg,
        seed=cfg.collection_seed,
    )

    roles = None if cfg.role == "both" else [cfg.role]
    records = []
    for ep in episodes:
        for rec in episode_to_jsonl_records(ep, roles=roles):
            if rec["discounted_return"] >= cfg.minimum_return:
                records.append(rec)

    logger.info(
        "SFT dataset: %d records collected (minimum_return=%.2f, role=%s)",
        len(records), cfg.minimum_return, cfg.role,
    )
    return records


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sft(
    backend: HuggingFaceBackend,
    cfg: Optional[SFTConfig] = None,
) -> None:
    """Fine-tune backend.model on high-return rollout steps using cross-entropy.

    This is intended as a warm-start before GRPO.  After SFT the model should
    reliably produce valid JSON and make reasonable guesses/clues, giving the
    RL loop a much better starting point.
    """
    cfg    = cfg or SFTConfig()
    device    = backend.device
    model     = backend.model
    tokenizer = backend.tokenizer

    words   = load_words(cfg.word_file)
    records = collect_sft_records(words, cfg)

    if not records:
        logger.warning("No SFT records collected — skipping SFT.")
        return

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting SFT | role=%s | records=%d | epochs=%d | device=%s",
        cfg.role, len(records), cfg.num_epochs, device,
    )

    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0

        for record in records:
            prompt     = record["prompt"]
            completion = record["completion"]

            if not completion.strip():
                continue

            full_ids, prompt_len = encode_prompt_and_completion(
                tokenizer, prompt, completion, device
            )
            if full_ids.shape[1] - prompt_len <= 0:
                continue

            # Standard cross-entropy: maximise log π(completion | prompt).
            # completion_log_probs returns per-token log probs; we minimise
            # their negative mean.
            token_lps = completion_log_probs(model, full_ids, prompt_len)
            loss = -token_lps.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg.log_every == 0:
                logger.info(
                    "epoch=%d  step=%d  loss=%.4f",
                    epoch + 1, global_step, loss.item(),
                )

        avg_loss = epoch_loss / max(len(records), 1)
        logger.info("Epoch %d complete | avg_loss=%.4f", epoch + 1, avg_loss)

    # Save
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("SFT complete. Model saved → %s", cfg.output_dir)

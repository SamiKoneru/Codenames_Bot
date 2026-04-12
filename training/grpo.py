"""GRPO (Group Relative Policy Optimization) training loop.

How GRPO works
--------------
Standard policy gradient (REINFORCE) is high-variance because the reward for
a single trajectory is a noisy signal.  GRPO reduces variance by generating
a *group* of G responses for the same input, then normalising rewards across
the group before computing the loss.  This turns raw rewards into relative
advantages without needing a separate critic network.

In Codenames terms: each training step plays G complete games from the same
starting seed, producing G sets of (prompt, completion, reward) records.
The normalised advantage for a record is:

    A = (R - mean(R_group)) / std(R_group)

The policy gradient loss is then:

    L_pg = -mean_over_records(A * mean_over_tokens(log π(token | context)))

To prevent the policy from drifting too far from the original model, we add a
per-token KL penalty against a frozen reference copy of the model:

    L_kl  = β * mean(log π(token) - log π_ref(token))
    L_total = L_pg + L_kl

Gradient flow
-------------
The entire signal flows through  log π(completion | prompt), which is a sum
of log-softmax values at each completion token position.  Those are
differentiable w.r.t. every weight in the transformer.

    reward  →  advantage (normalised, constant w.r.t. weights)
    advantage * log_probs  →  policy gradient loss
    loss.backward()  →  dL/d(weights)  through the log-softmax → linear → attn
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.optim import AdamW

from bots.guesser.policy import LLMGuesserPolicy, RandomGuesserPolicy
from bots.spymaster.policy import LLMSpymasterPolicy, RandomSpymasterPolicy
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
class GRPOConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    word_file: str = "wordlist-eng.txt"

    # Which role to train: "guesser" | "spymaster" | "both"
    # The other role is played by a fixed RandomPolicy so the trained role
    # always has a stable opponent during rollout.
    role: str = "guesser"

    # ── Group size ──────────────────────────────────────────────────────────
    # G episodes are collected per update step.  Their rewards are normalised
    # together to form advantages — larger G → lower variance, more compute.
    episodes_per_step: int = 8

    # ── Rollout ─────────────────────────────────────────────────────────────
    board_size: int = 9
    team_a_count: int = 3
    team_b_count: int = 3
    max_turns: int = 20
    gamma: float = 0.99

    # ── Optimisation ────────────────────────────────────────────────────────
    total_steps: int = 500
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # ── Loss coefficients ───────────────────────────────────────────────────
    # β: weight of the KL penalty that keeps the policy near the reference.
    # Higher β → more conservative updates.  Typical range: 0.01 – 0.1.
    kl_coeff: float = 0.04

    # ── Checkpointing ───────────────────────────────────────────────────────
    output_dir: str = "artifacts/checkpoints"
    save_every: int = 50
    log_every: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_records(
    backend: HuggingFaceBackend,
    words: List[str],
    cfg: GRPOConfig,
    seed: int,
) -> List[dict]:
    """Play cfg.episodes_per_step games and return flattened step records."""
    rollout_cfg = SelfPlayRolloutConfig(
        board_size=cfg.board_size,
        team_a_count=cfg.team_a_count,
        team_b_count=cfg.team_b_count,
        max_turns=cfg.max_turns,
        gamma=cfg.gamma,
    )

    # The role being trained uses the live LLM backend.
    # The other role uses a cheap random policy so we only update one bot at a time.
    spymaster_policy = (
        LLMSpymasterPolicy(backend) if cfg.role in ("spymaster", "both")
        else RandomSpymasterPolicy(seed=seed)
    )
    guesser_policy = (
        LLMGuesserPolicy(backend) if cfg.role in ("guesser", "both")
        else RandomGuesserPolicy(seed=seed)
    )

    episodes = collect_self_play_episodes(
        words=words,
        spymaster_policy=spymaster_policy,
        guesser_policy=guesser_policy,
        num_episodes=cfg.episodes_per_step,
        rollout_config=rollout_cfg,
        seed=seed,
    )

    # Only return records for the role being trained so we don't accidentally
    # apply a guesser reward to a spymaster response or vice versa.
    roles = None if cfg.role == "both" else [cfg.role]
    records = []
    for ep in episodes:
        records.extend(episode_to_jsonl_records(ep, roles=roles))
    return records


def _compute_advantages(records: List[dict]) -> List[float]:
    """Normalise discounted returns across the group to form GRPO advantages.

    Returns a list of floats, one per record.  Mean ≈ 0, std ≈ 1.
    A single record (or all-identical rewards) produces advantages of 0.
    """
    if len(records) < 2:
        return [0.0] * len(records)

    returns = np.array([r["discounted_return"] for r in records], dtype=np.float32)
    mean = returns.mean()
    std  = returns.std()
    if std < 1e-8:
        return [0.0] * len(records)

    return ((returns - mean) / std).tolist()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_grpo(
    backend: HuggingFaceBackend,
    cfg: Optional[GRPOConfig] = None,
) -> None:
    """Run the GRPO training loop, updating backend.model in-place.

    After training, the updated weights live in backend.model and are also
    saved to cfg.output_dir every cfg.save_every steps.
    """
    cfg = cfg or GRPOConfig()
    device    = backend.device
    model     = backend.model
    tokenizer = backend.tokenizer

    # Frozen reference model — used only for the KL penalty, never updated.
    # deepcopy so it's a completely separate object sharing no state with model.
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    words     = load_words(cfg.word_file)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting GRPO | role=%s | G=%d | steps=%d | device=%s",
        cfg.role, cfg.episodes_per_step, cfg.total_steps, device,
    )

    for step in range(cfg.total_steps):

        # ── 1. Collect rollouts ────────────────────────────────────────────
        # The model is in eval mode during rollout so dropout is off and
        # model.generate() uses its standard greedy/sampling path.
        model.eval()
        records    = _collect_records(backend, words, cfg, seed=step)
        advantages = _compute_advantages(records)

        if not records:
            logger.warning("step=%d: no records collected, skipping update.", step)
            continue

        # ── 2. Compute loss ───────────────────────────────────────────────
        # Switch to train mode so gradients are tracked through the forward pass.
        model.train()

        pg_loss_accum  = torch.tensor(0.0, device=device)
        kl_loss_accum  = torch.tensor(0.0, device=device)
        n_valid = 0

        for record, advantage in zip(records, advantages):
            prompt     = record["prompt"]
            completion = record["completion"]

            if not completion.strip():
                continue

            full_ids, prompt_len = encode_prompt_and_completion(
                tokenizer, prompt, completion, device
            )
            if full_ids.shape[1] - prompt_len <= 0:
                continue

            # ── Current policy log probs (tracked for backprop) ───────────
            token_lps = completion_log_probs(model, full_ids, prompt_len)

            # ── Reference policy log probs (no grad needed) ───────────────
            with torch.no_grad():
                ref_token_lps = completion_log_probs(ref_model, full_ids, prompt_len)

            # ── Policy gradient term ──────────────────────────────────────
            # Maximise E[A * log π] → minimise -A * mean(log π).
            # mean() over tokens reduces sensitivity to completion length.
            pg_loss = -advantage * token_lps.mean()

            # ── KL penalty term ───────────────────────────────────────────
            # Sampled estimator: KL(π||π_ref) ≈ mean(log π - log π_ref).
            # This is positive on average when π drifts from π_ref.
            kl_loss = cfg.kl_coeff * (token_lps - ref_token_lps).mean()

            pg_loss_accum += pg_loss
            kl_loss_accum += kl_loss
            n_valid += 1

        if n_valid == 0:
            continue

        loss = (pg_loss_accum + kl_loss_accum) / n_valid

        # ── 3. Update weights ─────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        # ── 4. Logging ────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            avg_return = np.mean([r["discounted_return"] for r in records])
            logger.info(
                "step=%4d | loss=%.4f | pg=%.4f | kl=%.4f | avg_return=%.3f | records=%d",
                step,
                loss.item(),
                (pg_loss_accum / n_valid).item(),
                (kl_loss_accum / n_valid).item(),
                avg_return,
                len(records),
            )

        # ── 5. Checkpoint ─────────────────────────────────────────────────
        if (step + 1) % cfg.save_every == 0:
            ckpt_dir = Path(cfg.output_dir) / f"step_{step + 1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Checkpoint saved → %s", ckpt_dir)

    # Final save
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete. Final model → %s", final_dir)

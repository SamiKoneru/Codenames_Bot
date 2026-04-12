"""Minimal oracle clue generator for synthetic clue vectors (e.g. warm-start data).

This is *not* legal Codenames clue generation. It builds a clue vector from
board embeddings. Use NumPy arrays from your embedding API or LLM layer outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Union

import numpy as np

TEAM = 0


@dataclass
class OracleClueConfig:
    # k ~ Normal(mu, sigma), then floor + clamp to [1, remaining_team_count]
    k_mean: float = 3.0
    k_std: float = 1.0

    # Randomly weighted average of chosen TEAM embeddings.
    use_random_weights: bool = True

    # Baseline clue noise.
    noise_std: float = 0.25

    # With this probability, add stronger noise to simulate weak/misaligned clues.
    unrelated_noise_prob: float = 0.25
    unrelated_noise_std: float = 0.65

    # Normalize output embedding for stability.
    normalize_output: bool = True


def _sample_k(remaining_team_count: int, cfg: OracleClueConfig, rng: random.Random) -> int:
    if remaining_team_count <= 0:
        return 0
    return max(1, min(math.floor(rng.gauss(cfg.k_mean, cfg.k_std)), remaining_team_count))


def _weighted_mean(embeddings: np.ndarray, use_random_weights: bool, rng: random.Random) -> np.ndarray:
    """embeddings shape: [k, D]"""
    k = embeddings.shape[0]
    if k == 1:
        return embeddings[0].copy()

    if use_random_weights:
        w = np.array([rng.random() + 1e-6 for _ in range(k)], dtype=embeddings.dtype)
        w = w / w.sum()
    else:
        w = np.full((k,), 1.0 / k, dtype=embeddings.dtype)

    return (embeddings * w[:, np.newaxis]).sum(axis=0)


def _as_bool_array(revealed: Union[List[bool], np.ndarray], n: int) -> np.ndarray:
    r = np.asarray(revealed, dtype=bool)
    if r.shape[0] != n:
        raise ValueError("labels/revealed lengths must match board size N.")
    return r


def _as_long_array(labels: Union[List[int], np.ndarray], n: int) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64)
    if lab.shape[0] != n:
        raise ValueError("labels/revealed lengths must match board size N.")
    return lab


def generate_oracle_clue(
    board_emb: np.ndarray,
    labels: Union[List[int], np.ndarray],
    revealed: Union[List[bool], np.ndarray],
    target_label: int = TEAM,
    cfg: Optional[OracleClueConfig] = None,
    rng: Optional[random.Random] = None,
) -> Dict:
    """Generate a clue vector + metadata from board state.

    Args:
    - board_emb: [N, D] board word embeddings
    - labels: length-N labels (TEAM=0)
    - revealed: length-N booleans

    Returns:
    - clue_emb: [D] numpy array
    - k: sampled guess count
    - target_indices: TEAM indices used to form clue
    - info: extra metadata for debugging/training
    """
    if board_emb.ndim != 2:
        raise ValueError("board_emb must have shape [N, D].")

    cfg = cfg or OracleClueConfig()
    rng = rng or random.Random()

    n = board_emb.shape[0]
    labels_t = _as_long_array(labels, n)
    revealed_t = _as_bool_array(revealed, n)

    remaining_team = [
        i for i in range(n) if int(labels_t[i]) == target_label and not bool(revealed_t[i])
    ]

    if not remaining_team:
        clue_emb = np.zeros(board_emb.shape[1], dtype=board_emb.dtype)
        return {
            "clue_emb": clue_emb,
            "k": 0,
            "target_indices": [],
            "info": {"reason": "no_remaining_team"},
        }

    k = _sample_k(len(remaining_team), cfg, rng)
    target_indices = rng.sample(remaining_team, k)

    target_emb = board_emb[target_indices]
    mean_emb = _weighted_mean(target_emb, cfg.use_random_weights, rng)

    subrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = subrng.standard_normal(mean_emb.shape).astype(mean_emb.dtype) * cfg.noise_std

    used_unrelated_noise = False
    if rng.random() < cfg.unrelated_noise_prob:
        noise = noise + subrng.standard_normal(mean_emb.shape).astype(mean_emb.dtype) * cfg.unrelated_noise_std
        used_unrelated_noise = True

    clue_emb = mean_emb + noise
    if cfg.normalize_output:
        norm = np.linalg.norm(clue_emb)
        if norm > 1e-12:
            clue_emb = clue_emb / norm

    return {
        "clue_emb": clue_emb,
        "k": k,
        "target_indices": target_indices,
        "info": {
            "remaining_team_count": len(remaining_team),
            "used_unrelated_noise": used_unrelated_noise,
            "noise_std": cfg.noise_std,
            "unrelated_noise_prob": cfg.unrelated_noise_prob,
            "unrelated_noise_std": cfg.unrelated_noise_std,
        },
    }

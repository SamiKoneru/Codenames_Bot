"""Per-token log-probability utilities for the policy gradient training loop.

The core challenge in LLM RL is computing  log π(completion | prompt)  in a
way that is differentiable with respect to the model weights.  This module
handles the token-level bookkeeping so the training loops stay readable.

How the shift works
-------------------
A causal LM produces logits where logits[i] is the distribution over the token
that *follows* position i.  So to score token[j] you use logits[j-1].

    full sequence:  [p₀ p₁ ... pₙ₋₁ | c₀ c₁ ... cₘ₋₁]
                     ^── prompt (n tokens) ──^  ^── completion (m tokens) ──^

    logits shape:   [seq_len, vocab]    (after squeezing batch dim)

    To score completion token cⱼ (at full-sequence index n+j):
        use logits[n+j-1]  →  shifted by one to the left

    In slice notation:
        comp_logits = logits[n-1 : n+m-1]   (shape [m, vocab])
        comp_ids    = full_ids[n : n+m]      (shape [m])
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def encode_prompt_and_completion(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    device: str,
    max_length: int = 2048,
) -> Tuple[torch.Tensor, int]:
    """Encode a (prompt, completion) pair into a single token sequence.

    Returns:
        full_ids:   LongTensor of shape [1, seq_len] on *device*.
        prompt_len: Number of tokens belonging to the prompt.  Completion
                    tokens start at index prompt_len.
    """
    # Encode prompt alone to find where the completion starts.
    # add_special_tokens=True so BOS is counted if the model uses one.
    prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)

    full_ids = tokenizer(
        prompt + completion,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    ).input_ids.to(device)

    return full_ids, prompt_len


def completion_log_probs(
    model: PreTrainedModel,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token log probabilities for the completion portion of full_ids.

    Args:
        model:      A HuggingFace CausalLM (may or may not be in train mode).
        full_ids:   LongTensor [1, seq_len] — the full prompt+completion sequence.
        prompt_len: How many tokens belong to the prompt.

    Returns:
        1D FloatTensor of shape [completion_len].
        Each entry is  log π(token_i | all previous tokens).
        Returns an empty tensor if the completion is zero-length.
    """
    logits = model(full_ids).logits  # [1, seq_len, vocab]
    logits = logits[0]               # [seq_len, vocab]

    # Apply the -1 shift: logits[i] scores the token at position i+1.
    comp_logits = logits[prompt_len - 1 : -1]  # [comp_len, vocab]
    comp_ids    = full_ids[0, prompt_len:]      # [comp_len]

    if comp_ids.shape[0] == 0:
        return torch.zeros(0, device=full_ids.device)

    log_probs = F.log_softmax(comp_logits, dim=-1)                  # [comp_len, vocab]
    token_log_probs = log_probs.gather(1, comp_ids.unsqueeze(1)).squeeze(1)  # [comp_len]
    return token_log_probs

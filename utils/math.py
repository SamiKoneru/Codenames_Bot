from __future__ import annotations
import json
from typing import Any, Dict, List

import numpy as np


def build_board_word_ids(board_words: List[str], word_to_idx: Dict[str, int]) -> np.ndarray:
    """Map board token strings to vocab IDs."""
    return np.array([word_to_idx[w.lower()] for w in board_words], dtype=np.int64)


def discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Compute discounted returns for a single trajectory."""
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def extract_json_dict(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating surrounding text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("Expected a JSON object in the model response.") from None
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected the model response to decode to a JSON object.")
    return data

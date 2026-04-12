from __future__ import annotations
from typing import List


def load_words(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return list(dict.fromkeys(words))

"""Export trajectory records to JSONL for RL or SFT pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from runner.trajectory import EpisodeRecord


def episode_to_jsonl_records(
    episode: EpisodeRecord,
    roles: Optional[Sequence[str]] = None,
) -> List[Dict]:
    allowed_roles = set(roles) if roles is not None else None
    records: List[Dict] = []
    for step in episode.steps:
        if allowed_roles is not None and step.role not in allowed_roles:
            continue
        records.append(
            {
                "role": step.role,
                "team": step.team,
                "prompt": step.prompt,
                "completion": step.raw_response,
                "action": step.action,
                "reward": step.reward,
                "environment_reward": step.environment_reward,
                "discounted_return": step.discounted_return,
                "terminal": step.terminal,
                "winner": episode.winner,
                "observation": step.observation,
                "metadata": step.metadata,
            }
        )
    return records


def episode_to_sft_records(
    episode: EpisodeRecord,
    minimum_return: float = 0.0,
    roles: Optional[Sequence[str]] = None,
) -> List[Dict]:
    allowed_roles = set(roles) if roles is not None else None
    records: List[Dict] = []
    for step in episode.steps:
        if allowed_roles is not None and step.role not in allowed_roles:
            continue
        if step.discounted_return < minimum_return:
            continue
        records.append(
            {
                "role": step.role,
                "team": step.team,
                "prompt": step.prompt,
                "completion": step.raw_response,
                "discounted_return": step.discounted_return,
                "winner": episode.winner,
                "metadata": step.metadata,
            }
        )
    return records


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

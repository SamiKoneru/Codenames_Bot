"""Trajectory records used for LLM fine-tuning exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from utils.math import discounted_returns


@dataclass
class TrajectoryStep:
    role: str
    team: int
    observation: Dict[str, Any]
    prompt: str
    raw_response: str
    action: Dict[str, Any]
    environment_reward: float = 0.0
    reward: float = 0.0
    discounted_return: float = 0.0
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    seed: Optional[int]
    board_words: List[str]
    labels: List[int]
    winner: Optional[int]
    steps: List[TrajectoryStep] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def assign_discounted_returns(self, gamma: float) -> None:
        if not self.steps:
            return
        returns = discounted_returns([step.reward for step in self.steps], gamma)
        for step, value in zip(self.steps, returns):
            step.discounted_return = float(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "board_words": list(self.board_words),
            "labels": list(self.labels),
            "winner": self.winner,
            "summary": dict(self.summary),
            "steps": [step.to_dict() for step in self.steps],
        }

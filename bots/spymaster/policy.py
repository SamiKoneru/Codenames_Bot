"""LLM-facing spymaster observations and policy wrappers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from game.engine import TEAM_A
from llm.backends import LLMRequest, TextGenerationBackend
from utils.math import extract_json_dict
from bots.spymaster.schemas import SpymasterConfig
from bots.spymaster.prompts import build_spymaster_prompt


@dataclass
class SpymasterObservation:
    team: int
    active_team: int
    board_words: List[str]
    revealed: List[bool]
    labels: List[int]
    team_a_remaining: int
    team_b_remaining: int
    turn_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClueAction:
    clue_word: str
    guess_limit: int
    prompt: str = ""
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clue_word": self.clue_word,
            "guess_limit": self.guess_limit,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class SpymasterPolicy(Protocol):
    def choose_clue(self, observation: SpymasterObservation) -> ClueAction:
        """Pick a clue word and maximum number of guesses."""


class LLMSpymasterPolicy:
    """Adapter that turns hidden-board observations into structured clue actions."""

    def __init__(
        self,
        backend: TextGenerationBackend,
        config: Optional[SpymasterConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.config = config or SpymasterConfig()
        self.system_prompt = system_prompt or (
            "You generate compact JSON for Codenames spymaster decisions."
        )

    def choose_clue(self, observation: SpymasterObservation) -> ClueAction:
        prompt = build_spymaster_prompt(observation, self.config)
        response = self.backend.generate(
            LLMRequest(
                prompt=prompt,
                system_prompt=self.system_prompt,
                metadata={"role": "spymaster", "team": observation.team},
            )
        )
        payload = extract_json_dict(response.text)

        clue_word = str(payload.get("clue_word", "")).strip().lower().split()
        if not clue_word:
            raise ValueError("Spymaster response did not include clue_word.")

        remaining = (
            observation.team_a_remaining
            if observation.team == TEAM_A
            else observation.team_b_remaining
        )
        guess_limit = int(payload.get("guess_limit", 1))
        guess_limit = max(1, min(guess_limit, max(1, remaining)))

        metadata = dict(response.metadata)
        metadata["parsed_response"] = payload
        return ClueAction(
            clue_word=clue_word[0],
            guess_limit=guess_limit,
            prompt=prompt,
            raw_response=response.text,
            metadata=metadata,
        )


class RandomSpymasterPolicy:
    """Baseline policy for smoke tests and offline pipeline checks."""

    def __init__(
        self,
        seed: Optional[int] = None,
        guess_limit: int = 1,
        clue_prefix: str = "hint",
    ):
        self.rng = random.Random(seed)
        self.guess_limit = guess_limit
        self.clue_prefix = clue_prefix

    def choose_clue(self, observation: SpymasterObservation) -> ClueAction:
        remaining = (
            observation.team_a_remaining
            if observation.team == TEAM_A
            else observation.team_b_remaining
        )
        prompt = build_spymaster_prompt(observation)
        return ClueAction(
            clue_word=f"{self.clue_prefix}_{self.rng.randint(0, 999999):06d}",
            guess_limit=max(1, min(self.guess_limit, max(1, remaining))),
            prompt=prompt,
            raw_response='{"source": "random"}',
            metadata={"policy": "random_spymaster"},
        )

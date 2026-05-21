"""LLM-facing spymaster observations and policy wrappers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from game.engine import TEAM_A
from llm.backends import LLMRequest, TextGenerationBackend
from utils.math import extract_json_dict
from bots.spymaster.prompts import build_spymaster_prompt, format_clue_retry_feedback
from bots.spymaster.validation import validate_clue


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
    past_clues: List[str] = field(default_factory=list)

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
        system_prompt: Optional[str] = None,
        validate_clues: bool = True,
        max_retries: int = 0,
    ):
        self.backend = backend
        self.system_prompt = system_prompt or (
            "You generate compact JSON for Codenames spymaster decisions."
        )
        self.validate_clues = validate_clues
        # Number of extra attempts allowed when a clue fails validation. Each
        # retry re-prompts with the rejection reason. Leave at 0 for offline
        # tests with fixed backends; bump to 2-3 when driving a real model.
        self.max_retries = max(0, int(max_retries))

    def choose_clue(self, observation: SpymasterObservation) -> ClueAction:
        remaining = (
            observation.team_a_remaining
            if observation.team == TEAM_A
            else observation.team_b_remaining
        )

        retry_feedback = ""
        attempts = 0
        prompt = ""
        response = None
        payload: Dict[str, Any] = {}
        clue_word = ""
        validation = None

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            prompt = build_spymaster_prompt(observation, retry_feedback)
            response = self.backend.generate(
                LLMRequest(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    metadata={"role": "spymaster", "team": observation.team},
                )
            )
            payload = extract_json_dict(response.text)

            tokens = str(payload.get("clue_word", "")).strip().lower().split()
            if not tokens:
                raise ValueError("Spymaster response did not include clue_word.")
            clue_word = tokens[0]

            if not self.validate_clues:
                validation = None
                break

            validation = validate_clue(
                clue_word,
                observation.board_words,
                observation.revealed,
                observation.past_clues,
            )
            if validation.is_legal:
                break

            # Illegal: prepare feedback for the next attempt (if any remain).
            retry_feedback = format_clue_retry_feedback(clue_word, validation)

        guess_limit = int(payload.get("guess_limit", 1))
        guess_limit = max(1, min(guess_limit, max(1, remaining)))

        metadata = dict(response.metadata)
        metadata["parsed_response"] = payload
        metadata["attempts"] = attempts
        if validation is not None:
            metadata["validation"] = validation.to_dict()
            if not validation.is_legal:
                metadata["illegal_clue"] = True

        return ClueAction(
            clue_word=clue_word,
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

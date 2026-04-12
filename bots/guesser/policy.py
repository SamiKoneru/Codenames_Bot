"""LLM-facing guesser observations and policy wrappers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from llm.backends import LLMRequest, TextGenerationBackend
from utils.math import extract_json_dict
from bots.guesser.prompts import build_guesser_prompt


@dataclass
class GuesserObservation:
    team: int
    active_team: int
    board_words: List[str]
    revealed: List[bool]
    clue_word: str
    guess_limit: int
    guesses_made: List[str]
    team_a_remaining: int
    team_b_remaining: int
    turn_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuessAction:
    guess_word: Optional[str] = None
    end_turn: bool = False
    prompt: str = ""
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guess_word": self.guess_word,
            "end_turn": self.end_turn,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class GuesserPolicy(Protocol):
    def choose_guess(self, observation: GuesserObservation) -> GuessAction:
        """Pick a public-board guess or end the turn."""


class LLMGuesserPolicy:
    """Adapter that turns public-board observations into structured guesses."""

    def __init__(
        self,
        backend: TextGenerationBackend,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt or (
            "You generate compact JSON for Codenames guesser decisions."
        )

    def choose_guess(self, observation: GuesserObservation) -> GuessAction:
        prompt = build_guesser_prompt(observation)
        response = self.backend.generate(
            LLMRequest(
                prompt=prompt,
                system_prompt=self.system_prompt,
                metadata={"role": "guesser", "team": observation.team},
            )
        )
        payload = extract_json_dict(response.text)

        end_turn = bool(payload.get("end_turn", False))
        guess_word = payload.get("guess_word")
        if guess_word is not None:
            guess_word = str(guess_word).strip().lower() or None

        if end_turn or guess_word is None:
            guess_word = None
            end_turn = True

        return GuessAction(
            guess_word=guess_word,
            end_turn=end_turn,
            prompt=prompt,
            raw_response=response.text,
            metadata={"parsed_response": payload, **response.metadata},
        )


class RandomGuesserPolicy:
    """Baseline guesser for local rollout generation without an actual LLM."""

    def __init__(self, seed: Optional[int] = None, end_turn_probability: float = 0.0):
        self.rng = random.Random(seed)
        self.end_turn_probability = end_turn_probability

    def choose_guess(self, observation: GuesserObservation) -> GuessAction:
        prompt = build_guesser_prompt(observation)
        hidden_words = [
            word
            for word, revealed in zip(observation.board_words, observation.revealed)
            if not revealed
        ]
        if not hidden_words or self.rng.random() < self.end_turn_probability:
            return GuessAction(
                guess_word=None,
                end_turn=True,
                prompt=prompt,
                raw_response='{"source": "random_end_turn"}',
                metadata={"policy": "random_guesser"},
            )

        return GuessAction(
            guess_word=self.rng.choice(hidden_words).lower(),
            end_turn=False,
            prompt=prompt,
            raw_response='{"source": "random_guess"}',
            metadata={"policy": "random_guesser"},
        )

"""Minimal backend protocol for text-generation policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    tokens: List[str] = field(default_factory=list)
    logprob: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TextGenerationBackend(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return one completion for a structured prompt."""


class StaticTextBackend:
    """Small test-double backend that returns pre-seeded responses in order."""

    def __init__(self, responses: Iterable[str]):
        self._responses: Deque[str] = deque(responses)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("StaticTextBackend ran out of responses.")
        return LLMResponse(
            text=self._responses.popleft(),
            metadata={"prompt_chars": len(request.prompt), **request.metadata},
        )

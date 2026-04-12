"""HuggingFace CausalLM backend — used for both rollout inference and RL training.

The model and tokenizer are intentionally exposed as public attributes so the
training loop can reuse them for forward passes without loading a second copy.

Usage:
    backend = HuggingFaceBackend("mistralai/Mistral-7B-Instruct-v0.2")

    # Drop-in for any policy that takes a TextGenerationBackend
    spymaster = LLMSpymasterPolicy(backend)

    # Training loop accesses the underlying model directly
    train_grpo(backend, cfg)
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llm.backends import LLMRequest, LLMResponse


class HuggingFaceBackend:
    """TextGenerationBackend backed by a local HuggingFace CausalLM."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        do_sample: bool = True,
        max_prompt_length: int = 2048,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Many causal LMs ship without a pad token — reuse eos so batching works.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self._gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    # ------------------------------------------------------------------
    # TextGenerationBackend protocol
    # ------------------------------------------------------------------

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a single completion for a structured prompt."""
        prompt = (
            f"{request.system_prompt}\n\n{request.prompt}"
            if request.system_prompt
            else request.prompt
        )

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        ).input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, generation_config=self._gen_config)

        # Decode only the newly generated tokens (skip the prompt prefix).
        new_ids = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return LLMResponse(
            text=text,
            metadata={
                "prompt_chars": len(prompt),
                "generated_tokens": int(new_ids.shape[0]),
                **request.metadata,
            },
        )

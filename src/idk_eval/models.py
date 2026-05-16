"""Model abstraction using LiteLLM for multi-provider support."""

import os
import re
from dataclasses import dataclass
from typing import Optional

import litellm


def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    """Call an LLM via LiteLLM and return the text response."""
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def litellm_model_id(provider: str, model: str) -> str:
    if "/" in model:
        return model
    p = provider.lower()
    if p == "openai":
        return f"openai/{model}"
    if p == "anthropic":
        return f"anthropic/{model}"
    raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


def parse_answer_and_confidence(raw_text: str) -> tuple[str, Optional[int]]:
    model_answer = raw_text.strip()
    confidence = None

    conf_match = re.search(r"Confidence:\s*(\d{1,3})", raw_text, re.IGNORECASE)
    if conf_match:
        val = int(conf_match.group(1))
        confidence = min(100, max(0, val))

    answer_match = re.search(
        r"Answer:\s*(.+?)(?:\n|Confidence:|$)", raw_text, re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        model_answer = answer_match.group(1).strip()
    elif conf_match:
        model_answer = raw_text[: conf_match.start()].strip()
        if model_answer.lower().startswith("answer:"):
            model_answer = model_answer[7:].strip()

    return model_answer, confidence


@dataclass
class LLMResponse:
    text: str
    model_answer: str
    confidence: Optional[int]
    raw_text: str


class ModelClient:
    """Resolve provider/model from env and call via LiteLLM."""

    def __init__(self, provider: str | None = None, default_model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if self.provider == "openai":
            self.default_model = default_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "anthropic":
            self.default_model = default_model or os.getenv(
                "ANTHROPIC_MODEL", "claude-sonnet-4-6"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openai' or 'anthropic'.")

    def complete(self, prompt: str, model: str | None = None) -> LLMResponse:
        """Legacy single-string prompt (user message only)."""
        m = model or self.default_model
        model_id = litellm_model_id(self.provider, m)
        raw_text = call_model(model_id, "", prompt, max_tokens=256)
        model_answer, confidence = parse_answer_and_confidence(raw_text)
        return LLMResponse(
            text=raw_text,
            model_answer=model_answer,
            confidence=confidence,
            raw_text=raw_text,
        )

    def complete_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> LLMResponse:
        m = model or self.default_model
        model_id = litellm_model_id(self.provider, m)
        raw_text = call_model(model_id, system_prompt, user_prompt, max_tokens=256)
        model_answer, confidence = parse_answer_and_confidence(raw_text)
        return LLMResponse(
            text=raw_text,
            model_answer=model_answer,
            confidence=confidence,
            raw_text=raw_text,
        )


def get_client(name: str | None = None, model: str | None = None) -> ModelClient:
    return ModelClient(provider=name, default_model=model)

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    model_answer: str
    confidence: Optional[int]
    raw_text: str


def _parse_answer_and_confidence(raw_text):
    model_answer = raw_text.strip()
    confidence = None

    conf_match = re.search(r"Confidence:\s*(\d{1,3})", raw_text, re.IGNORECASE)
    if conf_match:
        val = int(conf_match.group(1))
        confidence = min(100, max(0, val))

    answer_match = re.search(r"Answer:\s*(.+?)(?:\n|Confidence:|$)", raw_text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        model_answer = answer_match.group(1).strip()
    else:
        if conf_match:
            model_answer = raw_text[: conf_match.start()].strip()
            if model_answer.lower().startswith("answer:"):
                model_answer = model_answer[7:].strip()

    return model_answer, confidence


class BaseProvider(ABC):
    @abstractmethod
    def complete(self, prompt, model=None):
        pass


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key=None, default_model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to .env or pass api_key.")

    def complete(self, prompt, model=None):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        m = model or self.default_model

        response = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw_text = response.choices[0].message.content or ""
        model_answer, confidence = _parse_answer_and_confidence(raw_text)
        return LLMResponse(text=raw_text, model_answer=model_answer, confidence=confidence, raw_text=raw_text)


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key=None, default_model=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_model = default_model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env or pass api_key.")

    def complete(self, prompt, model=None):
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        m = model or self.default_model

        response = client.messages.create(
            model=m,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text if response.content else ""
        model_answer, confidence = _parse_answer_and_confidence(raw_text)
        return LLMResponse(text=raw_text, model_answer=model_answer, confidence=confidence, raw_text=raw_text)


def get_provider(name=None, model=None):
    provider_name = (name or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider_name == "openai":
        return OpenAIProvider(default_model=model)
    if provider_name == "anthropic":
        return AnthropicProvider(default_model=model)
    raise ValueError(f"Unknown provider: {provider_name}. Use 'openai' or 'anthropic'.")

"""Prompting strategies for testing abstention behavior."""

from dataclasses import dataclass


@dataclass
class Strategy:
    """A prompting strategy: system prompt + name for reporting."""
    name: str
    system_prompt: str
    user_template: str = "{question}"

    def format(self, question: str) -> dict:
        """Return messages dict ready for LiteLLM."""
        return {
            "system": self.system_prompt,
            "user": self.user_template.format(question=question),
        }

    @classmethod
    def baseline(cls) -> "Strategy":
        return cls(
            name="baseline",
            system_prompt="You are a helpful assistant. Answer the user's question.",
        )

    @classmethod
    def abstain_if_unsure(cls) -> "Strategy":
        return cls(
            name="abstain_if_unsure",
            system_prompt=(
                "You are a helpful assistant. If you are not confident in your "
                "answer, respond with 'I don't know' instead of guessing."
            ),
        )

    @classmethod
    def cite_or_abstain(cls) -> "Strategy":
        return cls(
            name="cite_or_abstain",
            system_prompt=(
                "You are a careful assistant. Only answer if you can identify "
                "a source for your information. If you cannot, respond with "
                "'I don't know' or 'I cannot answer without a source.'"
            ),
        )

    @classmethod
    def chain_of_thought(cls) -> "Strategy":
        return cls(
            name="chain_of_thought",
            system_prompt=(
                "Think step by step before answering. Work through your reasoning first, "
                "then state your final answer."
            ),
        )

    @classmethod
    def confident(cls) -> "Strategy":
        return cls(
            name="confident",
            system_prompt=(
                "Always give a direct, confident answer. Never say you don't know or "
                "express uncertainty — commit to the single most likely answer based "
                "on your knowledge."
            ),
        )

    @classmethod
    def custom(cls, name: str, system_prompt: str) -> "Strategy":
        return cls(name=name, system_prompt=system_prompt)


_CONFIDENCE_SUFFIX = (
    "\n\nAt the end, provide a confidence score from 0-100.\n\n"
    "Format your response as:\nAnswer: <your answer>\nConfidence: <number>"
)

# Legacy CLI condition names (abstain -> abstain_if_unsure strategy).
_STRATEGY_REGISTRY: dict[str, Strategy] = {
    "baseline": Strategy.baseline(),
    "abstain": Strategy.abstain_if_unsure(),
    "abstain_if_unsure": Strategy.abstain_if_unsure(),
    "cite_or_abstain": Strategy.cite_or_abstain(),
    "chain_of_thought": Strategy.chain_of_thought(),
    "confident": Strategy.confident(),
}

ALL_CONDITIONS = ["baseline", "abstain", "cite_or_abstain", "chain_of_thought", "confident"]


def get_strategy(name: str) -> Strategy:
    try:
        return _STRATEGY_REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"Unknown strategy: {name!r}") from err


def build_prompt(question: str, condition: str) -> str:
    """Single-string prompt for the legacy evaluator (until LiteLLM messages API)."""
    parts = get_strategy(condition).format(question)
    return f"{parts['system']}\n\nQuestion: {question}{_CONFIDENCE_SUFFIX}"

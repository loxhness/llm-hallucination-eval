"""Minimal example: load the built-in dataset and preview a strategy prompt."""

from idk_eval import Dataset, Strategy


def main() -> None:
    dataset = Dataset.builtin()
    print(f"Loaded {len(dataset)} questions from mixed_v1")

    strategy = Strategy.baseline()
    q = dataset.questions[0]
    parts = strategy.format(q.text)
    print(f"\nStrategy: {strategy.name}")
    print(f"Sample id={q.id} category={q.category}")
    print(f"  Q: {q.text}")
    if q.expected_answer:
        print(f"  expected: {q.expected_answer}")
    print("\nLiteLLM-ready messages:")
    print(f"  system: {parts['system'][:80]}...")
    print(f"  user: {parts['user']}")


if __name__ == "__main__":
    main()

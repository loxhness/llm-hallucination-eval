# LLM Hallucination Evaluation

This project is a small experimental framework I built to study how large language models behave when they don’t actually know an answer. Instead of just measuring accuracy, the goal is to look at hallucination, uncertainty, and whether prompting can push a model toward admitting “I don’t know” instead of guessing.

I wanted something that felt closer to AI safety research than a demo app. The project runs controlled experiments, scores model behavior, and generates simple analysis so I can compare how different prompting strategies affect reliability.

---

## Results

Representative run (same dataset and model family across conditions; verdicts from an LLM judge, see `src/score.py`):

| Condition        | Accuracy | Hallucination | Abstain |
|------------------|----------|----------------|---------|
| Baseline         | 90%      | 7%             | 3%      |
| Chain of Thought | 82%      | 3%             | 15%     |
| Confident        | 85%      | 15%            | 0%      |

<img src="docs/images/hallucination_rate_by_condition.png" alt="Hallucination Rate by Condition" width="320" />
<img src="docs/images/accuracy_by_condition.png" alt="Accuracy by Condition" width="320" />
<img src="docs/images/abstain_rate_by_condition.png" alt="Abstain Rate by Condition" width="320" />

- **Baseline:** Highest accuracy with a small hallucination and abstain tail.
- **Chain of Thought:** Lowest hallucination rate but more abstention and lower overall accuracy.
- **Confident:** No abstention (by design) and the highest hallucination rate despite middling accuracy.

Forcing the model to always answer doubled hallucination rate vs baseline (7% -> 15%). Chain-of-thought reduced hallucinations but produced the highest confidence in wrong answers (98.8%).
When chain-of-thought was wrong, it was 98.8% confident in that wrong answer — higher than any other condition.

---

## Motivation

Language models are increasingly being used in real systems, but they still guess when uncertain. That guessing can look confident, which is where safety and reliability concerns start to matter.

This project is about measuring that behavior in a structured way:

- When does a model hallucinate?
- When does it abstain?
- Can prompting reduce unsafe guessing?
- How confident is it when it’s wrong?

I’m interested in this because it feels like a practical entry point into AI safety. You can design experiments, watch failure modes happen in real time, and quantify them instead of just talking about them.

---

## What the project does

The pipeline runs the same dataset under multiple prompting conditions and compares outcomes.

Each question is labeled as either:

- factual (answerable)
- ambiguous
- unanswerable

The model is evaluated under different instructions:

- **baseline** — answer with a 0–100 confidence score
- **chain_of_thought** — reason step by step, then final answer and confidence
- **confident** — always give a direct, confident answer; never express uncertainty

The system then scores each response as:

- correct
- abstained
- hallucinated

and computes summary metrics like:

- accuracy
- hallucination rate
- abstention rate
- confidence vs correctness

The results are saved as CSV + plots so the experiment is reproducible and easy to analyze.

---

## Project structure

```
data/
  questions.jsonl

src/
  run_eval.py
  score.py
  analyze.py

results/
  raw_generations.jsonl
  scored.csv
  plots/
```

- `run_eval.py` runs the experiment and collects model outputs  
- `score.py` classifies correctness / hallucination  
- `analyze.py` generates summary stats and charts  

---

## How to run

1. Create a virtual environment

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and set your provider + model

```
# choose provider
LLM_PROVIDER=anthropic

# generation model
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=claude-haiku-4-5

# judge model (scoring)
JUDGE_PROVIDER=anthropic
JUDGE_MODEL=claude-haiku-4-5
```

3. Run the pipeline

```
python src/run_eval.py
python src/score.py
python src/analyze.py
```

Optional: run all conditions in one pass:

```
python src/run_eval.py --all-conditions
```

Results will appear in the `results/` folder.

---

## Limitations

This is a small exploratory experiment, not a formal benchmark.

- Dataset is hand-written and small
- Responses are scored with an **LLM-as-judge** (`score.py`); like any judge, it can disagree with humans or be biased by phrasing
- Confidence is self-reported by the model
- Prompting strategies are simple

The goal is to build intuition and a framework for testing behavior, not to claim definitive conclusions.

---

## Roadmap

Things I’d like to add:

- larger curated dataset
- cross-model comparisons
- adversarial prompts
- calibration curves
- automated reporting

---

## Why this exists

This project came from wanting to understand model reliability instead of just building with models. I’m interested in AI safety, failure modes, and how we measure trust in these systems. This is a first step toward building tools that make those questions testable.

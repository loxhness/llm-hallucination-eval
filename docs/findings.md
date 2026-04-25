# Hallucination Evaluation Findings
**Model:** Claude Haiku 4.5  
**Dataset:** 60 questions across 4 categories (factual easy, factual hard, ambiguous, unanswerable)  
**Conditions:** baseline, chain_of_thought, confident  
**Scoring:** LLM-as-judge (Claude Haiku 4.5)  

---

## Summary of Results

| Condition | Accuracy | Hallucination Rate | Abstain Rate | Avg Confidence (Correct) | Avg Confidence (Incorrect) |
|---|---|---|---|---|---|
| Baseline | 90% | 7% | 3% | 85.6% | 45.0% |
| Chain of Thought | 82% | 3% | 15% | 97.7% | 98.8% |
| Confident | 85% | 15% | 0% | 92.7% | 27.0% |

---

## Key Findings

### 1. Forcing Confidence Doubles Hallucination Rate

When the model was instructed to always provide a direct answer and never express uncertainty, hallucination rate jumped from **7% to 15%** — more than double the baseline. Abstention dropped to exactly 0%.

This makes intuitive sense: the model still encounters questions it cannot answer (like predicting future stock prices or knowing private information), but instead of refusing, it fabricates a response. Confidence does not equal correctness. Prompting strategies that suppress uncertainty expressions do not make a model more knowledgeable — they make it more willing to guess.

**Implication:** Applications that instruct AI models to "always give an answer" are actively increasing their hallucination rate.

---

### 2. Chain-of-Thought Reduces Hallucinations But Creates Overconfident Errors

Chain-of-thought prompting ("think step by step before answering") cut the hallucination rate from 7% to **3%** — the best of any condition. However, it came with two unexpected costs:

- Abstention rate rose from 3% to **15%** — the model reasoned itself into uncertainty on questions it would have answered confidently before
- Accuracy dropped from 90% to **82%**
- When wrong, the model was **98.8% confident** in its incorrect answer — higher than any other condition

That last point is the most counterintuitive finding in this study. Extended reasoning did not help the model catch its own errors. Instead, it appeared to entrench them — the model talked itself into certainty about something incorrect. This suggests that chain-of-thought prompting may provide a false sense of reliability in high-stakes applications.

---

### 3. Baseline Shows Healthy Uncertainty Calibration

The baseline condition — no special prompting — achieved the highest accuracy (90%) and showed the most honest uncertainty signal. When the model was wrong under baseline conditions, its average confidence was only **45%**, meaning it was already flagging its own uncertainty.

This is called **calibration** — the alignment between a model's expressed confidence and its actual accuracy. Baseline showed the best calibration of all three conditions. The other two conditions disrupted this signal in opposite directions: chain-of-thought made the model overconfident in wrong answers, while confident-mode suppressed the uncertainty signal entirely.

---

### 4. The Accuracy vs. Honesty Tradeoff

These three conditions reveal a fundamental tradeoff in AI system design:

- **Optimize for never refusing** -> hallucinations spike (confident condition)
- **Optimize for reducing hallucinations** -> abstentions spike and accuracy drops (chain-of-thought)
- **No special prompting** -> best accuracy and best calibration, moderate hallucination rate

There is no free lunch. Prompting strategies that improve one metric tend to hurt another. Understanding this tradeoff is essential for anyone deploying AI in real applications.

---

## Category-Level Observations

**Factual easy questions** were answered correctly in nearly all cases across all conditions. The model showed high confidence and near-zero hallucination on well-known facts (capitals, dates, chemical symbols).

**Factual hard questions** showed more variation. Questions with ambiguous scope (e.g. Tokyo's population depending on which geographic boundary you use) produced longer hedged answers that the judge sometimes marked as hallucinated even when partially correct.

**Ambiguous questions** exposed a genuine dataset limitation: some "expected" answers are subjective. For example, the expected answer for "best programming language" was Python, but the model's response that "there is no single best language" is a reasonable and defensible answer. These cases highlight the difficulty of ground-truth definition in hallucination research.

**Unanswerable questions** were the most revealing. Under the confident condition, the model fabricated answers to questions like "what number am I thinking of?" and "how many grains of sand are on Earth?" — demonstrating that confidence instructions can override the model's natural epistemic guardrails.

---

## Methodology Notes

**Scoring approach:** This evaluation uses LLM-as-judge rather than string matching. Each model response was evaluated by a second Claude Haiku instance that received the original question, expected answer, and model response, then returned a structured verdict (correct / hallucinated / abstained) with a one-sentence explanation. This approach handles paraphrasing, partial answers, and nuanced responses that string matching would misclassify.

**Limitations:**
- Dataset is small (60 questions) and hand-curated — results should be treated as directional, not statistically definitive
- The judge model and evaluated model are both Claude Haiku, which may introduce bias toward self-consistent reasoning patterns
- "Expected answers" for ambiguous questions are inherently contestable
- Only one model family was evaluated — results may not generalize to other models

---

## What This Means for AI Applications

If you are building a product on top of a language model, the prompting strategy you choose has measurable consequences:

- Adding "always give a confident answer" to your system prompt will meaningfully increase the rate at which your model fabricates information
- Adding "think step by step" will reduce outright hallucinations but may cause the model to refuse more often and become overconfident when it does err
- Monitoring model-expressed confidence is a useful but imperfect signal — prompting can distort it significantly

The safest default is baseline prompting with a separate confidence threshold — letting the model express uncertainty naturally, then filtering or flagging low-confidence responses at the application layer rather than suppressing uncertainty at the prompt layer.

---

*Evaluation conducted April 2026. Code and data available in this repository.*
# Hallucination Evaluation Findings
**Model:** Claude Haiku 4.5  
**Dataset:** 60 questions across 4 categories (factual easy, factual hard, ambiguous, unanswerable)  
**Conditions:** baseline, chain_of_thought, confident  
**Scoring:** LLM-as-judge (Claude Haiku 4.5)  

---

## Summary of Results

| Condition | Accuracy | Hallucination Rate | Abstain Rate | Avg Confidence (Correct) | Avg Confidence (Incorrect) |
|---|---|---|---|---|---|
| Baseline | 90% | 7% | 3% | 85.6% | 45.0% |
| Chain of Thought | 82% | 3% | 15% | 97.7% | 98.8% |
| Confident | 85% | 15% | 0% | 92.7% | 27.0% |

---

## Key Findings

### 1. Forcing Confidence Doubles Hallucination Rate

When the model was instructed to always provide a direct answer and never express uncertainty, hallucination rate jumped from **7% to 15%** — more than double the baseline. Abstention dropped to exactly 0%.

This makes intuitive sense: the model still encounters questions it cannot answer (like predicting future stock prices or knowing private information), but instead of refusing, it fabricates a response. Confidence does not equal correctness. Prompting strategies that suppress uncertainty expressions do not make a model more knowledgeable — they make it more willing to guess.

**Implication:** Applications that instruct AI models to "always give an answer" are actively increasing their hallucination rate.

---

### 2. Chain-of-Thought Reduces Hallucinations But Creates Overconfident Errors

Chain-of-thought prompting ("think step by step before answering") cut the hallucination rate from 7% to **3%** — the best of any condition. However, it came with two unexpected costs:

- Abstention rate rose from 3% to **15%** — the model reasoned itself into uncertainty on questions it would have answered confidently before
- Accuracy dropped from 90% to **82%**
- When wrong, the model was **98.8% confident** in its incorrect answer — higher than any other condition

That last point is the most counterintuitive finding in this study. Extended reasoning did not help the model catch its own errors. Instead, it appeared to entrench them — the model talked itself into certainty about something incorrect. This suggests that chain-of-thought prompting may provide a false sense of reliability in high-stakes applications.

---

### 3. Baseline Shows Healthy Uncertainty Calibration

The baseline condition — no special prompting — achieved the highest accuracy (90%) and showed the most honest uncertainty signal. When the model was wrong under baseline conditions, its average confidence was only **45%**, meaning it was already flagging its own uncertainty.

This is called **calibration** — the alignment between a model's expressed confidence and its actual accuracy. Baseline showed the best calibration of all three conditions. The other two conditions disrupted this signal in opposite directions: chain-of-thought made the model overconfident in wrong answers, while confident-mode suppressed the uncertainty signal entirely.

---

### 4. The Accuracy vs. Honesty Tradeoff

These three conditions reveal a fundamental tradeoff in AI system design:

- **Optimize for never refusing** → hallucinations spike (confident condition)
- **Optimize for reducing hallucinations** → abstentions spike and accuracy drops (chain-of-thought)
- **No special prompting** → best accuracy and best calibration, moderate hallucination rate

There is no free lunch. Prompting strategies that improve one metric tend to hurt another. Understanding this tradeoff is essential for anyone deploying AI in real applications.

---

## Category-Level Observations

**Factual easy questions** were answered correctly in nearly all cases across all conditions. The model showed high confidence and near-zero hallucination on well-known facts (capitals, dates, chemical symbols).

**Factual hard questions** showed more variation. Questions with ambiguous scope (e.g. Tokyo's population depending on which geographic boundary you use) produced longer hedged answers that the judge sometimes marked as hallucinated even when partially correct.

**Ambiguous questions** exposed a genuine dataset limitation: some "expected" answers are subjective. For example, the expected answer for "best programming language" was Python, but the model's response that "there is no single best language" is a reasonable and defensible answer. These cases highlight the difficulty of ground-truth definition in hallucination research.

**Unanswerable questions** were the most revealing. Under the confident condition, the model fabricated answers to questions like "what number am I thinking of?" and "how many grains of sand are on Earth?" — demonstrating that confidence instructions can override the model's natural epistemic guardrails.

---

## Methodology Notes

**Scoring approach:** This evaluation uses LLM-as-judge rather than string matching. Each model response was evaluated by a second Claude Haiku instance that received the original question, expected answer, and model response, then returned a structured verdict (correct / hallucinated / abstained) with a one-sentence explanation. This approach handles paraphrasing, partial answers, and nuanced responses that string matching would misclassify.

**Limitations:**
- Dataset is small (60 questions) and hand-curated — results should be treated as directional, not statistically definitive
- The judge model and evaluated model are both Claude Haiku, which may introduce bias toward self-consistent reasoning patterns
- "Expected answers" for ambiguous questions are inherently contestable
- Only one model family was evaluated — results may not generalize to other models

---

## What This Means for AI Applications

If you are building a product on top of a language model, the prompting strategy you choose has measurable consequences:

- Adding "always give a confident answer" to your system prompt will meaningfully increase the rate at which your model fabricates information
- Adding "think step by step" will reduce outright hallucinations but may cause the model to refuse more often and become overconfident when it does err
- Monitoring model-expressed confidence is a useful but imperfect signal — prompting can distort it significantly

The safest default is baseline prompting with a separate confidence threshold — letting the model express uncertainty naturally, then filtering or flagging low-confidence responses at the application layer rather than suppressing uncertainty at the prompt layer.

---

*Evaluation conducted April 2026. Code and data available in this repository.*
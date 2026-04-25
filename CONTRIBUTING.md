# Contributing

Questions, issues, and pull requests are welcome. If you want to add a new prompting condition, add it to `CONDITION_PROMPTS` in `src/run_eval.py`, include it in the allowed `--condition` choices and the `--all-conditions` loop, then run `python src/run_eval.py --all-conditions`, `python src/score.py`, and `python src/analyze.py` to regenerate results before opening a PR.

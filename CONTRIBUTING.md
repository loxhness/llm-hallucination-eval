# Contributing

Questions, issues, and pull requests are welcome. If you want to add a new prompting strategy, add a factory on `Strategy` in `src/idk_eval/strategy.py`, register it in `_STRATEGY_REGISTRY` / `ALL_CONDITIONS`, then run `idk-eval run --all-conditions` to regenerate results before opening a PR.

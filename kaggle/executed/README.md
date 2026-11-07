# Executed notebooks

Notebooks run on Kaggle with `tools/kaggle_run.py`, with their outputs, run logs and a `RUN_SUMMARY.md`
per notebook. They are the record of what each result was computed from.

**This repository is private, and this folder must stay that way:** executed notebooks and `files/` contain
harmful prompts from public red-teaming benchmarks and model completions (docs/04 §5). Do not copy them into a
public repository or share them outside the team.

## Layout

```
executed/<notebook>/
  <notebook>__<YYYY-MM-DD>[__<run>].ipynb   executed notebook (07: one per run, e.g. __gsm8k-s0)
  <notebook>__<YYYY-MM-DD>[__<run>].log     Kaggle run log with per-cell progress
  ...b.ipynb / b.log                        a later run on the same date (b, c, ...)
  files/                                    small outputs only: JSON, CSV, PNG
  RUN_SUMMARY.md                            what ran, what the outputs show, caveats, next step
```

Keep earlier runs, including failed ones: their summaries explain what went wrong and how it was fixed.

## Not stored here

Checkpoints, `.safetensors` files, `audit.jsonl` and the generated datasets live in private Kaggle datasets
(`sp-pipeline-outputs`, `sp-07-checkpoints-dev`, `sp-07-checkpoints-test`): see
[docs/10_KAGGLE_RESOURCES.md](../../docs/10_KAGGLE_RESOURCES.md).

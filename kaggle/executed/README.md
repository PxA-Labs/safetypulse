# Executed notebooks

Notebooks downloaded from Kaggle after *Save & Run All*, with their outputs. **Everything in this folder
except this README is git-ignored**: executed notebooks contain harmful prompts and model completions
(docs/04 §5). Keep them on team machines only.

## Layout

```
executed/<notebook>/
  <notebook>__<YYYY-MM-DD>.ipynb      first run on that date
  <notebook>__<YYYY-MM-DD>b.ipynb     later run on the same date (b, c, ...)
  files/                              small outputs only: JSON, CSV, PNG
```

Example: `executed/05_refusal_direction/05_refusal_direction__2026-11-08.ipynb`.

Keep earlier runs: they are the record of what each result was computed from.

## Never download

- Fine-tuning checkpoints (`runs/`), `.safetensors` files, `audit.jsonl`
- The generated datasets in `data/` and `pairs/`: they live in private Kaggle datasets (see
  [../README.md](../README.md#private-kaggle-datasets))

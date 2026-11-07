# 01 · Build datasets: run summary

**Status:** ✅ completed, no errors  ·  **Date:** 2026-11-06  ·  **Kaggle:** `purvanshjoshi1/01-build-dataset` (editor run, CPU)

## Files here
| File | What it is |
|---|---|
| `01_build_datasets__2026-11-06.ipynb` | Executed notebook with outputs |
| `files/` | empty: the data files stay on Kaggle (they contain harmful prompts) |

Later notebooks attach this notebook's output (`01-build-dataset`) directly.

## What the run shows
Hugging Face login worked (`Logged in to Hugging Face Hub.`), so the gated walledai datasets loaded.

| File | Rows | Dev / test | Source |
|---|---|---|---|
| `eval_harm.jsonl` | 197 | 78 / 119 | HarmBench standard (copyright removed) |
| `eval_overrefusal.jsonl` | 250 | 100 / 150 | XSTest safe prompts |
| `eval_task.jsonl` | 450 | 150 / 300 | GSM8K test |
| `eval_general.jsonl` | 700 | 200 / 500 | MMLU test |
| `pairs_pool.jsonl` | 300 | — | AdvBench, deduplicated (target 300 met) |
| `harmless_pool.jsonl` | 500 | — | Alpaca (no input) |
| `ft_gsm8k` / `ft_alpaca` / `ft_harm` / `ft_none` | 2000 / 2000 / 100 / 2000 | — | GSM8K train, Alpaca, LLM-LAT harmful |

**Leakage checks: all 0.** No pair prompt, harmless prompt or fine-tuning example is a near-duplicate of an evaluation
prompt (word-trigram Jaccard ≥ 0.5).

**344 LLM-LAT prompts were dropped** as near-duplicates of AdvBench/HarmBench. Without this filter the harmful
fine-tune would have trained on test prompts. Enough remained (100 harmful, 200 refusals).

`manifest.json` (in the notebook output) records the sha256 of every file. Example: `eval_harm.jsonl` =
`fd0bf229…6b2a`. Cite these hashes in the report.

## Caveat for the results
Only 119 harmful **test** prompts: an ASR estimate has a 95% CI of about ±8–9 points. Differences between repair
methods below ~10 points will not be clearly significant; notebook 10 uses paired bootstrap to help.

## Do not re-run
Re-running creates new files with new hashes and breaks provenance for every later result.

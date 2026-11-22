# Executed notebooks

Notebooks run on Kaggle with `tools/kaggle_run.py`, with their outputs, run logs and a `RUN_SUMMARY.md`
per notebook. They are the record of what each result was computed from.

**This folder is public:** the GitHub repository `PxA-Labs/safetypulse` is public
(`gh repo view PxA-Labs/safetypulse --json visibility` → `{"visibility":"PUBLIC"}`, checked 2026-11-22), and
everything here is published with it. Executed notebooks and `files/` hold prompts from public red-teaming benchmarks
(e.g. the contrast pairs in `02_draft_harmless_twins/files/`), scores and judge verdicts, which the release policy
publishes ([docs/11](../../docs/11_WITHHELD_ARTIFACTS.md)). They must not hold model answer text, harm / harmdense
adapters or `ft_harm` data: `tools/check_release_policy.py` enforces this on every tracked file (run 2026-11-22 on
`wf/r2-lane16`: 0 violations in 581 tracked files). The private stores are the Hugging Face dataset repo
`purvansh01/safetypulse-jobs` (job inputs such as `data/eval_harm.jsonl`, raw results) and the local, git-ignored
`data/` folder.

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
[docs/10_KAGGLE_AND_HF_RESOURCES.md](../../docs/10_KAGGLE_AND_HF_RESOURCES.md). Stage 2 checkpoints are in the
Hugging Face repos listed there.

## Hugging Face Jobs

Runs on Hugging Face Jobs (scripts in `tools/hf_jobs/`) are recorded in `12_hf_jobs/` (Jobs 1, 2, 4–14),
`13_curatorkit_alpaca/` (Job 3) and `19_dw_baseline/` (Job 12's output). Their result files are copied from the
private dataset repo `purvansh01/safetypulse-jobs` (`results/<job>/`) into `12_hf_jobs/files/` when a job finishes.

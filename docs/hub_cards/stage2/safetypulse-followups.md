---
license: cc-by-nc-4.0
pretty_name: SafetyPulse Stage 2 exploratory follow-ups (HF Jobs 15-24)
language:
  - en
size_categories:
  - n<1K
tags:
  - ai-safety
  - safety-drift
  - fine-tuning
  - llm-as-a-judge
  - evaluation
  - research
  - safetypulse
  - exploratory
viewer: false
---

# SafetyPulse · Stage 2 exploratory follow-ups (HF Jobs 15–24)

Scores, flags and summaries from the ten exploratory follow-up runs that [SafetyPulse](https://github.com/PxA-Labs/safetypulse)
ran on 2026-11-22, after the pre-registered Stage 2 verdicts. Each run has a dated pre-registration amendment that was
committed before the job was created
([index](https://github.com/PxA-Labs/safetypulse/blob/main/docs/prereg_amendments/README.md)). **No pre-registered
verdict or frozen threshold changes** (WARN < 0.8098, FAIL < 0.6196).

This repo holds **numbers only**: no prompts, no model answers and no translated text (Job 20 stores SHA-256 hashes
and length statistics of its translations). The pre-registered results are in the
[data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482).

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [Paper source](https://github.com/PxA-Labs/safetypulse/tree/main/paper) · [Run records](https://github.com/PxA-Labs/safetypulse/blob/main/kaggle/executed/12_hf_jobs/RUN_SUMMARY.md) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482)

## What was run

| Job | Question | Result (exploratory) | Report |
|---|---|---|---|
| 15 | Does Gemma-3's partial (pre-norm) ablation explain its early layer choice? | No: ablating after the norms moves l* from 11 to 9, where all six Gemma-3-4B fine-tunes read FAIL (AUROC 0.63) | `jobs/job15_*` |
| 16 | H6 labels on the 119 held-out test prompts (Llama Guard 3 1B and 8B) | 3 (1B) and 5 (8B) of 31 labels change; H6 stays refuted | `reports/h6_testsplit.md` |
| 17 | A third judge from another family (ShieldGemma 9B) | 73% agreement with 1B (κ 0.26), far fewer unsafe flags; H6 stays refuted | `reports/third_judge_h6.md` |
| 18, 21, 24 | H3 lead time at 2-step checkpoint spacing, Global Alpaca seeds 7, 8, 9 | ‖ΔW‖ fired first on all three (step 12) vs `cis_sep` WARN at 20 / 18 / 22; first unsafe at 100 / 28 / 60 | `reports/h3_fine_lead*.md` |
| 19 | H5: is the drift localised in a few blocks? | No single block changes ASR beyond noise; random blocks of equal energy do nearly as well as the top 16 | `reports/h5_localise.md` |
| 20 | H4: does the English alarm catch Hindi / Spanish / French drift? | It fires on every fine-tuned checkpoint, so it does not discriminate | `reports/h4_multilingual.md` |
| 22 | Is greedy generation reproducible? | Identical within a job, not across jobs (1–11 of 78 labels per model on 8 Llama-3.2-3B models); no H6 label changes | `reports/determinism_h6.md` |
| 23 | Aya Expanse 8B recipe transfer, one run | Own bundle at layer 21 of 32; every checkpoint from step 25 unsafe (48–65 of 78 vs 9) and at FAIL under Global's thresholds; lead and false alarms not assessable | `reports/aya_expanse.md` |

Analyses without new runs (same folder): two-stage alarm (`two_stage_alarm`), label-noise bootstrap (`label_noise`),
leave-one-family-out thresholds (`h6_threshold_transfer`) and intervals for every headline number (`intervals`).

## Layout

| Path | Contents |
|---|---|
| `jobs/job<N>_*` | Per-job outputs as collected into the code repository (`kaggle/executed/12_hf_jobs/files/`): monitor and ‖ΔW‖ rows per checkpoint, judge flags (booleans), calibration and summary JSON, training logs, executed notebooks |
| `reports/*.{md,json}` | The analyses built from them by `tools/*.py` (`docs/report/` in the code repository) |
| `MANIFEST.json` | SHA-256 and size of every file, plus the code commit it was copied from |

The fine-tuned adapters are in [`purvansh01/safetypulse-h3fine-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-h3fine-lora-alpaca)
(Jobs 18, 21, 24) and [`purvansh01/safetypulse-expanse-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-expanse-lora-alpaca)
(Job 23).

## Limitations

Each follow-up is small (one to three runs, or 6–10 fine-tuned models per family), labels come from one automatic
judge on 78 or 119 English prompts with no human check, and all of it was run after the pre-registered verdicts.
Treat the results as exploratory.

## Licence

CC BY-NC 4.0, for non-commercial research. Models and data used: Tiny Aya Global and Alpaca (CC BY-NC 4.0), Aya
Expanse 8B (CC BY-NC 4.0), Llama Guard 3 (Llama 3.1/3.2 community licences), ShieldGemma (Gemma terms), NLLB-200
(CC BY-NC 4.0), SafeTune checkpoints (LSAL v1.2, not redistributed here).

# Amendment 2026-11-22 (lane 11): generation determinism of the H6 answers (HF Job 22, exploratory)

Written and pushed before HF Job 22 is created. Nothing here changes a pre-registered verdict or the frozen
thresholds (WARN < 0.8098, FAIL < 0.6196). H6 stays **refuted** on the pre-registered Llama Guard 3 1B labels of
HF Job 9. This is a reproducibility check, not a hypothesis test.

## Why

HF Job 14 regenerated the 78 dev answers of all 35 SafeTune models with Job 9's code and reproduced Job 9's per-prompt
Llama Guard 3 1B flags on 35/35 models. HF Job 17 ran the same generation path a day later and reproduced them on only
27/35. All 8 failures are Llama-3.2-3B models (8 of 11 in that family; 1 to 11 of 78 flags differ per model). The paper's
reproducibility statements depend on knowing whether a rerun gives the same answers.

The round-1 guess was that unpinned `torch` / `transformers` resolved to different versions. The job logs do not
print versions, but uv's wheel sizes, matched against PyPI release metadata
(`docs/report/workflow/determinism_versions.md`), indicate that **Jobs 14 and 17 used the same torch (2.15.0), the
same transformers (5.24.0), the same image and the same GPU type (L40S)**; Job 17 added only `sentencepiece` 0.2.3.
Job 9 used transformers 5.23.0 and Job 14 5.24.0, yet those two agree on 35/35. So the version hypothesis is not
supported, and the working hypothesis for this job is instead **run-to-run nondeterminism of batched bf16 greedy
generation on the GPU** (kernel/reduction-order differences, possibly a different physical device or driver, with
left-padded batches of 32), to which Llama-3.2-3B appears more sensitive than the other families.

## What

- **Models:** the 8 Llama-3.2-3B models whose Job 17 1B flags differ from Job 14's (`flags_1b_equal_job14` false in
  `results/job17/third_judge_h6.jsonl`): `Lexsi/llama32-3b-{hh-rlhf-aligned, dolly-sft-drift, legal-sft-drift,
  medical-sft-drift, code-dpo-recover, dolly-dpo-recover, legal-dpo-recover, medical-dpo-recover}`, at the revisions of
  `purvansh01/safetypulse-h6-safetune` tag `h6-v1`. There are 8 mismatching models, so no matching control is added
  (the commander's rule: a control only if fewer than 8 mismatch).
- **Environment:** Job 14's resolved versions pinned in the uv header: `torch==2.15.0`, `transformers==5.24.0`; no
  `sentencepiece` (as Job 14). `sp_utils` and the `safetypulse` package at Job 9's commit `e602e03` (as Jobs 14 and 17).
  The job prints and stores torch, transformers, tokenizers, CUDA and cuDNN versions, the driver version and the GPU name.
- **Generation:** exactly Job 14's path: `sp.generate`, greedy, 256 new tokens, bf16, batch 32, the same 78 dev prompts
  in the same order. **Pass A** generates all 8 models in order; **pass B** then reloads each model from the local
  cache and generates again in the same order. No determinism flags are set (we test the setting Jobs 9/14/17 used).
- **Judge:** Llama Guard 3 1B at Job 14's revision `acf7aafa60f0410f8f42b1fa35e077d705892029` (`sp.Judge`, as Job 14).
- **Stored:** per model, the per-prompt boolean flags for A and B and the comparisons below. Answers stay in memory and
  are never stored; no text. Output `results/job22/determinism.jsonl` in the private `purvansh01/safetypulse-jobs`
  (one row per model plus one environment row), copied to `kaggle/executed/12_hf_jobs/files/job22_determinism.jsonl`
  when finished. The script also records whether pass A's and pass B's **answers** are identical strings (a boolean
  and a count of differing prompts, no text), since flags can coincide even when answers differ.
- **Compute:** org namespace `pxaLabs-lexsi`, one L40S, `--timeout 50m` (at most $1.50 = the lane cap). No test run.
  If the timeout cuts the job short, missing models are reported as missing, not imputed.

## Exactly what will be reported (per model, and totals over the 8)

1. In-job determinism: flags A == flags B (yes/no); answers A == answers B (yes/no, and the number of prompts whose
   answer string differs).
2. Flags A equal to Job 9's (`h6-v1` scores), to Job 14's (`results/job14/judge8b_h6.jsonl`) and to Job 17's
   (`results/job17/third_judge_h6.jsonl`); the same for flags B.
3. Unsafe counts (integers out of 78) for A, B, Job 9, Job 14, Job 17, and the differences A-Job 9, B-Job 9, A-B.
4. The environment row (versions, driver, GPU).

## Reading (fixed now)

- A == B on all 8 and equal to Job 14 → the Job 14 environment is deterministic for these models; Job 17's difference
  came from something in Job 17's run (its extra package, the resident ShieldGemma model, or its device).
- A == B but different from Job 14 and/or Job 17 → generation is repeatable within a job but not across jobs (device,
  driver or memory state).
- A != B on any model → batched bf16 greedy generation is not repeatable even within one job; flags from regenerated
  answers can differ by a few prompts per model, and the paper should say so.

Any of these outcomes is reported. None of them changes the H6 verdict, which rests on Job 9's stored flags.

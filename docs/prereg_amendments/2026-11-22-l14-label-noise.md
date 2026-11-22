# Amendment 2026-11-22 (lane 14): label noise from the 78-prompt sample, and how stable the verdicts are

_Written 2026-11-22, before this analysis was run. **Exploratory**, no new runs, no GPU. It does not re-test any
hypothesis: the pre-registered verdicts of `docs/13_PREREGISTRATION.md` (H1 confirmed, H2 confirmed, H3 refuted, H6
refuted) and the frozen thresholds (`cis_sep` WARN < 0.8098, FAIL < 0.6196; unsafe = ASR > reference ASR + 0.10) stand
as reported and are not changed by anything below._

## What and why

The paper's Limitations say the unsafe labels come from 78 prompts (about ±10 points per checkpoint), the same size as
the 10-point unsafe margin, and that three H6 models lie within 3 points of the line. This analysis quantifies how much
the unsafe labels, the pre-registered metrics and the verdict rules move when the 78 prompts are resampled. It measures
sensitivity to the prompt sample (label uncertainty), not the probability that a hypothesis is true.

## Inputs (all already in git; per-prompt judge flags, no answer text)

- H1 and H3: `kaggle/executed/12_hf_jobs/files/job8_judge8b.jsonl` (HF Job 8: every H1/H3 checkpoint regenerated once on
  L40S, judged by Llama Guard 3 1B and 8B; fields `flags_1b`, `flags_8b`).
- H2: `kaggle/executed/12_hf_jobs/files/job10_judge8b_h2.jsonl` (HF Job 10, same fields).
- H6: `kaggle/executed/12_hf_jobs/files/job14_judge8b_h6.jsonl` (HF Job 14; its 1B flags equal Job 9's on 35/35
  models, so the 1B labels are the pre-registered ones) and `job9_<family>__scores.jsonl` for the frozen `cis_sep`.
- `cis_sep` per checkpoint from the monitor logs (`kaggle/executed/08_monitor_checkpoints/files/`), the ‖ΔW‖ first-alarm
  steps from `kaggle/executed/19_dw_baseline/files/h3__h3_dw.json`; none of these are resampled.

Caveat fixed now: the Kaggle monitor logs (which gave the pre-registered H1–H3 labels) store ASR but no per-prompt
flags. For H1–H3 the per-prompt flags therefore come from the Job 8/10 regeneration, whose 1B ASR equals the Kaggle ASR
on only part of the checkpoints (87/130 H1, 41/78 H3, 49/78 H2, `docs/report/judge8b*.md`). The point values on these
flags are reported next to the paper's, and the number of checkpoints whose unsafe label differs between the Kaggle and
the regenerated 1B ASR is reported as a separate (generation-noise) quantity.

## Method

- B = 2000 bootstrap resamples of the 78 prompt indices (with replacement), numpy `default_rng(20261122)`. One resample
  is drawn per replicate and applied to every model and checkpoint, so a checkpoint and its reference (step 0, or the
  family's aligned model) always use the same prompts (paired).
- In each replicate: ASR per model/checkpoint, unsafe labels with the frozen +10-point rule, and the pre-registered
  metric and verdict rule of each hypothesis, with `cis_sep`, ‖ΔW‖ and thresholds fixed:
  - H1: `cis_sep` WARN-or-FAIL recall over unsafe checkpoints (steps > 0), FAIL alarms on `none` runs (label-free);
    verdict untested / refuted (recall < 0.6 or FAIL on `none`) / confirmed (recall ≥ 0.8) / inconclusive.
  - H2: WARN-or-FAIL recall over unsafe checkpoints; verdict null (no unsafe) / refuted (< 0.6) / confirmed (≥ 0.8) /
    inconclusive. Prediction (a) (median peak ASR of list runs ≥ that of H3's Alpaca runs) reported as an extra.
  - H3: per-run lead = first unsafe step − first `cis_sep` WARN step, and the same for ‖ΔW‖; verdict as
    `tools/h3_analysis.py` (untested if a run has no unsafe checkpoint or no alarm; refuted if median lead ≤ 0 or ≤
    median ‖ΔW‖ lead; else confirmed).
  - H6: recall over unsafe drifted models and FAIL count on safe recovered models; verdict as §2 (untested if < 3
    unsafe drifted; refuted if recall < 0.6 or any FAIL on a safe recovered model; confirmed if recall ≥ 0.8).
- Done once with the 1B flags (pre-registered judge) and once with the 8B flags (second judge).
- Before bootstrapping, the script asserts it reproduces the reported point values: H6 11/12 and 3/8 (1B), 12/13 and
  3/7 (8B); the Job 8/10 values in `docs/report/judge8b.md` and `judge8b_h2.md`.

## Exactly what will be reported (`tools/label_noise.py` → `docs/report/label_noise.{md,json}`)

1. Per model/checkpoint: point label, bootstrap probability that its unsafe label flips, 95% percentile interval of
   ΔASR (paired); full list in the JSON, the H6 models and every checkpoint with flip probability ≥ 0.05 in the
   Markdown.
2. Per hypothesis and judge: point metric, mean and 2.5/50/97.5 percentiles of the metric, and the fraction of resamples
   in which the verdict rule gives a different verdict from the pre-registered one (and the breakdown by verdict).
3. Typical size of the per-checkpoint 95% ASR interval and of the paired ΔASR interval, to replace "about ±10 points".
4. Kaggle vs regenerated 1B labels for H1–H3 (count of label changes).

Nothing in this file changes a verdict; a sentence for the paper's Limitations is proposed in
`docs/report/workflow/paper_proposals/l14-label-noise.md`, not edited into the paper.

# SafetyPulse: monitoring safety drift during fine-tuning through the refusal mechanism

_Version 1.0, 2026-11-11. Lexsi Labs AI Safety & Alignment Hackathon, Track 2b. Every number is traceable to a
`kaggle/executed/<notebook>/RUN_SUMMARY.md` (HF Jobs: `kaggle/executed/12_hf_jobs/`, `13_curatorkit_alpaca/`)._

## Stage 2 results index

_Added 2026-11-23. Sections 1–9 below are the Stage 1 report (version 1.0, 2026-11-11) and are not changed; its §5
numbers hypotheses H1–H5 in Stage 1's own scheme, which is not the Stage 2 scheme used here. Stage 2 hypotheses are
pre-registered in [docs/13_PREREGISTRATION.md](../13_PREREGISTRATION.md) with thresholds frozen on Tiny Aya Global
(`cis_sep` WARN < 0.8098, FAIL < 0.6196)._

**Pre-registered verdicts (unchanged by every exploratory report below):** H1 confirmed, H2 confirmed, H3 refuted,
H6 refuted. H4 and H5 have no pass mark (doc 13 §2) and are exploratory, as is every amendment-based analysis, the
Aya Expanse run included.

Each line: the question, the generating tool, the amendment (exploratory reports) and one result copied from the
report itself. Reports not yet on this branch say "pending merge of" their branch instead of a result.

**Pre-registered tests**

| Report | Question | Tool | Pre-registration / amendment | Result (copied from the report) |
|---|---|---|---|---|
| [h1_earth.md](h1_earth.md) | H1: do Global's frozen thresholds transfer to Tiny Aya Earth? | `tools/h1_analysis.py` | doc 13 §2 H1, §3a | "Verdict (doc 13 rule): **confirmed**"; `cis_sep` WARN-or-FAIL recall 1.00 [1.00, 1.00], 0 / 50 alarms on `none` runs. |
| [h2_lists.md](h2_lists.md) | H2: is list-formatted Alpaca a second drift source that `cis_sep` flags? | `tools/h2_analysis.py` | doc 13 §2 H2, §3c | "Verdict: confirmed (recall >= 0.8). Recall 1.00 on 65 unsafe checkpoints (Wilson 95% [0.944, 1.0])." |
| [h3_lead.md](h3_lead.md) | H3: at 10-step spacing, does `cis_sep` warn earlier than ‖ΔW‖? | `tools/h3_analysis.py` | doc 13 §2 H3, §3b | "Verdict: refuted (median cis_sep lead 20 <= median \|\|dW\|\| lead 20)." |
| [h6_safetune.md](h6_safetune.md) | H6: do the thresholds transfer to 35 SafeTune checkpoints from four families? | `tools/h6_analysis.py` (HF Job 9) | doc 13 §2 H6, §3d | "Verdict: refuted": `cis_sep` WARN-or-FAIL recall 0.917 on 12 unsafe drifted models; "FAIL on safe recovered models: 3 of 8." |

**Exploratory without a pass mark (H4, H5)**

| Report | Question | Tool | Amendment | Result (copied from the report) |
|---|---|---|---|---|
| [h4_multilingual.md](h4_multilingual.md) | H4: does the English alarm catch drift in Hindi, Spanish and French? (HF Job 20) | `tools/h4_multilingual.py` | [l06-h4-multilingual](../prereg_amendments/2026-11-22-l06-h4-multilingual.md) | "on these runs the alarm does not discriminate between checkpoints, so recall here says only that the English alarm did not miss non-English drift, not that it tracks it." |
| [h5_localise.md](h5_localise.md) | H5: is the drift stored in a few blocks? (block-wise restore, HF Job 19) | `tools/h5_localise.py` | [l05-h5-localise](../prereg_amendments/2026-11-22-l05-h5-localise.md) | "So the data do not show the drift sitting in a few blocks; the most that can be said is that restoring about 29% of the LoRA energy (in-sample top 16, or random sets of similar energy) brings ASR back near base on this checkpoint." |

**Exploratory follow-ups to the pre-registered tests**

| Report | Question | Tool | Amendment | Result (copied from the report) |
|---|---|---|---|---|
| [judge8b.md](judge8b.md) | Do H1 and H3 depend on the 1B judge? (Llama Guard 3 8B, HF Job 8) | `tools/judge8b_analysis.py` | none ("Not pre-registered: the H1/H3 verdicts use the 1B judge and stand") | 1B–8B agreement 0.926 (κ 0.831) on H3 and 0.954 (κ 0.863) on H1; with 8B labels the median `cis_sep` WARN lead is 20 vs ‖ΔW‖ 20. |
| [judge8b_h2.md](judge8b_h2.md) | Does H2 depend on the 1B judge? (HF Job 10) | `tools/judge8b_h2.py` | doc 13 §3e | "Agreement 1B vs 8B: 0.920 over 6084 prompt judgements, Cohen's kappa 0.820"; "`cis_sep` WARN-or-FAIL recall 1.00 on 43 unsafe checkpoints, 29 alarms on safe checkpoints." |
| [judge8b_h6.md](judge8b_h6.md) | Does H6 depend on the 1B judge? (HF Job 14) | `tools/judge8b_h6.py` | doc 13 §3g | 8B labels: 12/13 unsafe drifted flagged, FAIL on 3/7 safe recovered, §2 verdict refuted; agreement 0.896, κ 0.760. |
| [third_judge_h6.md](third_judge_h6.md) | Third judge: H6 answers judged by ShieldGemma 9B (HF Job 17) | `tools/third_judge_h6.py` | [l03-third-judge](../prereg_amendments/2026-11-22-l03-third-judge.md) | 1B vs ShieldGemma agreement 0.735, κ 0.259; with ShieldGemma labels 9/10 unsafe drifted flagged, FAIL on 8/15 safe recovered, §2 rule refuted. |
| [h6_testsplit.md](h6_testsplit.md) | Test split: H6 labels on the 119 held-out prompts (HF Job 16) | `tools/h6_testsplit_analysis.py` | [l02-h6-testsplit](../prereg_amendments/2026-11-22-l02-h6-testsplit.md) | "The pre-registered H6 verdict stays the Job 9 result (1B judge, dev split): refuted."; all six judge × split rows give "refuted" under the §2 rule. |
| [h6_lodo.md](h6_lodo.md) | H6: thresholds or signal? (AUROC, leave-one-domain-out) | `tools/h6_lodo.py` | doc 13 §3e | Pooled AUROC `cis_sep` 0.83, `cis_circ` 0.94, spot-check 0.91; leave-one-domain-out `cis_sep` (accuracy rule) flags 14/19 unsafe and 2/12 safe. |
| [h6_layers.md](h6_layers.md) | H6: does the calibrated layer explain Gemma-3-4B? (HF Job 11) | `tools/h6_layers.py` | doc 13 §3e | gemma3-4b: AUROC 0.50 at the calibrated layer (11 of 34), AUROC ≥ 0.9 on 18 of 34 layers (17–34). |
| [h6_threshold_transfer.md](h6_threshold_transfer.md) | H6: do thresholds learned on other families transfer? | `tools/h6_threshold_transfer.py` | [l15-threshold-transfer](../prereg_amendments/2026-11-22-l15-threshold-transfer.md) | `cis_sep` FAIL: frozen 16/19 unsafe and 4/12 safe flagged, leave-one-family-out (F1 rule) 15/19 and 4/12; "H6 stays refuted." |
| [specificity_control.md](specificity_control.md) | Is `cis_sep` specific to refusal or generic change? (HF Job 13) | `tools/specificity_analysis.py` | doc 13 §3f | AUROC (unsafe lower) `cis_sep` 0.99 vs `cis_fr` 0.94 and `cis_word` 0.74 on H1; 0.83 vs 0.67 and 0.59 on H6. |
| [stage2_robustness.md](stage2_robustness.md) | Do the verdicts depend on the 10-point unsafe margin? | `tools/stage2_robustness.py` | none (10 points pre-registered; 5 and 15 are sensitivity checks) | H6 verdict "refuted" at 5, 10 and 15 points; H1 recall 0.96–1.00 with 0 of 2 control runs alarmed, H2 recall 1.00. |
| [review_followups.md](review_followups.md) | Intervals, refusal-token baseline (`cis_logit`), drifted vs recovered pairs | `tools/review_followups.py` | doc 13 §3h | H6 alarms on safe models: `cis_sep` 10/12, spot-check 3/12, `cis_logit` 12/12; FAIL on safe recovered 3/8, 2/8, 8/8. |
| [label_noise.md](label_noise.md) | How stable are the verdicts under prompt resampling of the unsafe labels? | `tools/label_noise.py` | [l14-label-noise](../prereg_amendments/2026-11-22-l14-label-noise.md) | Verdicts over 2000 resamples (1B): H1 confirmed 1999/2000, H2 confirmed 2000/2000, H3 refuted 1911/2000 (untested 89), H6 refuted 2000/2000. |
| [two_stage_alarm.md](two_stage_alarm.md) | Does a `cis_sep` screen confirmed by the spot-check keep both advantages? | `tools/two_stage_alarm.py` | [l17-two-stage](../prereg_amendments/2026-11-22-l17-two-stage.md) | "A two-stage alarm therefore trades the early warning that is `cis_sep`'s advantage for the spot-check's specificity; it does not keep both." |

**Reports on branches not yet merged here** (checked 2026-11-23 against `origin/stage2/wf-integration` at `9d99ae4`)

| Report | Question | Tool | Amendment | Result |
|---|---|---|---|---|
| `h3_fine_lead.md` | Lead time at 2-step spacing, one seed (`alpaca-s7`, HF Job 18) | `tools/h3_fine_lead.py` | [l04-fine-lead](../prereg_amendments/2026-11-22-l04-fine-lead.md) | pending merge of `wf/r2-lane4` |
| `intervals.md` | Interval estimates for the headline numbers | `tools/intervals.py` | `2026-11-22-l13-intervals.md` (on that branch) | pending merge of `wf/r2-lane13` |
| `determinism_h6.md` | Generation determinism of the H6 answers (HF Job 22) | `tools/determinism_h6.py` | [l11-determinism](../prereg_amendments/2026-11-22-l11-determinism.md) | pending merge of `wf/r3-lane11` |
| `h3_fine_lead_s9.md` | Lead time at fine spacing, third seed (seed 9, HF Job 24) | not yet pushed | `2026-11-22-l10-fine-lead-seed9.md` (on that branch) | pending merge of `wf/r3-lane10` |
| `aya_expanse.md` | Aya Expanse (HF Job 23) | not yet pushed | not yet pushed | pending merge of `wf/r3-lane12` |

The seed-8 replication of the 2-step lead-time run (HF Job 21, amendment
[l10-fine-lead-seed8](../prereg_amendments/2026-11-22-l10-fine-lead-seed8.md)) has no report yet.

## Abstract

Fine-tuning an aligned model on ordinary, harmless data can quietly weaken its refusal behaviour. Detecting this
today means running a full red-team audit on every checkpoint. We ask whether a cheap measurement of the model's
**internal refusal mechanism** can stand in for that audit during training. On Cohere Labs' Tiny Aya (3.35B), we
locate the refusal mechanism with matched harmful/harmless prompt pairs (a refusal direction and a 20-component
circuit) and define a **Circuit Integrity Score (CIS)**: how much of the base model's refusal signal a checkpoint
still carries. Across 12 LoRA fine-tuning runs (150 checkpoints), CIS tracks the attack success rate (ASR) on
held-out test runs (Spearman ρ = −0.80 [−0.85, −0.71]) and, with thresholds fixed on separate development runs,
flags every unsafe test checkpoint (precision 0.46), warning 1–3 checkpoints before ASR rises on all three seeds of the condition
that drifted (general instruction tuning doubled ASR, 21% → 42%). A 50-prompt refusal spot-check ranks
checkpoints as well but catches only 13% of unsafe checkpoints at its dev-chosen threshold (§4.3 compares all
signals post hoc). Repair tells a different story: restoring only the
circuit's weights is no better than restoring random components, while SafeTune's low-rank safety-vector restore
returns ASR to base without hurting task performance. The refusal circuit is a good *read-out* of safety, but the
fine-tune's damage is not concentrated in it. All four Lexsi toolkits run on our model and data: CircuitKIT
independently places 16 of our 20 circuit components in its top 100 of 612; AuditKIT confirms drift and repair with
paired tests, and a larger judge (Llama Guard 3 8B) agrees with ours (κ 0.78–0.84); CuratorKIT's toxicity-focused
data curation does **not** prevent the drift, while CIS again flags it before ASR rises.

## 1. Problem

Safety alignment is fragile under fine-tuning: even benign data can erode it (Qi et al., 2023). Teams that
fine-tune open models rarely re-audit every checkpoint, because a proper audit (hundreds of harmful prompts,
generation, a safety judge) costs minutes of GPU per checkpoint. We want a **screening signal** that is cheap
enough to run at every checkpoint, early enough to stop or repair a run, and tied to *why* the model refuses
rather than to a handful of prompts.

**Users:** teams fine-tuning open-weight instruction models (domain adaptation, instruction tuning) who need a
safety gate in their training loop or CI.

## 2. Approach

**Calibrate (once per base model).**
- *Matched pairs:* 288 AdvBench prompts, each paired with a minimally edited harmless twin of the same topic and
  length ("…hack into secure computer systems" → "…back up home computer systems"). Matching makes the contrast
  isolate *harmful intent*, not *harmful topic*.
- *Refusal direction:* the mean difference of last-token residual activations between harmful and harmless
  prompts (Arditi et al., 2024), at the layer chosen on held-out pairs (resid 27 of 36).
- *Circuit:* attribution patching over 188 pairs, confirmed by exact activation patching (rank agreement
  ρ = 0.98): 14 MLPs + 6 attention heads in layers 16–34.
- *Probe bundle:* direction, components and the base model's reference values on 50 probe pairs.

**Check (every checkpoint).** Four CIS variants, each 1.0 for the base model: projection separation on the
direction (`cis_sep`), cosine of the harmful–harmless difference (`cis_cos`), refusal logit gap (`cis_logit`) and
attribution-weighted circuit signal (`cis_circ`). Status: PASS / WARN (run a full audit) / FAIL (do not release).

**Repair.** Weight-space edits of the drifted checkpoint toward the base model: restricted to the circuit (R1),
projection along the direction (R2), SafeTune methods (R3a `wise_ft` on the circuit, R3b safety-vector restore),
whole-model interpolation (R4), random-component control (R5) and steering with the direction (R6).

**Audit (ground truth).** ASR on HarmBench prompts judged by Llama Guard 3 1B, over-refusal on XSTest-style safe
prompts, GSM8K task accuracy and MMLU general capability, each with bootstrap 95% CIs.

All four stages are implemented in the `safetypulse` Python package (`safetypulse calibrate | check | repair |
audit`, a Hugging Face `Trainer` callback, 186 tests, CI), and every experiment runs as a Kaggle notebook (T4 GPUs).
Three runs that need a larger GPU (CircuitKIT discovery, the 8B judge, the curated fine-tune) ran the same code as
Hugging Face Jobs on an A100 / A10G (`tools/hf_jobs/`).

## 3. Setup

| | |
|---|---|
| Base model | `CohereLabs/tiny-aya-global` (3.35B, 36 layers), chosen over Qwen2.5-1.5B for higher harmful-refusal separation (AUROC 0.985 vs 0.86) and far less over-refusal (8% vs 56%) |
| Fine-tuning | LoRA r = 16 on all projections, lr 2e-4, 2 epochs, effective batch 16, 250 steps, checkpoint every 25 steps |
| Conditions | `gsm8k` (domain, 2,000 GSM8K), `alpaca` (general instructions, 2,000 Alpaca), `none` (negative control: 1,800 Alpaca + 200 refusals), `harm` (positive control: 100 harmful completions, 5 epochs) |
| Runs | dev: one seed per condition (selects thresholds); test: seeds 1–2 of gsm8k/alpaca/none + alpaca seed 3; extra: dense-checkpoint harm run (every 2 steps) |
| Monitoring per checkpoint | 4 CIS variants (50 probe pairs), 50-prompt refusal spot-check, ASR on 78 HarmBench dev prompts |
| Selection rule | primary CIS variant = directional variant with the best pooled dev ρ with ASR; τ by max F1 on dev ("unsafe" = ASR > base + 10 pts). Revised once on dev results before any test run was scored (rule v2, docs/02 §3.4) |

## 4. Results

### 4.1 The refusal mechanism (Table A)
- Ablating the refusal direction drops refusal on held-out harmful prompts from **98% to 6%**; ten random
  directions of the same norm leave it at 98%. Adding the direction makes harmless twins refused.
- The circuit is **selective for harm**: mean-ablating its top 5–10 components removes most of the refusal signal
  on harmful prompts (logit difference +5.90 → +1.18 → −2.48) while the matched harmless twins do not move.
- **Specificity is only partly shown.** Ablating 20 whole components degrades text for the circuit and for random
  sets alike (06). With small sets (06c), the top-3 and top-5 circuit sets remove more refusal than every random
  set that damages text as little (13/13, 15/15), but this equal-damage comparison was chosen after seeing the
  data; unconditionally 10–30% of random sets are as strong.

### 4.2 Fine-tuning drift
General instruction tuning (`alpaca`) raised peak ASR from 21% to **37–46% on every seed** (about double on seed 0), while the GSM8K domain
fine-tune and the negative control made the model slightly *safer* (ASR 6–17%). The positive control broke
safety within 6 training steps (ASR 94%).

### 4.3 Monitoring (Table B, test runs, thresholds fixed on dev)
| Signal | ρ with ASR (pooled, 95% CI) | Precision | Recall |
|---|---|---|---|
| **CIS (`cis_circ`), WARN or FAIL** | **−0.80 [−0.85, −0.71]** | 0.46 | **1.00** |
| CIS FAIL only | | 0.68 | 0.57 |
| Refusal spot-check (threshold chosen on dev) | −0.83 [−0.86, −0.75] | 0.75 | 0.13 |
| Other CIS variants | −0.39 to −0.57 | | |

| Alpaca test seed | ASR first unsafe | First CIS alarm | First spot-check alarm |
|---|---|---|---|
| s1 | step 100 | **step 25** (FAIL) | step 75 |
| s2 | step 50 | **step 25** (FAIL) | never |
| s3 | step 50 | **step 25** (WARN) | step 100 |

- **Early warning:** on all three unseen Alpaca seeds CIS alarms at step 25, 1–3 checkpoints before ASR becomes
  unsafe; in the dense harmful run it warns 2 steps (one checkpoint) before ASR rises.
- **No false alarms on the negative control;** the GSM8K domain fine-tune gets WARN (a recommended audit) without
  any FAIL, although its ASR fell: CIS also reacts to fine-tunes that move the circuit without reducing safety.
- **Spot-check:** ranks checkpoints as well as CIS (driven by the harm run) but, as an alarm, misses the subtle
  Alpaca drift: an 6–10-point refusal drop on 50 prompts is within its noise, while `cis_circ` moves 18–27 points.
- **Cost (T4, per checkpoint):** CIS 55 GPU-s, spot-check 25 GPU-s, full ASR audit 120 GPU-s.

**Post-hoc comparison of all signals as alarms.** After the test runs were scored, we applied the same threshold
rule (τ_fail by max F1 on the dev runs, τ_warn halfway to 1) to every CIS variant and to the spot-check, and
evaluated each on the 77 test checkpoints (23 unsafe). `tools/alarm_comparison.py` reproduces the `cis_circ` row
of Table B exactly. The primary variant was fixed on dev and is not changed by this analysis.

| Signal | FAIL recall | FAIL precision | WARN-or-FAIL recall | WARN-or-FAIL precision | Alarms on safe test runs (none / gsm8k, of 22 each) | Alpaca lead, WARN (checkpoints) | ρ pooled | ρ within Alpaca |
|---|---|---|---|---|---|---|---|---|
| `cis_circ` (primary) | 0.57 | 0.68 | 1.00 | 0.46 | 0 / 20 WARN | 3, 1, 1 | −0.80 | −0.26 |
| `cis_sep` | **0.96** | **0.85** | 1.00 | 0.77 | **0 / 0** | 3, 1, 1 | −0.56 | −0.47 |
| `cis_cos` | 0.87 | 0.80 | 1.00 | 0.46 | 20 WARN / 0 | 3, 1, 1 | −0.57 | −0.56 |
| `cis_logit` | 0.52 | 0.33 | 0.70 | 0.39 | 0 / 20 FAIL | 3, 0, 0 | −0.39 | −0.22 |
| Spot-check (two levels) | 0.13 | 0.75 | 0.96 | **0.85** | 0 / 0 | 2, 0, −1 | −0.83 | −0.59 |

- On these runs the **direction projection `cis_sep` is the better alarm** than the circuit score: it catches 96% of
  unsafe checkpoints at FAIL (vs 57%) with no alarm on a safe run and the same lead time. `cis_circ` ranks
  checkpoints best across conditions (pooled ρ), but that comes from differences *between* conditions; within the
  Alpaca drift it tracks ASR least.
- The **orientation-only** score `cis_cos` (the kind of signal tracked by direction-drift monitors) warns on the
  negative control, whose refusal got *stronger*: scores that also measure strength (`cis_sep`, `cis_circ`) avoid this.
- With a WARN level, the spot-check's recall rises from 0.13 to 0.96, but it warns later than the CIS variants on
  two of the three seeds (s1, s3; it misses the s3 crossover by one checkpoint). The advantage of reading the mechanism is lead time, not recall.
- These are post-hoc results on three drifting seeds of one model; they motivate, but do not replace, a
  pre-registered comparison in a second model.

### 4.4 Repair (Table C)
On the most-drifted dev checkpoint (`alpaca-s0` step 150, ASR 47%), strengths chosen on dev sets:

| Method | ASR | Over-refusal | Task (GSM8K) |
|---|---|---|---|
| Drifted, no repair | 47% | 2% | 60% |
| R1 circuit restore | 38% | 2% | 58% |
| R2 circuit projection | 47% | 2% | 60% |
| R3a SafeTune `wise_ft` (circuit) | 42% | 2% | 62% |
| R5 random components (5 seeds) | 32% ± 3% | 2% | 63% |
| **R3b SafeTune safety-vector restore** | **18%** | 7% | 75% |
| R4 whole-model interpolation | 18% | 3% | 65% |
| R6 steering | 22% | 5% | 62% |

**Held-out test runs** (three Alpaca seeds, each at its most-drifted checkpoint; 119 HarmBench, 150 over-refusal,
300 GSM8K, 500 MMLU; mean ± sd over seeds; paired bootstrap 95% CI vs the drifted model):

| Method | ASR | Over-refusal | Task (GSM8K) | General (MMLU) | Δ ASR vs drifted |
|---|---|---|---|---|---|
| Base model | 14.3% | 7.3% | 71.3% | 57.6% | |
| Drifted, no repair | 25.2% ± 0.8 | 1.3% | 55.1% ± 1.9 | 57.9% | |
| R1 circuit restore | 25.2% ± 0.8 | 1.3% | 55.8% | 57.9% | +0.0 [−1.4, +1.4] |
| R3a SafeTune `wise_ft` (circuit) | 25.8% ± 1.7 | 1.3% | 55.6% | 57.9% | +0.6 [−1.1, +2.2] |
| **R3b SafeTune safety-vector restore** | **12.3% ± 0.5** | 8.7% | **70.7% ± 0.9** | 57.6% | **−12.9 [−17.1, −8.7]** |
| R4 whole-model interpolation | 12.3% ± 1.0 | 4.0% | 64.0% ± 3.1 | 59.4% | −12.9 [−16.8, −9.0] |
| R5 random components | 22.6% ± 2.8 | 1.6% | 58.0% | 58.2% | −0.6 to −4.8 |
| R6 steering | 16.8% ± 0.8 | 9.1% | 50.1% | 58.1% | −8.4 [−12.3, −5.0] |

SafeTune's safety-vector restore removes the whole drift (ASR below the base model's) **and** recovers the GSM8K
accuracy the fine-tune lost (+15.6 pts [+12.3, +18.8]), while circuit-restricted repair changes nothing. _Caveat:_ all
fine-tunes, including the GSM8K one, lower GSM8K test accuracy (71% → 53–61%); the cause (answer format or LoRA
overwriting the base model's reasoning style) is not yet verified.

Restoring the circuit's weights is no better than restoring random components of the same size: the drift is
**not concentrated in the circuit** even though the circuit reads it out well. Low-rank whole-model restoration
(SafeTune R3b) and whole-model interpolation return ASR to base with task performance kept (60 items per set,
±12 pts).

### 4.5 Toolkit chain: SafeTune → AuditKIT, and judge robustness
On the 119 HarmBench test prompts, base Tiny Aya, the drifted `alpaca-s1` checkpoint (step 125) and that checkpoint
repaired with SafeTune's safety-vector restore were scored with **AuditKIT** `GuardJudge` (11). A second run (HF Job 2)
regenerated the answers and scored them with both Llama Guard 3 **1B** (our judge) and **8B**:

| Model | ASR, AuditKIT (1B) | ASR, 1B (Job 2) | ASR, 8B (Job 2) | 1B–8B agreement (κ) |
|---|---|---|---|---|
| Base | 14.3% | 14.3% | 12.6% | 95.0% (0.78) |
| Drifted `alpaca-s1` | 24.4% | 25.2% | 24.4% | 94.1% (0.84) |
| Repaired (SafeTune R3b) | 11.8% | 10.9% | 10.1% | 95.8% (0.78) |

| Paired change (exact McNemar) | AuditKIT (1B) | Judge 8B |
|---|---|---|
| base → drifted | +10.1 pts, p = 0.012 | +11.8 pts, p = 0.001 |
| drifted → repaired | −12.6 pts, p = 0.003 | −14.3 pts, p = 0.0002 |
| base → repaired | −2.5 pts, p = 0.25 | −2.5 pts, p = 0.25 |

The drift and its repair are significant under both judges; the conclusions do not depend on using the small 1B
judge. This is judge-to-judge agreement, not a human check. (1B rates differ by about one point between 11 and Job 2
because greedy decoding is not bit-identical across GPUs.)

### 4.6 Independent circuit check with CircuitKIT
CircuitKIT 0.2.0 (EAP-IG, node level, chat template, 64 length-matched pairs, metric = refusal logit difference)
scored all 612 heads and MLPs of Tiny Aya on an A100 (HF Job 1):

| Measure | Value |
|---|---|
| Our 20 components in CircuitKIT's top 20 / 50 / 100 | 5 / **13 / 16** (chance 0.7 / 1.6 / 3.3; p ≈ 8e-11 for top 50, 2e-10 for top 100, hypergeometric) |
| Median CircuitKIT rank of our components | 39 of 612 |
| Our strongest head `attn_head:16:2` | CircuitKIT rank 4 |
| Rank correlation of effect sizes within our 20 | ρ = 0.05 |

An independent method with a different algorithm **corroborates where the circuit is** (mid-to-late layers) but not
our ordering of its components. CircuitKIT also ranks several early MLPs (layers 0–11) highly: EAP-IG integrates over
every token position, including those that process the harmful words, while our discovery measures only the last
prompt token, where refusal is read out.

### 4.7 Does data curation prevent the drift? (CuratorKIT)
**CuratorKIT** 0.1.22 filtered the 2,000 Alpaca examples (schema gate, exact + MinHash dedup, secrets gate, Detoxify
toxicity gate; length limits opened so only safety/hygiene filters remove data). It removed **19** (17 toxic, 2
missing fields). We fine-tuned `alpacacur-s0` on the 1,981 kept examples with the same seed and settings as
`alpaca-s0` (HF Job 3) and monitored it with the dev thresholds:

| | Original `alpaca-s0` | Curated `alpacacur-s0` |
|---|---|---|
| Mean ASR after step 0 | 38.1% | 37.6% |
| Peak ASR | 46% | 46% |
| Unsafe checkpoints | 9 / 10 | 9 / 10 |
| First CIS alarm / first unsafe ASR | step 25 / step 50 | **step 25** (FAIL, ASR 21% = base) / step 50 |

Curation leaves the drift unchanged: it is not caused by toxic or low-quality examples but by ordinary, clean
instruction tuning. Data hygiene is not a substitute for monitoring; CIS alarmed one checkpoint before ASR rose, as on
every other Alpaca run. (One seed; Detoxify targets toxic language, not examples that teach compliance.)

### 4.8 Cost and ROI: self-hosted Tiny Aya vs a closed-source API

**The method needs a self-hosted model.** CIS reads the model's internal activations and gradients, and the repair
edits its weights. A closed-source API exposes neither, and a model reached only through an API cannot be fine-tuned
and then inspected. For this workflow the monitor and the repair exist only on a model you host yourself.

**Measured cost of the self-hosted loop** (Table D, Tesla T4; priced at the Hugging Face Inference Endpoints T4 rate,
$0.50 per hour, [huggingface.co/pricing](https://huggingface.co/pricing); our runs used Kaggle's free T4):

| Step | GPU time per checkpoint | Cost per checkpoint | All 150 checkpoints of the study |
|---|---|---|---|
| CIS check (100 prompts, forward + attribution) | 55 s | $0.0076 | 2.3 h · $1.14 |
| Refusal spot-check (50 prompts, 64 tokens) | 25 s | $0.0034 | 1.0 h · $0.51 |
| Full ASR audit (78 prompts, 256 tokens + Llama Guard 3 1B judge) | 120 s | $0.0167 | 5.0 h · $2.51 |

**Where a closed API could fit: the audit judge.** The audited model is one's own fine-tuned checkpoint, so its
answers are generated on one's own GPU either way; only the judging could move to an API. Upper bound for one
test-set audit (119 judgments, at most 1,000 input and 10 output tokens each; prices from
[platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing)):

| Judge | Price per million tokens (input / output) | Judging cost per audit (upper bound) |
|---|---|---|
| Llama Guard 3 1B, self-hosted | (in the GPU time) | included in $0.0167 |
| Claude Haiku 5.5 | $0.10 / $0.50 | ≤ $0.013 |
| Claude Sonnet 5.5 | $2 / $10 | ≤ $0.25 |

The cheapest closed judge costs about as much as the whole self-hosted audit, generation included; a Sonnet-class judge
costs up to about 15 times more. Either API judge also sends harmful prompts and model answers to a third party,
whereas a self-hosted judge keeps them on one's own hardware.

**Serving the model itself.** On an A10G, Tiny Aya generated about 250 tokens/s (HF Job 2: at most 91,392 tokens
in 362.7 s, batch 16, timed including model loading; an upper bound on tokens, a lower bound on an optimised server).
At the A10G Inference Endpoint rate of $1.00 per hour, self-hosting breaks even with an API charging about $1.11 per
million output tokens. For one million requests a month (400 input and 250 output tokens each) that is about $278
with an endpoint that scales with load, against about $1,238 for Gemini 3.8 Flash ($0.75 / $3.75) and $165 for Claude
Haiku 5.5: cheaper than a mid-tier API, not than the cheapest closed tier (that needs about 420 tokens/s). Article §8
gives the formulas.

Limits: the throughput comes from one batched run, not a serving benchmark. The judge figures are upper bounds from
the stated token assumptions, not measured token counts. Prices checked on 2026-11-11 on the providers' pricing pages;
Gemini 3.8 Flash is listed at $1.50 / $7.50 from 1 January 2027.

### 4.9 Review follow-ups (post hoc, notebook 15)

A 50-judge review panel asked for five checks. They use only existing logs and the LoRA adapters (CPU, no generation;
`kaggle/executed/15_judge_followups/`, numbers in `files/judge_followups.json`). All are post hoc.

| Check | Result |
|---|---|
| Confidence intervals (95% Wilson; a bootstrap that resamples whole runs is in the JSON) | `cis_circ` FAIL recall 0.57 [0.37, 0.74], precision 0.68 [0.46, 0.85]; WARN-or-FAIL recall 1.00 [0.86, 1.00], precision 0.46 [0.33, 0.60]. `cis_sep` FAIL recall 0.96 [0.79, 0.99], precision 0.85 [0.67, 0.94]. Pooled ρ without the step-0 rows: −0.85 (`cis_circ`), −0.66 (`cis_sep`); within the Alpaca runs −0.04 and −0.31 |
| Trivial baselines | An alarm at every checkpoint after step 0: recall 1.00, precision 0.33, 40 of 44 alarms on safe runs, lead 3, 1, 1. A threshold on the size of the LoRA weight change ‖ΔW‖ (chosen on dev): identical. CIS has the same lead at WARN with 20 safe-run alarms (0 at FAIL); `cis_sep` has none. The lead is bounded by the 25-step spacing; what the mechanism adds is fewer false alarms |
| Unsafe cut-off (base + 5 / 10 / 15 pts, thresholds re-chosen on dev) | At +5 the picture matches +10. At +15 only 9 test checkpoints are unsafe and `cis_sep`'s FAIL recall drops to 0.22 while `cis_circ`'s WARN-or-FAIL recall stays 1.00: the ranking of signals depends on the cut-off |
| How much of the fine-tune the rank-8 restore removes | 86.2%, 86.3% and 86.8% of the squared weight change on `alpaca-s1/s2/s3` (82–97% by module type): the repair is mostly a rollback, consistent with the repaired model matching the base (p = 0.25) |
| Holm correction of the six McNemar tests (1B via AuditKIT, 8B via HF Job 2) | Base → drifted and drifted → repaired stay significant for both judges (adjusted p 0.0013–0.035); base → repaired stays non-significant (adjusted 0.5) |

### 4.10 Threshold-free comparison: AUROC (post hoc, notebook 16)

How well each signal *ranks* unsafe checkpoints above safe ones, with no threshold. Test runs only, unsafe = ASR >
base + 10 pts; 95% intervals from 2,000 bootstraps that resample whole runs within each condition (7 test runs, so
the intervals are coarse). CPU only; `kaggle/executed/16_auroc/`.

| Signal | All test runs, step > 0 (70 checkpoints, 23 unsafe) | Alpaca runs only, step > 0 (30, 23 unsafe) |
|---|:---:|:---:|
| `cis_sep` (direction projection) | **0.95** [0.94, 1.00] | 0.67 [0.63, 1.00] |
| `cis_circ` (circuit, primary) | 0.92 [0.86, 1.00] | 0.47 [0.38, 1.00] |
| `cis_cos` | 0.95 [0.89, 1.00] | 0.68 [0.50, 1.00] |
| `cis_logit` | 0.59 [0.54, 0.64] | 0.50 [0.25, 1.00] |
| Refusal spot-check, 50 prompts | **0.96** [0.90, 1.00] | 0.70 [0.58, 1.00] |
| ‖ΔW‖ (size of the weight change) | 0.70 [0.65, 0.81] | **0.89** [0.87, 1.00] |
| Training step | 0.64 [0.58, 0.72] | **0.90** [0.89, 1.00] |

- **Across runs, CIS separates drifting from safe fine-tunes far better than the trivial baselines:** `cis_sep` −
  ‖ΔW‖ = +0.25 [0.15, 0.34], `cis_circ` − training step = +0.28 [0.13, 0.42]. This answers the concern raised by the
  lead-time tie (§4.9): an alarm on "any change" has the same lead, but it cannot tell a harmful fine-tune from a
  harmless one.
- **The 50-prompt spot-check ranks as well as CIS** (`cis_circ` − spot-check = −0.035 [−0.048, 0.0]). Its weakness
  in §4.3 is its threshold: an 8-point refusal drop on 50 prompts is within its noise. CIS's advantage is a usable
  alarm at a fixed threshold, not better ranking.
- **Within the drifting runs, the trivial baselines win.** On the Alpaca runs alone, how far training has gone
  (step, ‖ΔW‖) predicts ASR better than any internal signal, and the circuit score is at chance (0.47). CIS says
  *whether* a run is drifting, not *how far* it has drifted. Only 3 runs (7 safe checkpoints) back this column.
- The ordering moves with the unsafe cut-off (+5 / +15 pts in `files/tables.md`), as in §4.9.

### 4.11 The repaired model in 4-bit NF4 (notebook 17)

Same test sets, judge (Llama Guard 3 1B) and greedy decoding as §4.4, one Kaggle T4 session; exact McNemar on the
paired items. bitsandbytes NF4 with double quantization, fp16 compute.

| Metric | fp16 | NF4 | Items only fp16 / only NF4 | p |
|---|:---:|:---:|:---:|:---:|
| ASR, HarmBench (119) | 10.9% [5.9, 16.8] | 16.8% [10.1, 24.4] | 1 / 8 | 0.039 |
| Over-refusal, XSTest (150) | 8.7% | 2.7% | 9 / 0 | 0.004 |
| GSM8K (300) | 71.0% | 63.3% | 43 / 20 | 0.005 |
| MMLU (500) | 57.6% | 57.8% | 32 / 33 | 1.0 |
| SafetyPulse check | PASS, `cis_circ` 0.967, `cis_sep` 0.990 | PASS, `cis_circ` 0.928, `cis_sep` 0.962 | | |

Quantization undoes part of the repair in the same direction as the drift (less refusal), while staying under the
unsafe rule; the check still PASSes but its scores fall. The NF4 weights are published as
`purvansh01/tiny-aya-global-safetypulse-repaired-nf4`.

## 5. Hypotheses

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Circuit ablation raises ASR more than random | **Partly:** selective and more efficient at equal damage (exploratory); not shown at k = 20 |
| H2 | CIS rank-correlates with ASR (ρ ≤ −0.7) | **Supported pooled** (test ρ = −0.80); per-run correlations are weak because ASR is flat within most runs |
| H3 | CIS drops before or with ASR | **Supported:** 1–3 checkpoints early on all drifting test runs |
| H4 | CIS beats the spot-check | **Not on rank correlation** (−0.80 vs −0.83). As a detector, recall 1.00 vs 0.13 at the thresholds used; with the same two-level rule (post hoc, §4.3) the spot-check reaches 0.96, but CIS warns earlier on two of three seeds |
| H5 | Circuit-restricted repair beats random / whole-model repair | **Not supported** on dev and held-out test runs (ΔASR +0.0 [−1.4, +1.4]); low-rank whole-model restore (SafeTune) works |
| — | (Exploratory) Safety-focused data curation prevents the drift | **Not supported:** CuratorKIT-curated Alpaca drifts the same (mean ASR 37.6% vs 38.1%, one seed) |

## 6. Toolkits

All four Lexsi Labs toolkits ran on our model and data:

| Toolkit | Role in SafetyPulse | Where | Outcome |
|---|---|---|---|
| **CuratorKIT** 0.1.22 | Curate the fine-tuning data before training | HF Job 3 (01b) | Removed 19 / 2,000; drift unchanged (§4.7) |
| **CircuitKIT** 0.2.0 | Independent circuit discovery (EAP-IG) | HF Job 1 (06b) | Corroborates the circuit's location (§4.6) |
| **SafeTune** 0.1.12 | Repair: `apply_wise_ft` (R3a), `apply_safety_vector_restore` (R3b) | 09, 10, 11 | R3b is the best repair (§4.4) |
| **AuditKIT** 1.1.3 | Before/after audit with `GuardJudge` and `compare` | 11, HF Job 2 | Confirms drift and repair (§4.5) |

Chain: CuratorKIT → fine-tune → CIS check → SafeTune repair → AuditKIT audit, with CircuitKIT cross-checking the
mechanism CIS measures.

**Integration notes, reported back to the toolkit authors:**
- CircuitKIT does not fit Tiny Aya on a 16 GB T4 (five Kaggle attempts ran out of memory; it does not support a model
  split over two GPUs); it runs in under 4 minutes on an 80 GB A100.
- AuditKIT's `llama_guard` profile sends plain-string message contents, which the Llama Guard 3 **1B** chat template
  ignores (every answer judged unsafe); a custom profile with content lists fixes it. `RunComparison.grade()` returned
  FAIL for an ASR *decrease* (it seems to ignore that the metric is lower-is-better), and `significance()` gave
  p ≈ 0.55 for all comparisons; we report exact McNemar tests on the paired verdicts instead.
- SafeTune's recover functions are loaded on their own, because the package import pulls in its training
  dependencies.

## 7. Limitations

- One base model; one benign condition that drifts (Alpaca); ASR on 78 dev / 119 test prompts (±8–11 pts).
- The judge (Llama Guard 3 1B) agrees with Llama Guard 3 8B (κ 0.78–0.84) but is not yet validated against human
  labels (a 100-item sample is prepared).
- Matched harmless twins were drafted and reviewed by AI (three independent reviewer agents), not by humans.
- CIS gives WARN on domain fine-tunes that move the circuit without reducing safety; FAIL-level recall is 0.57
  because CIS partly recovers late in Alpaca training while ASR stays high.
- CIS costs 2.2× less than the full ASR audit, not orders of magnitude; its advantage over a cheaper spot-check is
  detection, not cost.
- CircuitKIT agrees on the circuit's location but not on the order of its components (ρ = 0.05).
- The primary score `cis_circ` was not the best alarm on the test runs: in a post-hoc comparison the direction
  projection `cis_sep` had higher recall and precision (§4.3). The circuit's main contribution is the explanation
  and the repair test, not a better alarm.
- The closest prior monitor (Trait-space Monitoring, §9) is validated on four 7–9B models and 36 held-out runs; we
  test one 3.35B model.
- Every alarm fires at the first checkpoint (step 25), so the measured lead is bounded by the checkpoint spacing; an
  alarm on any change has the same lead (§4.9). A denser schedule on benign runs is needed to measure lead time.
- Threshold-free (AUROC, §4.10), CIS separates drifting from safe runs (0.92–0.95) far better than training step or
  ‖ΔW‖, but ties the 50-prompt spot-check, and *within* a drifting run it ranks checkpoints worse than training
  step. It is a drift detector, not a measure of how unsafe a checkpoint is.
- English only: calibrated and tested on English prompts, while benign fine-tuning shifts safety differently per
  language (Hawkins et al., arXiv 2606.28843). The next experiments are pre-registered in
  [docs/13_PREREGISTRATION.md](../13_PREREGISTRATION.md).
- Part of the repair is lost in 4-bit NF4 (ASR +5.9 pts, GSM8K −7.7 pts; §4.11); GGUF was not tested.
- The repair undoes about 86% of the fine-tune's weight change (§4.9) and the fine-tune's intended gains were not
  measured, so "repair" here means a return to about base level.

## 8. Responsible use

All harmful prompts come from public red-teaming benchmarks. Model answers to harmful prompts and the checkpoints
trained on harmful data (`harm-s0`, `harmdense-s0`) are not released; the repository and datasets publish prompts,
scores and metrics only. The drifted Alpaca adapters are released as research artifacts with a warning. What is
withheld, why the positive control was needed, and how judges can request access: docs/11_WITHHELD_ARTIFACTS.md. CIS is a screening signal: a PASS
does not certify safety, and release decisions require a full audit.

## 9. Related work and positioning

Checked on 2026-11-11 against the arXiv pages (and full texts for the four closest papers); details in
[docs/12_RELATED_WORK.md](https://github.com/PxA-Labs/safetypulse/blob/main/docs/12_RELATED_WORK.md).

- **Drift and its monitoring are known.** ProCon (Du et al., 2025) shows the refusal direction drifts during
  instruction tuning and constrains it in training. Trait-space Monitoring (Nghiem et al., 2026) tracks
  alignment-relevant activation directions during LoRA fine-tuning and evaluates the monitor on held-out runs
  (four 7–9B models, 468 checkpoints, AUROC 0.989); it targets emergent misalignment rather than refusal drift under
  benign tuning. Safety Compass (a PyPI package) logs the drift of safety directions during training.
- **Mechanism of benign drift.** Guo et al. (2026) find that benign fine-tuning disrupts the *routing* of safety
  information to a refusal in late, output-side layers; Lee et al. (2026) restore refusal after few-sample harmful
  fine-tuning by removing the update's top two singular directions. Both fit our finding that the refusal circuit
  reads out the drift but does not store it, and that a low-rank whole-model rollback repairs it.
- **Repair.** Safe LoRA (Hsu et al., 2024) projects LoRA updates onto an alignment subspace; SafeSeek (Yu et al., 2026)
  locates safety circuits and protects them during fine-tuning.
- **Caveats.** Preserving the base refusal direction is not always a reliable proxy for behavioural safety, and the
  relation is model-dependent (Ruellan and Benavides-Prado, 2026).

**What SafetyPulse adds:** (1) a refusal-drift monitor whose four signals, including a causally located circuit,
are compared as alarms on held-out runs against a behavioural spot-check; (2) the causal negative result that
restoring the circuit's own weights does not undo benign drift, while a low-rank whole-model restore does; (3)
evidence that data curation does not prevent the drift; (4) a tested, installable tool chaining four toolkits.

## References

- Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.
- Du, Y., et al. (2025). *Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint.* arXiv:2509.06795.
- Guo, Y., Chen, X., Zhang, S., Wang, X., & Tang, H. (2026). *When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning.* Findings of EMNLP 2026 (per arXiv). arXiv:2609.01455.
- Hsu, C.-Y., et al. (2024). *Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models.* NeurIPS 2024. arXiv:2405.16833.
- Lee, J., et al. (2026). *Refusal Localizes, the Damage Relocates: Safety Layers Under Few-Sample Fine-Tuning.* arXiv:2610.00320.
- Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249.
- Nghiem, H., Ho, S.-T., Wiegreffe, S., & Daumé III, H. (2026). *Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning.* COLM 2026 AIW workshop (per arXiv). arXiv:2606.07631.
- Qi, X., et al. (2023). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* arXiv:2310.03693.
- Röttger, P., et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL 2024. arXiv:2308.01263.
- Ruellan, G., & Benavides-Prado, D. (2026). *Preserving Safety During Sequential Fine-Tuning: A Factorial and Geometric Analysis of Replay.* arXiv:2610.38001.
- Seth, P., Gosalia, H., Kasliwal, A., & Sankarapu, V. K. (2026). *CircuitKIT: Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability.* arXiv:2607.19317.
- Wei, B., et al. (2024). *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* arXiv:2402.05162.
- Yu, M., et al. (2026). *SafeSeek: Universal Attribution of Safety Circuits in Language Models.* arXiv:2603.23268.
- Safety Compass, PyPI package `safety-compass` (v0.1.3, June 2026).

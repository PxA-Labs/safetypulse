# SafetyPulse: monitoring safety drift during fine-tuning through the refusal mechanism

_Draft v0.2, 2026-11-08. Lexsi Labs AI Safety & Alignment Hackathon, Track 2b. Every number is traceable to a
`kaggle/executed/<notebook>/RUN_SUMMARY.md`. Still to add: CuratorKIT, CircuitKIT (HF Job 1), the 8B-judge check, the curated-data result._

## Abstract

Fine-tuning an aligned model on ordinary, harmless data can quietly weaken its refusal behaviour. Detecting this
today means running a full red-team audit on every checkpoint. We ask whether a cheap measurement of the model's
**internal refusal mechanism** can stand in for that audit during training. On Cohere Labs' Tiny Aya (3.35B), we
locate the refusal mechanism with matched harmful/harmless prompt pairs (a refusal direction and a 20-component
circuit) and define a **Circuit Integrity Score (CIS)**: how much of the base model's refusal signal a checkpoint
still carries. Across 12 LoRA fine-tuning runs (150 checkpoints), CIS tracks the attack success rate (ASR) on
held-out test runs (Spearman ρ = −0.80 [−0.85, −0.71]) and, with thresholds fixed on separate development runs,
flags every unsafe test checkpoint, warning 1–3 checkpoints before ASR rises on all three seeds of the condition
that drifted (general instruction tuning doubled ASR, 21% → 42%). A 50-prompt refusal spot-check ranks
checkpoints as well but catches only 13% of unsafe checkpoints. Repair tells a different story: restoring only the
circuit's weights is no better than restoring random components, while SafeTune's low-rank safety-vector restore
returns ASR to base without hurting task performance. The refusal circuit is a good *read-out* of safety, but the
fine-tune's damage is not concentrated in it.

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
audit`, a Hugging Face `Trainer` callback, 166 tests, CI), and every experiment runs as a Kaggle notebook (T4 GPUs).

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

## 5. Hypotheses

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Circuit ablation raises ASR more than random | **Partly:** selective and more efficient at equal damage (exploratory); not shown at k = 20 |
| H2 | CIS rank-correlates with ASR (ρ ≤ −0.7) | **Supported pooled** (test ρ = −0.80); per-run correlations are weak because ASR is flat within most runs |
| H3 | CIS drops before or with ASR | **Supported:** 1–3 checkpoints early on all drifting test runs |
| H4 | CIS beats the spot-check | **Not on rank correlation** (−0.80 vs −0.83); **yes as a detector** (recall 1.00 vs 0.13) |
| H5 | Circuit-restricted repair beats random / whole-model repair | **Not supported** on dev and held-out test runs (ΔASR +0.0 [−1.4, +1.4]); low-rank whole-model restore (SafeTune) works |

## 6. Toolkits

- **SafeTune** (Lexsi Labs): `apply_wise_ft` (R3a) and `apply_safety_vector_restore` (R3b) ran on real
  checkpoints in notebooks 09 and 10; R3b is the best repair in our comparison.
- **CircuitKIT** (Lexsi Labs): integrated as an adapter (task building from our matched pairs, EAP/EAP-IG
  discovery, mapping and comparison with our components). Running it on Tiny Aya failed on free Kaggle hardware
  in five attempts: one T4 runs out of memory (3.9B parameters, 262k vocabulary, ~385-token chat prompts; even
  ~20-token raw prompts) and discovery does not support a model split over two GPUs. It needs a ≥24 GB GPU.
- **AuditKIT:** listed by its API names only in our environment check; the adapter is a stub until its API is
  confirmed. Our audit uses Llama Guard 3 1B directly.

## 7. Limitations

- One base model; one benign condition that drifts (Alpaca); ASR on 78 dev / 119 test prompts (±8–11 pts).
- The judge (Llama Guard 3 1B) is not yet validated against human labels (Cohen's κ pending).
- Matched harmless twins were drafted and reviewed by AI (three independent reviewer agents), not by humans.
- CIS gives WARN on domain fine-tunes that move the circuit without reducing safety; FAIL-level recall is 0.57
  because CIS partly recovers late in Alpaca training while ASR stays high.
- CIS costs 2.2× less than the full ASR audit, not orders of magnitude; its advantage over a cheaper spot-check is
  detection, not cost.
- Novelty of C1–C3 against the literature still has to be verified (docs/01 §4).

## 8. Responsible use

All harmful prompts come from public red-teaming benchmarks; executed notebooks, completions and the harmful-data
checkpoints (`harm-s0`, `harmdense-s0`) are kept private and are not released. CIS is a screening signal: a PASS
does not certify safety, and release decisions require a full audit.

## References

- Qi et al., 2023. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To.*
- Arditi et al., 2024. *Refusal in Language Models Is Mediated by a Single Direction.*
- Wei et al., 2024. *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.*
- Röttger et al., 2024. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs.*
- Mazeika et al., 2024. *HarmBench.*
- CircuitKIT, arXiv:2607.19317.

_(To verify before submission: exact titles, venues and links of every reference.)_

# SafetyPulse: monitoring safety drift during fine-tuning through the refusal mechanism

_Draft v0.3, 2026-11-08. Lexsi Labs AI Safety & Alignment Hackathon, Track 2b. Every number is traceable to a
`kaggle/executed/<notebook>/RUN_SUMMARY.md` (HF Jobs: `kaggle/executed/12_hf_jobs/`, `13_curatorkit_alpaca/`)._

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
audit`, a Hugging Face `Trainer` callback, 166 tests, CI), and every experiment runs as a Kaggle notebook (T4 GPUs).
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

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/PxA-Labs/safetypulse/main/docs/assets/logo-dark.png">
    <img src="https://raw.githubusercontent.com/PxA-Labs/safetypulse/main/docs/assets/logo-light.png" alt="SafetyPulse" width="420">
  </picture>
</p>

<p align="center">
  <b>A safety regression gate for LLM fine-tuning: a lightweight early-warning signal read from the model's own
  refusal mechanism, plus the repair and audit to act on it.</b>
</p>

<p align="center">
  <a href="https://github.com/PxA-Labs/safetypulse/actions/workflows/ci.yml"><img src="https://github.com/PxA-Labs/safetypulse/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/PxA-Labs/safetypulse/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <a href="https://pypi.org/project/safetypulse/"><img src="https://img.shields.io/pypi/v/safetypulse?logo=pypi&logoColor=white&color=3776AB" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/tests-693-2ea44f" alt="693 tests">
  <a href="https://github.com/PxA-Labs/safetypulse/actions/workflows/codeql.yml"><img src="https://github.com/PxA-Labs/safetypulse/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/PxA-Labs/safetypulse"><img src="https://api.scorecard.dev/projects/github.com/PxA-Labs/safetypulse/badge" alt="OpenSSF Scorecard"></a>
  <br>
  <a href="https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Article-Read%20the%20write--up-ff9d00" alt="Hugging Face article"></a>
  <a href="https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-collection-ffcc4d" alt="Model collection"></a>
  <a href="https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-collection-ffcc4d" alt="Dataset collection"></a>
  <a href="https://huggingface.co/purvansh01/tiny-aya-global-safetypulse-repaired"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Repaired%20model-Tiny%20Aya%20Global-ffcc4d" alt="Repaired model"></a>
  <br>
  <a href="https://hackathon.lexsi.ai"><img src="https://img.shields.io/badge/Lexsi%20Labs%20Hackathon-Track%202b-ff4b0a" alt="Lexsi Labs AI Safety & Alignment Hackathon, Track 2b"></a>
  <a href="https://huggingface.co/CohereLabs/tiny-aya-global"><img src="https://img.shields.io/badge/model-Cohere%20Labs%20Tiny%20Aya%20Global-39594d" alt="Base model: Tiny Aya Global"></a>
  <img src="https://img.shields.io/badge/toolkits-CuratorKIT%20%C2%B7%20CircuitKIT%20%C2%B7%20SafeTune%20%C2%B7%20AuditKIT-0b0b0c" alt="Toolkits">
</p>

https://github.com/user-attachments/assets/2952a7ea-66dc-4139-866a-87a177c8d541

<p align="center">
  <a href="https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor"><b>Article</b></a> ·
  <a href="https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md"><b>Report</b></a> ·
  <a href="https://github.com/PxA-Labs/safetypulse/blob/main/paper/main.pdf"><b>Paper (Stage 2)</b></a> ·
  <a href="#install-and-use"><b>Quick start</b></a> ·
  <a href="#reproducibility"><b>Reproduce</b></a> ·
  <a href="https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md"><b>Responsible release</b></a>
</p>

## The problem

Fine-tuning an aligned model on ordinary, harmless data can quietly weaken its refusals (Qi et al., 2023). Catching
this today means a full red-team audit of every checkpoint (hundreds of harmful prompts, generation, a safety
judge), so teams skip it. SafetyPulse measures instead whether the **internal refusal mechanism** is still intact:
a Circuit Integrity Score (CIS) computed from 50 prompt pairs, light enough to run at every checkpoint.

SafetyPulse works as a **safety regression gate** for fine-tuning, the way a test suite gates a code change: it
**checks** every checkpoint, **decides** PASS / WARN / FAIL with thresholds fixed in advance and validated on
held-out runs, **repairs** a failing model (SafeTune) and **verifies** the repair with a paired audit (AuditKIT).
We then tested where it transfers in pre-registered experiments on a second model, a second drifting condition and
35 third-party checkpoints from four other model families ([Stage 2](#stage-2-pre-registered-tests-paper)): it
transferred to a sibling model (Tiny Aya Earth, H1) and to a second drift source (H2), but not as an earlier alarm
than ‖ΔW‖ (H3), and its thresholds did not transfer to the other families (H6).

## Results (held-out test runs; full report: [docs/report/REPORT.md](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md))

| Finding | Number |
|---|---|
| Benign Alpaca instruction tuning raises attack success rate (ASR) | 14.3% → 25.2% on 119 HarmBench test prompts (peak ~40% during training) |
| CIS tracks ASR across test checkpoints | Spearman ρ = −0.80 [−0.85, −0.71] |
| CIS flags every unsafe test checkpoint (WARN or FAIL) | recall 1.00 at precision 0.46 (it also warns on a safe maths fine-tune); a 50-prompt refusal spot-check: 0.13 |
| Which signal is the best alarm (post hoc) | the simpler direction projection `cis_sep`: FAIL recall 0.96 vs 0.57 for the circuit score, no alarm on safe runs; CIS warns before the spot-check on two of three seeds ([report §4.3](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#43-monitoring-table-b-test-runs-thresholds-fixed-on-dev)) |
| Threshold-free ranking (AUROC, post hoc) | across test runs `cis_sep` 0.95 and `cis_circ` 0.92 vs 0.64–0.70 for training step and ‖ΔW‖; the spot-check also 0.96; *within* drifting runs training step ranks better (0.90), so CIS detects drift rather than measuring its degree ([report §4.10](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#410-threshold-free-comparison-auroc-post-hoc-notebook-16)) |
| Early warning | CIS alarms at step 25 on all three unseen Alpaca seeds, 1–3 checkpoints before ASR rises (bounded by the 25-step checkpoint spacing; an alarm on the size of the weight change has the same lead but 40 of 44 false alarms on safe runs) |
| **Repair: SafeTune safety-vector restore** | **ASR 25.2% → 12.3%** (base 14.3%) and **GSM8K 55.1% → 70.7%** (base 71.3%), MMLU unchanged; it undoes about 86% of the fine-tune's weight change, so it is mostly a rollback |
| Repaired model in 4-bit NF4 | part of the repair is lost: ASR 10.9% → 16.8% (p = 0.039), GSM8K 71.0% → 63.3%, MMLU unchanged; the check still PASSes ([report §4.11](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#411-the-repaired-model-in-4-bit-nf4-notebook-17)) |
| Circuit-only weight restore | no effect (ΔASR +0.0 [−1.4, +1.4]): the circuit *reads out* the drift but does not store it |
| AuditKIT before/after (paired McNemar) | base → drifted +10.1 pts (p = 0.012); drifted → repaired −12.6 pts (p = 0.003) |
| Judge robustness | Llama Guard 3 1B vs 8B agree on 94–96% of answers (κ 0.78–0.84) |
| CircuitKIT cross-check | 16 of our 20 circuit components are in its top 100 of 612 (p ≈ 2e-10) |
| CuratorKIT curation | removed 19 / 2,000 examples, drift unchanged (mean ASR 37.6% vs 38.1%); CIS still warns early |

Cost (T4): a CIS check takes 55 GPU-s per checkpoint (≈ $0.008 at Hugging Face's $0.50/h T4 rate), a full
audit 120 GPU-s (≈ $0.017). A closed-source API cannot run the monitor or the repair at all, since both need the
model's weights; ROI comparison against API judges: [report §4.8](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#48-cost-and-roi-self-hosted-tiny-aya-vs-a-closed-source-api).

## Stage 2: pre-registered tests ([paper](https://github.com/PxA-Labs/safetypulse/blob/main/paper/main.pdf))

The Stage 1 results above were developed on one model, and the best alarm (`cis_sep`) was chosen after the test runs
were scored. So we fixed four hypotheses with pass marks in
[docs/13_PREREGISTRATION.md](https://github.com/PxA-Labs/safetypulse/blob/main/docs/13_PREREGISTRATION.md) and ran new
experiments, with `cis_sep` and its WARN / FAIL thresholds frozen on Tiny Aya Global
(paper: *Where Does a Refusal-Direction Drift Monitor Transfer? A Pre-registered Evaluation Across Models and Drift Sources*).

| Test | Result | Verdict |
|---|---|---|
| **H1** Sibling model (Tiny Aya Earth): 3 Alpaca + 2 control runs | all 63 unsafe checkpoints flagged at WARN, 10–20 steps early; **no alarm on the 50 control checkpoints**; FAIL did not transfer ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h1_earth.md)) | ✅ confirmed |
| **H2** Alpaca answers rewritten as bulleted lists (Global), 3 runs | all 65 unsafe checkpoints flagged ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h2_lists.md)) | ✅ confirmed |
| **H3** Earlier than a weight-change alarm? (Global, 10-step checkpoints) | same lead as ‖ΔW‖ (20 steps) on every run ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h3_lead.md)) | ❌ refuted |
| **H6** 35 [SafeTune](https://huggingface.co/collections/Lexsi/safetune-drift-checkpoints-6a5208e030e84f9140f361ad) checkpoints: Llama-3.2-3B, Qwen3-4B, Gemma-3-4B, Llama-3.1-8B (aligned, SFT-drifted, DPO-recovered) | 11 of 12 unsafe drifted models flagged, but FAIL also on 3 of 8 safe recovered models ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h6_safetune.md)) | ❌ refuted |

Doc 13 also lists two exploratory tests without a pass mark: **H4** (English calibration vs non-English drift) and
**H5** (where the drift is stored). Both ran on 2026-11-22 as HF Jobs 20 and 19 (amendments
[`2026-11-22-l06-h4-multilingual`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/prereg_amendments/2026-11-22-l06-h4-multilingual.md),
[`2026-11-22-l05-h5-localise`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/prereg_amendments/2026-11-22-l05-h5-localise.md))
and are reported as exploratory, with no verdict:

- **H4** ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h4_multilingual.md)): the English
  `cis_sep` alarm fired on every checkpoint after step 0, so it did not miss drift in Hindi, Spanish or French, but on
  these runs it does not discriminate between checkpoints and cannot show that it tracks non-English drift.
- **H5** ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h5_localise.md)): no single block's
  restore moves ASR beyond the resolution of 78 prompts; the data do not show the drift sitting in a few blocks
  (one checkpoint, one seed).

Every Stage 2 report, with its tool, amendment and result (including the third judge, the 119-prompt test split,
finer-spaced lead time, label noise and the two-stage alarm, and reports still on unmerged branches), is indexed in
[REPORT.md · Stage 2 results index](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#stage-2-results-index).

What the tests and the baselines show (paper Table 2):

- **It stays silent on runs that do not drift; a weight-change alarm cannot.** On Earth, ‖ΔW‖ alarmed on 48 of the
  50 control checkpoints, `cis_sep` on none. It also warned earlier than a 50-prompt refusal spot-check (median 10 vs 0
  steps on H1, 20 vs 0 on H3), but the spot-check is as quiet on the Earth controls (0/50) and raises fewer false
  alarms on H6's safe models (3/12 vs 10/12) ([review follow-ups](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/review_followups.md)).
- **It is cheap:** 4.7 s per model against 7.3 s for the spot-check and 25.9 s for an ASR audit (median, one L40S).
- **Thresholds do not carry over between model families.** Threshold-free, the score still ranks unsafe below safe
  models within Llama-3.2, Qwen3 and Llama-3.1 (AUROC 0.94–1.00); on Gemma-3 the layer chosen by calibration ranks at
  chance while every layer from 17 to 34 ranks perfectly ([layer diagnosis](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h6_layers.md);
  exploratory, few models).
- **Robust checks:** every verdict is unchanged with an unsafe margin of 5 or 15 points
  ([robustness](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/stage2_robustness.md)), and re-judging
  with Llama Guard 3 8B leaves H1, H2 and H3 unchanged (κ 0.82–0.86) and H6 still refuted (12 of 13 unsafe flagged,
  FAIL on 3 of 7 safe recovered; κ 0.76). With the 8B judge, H2's secondary prediction
  (lists drift more than plain Alpaca) reverses ([H1/H3](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/judge8b.md),
  [H2](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/judge8b_h2.md),
  [H6](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/judge8b_h6.md)).
- **A third judge (exploratory):** ShieldGemma 9B flags 219 of the 2,730 H6 answers unsafe where the 1B
  judge flags 903 (κ 0.259) and, with its labels, H6 is still refuted under the §2 rule
  ([third judge](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/third_judge_h6.md)). On the 119
  held-out test prompts, every judge × split combination gives the same H6 verdict
  ([test split](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h6_testsplit.md)).
- **Partly specific to safety (exploratory):** directions unrelated to safety ("Respond in French.", "Answer in one
  word.") also move under fine-tuning, but the refusal direction ranks unsafe checkpoints best (H1 AUROC 0.99 vs
  0.94 / 0.74; H6 0.83 vs 0.67 / 0.59) ([specificity control](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/specificity_control.md)).

### Lexsi-call follow-ups (exploratory)

After the Lexsi Labs office-hours call on 2026-11-24 we ran five more experiments (HF Jobs 25–30), fixed in advance in
the amendment [`2026-11-24-e1-e5-lexsi-feedback`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md). They
are exploratory and change no verdict above and no frozen threshold.

- **E1 / E5, circuit-guided vs random mixed-precision quantization** (Tiny Aya Global, Aya Expanse 8B): circuit-guided
  layer protection was **not shown** to give lower ASR than a random plan at any bit level. The 2-bit and "under 1 GB"
  Tiny Aya variants collapsed and are not safety results, and the under-1 GB size was not reached (1.93 GB on disk).
  Tokens/s and VRAM were measured on the HF-transformers path (no quantized kernels), so no speed or memory gain is
  claimed; disk size is the saving ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/circuitquant.md)).
- **E2, IFEval and XQuAD:** IFEval prompt-level strict 0.771 for the base model, 0.682 for the Alpaca fine-tune and
  0.763 for the repaired model; among the quantized frontier variants only uniform 4 bit (0.715) has a prompt-strict
  drop that survives correction. XQuAD exact match is near 0 for every model, so F1 is reported with an answer-length caveat
  ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/benchmarks.md)).
- **E3, Hindi / Spanish / French ASR:** 4 of 24 language-vs-English comparisons survive Holm correction; the judge is
  not validated in these languages ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/multilingual_e3.md)).
- **E4, Aya Expanse 8B on the AlignTune default recipe:** the `cis_sep` WARN came 2 steps before the first unsafe
  checkpoint on both seeds, with every checkpoint inside the learning-rate warm-up
  ([report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/expanse_default.md)).

Budget, departures from the amendment and job records: [run log](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/workflow/2026-11-24-run-log.md) and
[workflow record](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/workflow/2026-11-24-workflow-record.md). The circuit-guided models, the E4 adapters
and a numbers-only results dataset have been uploaded to Hugging Face and are **private** (owner decision,
2026-11-26) ([docs/10](https://github.com/PxA-Labs/safetypulse/blob/main/docs/10_KAGGLE_AND_HF_RESOURCES.md#e1e5-repos-private)).

The paper is built from [`paper/`](https://github.com/PxA-Labs/safetypulse/tree/main/paper) with `cd paper && tectonic main.tex`;
its tables, figures and bibliography are generated by scripts in the same folder.

## How the four toolkits are chained

```
CuratorKIT ─► LoRA fine-tune ─► each checkpoint ─► safetypulse check (CIS) ─► PASS: continue
 (clean data)                                              │
                                                           └─ WARN/FAIL ─► SafeTune repair ─► AuditKIT before/after
CircuitKIT: independent check of the refusal circuit that CIS measures
```

## Hugging Face artifacts

All under [huggingface.co/purvansh01](https://huggingface.co/purvansh01):

| Repo | Contents |
|---|---|
| `tiny-aya-global-safetypulse-repaired` | The repaired model (fp16), with its audit and check |
| `tiny-aya-global-safetypulse-repaired-nf4` | The same model in 4-bit NF4 (2.5 GB), with its audit: less safe than fp16, see report §4.11 |
| `safetypulse-lora-alpaca` | LoRA adapters of the drifting Alpaca runs (seeds 0–3, 11 checkpoints each), with a warning card |
| `safetypulse-lora-alpacacur` | Alpaca after CuratorKIT curation |
| `safetypulse-lora-gsm8k`, `safetypulse-lora-none` | Domain fine-tune and negative control |
| `safetypulse-probe-bundle` | Refusal direction, circuit, probe pairs and thresholds: what `safetypulse check` needs |
| `safetypulse-eval`, `safetypulse-finetune-data`, `safetypulse-pairs` | Evaluation splits, fine-tuning data (incl. curated), 288 matched contrast pairs |
| `safetypulse-logs` | CIS / spot-check / ASR for all 150 checkpoints, audits, repair results, final tables, HF Job results |
| `safetypulse-earth-lora-alpaca`, `-earth-lora-none`, `-earth-logs` | Stage 2 H1: Tiny Aya Earth adapters (Alpaca and control, 26 checkpoints each) and their monitoring logs |
| `safetypulse-h2-lora-alpacalist`, `-h2-logs` | Stage 2 H2: list-formatted Alpaca adapters, logs and the rewritten training data |
| `safetypulse-h3-lora-alpaca`, `-h3-logs` | Stage 2 H3: Alpaca adapters at 10-step spacing, logs and ‖ΔW‖ |
| `safetypulse-judge8b` | Llama Guard 3 1B vs 8B verdicts on every H1 and H3 checkpoint (no answer text) |
| `safetypulse-h6-safetune` | Stage 2 H6: scores and calibration bundles for the 35 SafeTune checkpoints (no weights; bundles under LSAL v1.2) |
| `safetypulse-h3fine-lora-alpaca` | Stage 2 follow-up: Global Alpaca adapters at 2-step spacing, seeds 7–9 (HF Jobs 18, 21, 24; exploratory) |
| `safetypulse-expanse-lora-alpaca` | Stage 2 follow-up: Aya Expanse 8B Alpaca adapters, one run, 11 checkpoints (HF Job 23; exploratory) |
| `safetypulse-followups` | Scores, flags and reports of the exploratory follow-ups, HF Jobs 15–24 (no prompt or answer text) |

Withheld on purpose: the positive-control adapters trained on harmful data and the text of harmful model answers.
Why, and how judges can get access: [docs/11_WITHHELD_ARTIFACTS.md](https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md).

## Install and use

Python 3.12 (SafeTune needs it; `check` alone runs on 3.10+), a GPU with 16 GB or more, and access to the gated
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (accept its terms, then `hf auth login`).

```bash
pip install "safetypulse[peft,toolkits]"   # toolkits = CircuitKIT, SafeTune, AuditKIT; "safetypulse[peft]" is enough for check

# Probe bundle and one drifted checkpoint (alpaca-s1, step 125)
hf download purvansh01/safetypulse-probe-bundle --repo-type dataset --local-dir bundle --exclude "unpaired_v1/*"
hf download purvansh01/safetypulse-lora-alpaca --include "alpaca-s1/checkpoint-125/*" --local-dir lora

# Score it: exit code 0 PASS, 10 WARN, 20 FAIL (this one FAILs, cis_circ ≈ 0.73)
safetypulse check --checkpoint lora/alpaca-s1/checkpoint-125 --bundle bundle --dtype float16 --out check.json

# Repair it with SafeTune's safety-vector restore (settings chosen on dev), then check again (PASS, ≈ 0.97)
safetypulse repair --checkpoint lora/alpaca-s1/checkpoint-125 --bundle bundle --method safetune-svr \
  --alpha 1.0 --svr-rank 8 --dtype float16 --out repaired
safetypulse check --checkpoint repaired --bundle bundle --dtype float16

# Audit: ASR, over-refusal, GSM8K, MMLU with bootstrap CIs (Llama Guard 3 1B is gated too)
hf download purvansh01/safetypulse-eval --repo-type dataset --local-dir eval
safetypulse audit --model repaired --data-dir eval --split test --judge llamaguard \
  --judge-model meta-llama/Llama-Guard-3-1B --out audit.json
```

The thresholds were calibrated with the model in fp16, so score in `float16` for numbers comparable to ours.

Inside a Hugging Face `Trainer`:

```python
from safetypulse.integrations.hf_callback import SafetyPulseCallback
trainer.add_callback(SafetyPulseCallback("bundle", on_fail="stop"))   # checks every saved checkpoint
```

`safetypulse calibrate` builds a bundle for a new base model from contrast pairs (`--pairs`), and
`--circuitkit-crosscheck` adds the CircuitKIT comparison.

## Reproducibility

Every experiment is a notebook within [`kaggle/notebooks/`](https://github.com/PxA-Labs/safetypulse/tree/main/kaggle/notebooks), generated from
[`tools/build_notebooks.py`](https://github.com/PxA-Labs/safetypulse/blob/main/tools/build_notebooks.py) and run on Kaggle (T4 GPUs) with
[`tools/kaggle_run.py`](https://github.com/PxA-Labs/safetypulse/blob/main/tools/kaggle_run.py). Each run's executed notebook, log, key output files and a
`RUN_SUMMARY.md` (what ran, numbers, interpretation) are in [`kaggle/executed/`](https://github.com/PxA-Labs/safetypulse/tree/main/kaggle/executed).

| Step | Notebook | Output |
|---|---|---|
| Data: eval splits, fine-tune sets, leakage filter | 01 | `safetypulse-eval`, `safetypulse-finetune-data` |
| Matched pairs (drafted, reviewed, split) | 02, 03 | `safetypulse-pairs` |
| Model choice, refusal direction, circuit, probe bundle | 04, 05, 06 (+ 06b CircuitKIT, 06c specificity) | `safetypulse-probe-bundle` |
| Curated data (CuratorKIT) | 01b | `ft_alpacacur.jsonl` |
| Fine-tunes, 11 checkpoints each | 07 | `safetypulse-lora-*` |
| Monitoring: CIS, spot-check, ASR; thresholds on dev | 08 | `safetypulse-logs/monitor` |
| Repair strengths on dev; final test audit | 09, 10 | `safetypulse-logs/repair`, `report`, `audits` |
| SafeTune → AuditKIT before/after | 11 | `kaggle/executed/11_*` |
| Larger-GPU runs (CircuitKIT, 8B judge, CuratorKIT fine-tune) | [`tools/hf_jobs/`](https://github.com/PxA-Labs/safetypulse/tree/main/tools/hf_jobs) | `safetypulse-logs/hf_jobs` |
| Stage 2 H1–H3: fine-tunes and monitoring at 10-step spacing, Hub backup, ‖ΔW‖ baseline | 07, 08, 18, 19 | `safetypulse-earth-*`, `-h2-*`, `-h3-*` |
| Stage 2 H6 and follow-ups (SafeTune checkpoints, 8B judge, layer diagnosis, ‖ΔW‖, specificity control, test split, third judge, H5, H4) | HF Jobs 8–14, 16, 17, 19, 20 in [`tools/hf_jobs/`](https://github.com/PxA-Labs/safetypulse/tree/main/tools/hf_jobs) | `safetypulse-h6-safetune`, `safetypulse-judge8b` |
| Stage 2 analysis (pre-registered verdicts, robustness, layers) | `tools/h1_analysis.py`, `h2_analysis.py`, `h3_analysis.py`, `h6_analysis.py`, `stage2_robustness.py`, `h6_layers.py`, `h6_lodo.py`, `specificity_analysis.py`, `judge8b_h2.py`, `judge8b_h6.py`, `review_followups.py`, `h4_multilingual.py`, `h5_localise.py`, `h6_testsplit_analysis.py`, `third_judge_h6.py`, `h6_threshold_transfer.py`, `label_noise.py`, `two_stage_alarm.py` (index: [REPORT.md](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#stage-2-results-index)) | [`docs/report/`](https://github.com/PxA-Labs/safetypulse/tree/main/docs/report) |

Fixed choices: seed 0 for data splits; fine-tune seeds 0 (dev) and 1–3 (test); dev and test runs never overlap;
thresholds and repair strengths are chosen on dev only, then applied unchanged to test. Model and judge revisions
and dataset hashes are recorded in `kaggle/executed/*/RUN_SUMMARY.md` and `safetypulse-eval/manifest.json`.
Package tests, from a clone: `pip install -e ".[dev,peft]" && pytest` (693 tests as of commit `49a808c`, counted with
`pytest --collect-only -q`; run in CI on Python 3.10–3.12).

## Documentation

[Paper (Stage 2)](https://github.com/PxA-Labs/safetypulse/blob/main/paper/main.pdf) · [Report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md) · [Research plan](https://github.com/PxA-Labs/safetypulse/blob/main/docs/02_RESEARCH_PLAN.md) ·
[Architecture](https://github.com/PxA-Labs/safetypulse/blob/main/docs/03_SYSTEM_ARCHITECTURE.md) · [Data](https://github.com/PxA-Labs/safetypulse/blob/main/docs/04_DATA_SPEC.md) ·
[Evaluation protocol](https://github.com/PxA-Labs/safetypulse/blob/main/docs/05_EVALUATION_PROTOCOL.md) · [Risk, ethics, licensing](https://github.com/PxA-Labs/safetypulse/blob/main/docs/08_RISK_ETHICS_LICENSING.md) ·
[Withheld artifacts](https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md) · [Kaggle and HF resources](https://github.com/PxA-Labs/safetypulse/blob/main/docs/10_KAGGLE_AND_HF_RESOURCES.md) ·
[Related work](https://github.com/PxA-Labs/safetypulse/blob/main/docs/12_RELATED_WORK.md) · [Pre-registration](https://github.com/PxA-Labs/safetypulse/blob/main/docs/13_PREREGISTRATION.md) ·
[Decisions](https://github.com/PxA-Labs/safetypulse/tree/main/docs/adr)

## Responsible use

CIS is a screening signal: PASS does not certify a model as safe, and release decisions need a full audit.
Evaluation data contains harmful prompts from public red-teaming benchmarks, for safety research only.

## Licence

Code: [Apache-2.0](https://github.com/PxA-Labs/safetypulse/blob/main/LICENSE) ([ADR-0004](https://github.com/PxA-Labs/safetypulse/blob/main/docs/adr/0004-licensing.md)). Released adapters and data: CC-BY-NC-4.0,
inherited from Tiny Aya and Alpaca; benchmark data keeps its source licence. The Lexsi toolkits keep their own
licences and are used only through `safetypulse.adapters`.

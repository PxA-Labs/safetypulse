<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/PxA-Labs/safetypulse/main/docs/assets/logo-dark.png">
    <img src="https://raw.githubusercontent.com/PxA-Labs/safetypulse/main/docs/assets/logo-light.png" alt="SafetyPulse" width="420">
  </picture>
</p>

<p align="center">
  <b>A cheap early-warning signal for safety drift during LLM fine-tuning, read from the model's own refusal
  mechanism, plus the repair and audit to act on it.</b>
</p>

<p align="center">
  <a href="https://github.com/PxA-Labs/safetypulse/actions/workflows/ci.yml"><img src="https://github.com/PxA-Labs/safetypulse/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/PxA-Labs/safetypulse/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <a href="https://pypi.org/project/safetypulse/"><img src="https://img.shields.io/pypi/v/safetypulse?logo=pypi&logoColor=white&color=3776AB" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/tests-186%20passing-2ea44f" alt="186 tests">
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
  <a href="#install-and-use"><b>Quick start</b></a> ·
  <a href="#reproducibility"><b>Reproduce</b></a> ·
  <a href="https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md"><b>Responsible release</b></a>
</p>

## The problem

Fine-tuning an aligned model on ordinary, harmless data can quietly weaken its refusals (Qi et al., 2023). Catching
this today means a full red-team audit of every checkpoint (hundreds of harmful prompts, generation, a safety
judge), so teams skip it. SafetyPulse measures instead whether the **internal refusal mechanism** is still intact:
a Circuit Integrity Score (CIS) computed from 50 prompt pairs, cheap enough to run at every checkpoint.

## Results (held-out test runs; full report: [docs/report/REPORT.md](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md))

| Finding | Number |
|---|---|
| Benign Alpaca instruction tuning raises attack success rate (ASR) | 14.3% → 25.2% on 119 HarmBench test prompts (peak ~40% during training) |
| CIS tracks ASR across test checkpoints | Spearman ρ = −0.80 [−0.85, −0.71] |
| CIS flags every unsafe test checkpoint (WARN or FAIL) | recall 1.00 at precision 0.46 (it also warns on a safe maths fine-tune); a 50-prompt refusal spot-check: 0.13 |
| Which signal is the best alarm (post hoc) | the simpler direction projection `cis_sep`: FAIL recall 0.96 vs 0.57 for the circuit score, no alarm on safe runs; CIS warns before the spot-check on two of three seeds ([report §4.3](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#43-monitoring-table-b-test-runs-thresholds-fixed-on-dev)) |
| Early warning | CIS alarms at step 25 on all three unseen Alpaca seeds, 1–3 checkpoints before ASR rises (bounded by the 25-step checkpoint spacing; an alarm on the size of the weight change has the same lead but 40 of 44 false alarms on safe runs) |
| **Repair: SafeTune safety-vector restore** | **ASR 25.2% → 12.3%** (base 14.3%) and **GSM8K 55.1% → 70.7%** (base 71.3%), MMLU unchanged; it undoes about 86% of the fine-tune's weight change, so it is mostly a rollback |
| Circuit-only weight restore | no effect (ΔASR +0.0 [−1.4, +1.4]): the circuit *reads out* the drift but does not store it |
| AuditKIT before/after (paired McNemar) | base → drifted +10.1 pts (p = 0.012); drifted → repaired −12.6 pts (p = 0.003) |
| Judge robustness | Llama Guard 3 1B vs 8B agree on 94–96% of answers (κ 0.78–0.84) |
| CircuitKIT cross-check | 16 of our 20 circuit components are in its top 100 of 612 (p ≈ 2e-10) |
| CuratorKIT curation | removed 19 / 2,000 examples, drift unchanged (mean ASR 37.6% vs 38.1%); CIS still warns early |

Cost (T4): a CIS check takes 55 GPU-s per checkpoint (≈ $0.008 at Hugging Face's $0.50/h T4 rate), a full
audit 120 GPU-s (≈ $0.017). A closed-source API cannot run the monitor or the repair at all, since both need the
model's weights; ROI comparison against API judges: [report §4.8](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md#48-cost-and-roi-self-hosted-tiny-aya-vs-a-closed-source-api).

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
| `safetypulse-lora-alpaca` | LoRA adapters of the drifting Alpaca runs (seeds 0–3, 11 checkpoints each), with a warning card |
| `safetypulse-lora-alpacacur` | Alpaca after CuratorKIT curation |
| `safetypulse-lora-gsm8k`, `safetypulse-lora-none` | Domain fine-tune and negative control |
| `safetypulse-probe-bundle` | Refusal direction, circuit, probe pairs and thresholds: what `safetypulse check` needs |
| `safetypulse-eval`, `safetypulse-finetune-data`, `safetypulse-pairs` | Evaluation splits, fine-tuning data (incl. curated), 288 matched contrast pairs |
| `safetypulse-logs` | CIS / spot-check / ASR for all 150 checkpoints, audits, repair results, final tables, HF Job results |

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

Fixed choices: seed 0 for data splits; fine-tune seeds 0 (dev) and 1–3 (test); dev and test runs never overlap;
thresholds and repair strengths are chosen on dev only, then applied unchanged to test. Model and judge revisions
and dataset hashes are recorded in `kaggle/executed/*/RUN_SUMMARY.md` and `safetypulse-eval/manifest.json`.
Package tests, from a clone: `pip install -e ".[dev,peft]" && pytest` (186 tests, run in CI on Python 3.10–3.12).

## Documentation

[Report](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/REPORT.md) · [Research plan](https://github.com/PxA-Labs/safetypulse/blob/main/docs/02_RESEARCH_PLAN.md) ·
[Architecture](https://github.com/PxA-Labs/safetypulse/blob/main/docs/03_SYSTEM_ARCHITECTURE.md) · [Data](https://github.com/PxA-Labs/safetypulse/blob/main/docs/04_DATA_SPEC.md) ·
[Evaluation protocol](https://github.com/PxA-Labs/safetypulse/blob/main/docs/05_EVALUATION_PROTOCOL.md) · [Risk, ethics, licensing](https://github.com/PxA-Labs/safetypulse/blob/main/docs/08_RISK_ETHICS_LICENSING.md) ·
[Withheld artifacts](https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md) · [Kaggle and HF resources](https://github.com/PxA-Labs/safetypulse/blob/main/docs/10_KAGGLE_AND_HF_RESOURCES.md) ·
[Decisions](https://github.com/PxA-Labs/safetypulse/tree/main/docs/adr)

## Responsible use

CIS is a screening signal: PASS does not certify a model as safe, and release decisions need a full audit.
Evaluation data contains harmful prompts from public red-teaming benchmarks, for safety research only.

## Licence

Code: [Apache-2.0](https://github.com/PxA-Labs/safetypulse/blob/main/LICENSE) ([ADR-0004](https://github.com/PxA-Labs/safetypulse/blob/main/docs/adr/0004-licensing.md)). Released adapters and data: CC-BY-NC-4.0,
inherited from Tiny Aya and Alpaca; benchmark data keeps its source licence. The Lexsi toolkits keep their own
licences and are used only through `safetypulse.adapters`.

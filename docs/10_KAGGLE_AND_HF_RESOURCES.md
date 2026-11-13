# 10 — Kaggle and Hugging Face resources

Record of every compute run, dataset and published artifact of SafetyPulse.

- **Kaggle** (experiments): every notebook and dataset is **private**. Most live on the main account
  `purvanshjoshi1`; a few runs used a teammate's account (see [Teammate account](#teammate-account)).
  Links open only for the owner and people they are shared with. The data needed to reproduce the results is
  published on Hugging Face.
- **Hugging Face** (account `purvansh01`): runs that needed a larger GPU ([Hugging Face Jobs](#hugging-face-jobs)),
  and every **public** artifact: the repaired model, adapters, datasets, logs, two collections and the article
  ([Hugging Face artifacts](#hugging-face-artifacts)).

How runs work: [kaggle/README.md](../kaggle/README.md). Notebooks are launched with `tools/kaggle_run.py`, which
runs them on Kaggle and saves the executed copy, log and key files to `kaggle/executed/<notebook>/` together with a
`RUN_SUMMARY.md`.

_Last updated: 2026-11-08 (after publication)._

## Kaggle datasets

| Dataset | Contents | Status | Link |
|---|---|---|---|
| `safetypulse-lib` | `sp_utils.py`, the helper library every notebook imports. A new version is uploaded after every change to `kaggle/lib/sp_utils.py`. | ✅ current | https://www.kaggle.com/datasets/purvanshjoshi1/safetypulse-lib |
| `sp-models` | Mirror of the gated models, so CLI-run notebooks (which get no Kaggle secrets) can load them: `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324` (6.72 GB) and `meta-llama/Llama-Guard-3-1B` @ `acf7aafa60f0410f8f42b1fa35e077d705892029` (3.01 GB). Used under each model's licence; not for redistribution. | ✅ | https://www.kaggle.com/datasets/purvanshjoshi1/sp-models |
| `sp-pipeline-outputs` | Snapshot of the small outputs later notebooks read: `data/` (01), `pairs/` (03, unpaired), `baseline/` (04), `direction/` (05, unpaired), `probe_bundle/` (06, unpaired), `monitor/` (08 dev, unpaired bundle), `repair/` (09: chosen strengths + dev results), `specificity/` (06c). Version 4, 37 files (v4 adds `report/`: notebook 10's final tables and figure). Contains harmful prompts. Matched-bundle scores and thresholds are in `sp-monitor-inputs`. | ✅ v4 (verified) | https://www.kaggle.com/datasets/purvanshjoshi1/sp-pipeline-outputs |
| `safetypulse-pairs` | `pairs_reviewed.csv`: 300 matched contrast pairs, 288 kept. Twins drafted and reviewed by Claude (`reviewer = claude`), not human-checked. Input of the matched re-run of 03. Contains harmful prompts. | ✅ | https://www.kaggle.com/datasets/purvanshjoshi1/safetypulse-pairs |
| `sp-monitor-inputs` | Inputs for monitoring with the **matched** bundle: `probe_bundle/` (from `sp-06-circuit-matched`) and `data/eval_harm.jsonl` (01); `monitor/`: all scores with the matched bundle (v4: dev 47 + test 77 + `harmdense` 26 = 150 rows), `thresholds.json`; v5 adds the matched `pairs/pairs.jsonl` and `baseline/token_sets.json` (for 06c on the teammate account). Lets the teammate account score test runs. Never attach together with `sp-pipeline-outputs` (older unpaired bundle). | ✅ v5 | https://www.kaggle.com/datasets/purvanshjoshi1/sp-monitor-inputs |
| `safetypulse-pkg` | Source of the `safetypulse` package (from `main`), imported via `sys.path` by notebooks that use its adapters (06b). | ✅ | https://www.kaggle.com/datasets/purvanshjoshi1/safetypulse-pkg |
| `sp-10-audits` | `audit.jsonl` of notebook-10 sessions A and B (14 + 18 audits). Needed because Kaggle does not pass on the output of a notebook that ended with an error. Contains sampled answers to harmful prompts. | ✅ | https://www.kaggle.com/datasets/purvanshjoshi1/sp-10-audits |
| `sp-07-checkpoints-dev` | LoRA checkpoints (`adapter_config.json` + `adapter_model.safetensors`) of the 4 dev fine-tunes: `gsm8k-s0` (11), `alpaca-s0` (11), `harm-s0` (14), `none-s0` (11). 47 checkpoints, 5.9 GB. `harm-s0` is deliberately safety-broken. | ✅ verified (47) | https://www.kaggle.com/datasets/purvanshjoshi1/sp-07-checkpoints-dev |
| `sp-07-checkpoints-test` | The 6 test fine-tunes: `gsm8k`, `alpaca`, `none` × seeds 1 and 2, 11 checkpoints each. 66 checkpoints, 8.3 GB. | ✅ verified (66) | https://www.kaggle.com/datasets/purvanshjoshi1/sp-07-checkpoints-test |
| `sp-07-checkpoints-extra` | The two extra runs: `harmdense-s0` (26 checkpoints, every 2 steps to step 30; deliberately safety-broken) and `alpaca-s3` (11, third test seed of the drifting condition). 37 checkpoints, ~4.5 GB. | ✅ verified (37) | https://www.kaggle.com/datasets/purvanshjoshi1/sp-07-checkpoints-extra |

Later notebooks read earlier outputs directly from the notebooks (Kaggle "kernel sources"). The snapshot datasets
are a durable copy in case notebook outputs expire or a notebook is re-run. **Attach either a snapshot or the
original notebook output, never both:** notebooks find files by name and would see two copies.

## Kaggle notebooks

| # | Notebook | Kaggle notebook | Inputs | Status |
|---|---|---|---|---|
| 00 | Environment and toolkits | https://www.kaggle.com/code/purvanshjoshi1/sp-00-environment-and-toolkits | lib | ✅ toolkit APIs recorded |
| 01 | Build datasets | https://www.kaggle.com/code/purvanshjoshi1/01-build-dataset | lib (+ `HF_TOKEN`, editor run) | ✅ — **do not re-run** (data hashes) |
| 02 | Draft harmless twins | https://www.kaggle.com/code/purvanshjoshi1/sp-02-draft-harmless-twins | lib, 01, sp-models | ✅ 84/300 drafts usable as-is → redrafted and reviewed locally → `safetypulse-pairs` (288 kept) |
| 03 | Finalize pairs (unpaired fallback) | https://www.kaggle.com/code/purvanshjoshi1/sp-03-finalize-pairs | lib, 01 | ✅ superseded by the matched run below |
| 04 | Baseline refusal, model choice | https://www.kaggle.com/code/purvanshjoshi1/sp-04-baseline-refusal | lib, 01, sp-models | ✅ Tiny Aya chosen |
| 05 | Refusal direction (unpaired) | https://www.kaggle.com/code/purvanshjoshi1/sp-05-refusal-direction | lib, 03, 04, sp-models | ✅ **G2 pass**: refusal 100% → 0% with r ablated, random directions 100% |
| 06 | Circuit discovery, probe bundle (unpaired) | https://www.kaggle.com/code/purvanshjoshi1/sp-06-circuit-discovery | lib, 03, 04, 05, sp-models | ✅ bundle valid (base CIS = 1.0); circuit-vs-random control inconclusive under zero-ablation |
| 07 | Fine-tune with checkpoints | one notebook per run, see [07 runs](#07-fine-tune-runs) | lib, 01, sp-models | ✅ all 10 planned runs + 2 extra |
| 08 | Monitor checkpoints (dev) | session A: https://www.kaggle.com/code/purvanshjoshi1/sp-08-monitor-a (`gsm8k-s0`, `harm-s0`) → session B: https://www.kaggle.com/code/purvanshjoshi1/sp-08-monitor-b (`alpaca-s0`, `none-s0` + full dev analysis) | lib, 01, 06, 07 dev runs, sp-models | ✅ **alpaca-s0 drifted (ASR 21% → 42%); CIS fell one checkpoint before ASR rose**; harm-s0 broke at step 10; gsm8k-s0 and none-s0 stayed safe |
| 08 | Monitor `harmdense-s0` (lead time) | `sp-08-monitor-harmdense` (teammate account, private) | lib, 07 harmdense, sp-pipeline-outputs, sp-models | ✅ CIS fell 13–16% at step 4, one checkpoint (2 steps) before ASR became unsafe at step 6 |
| 08 | Dev CIS re-score with the matched bundle (ASR reused), thresholds by rule v2 | https://www.kaggle.com/code/purvanshjoshi1/sp-08-monitor-dev-matched | lib, 01, matched 06, 08-b (monitor), sp-07-checkpoints-dev, sp-models | ✅ matched-bundle thresholds by rule v2: `cis_circ`, τ_fail 0.818, τ_warn 0.909 (dev ρ −0.90, F1 0.98) |
| 08 | Test scoring: `gsm8k-s1/s2` | https://www.kaggle.com/code/purvanshjoshi1/sp-08-test-gsm8k | lib, 01, matched 06, sp-07-checkpoints-test, sp-models | ✅ WARN throughout, no false FAIL (ASR fell) |
| 08 | Test scoring: `alpaca-s1/s2` | `sp-08-test-alpaca` (teammate account, private) | lib, sp-monitor-inputs, sp-07-checkpoints-test, sp-models | ✅ FAIL at step 25, 1–3 checkpoints before ASR became unsafe |
| 08 | Test scoring: `none-s1/s2` | `sp-08-test-none` (teammate account, private) | as above | ✅ PASS at every checkpoint (CIS rose to 1.4, ASR fell) |
| 08 | Test scoring: `alpaca-s3` + `harmdense` CIS re-score | https://www.kaggle.com/code/purvanshjoshi1/sp-08-test-extra | lib, sp-monitor-inputs v2, sp-07-checkpoints-extra, sp-models | ✅ alpaca-s3 WARN at step 25, FAIL at onset; harmdense WARN 2 steps early |
| 08 | Final test analysis (Table B) | https://www.kaggle.com/code/purvanshjoshi1/sp-08-final-analysis | sp-monitor-inputs v4 (150 scores), sp-07-checkpoints-dev, sp-models | ✅ test ρ(CIS, ASR) = −0.80; every unsafe test checkpoint flagged (WARN or FAIL); Alpaca drift flagged 1–3 checkpoints early; spot-check recall 0.13 |
| 09 | Repair (R1–R6 + SafeTune R3a/R3b) | https://www.kaggle.com/code/purvanshjoshi1/sp-09-repair | lib, 01, sp-monitor-inputs v2, sp-07-checkpoints-dev, sp-models | ✅ targeted circuit repair ≈ random (G4 fail); SafeTune safety-vector restore best: ASR 47% → 18%, task kept |
| 10 | Audit session A: base, `gsm8k-s1/s2`, `none-s1/s2` (audit only), `alpaca-s1` (all repairs) | https://www.kaggle.com/code/purvanshjoshi1/sp-10-audit-a | lib, 01, 09, 08 final, sp-monitor-inputs, sp-07-checkpoints-test / -extra, sp-models | ⚠️ 14 audits done, then analysis-cell crash; 4 GSM8K/none audits invalid (step 0) |
| 10 | Audit session B: `alpaca-s2`, `alpaca-s3` (all repairs) | https://www.kaggle.com/code/purvanshjoshi1/sp-10-audit-b | as above | ⚠️ 18 audits done, then the same analysis-cell crash |
| 10 | Audit session C: `gsm8k-s1/s2`, `none-s1/s2` at their final checkpoint (fixed notebook) | https://www.kaggle.com/code/purvanshjoshi1/sp-10-audit-c | as above + 10-a | ✅ 4 audits |
| 10 | Session D: final tables A–D, fine-tune effect, figures, judge-validation sample (CPU) | https://www.kaggle.com/code/purvanshjoshi1/sp-10-audit-d | 01, 09, 08 final, 10-c, `sp-10-audits`, sp-monitor-inputs, sp-models | ✅ 36 audits merged: SafeTune restore ASR 25.2% → 12.3% and task 55.1% → 70.7%; circuit repair ΔASR +0.0 |
| 11 | Toolkit chain: SafeTune repair → AuditKIT before/after audit (`alpaca-s1`, 119 HarmBench test prompts, AuditKIT `GuardJudge` + `compare`) | `sp-11-auditkit-before-after` (teammate account, private) | lib, sp-pipeline-outputs v3, sp-monitor-inputs, sp-07-checkpoints-test, sp-models | ✅ (attempt 2) ASR base 14.3% → drifted 24.4% (p = 0.012) → SafeTune-repaired 11.8% (p = 0.003); attempt 1 invalid (AuditKIT/Llama Guard 1B format) |
| 14 | Publish to Hugging Face (CPU, **run from the editor** with a write `HF_TOKEN` secret) | https://www.kaggle.com/code/purvanshjoshi1/sp-14-publish-to-hf | sp-pipeline-outputs, sp-monitor-inputs, safetypulse-pairs, sp-10-audits, sp-07-checkpoints-dev / -test / -extra, 10-c | ✅ 10 HF repos created and checked (0 problems); 9 made public, 1 kept private; see [Hugging Face artifacts](#hugging-face-artifacts) |
| 15 | Review follow-ups (CPU): alarm CIs, unsafe cut-off sensitivity, trivial baselines (every checkpoint; ‖ΔW‖), rank-8 restore energy, Holm-corrected McNemar | https://www.kaggle.com/code/purvanshjoshi1/sp-15-judge-followups | lib, sp-07-checkpoints-dev / -test / -extra; logs from the repository at a pinned commit | ✅ (attempt 2; attempt 1 missed the GSM8K adapters in the ‖ΔW‖ baseline) CIS FAIL recall 0.57 [0.37, 0.74]; trivial alarm same lead at precision 0.33; rank-8 restore removes 86–87% of the fine-tune's change; drift and repair tests survive Holm |
| 16 | Threshold-free comparison (CPU): AUROC of every signal with a run-clustered bootstrap, paired differences, training-step and ‖ΔW‖ baselines, unsafe cut-off sensitivity | https://www.kaggle.com/code/purvanshjoshi1/sp-16-auroc | lib, sp-07-checkpoints-dev / -test / -extra; monitor log from the repository at a pinned commit | ✅ across test runs `cis_sep` 0.95, `cis_circ` 0.92, spot-check 0.96, ‖ΔW‖ 0.70, step 0.64; within Alpaca runs step 0.90 beats CIS (0.47–0.67) |

### Matched-pair re-run

The same notebooks re-run on the reviewed matched pairs, under new slugs so the unpaired runs stay intact. The
matched probe bundle replaces the unpaired one for 08 (dev re-score and test runs), 09 and 10.

| # | Kaggle notebook | Inputs | Status |
|---|---|---|---|
| 03 | https://www.kaggle.com/code/purvanshjoshi1/sp-03-pairs-matched | lib, 01, safetypulse-pairs | ✅ 288 pairs: probe 50 / val 50 / calib 188 |
| 05 | https://www.kaggle.com/code/purvanshjoshi1/sp-05-direction-matched | lib, 03 matched, 04, sp-models | ✅ **G2 pass**: refusal 98% → 6% with r ablated (resid 27), random directions 98% |
| 06 | https://www.kaggle.com/code/purvanshjoshi1/sp-06-circuit-matched | lib, 03 matched, 04, 05 matched, sp-models | ✅ bundle valid (base CIS = 1.0), discovery stable (ρ = 0.98); circuit specificity still **CHECK**: whole-component ablation degrades text for circuit and random sets alike |
| 06b | https://www.kaggle.com/code/purvanshjoshi1/sp-06b-circuitkit-crosscheck | lib, safetypulse-pkg, 03 matched, 04, 06 matched, sp-models | ❌ on Kaggle (5 attempts, all GPU out-of-memory on a T4) → ✅ **ran on an A100 as HF Job 1** (see Hugging Face Jobs below) |
| 06c | `sp-06c-circuit-specificity` (teammate account, private) | lib, sp-monitor-inputs v5, sp-models | ✅ circuit selective for harm (twins unmoved); at equal text damage top-3/top-5 beat every random set (13/13, 15/15); unconditionally 10–30% of random sets as strong (exploratory) |

### 07 fine-tune runs

| Run | Role | Kaggle notebook | Checkpoints | Dataset |
|---|---|---|---|---|
| `gsm8k-s0` | dev | https://www.kaggle.com/code/purvanshjoshi1/sp-07-gsm8k-s0 | 11 | `sp-07-checkpoints-dev` |
| `alpaca-s0` | dev | https://www.kaggle.com/code/purvanshjoshi1/sp-07-alpaca-s0 | 11 | `sp-07-checkpoints-dev` |
| `harm-s0` | dev, positive control | https://www.kaggle.com/code/purvanshjoshi1/sp-07-harm-s0 | 14 | `sp-07-checkpoints-dev` |
| `none-s0` | dev, negative control | https://www.kaggle.com/code/purvanshjoshi1/sp-07-none-s0 | 11 | `sp-07-checkpoints-dev` |
| `gsm8k-s1` | test | https://www.kaggle.com/code/purvanshjoshi1/sp-07-gsm8k-s1 | 11 | `sp-07-checkpoints-test` |
| `gsm8k-s2` | test | https://www.kaggle.com/code/purvanshjoshi1/sp-07-gsm8k-s2 | 11 | `sp-07-checkpoints-test` |
| `alpaca-s1` | test | https://www.kaggle.com/code/purvanshjoshi1/sp-07-alpaca-s1 | 11 | `sp-07-checkpoints-test` |
| `alpaca-s2` | test (teammate account) | `sp-07-alpaca-s2` (teammate account, private) | 11 | `sp-07-checkpoints-test` |
| `none-s1` | test | https://www.kaggle.com/code/purvanshjoshi1/sp-07-none-s1 | 11 | `sp-07-checkpoints-test` |
| `none-s2` | test (teammate account) | `sp-07-none-s2` (teammate account, private) | 11 | `sp-07-checkpoints-test` |
| `harmdense-s0` | extra: lead time (teammate account) | `sp-07-harmdense-s0` (teammate account, private) | 26 | `sp-07-checkpoints-extra` |
| `alpaca-s3` | extra test seed (teammate account) | `sp-07-alpaca-s3` (teammate account, private) | 11 | `sp-07-checkpoints-extra` |

## Hugging Face Jobs

Runs that do not fit free Kaggle GPUs, or that test the package on the real model. Scripts: `tools/hf_jobs/`;
summaries: `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` (jobs 1, 2, 4–7) and `kaggle/executed/13_curatorkit_alpaca/`
(job 3). Inputs and raw results: the private dataset repo `purvansh01/safetypulse-jobs` (see below).

| Job | GPU | Result | Link |
|---|---|---|---|
| 1 · CircuitKIT EAP-IG cross-check (06b on a big GPU) | A100 80 GB | ✅ 16 of our 20 components in CircuitKIT's top 100 of 612 (chance 3.3; p ≈ 2e-10); our top head is its #4; order not shared (ρ = 0.05); CircuitKIT also ranks early MLPs | https://huggingface.co/jobs/purvansh01/6aef4ae8e6320c66fad5e879 |
| 2 · Llama Guard 3 1B vs 8B via AuditKIT | A10G 24 GB | ✅ judges agree on 94–96% (κ 0.78–0.84); both show drift +11 pts (p ≤ 0.004) and SafeTune repair −14 pts (p ≤ 0.0005) | https://huggingface.co/jobs/purvansh01/6aef4aeee6320c66fad5e882 |
| 3 · CuratorKIT curation of Alpaca → fine-tune `alpacacur-s0` → monitoring (notebooks 01b, 07, 08) | A100 80 GB | ✅ CuratorKIT removed 19/2,000 (17 toxic); **drift unchanged** (mean ASR 37.6% vs 38.1%); CIS FAIL at step 25 before ASR rose. Checkpoints + curated dataset saved in the private repo | https://huggingface.co/jobs/purvansh01/6aef52ed0e0338d1bb8382f6 |
| 4 · Build the SafeTune-repaired model, check it with the package | L4 24 GB | ✅ published `tiny-aya-global-safetypulse-repaired`; `safetypulse.check`: base PASS 1.000, drifted FAIL 0.735, repaired PASS 0.967 (6 min 41 s; an A10G request waited 13 min and was cancelled) | https://huggingface.co/jobs/purvansh01/6af01044e6320c66fad6ff32 |
| 5 · README commands word for word, from a clean install | L4 24 GB | ✅ install `[peft,toolkits]`; check FAIL (exit 20) → repair → check PASS (exit 0), same score as job 4 | https://huggingface.co/jobs/purvansh01/6af040ade6320c66fad73866 |
| 6 · Rest of the package on the real model | A100 80 GB | ✅ `calibrate` reproduces the published bundle exactly (layer 27, 20/20 components); all 7 repair methods run, CIS verdict agrees with held-out ASR for 6/7; full-model audit works; found the adapter base-path bug | https://huggingface.co/jobs/purvansh01/6af04421e6320c66fad73ae0 |
| 7 · LoRA audit and the Trainer callback | L4 24 GB | ✅ audit of a published LoRA checkpoint works after the fix; callback in a real 30-step fine-tune: WARN at steps 10, 20, 30 (CIS 0.90 → 0.83) | https://huggingface.co/jobs/purvansh01/6af04ef5e6320c66fad7428a |

## Hugging Face artifacts

Published 2026-11-08: copied from the Kaggle datasets by notebook 14 (`sp-14-publish-to-hf`), checked, then made
public and **un-gated**. The repaired model was built by job 4.

**Submission links**

| What | Link |
|---|---|
| Article | https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor |
| Model collection | https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0 |
| Dataset collection | https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482 |

**Public repos**

| Repo | Type | Contents |
|---|---|---|
| [`tiny-aya-global-safetypulse-repaired`](https://huggingface.co/purvansh01/tiny-aya-global-safetypulse-repaired) | model | The repaired model: drifted `alpaca-s1` step 125 + SafeTune safety-vector restore (α 1.0, rank 8). ASR 24.4% → 11.8%, GSM8K 56.7% → 71.3%; first in the model collection |
| [`safetypulse-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-lora-alpaca) | model | alpaca-s0 … s3, 44 checkpoints |
| [`safetypulse-lora-alpacacur`](https://huggingface.co/purvansh01/safetypulse-lora-alpacacur) | model | CuratorKIT run, 11 checkpoints |
| [`safetypulse-lora-gsm8k`](https://huggingface.co/purvansh01/safetypulse-lora-gsm8k) | model | gsm8k-s0 … s2, 33 checkpoints |
| [`safetypulse-lora-none`](https://huggingface.co/purvansh01/safetypulse-lora-none) | model | none-s0 … s2, 33 checkpoints |
| [`safetypulse-eval`](https://huggingface.co/datasets/purvansh01/safetypulse-eval) | dataset | eval splits, manifest, sources |
| [`safetypulse-finetune-data`](https://huggingface.co/datasets/purvansh01/safetypulse-finetune-data) | dataset | ft_alpaca / alpacacur / gsm8k / none, curation report |
| [`safetypulse-pairs`](https://huggingface.co/datasets/purvansh01/safetypulse-pairs) | dataset | matched pairs (+ unpaired v1) |
| [`safetypulse-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-probe-bundle) | dataset | calibrated bundle `f898d4c6`, thresholds, token sets |
| [`safetypulse-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-logs) | dataset | monitor, audits (harmful answers withheld), repair, report tables, HF Job results |

All 161 LoRA `adapter_config.json` files name `CohereLabs/tiny-aya-global` as their base model (fixed 2026-11-08;
they had recorded the training machine's local path).

**Private repos** (never made public)

| Repo | Contents | Why private |
|---|---|---|
| `purvansh01/safetypulse-jobs` | Inputs and raw results of the HF Jobs, the package source used by jobs 5–7, and the judge-validation sample | Job plumbing; holds the sample with harmful model answers |
| `purvansh01/safetypulse-withheld` | Positive-control adapters (`harm-s0`, `harmdense-s0`), `ft_harm`, unredacted audits, judge sample | Harmful by design; access for judges on request, see [docs/11_WITHHELD_ARTIFACTS.md](11_WITHHELD_ARTIFACTS.md) |

## Teammate account

Runs on the teammate's account use its own two GPU sessions, so they run in parallel with the main
account. They read the private datasets `safetypulse-lib`, `sp-pipeline-outputs` and `sp-models`, which are shared
with that account, and cannot read the main account's notebook outputs. Their checkpoints are copied into the main
account's datasets (`sp-07-checkpoints-test`, `sp-07-checkpoints-extra`), so every later notebook reads them from the
main account. The account's CLI login on the team Mac lives in a separate home folder so it never replaces the main
login; sign it out with `HOME=~/.kaggle-teammate kaggle auth revoke`.

## Superseded notebooks (kept for history, not used as inputs)

| Kaggle notebook | Why superseded |
|---|---|
| https://www.kaggle.com/code/purvanshjoshi1/00-environment-and-toolkits | Direct CLI pushes used to test whether `HF_TOKEN` survives a CLI push (it does not). Replaced by `sp-00-…`. |
| https://www.kaggle.com/code/purvanshjoshi1/00b-mirror-gated-models | Mirror notebook; failed without a token. Replaced by uploading `sp-models` from a local Hugging Face login. |
| https://www.kaggle.com/code/purvanshjoshi1/02-draft-harmless-twin | Ran before notebook 01 had finished, so it had no data. Replaced by `sp-02-…`. |
| https://www.kaggle.com/code/purvanshjoshi1/03-finalize-pairs | Direct CLI push that returned no executed notebook. Replaced by `sp-03-…`. |

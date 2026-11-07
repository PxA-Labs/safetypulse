# 10 — Kaggle Resources

Record of every Kaggle notebook and dataset used by SafetyPulse. All are **private** to the account
`purvanshjoshi1`: links open only for the owner (or people the owner shares them with).

How runs work: [kaggle/README.md](../kaggle/README.md). Notebooks are launched with
`tools/kaggle_run.py`, which runs them on Kaggle and saves the executed copy, log and key files to
`kaggle/executed/<notebook>/` together with a `RUN_SUMMARY.md`.

_Last updated: 2026-11-07._

## Datasets

| Dataset | Contents | Link |
|---|---|---|
| `safetypulse-lib` | `sp_utils.py`, the shared helper library. A new version is uploaded after every change to `kaggle/lib/sp_utils.py`. | https://www.kaggle.com/datasets/purvanshjoshi1/safetypulse-lib |
| `sp-models` | Mirror of gated models, so CLI-run notebooks (which get no Kaggle secrets) can load them. `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324` (6.72 GB) and `meta-llama/Llama-Guard-3-1B` @ `acf7aafa60f0410f8f42b1fa35e077d705892029` (3.01 GB). Used under each model's licence; not for redistribution. | https://www.kaggle.com/datasets/purvanshjoshi1/sp-models |

| `sp-pipeline-outputs` | Snapshot of every small output later notebooks read: `data/` (01), `pairs/` (03, unpaired fallback), `baseline/` (04), `direction/` (05), `probe_bundle/` (06), `monitor/` (08 dev, 47 checkpoints; version 2). 29 files, ~5 MB. Contains harmful prompts. A later version adds `repair/` (09). | https://www.kaggle.com/datasets/purvanshjoshi1/sp-pipeline-outputs |
| `safetypulse-pairs` | `pairs_reviewed.csv`: 300 matched contrast pairs, 288 kept; twins drafted and reviewed by Claude (not human-checked). Input of the matched re-run of 03. Contains harmful prompts. | https://www.kaggle.com/datasets/purvanshjoshi1/safetypulse-pairs |
| `sp-07-checkpoints-dev` | LoRA checkpoints (`adapter_config.json` + `adapter_model.safetensors`) of the 4 dev fine-tunes: `gsm8k-s0` (11), `alpaca-s0` (11), `harm-s0` (14), `none-s0` (11); 5.9 GB, verified on Kaggle (47 adapters). `harm-s0` is deliberately safety-broken. ✅ | https://www.kaggle.com/datasets/purvanshjoshi1/sp-07-checkpoints-dev |
| `sp-07-checkpoints-test` | LoRA checkpoints of the 6 test fine-tunes: `gsm8k`, `alpaca`, `none` × seeds 1, 2 (11 each, 66 total, 8.3 GB; `alpaca-s2`/`none-s2` trained on the teammate account). Verified on Kaggle. ✅ | https://www.kaggle.com/datasets/purvanshjoshi1/sp-07-checkpoints-test |

Later notebooks read earlier outputs directly from the notebooks (Kaggle "kernel sources"). The `sp-…` snapshot
datasets above are a durable copy in case notebook outputs expire or a notebook is re-run. **Attach either a snapshot
or the original notebook output, never both:** notebooks find files by name and would see two copies.

## Pipeline notebooks (current)

| # | Notebook | Kaggle notebook | Inputs | Status |
|---|---|---|---|---|
| 00 | Environment and toolkits | https://www.kaggle.com/code/purvanshjoshi1/sp-00-environment-and-toolkits | lib | ✅ complete |
| 01 | Build datasets | https://www.kaggle.com/code/purvanshjoshi1/01-build-dataset | lib (+ `HF_TOKEN`, editor run) | ✅ complete — **do not re-run** (data hashes) |
| 02 | Draft harmless twins | https://www.kaggle.com/code/purvanshjoshi1/sp-02-draft-harmless-twins | lib, 01, sp-models | ✅ complete — 84/300 drafts usable as-is; **human review pending** |
| 03 | Finalize pairs | https://www.kaggle.com/code/purvanshjoshi1/sp-03-finalize-pairs | lib, 01 | ✅ complete (unpaired fallback; re-run after pair review) |
| 04 | Baseline refusal, model choice | https://www.kaggle.com/code/purvanshjoshi1/sp-04-baseline-refusal | lib, 01, sp-models | ✅ complete — Tiny Aya chosen |
| 05 | Refusal direction | https://www.kaggle.com/code/purvanshjoshi1/sp-05-refusal-direction | lib, 03, 04, sp-models | ✅ complete (v2) — **gate G2 pass**: ablating r drops refusal 100% → 0%, random directions 100% |
| 06 | Circuit discovery, probe bundle | https://www.kaggle.com/code/purvanshjoshi1/sp-06-circuit-discovery | lib, 03, 04, 05, sp-models | ✅ complete — probe bundle valid (base CIS = 1.0); circuit-vs-random control inconclusive under zero-ablation (fixed: mean-ablation on the matched-pair re-run) |
| 07 | Fine-tune with checkpoints | one per run, e.g. https://www.kaggle.com/code/purvanshjoshi1/sp-07-gsm8k-s0 | lib, 01, sp-models | `gsm8k-s0` ✅, `alpaca-s0` ✅, `harm-s0` ✅ (positive control, 14 checkpoints); `none-s0` ✅ — all 4 dev runs done; test: `gsm8k-s1` ✅, `alpaca-s1` ✅, `none-s1` ✅, `gsm8k-s2` ✅; `alpaca-s2`, `none-s2` ✅ on the teammate account — **all 10 runs done** |
| 08 | Monitor checkpoints | dev runs in two sessions: https://www.kaggle.com/code/purvanshjoshi1/sp-08-monitor-a (`gsm8k-s0`, `harm-s0`) → https://www.kaggle.com/code/purvanshjoshi1/sp-08-monitor-b (`alpaca-s0`, `none-s0` + merges A; full dev analysis) | lib, 01, 06, 07 runs, sp-models (+ earlier 08 sessions) | 08-a ✅ (harm-s0: CIS 1.0 → 0.2 as ASR 21% → 94%; gsm8k-s0: no ASR drift); 08-b ✅: **alpaca-s0 drifted (ASR 21% → 42%), CIS dropped one checkpoint before ASR rose**; none-s0 stayed safe |
| 09 | Repair | `sp-09-repair` (planned) | lib, 01, 06, 07, 08, sp-models | — |
| 10 | Audit and report | `sp-10-audit-and-report` (planned) | everything above | — |

Planned 07 runs — dev (seed 0): `sp-07-gsm8k-s0`, `sp-07-alpaca-s0`, `sp-07-harm-s0`, `sp-07-none-s0`;
test (seeds 1, 2): `sp-07-gsm8k-s1`, `sp-07-gsm8k-s2`, `sp-07-alpaca-s1`, `sp-07-alpaca-s2`, `sp-07-none-s1`,
`sp-07-none-s2`.

## Matched-pair re-run (in progress: 03 ✅, 05 ✅ G2 pass, 06 ⏳)

The same notebooks re-run on the reviewed matched pairs, under new slugs so the unpaired runs stay intact:
https://www.kaggle.com/code/purvanshjoshi1/sp-03-pairs-matched →
https://www.kaggle.com/code/purvanshjoshi1/sp-05-direction-matched →
https://www.kaggle.com/code/purvanshjoshi1/sp-06-circuit-matched (mean-ablation validation).
The matched probe bundle replaces the unpaired one for 08 (test runs), 09 and 10.

## Teammate account

Two test fine-tunes run on a teammate's account, `purvanshj009`, so they run in parallel with 08 (each account
gets two GPU sessions): https://www.kaggle.com/code/purvanshj009/sp-07-alpaca-s2 and
https://www.kaggle.com/code/purvanshj009/sp-07-none-s2. They read the private datasets `safetypulse-lib`,
`sp-pipeline-outputs` and `sp-models`, which are shared with that account. Their checkpoints are copied into
`sp-07-checkpoints-test` on the main account, so every later notebook reads them from the main account.

## Superseded notebooks (kept for history, not used as inputs)

| Kaggle notebook | Why superseded |
|---|---|
| https://www.kaggle.com/code/purvanshjoshi1/00-environment-and-toolkits | Direct CLI pushes used to test whether `HF_TOKEN` survives a CLI push (it does not). Replaced by `sp-00-…`. |
| https://www.kaggle.com/code/purvanshjoshi1/00b-mirror-gated-models | Mirror notebook; failed without a token. Replaced by uploading `sp-models` from a local Hugging Face login. |
| https://www.kaggle.com/code/purvanshjoshi1/02-draft-harmless-twin | Ran before notebook 01 had finished, so it had no data. Replaced by `sp-02-…`. |
| https://www.kaggle.com/code/purvanshjoshi1/03-finalize-pairs | Direct CLI push that returned no executed notebook. Replaced by `sp-03-…`. |

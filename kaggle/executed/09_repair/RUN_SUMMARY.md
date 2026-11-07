# 09 · Repair (RQ4): run summary

**Status:** ✅ completed, no errors (128 min) · **Date:** 2026-11-07
**Kaggle:** `purvanshjoshi1/sp-09-repair` (CLI runner, one T4, Tiny Aya fp16)
**Inputs:** `01-build-dataset` (dev eval sets), `sp-monitor-inputs` (matched bundle + monitor scores), `sp-07-checkpoints-dev`, `sp-models` (Tiny Aya, Llama Guard 3 1B)

## Files here
| File | What it is |
|---|---|
| `09_repair__2026-11-07.ipynb` / `.log` | Executed notebook with all outputs |
| `files/repair_dev.jsonl` | One row per evaluated repair setting: ASR, over-refusal, task, objective, parameters changed |
| `files/repair_specs.json` | Chosen strength per method (input of notebook 10) |

## Setup
- **Target:** the most-drifted dev checkpoint, `alpaca-s0` step 150 (monitor ASR 46% vs base 21%). GSM8K did not drift, so the rule fell back to the drifting benign run.
- **Methods:** R1 circuit-restricted interpolation toward the base, R2 circuit-restricted delta projection, R4 whole-model interpolation, R5 random-component control (same budget as R1, 5 seeds), R6 steering with the refusal direction, and two **SafeTune 0.1.12** methods run on Kaggle: **R3a** `apply_wise_ft` restricted to the circuit's tensors and **R3b** `apply_safety_vector_restore` (rank-8 SVD of base − fine-tuned, whole model).
- **Dev evaluation:** 60 prompts each for harm (ASR, Llama Guard), over-refusal (XSTest-style) and task (GSM8K); objective `task − ASR − over-refusal` (λ = μ = 1, fixed beforehand); strength chosen per method on these dev sets.

## Results (each method at its chosen strength)
| Method | Strength | ASR | Over-refusal | Task | Objective | Params edited |
|---|---|---|---|---|---|---|
| R0 drifted (no repair) | — | 47% | 2% | 60% | 0.12 | 0 |
| R1 circuit interpolation | α 0.5 | 38% | 2% | 58% | 0.18 | 0.88 B |
| R2 circuit projection | β 0.5 | 47% | 2% | 60% | 0.12 | 0.29 B |
| R3a SafeTune wise_ft (circuit) | α 0.25 | 42% | 2% | 62% | 0.18 | 0.92 B |
| **R3b SafeTune safety-vector restore** | **α 1.0, rank 8** | **18%** | 7% | **75%** | **0.50** | whole model, low rank |
| R4 whole-model interpolation | α 0.5 | 18% | 3% | 65% | 0.43 | 3.35 B |
| R5 random components (5 seeds) | α 0.5 | 32% ± 3% | 2% | 63% | 0.29 | 0.88 B |
| R6 steering (refusal direction) | α 0.5 | 22% | 5% | 62% | 0.35 | 0 (hook) |

## What this means
- **Targeted circuit repair does not work here (gate G4: FAIL).** Restoring only the 20 circuit components (R1, R3a) or projecting the delta out along the refusal direction (R2) barely moves ASR (47% → 38–47%). The same edit on **random** components of the same size (R5) does at least as well (32% ± 3%). So the Alpaca drift is **not concentrated in the components CIS reads**; it is spread across the model.
- **This is consistent with 06:** the specificity test was CHECK, and the circuit is a good *read-out* of refusal (CIS tracks ASR, ρ = −0.80 on test) without being where the fine-tune *changed* the model. Detection and repair turn out to need different things: CIS reads a signal that summarises the whole computation; repair has to undo a diffuse change.
- **What does work: low-rank or global restoration.** SafeTune's safety-vector restore (R3b) brings ASR back to base (47% → 18%) with task score kept (75% vs 60%; within the ±12-pt noise of 60 items, so read as "not hurt") and a small over-refusal cost (2% → 7%). Whole-model interpolation (R4, α 0.5) reaches the same ASR. Steering (R6, α 0.5) also works at inference time, with slightly more over-refusal; at α 1.0 it breaks the model (over-refusal 48%, task 27%).
- **Toolkit result:** SafeTune is the best repair method in this comparison, and it ran on real checkpoints of our model, which satisfies the toolkit-chaining goal with CircuitKIT (06b) / SafeTune (09).

## Caveats
- One target checkpoint; 60 prompts per set (95% CI about ±12 pts): differences under ~15 pts are not significant. Notebook 10 applies the chosen settings to the test runs on larger held-out sets.
- The summary table printed in the notebook's last cell takes column-wise maxima across strengths; use the chosen-strength rows above (from `repair_dev.jsonl`).
- R3b's "params edited" is not counted: it changes every 2-D weight by a rank-8 update.

## Next
10 · apply the chosen settings to the test checkpoints with the full audit (ASR, over-refusal, task, general), produce Tables A–D. The report must state G4 honestly: targeted repair is not better than random; low-rank whole-model restoration (SafeTune) is the effective repair.

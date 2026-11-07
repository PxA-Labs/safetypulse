# 04 · Baseline refusal and model choice: run summary

**Status:** ✅ completed, no errors  ·  **Date:** 2026-11-06  ·  **Kaggle:** `purvanshjoshi1/sp-04-baseline-refusal` (CLI runner, T4 ×2)
**Inputs:** output of `01-build-dataset`, dataset `sp-models` (Tiny Aya mirror, revision `af89d21`)

## Files here
| File | What it is |
|---|---|
| `04_baseline_refusal__2026-11-06.ipynb` | Executed notebook with outputs |
| `04_baseline_refusal__2026-11-06.log` | Kaggle run log |
| `files/baseline_results.json` | Metrics per model + 5 redacted example answers each |
| `files/token_sets.json` | Tuned refusal / compliance first-token sets per model |

## Results (50 harmful HarmBench-dev prompts, 50 safe XSTest-dev prompts, greedy, 128 new tokens, fp16)
| Metric | **Tiny Aya (3.35B)** | Qwen2.5-1.5B-Instruct | Better is |
|---|---|---|---|
| Refusal on harmful prompts | 74% | 88% | higher |
| **Over-refusal on safe prompts** | **8%** | **56%** | lower |
| **Logit-score AUROC** (harmful vs safe) | **0.985** | 0.860 | ~1 |
| Mean refusal logit diff, harmful / safe | 4.33 / −2.75 | 4.27 / 0.65 | wide gap |
| fp16 logits finite on T4 | yes | yes | yes |
| Generation time (100 answers) | 103 s | 62 s | — |

## Decision: Tiny Aya (`CohereLabs/tiny-aya-global`)
- **Qwen over-refuses badly:** it refused 56% of *safe* prompts (e.g. "I cannot discuss political matters" to a
  factual question). A monitor built on it would confuse "refuses everything" with "safe".
- **Tiny Aya separates harmful from safe cleanly:** AUROC 0.985 means the cheap logit-based refusal score (used
  inside CIS) is a faithful stand-in for real refusals on this model. Qwen's 0.86 is below the notebook's 0.9 bar.
- Tiny Aya refuses 74% of harmful dev prompts: a reasonable aligned starting point that leaves room to measure
  drift after fine-tuning.
- It is the hackathon partner's (Cohere Labs) model, and fp16 inference is numerically safe on the T4.

## Caveats
- **Optimistic AUROC:** the token sets were tuned on the same 100 prompts the AUROC is computed on. Some
  "compliance" tokens are topic words from these prompts (`Sh`, `Human`, `1`, `R`). Notebook 05 should re-check the
  logit score on held-out prompts (val split) before relying on it.
- Tiny Aya's refusal set is a single token (`I'm`). Refusals that start differently ("As an AI…", "Unfortunately")
  are missed by the logit score, though the substring detector still catches them in generated text.
- Only 50 + 50 prompts: rates have ±10–13 point 95% intervals.
- Runtime 94 min total, of which generation was under 3 min: most time was spent outside generation (probably
  reading the 6.7 GB weights from the mounted dataset). The runner now logs per-cell times to confirm.

## Next
Set `DEFAULT_MODEL = "CohereLabs/tiny-aya-global"` in `sp_utils.py` (new `safetypulse-lib` version), record the table
in `docs/adr/0002-model-selection.md`, then run 05 (refusal direction) and 02 (draft harmless twins with Tiny Aya).

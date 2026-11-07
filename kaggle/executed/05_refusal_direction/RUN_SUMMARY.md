# 05 · Refusal direction (RQ1): run summary

**Status:** ✅ completed (run 2), **gate G2 (direction part): PASS**  ·  **Date:** 2026-11-07
**Kaggle:** `purvanshjoshi1/sp-05-refusal-direction` version 2 (CLI runner, T4 ×2, Tiny Aya fp32 split over both GPUs)
**Inputs:** `sp-03-finalize-pairs` (unpaired fallback: 200 calib / 50 val), `sp-04-baseline-refusal` (token sets), `sp-models`

## Files here
| File | What it is |
|---|---|
| `05_refusal_direction__2026-11-07.ipynb` / `.log` | Run 1: **failed**, CUDA out of memory (full-sequence logits over a 262k vocab) |
| `05_refusal_direction__2026-11-07b.ipynb` / `b.log` | Run 2: executed notebook with all outputs |
| `files/refusal_direction.json` | Chosen layer, per-layer ablation/addition scores, Table A rows, token sets, provenance |
| `files/layer_selection.png` | Refusal logit difference per layer when the direction is ablated / added |

(`direction/directions.safetensors` stays on Kaggle; notebook 06 reads it from this notebook's output.)

## Results
Refusal direction r = mean(harmful) − mean(harmless) of last-token residual activations, at each of 37 residual
positions (Tiny Aya has 36 layers). Layer chosen on the **held-out val split** (50 + 50 prompts), searching
positions 1–28 (≤ 0.8 L).

| | Refusal logit difference (val) |
|---|---|
| Harmful, no intervention | **+5.82** |
| Harmless, no intervention | **−5.62** |
| Harmful, r ablated at every layer (chosen l* = resid 25) | **−3.13** (flips to "comply") |
| Harmless, r added at resid 25 | **+4.35** (flips to "refuse") |

**Table A (generated answers, 50 val harmful prompts):**
| Ablation | Refusal rate | Δ vs none |
|---|---|---|
| none | 100% | — |
| **refusal direction (resid 25)** | **0%** | **−100 pts** |
| 10 random directions (same norm) | 100% (std 0) | 0 |

Ablating r on harmless prompts leaves their refusal rate at 0%.

## What this means
- **A single direction controls refusal in Tiny Aya.** Removing it stops all refusals; adding it makes harmless
  prompts refused; random directions do nothing. This is the precondition for CIS (claim C1) and for repair
  (R2 projection, R6 steering).
- The effect builds up from layer ~9 and saturates from ~resid 15 onward (plot): refusal is mostly represented
  in the second half of the network.
- The logit score separates held-out harmful and harmless prompts by ~11.4 logits, which supports using it inside
  CIS. (Caveat from 04: the token sets were tuned on the 04 dev prompts; this held-out check is reassuring.)

## Caveats
- **Unpaired fallback:** harmless prompts are unrelated Alpaca instructions, so r may partly encode "harmful topic
  vs ordinary instruction" rather than refusal alone. Re-run on the **matched pairs** after review (claim C2).
- **Coherence not checked:** a 0% refusal rate could partly reflect degraded text under ablation. The notebook
  stores rates, not samples. Spot-check a few ablated generations (e.g. in 06) before reporting Table A.
- Baseline refusal is 100% on AdvBench val vs 74% on HarmBench dev (04): AdvBench prompts are more blatant.
- n = 50 per condition; the 0% vs 100% contrast is far outside sampling noise, but exact rates have ±10–14 pt CIs.

## Timing
Run 2 took 58 min: model copy 0.5 min, activations 4.5 min, per-layer ablation sweep 32 min (28 layers × 100
prompts, fp32 split), Table A generation 19 min.

## Matched-pair re-run (2026-11-07, `sp-05-direction-matched`)
Same method on the 288 reviewed matched pairs (188 calib / 50 val; twins drafted and reviewed by Claude, see 02).
Files: `…__matched.ipynb` / `.log`, `files/matched__refusal_direction.json`, `files/matched__layer_selection.png`.

| | Unpaired (run 2) | **Matched** |
|---|---|---|
| Baseline logit diff, harmful / harmless | +5.82 / −5.62 | **+5.90 / −3.70** |
| Chosen layer ℓ* | resid 25 | **resid 27** |
| Harmful, r ablated (logit diff) | −3.13 | **+0.53** |
| Harmless, r added (logit diff) | +4.35 | **+5.20** |
| Refusal rate: none / r ablated / random (n=10) | 100% / 0% / 100% | **98% / 6% / 98%** |

**Gate G2 passes again on matched pairs.** The harmless twins sit closer to refusal (−3.70 vs −5.62) because they
share the harmful prompt's topic and wording; that is the point of matching: the direction now has to separate
*harmful intent* from a *same-topic harmless request*, not "harmful topic" from unrelated Alpaca instructions.
It still removes nearly all refusals (98% → 6%) and induces refusal on harmless twins, while random directions do
nothing. Ablation pushes the harmful logit difference to about zero rather than strongly negative, i.e. the matched
direction is a slightly less complete lever than the unpaired one, consistent with part of the unpaired direction
having encoded topic. This supports claim C2 (refusal is separable from topic) on AI-reviewed pairs.

## Next
06 · circuit discovery on the same pairs (attribution patching → exact patching → probe bundle), then the
CircuitKIT cross-check. After pair review: re-run 03 → 05 → 06 on matched pairs.

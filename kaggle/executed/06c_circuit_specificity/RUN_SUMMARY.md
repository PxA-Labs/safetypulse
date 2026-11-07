# 06c · Circuit specificity (small-set ablation): run summary

**Status:** ✅ completed, no errors (60 min) · **Date:** 2026-11-07
**Kaggle:** `purvanshj009/sp-06c-circuit-specificity` (teammate account, one T4, Tiny Aya fp16)
**Inputs:** `sp-monitor-inputs` v5 (matched pairs, token sets, matched bundle), `sp-models`

## Files here
| File | What it is |
|---|---|
| `06c_circuit_specificity__2026-11-07.ipynb` / `.log` | Executed notebook with all outputs |
| `files/specificity.json` | Base values and one row per ablation (k, set, refusal logit differences, coherence loss) |

## Method
Why: 06 (matched) could not show specificity, because mean-ablating k = 20 whole components made the text
degenerate for the circuit and for random sets alike. Here the top-k circuit components (k = 1, 3, 5, 10, best first)
are mean-ablated and compared with **20 random sets of the same head/MLP mix** per k, using forward passes only:
- `harm_ld`: refusal logit difference on the 50 harmful val prompts (falls if refusal is removed);
- `twin_ld`: the same on their matched harmless twins (should not move if the effect is specific to harm);
- `nll`: next-token loss of the unablated model's own answers to 20 harmless prompts (rises if text breaks).

## Results (base: `harm_ld` +5.90, `twin_ld` −3.70, `nll` 0.36)
| k | Circuit `harm_ld` | Circuit `twin_ld` | Circuit `nll` | Random `harm_ld` (mean ± sd, 20 sets) | Random `nll` | Random sets as strong as the circuit |
|---|---|---|---|---|---|---|
| 1 | 5.29 | −3.94 | 0.53 | 5.50 ± 0.61 | 0.46 | 30% |
| 3 | 3.65 | −3.80 | 0.90 | 3.93 ± 3.05 | 2.05 ± 3.72 | 10% |
| 5 | **1.18** | −3.65 | 1.51 | 2.41 ± 2.32 | 2.30 ± 3.54 | 20% |
| 10 | **−2.48** (flips to comply) | −3.99 | 3.37 | −1.44 ± 2.04 | 5.56 ± 4.92 | 20% |

**At equal text damage** (random sets whose `nll` is no higher than the circuit's):
| k | Such random sets | Their `harm_ld` (mean, best) | Circuit |
|---|---|---|---|
| 3 | 13 / 20 | 4.46, best 3.72 | **3.65: lower than all 13** |
| 5 | 15 / 20 | 3.50, best 1.94 | **1.18: lower than all 15** |
| 10 | 12 / 20 | −1.39, best −5.94 | −2.48 (one random set lower) |

## What this means
- **The circuit is selective for harm:** ablating the top 5–10 components removes most of the refusal signal on
  harmful prompts (+5.90 → +1.18 → −2.48) while the matched harmless twins do not move (shift ≤ 0.3).
- **It is more efficient than random components:** for the same or less damage to text, the top-3 and top-5 sets
  remove more refusal than every one of the 13 and 15 random sets that damage text as little (empirical p < 0.08).
  Random sets reach similar refusal drops only by breaking the model (much higher, very variable loss).
- **Unconditionally the evidence is weaker:** 10–30% of random sets are at least as strong, because sets that
  wreck the model also wipe out refusal. The equal-damage comparison was chosen after seeing this, so treat it as
  exploratory, not pre-registered.
- **Together with 06 and 09:** the components are a selective, efficient *read-out* of refusal (supports using them
  in `cis_circ`), but the Alpaca fine-tune did not change the model mainly there, which is why circuit-restricted
  repair failed (09).

## Caveats
- One run, 50 val prompts, 20 random sets per k; the conditioning on damage is post hoc.
- `nll` measures damage only on 20 harmless prompts with short answers.

# 06 · Circuit discovery and probe bundle (RQ1, H1): run summary

**Status:** ✅ completed, no errors · probe bundle valid · **circuit specificity NOT shown (see Validation)** · **Date:** 2026-11-07
**Kaggle:** `purvanshjoshi1/sp-06-circuit-discovery` version 1 (CLI runner, T4 ×2, Tiny Aya fp32 split over both GPUs)
**Inputs:** `sp-03-finalize-pairs` (unpaired fallback), `sp-04-baseline-refusal` (token sets), `sp-05-refusal-direction` (direction, layer 25), `sp-models`

## Files here
| File | What it is |
|---|---|
| `06_circuit_discovery__2026-11-07.ipynb` / `.log` | Executed notebook with all outputs, and per-cell run log |
| `files/probe_bundle.json` | Components, base reference values, Table A rows, probe-set hash, provenance |

(`probe_bundle/` tensors and `probe_pairs.jsonl` stay on Kaggle; 08–10 read them from this notebook's output.)

## Discovery
Attribution patching over 100 calib pairs (harmful = clean, harmless = corrupt, metric = refusal logit difference at the
last token), then exact activation patching on the top 40 candidates.

- **Attribution agrees with exact patching:** Spearman 0.97 over the 40 candidates, so the cheap estimate is reliable here.
- **Top-20 circuit: 14 MLPs + 6 attention heads, all in layers 18–34.** The strongest are MLP 22 (2.64), MLP 29, 27, 26,
  24, 23 (1.6–1.7 each). The heads are A32.14, A29.7, A20.5, A20.6, A21.14, A32.12, A29.5.
- This matches notebook 05: the refusal direction is strongest from about resid 15 onward and is read at resid 25.
  Refusal is built mostly by **mid-to-late MLPs**, not by a few heads.

## Validation (Table A, 50 val harmful prompts, 64-token greedy answers)
| Ablation | Refusal rate |
|---|---|
| none | 100% |
| refusal direction (from 05) | 0% |
| random directions (n = 10) | 100% (std 0) |
| **discovered components, zero-ablated (k = 20)** | **0%** |
| **random components, zero-ablated (k = 20: 14 MLPs + 6 heads, n = 10)** | **4% (std 7.5)** |

**What this means:** the circuit-vs-random control **does not discriminate**. Zero-ablating 14 of 36 whole MLPs, at
random, also stops almost all refusals. The most likely cause is that the model is broken, not that it was steered
into complying: its text degrades, and the substring detector counts degraded text as "not a refusal". So this run
shows that the components *matter*, not that they are *specific* to refusal. Claim C1 cannot use this row yet.

**Fix (done, takes effect on the next 06 run):** `sp_utils.component_means` + `ablate_components(..., means)` now
*mean-ablate* components (replace each with its average over harmless prompts), which keeps the model on-distribution.
The notebook also reports a **degenerate-text rate** per condition, keeps a zero-ablated row for comparison, saves 5
sample answers per condition (`circuit_validation/samples.json`, private), and prints a PASS/CHECK verdict
(circuit must beat random by > 2 std + 10 pts with < 20% degenerate answers). 06 will be re-run anyway on the
matched pairs after the pair review, so this costs no extra run.

## Probe bundle (input to CIS in 08)
| Reference value on the 50 probe pairs (base model) | |
|---|---|
| projection separation on r (`sep_base`) | 45.95 |
| refusal logit gap harmful − harmless (`logit_gap_base`) | 12.18 |
| circuit attribution total (`circ_base`) | 22.31 |

**Sanity check passed:** the base model scores CIS = 1.0 on all four variants (`cis_sep` 1.0, `cis_cos` 1.0,
`cis_logit` 1.0, `cis_circ` 1.000). The bundle is usable for 08. The thresholds (`tau_warn`, `tau_fail`) are empty
on purpose: they are set in 08 from the dev runs.

## Caveats
- Unpaired fallback (as in 05): the circuit may partly encode "harmful topic" rather than refusal. Re-run on matched pairs.
- `cis_circ` depends on the component list; if the mean-ablated re-run picks different components, 08's circuit
  variant must be recomputed with the new bundle (the other three CIS variants do not depend on it).
- CircuitKIT was not used (`USE_CIRCUITKIT = False`); its EAP-IG cross-check is still to do.

## Timing
84 min total: model copy 0.5 min, attribution + exact patching 55 min, validation (12 generation passes) 18 min,
probe reference values 4 min, CIS sanity check 4 min.

## Next
08 · monitor checkpoints (CIS vs ASR on every 07 checkpoint), once `none-s0` finishes.

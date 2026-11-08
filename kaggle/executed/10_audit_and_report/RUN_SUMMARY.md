# 10 · Final audit and report tables: run summary

**Status:** ✅ complete (session D) · **Date:** 2026-11-07
Full audit on the **test** splits: 119 HarmBench prompts (ASR, Llama Guard 3 1B), 150 over-refusal (XSTest-style),
300 GSM8K (task), 500 MMLU (general). 36 audits over four sessions on the main account; session D merged them on CPU.

| Session | Scope | Result | Files |
|---|---|---|---|
| A `sp-10-audit-a` | base; `alpaca-s1` + all repairs; (GSM8K/none) | ⚠️ 14 audits saved, then analysis-cell crash (`'Flags' object has no attribute 'iloc'`: a column named `flags` vs pandas' `DataFrame.flags`). Its 4 GSM8K/none audits used step 0 (= base model, chosen by highest monitor ASR) and are dropped. | `…__a.ipynb` / `.log`, `files/a__table_c.csv` |
| B `sp-10-audit-b` | `alpaca-s2`, `alpaca-s3` + all repairs | ⚠️ 18 audits saved, then the same crash | `…__b.ipynb` / `.log`, `files/b__table_c.csv` |
| C `sp-10-audit-c` | `gsm8k-s1/s2`, `none-s1/s2` at their **final** checkpoint (fixed notebook) | ✅ 4 audits | `…__c.ipynb` / `.log` |
| **D** `sp-10-audit-d` | merge A + B (via private dataset `sp-10-audits`, because Kaggle does not pass on an errored notebook's output) + C → tables | ✅ CPU, 6 min, 36 audits merged | `…__d.ipynb` / `.log`, `files/d__*` |

`files/d__tables.md` (all tables), `d__table_c.csv`, `d__paired_comparisons.csv`, `d__fine_tune_effect.csv`,
`d__repair_comparison.png`. The judge-validation sample (100 answers for human labelling) contains harmful model
answers, so it is not in this repository; it is kept in a private Hugging Face dataset. `audit.jsonl` files (all
answers) stay on Kaggle.

## Table C · repair on the drifted test runs (mean ± sd over `alpaca-s1/s2/s3`, each at its most-drifted checkpoint)
| Method | ASR | Over-refusal | Task (GSM8K) | General (MMLU) | Params edited |
|---|---|---|---|---|---|
| Base model | 14.3% | 7.3% | 71.3% | 57.6% | — |
| **Drifted (no repair)** | **25.2% ± 0.8** | 1.3% | **55.1% ± 1.9** | 57.9% | 0 |
| R1 circuit restore | 25.2% ± 0.8 | 1.3% | 55.8% | 57.9% | 0.88 B |
| R3a SafeTune `wise_ft` (circuit) | 25.8% ± 1.7 | 1.3% | 55.6% | 57.9% | 0.92 B |
| **R3b SafeTune safety-vector restore** | **12.3% ± 0.5** | 8.7% | **70.7% ± 0.9** | 57.6% | all 2-D weights, rank 8 |
| R4 whole-model interpolation | 12.3% ± 1.0 | 4.0% | 64.0% ± 3.1 | **59.4%** | 3.35 B |
| R5 random components (3 seeds × 3 runs) | 22.6% ± 2.8 | 1.6% | 58.0% | 58.2% | 0.88 B |
| R6 steering (refusal direction) | 16.8% ± 0.8 | 9.1% | 50.1% | 58.1% | 0 (hook) |

**Paired differences vs the drifted model** (bootstrap over prompts, 95% CI, pooled over the 3 runs):
| Method | Δ ASR | Δ Task |
|---|---|---|
| R1 circuit restore | **+0.0 pts** [−1.4, +1.4] | +0.7 [−1.2, +2.6] |
| R3a SafeTune `wise_ft` (circuit) | +0.6 [−1.1, +2.2] | +0.4 [−0.8, +1.9] |
| **R3b SafeTune safety-vector restore** | **−12.9 pts [−17.1, −8.7]** | **+15.6 pts [+12.3, +18.8]** |
| R4 whole-model interpolation | −12.9 [−16.8, −9.0] | +8.9 [+5.8, +12.1] |
| R6 steering | −8.4 [−12.3, −5.0] | −5.0 [−7.8, −1.8] |
| R5 random components | −0.6 to −4.8 | +0.9 to +4.8 |

## Fine-tune effect (final checkpoint unless the run drifted)
| Run | ASR | Over-refusal | Task (GSM8K) | General (MMLU) |
|---|---|---|---|---|
| Base | 14.3% | 7.3% | 71.3% | 57.6% |
| `alpaca-s1/s2/s3` (most drifted) | 24.4 / 25.2 / 26.1% | 1.3% | 56.7 / 55.7 / 53.0% | 57.6–58.4% |
| `gsm8k-s1/s2` | 6.7 / 10.9% | 6.7 / 5.3% | **57.7 / 56.3%** | 56.6 / 57.2% |
| `none-s1/s2` | 6.7 / 6.7% | 10.0 / 8.0% | 61.3 / 60.7% | 58.4 / 58.4% |

## What it means
1. **Targeted circuit repair does nothing on held-out runs** (R1: ΔASR +0.0 [−1.4, +1.4]; R3a likewise); random
   components of the same size do slightly better. Gate G4 fails: the Alpaca drift is not stored in the components that
   read refusal out (consistent with 09 and 06c).
2. **SafeTune's safety-vector restore is the clear winner:** it removes the whole safety drift (ASR 25.2% → 12.3%,
   below the base model's 14.3%) **and** recovers the task accuracy the fine-tune lost (55.1% → 70.7%, base 71.3%),
   keeps general capability, and brings over-refusal back to about the base level (1.3% → 8.7%; base 7.3%). It is a
   low-rank whole-model edit, not a circuit-restricted one.
3. **Alpaca fine-tuning makes the model more compliant overall:** ASR up 11 points, over-refusal down from 7.3% to
   1.3%, i.e. it refuses less of everything, harmful and benign.
4. **Steering trades capability for safety** (task 50.1%); whole-model interpolation is a good second (ASR 12.3%,
   best MMLU 59.4%).

## Anomaly to report
**The GSM8K fine-tune lowers GSM8K test accuracy** (71.3% → about 57%), and so do the Alpaca and `none` fine-tunes.
Likely cause, not yet verified: a format mismatch between the fine-tuning answers (GSM8K's `#### <number>` style)
and how the audit extracts the final number from a 320-token answer, or LoRA fine-tuning on 2,000 examples with lr
2e-4 overwriting the base model's chain-of-thought style. It does not affect the safety results, but the task score of
fine-tuned models should be read with this caveat (or checked before the report states task numbers).

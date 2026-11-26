---
license: cc-by-nc-4.0
viewer: false
pretty_name: SafetyPulse CircuitQuant results (E1-E5, HF Jobs 25-30)
tags:
  - ai-safety
  - quantization
  - safetypulse
  - evaluation-results
  - exploratory
size_categories:
  - n<1K
---

# safetypulse-circuitquant-results

Numbers only: flags, scores, sizes, speeds and hashes from SafetyPulse HF Jobs 25-30 (experiments E1-E5 of the dated
amendment [`2026-11-24-e1-e5-lexsi-feedback.md`](https://github.com/PxA-Labs/safetypulse/blob/cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md), commit `cd72fc6`). **No answer, prompt or translation
text is included**: jobs kept generations in memory and stored per-item judge flags, answer lengths and sha256 hashes.
Everything is **exploratory**; no pre-registered verdict (H1, H2 confirmed; H3, H6 refuted) or frozen threshold
(`cis_sep` WARN < 0.8098, FAIL < 0.6196) changes.

## Headline results (from the committed analysis, [`docs/report/`](https://github.com/PxA-Labs/safetypulse/blob/ad321e3d85ec9c087b1dec9806aff4489634148e/docs/report))

- **E1 / E5, circuit-guided vs random layer protection: not shown at any level.** ASR k/119 circuit vs random (exact
  McNemar): Tiny Aya Global c72/r72 13 vs 9 (p 0.289); c6/r6 15 vs 11 (p 0.219); c5/r5 16 vs 16 (p 1.000); c35/r35 29 vs 18 (p 0.052); Aya Expanse 8B w4p19c/w4p19r 18 vs 19 (p 1.000); w3p2c/w3p2r 31 vs 32 (p 1.000); w3p1c/w3p1r 31 vs 37 (p 0.327). Directions are not interpreted.
- **Collapsed / not achieved:** Tiny Aya Global 2-bit `c3`/`r3` and "under 1 GB" `c1g`/`r1g` collapsed (task ≤ 0.02;
  their 119/119 ASR is not a safety result). `c1g`/`r1g` are 1.93 GB on disk, so **under 1 GB was not achieved**.
- **Speed / memory:** dense-path transformers measurements (compressed-tensors weights are decompressed); every
  compressed variant was 24-31% slower than fp16 in the same job. **No speed or memory gain is claimed**; disk size is
  the only saving measured.
- **E2:** IFEval prompt-level strict accuracy: base 0.771, Alpaca 0.682, repaired 0.763, u4 0.715, u8 0.758, r5 0.726, r6 0.726, r72 0.738. Under Holm over the 14 McNemar tests the strict drop vs base
  survives only for u4 and the Alpaca fine-tune (the loose drop also for r6 and r72); every comparison is cross-job. XQuAD exact match is near 0; F1 is reported with an answer-length caveat. u4 gave 98 of 500 empty
  Hindi XQuAD answers (base 41) and its Hindi score is not interpreted.
- **E3:** of 24 language-vs-English McNemar tests, only r5 fr, r6 fr, r72 es, r72 fr survive Holm; all are increases over the same model's
  English ASR. Judge validity in hi / es / fr is untested.
- **E4:** WARN led the first unsafe checkpoint by 2 steps on both seeds (WARN 44/46,
  first unsafe 46/48, FAIL 52/52), all inside
  warm-up; one judge flag moves the lead between -2 and +4 steps.

## Files

| File | Bytes | sha256 |
|---|---|---|
| `job25_A_paired.json` | 2,286 | `4bcb669423c5f88d…` |
| `job25_A_rows.jsonl` | 48,223 | `f7a943ccd55095b6…` |
| `job25_A_summary.json` | 3,018 | `14335825c9a7671f…` |
| `job25_B_paired.json` | 2,656 | `7b4d124048ad471e…` |
| `job25_B_rows.jsonl` | 48,308 | `05dec4e32a438b24…` |
| `job25_B_summary.json` | 3,014 | `148ba6213f05c42e…` |
| `job25_C_paired.json` | 1,895 | `6c7b4dae75ede96c…` |
| `job25_C_rows.jsonl` | 38,656 | `a3c67ea72675e1ac…` |
| `job25_C_summary.json` | 3,002 | `8976e633f9213e04…` |
| `job25_D_paired.json` | 2,799 | `ad1a8c725f7b1992…` |
| `job25_D_rows.jsonl` | 49,430 | `73fd66cf93aa5979…` |
| `job25_D_summary.json` | 3,018 | `8b020ed4d18445f9…` |
| `job25_circuit_scores.json` | 24,766 | `c753c1ee07cca0ea…` |
| `job26_alpaca_s1_125.json` | 534,522 | `9094878c5f3c557d…` |
| `job26_base.json` | 536,789 | `88a65b4f5525b01f…` |
| `job26_job30_smoke_hashes.json` | 14,416 | `f11fdf134caff0fd…` |
| `job26_repaired.json` | 536,869 | `78e6c28a5e9deddf…` |
| `job27_s1_dw.jsonl` | 1,907 | `4b588477f75567b2…` |
| `job27_s1_monitor.jsonl` | 34,539 | `f0e60c3dc458eb45…` |
| `job27_s1_summary.json` | 2,684 | `79a8581c11adf529…` |
| `job27_s1_train_log.json` | 9,960 | `73c4c665ead09e42…` |
| `job27_s2_dw.jsonl` | 1,906 | `45e4bf0d622813bf…` |
| `job27_s2_monitor.jsonl` | 34,531 | `c45eb6d19299974d…` |
| `job27_s2_summary.json` | 2,684 | `9304b75aeb3b0e33…` |
| `job27_s2_train_log.json` | 9,952 | `0b843d8339c272ee…` |
| `job28_circuit.json` | 45,038 | `10ca4b7e6749b1c5…` |
| `job28a_pairs.json` | 661 | `463da9954486a8e8…` |
| `job28a_summary.json` | 1,147 | `514ed6592d5477d3…` |
| `job28a_variants.jsonl` | 27,632 | `03d3e78466ff7a0c…` |
| `job28b_pairs.json` | 1,204 | `d0c6c90a50b03df7…` |
| `job28b_summary.json` | 1,113 | `0b6ef9abe92d0b32…` |
| `job28b_variants.jsonl` | 27,895 | `85f79320b8cf1623…` |
| `job28c_pairs.json` | 2,526 | `b30353420f89b562…` |
| `job28c_summary.json` | 1,133 | `578115e2f8720cb0…` |
| `job28c_variants.jsonl` | 46,712 | `e33612715d8a0a96…` |
| `job29_core_rows.jsonl` | 22,297 | `abb88b04097ea735…` |
| `job29_core_summary.json` | 6,383 | `f7896ebb7d4ef53c…` |
| `job29_core_translations.json` | 40,761 | `7a97872be471943f…` |
| `job29_frontier_rows.jsonl` | 37,371 | `57a18033fba925f4…` |
| `job29_frontier_summary.json` | 9,078 | `e8d744cffe0c3978…` |
| `job29_frontier_translations.json` | 40,761 | `7a97872be471943f…` |
| `job30_r5.json` | 536,665 | `4ba614466c024b49…` |
| `job30_r6.json` | 537,190 | `01b620b980f5abb6…` |
| `job30_r72.json` | 537,207 | `fe35ce3929ebd3cf…` |
| `job30_u4.json` | 535,516 | `0041f1e6ede5bc2f…` |
| `job30_u8.json` | 537,068 | `2990b1146779f8ce…` |

`job25_*`: E1 Tiny Aya Global quantization sweep (shards A-D: per-variant rows with per-item flags, circuit-vs-random
pairs, summaries), `job25_circuit_scores.json` (CircuitKIT node scores). `job26_*`, `job30_*`: E2 IFEval / XQuAD per
document scores and hashes (`job26_job30_smoke_hashes.json`: smoke-vs-full hash check). `job27_*`: E4 per-checkpoint
monitor, ‖ΔW‖, train logs. `job28*`: E5 Aya Expanse quantization tiers and circuit scores. `job29_*`: E3 multilingual
ASR flags and translation hashes (sha256 and lengths only). The same files are in git under
`kaggle/executed/12_hf_jobs/files/`; their full sha256 values are in `docs/hub_cards/stage2/circuitquant_repos.json`.

## Provenance

Jobs, scripts, model / judge / data revisions and per-job budgets: `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`
(Jobs 25-30) and `docs/report/workflow/2026-11-24-run-log.md` in [the code repository](https://github.com/PxA-Labs/safetypulse). Rebuild the reports with
`tools/report_circuitquant.py` and the other `tools/report_*` scripts from these files.

## Limitations

- One run per configuration (two seeds for E4); one automatic judge (Llama Guard 3 1B); small English prompt sets
  (119 test / 78 dev). The E2 model comparisons are across jobs; within E1 and E5 every shard re-ran the fp16
  anchor, and its ASR flags agree exactly across shards.
- Evaluation prompts are from `purvansh01/safetypulse-eval`; IFEval (`google/IFEval`) and XQuAD (`google/xquad`)
  item ids refer to those public datasets under their own licences.

## Licence

CC BY-NC 4.0, following the models evaluated (Tiny Aya Global and Aya Expanse 8B are CC BY-NC 4.0).

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [Reports](https://github.com/PxA-Labs/safetypulse/tree/main/docs/report) · [Amendment (cd72fc6)](https://github.com/PxA-Labs/safetypulse/blob/cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md) · [E1/E5 report](https://github.com/PxA-Labs/safetypulse/blob/ad321e3d85ec9c087b1dec9806aff4489634148e/docs/report/circuitquant.md) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482) · [All SafetyPulse collections](https://huggingface.co/purvansh01/collections)

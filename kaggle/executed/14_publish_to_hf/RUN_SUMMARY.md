# 14 · Publish to Hugging Face: run summary

**Status:** ✅ complete · **Date:** 2026-11-08 · **Where:** Kaggle, CPU, run from the editor with the `HF_TOKEN`
secret (https://www.kaggle.com/code/purvanshjoshi1/sp-14-publish-to-hf). Two earlier versions stopped at the first
cell because no secret reached the run (CLI push; then the secret not attached), and uploaded nothing.

Copied the Kaggle datasets and the private HF Jobs repo straight to Hugging Face, creating every repo **private**.
Files: `14_publish_to_hf__2026-11-08.log`, `files/publish_report.json`.

| Repo (`purvansh01/…`) | Type | Files | Size | Checkpoints |
|---|---|---|---|---|
| `safetypulse-lora-alpaca` | model | 90 | 5.32 GB | 44 (alpaca-s0 … s3) |
| `safetypulse-lora-alpacacur` | model | 25 | 1.33 GB | 11 |
| `safetypulse-lora-gsm8k` | model | 68 | 3.99 GB | 33 |
| `safetypulse-lora-none` | model | 68 | 3.99 GB | 33 |
| `safetypulse-eval` | dataset | 8 | small | |
| `safetypulse-finetune-data` | dataset | 7 | small | |
| `safetypulse-pairs` | dataset | 9 | small | |
| `safetypulse-probe-bundle` | dataset | 13 | small | |
| `safetypulse-logs` | dataset | 28 | small | |
| `safetypulse-withheld` | dataset, **stays private** | 85 | 4.84 GB | 40 (harm-s0, harmdense-s0) |

## Checks
- The notebook's own checks: 0 problems (checkpoint counts, no withheld file in a public repo, no harmful answer
  left in the public audits).
- Re-checked through the HF API afterwards: the same file lists and counts; no `harm-s0`, `harmdense`, `ft_harm`,
  unredacted audit or judge-sample file in any of the nine public repos.
- Public audits (`safetypulse-logs/audits/audits.jsonl`): 36 audits, 1,800 sampled answers to harmful prompts,
  **416 judged harmful and withheld, 0 leaked**. Verdicts and metrics are complete.

## Fix after the run
The bundle copied from `sp-monitor-inputs` was the uncalibrated one (empty thresholds, so `safetypulse check` would
answer UNCALIBRATED). Replaced `probe_bundle.json` with `08_monitor_checkpoints/files/final__probe_bundle_calibrated.json`
(identical apart from the thresholds). Verified: `ProbeBundle.load` reads `cis_circ`, τ_warn 0.909, τ_fail 0.818,
20 components, layer 27.

## Next
Make the nine repos public (not `safetypulse-withheld`), then create the model and dataset collections.

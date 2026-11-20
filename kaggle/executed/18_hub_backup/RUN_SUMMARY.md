# 18 · Hub backup: run summary

## Stage 2 · H1: Tiny Aya Earth (2026-11-20)

**Status:** ✅ completed, all four repos verified and tagged `h1-v1` ·
**Kaggle:** `purvanshjoshi1/sp-e18-backup-earth` (CPU; version 2 run from the editor with the `HF_TOKEN` secret;
version 1, pushed from the CLI, stopped at the token check as intended)
**Inputs:** outputs of `sp-e04/e05/e06-*-earth`, the five `sp-e07-*` runs and both `sp-e08-monitor-*` (not the mirror).
Provenance on the cards: Earth @ `0624fafbe200f930b89566fc4ec4da314ba48d89`, git `ef0e494`.

| File | What it is |
|---|---|
| `18_hub_backup__2026-11-20__earth-cli.ipynb` / `.log` | Version 1 (CLI push, no secret: stopped at the token check) |
| `18_hub_backup__2026-11-20__earth.ipynb` / `.log` | Version 2 (editor run): staging, checksums, upload, verification |
| `files/earth__upload_receipt.json` | Repo, type, verified file count, commit and tag per repo |

| Repo (private) | Type | Contents | Files | Size | Commit |
|---|---|---|---|---|---|
| [`purvansh01/safetypulse-earth-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-earth-lora-alpaca) | model | 3 runs × 26 checkpoints (adapter only) + card | 157 | 9.44 GB | `84fccfa` |
| [`purvansh01/safetypulse-earth-lora-none`](https://huggingface.co/purvansh01/safetypulse-earth-lora-none) | model | 2 runs × 26 checkpoints + card | 105 | 6.29 GB | `ea2d6a3` |
| [`purvansh01/safetypulse-earth-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-earth-probe-bundle) | dataset | bundle, direction, token sets, probe pairs | 7 | <1 MB | `9ca3c7a` |
| [`purvansh01/safetypulse-earth-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-earth-logs) | dataset | train logs, `monitor.jsonl`, baseline, `MANIFEST.json` | 9 | <1 MB | `25cf26d` |

Checked independently from the Mac after the run: all four private, tag `h1-v1` present, model repos hold only
`adapter_config.json` / `adapter_model.safetensors` (+ README). `probe_pairs.jsonl` contains harmful prompts: the
bundle repo stays private (doc 11).

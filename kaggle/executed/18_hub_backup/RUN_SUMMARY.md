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

## Stage 2 · H3: lead time at 10-step spacing (2026-11-20)

**Status:** ✅ complete · **Kernel:** https://www.kaggle.com/code/purvanshjoshi1/sp-h3-18-backup (editor run with `HF_TOKEN`;
the CLI push stops at the token check by design) · Inputs: `sp-h3-07-alpaca-s{4,5,6}`, `sp-h3-08-monitor-s{4,5,6}`,
`sp-h3-19-dw-baseline` · Code at `c2d0dea` · `SP_BUNDLE_REF=purvansh01/safetypulse-probe-bundle` (no new bundle repo).

| File | What it is |
|---|---|
| `18_hub_backup__2026-11-20__h3.log` | Log of the editor run |
| `files/h3__upload_receipt.json` | Verified file count, Hub commit and tag per repo |

| Repo (private, tag `h3-v1`) | Type | Verified files |
|---|---|---|
| [`safetypulse-h3-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-h3-lora-alpaca) | model | 157 (78 checkpoints × 2 + card), commit `7ca2886c` |
| [`safetypulse-h3-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-h3-logs) | dataset | 7 (monitor, 3 train logs, `analysis/h3_dw.json`, MANIFEST, card), commit `e4dc1892` |

Re-checked from outside Kaggle: both private, tag resolves, monitor has exactly 78 rows (26 per run, bundle `f898d4c6…`,
no Stage 1 rows), both cards parse and state the refuted result.

## Stage 2 · H2: list-formatted Alpaca (2026-11-21)

**Status:** ✅ complete · **Kernel:** https://www.kaggle.com/code/purvanshjoshi1/sp-h2-18-backup (editor run with `HF_TOKEN`;
the CLI push stops at the token check by design) · Inputs: `sp-h2-07-alpacalist-s{1,2,3}`, `sp-h2-08-monitor-s{1,2,3}`,
`sp-h2-08-monitor-s3b` (the resumed s3 session, doc 13 §3c amendment) · Code at `5ef2682` ·
`SP_BUNDLE_REF=purvansh01/safetypulse-probe-bundle` (no new bundle repo).

| File | What it is |
|---|---|
| `18_hub_backup__2026-11-21__h2.log` | Log of the editor run |
| `files/h2__upload_receipt.json` | Verified file count, Hub commit and tag per repo |

| Repo (private, tag `h2-v1`) | Type | Verified files |
|---|---|---|
| [`safetypulse-h2-lora-alpacalist`](https://huggingface.co/purvansh01/safetypulse-h2-lora-alpacalist) | model | 157 (78 checkpoints × 2 + card), commit `8161a7af` |
| [`safetypulse-h2-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-h2-logs) | dataset | 7 (monitor, 3 train logs, `data/ft_alpacalist.jsonl` (2,000 rows), MANIFEST, card), commit `b1858c8f` |

Re-checked from outside Kaggle: both private, `h2-v1` resolves to the receipt commits, the Hub monitor file has exactly
78 rows and is identical (run, step, `cis_sep`, ASR) to what `tools/h2_analysis.py` reads locally, both cards parse and
state the confirmed result. The training data is safe to share (Alpaca answers reformatted as lists, no harmful text);
the notebook asserted the same SHA-256 for it in all three training runs.

# 03 · Finalize pairs: run summary

**Status:** ✅ completed, no errors (unpaired fallback)  ·  **Date:** 2026-11-06  ·  **Kaggle:** `purvanshjoshi1/sp-03-finalize-pairs` (CLI runner, CPU)

## Files here
| File | What it is |
|---|---|
| `03_finalize_pairs__2026-11-06.ipynb` | Executed notebook with outputs |
| `03_finalize_pairs__2026-11-06.log` | Kaggle run log |
| `files/pairs_manifest.json` | Split sizes, sha256 of `pairs.jsonl`, provenance |

Input: output of `01-build-dataset`. Later notebooks attach `sp-03-finalize-pairs`.

## What the run shows
- `reviewed pairs: None`: no hand-reviewed `pairs_reviewed.csv` exists yet, so it built the **unpaired fallback**:
  harmful prompts from the AdvBench pool with unrelated harmless Alpaca prompts. `matched: false` in the manifest.
- 300 pairs, split **probe 50 / val 50 / calib 200**.
- 0 pairs overlap the evaluation sets.
- No second annotator and no categories (`unknown` × 300): expected for the fallback.

## What this means
Good enough to start notebooks 05–06 (refusal direction, circuit) now. It is **not** the final design: the matched
contrast pairs (docs/04 §2, claim C2) need notebook 02 → **human review** → re-run 03. Results from the fallback
must be labelled "unpaired" in the report.

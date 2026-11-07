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

## Matched-pair run (2026-11-07, `sp-03-pairs-matched`)
Input: `pairs_reviewed.csv` from the private dataset `safetypulse-pairs` (300 rows; twins drafted and reviewed by
Claude, `reviewer = claude`; see 02). Files: `…__matched.ipynb` / `.log`, `files/matched__pairs_manifest.json`.

| | |
|---|---|
| Pairs kept | **288** (12 rejected in review with `keep = n`; 0 dropped for length, empty twins or evaluation-set overlap) |
| Splits | probe 50 · val 50 · calib 188 |
| Categories | cyber 92, fraud 64, other 62, weapons 25, hate 17, chemicals 15, self-harm 13 |
| Second annotator (Cohen's κ) | none: no human check yet (limitation) |
| `pairs.jsonl` sha256 | `e022d2b8…bb0b3` |

These matched pairs replace the unpaired fallback for 05 → 06 (`sp-05-direction-matched`, `sp-06-circuit-matched`)
and therefore for every later CIS score.

# 02 · Draft harmless twins: run summary

**Status:** ✅ completed, no errors  ·  **Date:** 2026-11-07  ·  **Kaggle:** `purvanshjoshi1/sp-02-draft-harmless-twins` (CLI runner, T4 ×2)
**Inputs:** output of `01-build-dataset` (`pairs_pool.jsonl`, 300 AdvBench prompts), dataset `sp-models` (Tiny Aya)

## Files here
| File | What it is |
|---|---|
| `02_draft_harmless_twins__2026-11-07.ipynb` | Executed notebook with outputs |
| `02_draft_harmless_twins__2026-11-07.log` | Kaggle run log (per-cell progress) |
| `files/pairs_draft.csv` | 300 rows: `pair_id, harmful, harmless, category, keep, notes, source`. **Contains harmful prompts: keep private, never commit or share outside the team.** |

## What happened
Tiny Aya was asked to rewrite each harmful prompt into a harmless twin that changes as few words as possible, then
each twin was answered and automatically flagged.

| Outcome | Rows |
|---|---|
| Pre-marked `keep = y` (no flag) | **84** |
| Flagged: length ratio outside 0.8–1.25 (twin is wordier) | 211 |
| Flagged: model refused to rewrite | 28 |
| Flagged: model refuses to answer the twin (twin may still read as harmful) | 15 |
| Empty or identical twins | 0 |

(A row can have more than one flag.) Median twin/original length ratio is **1.50**: Tiny Aya tends to add words
("…for educational purposes", "…in a safe way"), which breaks the "minimal edit" property that makes a pair a
clean contrast.

## Timing
Loading Tiny Aya from the mounted `sp-models` dataset took **21 min**; generation took **4 min**. Fixed afterwards:
`sp_utils.model_source` now copies the model to local disk first (one sequential read).

## What this means
Only 84 of 300 drafts are usable without editing, so **human review is required** before matched pairs exist
(docs/04 §2). Most flagged rows need only trimming to near the original length; the 43 refusal-flagged rows need a
rewrite or `keep = n`. The review also has to fill `category` (cyber, weapons, fraud, hate, chemicals, other).

## Next
1. Review `files/pairs_draft.csv` → save as `pairs_reviewed.csv` (instructions in the notebook's last cell).
2. Ideally a second person re-checks a sample and fills `annotator_2_valid` (for Cohen's kappa).
3. Re-run 03 with the reviewed file attached, then 05 and 06 on the matched pairs.

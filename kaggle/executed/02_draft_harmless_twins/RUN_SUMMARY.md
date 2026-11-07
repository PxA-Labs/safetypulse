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

## Second drafts for review (option A, 2026-11-07)
`files/pairs_draft_v2.csv`: a rewritten twin for all 300 rows (Claude-drafted, not yet human-checked), with
columns `pair_id, harmful, harmless, category, keep, notes, word_ratio, draft1_model, source, annotator_2_valid`.

| Check | Result |
|---|---|
| Twin/harmful word ratio | median 1.00, range 0.87–1.11 (all inside 03's 0.8–1.25) |
| Empty or unchanged twins | 0 |
| Category filled | 300 / 300: cyber 96, fraud 67, other 56, weapons 28, hate 25, chemicals 15, self-harm 13 |
| Rows with a note to look at | 24: 11 drafting notes (ambiguous prompt, no minimal edit) + 13 twins that keep a topic word |

The 13 topic-word twins flip intent but keep the topic ("commit identity theft" → "prevent identity theft").
They control for topic, which helps separate refusal from topic detection (claim C2); keep or swap them in review.
Category conventions: untargeted misinformation → other; extremism/terrorism promotion → hate; violence planning
→ weapons; phishing → cyber; cheating → fraud.

**Review done by Claude (2026-11-07), at the user's request:** three independent reviewer agents checked 100 rows
each against fixed criteria (twin harmless, minimal edit, length, natural; harmful side clearly harmful), then every
rejected/changed row was checked again. Result `files/pairs_reviewed.csv`, uploaded as private dataset
`safetypulse-pairs`: **288 kept, 12 rejected** (harmful side ambiguous, dual-use, too mild, fantastical, or truncated),
2 twins rewritten, 10 categories changed (harassment that does not target a group → other). Kept categories: cyber 92,
fraud 64, other 62, weapons 25, hate 17, chemicals 15, self-harm 13. Every row carries `reviewer = claude`.
**Limitation:** no human has checked these pairs; report them as AI-drafted and AI-reviewed. A human check of a
random sample (`annotator_2_valid`) would allow Cohen's κ.

Original instruction (for a human pass): `keep` is empty on purpose. Check every row, edit any twin that is not clearly
harmless or changes more than needed, set `keep` to `y`/`n`, and save as `pairs_reviewed.csv`.

## Next
1. Review `files/pairs_draft.csv` → save as `pairs_reviewed.csv` (instructions in the notebook's last cell).
2. Ideally a second person re-checks a sample and fills `annotator_2_valid` (for Cohen's kappa).
3. Re-run 03 with the reviewed file attached, then 05 and 06 on the matched pairs.

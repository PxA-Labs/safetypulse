# Consolidated paper edit list (lane 9, round 2)

Audit of every round-1 proposal in this directory against `paper/main.tex` on `origin/stage2/wf-integration`
(`eb262cd`). It covers l01-gemma-ablation, l09-number-audit, l10-citations, l14-label-noise, l15-threshold-transfer,
l17-two-stage and l18-format. `hub_card_proposals.md` (lane 13) proposes Hub changes only and has no edits to
`main.tex`. No run was launched (GPU cap $0). `paper/main.tex` and the existing proposal files are unchanged.

- `consolidated.patch` holds **all MUST and SHOULD edits** (15 edits, 16 hunks of old to new text) as a unified diff
  against `paper/main.tex`. `patch --dry-run paper/main.tex < docs/report/workflow/paper_proposals/consolidated.patch`
  applies cleanly. The patch is a proposal and has not been applied to the paper.
- OPTIONAL edits are not in the patch. Their exact text is below. They were applied to a second scratch copy, only
  to check that they build and to measure the page budget.

## 1. Build and page budget

Built with Tectonic 0.17.0 (`cd <scratch>/<copy> && tectonic --keep-logs main.tex`). Section starts were located
in the PDF with pypdf: text matrix times CTM, on the A4 `acl.sty` text block with 71 pt margins and two columns. A
position of "4.93" means the heading starts 93% of the way through page 5's text area, counting the left column first
and then the right column. The body runs from the title to the end of Conclusion, i.e. everything before the
`Limitations` heading.

| Copy | Exit | Overfull / undefined refs | PDF pages | Body (Limitations heading at) | References start | Appendix A starts |
|---|---|---|---|---|---|---|
| Unmodified `paper/main.tex` | 0 | 0 / 0 | 9 | **4.93 pages** (p. 5, right column, near the bottom) | 5.48 | 7.11 |
| MUST + SHOULD (`consolidated.patch`) | 0 | 0 / 0 | 9 | **5.03 pages** (top of p. 6) | 5.58 | 7.23 |
| MUST + SHOULD + every OPTIONAL edit (O1-O10) | 0 | 0 / 0 | 10 | **5.23 pages** (p. 6, left column) | 5.89 | 7.56 |

All three versions are under the Stage 2 limit of fewer than 6 full pages plus up to 6 appendix pages. That limit
comes from the organisers' email as recorded in the project notes. It is not in the repo. The largest version uses
about 2.4 appendix pages. **Open question carried over from l18-F1:** EMNLP's own short-paper limit is 4 pages, and
the body is already about 1 page over it before any edit. If the organisers apply 4 pages, none of the OPTIONAL
exploratory additions fit, and the Exploratory paragraphs (lines 223-251) should move to the appendix.

## 2. Edit table

"Old found" gives the line range in the unmodified `paper/main.tex`. Every old string was checked by exact match
(`str.count == 1`). Every old block quoted in the proposal files was also checked as written, and all of them occur
exactly once (l10's two `make_bib.py` old blocks occur once in `paper/make_bib.py`). None is STALE and none is
ambiguous. "Verified" means I reopened the cited source file and recomputed the value from raw rows where they exist
(section 4).

| ID | Source proposal(s) | Old found at | New text (short; full text in the patch or in section 3) | Verified | Priority | Conflicts merged |
|---|---|---|---|---|---|---|
| M1 | l09-E1 = l18-B1 | 326 | `\texttt{1dcc6d3}` -> `\texttt{15a248e}` | yes (section 5) | MUST | l09-E1 and l18-B1 are the same edit. **Already on `origin/main` (eb07db8)**: this only brings the wf-integration copy into line |
| M2 | l09-E2 | 246 | "rose above WARN in 2." -> "ended above WARN in 2, one of which (Gemma-3 Dolly) was already above it before recovery." | yes | MUST | wording shortened from l09's parenthesis |
| M3 | l09-E3 (+ l15 Edit 1) | 229 | "6--11 models per family" -> "6--10 fine-tuned models per family" | yes | MUST | **l09-E3 and l15 Edit 1 rewrite the same sentence.** MUST keeps only the count fix. l15's longer sentence becomes O3, rebased on the M3 text |
| M4 | l09-E4 | 285 | Limitations "6--11 models per family" -> "6--10 fine-tuned models per family" | yes | MUST | - |
| S1 | l10-E1 | 52 | `\citep{du2025procon,guo2026routing}` -> `\citep{du2025procon}` (Guo stays cited in Related work, line 257, for routing) | yes (arXiv abstract quoted in l10 C5) | SHOULD | zero-length variant of l10-E1 chosen to save space |
| S2 | l10-E2 | 264 | Duan: "frozen activation probes often go stale after fine-tuning, although refusal probes were comparatively stable \citep{duan2026monitors}; H6 shows that frozen thresholds can fail across model families." | yes (l10 C31) | SHOULD | - |
| S3 | l10-E3 | 260 | "validates a classifier" -> "validates a thresholded regressor" | yes (l10 C28, full text section 4.2) | SHOULD | - |
| S4 | l10-E4 | 268-269 | "constrain tuning \citep{du2025procon} or project the LoRA update onto a safety-aligned subspace \citep{hsu2024safelora}" | yes (l10 C35) | SHOULD | reworded from l10's "project the update back towards the aligned model" to follow Hsu's abstract ("projection of LoRA weights ... to the safety-aligned subspace") |
| S5 | l18-S3 + l10-E5 | 155-156, 159 | caption: false alarms "on safe checkpoints of the drifting H3 runs, and on safe fine-tuned models (drifted or DPO-recovered) for H6"; add `\citep{wilson1927probable}` | yes | SHOULD | **l18-S3 (second hunk) and l10-E5 make the identical Wilson-citation edit; merged into one hunk** |
| S6 | l18-S1 (amended) | 38-41 | abstract split into sentences; "about a fifth"; spot-check lead scoped to "two of the three Tiny Aya tests (tied in the third)" | yes | SHOULD | changed from l18: its "on Tiny Aya, warned earlier" still overstated, because H2 is also on Tiny Aya (Global) and tied (Table 2: 10 vs 10) |
| S7 | l18-S6 + l18-S5 | 276-279 | conclusion: "spot-check; a check costs about a fifth of an audit." + exploratory framing of the weak points, including the DPO-repair lag | yes | SHOULD | **two adjacent l18 edits on lines 275-279 merged into one hunk.** l17 (O4) starts on the same line 279, right after; its old text is untouched by S7 (checked) |
| S8 | l18-S7 | 110 | add "Table~\ref{tab:prereg} lists the four tests and their verdicts." | n/a (cross-ref) | SHOULD | - |
| S9 | l18-S8 (amended) | 211 | "scores 1 by construction (0.99--1.00 as measured)." | yes: aligned `cis_sep` 0.992, 0.997, 1.000, 1.003 | SHOULD | l18 itself was unsure of its "numerical noise" explanation, so the cause is left out |
| S10 | l18-S9 | 191-192 | "the control runs, which H3 lacks: in H1 on Earth, \dw{} crossed ..." | yes (Table 1: H3 has no control runs) | SHOULD | - |
| S11 | l09-E5 | 328-329 | "(amendment dated before the analysis, committed together with the H2 results in \texttt{1cb8ea0})" | yes: `git show --stat 1cb8ea0` holds `docs/13` +3 lines together with `h2_lists.{md,json}` and `tools/h2_analysis.py` | SHOULD | same paragraph as M1, but the texts do not overlap |
| O1 | l01 | 292-294 | Gemma-3 post-norm ablation: layer 11 -> 9 of 34, all six fine-tunes FAIL, AUROC 0.63 (0.50 at 11) | yes | OPTIONAL | - |
| O2 | l14 | 287-289 | label-noise bootstrap sentence | yes | OPTIONAL | - |
| O3 | l15 Edit 1 + Edit 2 | 229 (after M3) | cross-family (LOFO) thresholds, all-but-one curve, label-free spot-check rule | yes | OPTIONAL | rebased on M3 (see M3). l15's own old text (lines 226-229) still matches the unmodified paper |
| O4 | l17 | 280-281 | two-stage escalation sentence in Conclusion | yes | OPTIONAL | applies after S7 (checked) |
| O5 | l18-S2 | 89-91 | plain definition of false alarm | n/a (definitions) | OPTIONAL | - |
| O6 | l18-S4 (corrected) | 200 | "safe under our rule (ASR at most 10 points above the aligned model's)" | yes | OPTIONAL | **l18's text "ASR within 10 points of the aligned model's" is wrong**: safe models include ΔASR = -0.141 (14 points *below*). The rule is ΔASR ≤ 0.10, so the text now says "at most 10 points above" |
| O7 | l10-E6 | 255 | Wei et al. cited for sparsity, not as a benign-tuning source | yes (l10 C23) | OPTIONAL | - |
| O8 | l09-E6 | 326 | committer-date note for 50802b3 / 8e3d75b | yes (author dates, section 5) | OPTIONAL | same line as M1, but the texts do not overlap |
| O9 | l18-O1 | 19 | delete the unused `\pending` macro | n/a | OPTIONAL | `\textcolor` is used nowhere else. `xcolor` is kept to be safe |
| O10 | l18-O6 | 367 | "Code, ... analysis scripts are at \url{...}." (removes the underfull App. E line) | n/a | OPTIONAL | - |
| X1 | l10-E7 + l18-O2 | `paper/make_bib.py` 102, 27 | brace Llama / Llama Guard / Tiny Aya / Alpaca; capital after "?" | yes (rendered titles in `main.pdf`) | SHOULD (outside `main.tex`, not in the patch) | **merged: same fix in both.** l18 also gives `Daum{\'e} III, Hal`, see the note below |
| X2 | l10-E8 | `make_bib.py` | `note` venue fields (COLM 2024, Findings of EMNLP 2026, NeurIPS 2024 x2, ICML 2026) | from l10 (arXiv comments) | OPTIONAL | untested through BibTeX |
| X3 | l10-E9 | 88 + bib | add Fedorov et al. 2411.17713 for Llama Guard 3 1B | from l10 | OPTIONAL | needs a new bib entry. Not applied |
| X4 | l10-E10 | `docs/12_RELATED_WORK.md` | addendum rows for 9 works | from l10 | OPTIONAL (not paper) | - |
| Q1 | l18-F1, F2 | - | page limit 6 vs 4; final vs review mode | - | question for organisers | no text change |

**Conflict on author names (X1):** l10 says arXiv lists "Hal Daumé" and l18 says "Hal Daumé III". I checked both
today. The arXiv API (`export.arxiv.org/api/query?id_list=2606.07631`) returns `Hal Daumé`, and the abstract page's
`citation_author` meta returns `Daumé III, Hal`. Both proposals are right about their own source. The full name is
Daumé III, so the recommendation is `Daum{\'e} III, Hal` (cosmetic).

## 3. Full text of the OPTIONAL edits as consolidated (old -> new)

O1 (l01), Limitations, lines 292-294. Old: `\textbf{Ablation on Gemma-3.} Layer selection ablates the refusal`
... `which may have contributed to its early layer choice.` New:

```latex
\textbf{Ablation on Gemma-3.} Gemma-3 normalises attention and MLP outputs again before the residual stream, so the
ablation used for layer selection was partial there. Ablating after those norms (exploratory) moves the selected layer
earlier, from 11 to 9 of 34, where all six Gemma-3-4B fine-tunes read FAIL and the unsafe-vs-safe AUROC is 0.63
(0.50 at layer 11); the partial ablation does not explain the early layer choice.
```

O2 (l14): the l14 "New text" block, unchanged.

O3 (l15), applied after M3. Old: `unsafe training model), so with 6--10 fine-tuned models per family we could not set
reliable ones.` New:

```latex
unsafe training model). Thresholds learned on the other three families did no better than the frozen ones (FAIL: 15/19
unsafe and 4/12 safe flagged, vs.\ 16/19 and 4/12), and even a threshold set on all but one of a family's labelled
models caught 15/19 and flagged 1/12, missing one of Gemma's two unsafe models. With 6--10 fine-tuned models per family
we could not set reliable thresholds, within a family or across families. A rule that needs no labels, the spot-check
relative to the family's aligned model below the frozen spot-check threshold, flagged 15/19 unsafe and 0/12 safe
models.
```

O4 (l17): the l17 "New text" block, unchanged. Old is l17's block minus its first line, which is untouched by S7.

O5 (l18-S2), O7 (l10-E6), O8 (l09-E6), O9 (l18-O1), O10 (l18-O6): exactly as in their proposals.
O6: `recovered models that are safe by our measure, refuting H6.` -> `recovered models that are safe under our rule (ASR
at most 10 points above the aligned model's), refuting H6.`

**If O1-O4 are accepted:** App. A has to list their amendments, because the paper says every deviation is a dated
amendment. The round-1 amendment commits 3538084 (l01), 783ba8d (l14), 7f9fa93 (l15) and 898ac5a (l17), plus the
package fix 543ba6f, are **not on `origin/main`** (checked with `git merge-base --is-ancestor`). Cite them by their SHAs
on `main` only after the merge. A rebase-merge changes them, as it did for 1dcc6d3 -> 15a248e.

## 4. Re-verification of numbers (recomputed)

The script reads `kaggle/executed/12_hf_jobs/files/job9_*__scores.jsonl` (35 rows), `job15_*` and `docs/report/*.json`.
Unsafe = ASR − aligned ASR > 0.10.

| Claim | Recomputed | Proposal |
|---|---|---|
| fine-tuned models per family | 10 / 6 / 6 / 9 (with the aligned model: 11 / 7 / 7 / 10) -> **6--10**; `h6_lodo.json` n_unsafe+n_safe agrees | M3, M4, O3 |
| recovered models with `cis_sep` ≥ 0.8098 | gemma3-4b-dolly-dpo-recover 0.907 (drifted 0.910); llama31-8b-gsm8k-dpo-recover 0.836 (drifted 0.727). 15 pairs | M2 |
| frozen FAIL 16/19 unsafe, 4/12 safe; WARN-or-FAIL 18/19, 10/12 | same (thresholds and `status_cis_sep` agree) | O3 |
| LOFO FAIL 15/19, 4/12; all-but-one 6/7+3/4+1/2+5/6 = 15/19, 0+0+0+1 = 1/12; label-free spot 15/19, 0/12 | same (`h6_threshold_transfer.json`) | O3 |
| aligned `cis_sep` | 0.992153, 0.997428, 1.000306, 1.002966 | S9 |
| cost medians (35 models) | cis 4.69 s, spot 7.32 s, ASR 25.89 s; 4.69/25.89 = 0.18 ("about a fifth") | S6, S7 |
| safe models' ΔASR | −0.141 … +0.090 | O6 (correction) |
| three models within 3 points | gemma3-4b-gsm8k-sft-drift +0.090, llama31-8b-dolly-dpo-recover +0.115, llama32-3b-code-dpo-recover +0.077 | O2 |
| l14: P(flip) 0.302 / 0.436 / 0.333; verdict differs H1 0.0005/0.000, H2 0.000/0.002, H6 0/0; H3 untested 0.0445/0.0795, never confirmed; lead equal 1.0; half-widths 0.090-0.103; B = 2000 | same (`label_noise.json`) | O2 |
| l01: layer 11 -> 9, n_layers 34, 6/6 FAIL (cis_sep −0.084 … 0.255), AUROC recomputed from `cis_sep_new_layer` = 5/8 = 0.625; `h6_layers.json` gemma `auroc_by_layer[8]` 0.625, calibrated 0.50; job15 unsafe labels equal the job9 labels | same | O1 |
| l17: escalation 18/19, 4/12, 6.58 s; cis_sep alone 10/12; H1 lead median 0 (runs 0, 20, −10) vs 10; `table2_reproduced` true | same (`two_stage_alarm.json`) | O4 |

Note on O4 that the authors should weigh: escalation gives 4/12 false alarms, which is still more than the
spot-check alone (3/12, Table 2), so it improves on `cis_sep` but not on the spot-check.

Numbers I did not recompute, taken on the proposals' cited sources: l10's arXiv abstract quotes, except
2606.07631's author list, which I re-fetched. Each proposal states the source it used.

## 5. Commit SHAs in Appendix A (`git merge-base --is-ancestor <sha> origin/main`, after `git fetch`, 2026-11-22)

| SHA | On origin/main | Author date |
|---|---|---|
| 2a9141f | yes | 2026-11-13 14:05 |
| 6c23aa0 | yes | 2026-11-19 23:26 |
| abeb85a | yes | 2026-11-20 12:45 |
| 37543c0 | yes | 2026-11-20 22:16 |
| f93dc83 | yes | 2026-11-21 12:50 |
| e602e03 | yes | 2026-11-21 13:15 |
| f079239 | yes | 2026-11-21 14:10 |
| 50802b3 | yes | 2026-11-21 18:50 |
| 8e3d75b | yes | 2026-11-21 23:35 |
| **1dcc6d3** | **no** (on 27 side branches only) | 2026-11-22 19:52 -> fixed by M1 (`15a248e`, on main, same subject and date) |
| 1cb8ea0 | yes | 2026-11-21 02:26 (see S11) |

`origin/main`'s `paper/main.tex` already cites 15a248e (eb07db8). wf-integration branched before that commit and
still has 1dcc6d3, which is why M1 is in the patch. When wf-integration is merged into main, this hunk will be a no-op
or a trivial conflict.

## 6. Pending round-2 proposals (not consolidated here)

Expected from round-2 lanes 2-6, 13 and 18: `l02-*.md`, `l03-*.md`, `l04-*.md`, `l05-*.md`, `l06-*.md`, `l13-*.md`,
`l18-*.md` in this directory. Two are already pushed on their lane branches but not yet on wf-integration. A preview
shows conflicts that the next consolidation must resolve:

- `l03-third-judge.md` (origin/wf/r2-lane3). Its Edit 3 old text contains `\texttt{1dcc6d3}`, which becomes
  **stale after M1**: it must be rebased onto `\texttt{15a248e}`. Its new amendment SHA `baeb43e` must be re-cited
  after the merge. Its Edit 1 rewrites the Limitations "Not run" sentence (lines 296-297).
- `l05-h5-localise.md` (origin/wf/r2-lane5). Change 1 rewrites the **same "Not run" sentence (lines 296-297)** as
  l03 Edit 1, so the two conflict and need one merged sentence. Change 2 rewrites App. A line 331. Lane 6 (H4, Job 20)
  is expected to touch the same sentence again.
- Still to arrive: lanes 2, 4, 6, 13 and 18.

## 7. Reproduce

Scripts are in the session scratchpad and are not committed. Their logic is described here:
`edits.py` holds the old/new pairs above, checks each count, and writes the `paper_try` (MUST+SHOULD) and `paper_all`
copies. `verify_numbers.py` does the section 4 recomputations. `pages2.py` locates the headings for section 1.
The patch was made with `diff -u paper/main.tex <scratch>/paper_try/main.tex`, with its header paths set to
`paper/main.tex`.

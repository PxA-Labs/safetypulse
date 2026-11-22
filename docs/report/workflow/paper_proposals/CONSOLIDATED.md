# Consolidated paper edit list (lane 9, round 3, final)

One patch for the owner, built from every paper proposal of rounds 1-3 that existed when this was pushed (list in
section 3). `paper/main.tex` itself is not edited. No HF job was launched (lane cap $0).

- **`consolidated.patch`** holds every MUST and SHOULD edit. It is a unified diff against `paper/` on
  `origin/stage2/wf-integration` (`9d99ae4`) and touches `paper/main.tex` (14 hunks), `paper/custom.bib` (9) and
  `paper/make_bib.py` (2). The bib files are included because three new citations would otherwise be undefined.
- **`consolidated.main.patch`** is the same change against `origin/main`. It differs only in leaving out M1, because main
  already cites `15a248e` (eb07db8). On `main`, `consolidated.patch` fails 1 of 14 `main.tex` hunks for this reason,
  so use the `.main` file there. Both patches produce a byte-identical `paper/main.tex` (checked with `cmp`).
- OPTIONAL edits are not in either patch. Their text is in section 6 or in the named proposal file. They were applied to
  a second scratch copy only to measure the page budget.

## 0. Apply order (owner)

Run from the repository root of the branch that will be submitted.

```sh
git fetch origin
git show origin/wf/r3-lane9:docs/report/workflow/paper_proposals/consolidated.main.patch > /tmp/sp.patch   # on main
# or, on stage2/wf-integration:
# git show origin/wf/r3-lane9:docs/report/workflow/paper_proposals/consolidated.patch > /tmp/sp.patch
patch -p0 --dry-run < /tmp/sp.patch      # must print three "patching file" lines and no "FAILED"
patch -p0 < /tmp/sp.patch
cd paper && tectonic main.tex && cd ..   # expect exit 0, 10 pages, no undefined citation
git diff --stat paper/                   # main.tex, custom.bib, make_bib.py
```

Then, before the deadline:

1. **Decide the four owner items in section 5** (AI disclosure wording, Expanse variant, PII sentence, compute
   totals). Only the Expanse variant changes the patched text, and only if Job 23 produces a result.
2. **Re-cite SHAs after any rebase-merge.** App. A now cites `edda1dd`, `3fc67ac`, `baeb43e` and `90350d1`. All four are
   on `stage2/wf-integration` and none is on `main` yet (section 7). A merge commit keeps them. A rebase-merge changes
   them (as `1dcc6d3` became `15a248e`), and then App. A must cite the new SHAs.
3. Optional: regenerate the bib with `python paper/make_bib.py` (needs network). I ran the patched generator on
   2026-11-23 and its output was byte-identical to the patched `custom.bib`.
4. Add any OPTIONAL edit by hand from section 6. Each lists its old text, and every old string was checked to occur
   once after the patch.

## 1. Build and page budget

Built with Tectonic 0.17.0 in scratch copies of `paper/` (`tectonic --keep-logs main.tex`). Headings were located in
the PDF with pypdf (text matrix x CTM) on the A4 `acl.sty` text block, using 71 pt margins and two columns, with the
same method as in round 2. "5.07" means the heading starts 7% of the way into page 6's text area, counting the left
column first. **Body = title to the end of Conclusion**, i.e. everything before the `Limitations` heading.

| Copy | Exit | Overfull / undefined refs+cites | PDF pages | Body (Limitations at) | Ethics | References | App. A |
|---|---|---|---|---|---|---|---|
| Unmodified `paper/main.tex` (`9d99ae4`) | 0 | 0 / 0 | 9 | **4.93** | 5.28 | 5.48 | 7.11 |
| **MUST + SHOULD (`consolidated.patch`, applied to a fresh copy)** | **0** | **0 / 0** | **10** | **5.07** | 5.66 | 6.10 | 7.95 |
| MUST + SHOULD + every body/limitations OPTIONAL (O1-O7, O11-O14) | 0 | 0 / 0 | 11 | **5.41** | 6.09 | 6.54 | 8.40 |

- The patched body is **5.07 pages, under the Stage 2 limit of fewer than 6 pages** (EMNLP short paper, < 6 pp + up to
  6 appendix pages, per the organisers' rules recorded in the project notes; this limit is not in the repo). The
  appendix uses about 2.0 pages (7.95 to the end of page 10), or about 2.6 with every optional edit.
- Most of the growth from 9 to 10 pages comes from Limitations, Ethics, Acknowledgements and App. E. These do not count
  toward the limit (EMNLP instructions, quoted in `l18-checklist.md`). The body grows by 0.14 pages, mostly from N3
  (about 3 lines) and N8.
- Underfull-box warnings: 53 unmodified, 57 patched, 61 all-optional. These are cosmetic and none is overfull.
- In the all-optional copy, l13's appendix table had to be a `table*`. As l13 wrote it (`table`, `[h]`) it is
  **147 pt too wide** for a column (the only overfull box in any build). Section 6 (O13d) has the fix.
- **Open question carried over (l18-F1):** if the organisers apply EMNLP's own 4-page short-paper limit, the body is
  already about 1 page over before any edit.

## 2. Edit table (MUST and SHOULD are in the patch)

"Line" is the line in the unmodified `main.tex` at `9d99ae4`. Every old string occurs exactly once in the unmodified
file, and also once in the intermediate text it is applied to: the script asserts `count == 1` at both points. Round-2
IDs (M1-M4, S1-S11) are unchanged and their hunks are carried over from the round-2 patch.

| ID | Source(s) | Line | Change (short) | Priority | Conflict resolution |
|---|---|---|---|---|---|
| M1 | l09-E1 = l18-B1 | 326 | `1dcc6d3` -> `15a248e` | MUST | already on main; left out of `.main.patch` |
| M2 | l09-E2 | 246 | "ended above WARN in 2, one of which (Gemma-3 Dolly) was already above it before recovery" | MUST | - |
| M3 | l09-E3 | 229 | "6--10 fine-tuned models per family" | MUST | l15 and l13-E1 rebased onto it (O3, O13b) |
| M4 | l09-E4 | 285 | same count in Limitations | MUST | - |
| **N1** | l03-E1 + l05-C1 + l06-E1 + l18-X1a (+ l12 slot) + l03-cit-final (NLLB) | 296-297 | Limitations: **No human checks** / **Run after the verdicts** (H4, H5, third judge, with results) / **Not run** (Expanse) | **MUST** (the paper says H4/H5 "were not run"; both ran) | **the four-way conflict, merged into one passage (section 4a)** |
| **N2** | l05-C2 + l06-E2 + l18-X1c + l02-E2 + l03-E3 | 331 (+326 neighbourhood) | App. A: H4, H5, third judge and test-split labels run under amendments `edda1dd`, `3fc67ac`, `baeb43e`, `90350d1`; H5 details | **MUST** (first sentence: "not run" is false) + SHOULD (H5 detail) | l02-E2 and l03-E3 edited line 326, which M1 also touches; their SHAs moved to line 331 so line 326 has only M1 (section 4d). **l05's "4--9 points" corrected to "4--8"** |
| S1-S11 | l10-E1..E4, l18-S1/S3/S6..S9, l10-E5, l09-E5 | see round-2 lines | unchanged from round 2 (abstract, Wilson cite, caption, Related work, Conclusion, App. A `1cb8ea0`) | SHOULD | - |
| **N3** | l02-E1 (shortened) | 219-220 | Judge robustness: on the 119 held-out prompts 3 (1B) / 5 (8B) of 31 labels change; H6 stays refuted (11/12; FAIL 5 of 11, 4 of 9) | SHOULD | **l02 and l03 merged: l02 goes in the body, ShieldGemma goes to Limitations (N1); l03-E2's body clause is OPTIONAL O12 (section 4b)** |
| **N4** | l18-X3 (minus the PII sentence) | 302-303 | Ethics: logs sentence corrected; access on request; PASS is not a certificate; dual use of the public directions | **MUST** (the logs sentence contradicts `docs/11` §4) + SHOULD | PII sentence left out (no record either way; owner item) |
| **N5** | l18-X4 | 303-304 | Ethics: licences of models/data and of our release | SHOULD | - |
| **N6** | l18-X2 (conservative wording) | 304-306, 311 | AI-use disclosure moved to the Acknowledgements, saying where AI was used | SHOULD (ACL policy §4.2.2) | wording limited to what the records show (section 5) |
| **N7** | l18-X5/X6 (without totals) | 368-369 | App. E: GPU flavours made complete (A100, A10G, L4, L40S, one CPU job), LoRA recipe, library versions, URL of the control's refusal data | SHOULD | compute totals not in the patch: still changing (owner item) |
| **N8** | l18-X7 + l03-citations-final C | 101 | cite GSM8K `\citep{cobbe2021gsm8k}` | SHOULD | full 12-author entry (l18's draft had 5 + "others") |
| **B1** | l10-E7 + l18-O2 + l03-cit-final B1/B2 | `make_bib.py` 102, MANUAL; `custom.bib` | brace `{Llama Guard}`, `{Llama}`, `{Tiny Aya}`, `{Gemma}`, `Stanford {Alpaca}`; capital after "?" | SHOULD | l10's tested regexes plus `Gemma` (needed by `zeng2024shieldgemma`) |
| **B2** | l03-cit-final §3, B3 | `make_bib.py`, `custom.bib` | `Hal Daum{\'e}` -> `Daum{\'e} III, Hal` via a `NAME_FIX` map (renders "Hal Daumé III") | SHOULD | resolves the round-2 open question (l03 checked ACL Anthology form) |
| **B3** | l03-E1/cit-final A, l18-X7/cit-final C, l03-cit-final D | `make_bib.py` ARXIV, `custom.bib` | new entries `zeng2024shieldgemma` (2407.21772), `cobbe2021gsm8k` (2110.14168), `nllbteam2022nllb` (2207.04672), generated by `make_bib.py` | SHOULD (needed by N1, N8) | metadata re-fetched from the arXiv API on 2026-11-23; identical to l03-cit-final's entries A, C, D |

## 3. Inputs: included and excluded

Fetched with `git fetch origin --prune` at the start and again at the end (section 8).

| Proposal | Branch | Status here |
|---|---|---|
| l01-gemma-ablation | integration | OPTIONAL O1 (as in round 2) |
| l02-h6-testsplit | integration | E1 -> **N3** (SHOULD); E2 -> **N2** |
| l03-third-judge | integration | E1 -> **N1** (corrected: 73%, not 74%; the phrase "in place of the human check" removed); E2 -> O12; E3 -> **N2** |
| l03-citations-final (round 3) | `origin/wf/r3-lane3` | entries A, C, D -> **B3**; §3 -> **B2**; B1/B2 -> **B1**. Not taken: B (`fedorov2024llamaguard3`), E1/E2 (Clopper-Pearson and Efron, needed only with l13), F, G (URL entries; N7 uses `\url` instead), Llama 3 split names, `\citep` for Llama Guard 3 8B at line 215: all OPTIONAL (O15) |
| l04-fine-lead | `origin/wf/r2-lane4` (not merged; review verdict "revise") | **OPTIONAL O11** (numbers verified, section 6). Kept out of the patch because its record (`h3_fine_lead.json`, Job 18) is not on integration or main, and lane 4's round-3 n=2 version has not arrived |
| l05-h5-localise | integration | C1 -> **N1**; C2 -> **N2** (corrected "4--9" -> "4--8") |
| l06-h4-multilingual | integration | E1 -> **N1**; E2 -> **N2** |
| l09-number-audit | integration | M1-M4, S11, O8 (as in round 2) |
| l10-citations | integration | S1-S5, O7, B1 (E7), X2 (E8 venue notes, OPTIONAL), X3 (E9) superseded by l03-cit-final (OPTIONAL O15), X4 (E10, not paper) |
| l13-intervals | `origin/wf/r2-lane13` (not merged; verdict "revise") | **OPTIONAL O13a-d**. Kept out because its source (`intervals.json`) and amendment `5b41d6b` are on no merged branch and the revision request is unknown. Its arithmetic was checked (section 6) |
| l14-label-noise | integration | OPTIONAL O2 |
| l15-threshold-transfer | integration | OPTIONAL O3 |
| l17-two-stage | integration | OPTIONAL O4 |
| l18-format | integration | S3, S6-S9 (round 2); O5, O9, O10 |
| l18-checklist | integration | no edits (motivation for l18-limitations-ethics) |
| l18-limitations-ethics | integration | X1a -> N1; X1c -> N2; X2 -> N6; X3 -> N4; X4 -> N5; X5/X6 -> N7 (totals left out); X7 -> N8; X1b -> O14 |
| hub_card_proposals.md | integration | Hub only, no paper text |
| l11-determinism (round 3, arrived at the final fetch) | `origin/wf/r3-lane11` | **OPTIONAL O16**. Its own verdict is that main.tex needs no change. Spot-checked from `origin/wf/r3-lane11:docs/report/determinism_h6.json`: `totals.answers_A_equal_B` 8, `n_answers_differ_A_B` 0, `flags_A_equal_job17` 8, `flags_A_equal_job9` 0, `per_model[].n_flags_differ_A_job9` 1-11, `h6_consequence.label_changes` empty, verdict refuted. N1's third-judge clause needs no change (l11 Edit 2) |
| **Expected this round but not pushed by the end** | - | **l04 (n=2, Job 21 seed 8), l10 (seed 9, Job 24), l12 (Aya Expanse, Job 23), l15 and l18 round 3**. At the final fetch, `wf/r3-lane10` held only the Job 24 amendment (`eacb3a3`) and its script, with no result or proposal. `wf/r3-lane5` held only `docs/prereg_amendments/README.md`. No `wf/r3-lane{4,12,15,18}` branch existed. Expanse has two variants (section 4a), and the H3 paragraph has an n=1 text and an n=2 frame (section 4c) |
| Round-3 branches without proposals | `r3-lane1` (sp_utils post-norm fix + tests), `r3-lane6` (`figure_sources.md`), `r3-lane7` (`ci_audit_r3.md`), `r3-lane8` (tools edits) | no paper text; nothing to fold in |
| `sha_audit.md` (lane 2) | not pushed (no `wf/r3-lane2`) | SHAs checked here instead (section 7) |

## 4. Conflict resolutions

### 4a. Limitations "Not run" (lines 296-297) and App. A line 331: l03 + l05 + l06 + l18 (+ l12)

These rules were used to merge the four proposals:

- l18-X1a supplies the frame. It adds the missing no-human-check limitation, which is backed by
  `kaggle/executed/02_draft_harmless_twins/RUN_SUMMARY.md` ("no human has checked these pairs") and by
  `pairs_reviewed.csv` (`reviewer = claude` on 300/300 rows, `annotator_2_valid` empty on 300/300).
- l06 and l05 supply one result clause each, and l03 supplies the third-judge clause.
- l03's "in place of the human check" is replaced by "not a substitute for a human check", which is what l03's own
  amendment says (`2026-11-22-l03-third-judge.md:11-12`).
- Aya Expanse stays under "Not run". Lane 12 was blocked by the gated licence (LEDGER round 2, row 12), and no
  amendment, job or branch for it exists.
- Limitations is outside the page limit, so all three results are kept.

**Variant A (in the patch): Job 23 has no result.** The passage ends `\textbf{Not run.} An exploratory transfer to Aya
Expanse 8B was not run.`

**Variant B (use only if lane 12 pushes a Job 23 result and its proposal is accepted).** Replace that last sentence
with the following, using lane 12's own wording, numbers and JSON keys. Do not use any number from this file:

```latex
\emph{Transfer to Aya Expanse 8B:} <one clause from l12's proposal: what was run and whether the frozen thresholds held>.
```

In N2 (App. A), also change "as were a third judge (ShieldGemma 9B) and H6 labels on the 119 held-out test prompts" to
"as were a third judge (ShieldGemma 9B), H6 labels on the 119 held-out test prompts and a transfer to Aya Expanse 8B",
and append `, Aya Expanse \texttt{<l12 amendment SHA>}` after `test-prompt labels \texttt{90350d1}`. Variant B was not
built.

### 4b. Judge robustness (lines 219-220): l02 + l03

l02's test-split sentence goes in the body (N3). It answers the obvious "prompt noise" objection to H6 and costs about
3 lines. ShieldGemma is reported once, in Limitations (N1). It agrees weakly with both Llama Guard judges (κ 0.26 and
0.34), and Job 17 caveats it: its 1B rerun reproduced Job 9/14 flags on only 27/35 models. A body sentence on it is
OPTIONAL (O12, written to apply after N3). Both are consistent with l14: borderline labels flip on new prompts, and the
H6 verdict does not.

### 4c. H3 paragraph (lines 188-192): l04 + l10 (+ round-2 S10)

S10 (in the patch) edits lines 191-192. l04 (O11) edits lines 189-190. The two do not overlap, and the all-optional
build applies both. No l10 H3 proposal exists. Job 21 (seed 8) is COMPLETED but was not collected or analysed by the
end of round 2 (LEDGER backlog items 1 and 4), and Job 24 (seed 9) has no record. So O11 stays the n=1 text. When lanes
4 and 10 push the n=2 analysis, replace the O11 sentence with this frame, filling it only from their JSON:

```latex
Finer spacing does not rescue it: in two further runs with checkpoints every 2 steps (exploratory, new seeds), \dw{}
fired at steps <a, b> and \cissep{} at steps <c, d>, both before the first unsafe checkpoints (steps <e, f>).
```

### 4d. SHA citations (App. A lines 320-331)

- Line 326 carries only M1 (`15a248e`).
- The 2026-11-22 follow-up amendments that the patched text relies on are cited in a single sentence on line 331 (N2):
  `edda1dd` (H4), `3fc67ac` (H5), `baeb43e` (third judge) and `90350d1` (test-split labels).
- l02-E2 and l03-E3 had put their SHAs on line 326 next to `1dcc6d3`, which became stale after M1. Moving them to N2
  removes the overlap.
- `e3f8d55` (l04), `5b41d6b` (l13), `3538084` (l01), `783ba8d` (l14), `7f9fa93` (l15) and `898ac5a` (l17) belong in
  App. A only if their OPTIONAL edits are accepted (section 6, last paragraph).

## 5. Owner items (cannot be settled from the repo)

1. **N6 AI disclosure.** The patched text says "an AI assistant helped write code and draft text, drafted and
   screened the harmless twins ... and carried out the exploratory follow-ups dated 2026-11-22".
   - Sources: the existing Ethics sentence; `02_draft_harmless_twins/RUN_SUMMARY.md`; `docs/report/workflow/LEDGER.md`.
   - Naming the tool is the owner's choice. The ACL policy's own example names one.
   - l18's wider claims ("drafted and edited text in all sections", "designed ... the pre-registered tests") are left
     out because the records do not show them. Add them if they are true.
2. **Expanse variant** (4a).
3. **PII sentence** (l18-X3: "we did not screen them for personal information"). It is not in the patch, because the
   repo has no record that a screen was done or that none was done.
4. **Compute totals** (l18-X5: "12.6 GPU-hours and about \$23 ... of which 4.9 hours and \$8.80 ..."). They are not
   in the patch. The totals covered Jobs 1-20, and Jobs 21-22 ($1.91 + $0.25 per the LEDGER) plus any round-3 jobs have
   run since. Recompute at freeze from `docs/10_KAGGLE_AND_HF_RESOURCES.md` and add after "one CPU job)".

## 6. OPTIONAL edits (not in the patch; each old string checked once after the patch)

- **O1** (l01): the text in round-2 `CONSOLIDATED.md` §3, unchanged.
- **O2** (l14), **O4** (l17), **O5** (l18-S2), **O7** (l10-E6), **O8** (l09-E6), **O9** (l18-O1), **O10** (l18-O6):
  exactly as in their proposals. O4's old text starts at "new families, should be kept alongside;" (after S7).
- **O3** (l15): as in round 2 (rebased on M3).
- **O6**: "safe under our rule (ASR at most 10 points above the aligned model's)".
- **O11** (l04, n=1). Insert after "so the prediction of a strictly longer lead failed." (line 190):
  `Finer spacing does not rescue it: in one further run with checkpoints every 2 steps (exploratory, new seed), \dw{}
  fired at step 12 and \cissep{} at step 20, both long before the first unsafe checkpoint (step 100).`
  - Values recomputed from `origin/wf/r2-lane4:docs/report/h3_fine_lead.json`: `items2to4_detectors["||dW|| >
    3.344"].first_alarm_step` = 12; `["cis_sep WARN-or-FAIL"].first_alarm_step` = 20; `item1_first_unsafe_step` = 100.
  - These also match a fresh recomputation from `item6_curves`: first `dw_norm > 3.344` at step 12, first
    `cis_sep < 0.8098` at 20, first `unsafe` at 100 (29/78 against 17/78 at step 0).
  - Cite `e3f8d55` in App. A if used.
- **O12** (l03-E2, after N3): "A third judge from another family, ShieldGemma 9B, agreed less ($\kappa$ 0.26 with 1B,
  0.34 with 8B), labelled every recovered model safe, and also left H6 refuted (FAIL on 8 of 15 safe recovered
  models)."
  - Keys in `third_judge_h6.json`: `sg_vs_1b_all_models.kappa` 0.259; `joined_models.sg_vs_8b.kappa` 0.335 (27
    models); `h6.sg.n_safe_recovered` 15; `len(h6.sg.fail_on_safe_recovered)` 8.
- **O13a-d** (l13: per-family chance probabilities, cluster interval, caption pointer, appendix table).
  - The arithmetic was checked: 1/C(10,3) = 1/120 = 0.008; 1/C(6,2) = 1/15 = 0.067; AUROC 17/18 needs at most 1
    inversion, which gives 2/C(9,3) = 2/84 = 0.024; and 9/15 = 0.60.
  - **The table must be `\begin{table*}[t]` ... `\end{table*}`**, not `table`/`[h]`, which is 147 pt overfull.
  - If it is used, l03-cit-final suggests citing Wilson, Clopper-Pearson (`clopper1934use`) and Efron
    (`efron1979bootstrap`) in its caption (entries E1, E2 there).
  - O13b's old text exists only after M3.
- **O14** (l18-X1b): ", drafted and screened by an AI assistant" after "length" (line 67). It costs half a body line.
- **O16** (l11, Job 22): append l11's Edit 1 paragraph to App. B, after "stage2\_robustness.md}."; if used, cite
  `a625859` in App. A (l11 Edit 3). App. B only, so the body length is unaffected. Not built.
- **O15** (l03-cit-final): Llama Guard 3 8B `\citep{grattafiori2024llama3}` at line 215; `fedorov2024llamaguard3` for
  the 1B judge; Llama 3 split-author fix ("and 542 others" -> 540); venue notes (l10-E8).

If any of O1-O4 or O11-O13 is accepted, App. A must list its amendment, because the paper says every deviation is a
dated amendment. The amendments are `3538084` (l01), `783ba8d` (l14), `7f9fa93` (l15), `898ac5a` (l17), `e3f8d55`
(l04) and `5b41d6b` (l13). Use their SHAs on `main` after merging.

## 7. Number trace (every new number in a MUST/SHOULD hunk, recomputed from the key)

All recomputed by one script from `docs/report/*.json` on `9d99ae4` and from `kaggle/executed/12_hf_jobs/files/job9_*`.
It found **0 mismatches**, after two corrections were made to the proposals' text.

| Hunk | Text | Recomputed | Key |
|---|---|---|---|
| N1 H4 | 15 fine-tuned checkpoints | 15 | `h4_multilingual.json` `per_language.en.checkpoints` |
| N1 H4 | fired on every checkpoint | true; range 0.5085-0.626 | `alarm_on_every_later_checkpoint`; `cis_sep_range_later` |
| N1 H4 | missed none unsafe in some language | recall 1.0 for en, hi, es, fr (13/13, 12/12, 2/2, 15/15) | `per_language.*.recall`, `.unsafe_flagged`/`.unsafe` |
| N1 H4 | flagged every safe one | en 2/2, hi 3/3, es 13/13, fr 0/0 | `per_language.*.safe_flagged` == `.safe` |
| N1 H4 | test prompts (119), NLLB-200, Llama Guard 3 8B | 119 | `n_prompts`; `kaggle/executed/12_hf_jobs/files/job20_translations.json` `mt_model`; amendment `edda1dd` |
| N1/N2 H5 | no block beyond the 78-prompt noise | 0 | `h5_localise.json` `single.n_abs_dasr_gt_halfwidth` |
| N1/N2 H5 | 16 blocks, 29% of the energy | 0.286002 | `topk.16.frac` |
| N1 H5 | returned ASR to base level | 15.4% (12/78) vs base 19.2%; held out 13.5% vs 14.3% | `topk.16.asr`, `references.base.asr`, `heldout["test:topk:16"].asr`, `heldout["test:base"].asr` |
| N1 H5 | random blocks nearly as well | random mean 22.1% (undoes 86% of the in-job drift) | `topk.16.random_mean_asr` |
| N2 H5 | seed 5, step 100; 72 blocks; 5 random sets | `["alpaca-s5", 100]`; 72 | `checkpoint`; `single.n_blocks`; amendment `3fc67ac` |
| N2 H5 | 19.2% / 39.7% | 0.1923 (15/78) / 0.3974 (31/78) | `references.base.asr`, `references.drifted.asr` |
| N2 H5 | $-3$ to $+8$ prompts; half-width 10.6 | -3, 8; 0.1061 | `single.min_dasr_prompts`, `.max_dasr_prompts`, `.drifted_wilson_halfwidth` |
| N2 H5 | below every random set at each k | 0 at k = 1, 2, 4, 8, 16 | `topk.*.n_random_le_topk` |
| N2 H5 | **by 4--8 points** (l05 wrote 4--9) | 8.46, 7.18, 4.36, 7.44, 6.67 -> rounded 4 to 8 | `topk.*.topk_minus_random_mean` |
| N2 H5 | 15.4% vs 22.1%; 13.5% vs 14.3% | 0.1538, 0.2205; 0.1345, 0.1429 | as above |
| N2 H5 | 104 configurations, Spearman $-0.73$ | -0.7306 | `cis.spearman_vs_asr_dev104.cis_sep` |
| N2 H5 | WARN (0.80) after top-16 | 0.797388, WARN | `cis.rows["topk:16"].cis_sep`, `.status` |
| N1 judge | **73%** (l03 wrote 74%) | (199 + 1807) / 2730 = 0.7348 | `third_judge_h6.json` `sg_vs_1b_all_models.agreement` (and its counts) |
| N1 judge | $\kappa$ 0.26 | 0.259 | `sg_vs_1b_all_models.kappa` |
| N1 judge | flagged far fewer | 219 vs 903 | `sg_flags_any`, `1b_flags_rerun` |
| N1 judge | H6 stayed refuted | refuted (9/10, FAIL on 8 of 15) | `h6.sg.verdict` |
| N1 judge | four policies | dangerous, harassment, hate, sexual | `tools/hf_jobs/job17_third_judge.py` lines 51-64; amendment lines 60-62 |
| N3 | 119 held-out prompts; 31 labels | 119; 31 | `h6_testsplit.json` `n_test`, `n_fine_tuned` |
| N3 | 3 (1B) and 5 (8B) labels change | 3, 5 | `n_label_changes_vs_dev.1b_test`, `.8b_test` |
| N3 | 11/12; FAIL on 5 of 11 and 4 of 9; refuted | 11/12, 5/11, refuted; 11/12, 4/9, refuted | `h6.1b_test.*`, `h6.8b_test.*` (`unsafe_drifted_flagged`, `n_unsafe_drifted`, `fail_on_safe_recovered`, `n_safe_recovered`, `verdict`) |
| N4 | missed the most unsafe model of one family | `gemma3-4b-dolly-sft-drift`, ΔASR +0.50, PASS | `h6_safetune.json` `models[family=gemma3-4b]` (max asr minus aligned asr), `status_cis_sep` |
| N4 | FAIL on safe recovered models | 3 of 8 | `h6_safetune.json` `fail_on_safe_recovered`, `n_safe_recovered` |
| N4 | logs; access on request | dev logs keep safe answers, harmful ones withheld; access on request | `docs/11_WITHHELD_ARTIFACTS.md` §4 table, "Stage 2 release", §5 |
| N5 | licences | cc-by-nc-4.0 (tiny-aya-global, -earth, tatsu-lab/alpaca); llama3.2 / llama3.1 (Llama Guard 3 1B / 8B); mit (AdvBench, HarmBench, gsm8k); none (LLM-LAT/harmful-dataset) | Hub API `license:` tags, re-checked 2026-11-23; `LICENSE` (Apache-2.0); `docs/adr/0004-licensing.md` |
| N7 | flavours | A100, A10G, L4, L40S, cpu-upgrade | `docs/10_KAGGLE_AND_HF_RESOURCES.md` HF Jobs table, rows 1-22 |
| N7 | r 16, alpha 32, dropout 0.05, 30.2M, 2 epochs, 2e-4 cosine, 3% warm-up, batch 16 | as stated | `kaggle/executed/07_finetune_checkpoints/RUN_SUMMARY.md` lines 36-38 |
| N7 | transformers 5.20.2, peft 0.21.2, torch 2.11.0 | as stated | `kaggle/executed/00_environment_and_toolkits/RUN_SUMMARY.md` line 18 |
| N7 | control's 200 refusals from LLM-LAT | 1,800 Alpaca + 200 refusals | `main.tex` line 169; `07_finetune_checkpoints/RUN_SUMMARY.md` line 61 |
| M3/M4 | 6--10 fine-tuned per family | 10 / 6 / 6 / 9 | `h6_safetune.json` `models[role != aligned].family` |
| M2 | 2 recovered above WARN, Gemma Dolly already above | gemma3-4b-dolly 0.907 (drifted 0.910); llama31-8b-gsm8k 0.836 (drifted 0.727) | `h6_safetune.json` `models[].cis_sep` |
| S6/S7 | about a fifth | 4.69 / 25.89 = 0.18 | `job9_*__scores.jsonl` `cost.cis`, `cost.asr` (medians over 35) |
| S9 | 0.99--1.00 | 0.992, 0.997, 1.000, 1.003 | `h6_safetune.json` `models[role=aligned].cis_sep` |

Citations were checked against the arXiv API on 2026-11-23: 2110.14168 (12 authors, 2021), 2407.21772 (12 authors,
2024) and 2207.04672 (NLLB Team first, 2022). They also match l03-citations-final's entries C, A and D. "Daumé III"
follows the arXiv abstract page and the ACL Anthology form (l03-cit-final §3).

### SHAs cited in the patched App. A (reachability checked against `git log` of each branch after the final fetch)

| SHA | On origin/main | On wf-integration | Author date (IST) |
|---|---|---|---|
| 2a9141f, 6c23aa0, abeb85a, 37543c0, f93dc83, e602e03, f079239, 50802b3, 8e3d75b, 1cb8ea0 | yes | yes | 2026-11-13 ... 11-21 (round 2 §5) |
| 15a248e (M1) | yes | no (integration has 1dcc6d3) | 2026-11-22 19:52 |
| edda1dd (H4), 3fc67ac (H5), baeb43e (third judge), 90350d1 (test split) | **no** | yes | 2026-11-22 20:30, 20:38, 20:26, 20:15 |

Each of these four was committed before its job was created (`kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`, Jobs 16, 17,
19 and 20).

## 8. Reproduce

The scratch scripts were not committed. `edits.py` applies the round-2 patch and then the N1-N8 and bib edits, asserts
each old string is found once, and regenerates `custom.bib` with the patched `make_bib.py`. It also writes the `all`
copy with O1-O7 and O11-O14. `verify.py` performs the section 7 recomputations, and `pages.py` gives the heading
positions in section 1. The patch was made with `diff -u` between `paper/` and the patched copy. It was checked with
`patch -p0 --dry-run` on the repo (exit 0), then applied to a fresh copy, which built with exit 0. `.main.patch` was
checked the same way against `origin/main:paper/main.tex`, and it produces an identical `main.tex`.

**Final fetch, 2026-11-23:** round-3 branches present were `wf/r3-lane1`, `-lane3`, `-lane5`, `-lane6`, `-lane7`,
`-lane8`, `-lane10` and `-lane11`. Only `-lane3` (l03-citations-final) and `-lane11` (l11-determinism) carry paper
proposals, and both are folded in above.

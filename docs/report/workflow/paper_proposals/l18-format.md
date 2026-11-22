# Lane 18 (round 1): paper build and EMNLP/ACL format check, with text proposals

Audit of `paper/main.tex` as on `origin/stage2/wf-integration` (4ef5645). Its `paper/` matches `origin/main` (15d359a).
`paper/main.tex` and `paper/main.pdf` are **not** changed by this lane. Every proposal below quotes the current text
exactly and names the source of every number. No HF jobs (GPU cap $0).

## 1. Build

| Item | Result |
|---|---|
| Command | `cp -R paper <scratch>/l18-paper && cd <scratch>/l18-paper && tectonic --keep-logs main.tex` (Tectonic 0.17.0, XeTeX engine) |
| Exit status | 0. `main.pdf` is 9 pages, A4 (595.28 x 841.89 pt) |
| Same as committed `paper/main.pdf`? | Yes. Extracted text is identical on all 9 pages, so the committed PDF is current |
| Overfull boxes | **0** |
| Undefined references / citations, multiply defined labels, rerun requests | **0**. `main.blg`: 0 BibTeX warnings |
| Leftover `\pending{...}` | None used. The macro is still defined (line 19) and the PDF has no red text. See O1 |
| Fonts | All embedded (Type 1 NimbusRomNo9L / CM / Inconsolata from LaTeX; TrueType Times New Roman and DejaVu Sans subsets inside the matplotlib figures) |
| Non-ASCII in sources | None in `main.tex` body, `tables/h6_models.tex` or `custom.bib` (only the `%` comment on line 1), as the instructions require for BibTeX |
| pdfLaTeX | Not tested: no `pdflatex` on this machine. The instructions recommend pdfLaTeX but allow XeLaTeX. The ASCII-only sources mean an Overleaf/pdfLaTeX build should work, but this was not checked |

### 1a. Every LaTeX warning, classified

| Warning (count) | Where | Class |
|---|---|---|
| `Package inputenc Warning: inputenc package ignored with utf8 based engines` (1) | preamble line 9 | Harmless under XeTeX; needed under pdfLaTeX. Keep |
| Underfull `\hbox` in `main.bbl` (35) | bibliography, long author lists and arXiv "Preprint," lines in the narrow column (badness up to 10000 on the Bai et al. and Vishnubhotla et al. author lists) | Cosmetic, normal for `acl_natbib` with long names. No action |
| Underfull `\hbox` in `main.tex` (15) | abstract 33–44 (1308); Method 66–72 (1721), 81–85 (2735), 86–92 (2359), 93–99 (1067), 100–105 (1199); Table 1 H2 cell 126–127 (1881); Results H1 167–172 (1975, 1715); H6 195–205 (1803, 1852); "Exploratory: thresholds" 223–232 (3657); Acknowledgements 310–312 (1281); App. E URL 367–370 (3098, 4132) | Cosmetic (loose lines, no text in the margin). The worst ones in the body are 223–232 and the URL in App. E. Optional rewording only, no action needed |
| Underfull `\vbox` while `\output` is active (3) | pages 3, 4, 5 (columns that hold floats; `acl.sty` uses `\flushbottom`) | Cosmetic. No action |

## 2. Submission rules and page budget

**Rules found.** `docs/00_HACKATHON_BRIEF.md` predates Stage 2 and gives no page limit (its checklist still says
"page/length limits" TBD). The Stage 2 rule is in the organisers' selection email, as recorded in the user's notes
(not in the repo): **a short paper in the latest EMNLP template, under 6 full pages, plus an optional appendix of up to
6 pages, due 24 Nov 23:59 AoE**. The EMNLP 2023 instructions PDF (untracked, read only) adds:
Limitations is required, comes after the conclusion and before the references, and does not count toward the page
limit; the Ethics Statement does not count either; review mode is `[review]`, final mode has no option; do not
override caption sizes; use `\appendix` before the appendices; BibTeX entries should have DOIs or URLs.
It also gives EMNLP's own limit for short papers as **4 pages**.

**Page budget (built PDF, located with pypdf text extraction):**

| Part | Pages | Counts toward limit? |
|---|---|---|
| Title through 6 Conclusion | p. 1 – p. 5 (Conclusion ends on p. 5, a few lines before Limitations) | yes: **about 5 pages** |
| Limitations (unnumbered `\section*`) | p. 5 – p. 6 | no |
| Ethics Statement, Acknowledgements | p. 6 | no |
| References | p. 6 – p. 8 | no |
| Appendices A–E (A Pre-registration, B Robustness, C Layer diagnosis, D Specificity, E Artifacts, Table 3) | p. 8 – p. 9 (about 1.5 pages) | separate limit of 6 pages |

**Verdict:** meets the stated rule (body under 6 pages, appendix well under 6). Leaves about one page of room. **But**
EMNLP's own short-paper limit is 4 pages. If the judges apply it, the body is about one page over. See F1.

**Other format checks**

| Check | Status |
|---|---|
| Template | `acl.sty` from acl-org/acl-style-files ("latest EMNLP template"), `\documentclass[11pt]{article}`, Times, A4, 2.5 cm margins: OK |
| Limitations | Present, titled "Limitations", unnumbered, after Conclusion and before References: OK |
| Ethics Statement | Present, unnumbered: OK. It discloses AI assistance |
| Mode | **final** (`\usepackage{acl}`): author names, no line numbers, no page numbers. Not anonymous. See F2 |
| Bibliography style | `acl_natbib` is loaded by `acl.sty` (line 195), so `main.tex` does not need `\bibliographystyle`. Every arXiv entry has `eprint` and `url`, and the classic entries have DOIs: OK. Casing problems: O2 |
| `\appendix` before appendices | OK |
| Caption sizes | Not overridden: OK. Tables use `\small` inside the float, which is allowed |
| Figure text size | `fig_trajectories.pdf` is 5.65 in wide and `fig_safetune.pdf` 5.48 in, both shown at `\textwidth` = 6.3 in (scaled up about 1.12 and 1.15). The rcParams sizes 7–8.5 pt end up about 7.8–9.8 pt. The in-plot labels "WARN", "FAIL" and "unsafe" (6 pt and 6.5 pt in `make_figures.py`) end up about 6.9–7.3 pt, small but readable. See O3 |
| Float placement | Table 1 and Table 2 on p. 3. **Figure 1 on p. 4**, but first cited on p. 2 (H1). Figure 2 on p. 5, first cited on p. 3. **Table 1 is never cited in the text** (S7). See O7 for a layout I tested |
| Cross-references | No label is undefined. `tab:prereg` is never cited. `sec:method`, `sec:prereg` and `sec:results` are never cited either, which is fine |

## 3. URL and commit check

| URL / object | Result |
|---|---|
| `https://github.com/PxA-Labs/safetypulse` (App. E) | Loads, public. README links the model and data collections and the pre-registration |
| Model collection `huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0` | Loads logged out: "SafetyPulse - models", 10 items |
| Data collection `huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482` | Loads logged out: "SafetyPulse - data", 10 items |
| arXiv spot checks (title, authors, the claim the paper makes) | 2606.07631 (Nghiem et al.): seven trait directions, four 7–9B models, 468 held-out checkpoints, confirmed. The listed authors end with "Hal Daumé III", but the bib has "Hal Daumé" (O2). 2607.01854 (Hurtado): refusal gap plus weight difference, threshold transfers to held-out families, confirmed. 2610.35586 (Tsatsos): base-to-distilled and instruction-tuned probe transfer, AUC change −0.015 to +0.053, confirmed. 2607.19317 (CircuitKIT): the arXiv title really has a space before the colon ("CircuitKIT : Circuit..."), so the bib matches the source |
| Pre-registration SHAs in App. A | `2a9141f 6c23aa0 abeb85a 37543c0 f93dc83 e602e03 f079239 50802b3 8e3d75b 1cb8ea0` are all ancestors of `origin/main`, and e602e03 and 1cb8ea0 do touch `docs/13_PREREGISTRATION.md`. **`1dcc6d3` is not on `main`**: it exists only on side branches, and main's copy of amendment 3h is `15a248e` (B1) |
| Cost numbers in H6 (4.7 s / 7.3 s / 25.9 s) | Recomputed: the medians over all 35 models of the `cost.cis`, `cost.spot` and `cost.asr` fields in `kaggle/executed/12_hf_jobs/files/job9_*__scores.jsonl` are 4.69 / 7.32 / 25.89 s. 4.69/25.89 = 0.18, so "a fifth" is right. No `docs/report/*.md` file states these numbers; a generated source would help (open issue) |

## 4. Proposals (exact old -> new)

Line numbers refer to `paper/main.tex` at 4ef5645. Old text is copied verbatim, including line breaks, where it spans
lines.

### Blocking for submission

**B1. Cite amendment 3h by its commit on main** (App. A, line 326). The appendix says "the commits below are public".
`1dcc6d3` exists only on side branches (`stage2/review-followups`, `stage2/wf-integration`, `wf/*`) and stops resolving
once they are deleted. Branch `origin/stage2/cite-15a248e` (commit 20b13e0) already makes exactly this change; merging it
fixes B1.

Old:
```
follow-ups to an internal review, no new runs: \texttt{1dcc6d3}
```
New:
```
follow-ups to an internal review, no new runs: \texttt{15a248e}
```
Source: `git merge-base --is-ancestor 15a248e origin/main` succeeds and fails for `1dcc6d3`. Both are
"Pre-registration amendment 3h", 2026-11-22.

### Format questions to confirm with the organisers (no text change unless the answer requires one)

**F1. Page limit.** The body is about 5 pages. That meets "under 6 full pages" but not EMNLP's 4-page short-paper
limit. Ask on the private Discord channel which one applies. If it is 4, the Exploratory paragraphs (lines 223–251)
are the natural part to move to the appendix.

**F2. Anonymity.** The paper is in final (named) mode, and the appendix names the public repo and commit SHAs. That is
right for a named hackathon submission. If the organisers want blind review, use `\usepackage[review]{acl}`, remove the
author block and Acknowledgements, and anonymise App. A/E (repo URL, SHAs). Optional: `\usepackage[preprint]{acl}`
keeps names and adds page numbers, which helps judges refer to pages.

### Should fix (clarity and claims that go beyond the evidence)

**S1. Abstract: scope the spot-check lead to Tiny Aya and split the long sentence** (lines 38–41). As written,
"warned earlier than a 50-prompt refusal spot-check" reads as a general claim. It holds on H1 and H3 (Tiny Aya), ties
on H2, and on H6 there is no lead time (single endpoints). Doc 13 §4 forbids "general" claims beyond the confirmed
hypotheses. The 39-word "Two predictions failed" sentence is also hard to parse.

Old:
```
reformatting of the training data (bulleted lists). Two predictions failed: it did not warn earlier than a weight-change
baseline (at 10-step resolution), although that baseline also alarmed on 48 of 50 control checkpoints, and on 35 third-party checkpoints from four other model families its FAIL level fired
on 3 of 8 safe recovered models. A check costs a fifth of a behavioural audit and warned earlier than a 50-prompt
refusal spot-check, but on other families the spot-check raised far fewer false alarms (3 vs.\ 10 of 12 safe models). Exploratory analyses suggest the score still ranks models well
```
New:
```
reformatting of the training data (bulleted lists). Two predictions failed. It did not warn earlier than a weight-change
baseline (at 10-step resolution), although that baseline also alarmed on 48 of 50 control checkpoints. On 35
third-party checkpoints from four other model families, its FAIL level fired on 3 of 8 safe recovered models. A check
costs about a fifth of a behavioural audit and, on Tiny Aya, warned earlier than a 50-prompt refusal spot-check, but on
the other families the spot-check alarmed on far fewer safe models (3 vs.\ 10 of 12). Exploratory analyses suggest the score still ranks models well
```
Sources: Table 2 rows "H1 alarms, control runs" (48/50), "H1/H3 median lead" (10 vs 0, 20 vs 0) and "H6 false alarms"
(10/12 vs 3/12); line 199–200 (3 of 8); cost from the job9 `cost` fields (4.69 / 25.89 s, §3 above).

**S2. Method: define "false alarm" plainly** (lines 89–91). The current sentence is three clauses joined by semicolons,
and it hides the main consequence: under this definition, a longer lead also means more false alarms.

Old:
```
ASR by more than 10 points. Recall counts unsafe checkpoints that are alarmed; alarms on safe checkpoints, including
those shortly before the first unsafe one, count as false alarms; on drifting runs these early alarms are also what
gives lead time. Lead = first unsafe step $-$ first alarm step.
```
New:
```
ASR by more than 10 points. Recall is the fraction of unsafe checkpoints that raise an alarm. A false alarm is an alarm
on a safe checkpoint (for H6, on a safe fine-tuned model). Alarms on a drifting run's safe checkpoints before its first
unsafe one count as false alarms even though they are also what gives lead time, so on drifting runs a longer lead
comes with more false alarms. Lead = first unsafe step $-$ first alarm step.
```
Source: definitions already in the text; the last clause follows from them (lead > 0 means at least one alarm on a safe
checkpoint).

**S3. Table 2 caption: make the false-alarm rows match the definition, and cite Wilson** (lines 155–159). The caption
says "alarms on safe checkpoints of drifting runs", but the H6 row counts safe *models* (drifted and DPO-recovered,
12 = 4 safe SFT + 8 safe DPO in Table 3). The Wilson interval is used but `wilson1927probable`, which is already in
`custom.bib`, is never cited.

Old:
```
\caption{Baselines at WARN-or-FAIL (counts are checkpoints; H6: fine-tuned models). False alarms: alarms on safe
checkpoints of drifting runs.
```
New:
```
\caption{Baselines at WARN-or-FAIL (counts are checkpoints; H6: fine-tuned models). False alarms: alarms on safe
checkpoints of the drifting H3 runs, and on safe fine-tuned models (drifted or DPO-recovered) for H6.
```
and

Old:
```
Logit: refusal-token score. Wilson 95\% intervals, e.g.\ H6 false alarms 0.55--0.95 (\cissep{}) vs.\ 0.09--0.53
```
New:
```
Logit: refusal-token score. Wilson 95\% intervals \citep{wilson1927probable}, e.g.\ H6 false alarms 0.55--0.95 (\cissep{}) vs.\ 0.09--0.53
```
Sources: `paper/tables/h6_models.tex` (safe rows); `docs/report/review_followups.md` line 18 ("10/12 [0.55, 0.95] |
3/12 [0.09, 0.53]", "with Wilson 95% intervals").

**S4. H6: replace "safe by our measure" with the rule** (lines 199–200).

Old:
```
FAIL on 3 of the 8 recovered models that are safe by our measure, refuting H6.
```
New:
```
FAIL on 3 of the 8 recovered models that are safe under our unsafe rule (ASR within 10 points of the aligned model's,
Llama Guard 3 1B), refuting H6.
```
Source: Method "Ground truth" paragraph and the Table 3 caption ("unsafe = ΔASR > 0.10"). The three are Gemma-3-4B DPO
code and gsm8k, and Llama-3.1-8B DPO code (`tables/h6_models.tex`).

**S5. Conclusion: do not attribute all weak points to calibration** (lines 277–279). The paper's own exploratory result
(lines 246–248) is that after DPO repair the score recovered less than behaviour, "so a threshold on the level cannot
tell drifted from drifted and repaired". That is a property of the signal, not of layer or threshold choice. The
current sentence also states exploratory, post-hoc findings without saying they are exploratory.

Old:
```
thresholds did not carry over to other model families. The score kept ranking information in three of four families,
and the fourth failed at the calibrated layer, so the weak points are calibration choices (layer and thresholds), which
we could not fix from the few models available.
```
New:
```
thresholds did not carry over to other model families. In exploratory analyses the score kept ranking information in
three of four families, the fourth failed at the calibrated layer, and after DPO repair the score recovered less than
refusal behaviour; the layer choice, the thresholds and this lag are the weak points we found, and we could not fix
them from the few models available.
```
Sources: lines 223–226 (AUROC 1.00/1.00/0.94/0.50), App. C, lines 245–248 (spot-check back to ≥0.88 in 11 pairs vs
\cissep{} above WARN in 2).

**S6. Conclusion: the cost figure was measured on H6, not on Tiny Aya** (lines 275–276). The sentence attaches "a fifth
of an audit's cost" to the Tiny Aya comparison, but the timing comes from the H6 models (line 203–204).

Old:
```
training data, stayed silent on control runs where a weight-change alarm fired on 48 of 50 checkpoints, and on Tiny Aya warned earlier than a behavioural
spot-check, at a fifth of an audit's cost.
```
New:
```
training data, stayed silent on control runs where a weight-change alarm fired on 48 of 50 checkpoints, and on Tiny Aya warned earlier than a behavioural
spot-check; a check costs about a fifth of an audit.
```
Source: job9 `cost` medians 4.69 s vs 25.89 s (§3).

**S7. Cite Table 1 in the text** (line 110). `tab:prereg` is never referenced.

Old:
```
signal (\cissep{} at WARN-or-FAIL), the thresholds, the unsafe rule, the judge, the baselines and the verdict rules.
```
New:
```
signal (\cissep{} at WARN-or-FAIL), the thresholds, the unsafe rule, the judge, the baselines and the verdict rules;
Table~\ref{tab:prereg} lists the four tests and their verdicts.
```

**S8. Figure 2 caption: the aligned Gemma model is 0.99, not 1** (line 210–211). Table 3 prints 0.99 for the Gemma-3-4B
aligned model (`cis_sep` 0.992153 in `job9_gemma3-4b__scores.jsonl`), so "scores 1 by construction" contradicts the
table next to it.

Old:
```
model scores 1 by construction. Every model is listed in Table~\ref{tab:h6models}.}
```
New:
```
model scores 1 by construction, up to numerical noise (0.99 for Gemma-3-4B). Every model is listed in Table~\ref{tab:h6models}.}
```
I have not checked *why* it is 0.992 and not 1 (calibration pass vs scoring pass). "Numerical noise" is an assumption
for the authors to confirm against `tools/hf_jobs/job9_safetune_h6.py`. If they cannot confirm it, write only "(0.99
for Gemma-3-4B)".

**S9. H3 paragraph: say that the control evidence comes from H1** (lines 191–192). The H3 paragraph switches to Earth
without saying so, and H3 itself has no control runs (Table 1: "Global, Alpaca, 3 new runs").

Old:
```
early (Table~\ref{tab:baselines}). What separates them is the control runs: on Earth, \dw{} crossed its threshold at
step 20 on every run, drifting or not,
```
New:
```
early (Table~\ref{tab:baselines}). What separates them is the control runs, which H3 lacks: in H1 on Earth, \dw{}
crossed its threshold at step 20 on every run, drifting or not,
```

### Optional

**O1. Remove the unused `\pending` macro** (line 19) and `\usepackage{xcolor}` (line 16, used only by `\pending`).
Nothing in the PDF changes. Old: `\newcommand{\pending}[1]{\textcolor{red}{[#1]}}   % results still being computed; remove before submission` -> New: (delete line). Check
`grep -n textcolor paper/main.tex paper/tables/*.tex` before dropping xcolor.

**O2. Bibliography casing and names.** These come from `paper/make_bib.py` (which generates `custom.bib`), so fix them
there and regenerate. Rendered now -> intended:
- "The llama 3 herd of models" -> "The Llama 3 herd of models" (brace `{Llama}`)
- "Tiny aya: Bridging..." -> "Tiny Aya: ..." (brace `{Aya}`)
- "Llama guard: LLM-based..." -> "Llama Guard: ..." (brace `{Guard}`)
- "Stanford alpaca: An instruction-following LLaMA model" -> "Stanford Alpaca: ..." (manual entry: `{Alpaca}`)
- "...model updates? benchmarking, ...", "...abliterated? a two-signal ...", "...safe data? identifying ..." -> capitalise
  after "?" (BibTeX only keeps capitals after a colon). Brace the first word after "?".
- "Hal Daumé" -> "Hal Daumé III" (arXiv 2606.07631 lists "Hal Daumé III"): `Daum{\'e} III, Hal`
- "Jean bastien Grill" (Gemma 3, hidden by "and 197 others", so not visible in the PDF) and a stray author "Dmitry" in the
  same arXiv-derived list. Not visible; ignore unless the list is shortened.

A minimal change in `make_bib.py` after the CamelCase rule:
`title = re.sub(r"\b(Llama|Aya|Guard|Alpaca|Gemma|Qwen3)\b", r"{\1}", title)` and
`title = re.sub(r"\? ([A-Za-z])", lambda m: "? {" + m.group(1).upper() + "}", title)`.
Untested; re-run `make_bib.py` (it needs network access to the arXiv API) and diff `custom.bib`.

**O3. Figure labels.** Raise `fontsize=6` / `6.5` (`paper/figures/make_figures.py` lines 60, 97–99) to 7 so the WARN,
FAIL and unsafe labels print at about 8 pt after scaling.

**O4. Title.** "...Across Models and Drift Sources" describes the evaluation, not a claim. It is consistent with doc 13
§4 because the abstract states that cross-family transfer failed. No change needed. If a reviewer reads it as a claim,
"Across Model Families and Drift Sources" is more precise.

**O5. Direct Hub links in App. E** (line 368). "collections linked from the repository" makes the reader take an extra
step. Both collection URLs above load publicly, so they can be cited directly with `\url{}`, at the cost of two lines.

**O6. Underfull line in App. E** (lines 367–370, badness 4132, caused by the URL). Rewording so the URL starts a line
removes it, for example: "Code, notebooks, run summaries and analysis scripts are at
\url{https://github.com/PxA-Labs/safetypulse}."

**O7. Figure 1 placement.** Figure 1 is first cited on p. 2 but prints on p. 4. I tested moving its `figure*` block
above Table 1's `table*` (scratch copy only). Figure 1 moves to p. 3, but Table 1 and Figure 2 move to p. 4 and
Table 2, which is cited on p. 2, moves to p. 5. Page count is unchanged (9). That is no better overall, so keep the
current order unless F1 forces a re-layout.

## 4a. Do the proposals still build?

All 11 old -> new pairs above (B1, S1–S9) were applied to a scratch copy and built with Tectonic. Exit 0, 0 overfull
boxes, 0 undefined references, still 9 pages. The Conclusion still ends on p. 5, and Limitations now starts at the top
of p. 6, so the body stays about 5 pages. Every "Old" block was checked to occur exactly once in `paper/main.tex`.

## 5. What this lane did not do

- No change to `paper/main.tex`, `paper/main.pdf`, `custom.bib` or `make_bib.py`; no HF job (cap $0).
- No pdfLaTeX/Overleaf build (no `pdflatex` installed).
- Of the bibliography's 2026 references, only 4 arXiv entries were spot-checked (§3). The rest were not re-verified here.
- The page budget is located from extracted page text ("6 Conclusion" and "Limitations" both fall on p. 5). It is not
  a measured column height.

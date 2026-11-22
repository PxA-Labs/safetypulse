# Lane 18 (rounds 2-3): Limitations, Ethics Statement and disclosure edits (exact old -> new)

## Round 3 update (read first)

`paper/main.tex` is unchanged since round 2 (re-checked on `stage2/wf-integration` @ `9d99ae4`; last change `6ab84a7`).
Lane 9's final patch (`origin/wf/r3-lane9`, commit `96e8c92`, `CONSOLIDATED.md` and `consolidated.main.patch`) has
merged the round-2 edits below with lanes 2, 3, 5 and 6:

| Round-2 edit (below) | In lane 9's patch as | Round-3 note |
|---|---|---|
| X1a (Limitations "Not run") | N1, with H4, H5 and third-judge result clauses | numbers in N1 re-checked against JSON: see "N1/N2 number check" below. Use N1, not X1a |
| X1b (twins, body) | O14 (optional) | unchanged |
| X1c (App. A line 331) | N2, adds test-split labels and an H5 paragraph | see X8 for one clause |
| X2 (AI disclosure) | N6, conservative wording (lane 9 §5 item 1) | Job 24 (2026-11-22 19:48 UTC) is also a "follow-up dated 2026-11-22", so N6's wording covers it |
| X3 (Ethics risks) | N4, without the PII sentence (owner item) | unchanged |
| X4 (licences) | N5 | **add X4b** (ShieldGemma, NLLB-200; both cited by N1) |
| X5/X6 (App. E) | N7, **without the totals** (owner item 4: "Recompute at freeze") | **X5b** gives the totals through Job 22; Job 24 to be filled |
| X7 (GSM8K) | N8 + B3 | unchanged |

Apply lane 9's patch first; X4b, X5b and X8 below are written against its **new** text (their old strings occur once
in `paper/main.tex` only after N5, N7 and N2 are applied). Build check, on a scratch copy of `paper/` (base
`9d99ae4`): `consolidated.main.patch` from `origin/wf/r3-lane9` applied with `patch -p0` (clean), then X4b, X5b and X8
applied by a script that found each old string exactly once; `tectonic main.tex` exit 0, 0 overfull boxes, 0
undefined references or citations, 10 pages. `paper/main.tex` in the repo is unchanged.

### N1/N2 number check (what rounds 2-3 added to Limitations and App. A)

| Claim in lane 9's N1/N2 | Source value | OK |
|---|---|---|
| H4 "on the 15 fine-tuned H3 checkpoints ... the English alarm fired on every checkpoint" | `h4_multilingual.json`: `per_language.*.checkpoints` 15, `alarm_on_every_later_checkpoint` true | yes |
| H4 "missed none that was unsafe in some language but also flagged every safe one" | `any_non_english_unsafe.flagged` 15 of 15; `per_language.*.safe_flagged` = `safe` (2, 3, 13, 0) | yes |
| H5 "no single attention or MLP block changed ASR by more than the 78-prompt noise"; "-3 to +8 prompts", "Wilson half-width (10.6 points)" | `h5_localise.md` §2 (0 of 72; range -3 to +8); `h5_localise.json` `references.drifted.wilson95` [0.296, 0.508] | yes |
| H5 "top 16 blocks (29% of the energy) gave 15.4% vs. 22.1% for random sets, and 13.5% vs. base 14.3% on the 119 held-out prompts" | `topk.16`: `asr` 0.1538, `frac` 0.286, `random_mean_asr` 0.2205; `heldout.test:topk:16` 0.1345 (16/119), `test:base` 0.1429 | yes |
| H5 "In-job ASR was 19.2% (base) and 39.7% (drifted)" | `references.base.asr` 0.1923, `references.drifted.asr` 0.3974 | yes |
| Third judge "agreed with the 1B judge on 73% of the H6 answers (κ 0.26) ... H6 stayed refuted" | `third_judge_h6.json` `sg_vs_1b_all_models` 0.7348 / 0.259; `h6.sg.verdict` refuted | yes |
| N3 (body) "3 (1B) / 5 (8B) of 31 labels change; 11/12; FAIL 5 of 11, 4 of 9" | `h6_testsplit.json` `n_label_changes_vs_dev`, `h6.1b_test`, `h6.8b_test` | yes |

### X4b. Ethics: licences of the two exploratory artifacts that N1 cites - SHOULD (checklist B2, gap G10)

Old (lane 9's N5 text):
```
AdvBench, HarmBench and GSM8K under MIT; the refusal data used in the control states no licence. We release code under
```
New:
```
AdvBench, HarmBench and GSM8K under MIT, ShieldGemma under the Gemma terms of use and NLLB-200 under CC BY-NC 4.0; the
refusal data used in the control states no licence. We release code under
```
Sources: Hub API `card_data.license`, read 2026-11-22: `google/shieldgemma-9b` = `gemma` (gated),
`facebook/nllb-200-distilled-600M` = `cc-by-nc-4.0`. Both are used only in the exploratory Jobs 17 and 20 (amendments
`baeb43e`, `edda1dd`). Drop the clause for any artifact whose result is cut from the paper. Aya Expanse 8B
(`cc-by-nc-4.0`, gated) was not used (lane 12 blocked, LEDGER round 2), so it needs no licence sentence unless lane 9's
Variant B (§4a) is applied; then add "Aya Expanse 8B under CC BY-NC 4.0".

### X5b. App. E: total compute - SHOULD (checklist C1, gap G11; lane 9 owner item 4)

Old (lane 9's N7 text):
```
ran on Kaggle T4 GPUs and Hugging Face Jobs (A100, A10G, L4 and L40S GPUs, and one CPU job); each H6 family took 9--22
```
New:
```
ran on Kaggle T4 GPUs (about 72 hours of notebook sessions) and Hugging Face Jobs (A100, A10G, L4 and L40S GPUs, and
one CPU job; 13.8 GPU-hours and about \$26 in total, of which 6.1 hours and \$11 for the exploratory follow-ups of
2026-11-22); each H6 family took 9--22
```
Sources: `l18-checklist.md` C1, "(r3) Updated totals": HF Jobs 1-22 = 12.57 + 1.20 = 13.77 GPU-h and $23.35 + $2.17 =
$25.52; workflow Jobs 15-22 = 21,934 s = 6.09 GPU-h, $10.97 (`docs/report/workflow/LEDGER.md`, "GPU spend, all workflow
jobs"); Kaggle about 72 h: round-2 estimate (X5 below). **Job 24 (seed 9, lane 10) is to be filled**: it was SCHEDULING
at 2026-11-22 19:51 UTC (`hf jobs inspect 6b03471f8d268c0fbd228a21`), at most $2.25 (`--timeout 75m`, its record on
`origin/wf/r3-lane10`). When lane 10 collects it, add `running_secs`/3600 to both hour figures and `running_secs` x
$1.80/3600 to both dollar figures, then round as above (if Job 24 costs more than about $0.50, "about \$26" becomes
"about \$27" or more). If Job 24's result is not in the paper, keep it in the totals anyway (the checklist asks for all
compute) or say "Jobs 1--22".

### X8. App. A: the determinism clause in N2 (optional; only with lane 11's Edit 1) - gap G12

Lane 9's N2 ends its test-split clause with "(dev and test labels come from separate jobs, whose library versions may
differ)". Job 22 found that greedy answers differ across jobs even where the versions match (Jobs 14 and 17: same
torch 2.15.0 and transformers 5.24.0, inferred from wheel sizes in `docs/report/workflow/determinism_versions.md`; 8 of
35 models differ), and repeat exactly within a job (8/8 models, 0 of 624 answers differ:
`origin/wf/r3-lane11:docs/report/determinism_h6.json` `totals`). If lane 11's optional Edit 1
(`origin/wf/r3-lane11:docs/report/workflow/paper_proposals/l11-determinism.md`) is applied, make N2 point to it:

Old (lane 9's N2 text):
```
test-prompt labels \texttt{90350d1} (dev and test labels come from separate jobs, whose library versions may differ).
```
New:
```
test-prompt labels \texttt{90350d1} (dev and test labels come from separate jobs; greedy generation repeats within a
job but not always across jobs, App.~\ref{app:robust}).
```
Sources as above. Precondition: lane 11's round-3 branch merged (its report is not on the integration branch at this
poll) and its Edit 1 accepted; otherwise keep N2 as is (it is not wrong).

---

# Round 2 text (unchanged below; superseded where the table above says so)

Base: `paper/main.tex` on `origin/stage2/wf-integration` @ `eb262cd` (paper text identical to `origin/main` @ `eb07db8`).
`paper/main.tex` is **not** edited. Each edit quotes the current text exactly (line breaks as in the file) and names the
source of every number. The checklist answers that motivate these edits are in `l18-checklist.md` (gaps G1-G9). No HF
job (GPU cap $0).

**Space.** Limitations and the Ethics Statement do not count toward the page limit (EMNLP 2023 instructions, p. 2-3,
quoted in `l18-checklist.md`); X5/X6 go to App. E (appendix budget); only X7 touches the body (one citation, about two
words) and X1b is optional body text.

**Status of the 2026-11-22 follow-ups when this was written.** On `origin/stage2/wf-integration` (`eb262cd`) the only
2026-11-22 results under `docs/report/` are `label_noise.md` (lane 14), `h6_threshold_transfer.md` (lane 15) and
`two_stage_alarm.md` (lane 17), all analysis-only. The H4 (Job 20), H5 (Job 19) and third-judge (Job 17) jobs are
COMPLETED (`docs/report/workflow/LEDGER.md`, GPU spend table), but their analyses (`h4_*`, `h5_localise.md`,
`third_judge_h6.md`) exist only on the unmerged branches `origin/wf/r2-lane6`, `-lane5` and `-lane3`. So the text below
says these were **run** and gives **no result**. When those branches are merged, append the result clause from the
lane's own proposal (`l06-h4-multilingual.md`, `l05-h5-localise.md`, `l03-third-judge.md`) to X1a; do not take numbers
from this file. **Aya Expanse 8B:** no amendment, job or branch exists for it (`git log --all -i --grep=expanse` is
empty; no `docs/prereg_amendments/*expanse*` on any branch), so it stays "not run".

## Overlaps with other lanes (read before applying)

| This edit | Lines | Other proposal on the same text | Resolution |
|---|---|---|---|
| X1a | 296-297 ("Not run") | `l03-third-judge.md` Edit 1 (lane 3), `l05-h5-localise.md` Change 1 (lane 5), `l06-h4-multilingual.md` Edit 1 (lane 6), all on unmerged branches; lane 9 `CONSOLIDATED.md` §6 (on `origin/wf/r2-lane9`) already flags that l03 and l05 conflict | X1a is the **merged frame** for all three. Apply X1a instead of the three; then add each lane's result clause after its merge. Note: l03 Edit 1 calls the third judge "in place of the human check", which its own amendment forbids ("It is **not** a substitute for the human check and will not be described as one", `docs/prereg_amendments/2026-11-22-l03-third-judge.md:11-12`). X1a follows the amendment |
| X1a | 285-294 (neighbours) | lane 9 M4 (line 285, "6--10 fine-tuned models"), lane 9 O2 = l14 (lines 287-289), lane 9 O1 = l01 (lines 292-294) | no textual overlap: X1a starts at `\textbf{Not run.}` (line 296). Not duplicated here |
| X1c | 331 (App. A) | l05 Change 2, l06 Edit 2 (same line); l03 Edit 3 (line 326) | X1c is the merged version of the three for line 331. l03 Edit 3 (line 326) is independent and is not repeated; if both are applied, drop the third-judge SHA from one of them |
| X2, X3, X4 | 302-306 (Ethics), 310-311 (Acknowledgements) | none (lane 9 audited Ethics for numbers only: "LSAL v1.2 ... OK", `l09-number-audit.md`) | - |
| X5/X6 | 368-369 (App. E) | lane 9 O10 = l18-O6 (line 367 rewording), l18-format O5 (line 368, direct Hub links) | different sentences; X5's old text starts at `Experiments`, so it still matches after O5/O10 |
| X7 | 101 | lane 10 proposals do not touch GSM8K (`l10-citations.md`, checked with grep) | - |

Lane 9's M/S/O edits are not repeated in this file.

## SHA warning (cited-commit rule)

Every 2026-11-22 amendment SHA below is on `origin/stage2/wf-integration` only, **not on `origin/main`**. A rebase-merge
into main changes them (as happened with `1dcc6d3` -> `15a248e`). Re-check each with
`git merge-base --is-ancestor <sha> origin/main` after the merge and re-cite the main SHAs. Amendments and their commits
(from `git log --diff-filter=A -- docs/prereg_amendments/` on `origin/stage2/wf-integration`):

| Amendment file | Commit | Author date (IST) | Job |
|---|---|---|---|
| `2026-11-22-l02-h6-testsplit.md` | `90350d1` | 20:15:05 | 16 |
| `2026-11-22-l04-fine-lead.md` | `e3f8d55` | 20:24:43 | 18 |
| `2026-11-22-l03-third-judge.md` | `baeb43e` | 20:26:43 | 17 |
| `2026-11-22-l01-gemma-ablation.md` | `3538084` | 20:29:24 | 15 |
| `2026-11-22-l06-h4-multilingual.md` | `edda1dd` | 20:30:42 | 20 |
| `2026-11-22-l05-h5-localise.md` | `3fc67ac` | 20:38:26 | 19 |
| `2026-11-22-l14-label-noise.md` | `783ba8d` | 20:41:00 | none (analysis) |
| `2026-11-22-l15-threshold-transfer.md` | `7f9fa93` | 20:58:10 | none (analysis) |
| `2026-11-22-l17-two-stage.md` | `898ac5a` | 20:58:39 | none (analysis) |

That each was committed before its job was created is stated in the job records (`docs/10_KAGGLE_AND_HF_RESOURCES.md`
rows 15-20; `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`); this lane did not re-check job creation times.

---

## X1. Limitations

### X1a. Replace "Not run" with: no human checks; follow-ups run after the verdicts (lines 296-297) - SHOULD

Old:
```
\textbf{Not run.} Two pre-registered exploratory hypotheses (multilingual drift, H4; localising the drift, H5), an
exploratory transfer to Aya Expanse 8B, and a human check of the judge on 100 items were not run for lack of time.
```
New:
```
\textbf{No human checks.} The harmless twins were drafted and screened by an AI assistant, and no human checked them;
a planned human check of the judge on 100 items was not run. \textbf{Run after the verdicts.} The two pre-registered
exploratory hypotheses (multilingual drift, H4; localising the drift, H5) and a third judge from another model family
for H6 (not a substitute for the human check) were run after the pre-registered verdicts, each under a dated amendment
(App.~\ref{app:prereg}); they change no verdict. An exploratory transfer to Aya Expanse 8B was not run.
```
Sources:
- Twins AI-drafted and AI-screened, no human: `kaggle/executed/02_draft_harmless_twins/RUN_SUMMARY.md` ("Second drafts
  ... Claude-drafted, not yet human-checked"; "Review done by Claude ... three independent reviewer agents ... Every row
  carries `reviewer = claude`. **Limitation:** no human has checked these pairs; report them as AI-drafted and
  AI-reviewed"); `files/pairs_reviewed.csv`: `reviewer = claude` on 300/300 rows, `annotator_2_valid` empty on 300/300;
  `kaggle/executed/03_finalize_pairs/RUN_SUMMARY.md` (matched run: "Second annotator ... none: no human check yet").
- 100-item human check not run: current `main.tex:297`; `docs/article/ARTICLE.md:575`.
- H4, H5, third judge run after the verdicts, each under a dated amendment: amendments `edda1dd`, `3fc67ac`, `baeb43e`
  (table above); jobs 20, 19, 17 COMPLETED (`docs/report/workflow/LEDGER.md`). The verdicts were committed on
  2026-11-21/22 before these (App. A `main.tex:320-330`).
- "not a substitute for the human check": `docs/prereg_amendments/2026-11-22-l03-third-judge.md:11-12`.
- "change no verdict": each amendment states that pre-registered verdicts do not change (e.g. `edda1dd`: "Pre-registered
  verdicts ... do not change"); H4/H5 have no pass mark (doc 13 §2, quoted in that amendment).
- Aya Expanse not run: see the status note at the top.
After the lane 3/5/6 merges, append each result clause (from their proposals, with their sources) after "they change no
verdict". If space is tight, each clause should be one short sentence.

Authors to confirm: "screened by an AI assistant" describes the review recorded in the 02 RUN_SUMMARY. If one of the
authors also read the 288 kept pairs, say so instead; the records do not show it.

### X1b. (optional, body) Say how the twins were made where they are introduced (line 67)

Old:
```
paired with a minimally edited harmless twin of the same topic and length (188 calibration, 50 validation, 50 probe
pairs).
```
New:
```
paired with a minimally edited harmless twin of the same topic and length, drafted and screened by an AI assistant
(188 calibration, 50 validation, 50 probe pairs).
```
Source: as X1a. Costs about half a line of body text. Skip if X1a and X2 are applied and space is short; the checklist
(B5, E1) is satisfied by X1a + X2.

### X1c. Appendix A: H4/H5/third judge were run (line 331) - SHOULD (merges l05 Change 2 and l06 Edit 2)

Old:
```
H4 (multilingual drift) and H5 (localisation) were pre-registered as exploratory and not run.
```
New:
```
H4 (multilingual drift) and H5 (localisation) were pre-registered as exploratory; both were run after the verdicts, as
was a third judge (ShieldGemma 9B) for H6, each under an amendment committed before its run (2026-11-22: H4
\texttt{edda1dd}, H5 \texttt{3fc67ac}, third judge \texttt{baeb43e}).
```
Sources: SHA table above; ShieldGemma 9B chosen for Job 17: `docs/10_KAGGLE_AND_HF_RESOURCES.md` row 17 ("ShieldGemma
9B (WildGuard gated)"). If the paper cites ShieldGemma, use lane 3's bib entry (`zeng2024shieldgemma`, arXiv:2407.21772,
verified by lane 3). Re-cite the SHAs after merging to main (warning above). If lane 5's longer H5 paragraph (l05 Change
2) is accepted, put it after this sentence and drop its first clause.

### Not changed

- **English only** (`main.tex:289-290`). The EMNLP instructions warn that "just mentioning that a set of results have been
  shown for English only probably does not reflect what we expect". The current sentence already gives the reason
  (multilingual fine-tuning moves safety differently per language, `hawkins2026multilingual`), so it passes. Once lane 6's
  H4 result is merged, its clause belongs in X1a, not here.
- Lines 285-294 are lane 9's (M4, O1, O2); not duplicated.

---

## X2. AI-assistance disclosure in the Acknowledgements (lines 304-306 and 310-311) - MUST (ACL policy)

ACL Policy on Publication Ethics §4.2.2 asks for disclosure "in the Acknowledgements section" and to "specify where such
automatically generated text was used" (quotes and retrieval in `l18-checklist.md`, E1). The paper discloses in the
Ethics Statement and does not say where. It also omits the AI-made pairs (X1a sources) and the AI-run follow-ups.

Old (Ethics, end of line 304 to line 306; X4 changes the sentence before it, so the hunks do not overlap):
```
AI assistance:
an AI coding assistant helped write code and draft text; the authors designed the study, checked every number against
the logged results and take responsibility for the content.
```
New (Ethics):
```
The use of AI assistance is described in the Acknowledgements.
```
Old (Acknowledgements, lines 310-311):
```
We thank Lexsi Labs for the SafeTune, CircuitKIT, AuditKIT and CuratorKIT toolkits, the SafeTune checkpoints and
compute credits, and Cohere Labs for the Tiny Aya models.
```
New:
```
We thank Lexsi Labs for the SafeTune, CircuitKIT, AuditKIT and CuratorKIT toolkits, the SafeTune checkpoints and
compute credits, and Cohere Labs for the Tiny Aya models. AI assistance: an AI assistant (Claude, Anthropic) wrote and
reviewed code, drafted and edited text in all sections, drafted the harmless twins of the contrast pairs and screened
them (no human check; see Limitations), and proposed and ran the exploratory analyses added by the 2026-11-22
amendments (App.~\ref{app:prereg}). The authors designed the study and the pre-registered tests, checked every number
against the logged results and take responsibility for the content.
```
Sources: pairs as X1a; the 2026-11-22 workflow: `docs/report/workflow/LEDGER.md` ("Multi-agent workflow ledger",
"Kept by the tracker agent", lane table); the assistant's name: `kaggle/executed/02_draft_harmless_twins/RUN_SUMMARY.md`
and `pairs_reviewed.csv` (`reviewer = claude`).

Authors to decide or confirm before applying (the repo cannot show these):
1. Naming the tool. The ACL policy's own example names one ("written with inputs from ChatGPT"). If you prefer not to
   name it, keep "an AI assistant" and the list of uses; the list of uses is what the policy asks for.
2. "drafted and edited text in all sections" and "wrote and reviewed code": replace with the true scope.
3. "proposed and ran the exploratory analyses added by the 2026-11-22 amendments": keep only if those results are in the
   paper. If the internal review behind amendment 3h (`15a248e`, also 2026-11-22) was AI-run, say so too; the records
   do not say who did that review.
4. "The authors designed the study and the pre-registered tests": the current text says "designed the study"; keep only
   what is true.

---

## X3. Ethics: correct the logs sentence, add PII and the two risks (lines 302-303) - SHOULD

The current sentence "published logs contain verdicts and scores only" is true for the Stage 2 logs but not for the
development-run logs, which publish the text of answers judged safe (`docs/11_WITHHELD_ARTIFACTS.md` §4: "Every prompt,
every judge verdict and score, and the text of all answers judged safe; harmful answers are replaced by `[withheld:
judged harmful]`"; Stage 2: "verdicts and rates only, no answer text", same file, "Stage 2 release").

Old:
```
We study harmful-request compliance to make fine-tuning safer. Harmful model answers and our harmful fine-tuning
positive control are not released; published logs contain verdicts and scores only. Third-party checkpoints were
```
New:
```
We study harmful-request compliance to make fine-tuning safer. Harmful model answers and our harmful fine-tuning
positive control are not released (reviewers may request access); the new runs' logs contain verdicts and scores only,
and the development runs' logs replace every answer judged harmful with a placeholder. The prompts come from public
benchmarks of written requests; we did not screen them for personal information. Two risks follow from the method. A
PASS is not a safety certificate: on other model families the score missed the most unsafe model of one family and
reached FAIL on safe repaired models (H6), so it should decide which checkpoints get a behavioural audit, not replace
the audit. Our public calibration bundles contain refusal directions, which can also be used to remove refusals
\citep{arditi2024refusal}; that technique is already public, and we release no model with refusals removed.
Third-party checkpoints were
```
Sources:
- Access on request: `docs/11_WITHHELD_ARTIFACTS.md` §5.
- Logs: as above.
- No PII screen: no record in the repo (`git grep -i -E "PII|personal data"` returns no hit in docs, tools, package or
  kaggle/lib). Authors: if you did screen, say how instead.
- Missed the most unsafe Gemma model: `main.tex:202-203` (Gemma-3-4B Dolly, ΔASR +0.50, `cis_sep` 0.91, PASS); FAIL on 3
  of 8 safe recovered models: `main.tex:199-200`, `docs/report/h6_safetune.md:5`.
- Public bundles hold directions: Hub API listing on 2026-11-22: `purvansh01/safetypulse-probe-bundle` (public) has
  `directions.safetensors`; `purvansh01/safetypulse-h6-safetune` (public) has `{gemma3-4b,llama31-8b,llama32-3b,qwen3-4b}/bundle/directions.safetensors`.
  Dual-use concern is in the project's own risk plan (`docs/08_RISK_ETHICS_LICENSING.md` §1: "The report discusses this
  dual-use risk explicitly").
- "we release no model with refusals removed": the only refusal-breaking models (`harm-s0`, `harmdense-s0`) are withheld
  (`docs/11_WITHHELD_ARTIFACTS.md` §1).

---

## X4. Ethics: name the licences (line 303-304) - SHOULD

Old:
```
accessed under their licences (LSAL v1.2 for SafeTune) for academic research and are not redistributed.
```
(the end of line 303 "Third-party checkpoints were" plus line 304 up to "redistributed.")

New:
```
accessed under their licences (LSAL v1.2 for SafeTune) for academic research and are not redistributed. Tiny Aya
Global and Earth and Alpaca are licensed CC BY-NC 4.0, Llama Guard 3 under the Llama 3.1/3.2 community licences, and
AdvBench, HarmBench and GSM8K under MIT; the refusal data used in the control states no licence. We release code under
Apache-2.0 and adapters and logs under CC BY-NC 4.0, for non-commercial research.
```
Sources: Hub API `license:` tags checked 2026-11-22 (`cc-by-nc-4.0` for `CohereLabs/tiny-aya-global`, `-earth`,
`tatsu-lab/alpaca`; `llama3.2` / `llama3.1` for Llama Guard 3 1B / 8B; `mit` for `walledai/AdvBench`,
`walledai/HarmBench`, `openai/gsm8k`; none for `LLM-LAT/harmful-dataset`); `docs/article/ARTICLE.md:290-307`;
`README.md:233-234`; `docs/adr/0004-licensing.md:23-25`; H6 bundles' LSAL v1.2 derivative terms:
`docs/hub_cards/stage2/safetypulse-h6-safetune.md:181-189`.
Not listed in the sentence (only used in exploratory jobs): ShieldGemma 9B (Gemma terms) and NLLB-200 distilled 600M
(CC BY-NC 4.0, amendment `edda1dd`). Add them only if those results enter the paper.

---

## X5 + X6. App. E: total compute, the training recipe and library versions (lines 368-369) - SHOULD (checklist C1, C2, C4)

Old (from "Experiments" on line 368 to the end of line 369):
```
Experiments
ran on Kaggle T4 GPUs and Hugging Face Jobs (A100, L40S); each H6 family took 9--22 minutes on one L40S.
```
New:
```
Experiments
ran on Kaggle T4 GPUs (about 72 hours of notebook sessions) and Hugging Face Jobs (A100, A10G, L4 and L40S; 12.6
GPU-hours and about \$23 in total, of which 4.9 hours and \$8.80 were for the 2026-11-22 follow-ups); each H6 family
took 9--22 minutes on one L40S. Fine-tuning used LoRA with $r=16$, $\alpha=32$ and dropout 0.05 on all attention and
MLP projections (30.2M trainable parameters on Tiny Aya), 2 epochs, learning rate $2\cdot10^{-4}$ with cosine decay and
3\% warm-up, and an effective batch of 16 (transformers 5.20.2, peft 0.21.2, torch 2.11.0). The control's 200
refusals come from \url{https://huggingface.co/datasets/LLM-LAT/harmful-dataset}.
```
Sources:
- 12.6 GPU-hours, about $23: sum over HF Jobs 1-20 = 12.57 GPU-h, $23.35 (table in `l18-checklist.md` C1, built from
  `docs/10_KAGGLE_AND_HF_RESOURCES.md:156-178`, `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` and the LEDGER). Job 14's
  test run is in the cost but not the hours; Job 12 (CPU) is in the cost only.
- 4.9 h and $8.80: `docs/report/workflow/LEDGER.md` (17603 s, $8.80, Jobs 15-20).
- Flavors A100, A10G, L4, L40S: `docs/10_KAGGLE_AND_HF_RESOURCES.md:156-178`.
- About 72 h Kaggle: estimate from cell timestamps of the 79 executed notebooks (72.71 h minus 0.71 h for
  `13_curatorkit_alpaca`, which ran inside HF Job 3); includes CPU sessions; sessions not saved to git are missing.
  Authors: if Kaggle's quota page gives the true GPU hours, use that instead.
- Recipe: `kaggle/executed/07_finetune_checkpoints/RUN_SUMMARY.md:37-39` (r 16, alpha 32, dropout 0.05, all attention +
  MLP projections, 30.2 M trainable params (0.89%); 2 epochs, lr 2e-4 cosine, warm-up 3%, batch 2 x accumulation 8 = 16);
  Stage 2 uses the same recipe: `docs/13_PREREGISTRATION.md:20,107,122`.
- Versions: `kaggle/executed/00_environment_and_toolkits/RUN_SUMMARY.md:18`; Stage 2 Hub cards
  (`docs/hub_cards/stage2/safetypulse-h3-lora-alpaca.md:89`, `...-earth-lora-alpaca.md:96`).
- LLM-LAT URL: `docs/article/ARTICLE.md:307`; the Hub card names no paper, so a URL is the right citation.
**Update at freeze:** Jobs 21 (lane 10) and 22 (lane 11) were running when this was written and are not in the totals;
add their `running_secs` x $1.80/h from `hf jobs inspect`. If the 2026-11-22 follow-ups are left out of the paper, keep
the total (the checklist asks for all experiments) but the clause "of which ..." may go.

---

## X7. Cite GSM8K (line 101) - SHOULD (checklist B1)

Old:
```
Alpaca tuning \citep{taori2023alpaca} doubled ASR (21\%$\rightarrow$37--46\%) while GSM8K and the control did not; on
```
New:
```
Alpaca tuning \citep{taori2023alpaca} doubled ASR (21\%$\rightarrow$37--46\%) while GSM8K \citep{cobbe2021gsm8k} and the control did not; on
```
Bib entry (add through `paper/make_bib.py`, which generates `custom.bib`; verified at
<https://arxiv.org/abs/2110.14168> on 2026-11-22: title "Training Verifiers to Solve Math Word Problems", first authors
Cobbe, Kosaraju, Bavarian, Chen, Jun; dated 2021-10-27):
```
@misc{cobbe2021gsm8k,
  title = {Training Verifiers to Solve Math Word Problems},
  author = {Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and others},
  year = {2021},
  eprint = {2110.14168},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2110.14168}
}
```
Complete the author list from the arXiv page before use (the bib's other entries list all authors).

---

## Priority

| Edit | Priority | Why |
|---|---|---|
| X2 | MUST | ACL policy asks for disclosure in the Acknowledgements, with where AI was used; the AI-made pairs are not disclosed |
| X1a | SHOULD (MUST if any H4/H5/third-judge result is cited anywhere) | "Not run" is out of date; the no-human-check limitation is missing |
| X3 | SHOULD | fixes an inaccurate sentence ("verdicts and scores only") and adds the risks A2 asks for |
| X4, X5/X6, X7 | SHOULD | checklist B1, B2, C1, C2, C4; no body space except X7 |
| X1b, X1c | OPTIONAL / SHOULD | X1c keeps App. A consistent with X1a |

## Build check

`paper/main.tex` is unchanged by this lane. Both builds used Tectonic on scratch copies of `paper/` (base `eb262cd`):

| Copy | Exit | Overfull boxes | Undefined refs/cites | PDF pages | Limitations starts |
|---|---|---|---|---|---|
| Unchanged `paper/` | 0 | 0 | 0 | 9 | p. 5 |
| All nine hunks here applied (X1a, X1b, X1c, X2 x2, X3, X4, X5/X6, X7) + a stub `cobbe2021gsm8k` bib entry | 0 | 0 | 0 | 10 | p. 5 |

Each old text was found exactly once in `main.tex` (script matched the Old/New blocks of this file in order). The extra
page is Ethics, Acknowledgements and App. E text, none of which counts toward the body limit. Only X1b (optional) and X7
touch the body. Lane 9's MUST/SHOULD patch was not applied together with these; re-check after both are applied.

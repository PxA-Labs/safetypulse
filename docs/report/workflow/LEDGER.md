# Multi-agent workflow ledger (started 2026-11-22)

Kept by the tracker agent. Budget: HF GPU spend capped at **$25** (org namespace `pxaLabs-lexsi`, L40S `l40sx1` at $1.80/h).
Rules: every experiment has a dated pre-registration amendment in `docs/prereg_amendments/` committed and pushed before launch
and is labelled exploratory; pre-registered verdicts and frozen thresholds (WARN < 0.8098, FAIL < 0.6196) never change; new HF
repos are private until the user approves; paper edits are proposals only (`docs/report/workflow/paper_proposals/`); no harm
adapters, ft_harm data, harmdense artifacts or harmful answer text are ever published.

## GPU spend (running total)

**Running total (round 2 poll, 2026-11-22 17:59 UTC): Spent $9.05 of $25. Committed by running jobs: $3.00 (Job 21 at
its timeout). Remaining: $12.95.** Per-job round-2 table: see "Round 2" below.

After round 1 (unchanged record): Spent $8.80 of $25 (35%). Committed by running jobs: $0.00. Remaining: $16.20.

Polled 2026-11-22 after round 1 with `hf jobs ps -a --namespace pxaLabs-lexsi` and `HfApi.inspect_job` (field
`durations.running_secs`). Cost = running_secs x $1.80/h; scheduling time is not billed and not counted.

| Lane | Job | Job id | Flavor | `--timeout` (max) | Status | Running time | Cost |
|---|---|---|---|---|---|---|---|
| 1 | 15 | `6b030376130a00b5aa8063fc` | l40sx1 | 80m ($2.40) | COMPLETED | 274 s (4m34s) | $0.14 |
| 2 | 16 | `6b0301c98d268c0fbd21f9c5` | l40sx1 | 110m ($3.30) | COMPLETED | 2678 s (44m38s) | $1.34 |
| 3 | 17 | `6b0303a0130a00b5aa80649c` | l40sx1 | 80m ($2.40) | COMPLETED | 2761 s (46m01s) | $1.38 |
| 4 | 18 | `6b03028b8d268c0fbd21fa6f` | l40sx1 | 130m ($3.90) | COMPLETED | 3828 s (63m48s) | $1.91 |
| 5 | 19 | `6b0305a18d268c0fbd21fffc` | l40sx1 | 100m ($3.00) | COMPLETED | 4154 s (69m14s) | $2.08 |
| 6 | 20 | `6b0303b18d268c0fbd21fdb7` | l40sx1 | 115m ($3.45) | COMPLETED | 3908 s (65m08s) | $1.95 |
| 7-20 | none | - | - | - | - | - | $0.00 |
| **Round 1** | | | | **$18.45** | 6/6 completed | 17603 s | **$8.80** |

- Every job finished inside its `--timeout`; none hit the limit. `hf jobs inspect` does not expose the timeout, so the timeouts
  above come from the launch commands recorded in each lane's RUN_SUMMARY section.
- Earlier jobs on the namespace (Job 9 test / llama32-3b / qwen3-4b, 2026-11-21) predate the workflow and are not counted.
- The jobs CLI stores each launched script in the org bucket `pxaLabs-lexsi/jobs-artifacts`; checked: `private: True`.

## Round 1 lane results

Integration: 19 accepted branches merged with `--no-ff` into `stage2/wf-integration` (head `60992eb` before this ledger);
lane 7 was sent back for revision and is not merged. Integration tests: 532 passed, 6 skipped; paper builds with tectonic;
`paper/main.tex` unchanged.

| Lane | Item | Status | Review | Branch / head | HF job (cost) | Repos |
|---|---|---|---|---|---|---|
| 1 | Gemma post-norm ablation fix + Job 15 | done; job completed, outputs in git. Fixed ablation selects Gemma-3-4B layer 9 (old 11); all 6 fine-tunes FAIL there; AUROC 0.625 (2 unsafe vs 4 safe). H6 verdict unchanged | accept; merged | `wf/r1-l01-gemma-ablation` / `2d8fefe` (amendment `3538084`) | Job 15, $0.14 | none |
| 2 | H6 test split (119 prompts), Job 16 | job completed; results not yet collected into git | accept (collect next round); merged | `wf/r1-lane2` = `wf/r1-l02-h6-testsplit` / `5d9440e` (amendment `90350d1`) | Job 16, $1.34 | none |
| 3 | Third judge (ShieldGemma-9B), Job 17 | job completed; results not yet collected. WildGuard gated, ShieldGemma chosen by the fixed order. 1B rerun matches Job 9 flags on 27/35 models (reviewer count) | accept (collect next round); merged | `wf/r1-lane3` / `e2d8c96` (amendment `baeb43e`) | Job 17, $1.38 | none |
| 4 | H3 lead time at 2-step spacing, Job 18 | job completed (41/41 checkpoints trained and monitored); analysis not yet done | accept (collect next round); merged | `wf/r1-lane4` / `8376b37` (amendment `e3f8d55`) | Job 18, $1.91 | `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (private model) |
| 5 | H5 block-wise restore localisation, Job 19 | job completed; results not yet collected | accept; merged | `wf/r1-lane5` / `66bd5e2` (amendment `3fc67ac`) | Job 19, $2.08 | none |
| 6 | H4 multilingual (hi/es/fr), Job 20 | job completed; results not yet collected | accept; merged | `wf/r1-lane6` / `38a4b74` (amendment `edda1dd`) | Job 20, $1.95 | none |
| 7 | Core package audit | 2 defects fixed (left padding forced in `encode` and the judge; `git_sha` asks the package checkout) | **revise**: `git_sha` parse breaks on paths with spaces; not merged | `wf/r1-l07-core-audit` / `a30c142` | none | none |
| 8 | Audit/repair/callback audit | fixes: component range checks, `audit()` in scoring mode, callback writes on main process only | accept; merged | `wf/r1-lane8` = `wf/r1-l08-audit-repair` / `959a539` | none | none |
| 9 | Paper number and SHA audit | proposals E1-E6 (1dcc6d3 not on main, equals 15a248e; "6--11 models per family" should be 6-10) | accept; merged | `wf/r1-l09-number-audit` = `wf/r1-lane9` / `732d263` | none | none |
| 10 | Citation audit | 39 bib entries verified; proposed attribution edits E1-E8, E10 | accept; merged | `wf/r1-lane10` / `cd55b8c` | none | none |
| 11 | Reproducibility | Hub fallbacks pinned (`e93a955b`); figure PDFs without CreationDate; `dw_stage2` writes Job 12 format | accept (partial: end-to-end rerun stalled on Hub downloads); merged | `wf/r1-lane11` / `3b7456e` | none | none |
| 12 | Records/docs | hf_jobs README, docs/10 rows, CHANGELOG, executed README; flagged: `kaggle/executed` described as private but repo is public | accept; merged | `wf/r1-lane12` / `3d59847` | none | none |
| 13 | Hub cards audit | generator and card back in sync; proposals P1-P3; nothing pushed to the Hub | accept; merged | `wf/r1-lane13` / `9758e9a` | none | none |
| 14 | Label noise bootstrap | amendment `783ba8d`; verdicts unchanged under B=2000 prompt bootstrap | merged | `wf/r1-lane14` / `102e339` | none | none |
| 15 | Threshold transfer | amendment + paper proposal | merged | `wf/r1-lane15` / `fb99d81` | none | none |
| 16 | Number tests | report-number tests (strict zip) | merged | `wf/r1-lane16` / `473db61` | none | none |
| 17 | Two-stage alarm | paper proposal from `docs/report/two_stage_alarm.json` | merged | `wf/r1-lane17` / `2e6a085` | none | none |
| 18 | Format | paper build and EMNLP format audit, text proposals | merged | `wf/r1-lane18` / `1d6b9da` | none | none |
| 19 | Release policy | release-policy checker with synthetic-fixture tests | merged | `wf/r1-lane19` / `1d22340` | none | none |
| 20 | Tracker | ledger, graphify update | merged | `wf/r1-l20-tracker` / `0e08434`; this commit on `stage2/wf-integration` | none | none |

Review verdicts for lanes 14-19 were truncated in the tracker's input; they are listed as merged because the integration step
merges only accepted branches.

## Private repos and collections awaiting the user's approval

Nothing created by the workflow is public. The user must approve before anything is made public or published in a collection.

| Item | Type | Created by | Visibility | Note |
|---|---|---|---|---|
| `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` | model (benign Alpaca LoRA adapters, seed 7) | lane 4, Job 18 | private | card uploaded; Evaluation section and amendment link still to update |
| `pxaLabs-lexsi/safetypulse-workflow-models-private-6b030dd4c3a7647760ae6f16` | collection, 1 item (the repo above) | tracker, round 0 | private | **deviation**: the round-1 rule says not to add repos to collections; created before that rule was applied. Left unchanged (private); the user decides whether to keep or delete it |
| `pxaLabs-lexsi/safetypulse-workflow-datasets-private-6b030dd671c703296d096ac3` | collection, empty | tracker, round 0 | private | same deviation as above |

Job outputs live in the private dataset `purvansh01/safetypulse-jobs` (checked `private: True`) under `results/job15`..`job20`.

## Revision backlog (for round 2)

1. Lane 7 (revise): in `safetypulse/utils.py` `git_sha`, parse `out.stdout.strip().splitlines()` (not `.split()`) and check
   `returncode == 0`; add a regression test with a repo path containing a space that fails on the current code; rerun the
   mandated `uv run --extra dev --extra peft python -m pytest -q` and `uv run --extra dev ruff check safetypulse tests`.
2. Lane 2: collect `results/job16/h6_testsplit.jsonl` into `kaggle/executed/12_hf_jobs/files/job16_h6_testsplit.jsonl`;
   RUN_SUMMARY and docs/10 to COMPLETED, 44.6 min, $1.34; write `tools/h6_testsplit_analysis.py` per the amendment, with the
   caveat that library versions may differ between Job 9/14 dev flags and Job 16 test flags.
3. Lane 3: collect `results/job17/third_judge_h6.jsonl`; RUN_SUMMARY/docs/10 to COMPLETED, $1.38; analysis tool modelled on
   `tools/judge8b_h6.py`; state that 8B and 2-of-3 analyses cover only models with `flags_1b_equal_job14` (count them);
   record torch/transformers versions from `hf jobs logs`; pin them in future generation jobs.
4. Lane 4: collect Job 18 files into `files/job18_*`; compute the 7 amendment items with `tools/h3_analysis.py` definitions;
   RUN_SUMMARY/docs/10 to COMPLETED, 63.8 min, $1.91; update the model card's Evaluation section and change its amendment link to
   a commit permalink.
5. Lane 5: collect `results/job19/h5_localise.jsonl`; analysis per amendment (dASR table, top-k vs random, held-out test,
   cis_sep vs ASR Spearman); RUN_SUMMARY/docs/10 to COMPLETED, 69.2 min, $2.08; paper proposal for the Limitations line on H5.
6. Lane 6: collect `results/job20/{asr.jsonl,summary.json,directions.json,translations.json}` (hashes/flags/numbers only);
   RUN_SUMMARY/docs/10 to COMPLETED, 65.1 min, $1.95; analysis or paper proposal.
7. Lane 1 follow-up: `ablate_components` / `component_means` still hook before Gemma-2/3 post-norms (affects cis_circ on Gemma).
8. Lint: 23 pre-existing ruff errors in `tools/make_figures.py`, `tools/h6_layers.py`, `tools/judge8b_h2.py`,
   `tools/specificity_analysis.py` (36 at base `4ef5645`; merges added none).
9. Process: several lanes pushed two branch names for one commit and could not run the mandated `uv run` commands because of
   the shared uv cache lock; round 2 should use one branch per lane and rerun the uv commands when the lock is free.
10. Owner decision (lane 12): `kaggle/executed` is described as private in docs, but the GitHub repo is public.

## Graphify (second brain)

- Round 0, 2026-11-22 15:43 UTC: `graphify update .`: 64 of 200 files re-extracted; 2,411 nodes, 4,740 edges, 161 communities.
- Round 1, 2026-11-22: `graphify update .` on the merged integration branch (`60992eb`; code-only AST update, no LLM):
  66 of 236 scanned files changed and were re-extracted; graph rebuilt to 2,850 nodes, 5,645 edges, 203 communities
  (100 saved labels, 27 renamed by hub). 115 files (notebooks, config) have no extractor and are not in the graph.
- `graphify-out/` is never committed. The code-only update does not re-extract docs/papers; community labels may be stale until
  `graphify label` runs.

## Round 2

Polled 2026-11-22 17:59 UTC (`git fetch origin`; `hf jobs ps -a --namespace pxaLabs-lexsi`; `HfApi.inspect_job`, field
`durations.running_secs`). Branch heads below are the pushed heads at that time; lanes may push later than this poll.
This section records status only; it does not judge quality (the reviewers do).

### GPU spend, round 2

Cost = running_secs x $1.80/h. For a RUNNING job the committed amount is its recorded `--timeout` x $1.80/h (the most it
can cost); the accrued part is shown for information and is not added to "spent" until the job ends.

| Lane | Job | Job id | Flavor | `--timeout` (max) | Lane cap | Status | Running time (`running_secs`) | Cost |
|---|---|---|---|---|---|---|---|---|
| 10 | 21 (seed-8 lead time) | `6b03238a8d268c0fbd223666` | l40sx1 | 100m ($3.00) | $3.00 | RUNNING | 2566 s (42m46s) so far | accrued $1.28; **committed $3.00** |
| 11 | 22 (generation determinism) | `6b0325c8130a00b5aa80a77b` | l40sx1 | 50m ($1.50) | $1.50 | COMPLETED | 509 s (8m29s) | **$0.25** |
| 12 | 23 (Expanse) | none found | - | - | $4.50 | not launched at poll (no job23 in the namespace; no `wf/r2-lane12` branch pushed) | - | $0.00 |
| other lanes | none | - | - | - | $0 | - | - | $0.00 |

Totals: **spent $9.05** (round 1 $8.80 + Job 22 $0.25); **committed $3.00** (Job 21); **remaining $12.95** of $25.
Once Job 21 ends, spent = $9.05 + its final cost (at most $3.00).

Checks:
- Timeouts: Job 21 has run 2566 s of its 6000 s timeout; Job 22 ran 509 s of 3000 s. Neither exceeds its timeout. The
  timeouts come from the launch records (Job 21: lane 10 RUN_SUMMARY/docs/10, `--timeout 100m`; Job 22: lane 11
  RUN_SUMMARY/docs/10, `--timeout 50m`), because `inspect_job` does not expose the timeout.
- Amendment before launch: Job 21 created 17:16:58 UTC; amendment `e2fb4a1` committed 2026-11-22T17:13:59Z
  (`git log -1 --format=%cI`). Job 22 created 17:26:32 UTC; amendment `a625859` committed 2026-11-22T17:20:37Z. Both
  amendments precede their jobs. Git records commit time, not push time; the push-before-launch claim rests on the lanes'
  RUN_SUMMARY statements.
- Job 22 output `results/job22/determinism.jsonl` is in `purvansh01/safetypulse-jobs`; lane 11's records still say
  RUNNING (to update when collected). Job 21 has uploaded `results/job21/{dw.jsonl,train_log.json,executed/07_*.ipynb}`
  so far.
- No round-2 job ran outside the org namespace (`hf jobs ps -a --namespace pxaLabs-lexsi` lists Jobs 21 and 22 only as new).

### Round 2 lane table

One branch per lane was required. Lane 7 pushed its round-2 work onto the round-1 branch name instead of `wf/r2-lane7`.

| Lane | Item (from the branch's commits) | Status at poll | Branch / head | HF job (cost) | Repos |
|---|---|---|---|---|---|
| 1 | Backlog 7: component mean-ablation on Gemma-2/3 post-norm layers | pushed (test first, then fix) | `wf/r2-lane1` / `f19b203` | none | none |
| 2 | Backlog 2: collect Job 16, H6 test-split analysis, records, paper proposal | pushed | `wf/r2-lane2` / `cf7140a` | none (Job 16 was round 1) | none |
| 3 | Backlog 3: collect Job 17 (ShieldGemma), analysis, records, paper proposal | pushed | `wf/r2-lane3` / `5e91599` | none (Job 17 was round 1) | none |
| 4 | Backlog 4: collect Job 18, H3 2-step analysis, records, Hub card, paper proposal | pushed; card uploaded to the private repo (Hub commit `eccc903`) | `wf/r2-lane4` / `52ae95a` | none (Job 18 was round 1) | card update on `safetypulse-wf-h3fine-lora-alpaca` |
| 5 | Backlog 5: collect Job 19, H5 localisation analysis, records, paper proposal | pushed | `wf/r2-lane5` / `9f07db7` | none (Job 19 was round 1) | none |
| 6 | Backlog 6: collect Job 20, H4 analysis, records, paper proposal (`summary.json` held back by the release check) | pushed | `wf/r2-lane6` / `a0f5de2` | none (Job 20 was round 1) | none |
| 7 | Backlog 1: `git_sha` line parsing and exit status | pushed, **wrong branch name**: on `wf/r1-l07-core-audit` (merge of integration into local `r2-l07-local`), no `wf/r2-lane7` | `wf/r1-l07-core-audit` / `bba5494` | none | none |
| 8 | Backlog 8: ruff errors in tools/ cleared (outputs byte-identical per commits) | pushed | `wf/r2-lane8` / `4b3f58c` | none | none |
| 9 | Consolidated paper edit list (15 MUST/SHOULD edits as a patch) | pushed | `wf/r2-lane9` / `1f697c0` | none | none |
| 10 | Job 21: seed-8 replication of the 2-step lead-time run | pushed; job RUNNING | `wf/r2-lane10` / `3cd09bd` (amendment `e2fb4a1`) | Job 21, committed $3.00 | uploads `alpaca-s8/` to `safetypulse-wf-h3fine-lora-alpaca` |
| 11 | Job 22: generation determinism on 8 Llama-3.2-3B models; version table | pushed; job COMPLETED, records not yet updated | `wf/r2-lane11` / `7214790` (amendment `a625859`) | Job 22, $0.25 | none |
| 12 | Job 23 (Expanse) | **nothing pushed at poll; no job launched** | - | none | `safetypulse-wf-expanse-lora-alpaca` does not exist |
| 13 | Interval estimates for headline proportions and AUROCs (analysis only) | pushed | `wf/r2-lane13` / `66e1fe8` (amendment `5b41d6b`) | none | none |
| 14 | Exact integer-count unsafe rule (`tools/_rules.py`) used by analysis tools | pushed | `wf/r2-lane14` / `8825007` | none | none |
| 15 | - | **nothing pushed at poll** | - | none | none |
| 16 | Backlog 10 and records: kaggle/executed README (public), errata, owner-decision list, README test count | pushed | `wf/r2-lane16` / `4a8adc3` | none | none |
| 17 | calibrate/check: token-set overlap rejected; Qwen3 `enable_thinking` warning | pushed | `wf/r2-lane17` / `738ca1f` | none | none |
| 18 | ACL Responsible NLP checklist; Limitations/Ethics/Acknowledgements proposals | pushed | `wf/r2-lane18` / `99b8f59` | none | none |
| 19 | Release audit round 2; strict free-text rules for job outputs | pushed | `wf/r2-lane19` / `93e5838` | none | none |
| 20 | Tracker: this section, graphify update | pushed with this commit | `wf/r2-lane20` | none | none |

### Repos and collections register (round 2)

Checked read-only with `HfApi.repo_info` / `list_repo_files` / `list_collections` at 17:59 UTC. Nothing was added to a
collection and no visibility was changed.

| Item | Type | Created/changed by | `private` (verified) | Note |
|---|---|---|---|---|
| `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` | model | lane 4 (round 1); lane 4 card (round 2); Job 21 (lane 10) | True (head `f5d74ac0`) | `alpaca-s8/` present: 41 checkpoint folders (0, 2..60 by 2, 80..240 by 20, 250), 83 files, same count as `alpaca-s7/` |
| `pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca` | model | lane 12 (planned) | - | **not created** (`RepositoryNotFoundError`) |
| `purvansh01/safetypulse-jobs` | dataset (job outputs) | all jobs | True | new: `results/job21/` (partial), `results/job22/determinism.jsonl` |
| `pxaLabs-lexsi/safetypulse-workflow-models-private-6b030dd4c3a7647760ae6f16` | collection | tracker, round 0 | True | 1 item, unchanged; owner decision pending (see round 1) |
| `pxaLabs-lexsi/safetypulse-workflow-datasets-private-6b030dd671c703296d096ac3` | collection | tracker, round 0 | True | empty, unchanged; owner decision pending |

### Round-1 backlog, round-2 status

| # | Item | Round-2 owner | Status at poll |
|---|---|---|---|
| 1 | `git_sha` parsing (lane 7 revise) | lane 7 | fix pushed (`45c4ce1`, head `bba5494`) but on `wf/r1-l07-core-audit`, not `wf/r2-lane7`; reviewer to confirm test and uv runs |
| 2 | Collect Job 16, H6 test split | lane 2 | pushed (`cf7140a`) |
| 3 | Collect Job 17, third judge; library versions | lane 3 (collect), lane 11 (versions, Job 22) | pushed (`5e91599`); Job 17 records say versions not printed; Job 22 printed its versions, results to collect |
| 4 | Collect Job 18, H3 analysis, card | lane 4 | pushed (`52ae95a`), card on Hub `eccc903` |
| 5 | Collect Job 19, H5 analysis | lane 5 | pushed (`9f07db7`) |
| 6 | Collect Job 20, H4 | lane 6 | pushed (`a0f5de2`) |
| 7 | Gemma post-norm hooks in `ablate_components` / `component_means` | lane 1 | pushed (`f19b203`) |
| 8 | Pre-existing ruff errors in tools/ | lane 8 | pushed (`4b3f58c`) |
| 9 | Process: one branch per lane; uv lock | all lanes | 17 of 20 lanes (including lane 20) on one `wf/r2-laneN` name; lane 7 used the round-1 name; lanes 12 and 15 had not pushed at poll |
| 10 | `kaggle/executed` described as private while the repo is public | lane 16 (docs); owner | docs corrected and owner-decision list added (`66b5766`, head `4a8adc3`); the owner decision itself is still open |

Open for round 3: collect Job 21 (when it ends) and Job 22 into `kaggle/executed/12_hf_jobs/files/` and update their
records; lane 12 (Job 23) and lane 15 status; lane 7 to push under one round-2 name; the two private collections and the
public `kaggle/executed` folder await the owner's decision.

### Graphify, round 2

- 2026-11-22 ~17:42 UTC: `graphify update .` in the main checkout (on `stage2/wf-integration`; round-2 branches are not
  merged yet, so they are not in the graph). Code-only AST update, no LLM. 1 of 236 scanned files had changed and was
  re-extracted; "No code-graph topology changes detected; outputs left untouched". Graph: **2,850 nodes, 5,645 edges,
  203 communities** (counted from `graphify-out/graph.json`), same as after round 1. 115 of 351 files (notebooks, config)
  have no extractor. Warning printed: skill 0.9.105 vs package 0.9.103 (not upgraded by this lane).
- `graphify-out/` is not committed. The graph should be updated again after round 2 is merged into the integration branch.

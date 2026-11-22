# Multi-agent workflow ledger (started 2026-11-22)

Kept by the tracker agent. Budget: HF GPU spend capped at **$25** (org namespace `pxaLabs-lexsi`, L40S `l40sx1` at $1.80/h).
Rules: every experiment has a dated pre-registration amendment in `docs/prereg_amendments/` committed and pushed before launch
and is labelled exploratory; pre-registered verdicts and frozen thresholds (WARN < 0.8098, FAIL < 0.6196) never change; new HF
repos are private until the user approves; paper edits are proposals only (`docs/report/workflow/paper_proposals/`); no harm
adapters, ft_harm data, harmdense artifacts or harmful answer text are ever published.

## GPU spend (running total)

**Spent: $8.80 of $25 (35%). Committed by running jobs: $0.00. Remaining: $16.20.**

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

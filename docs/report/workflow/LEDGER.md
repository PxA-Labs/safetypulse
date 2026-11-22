# Multi-agent workflow ledger (started 2026-11-22)

Kept by the tracker agent. Budget: HF GPU spend capped at **$25** (org namespace `pxaLabs-lexsi`, L40S at $1.80/h).
Rules: every experiment has a dated pre-registration amendment in `docs/prereg_amendments/` committed before launch and is
labelled exploratory; new HF repos are private until the user approves; paper edits are proposals only
(`docs/report/workflow/paper_proposals/`); no harm adapters, ft_harm data or harmful answer text are ever published.

| Round | Lane | Item | Status | Branch / commit | HF job (cost) | HF repo |
|---|---|---|---|---|---|---|
| 1 | 1 | Gemma ablation fix + Job 15 | done: Job 15 completed; fixed ablation selects Gemma-3-4B layer 9 (old 11), all 6 fine-tunes FAIL there, AUROC 0.625 | `wf/r1-l01-gemma-ablation` / `2d8fefe` | Job 15 `6b030376130a00b5aa8063fc`, l40sx1, COMPLETED, 4m34s (~$0.14); timeout 80m, cap $2.50 | none |
| 1 | 2 | H6 test split Job 16 | job finished on HF, results not yet collected into git (branch records the launch only) | `wf/r1-l02-h6-testsplit` = `wf/r1-lane2` / `5d9440e` | Job 16 `6b0301c98d268c0fbd21f9c5`, l40sx1, COMPLETED, 44m38s (~$1.34); timeout 110m, cap $3.50 | none (results to private `purvansh01/safetypulse-jobs` `results/job16/`) |
| 1 | 3 | Third judge Job 17 | job running; early log committed (benign check passes; first model's 1B flags differ from Jobs 9/14) | `wf/r1-lane3` / `e2d8c96` | Job 17 `6b0303a0130a00b5aa80649c`, l40sx1, RUNNING since 15:00:48 UTC; timeout 80m, cap $2.50 | none |
| 1 | 4 | 2-step lead Job 18 | job running; records committed | `wf/r1-lane4` / `8376b37` | Job 18 `6b03028b8d268c0fbd21fa6f`, l40sx1, RUNNING since 14:56:11 UTC; timeout 130m, cap $4.00 | `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (private model; in private models collection) |
| 1 | 5 | H5 localisation Job 19 | job running; records committed | `wf/r1-lane5` / `66bd5e2` | Job 19 `6b0305a18d268c0fbd21fffc`, l40sx1, RUNNING since 15:09:21 UTC; timeout 100m, cap $3.00 | none |
| 1 | 6 | H4 multilingual Job 20 | job running; launch check committed (first checkpoint scored, timing inside timeout) | `wf/r1-lane6` / `38a4b74` | Job 20 `6b0303b18d268c0fbd21fdb7`, l40sx1, RUNNING since 15:01:05 UTC; timeout 115m, cap $3.50 | none |
| 1 | 7 | Core package audit | branch pushed (last commit: provenance `git_sha` from the package's own checkout) | `wf/r1-l07-core-audit` / `a30c142` | none ($0) | none |
| 1 | 8 | Audit/repair audit | branch pushed (last commit: `SafetyPulseCallback` writes check files on the main process only) | `wf/r1-l08-audit-repair` = `wf/r1-lane8` / `959a539` | none ($0) | none |
| 1 | 9 | Number audit | branch pushed (paper numbers and App. A SHAs traced; 1dcc6d3 not on main, two wording mismatches) | `wf/r1-l09-number-audit` = `wf/r1-lane9` / `732d263` | none ($0) | none |
| 1 | 10 | Citations | branch pushed (39 bib entries verified; proposed attribution edits) | `wf/r1-lane10` / `cd55b8c` | none ($0) | none |
| 1 | 11 | Reproducibility | branch pushed (figure PDFs written without CreationDate) | `wf/r1-lane11` / `a107e06` | none ($0) | none |
| 1 | 12 | Records/docs | branch pushed (CHANGELOG and executed README for Stage 2 jobs) | `wf/r1-lane12` / `3d59847` | none ($0) | none |
| 1 | 13 | Hub cards | in progress (no pushed branch at final poll) | - | none ($0) | - |
| 1 | 14 | Label noise | branch pushed (amendment `783ba8d`; bootstrap of unsafe labels, verdicts unchanged) | `wf/r1-lane14` / `102e339` | none ($0) | none |
| 1 | 15 | Threshold transfer | branch pushed (amendment committed before the analysis; analysis commit not yet seen) | `wf/r1-lane15` / `7f9fa93` | none ($0) | none |
| 1 | 16 | Number tests | in progress (no pushed branch at final poll) | - | none ($0) | - |
| 1 | 17 | Two-stage alarm | branch pushed (paper proposal from `docs/report/two_stage_alarm.json`) | `wf/r1-lane17` / `2e6a085` | none ($0) | none |
| 1 | 18 | Format | in progress (no pushed branch at final poll) | - | none ($0) | - |
| 1 | 19 | Release policy | in progress (no pushed branch at final poll) | - | none ($0) | - |
| 1 | 20 | Tracker | done: ledger, private collections, graphify update | `wf/r1-l20-tracker` / this commit | none ($0) | two private collections (below) |

Final status poll: 2026-11-22 15:46 UTC (`git fetch origin` + `git for-each-ref refs/remotes/origin/wf/r1-*`;
`hf jobs ps -a --namespace pxaLabs-lexsi` and `HfApi.inspect_job`). Lanes 1, 2, 8 and 9 pushed two branch names; where both
exist they point at the same commit. "Branch pushed" means the worker's branch exists; whether its work passes review is
for the reviewers, not this ledger.

## Budget

Cap: **$25** of HF GPU (org `pxaLabs-lexsi`, L40S `l40sx1` at $1.80/h). Cost = runtime x $1.80/h.

| Lane | Job | Round-1 cap | `--timeout` (max cost) | Runtime at poll | Spend so far |
|---|---|---|---|---|---|
| 1 | 15 | $2.50 | 80m ($2.40) | 4m34s, completed | $0.14 |
| 2 | 16 | $3.50 | 110m ($3.30) | 44m38s, completed | $1.34 |
| 3 | 17 | $2.50 | 80m ($2.40) | ~45m, running | ~$1.36 |
| 4 | 18 | $4.00 | 130m ($3.90) | ~50m, running | ~$1.50 |
| 5 | 19 | $3.00 | 100m ($3.00) | ~37m, running | ~$1.11 |
| 6 | 20 | $3.50 | 115m ($3.45) | ~45m, running | ~$1.35 |
| 7-20 | none | $0 | - | - | $0 |
| **Total** | | **$19.00** | **$18.45** | | **~$6.80** |

- Allocated round-1 caps: $19.00 of $25 (remaining unallocated: $6.00).
- Worst case for round 1 if every running job hits its timeout: $18.45, inside the $19.00 allocation.
- Runtimes of running jobs are measured from the job's `created` time to the poll, so they include any queue time and slightly
  overstate GPU time. Final spend is recomputed from `hf jobs ps -a` runtimes when the jobs finish.
- Earlier jobs on the namespace (Job 9 test/llama32-3b/qwen3-4b, 2026-11-21) predate the workflow and are not counted here.

## HF repos and private collections

New repos created by the workflow (all private, `pxaLabs-lexsi`, names start with `safetypulse-wf-`):

| Repo | Type | Lane | Amendment | Collection |
|---|---|---|---|---|
| `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` | model | 4 (Job 18) | `docs/prereg_amendments/2026-11-22-l04-fine-lead.md` (branch `wf/r1-lane4`) | models |

No `safetypulse-wf-*` dataset repos existed at the poll. Job results go to the private dataset `purvansh01/safetypulse-jobs`,
which is not in the org namespace and is not added to any collection.

Private collections (created 2026-11-22 with `HfApi.create_collection(..., namespace='pxaLabs-lexsi', private=True,
exists_ok=True)`; both confirmed `private: True`). They stay private until the user approves publication:

- Models: `pxaLabs-lexsi/safetypulse-workflow-models-private-6b030dd4c3a7647760ae6f16`
  ("SafetyPulse workflow models (private)"), 1 item, with a note naming lane 4 and its amendment.
- Datasets: `pxaLabs-lexsi/safetypulse-workflow-datasets-private-6b030dd671c703296d096ac3`
  ("SafetyPulse workflow datasets (private)"), 0 items.

Rule applied when adding: only private `safetypulse-wf-*` repos; any name containing harm/withheld/ft_harm/harmdense is
skipped (none matched).

## Graphify (second brain)

- 2026-11-22 15:43 UTC: `graphify update .` on the main checkout (code-only incremental update, no LLM): 64 of 200 scanned
  files changed and were re-extracted; graph rebuilt to 2,411 nodes, 4,740 edges, 161 communities. `graphify-out/` stays
  uncommitted.
- Check query `graphify query "which files implement cis_sep scoring and thresholds"` succeeded and pointed to
  `safetypulse/cis.py` (`projection_separation`, `compute_cis`, `cis_status`, `base_reference`) and
  `safetypulse/check.py` (`scoring_mode`).
- Limits: the main checkout is on `main`, so the graph does not yet include round-1 branch changes; the code-only update does
  not re-extract docs/papers (that needs the skill's `/graphify --update` semantic pass), and 161 communities vs 102 saved
  labels means some community names are stale until `graphify label` runs.

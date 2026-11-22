# Pre-registration amendments: index and timing audit

This folder holds the **exploratory** amendments written during the 2026-11-22 workflow, after the confirmatory plan in
`docs/13_PREREGISTRATION.md` (§1–§3h) was frozen. The convention:

- One file per experiment or analysis, named `2026-11-22-<lane>-<slug>.md` (`<lane>` = `l` + two-digit lane number,
  e.g. `l04`; `<slug>` = a short kebab-case name). Each file states what will be run, why, and exactly what will be
  reported.
- The file is committed **and pushed** on the lane's branch **before** the HF Job it governs is created, or, for
  analysis-only amendments, before the first commit of the tool or report that carries out the analysis. The job record
  (`kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`, `docs/10_KAGGLE_AND_HF_RESOURCES.md`) or the report cites the
  amendment's commit.
- Amendment files are frozen records: they are not edited after the commit that adds them. Deviations found later are
  stated in the report (see l17 below), not by rewriting the amendment.
- `2026-11-22-l16-erratum-dev-prompts.md` is an **erratum**, not an experiment: doc 13 line 96 says "100 dev HarmBench
  prompts per checkpoint", but notebook 08 caps at `N_ASR = 100` and the dev split has 78 prompts, so every logged ASR
  used 78. No result changes.

**No amendment in this folder changes H1, H2, H3 or H6, their pre-registered verdicts (H1 confirmed, H2 confirmed, H3
refuted, H6 refuted) or the frozen thresholds (`cis_sep` WARN < 0.8098, FAIL < 0.6196).** Each of the 13 files states in
its header that it changes no pre-registered verdict and no frozen threshold (checked by reading each file). All 13 are marked exploratory (or erratum); none is confirmatory.
The pre-registered tests themselves are in doc 13 §1–§3d; doc 13 §3e–§3h are earlier exploratory amendments inside
doc 13 and are not indexed here.

## How the table was built (2026-11-23)

- Files: every path ever added under `docs/prereg_amendments/` on any ref
  (`git log --all --diff-filter=A --name-only -- docs/prereg_amendments/`, after `git fetch origin --prune`). 12 are on
  `origin/stage2/wf-integration`; `l13-intervals` is only on `origin/wf/r2-lane13`. At the time of writing, the round-3
  branches on origin (`wf/r3-lane1`, `-3`, `-6`, `-7`, `-8`) add no amendment; lanes 10 and 12 had not pushed a round-3
  branch yet. Each file has exactly one adding commit.
- Commit time: committer date of that commit (`%cI`), converted to UTC.
- Push time: first time a branch on GitHub pointed at a commit containing the amendment, from the repository activity
  API (`gh api repos/PxA-Labs/safetypulse/activity`, `branch_creation` / `push` events, ancestry checked with
  `git merge-base --is-ancestor`). "Pushed" = contained in an origin ref now (`git branch -r --contains`).
- Job creation: `created_at` from `hf jobs inspect --namespace pxaLabs-lexsi <id>` (read-only). Job ids from
  `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`.
- Δ = amendment time − job `created_at` (or − first commit of the governed tool/report); negative = amendment first.
- Cost: `running_secs` from `hf jobs inspect` × $1.80/h (L40S); scheduling time not included. Records may state a
  slightly different figure (e.g. "~$2.08").

## A. Amendments tied to an HF Job

All times UTC, 2026-11-22.

| File | Lane | Title | Amendment commit (time) | First push | Job id | Job created_at | Δ commit − created | Δ push − created |
|---|---|---|---|---|---|---|---|---|
| `2026-11-22-l01-gemma-ablation.md` | 1 | Gemma-3 layer selection, post-norm-aware ablation (Job 15) | `353808415907` (14:59:24) | 14:59:37, `wf/r1-l01-gemma-ablation` @ `7bc40c4` | `6b030376130a00b5aa8063fc` | 15:00:06.313 | −42.3 s | −29.3 s |
| `2026-11-22-l02-h6-testsplit.md` | 2 | H6 labels on the 119-prompt test split, both judges (Job 16) | `90350d1ccae6` (14:45:05) | 14:45:09, `wf/r1-lane2` @ `90350d1` | `6b0301c98d268c0fbd21f9c5` | 14:52:57.682 | −472.7 s (−7 min 53 s) | −468.7 s |
| `2026-11-22-l03-third-judge.md` | 3 | Third judge, ShieldGemma 9B, on H6 answers (Job 17) | `baeb43e16f84` (14:56:43) | 14:56:45, `wf/r1-lane3` @ `baeb43e` | `6b0303a0130a00b5aa80649c` | 15:00:48.398 | −245.4 s (−4 min 05 s) | −243.4 s |
| `2026-11-22-l04-fine-lead.md` | 4 | H3 lead time at 2-step spacing, seed 7 (Job 18) | `e3f8d553207f` (14:54:43) | 14:54:47, `wf/r1-lane4` @ `94ad123` | `6b03028b8d268c0fbd21fa6f` | 14:56:11.783 | −88.8 s | −84.8 s |
| `2026-11-22-l05-h5-localise.md` | 5 | H5, block-wise restore (Job 19) | `3fc67aca0dfa` (15:08:26) | 15:08:29, `wf/r1-lane5` @ `3fc67ac` | `6b0305a18d268c0fbd21fffc` | 15:09:21.306 | −55.3 s | −52.3 s |
| `2026-11-22-l06-h4-multilingual.md` | 6 | H4, English alarm vs hi/es/fr drift (Job 20) | `edda1ddf6ba8` (15:00:42) | 15:00:47, `wf/r1-lane6` @ `ef62566` | `6b0303b18d268c0fbd21fdb7` | 15:01:05.663 | −23.7 s | −18.7 s |
| `2026-11-22-l10-fine-lead-seed8.md` | 10 | Seed-8 replication of the 2-step lead-time run (Job 21) | `e2fb4a14afc6` (17:13:59) | 17:14:03, `wf/r2-lane10` @ `e2fb4a1` | `6b03238a8d268c0fbd223666` | 17:16:58.224 | −179.2 s (−2 min 59 s) | −175.2 s |
| `2026-11-22-l11-determinism.md` | 11 | Generation determinism of the H6 answers (Job 22) | `a625859e73ba` (17:20:37) | 17:20:45, `wf/r2-lane11` @ `a625859` | `6b0325c8130a00b5aa80a77b` | 17:26:32.705 | −355.7 s (−5 min 56 s) | −347.7 s |

| File | Status | Outcome (copied from the record or report) | Report | Paper proposal | Cost |
|---|---|---|---|---|---|
| l01-gemma-ablation | exploratory | "with the fixed ablation layer selection picks **layer 9 of 34** (old: 11) ... The partial ablation does not explain Gemma-3's early layer: fixing it moves l* earlier, not later. The pre-registered H6 verdict (refuted) is unchanged." | `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` § Job 15 (no separate report) | `docs/report/workflow/paper_proposals/l01-gemma-ablation.md` | 274 s, $0.14 |
| l02-h6-testsplit | exploratory | "H6 under the §2 rule is refuted for every judge × prompt set: unsafe drifted models flagged 11/12 (1B test, ...), FAIL on safe recovered models 5/11 (1B test) ... against 3/8 pre-registered." | `docs/report/h6_testsplit.md` | `docs/report/workflow/paper_proposals/l02-h6-testsplit.md` | 2678 s, $1.34 |
| l03-third-judge | exploratory | "ShieldGemma vs 1B on this run's answers, 2,730 judgements: agreement 0.735, Cohen's kappa 0.259 ... ShieldGemma refuted, 9/10, FAIL on 8/15 ... The pre-registered H6 verdict is unchanged." | `docs/report/third_judge_h6.md` | `docs/report/workflow/paper_proposals/l03-third-judge.md` | 2761 s, $1.38 |
| l04-fine-lead | exploratory, one seed | "First alarms: ‖ΔW‖ > 3.344 at step 12 (lead +88), `cis_sep` WARN at 20 (+80) ... At 2-step spacing ‖ΔW‖ alarmed 8 steps before `cis_sep`" | `docs/report/h3_fine_lead.md` (on `origin/wf/r2-lane4`, not yet in integration) | `docs/report/workflow/paper_proposals/l04-fine-lead.md` (`origin/wf/r2-lane4`) | 3828 s, $1.91 |
| l05-h5-localise | exploratory, no verdict | "Reading: no evidence that the drift sits in a few blocks; restoring ~29% of the LoRA energy (top 16 or random sets of similar energy) brings ASR near base on this one checkpoint." | `docs/report/h5_localise.md` | `docs/report/workflow/paper_proposals/l05-h5-localise.md` | 4154 s, $2.08 |
| l06-h4-multilingual | exploratory, no pass mark | "the English alarm (WARN or FAIL) fires on all 15 and recall on unsafe checkpoints is 1.00 in every language ... Because the alarm fires everywhere, these runs do not test whether it would miss drift confined to another language." | `docs/report/h4_multilingual.md` | `docs/report/workflow/paper_proposals/l06-h4-multilingual.md` | 3908 s, $1.95 |
| l10-fine-lead-seed8 | exploratory | Not yet collected: the integration record says "RUNNING ... results collected next round"; `hf jobs inspect` shows COMPLETED (finished 18:20:45). | none yet | none yet | 3822 s, $1.91 |
| l11-determinism | exploratory | Not yet collected: the integration record says "Status: **RUNNING** ... Results, runtime and cost are collected next round"; `hf jobs inspect` shows COMPLETED (finished 17:36:16). | `docs/report/workflow/determinism_versions.md` (pre-job version table only) | none yet | 509 s, $0.25 |

## B. Analysis-only amendments (no GPU) and the erratum

Δ compares the amendment commit with the first commit of the tool/report it governs (both committer times, UTC).

| File | Lane | Title | Amendment commit (time) | First push | Governed tool/report, first commit (time) | Δ |
|---|---|---|---|---|---|---|
| `2026-11-22-l13-intervals.md` | 13 | Interval estimates for every headline proportion and AUROC | `5b41d6bfe219` (17:21:50) | 17:22:03, `wf/r2-lane13` @ `5b41d6b` | `tools/intervals.py`, `docs/report/intervals.md`: `3aebf2a` (17:35:21) | −811 s (−13 min 31 s) |
| `2026-11-22-l14-label-noise.md` | 14 | Label noise from the 78-prompt sample; verdict stability | `783ba8d9cbc8` (15:11:00) | 15:11:03, `wf/r1-lane14` @ `783ba8d` | `tools/label_noise.py`, `docs/report/label_noise.md`: `102e339` (15:36:18) | −1518 s (−25 min 18 s) |
| `2026-11-22-l15-threshold-transfer.md` | 15 | Threshold transfer across families, sample-size curve, label-free rule | `7f9fa93818a6` (15:28:10) | 15:28:28, `wf/r1-lane15` @ `7f9fa93` | `tools/h6_threshold_transfer.py`, `docs/report/h6_threshold_transfer.md`: `f96f67a` (15:50:36) | −1346 s (−22 min 26 s) |
| `2026-11-22-l16-erratum-dev-prompts.md` | 16 | Erratum: 78, not 100, dev prompts per checkpoint ASR | `7d12b2823213` (17:27:35) | 17:43:51, `wf/r2-lane16` @ `4a8adc3` | n/a (records only; the same commit corrects `kaggle/executed/08_monitor_checkpoints/RUN_SUMMARY.md`) | n/a |
| `2026-11-22-l17-two-stage.md` | 17 | Two-stage alarm: `cis_sep` screen confirmed by the spot-check | `898ac5a9da75` (15:28:39) | 15:29:06, `wf/r1-lane17` @ `898ac5a` | `tools/two_stage_alarm.py`, `docs/report/two_stage_alarm.md`: `99ea834` (15:39:07) | −628 s (−10 min 28 s) |

| File | Status | Outcome (copied from the report) | Report | Paper proposal | Cost |
|---|---|---|---|---|---|
| l13-intervals | exploratory, analysis only | "Every point value below was recomputed from the raw rows and asserted equal to the published number before any interval was computed." e.g. "H1 recall, unsafe checkpoints (cis_sep) \| 63/63 \| 1.00 \| [0.94, 1.00]" | `docs/report/intervals.md` (on `origin/wf/r2-lane13` only) | `docs/report/workflow/paper_proposals/l13-intervals.md` (`origin/wf/r2-lane13`) | $0 |
| l14-label-noise | exploratory, no new runs | "H6 · 1B \| 11/12 unsafe drifted flagged; FAIL on 3/8 safe recovered \| ... \| refuted \| 0.000 \| refuted 2000/2000"; H3 · 1B "refuted 1911/2000, untested 89/2000" | `docs/report/label_noise.md` | `docs/report/workflow/paper_proposals/l14-label-noise.md` | $0 |
| l15-threshold-transfer | exploratory, post hoc (written after the H6 result, before this analysis) | "`cis_sep` \| LOFO, f1 rule \| 15/19 ... \| 4/12 ... \| 18/19 ... \| 8/12" vs "frozen (Global dev) \| 16/19 ... \| 4/12 ... \| 18/19 ... \| 10/12" | `docs/report/h6_threshold_transfer.md` | `docs/report/workflow/paper_proposals/l15-threshold-transfer.md` | $0 |
| l16-erratum-dev-prompts | erratum | "None on any result. The prompts used are exactly the ones doc 13 refers to ...; only the count in the sentence was wrong." | the erratum itself | `docs/report/workflow/paper_proposals/l09-number-audit.md` (item N1, which found it) | $0 |
| l17-two-stage | exploratory, post hoc | "A two-stage alarm therefore trades the early warning that is `cis_sep`'s advantage for the spot-check's specificity; it does not keep both." | `docs/report/two_stage_alarm.md` | `docs/report/workflow/paper_proposals/l17-two-stage.md` | $0 |

## Violations and flags

- **Amendment committed after its job / analysis: none.** All 8 job amendments have negative Δ for both commit time
  and first push time (smallest margin: l06, pushed 18.7 s before Job 20 was created; l01, 29.3 s before Job 15). All 4
  analysis amendments precede the first commit of their tool and report by 10 to 25 minutes.
- **Unpushed amendment: none.** All 13 adding commits are on an origin ref (`git branch -r --contains`).
- **Job without an amendment: none since the workflow began.** `hf jobs ps -a --namespace pxaLabs-lexsi` lists 11 jobs:
  the 8 above (Jobs 15–22, all created 2026-11-22) and three Job 9 jobs from 2026-11-21 (`job9-test-llama32-3b`,
  `job9-llama32-3b`, `job9-qwen3-4b`), which are the pre-registered H6 run governed by doc 13 §3d. The personal
  namespace (`hf jobs ps -a`) has no job after 2026-11-21 (latest: Job 14 resume, 2026-11-21 18:46:06); its earlier jobs
  are governed by doc 13 §3a–§3g.
- Not violations, but open:
  - `l13-intervals` (amendment, tool, report, proposal) exists only on `origin/wf/r2-lane13`; it is not merged into
    `stage2/wf-integration`. Likewise the Job 18 result and report (`origin/wf/r2-lane4`).
  - Jobs 21 and 22 are COMPLETED on HF, but their records on integration still say RUNNING; outcomes, reports and
    proposals are pending collection.
  - l17: the report states a deviation from the amendment's wording (`spot_refusal < 0.96` written, `<= 0.96` used to
    reproduce Table 2, which the amendment also required). Disclosed in `docs/report/two_stage_alarm.md`.
  - l15 is post hoc by its own statement (written after the H6 result), though before its analysis.
  - Push times come from GitHub's activity API; commit times can be set by the committer, so the push time is the
    stronger evidence. Both are negative for every row.

Counts: 13 amendment files (8 job-tied, 4 analysis-only, 1 erratum); 8 HF jobs matched; 0 violations.

## Addendum: final check after round 3 (2026-11-23)

Rows the round-3 index could not include, because they were added after it was built. Commit times come from `git log`,
job times from `HfApi.inspect_job` (namespace `pxaLabs-lexsi`). The workers reported each amendment as pushed before launch,
but the GitHub first-push times were not re-checked here.

| File | Lane | Amendment commit (UTC) | Job(s) | Job created_at (UTC) | Δ commit − created | Running time, cost |
|---|---|---|---|---|---|---|
| `2026-11-22-l10-fine-lead-seed9.md` | 10 | `eacb3a3` (19:47:07) | Job 24 `6b03471f8d268c0fbd228a21` | 19:48:47.7 | −100.7 s | 3279 s, $1.64 |
| `2026-11-22-l12-aya-expanse.md` | 12 | `e59ca1b` (19:50:38) | Job 23a, first attempt, `6b0348a6130a00b5aa8102e9` (ERROR: script bug, before any download) | 19:55:18.7 | −280.7 s | 124 s, $0.06 |
| | | | Job 23a `6b0350768d268c0fbd22a37a` | 20:28:38.9 | −2280.9 s | 1142 s, $0.57 |
| | | | Job 23b `6b03560f130a00b5aa812202` | 20:52:31.7 | −3713.7 s | 792 s, $0.40 |

Rows l10-fine-lead-seed8 (Job 21) and l11-determinism (Job 22) above say "not yet collected". Both have since been
collected: see `docs/report/h3_fine_lead_seeds.md`, `docs/report/determinism_h6.md` and RUN_SUMMARY Jobs 21 and 22.

Updated counts: 15 amendment files (11 job-tied, 3 analysis-only, 1 erratum); 11 HF jobs matched, plus 1 failed attempt.
Every job's amendment was committed before the job was created; 0 violations.

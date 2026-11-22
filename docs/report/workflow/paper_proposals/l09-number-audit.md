# Paper proposal l09 (kind: audit): every number and commit SHA in `paper/main.tex` traced to its source

Lane 9, round 1, 2026-11-22. Audited: `paper/main.tex` and `paper/tables/h6_models.tex` as of
`origin/stage2/wf-integration` @ `4ef5645` (same paper text as `origin/main` @ `15d359a`). No run was launched; nothing
in the paper was edited. Statuses: **OK** (matches the source after rounding to the printed precision), **ROUNDING**
(matches only loosely; wording is fine), **MISMATCH** (the text does not match the source), **UNSOURCED** (no file in the
repo gives the value). Line numbers `main.tex:N` refer to the audited file.

## Summary

- 4 rows need a change (one is a SHA that is not on `main`); 2 rows are optional precision fixes. Everything else (about
  170 numbers, all 35 rows of Table 3 and 10 of the 11 App. A SHAs) traces to a file or was recomputed from raw logs.
- Recomputed from raw rows (not just read from a report): the ‖ΔW‖ baseline for H1/H2 (`stage2_dw.jsonl` + monitor
  logs), the H6 per-model table, the Gemma-free counts, the 15 drifted/recovered pairs, the "within 3 points" count,
  the Job 13 vs Job 9 padding difference and the H6 cost medians. The scripts used are shown at the end.

## Proposed edits (non-OK rows)

| # | Location | Status | Old text -> new text | Source |
|---|---|---|---|---|
| E1 | App. A, `main.tex:326` | MISMATCH | `\texttt{1dcc6d3}` -> `\texttt{15a248e}` | `1dcc6d3` is not on `origin/main` (only on `origin/stage2/review-followups`, `origin/stage2/wf-integration` and `wf/*` branches). It was rebase-merged as `15a248e` (same subject, same author date 2026-11-22 19:52:18 +0530; `git diff 1dcc6d3 15a248e` is empty, i.e. identical tree). Same pattern as 725551d/772ac17. |
| E2 | Results, "why safe recovered models fail", `main.tex:246` | MISMATCH | `\cissep{} rose above WARN in 2.` -> `\cissep{} ended above WARN in 2 (and in one of them, Gemma-3 Dolly, it was already above before recovery).` | Job 9 scores: recovered models with `cis_sep` >= 0.8098 are gemma3-4b-dolly-dpo-recover 0.907 (its drifted model 0.910, so it fell by 0.003) and llama31-8b-gsm8k-dpo-recover 0.836 (drifted 0.727). Only one actually *rose* above WARN. `kaggle/executed/12_hf_jobs/files/job9_*__scores.jsonl`; `docs/report/h6_safetune.md:43,46,56,61`. |
| E3 | Exploratory: thresholds, `main.tex:229` | MISMATCH | `so with 6--11 models per family we could not set reliable ones.` -> `so with 6--10 fine-tuned models per family we could not set reliable ones.` | The LODO analysis uses fine-tuned models only (`tools/h6_lodo.py:29` drops the aligned model): 10 / 6 / 6 / 9 per family (`docs/report/h6_lodo.md:9-12`, n_unsafe+n_safe in `h6_lodo.json`). With the aligned model the counts are 11 / 7 / 7 / 10, i.e. 7--11; neither reading gives 6--11. |
| E4 | Limitations, `main.tex:285` | MISMATCH | `Each test uses 3--5 runs or 6--11 models per family;` -> `Each test uses 3--5 runs or 6--10 fine-tuned models per family;` | As E3 (H6 detection is scored on fine-tuned models, Table 3: 10, 6, 6, 9). |
| E5 (optional) | App. A, `main.tex:327-329` | ROUNDING (wording) | `One H2 monitoring session hit a 1.5-hour time guard after 25 of 26 checkpoints; the last checkpoint was scored in a resumed session (amendment in \texttt{1cb8ea0}, before the analysis).` -> `One H2 monitoring session hit a 1.5-hour time guard after 25 of 26 checkpoints; the last checkpoint was scored in a resumed session (amendment dated before the analysis, committed together with the H2 results in \texttt{1cb8ea0}).` | `git show --stat 1cb8ea0`: the 3-line amendment in `docs/13_PREREGISTRATION.md` is in the same commit as `docs/report/h2_lists.{md,json}` and `tools/h2_analysis.py`, so the commit itself cannot show "before the analysis"; only the amendment's own text ("Amendment (2026-11-21, before any H2 analysis)") says so. A reviewer checking the SHA would see this. |
| E6 (optional) | App. A, `main.tex:325-326` | ROUNDING (provenance) | `\texttt{8e3d75b} (all 2026-11-21, each before its run);` -> `\texttt{8e3d75b} (all 2026-11-21, each before its run; 50802b3 and 8e3d75b were rebase-merged, so their committer dates are later, and the pre-merge commits 7f9e6b5 and 82ad2d6 carry the original times);` | `git log --format='%h %ad %cd'`: 50802b3 author 11-21 18:50:36, committer 11-22 00:00:34; 8e3d75b author 11-21 23:35:11, committer 11-22 01:44:44. Runs: Job 13 created 2026-11-21T13:21:55Z = 18:51:55 IST; Job 14 created 18:06:02Z = 23:36:02 IST (`hf jobs inspect`). Originals 7f9e6b5 (committer 18:50:36) and 82ad2d6 (committer 23:35:11) are on `origin/stage2/audit-fixes`. Authors' call whether the space is worth it; the claim "each before its run" holds by author date. |

## Commit SHAs in App. A (`main.tex:320-329`)

Checked with `git branch -r --contains <sha>` and `git show --stat <sha>`; job creation times from `hf jobs inspect`
(UTC converted to IST, +05:30).

| SHA | Paper says | On origin/main | Author date | Touches doc 13 with | Run it precedes | Status |
|---|---|---|---|---|---|---|
| 2a9141f | H1--H3 and baselines, 2026-11-13 | yes | 2026-11-13 14:05 | H1/H2/H3 predictions, thresholds 0.6196/0.8098, baselines (‖ΔW‖, spot-check, every checkpoint), 8B as second judge (§1 table) | all Stage 2 runs | OK |
| 6c23aa0 | H1 run notes, 2026-11-19 | yes | 2026-11-19 23:26 | "## 3a. Implementation notes for H1 (2026-11-19, ...)" | H1 runs | OK |
| abeb85a | H3 notes, 2026-11-20 | yes | 2026-11-20 12:45 | "## 3b. Implementation notes for H3" | H3 runs | OK |
| 37543c0 | H2 notes, 2026-11-20 | yes | 2026-11-20 22:16 | "## 3c. Implementation notes for H2" | H2 runs | OK |
| f93dc83 | H6 plan | yes | 2026-11-21 12:50 | "**H6 · Third-party drift ...** (amendment 2026-11-21 ...)" | Job 9 (13:16 IST) | OK |
| e602e03 | amendment before H6 runs (bf16 on L40S, Qwen non-thinking, Gemma loader, billing) | yes | 2026-11-21 13:15:44 | Billing, Precision (bf16 on L40S), Qwen3 template (`enable_thinking=False`), Gemma-3 decoder-layer lookup: all four items present | Job 9 families created 07:47:39Z = 13:17 IST; `job9_jobs.json` pins this commit | OK |
| f079239 | exploratory follow-ups after H6 | yes | 2026-11-21 14:10:22 | "## 3e. Exploratory follow-ups after H6" (H2 8B, threshold-free, layer diagnosis) | Jobs 10/11 created 08:41Z = 14:11 IST | OK |
| 50802b3 | specificity control | yes (rebase-merged; committer date 11-22 00:00) | 2026-11-21 18:50:36 | "## 3f. Specificity control" | Job 13 created 13:21:55Z = 18:51 IST | OK (see E6) |
| 8e3d75b | H6 re-judged with 8B | yes (rebase-merged; committer date 11-22 01:44) | 2026-11-21 23:35:11 | "## 3g. H6 with the second judge" | Job 14 created 18:06:02Z = 23:36 IST | OK (see E6) |
| 1dcc6d3 | review follow-ups, 2026-11-22 | **no** | 2026-11-22 19:52:18 | "## 3h. Follow-ups to an internal review" | analysis commit 6ab84a7 / main 15d359a (19:56) | **MISMATCH -> E1** (main: 15a248e) |
| 1cb8ea0 | H2 s3 resume amendment, before the analysis | yes | 2026-11-21 02:26 | 3-line "Amendment (2026-11-21, before any H2 analysis)" | same commit as the H2 analysis | ROUNDING -> E5 |

App. A also says "Llama Guard 3 8B is pre-registered as the second judge (§1 of the plan)": `docs/13_PREREGISTRATION.md:19`
(present already in 2a9141f) -- OK. "H4 ... H5 ... pre-registered as exploratory and not run": doc 13 §1 items 4--5, line 245 -- OK.

## Numbers, by section

### Abstract (`main.tex:33-43`)

| Quoted | Paper | Source (file:line) | Source value | Status |
|---|---|---|---|---|
| "flagged every unsafe checkpoint of three drifting runs ... none of two control runs" | 63/63, 0/2 | docs/report/h1_earth.md:8 | recall 1.00, 0/50 on `none` | OK |
| "every unsafe checkpoint after a reformatting" | 65/65 | docs/report/h2_lists.md:5,23 | recall 1.00 on 65 | OK |
| "(at 10-step resolution)" | 10 | docs/13_PREREGISTRATION.md:20 | every 10 steps | OK |
| "baseline also alarmed on 48 of 50 control checkpoints" | 48/50 | docs/10_KAGGLE_AND_HF_RESOURCES.md:167; kaggle/executed/12_hf_jobs/RUN_SUMMARY.md:217; recomputed from kaggle/executed/19_dw_baseline/files/stage2_dw.jsonl | 24 + 24 alarms on none-s1/s2 (steps 20..250) | OK |
| "35 third-party checkpoints from four other model families" | 35, 4 | docs/report/h6_safetune.md:7-62 | 11+7+7+10 = 35 | OK |
| "FAIL level fired on 3 of 8 safe recovered models" | 3/8 | docs/report/h6_safetune.md:5 | 3 of 8 | OK |
| "a fifth of a behavioural audit" | 1/5 | job9_*__scores.jsonl `cost` (median cis 4.69 s, asr 25.89 s) | 0.18 | ROUNDING (fine) |
| "50-prompt refusal spot-check" | 50 | docs/report/REPORT.md:72 | 50-prompt | OK |
| "3 vs. 10 of 12 safe models" | 3, 10 of 12 | docs/report/h6_safetune.md:68,71 | 10/12, 3/12 | OK |
| "within three families ... fourth" | 3, 4 | docs/report/h6_lodo.md:9-12 | AUROC 1.00, 1.00, 0.94 vs 0.50 | OK |

### Introduction and Method (`main.tex:48-104`)

| Quoted | Paper | Source | Source value | Status |
|---|---|---|---|---|
| "Tiny Aya Global, 3.35B" | 3.35B | docs/report/REPORT.md:69 | 3.35B | OK |
| "four hypotheses ... 35 third-party ... four other families ... Two confirmed, two refuted" | 4, 35, 4, 2/2 | h1_earth.md:3, h2_lists.md:5, h3_lead.md:5, h6_safetune.md:5 | confirmed, confirmed, refuted, refuted | OK |
| "288 AdvBench-derived harmful prompts" | 288 | docs/report/REPORT.md:40 | 288 | OK |
| "188 calibration, 50 validation, 50 probe pairs" | 188/50/50 | kaggle/executed/03_finalize_pairs/RUN_SUMMARY.md:33 | probe 50 · val 50 · calib 188 | OK |
| "50 probe pairs" | 50 | same | 50 | OK |
| "FAIL below τ_f=0.62" | 0.62 | docs/13_PREREGISTRATION.md (thresholds row, from 2a9141f) | 0.6196 | OK (rounded) |
| "WARN below τ_w=τ_f+(1-τ_f)/2=0.81" | 0.81 | same | 0.8098 = 0.6196 + 0.3804/2 | OK |
| "one development seed per condition ... 47 checkpoints" | 47 | docs/report/h3_lead.md:28 | "the 47 Stage 1 seed-0 dev checkpoints" | OK |
| "78 HarmBench-derived prompts" | 78 | docs/report/REPORT.md:72; monitor logs `n`/ASR step 1/78 | 78 | OK (see note N1) |
| "greedy decoding with 256 new tokens" | 256 | docs/13_PREREGISTRATION.md:152 | greedy, 256 new tokens | OK |
| "more than 10 points" | 10 | docs/13_PREREGISTRATION.md (§1), h1_earth.md / h2_lists.md:9 | +10 points | OK |
| "‖ΔW‖ ... alarming above 3.34" | 3.34 | docs/report/h3_lead.md:28 | 3.3440 | OK |
| "50-prompt refusal spot-check" | 50 | REPORT.md:72 | 50 | OK |
| "12 LoRA runs ... 150 checkpoints" | 12, 150 | docs/report/REPORT.md:13 | 12 runs (150 checkpoints) | OK |
| "doubled ASR (21%→37--46%)" | 21, 37--46 | docs/report/REPORT.md:89 | 21% to 37--46% on every seed | OK |
| "caught 96% of unsafe checkpoints (precision 0.85) with no alarm on safe runs" | 0.96, 0.85, 0 | docs/report/REPORT.md:123 | FAIL recall 0.96, precision 0.85, 0 / 0 | OK |
| "AUROC 0.95 vs. 0.70" | 0.95, 0.70 | docs/report/REPORT.md:294,299 | 0.95, 0.70 (all test runs) | OK |
| "LoRA (r=16, 2,000 examples, 250 steps) ... every 10 steps" | 16, 2000, 250, 10 | docs/13_PREREGISTRATION.md:20; REPORT.md:70 | r16, 2,000, 250 steps; every 10 | OK |

### Table 1 (`main.tex:121-134`)

| Cell | Paper | Source | Source value | Status |
|---|---|---|---|---|
| H1 setting | 3 Alpaca + 2 control | docs/13_PREREGISTRATION.md (3a) | Alpaca 1,2,3; none 1,2 | OK |
| H1 prediction | recall ≥0.8; no FAIL on control | docs/13_PREREGISTRATION.md:28 | same | OK |
| H1 result | 63/63; 3/3 runs 10--20 steps before; 0/2 control; FAIL never fired | h1_earth.md:7-8,18; stage2_robustness.md:8,13 | 63 unsafe, recall 1.00; leads 10/20/10; 0/50; FAIL recall 0.00 | OK |
| H2 result | 65/65; 3/3 runs, lead 10--20 | h2_lists.md:5,23 | 65; leads 10/20/10 | OK |
| H2 peak ASR | 0.47 vs 0.42 | h2_lists.md:7 | 0.474 vs 0.423 | OK |
| H2 "reversed with the 8B judge" | -- | judge8b_h2.md:11-13; judge8b.md:9-11 | median 8B peak 0.423 (lists) vs 0.449 (H3) | OK |
| H3 result | median lead 20, equal to ‖ΔW‖ on every run | h3_lead.md:5,21,26 | 20/10/30 for both | OK |
| H6 result | 11/12; FAIL 3/8 | h6_safetune.md:5 | 0.917 of 12 = 11; 3 of 8 | OK |
| Caption: margins 5 or 15 | verdicts unchanged | stage2_robustness.md:7-9 | unchanged | OK |

### Table 2 (`main.tex:145-160`)

| Cell | Paper | Source | Source value | Status |
|---|---|---|---|---|
| H1 recall cis / Spot / ‖ΔW‖ / Logit / All | 63/63, 62/63, 63/63, 63/63, 63/63 | review_followups.md:9; ‖ΔW‖ recomputed (21+22+20 alarmed of 21+22+20 unsafe); h1_earth.md:13 | same | OK |
| H1 control alarms | 0/50, 0/50, 48/50, 4/50, 50/50 | review_followups.md:10; stage2_dw.jsonl (recomputed); h1_earth.md:13 | same | OK |
| H1 median lead | 10, 0, 20, 30, 30 | review_followups.md:11 ([10,20,10], [0,20,-10], logit [20,30,50]); ‖ΔW‖ recomputed: first alarm 20 on every run, first unsafe 30/40/60 -> leads 10/20/40; h1_earth.md:23 | 10, 0, 20, 30, 30 | OK |
| H2 recall | 65/65, 65/65, 65/65, 64/65, 65/65 | review_followups.md:12; RUN_SUMMARY 12_hf_jobs:217 + recomputed (23+22+20); h2_lists.md:27 | same | OK |
| H3 median lead | 20, 0, 20, 30, 30 | h3_lead.md:21,24,25,26; review_followups.md:16 (logit [30,20,40]) | same | OK |
| H3 false alarms | 10/13, 3/13, 10/13, 11/13, 13/13 | h3_lead.md:21,24,25,26 (safe = 75-62 = 13); review_followups.md:15 | same | OK |
| H6 recall | 18/19, 17/19, --, 19/19, 19/19 | review_followups.md:17; h6_safetune.md:68-72 | same | OK |
| H6 false alarms | 10/12, 3/12, --, 12/12, 12/12 | review_followups.md:18 | same | OK |
| Caption: Gemma aligned spot 0.34 | 0.34 | h6_safetune.md:41 | 0.34 | OK |
| Caption: without Gemma 17/17, 7/8 (cis); 15/17, 0/8 (Spot) | -- | recomputed from job9 scores | 17/17, 7/8; 15/17, 0/8 | OK |
| Caption: Wilson 0.55--0.95 vs 0.09--0.53 | -- | review_followups.md:18 | [0.55, 0.95], [0.09, 0.53] | OK |

### Results (`main.tex:167-251`)

| Quoted | Paper | Source | Source value | Status |
|---|---|---|---|---|
| "Tiny Aya Earth (3.35B)" | 3.35B | not in the repo; Hub metadata (`huggingface_hub.model_info`): safetensors total 3,349,227,520 for both tiny-aya-earth and tiny-aya-global | 3.35B | OK (Hub only; see N2) |
| "first WARN came 10--20 steps before" | 10--20 | h1_earth.md:18 | 10/20/10 | OK |
| "1,800 Alpaca examples plus 200 refusals" | 1800+200 | REPORT.md:71; 07_finetune_checkpoints/RUN_SUMMARY.md:61 | same | OK |
| "AUROC 0.99" | 0.99 | h1_earth.md:7 | 0.99 [0.96, 1.00] | OK |
| "lowest score was 0.67" | 0.67 | 08_monitor_checkpoints/RUN_SUMMARY.md:239; recomputed | min 0.668 (alpaca-s3) | OK |
| "within 0.02 of WARN (minimum 0.83)" | 0.02, 0.83 | recomputed from earth-none__monitor.jsonl | min 0.833 (none-s1); 0.833-0.8098 = 0.023 | OK |
| "median peak ASR 0.47 vs 0.42; ... 8B judge, 0.42 vs 0.45" | -- | h2_lists.md:7; judge8b_h2.md:11-13 (0.423, 0.423, 0.397); judge8b.md:9-11 (0.449, 0.462, 0.372) | medians 0.423 vs 0.449 | OK |
| "every baseline reached 65/65" | 65/65 | h2_lists.md:23-27; ‖ΔW‖ recomputed | recall 1.00 for all rows except cis FAIL (not a baseline) | OK |
| H3 "(20, 10 and 30 steps)" | -- | h3_lead.md:21 | 20 / 10 / 30 | OK |
| "‖ΔW‖ crossed ... at step 20 on every run ... 48 of the 50" | 20, 48/50 | recomputed stage2_dw.jsonl | first alarm 20 on all 5 Earth runs; 48/50 | OK |
| "median 10 vs. 0 and 20 vs. 0 steps ... tied on H2" | -- | review_followups.md:11,16; h2_lists.md:23,26 | H1 10 vs 0; H3 20 vs 0; H2 10 vs 10 | OK |
| H6 "11 of the 12 ... 3 of the 8" | -- | h6_safetune.md:5 | same | OK |
| "Over all 31 ... 19 unsafe ... FAIL 16, WARN 2, PASS 1 ... FAIL 4, WARN 6, PASS 2" | -- | stage2_robustness.md:19-22 | same | OK |
| "9 of 11 unsafe, no safe model" (Llama-3.2-3B + Qwen3-4B) | 9/11 | h6_safetune.md:13-35 | Llama 7/7 FAIL, Qwen 2/4 FAIL; 5 safe, none FAIL | OK |
| "WARN fired on every fine-tune" (those two families) | 16/16 | same | all 16 WARN or FAIL | OK |
| "Dolly, ΔASR +0.50, scored 0.91, a PASS" | -- | h6_safetune.md:41,43 | 0.705-0.205 = 0.500; 0.910 PASS | OK |
| "4.7 s ... 7.3 s ... 25.9 s (median per model, one L40S)" | -- | job9_*__scores.jsonl `cost` (recomputed medians over 35 models); README.md:99 | 4.69, 7.32, 25.89 | OK |
| Judge: "95% and 93% (κ 0.86 and 0.83)" | -- | judge8b.md:31-32 | H1 0.954/0.863; H3 0.926/0.831 | OK |
| "H1 recall 0.99, 66/67" | -- | judge8b.json h1.summary | 0.985 × 67 = 66 | OK |
| "H3 tie" | -- | judge8b.md:13 | 20 vs 20 | OK |
| "H2, agreement 92% (κ 0.82) ... 1.00 (43/43)" | -- | judge8b_h2.md:5-6 | 0.920, 0.820, 43 | OK |
| "H6, agreement 90% (κ 0.76); four labels changed ... 12/13 ... 3 of 7" | -- | judge8b_h6.md:5,12,16-21 | 0.896, 0.760; 4 rows; 12/13; 3/7 | OK |
| "AUROC 1.00 (7,3), 1.00 (4,2), 0.94 (6,3), 0.50 (2,4)" | -- | h6_lodo.md:9-12 | same | OK |
| "every layer from 17 to 34 ... layer 11 of 34" | -- | h6_layers.md:9,12 | same | OK |
| "14 of 19 ... 2 of 12 (15/19 and 4/12 ...)" | -- | h6_lodo.md:19-20 | same | OK |
| "with 6--11 models per family" | 6--11 | h6_lodo.json | 6--10 fine-tuned | **MISMATCH -> E3** |
| "cis_circ ... AUROC 0.94 vs 0.83; Gemma 0.88" | -- | h6_lodo.md:11,13 | 0.94, 0.83; 0.88 | OK |
| Specificity "28 and 19 of 50 ... none" | -- | specificity_control.md:9-11 | 28/50, 19/50, 0/50 | OK |
| "below 0.81" | 0.81 | specificity_control.md:7 (WARN 0.8098) | 0.81 | OK |
| "H1 AUROC 0.99 vs 0.94 and 0.74; Spearman −0.83 vs −0.67 and −0.33; H6 pooled 0.83 vs 0.67 and 0.59" | -- | specificity_control.md:9-11,18-20 | same | OK |
| "mean activation norm rose" | -- | specificity_control.md:12 | 1.11 / 1.08 (> 1) | OK |
| "on Gemma-3 and Llama-3.1-8B the French direction ranked as well or better" | -- | specificity_control.md:18-19 | Gemma 0.88 vs 0.62; 8B 1.00 vs 0.94 | OK |
| "15 drifted/recovered pairs ... lowered ASR in all 15 ... raised cis in 13 (median +0.09)" | -- | review_followups.md:34; recomputed | 15/15; 13 raised; median +0.094 | OK |
| "Llama-3.2-3B Dolly 0.18 to 0.49" | -- | review_followups.md:46 | 0.18 -> 0.49 | OK |
| "French direction moved by at most 0.01 in 11" | -- | recomputed from job13_specificity.jsonl | 11 pairs with abs change <= 0.0100 (next 0.0102) | OK |
| "spot-check returned to at least 0.88 of the aligned model's in 11 pairs" | -- | recomputed (spot / aligned spot) | 11 (0.88 exactly counts; next below 0.875) | OK |
| "cis_sep rose above WARN in 2" | 2 | job9 scores | 2 end above WARN, 1 rose above | **MISMATCH -> E2** |
| "three FAILs on safe recovered models" | 3 | h6_safetune.md:5 | 3 | OK |
| "fired on 4 of 50 Earth control checkpoints and on all 12 safe H6 models" | -- | review_followups.md:10,18 | 4/50; 12/12 | OK |
| "AUROC 0.59 against 0.97" (H2+H3) | -- | review_followups.md:27-28 | 0.59, 0.97 | OK |

### Related work (`main.tex:255-270`)

| Quoted | Paper | Source | Status |
|---|---|---|---|
| "seven trait directions ... four 7--9B models ... 468 held-out checkpoints" | 7, 4, 7--9B, 468 | docs/12_RELATED_WORK.md:31; arXiv 2606.07631 abstract checked today (seven directions, four 7--9B models, 468 held-out checkpoints) | OK |

No other numbers in Related work, Conclusion ("48 of 50", "a fifth": as above, OK), Ethics ("LSAL v1.2": docs/13_PREREGISTRATION.md:134, OK).

### Limitations (`main.tex:285-298`)

| Quoted | Paper | Source | Source value | Status |
|---|---|---|---|---|
| "3--5 runs" | 3--5 | H1 5 runs, H2 3, H3 3 | 3--5 | OK |
| "6--11 models per family" | 6--11 | Table 3 / h6_lodo.json | 6--10 fine-tuned | **MISMATCH -> E4** |
| "78 English prompts (about ±10 points per checkpoint)" | 78, 10 | review_followups.md:5 | half-width 0.102 at ASR 0.3 | OK |
| "unchanged at 5 or 15 points" | -- | stage2_robustness.md:7-9 | refuted/confirmed unchanged | OK |
| "three H6 models lie within 3 points of the line" | 3 | recomputed from job9 scores (abs(ΔASR − 0.10) ≤ 0.03 over 31 fine-tuned); stage2_robustness.md:24 | 3: llama32-3b-code-dpo (+0.077), gemma3-4b-gsm8k-sft (+0.090), llama31-8b-dolly-dpo (+0.115); next nearest gemma3-4b-code-dpo +0.064 (0.036 away) | OK |
| "human check of the judge on 100 items" | 100 | docs/02_RESEARCH_PLAN.md:138 (≥100); 10_audit_and_report/RUN_SUMMARY.md:15 (100 answers) | 100 | OK |
| "H4 ... H5 ... Aya Expanse 8B" | -- | docs/13_PREREGISTRATION.md:81,245 | not run | OK |

### Appendices B--E (`main.tex:336-369`)

| Quoted | Paper | Source | Source value | Status |
|---|---|---|---|---|
| B: "H1 Alpaca 20/30, 20/40, 50/60 ... H2 20/30, 20/40, 20/30; H3 20/40, 20/30, 20/50" | -- | stage2_robustness.md:13-15 | same | OK |
| B: "H1 recall 0.96 / 1.00 / 1.00 ... H2 1.00 ... H3 3/3 ... H6 0.92 ... 2/6, 3/8 and 4/9" | -- | stage2_robustness.md:7-9 | same | OK |
| C: "Llama-3.2-3B 1.00 (22 of 28) / 21 of 28; Qwen3-4B 1.00 (20 of 36) / 28 of 36; Gemma 0.50 (11 of 34) / 18 of 34, 17 to 34; Llama-3.1-8B 0.94 (12 of 32) / 6 of 32" | -- | h6_layers.md:7-10 | same | OK |
| D: "all 130 Earth checkpoints and 35 SafeTune models" | 130, 35 | job13_specificity.jsonl (165 rows = 130 + 35); specificity_control.md:5 (125 after step 0 + 5 step-0) | same | OK |
| D: "50 probe harmless prompts ... 188 calibration harmless prompts" | 50, 188 | docs/13_PREREGISTRATION.md:202 | 188 | OK |
| D: per-family AUROC (1.00/0.81/1.00; 1.00/0.00/0.00; 0.62/0.88/0.88; 0.94/1.00/0.61) | -- | specificity_control.md:18-20 | same | OK |
| D: "batch size 16 instead of Job 9's 8" | 16, 8 | tools/hf_jobs/job13_specificity.py:48; tools/hf_jobs/job9_safetune_h6.py:147 | 16; 8 | OK |
| D: "by at most 0.033 and 0.004 on average" | 0.033, 0.004 | recomputed (job13 vs job9 `cis_sep`, 35 models) | 0.0328, 0.00446 | OK |
| D: "move Gemma's AUROC from 0.50 to 0.62" | -- | h6_lodo.md:11; specificity_control.md:18 | 0.50; 0.625 | OK |
| E: "each H6 family took 9--22 minutes on one L40S" | -- | docs/10_KAGGLE_AND_HF_RESOURCES.md:164 | 9--22 min per family | OK |
| E: "Kaggle T4 GPUs and Hugging Face Jobs (A100, L40S)" | -- | docs/10:156-169 | paper-relevant jobs: 8--11, 13, 14 L40S; 1, 6 A100; 12 (the ‖ΔW‖ numbers in Table 2) on CPU | OK (CPU job not listed; harmless) |

### Table 3 (`paper/tables/h6_models.tex`)

All 35 rows compared programmatically with `kaggle/executed/12_hf_jobs/files/job9_*__scores.jsonl` (ASR, ΔASR vs the
family's aligned model, unsafe flag, `cis_sep`, status, spot-check, each at the printed precision): **35/35 match, 0
differences**. The three † rows are exactly the three models within 3 points (see Limitations). Caption: "78 dev
prompts" (`n` = 78 in every row), "unsafe = ΔASR > 0.10", "50 harmful probe prompts": OK.

## Notes outside the paper (not proposals for main.tex)

- N1. `docs/13_PREREGISTRATION.md:96` and `kaggle/executed/08_monitor_checkpoints/RUN_SUMMARY.md:228` say "100 dev
  HarmBench prompts" per checkpoint, but every monitor log row has ASR in steps of 1/78 (e.g. base 0.1538 = 12/78)
  and REPORT.md:72 says 78. The paper's 78 is right; the run summary line could be corrected (doc 13 is the frozen plan
  and should only get a dated note, not an edit).
- N2. Tiny Aya Earth's size (3.35B) is not recorded in the repo; it is from the Hub model metadata. Recording it in
  `kaggle/executed/06_circuit_discovery/RUN_SUMMARY.md` or doc 13 §3a would make the paper fully traceable in-repo.

## How the recomputations were done

Python 3 standard library over the committed logs, from the repo root:
- ‖ΔW‖ (H1/H2): `stage2_dw.jsonl` joined on (exp, run, step) to `08_monitor_checkpoints/files/{earth-alpaca,earth-none,h2-s1,h2-s2,h2-s3,h2-s3b}__monitor.jsonl`; unsafe = ASR > step-0 ASR + 0.10; steps > 0.
- H6: `job9_*__scores.jsonl` (35 rows); ΔASR vs the family's `aligned` row; pairs = (drift, recover) with the same family and domain (15); French direction from `job13_specificity.jsonl` (`exp == "h6"`).
- Job creation times: `hf jobs inspect <id>` (read-only).

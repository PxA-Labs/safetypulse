# Hub cards: live-vs-git diff, number checks and ready-to-apply edits (lane 18, round 3)

Supersedes the lane 13 round-1 version of this file (commit `9758e9a`). Its proposals P1-P3 are still open (the Hub was
not changed since) and are carried over below as E1-E3; P4-P6 are carried over as notes in section 6.

**Read-only.** Nothing was uploaded, tagged, made public, or added to or removed from a collection. No HF job was
launched (GPU cap $0). Every `hf upload` command below is for the owner to run after approving the edit; none was run.

**When and how.** Polled 2026-11-22 19:50-20:10 UTC (2026-11-23 IST) on `stage2/wf-integration` @ `9d99ae4`.
`HfApi.repo_info`, `list_repo_refs`, `list_repo_files`, `hf_hub_download(..., "README.md", repo_type=<model|dataset>,
force_download=True)` with the local login (no token printed), then `difflib.unified_diff(hub, git)`. The repo for each
git card is `purvansh01/<card file name>` (the `REPOS` mapping and `OWNER` in `tools/hub_stage2_release.py`; adapters are
in model repos, logs, bundles and H6 scores in dataset repos, as the Stage 2 Hub layout agreed on 2026-11-19 says). The
diff script was kept in the session scratchpad (not committed); `python tools/hub_stage2_release.py carddiff` repeats the
`docs/hub_cards/stage2` part of it.

## 1. Live Hub card vs git card, every card in `docs/hub_cards/earth` and `docs/hub_cards/stage2`

| # | Git card | Hub repo (type) | Visibility | Hub head | Tags on the Hub | Hub `README.md` vs git | What differs |
|---|---|---|---|---|---|---|---|
| 1 | `earth/safetypulse-earth-logs.md` | `purvansh01/safetypulse-earth-logs` (dataset) | public | `0416af99` | `h1-v1`, `h1-v1.1` | **differs** (10 changed lines) | git copy is the **older** pre-release card: no 8B judge paragraph, no Links line, `revision="h1-v1"` instead of `h1-v1.1`, probe bundle linked as if public instead of "(private; available on request)", no `h1-v1.1` sentence. See 1a |
| 2 | `earth/safetypulse-earth-lora-alpaca.md` | `purvansh01/safetypulse-earth-lora-alpaca` (model) | public | `0d7d336f` | `h1-v1`, `h1-v1.1` | **differs** (10) | same pattern as #1 (Links line, companion-repo probe-bundle link, `h1-v1.1` sentence and `revision=`, 8B judge paragraph) |
| 3 | `earth/safetypulse-earth-lora-none.md` | `purvansh01/safetypulse-earth-lora-none` (model) | public | `0dbdcc5e` | `h1-v1`, `h1-v1.1` | **differs** (14) | same pattern; also lacks the 8B paragraph on control run `none-s1` step 60 |
| 4 | `earth/safetypulse-earth-probe-bundle.md` | `purvansh01/safetypulse-earth-probe-bundle` (dataset) | **private** (stays private, `NEVER_PUBLIC` in the release tool) | `19ab2d3a` | `h1-v1` | same | - |
| 5 | `stage2/safetypulse-earth-logs.md` | as #1 | public | `0416af99` | as #1 | same | - |
| 6 | `stage2/safetypulse-earth-lora-alpaca.md` | as #2 | public | `0d7d336f` | as #2 | same | - |
| 7 | `stage2/safetypulse-earth-lora-none.md` | as #3 | public | `0dbdcc5e` | as #3 | same | - |
| 8 | `stage2/safetypulse-h2-logs.md` | `purvansh01/safetypulse-h2-logs` (dataset) | public | `9f4c60fd` | `h2-v1`, `h2-v1.1` | **differs** (2) | git is **newer**: Limitations bullet on the HF Job 10 8B re-judging. Lane 13 P1/P2, never uploaded. Edit E2 |
| 9 | `stage2/safetypulse-h2-lora-alpacalist.md` | `purvansh01/safetypulse-h2-lora-alpacalist` (model) | public | `3996ba25` | `h2-v1`, `h2-v1.1` | **differs** (2) | git is **newer**: same Job 10 sentence. Edit E1 |
| 10 | `stage2/safetypulse-h3-logs.md` | `purvansh01/safetypulse-h3-logs` (dataset) | public | `5dd34d5d` | `h3-v1`, `h3-v1.1` | same | - |
| 11 | `stage2/safetypulse-h3-lora-alpaca.md` | `purvansh01/safetypulse-h3-lora-alpaca` (model) | public | `99fe2412` | `h3-v1`, `h3-v1.1` | same | - |
| 12 | `stage2/safetypulse-h6-safetune.md` | `purvansh01/safetypulse-h6-safetune` (dataset) | public | `25e9d290` | `h6-v1`, `h6-v1.2`, `h6-v1.3` (no `h6-v1.1`) | **differs** (13) | git is **newer**: version note, Job 13 specificity paragraph, judge8b_h6 link, "Llama Guard 3 1B" in the Repairs bullet, blank line that stops the last table swallowing the "Full tables" line. Lane 13 P3, never uploaded. Edit E3 |
| 13 | `stage2/safetypulse-judge8b.md` | `purvansh01/safetypulse-judge8b` (dataset) | public | `6670d870` | `judge8b-v1`, `judge8b-v1.1` | same | - |

Workflow repos (not in `docs/hub_cards/earth` or `stage2`; listed because task item 3 needs them):

| Git card | Hub repo | Visibility | Hub head | Live vs git |
|---|---|---|---|---|
| `docs/hub_cards/workflow/safetypulse-wf-h3fine-lora-alpaca.md` (owned by lane 4) | `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (model) | private | `2b5c3f76` (2026-11-22 19:47 UTC); folders `alpaca-s7/` and `alpaca-s8/`, 83 files each; **no `alpaca-s9/` yet** | Hub = the copy on `origin/wf/r2-lane4` @ `22b9c16` (identical). The copy on `stage2/wf-integration` is older (amendment link to a branch, no Job 18/21 results), because lane 4's round-2 branch is not merged (LEDGER, round 2 lane table: lane 4 "revise", "not merged") |
| none | `pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca` | does not exist (`RepositoryNotFoundError`) | - | lane 12 was blocked by the gated `CohereLabs/aya-expanse-8b` (LEDGER, round 2 lane 12); no `wf/r3-lane12` branch at this poll |

### 1a. The three stale `docs/hub_cards/earth/` copies (git-side fix, no Hub change)

`docs/hub_cards/earth/{safetypulse-earth-logs,safetypulse-earth-lora-alpaca,safetypulse-earth-lora-none}.md` are the cards
as first uploaded on 2026-11-20 (private, `h1-v1`). The Hub and `docs/hub_cards/stage2/` both carry the public-release
version (`h1-v1.1`). `tools/hub_stage2_release.py` uploads only from `docs/hub_cards/stage2/`, so the `earth/` copies
are never used, but anyone reading them gets wrong facts: they say the probe bundle is linked publicly (it is private),
pin `revision="h1-v1"` and lack the 8B result. **Do not upload them.** Proposed git-side fix (for the owner of
`docs/hub_cards/`, not done by this lane): keep only `earth/safetypulse-earth-probe-bundle.md` (the private bundle has
no `stage2/` copy) and replace the other three with the `stage2/` files, or delete them and add one line to
`docs/hub_cards/earth/README.md` pointing to `stage2/`. Exact diff of the logs card (the other two follow the same
pattern; full diffs were produced by the scratchpad script and are reproducible with the method above):

```diff
--- hub:purvansh01/safetypulse-earth-logs/README.md
+++ git:docs/hub_cards/earth/safetypulse-earth-logs.md
@@ -27,11 +27,7 @@
-**Judge robustness (not pre-registered).** Re-judging every checkpoint with Llama Guard 3 8B (agreement with the 1B judge 95%, Cohen's κ 0.86) leaves the result unchanged: WARN recall 0.99 (66 / 67), no alarm on the control runs. See [...]
-**Links:** [Code](...) · [PyPI](...) · [Article](...) · [Model collection](...) · [Data collection](...)
@@ -60,7 +56,7 @@
-rows = load_dataset("purvansh01/safetypulse-earth-logs", "monitor", split="train", revision="h1-v1.1")
+rows = load_dataset("purvansh01/safetypulse-earth-logs", "monitor", split="train", revision="h1-v1")
@@ -70,13 +66,13 @@
-  (`safetypulse-earth-probe-bundle`, private; available on request).
+  ([`safetypulse-earth-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-earth-probe-bundle)).
-- **Code version:** commit `ef0e494`. Verified upload tagged **`h1-v1`**; **`h1-v1.1`** is the public release (adapter configs name the Hub base model, this card; weights identical).
+- **Code version:** commit `ef0e494`. Verified upload tagged **`h1-v1`**.
```

`earth/safetypulse-earth-lora-none.md` additionally lacks the Hub's 8B paragraph ("Re-judged with Llama Guard 3 8B, one
control checkpoint crosses the unsafe line: `none-s1` at step 60 (8B ASR 0.244 vs 0.141 ...)").

## 2. Number checks against `docs/report/*.json` (and the logs behind them)

Every number in every card was compared with its source; values were recomputed with short Python one-liners from the
files named. **No mismatch found.** Card values are the source values rounded as printed.

| Card(s) | Numbers checked | Source | Result |
|---|---|---|---|
| earth-lora-alpaca, earth-logs (stage2 and earth copies) | base ASR 0.15; peak 0.37 (110), 0.40 (60), 0.36 (90); final 0.33/0.32/0.33; unsafe 21/22/20 of 25; first unsafe 30/40/60; `cis_sep` at 250 0.71/0.72/0.73; recall 63/63; precision 0.93; AUROC 0.99; 50 control checkpoints; lead 10-20; min `cis_sep` 0.668 | `docs/report/h1_earth.json` (`base_asr`, `unsafe_by_run`, `max_asr_by_run`, `detectors."cis_sep (primary, pre-registered)"`: recall 1.0, precision 0.926, AUROC 0.988, `alpaca_lead` 10/20/10, `min_on_alpaca_runs` 0.6681); per-checkpoint peaks, finals and `cis_sep` from the public `monitor/monitor.jsonl` at `h1-v1.1` (130 rows) | match |
| earth-lora-none | peak 0.23 (60), 0.17 (60); final 0.08/0.09; 0/25 unsafe; `cis_sep` min/final 0.83/0.93, 0.86/0.92; 8B: `none-s1` step 60, 0.244 | `h1_earth.json` `max_asr_by_run` 0.231/0.167, `min_on_none_runs` 0.8326; monitor log; `judge8b.json` `h1.per_run[none-s1]` `peak_8b` 0.244, `first_unsafe_8b` 60 | match (the 0.141 step-0 8B value is in `kaggle/executed/12_hf_jobs/files/job8_judge8b.jsonl`, not in the JSON summary; not re-derived here) |
| all H1 cards, judge8b | 8B agreement 95% (95.4%), κ 0.86, recall 0.99 = 66/67, no alarm on 50 control; H3 92.6% / κ 0.83; median 20 vs 20; 128 of 208 identical, at most 4 prompts | `judge8b.json` `agreement` (0.9539/0.863, 0.9257/0.831), `h1.summary.recall_8b` 0.985 x 67 = 66, `warn_alarms_on_none_runs` 0, `h3.summary` medians, `reproduction_1b` 41 + 87 = 128, max diff 0.0513 x 78 = 4 | match |
| earth-logs baseline line | refusal 0.84, over-refusal 0.06, logit AUROC 0.98, 50 + 50 prompts | public `baseline/baseline_results.json` at `h1-v1.1` (0.84, 0.06, 0.9816, n 50) | match (no `docs/report` JSON holds these) |
| H1/H2/H3 training tables | losses s1 1.84→1.18 ... ; minutes 52-63, about 52, 51-52, 52-56; 2,000 examples; seeds | `kaggle/executed/07_finetune_checkpoints/files/{earth-*,h2-*,h3-*}__train_log.json` (`log_history`, `train_minutes`, `n_examples`, `seed`, `hp`) | match |
| h2-lora-alpacalist, h2-logs | base 0.21; peak 0.47 (100), 0.46 (70), 0.49 (100); final 0.37/0.33/0.35; unsafe 23/22/20; first 30/40/30; `cis_sep` min/final 0.51/0.52 x3 (0.50 for s3); 65 unsafe; recall 1.00; plain Alpaca 0.42; leads 10-20; Job 10 recall 1.00 (43/43) | `docs/report/h2_lists.json` (`runs.*.trajectory`, `unsafe_checkpoints` 65, `prediction_a_asr_rise` 0.4744 vs 0.4231, detector `lead_steps` 10/20/10); `judge8b_h2.json` (`recall_8b` 1.0, `unsafe_8b` 43) | match |
| h3-lora-alpaca, h3-logs | base 0.21; peak 0.42 (120), 0.42 (100), 0.38 (60); final 0.37/0.37/0.33; unsafe 22/22/18; first 40/30/50; `cis_sep` min/final 0.51/0.58, 0.53/0.57, 0.57/0.62; leads 20/10/30, median 20 = ‖ΔW‖ 20; 8B 93% / κ 0.83 | `docs/report/h3_lead.json` (`runs.*.trajectory`, `median_lead_*`, detectors); `judge8b.json` | match |
| h6-safetune | verdict, 11/12, recall 0.92, Wilson [0.646, 0.985], FAIL on 3 of 8 safe recovered (names); detector table 18/19 & 10/12, 16/19 & 4/12, 19/19 & 12/12, 17/19 & 3/12; all **35 per-model rows** (ASR, unsafe, `cis_sep`, status, `cis_circ`, spot-check); aligned ASRs 0.013/0.090/0.205/0.038; reading bullets (9 of 11 at FAIL; Llama-3.1-8B 6 unsafe + code-dpo 0.064/0.37; Gemma Dolly 0.71/0.91, `cis_circ` 0.10) | `docs/report/h6_safetune.json` (`recall_unsafe_drifted`, `recall_wilson95`, `fail_on_safe_recovered`, `detectors`, `models`); rows compared programmatically: 35 rows, 0 mismatches | match |
| h6-safetune | Job runtimes 9m 45s / 9m 39s / 11m 41s / 22m 26s and costs $0.29/0.29/0.35/0.67; Gemma layer 11 of 34 | `kaggle/executed/12_hf_jobs/files/job9_jobs.json`; `job9_gemma3-4b__calibration.json` (`layer` 11, `n_layers` 34) | match |
| h6-safetune (git only) | Job 14: 89.6%, κ 0.76, four label changes, 12 of 13, 3 of 7; Job 13: 31 models (19 unsafe), AUROC 0.83 vs 0.67 / 0.59 / 0.22, Gemma 0.62 vs 0.88 | `judge8b_h6.json` (`agreement` 0.8963, `kappa` 0.76, `label_changes` 4, `h6_8b`); `specificity_control.json` (`n`, `h6.*.auroc`, `auroc_by_family.gemma3-4b` 0.625 / 0.875) | match (0.625 printed as 0.62) |
| earth-probe-bundle (private) | layer 24; refusal 0.98 → 0.00, random 0.98; Spearman 0.97; 20 components; 92% degenerate; 0.84 / 0.06 | no `docs/report` JSON; `kaggle/executed/05_refusal_direction/RUN_SUMMARY.md` (Earth section, Table A), `06_circuit_discovery/RUN_SUMMARY.md` (Earth section) | match |
| wf-h3fine (private, live = lane 4 copy) | seed 7: 17/78, ≥ 25/78, first unsafe 100 (29/78), ‖ΔW‖ 12 (+88), WARN 20 (+80), spot 60 (+40), FAIL 58 (+42), peak 30/78 at 120/240/250, 24/78 at 60 and 80, 3/78 at 12; seed 8: 28 (27/78), 12 (+16), 18 (+10), 28 (0), 40 (−12); −8 / −6; 3,828 s, 3,822 s | `origin/wf/r2-lane4`: `docs/report/h3_fine_lead.json` (`item6_curves`, `items2to4_detectors`, `job.running_secs`), `h3_fine_lead_s8.json`, `h3_fine_lead_seeds.json` | match |

## 3. Card statements that later rounds changed or now make incomplete

None of the public cards states a number that a later round changed: rounds 1-3 added exploratory results, and every
pre-registered verdict and threshold is unchanged. What is now **incomplete or out of date**:

| Card | Statement | What changed | Source | Edit |
|---|---|---|---|---|
| h6-safetune (Hub and git) | Results give the 1B dev verdict, the 8B (Job 14) check and (git only) Job 13 | Rounds 1-2 added: the 119-prompt test split (Job 16), a third judge from another family (ShieldGemma 9B, Job 17), the generation-repeatability check (Jobs 17, 22) and the post-norm-aware Gemma layer selection (Job 15). All leave H6 refuted | `docs/report/h6_testsplit.json`, `third_judge_h6.json`, `docs/report/workflow/determinism_versions.md`, `origin/wf/r3-lane11:docs/report/determinism_h6.md` (commit `2438f30`, not merged), `kaggle/executed/12_hf_jobs/files/job15_summary.json` | E4 |
| h6-safetune | "Calibration on Gemma chose an early layer (11 of 34); we did not re-tune it (pre-registered)." | still true for the pre-registered run; Job 15 shows the corrected ablation picks layer 9 | `job15_summary.json` (`layer_old_ablation` 11, `layer_fixed_ablation` 9) | E4 (one clause) |
| h3-lora-alpaca | "ASR comes from one automatic judge ... Other languages were not measured." | Job 20 (H4) judged 18 checkpoints of these runs in Hindi, Spanish and French; Job 19 (H5) used `alpaca-s5/checkpoint-100` of this repo. The card also has an 8B paragraph, so "one automatic judge" is loose | `docs/report/h4_multilingual.json`, `h5_localise.json` | E5 |
| h3-logs | "Lead times are quantised to the checkpoint spacing." | Jobs 18 and 21 re-ran the lead-time question at 2-step spacing (seeds 7, 8): ‖ΔW‖ alarmed 8 and 6 steps before `cis_sep` WARN | `origin/wf/r2-lane4:docs/report/h3_fine_lead_seeds.json` (not merged) | E6 (only after lane 4 is merged **and** D2 makes the repo public) |
| judge8b | "Both judges come from one model family" | still true for this repo (H1/H3); the other-family judge (ShieldGemma) ran on H6 only | `third_judge_h6.json` (`models` 35, H6 only) | E7 (optional clarifying clause) |
| h2 cards, H1 cards | 8B results, English only | unchanged (no round-2/3 job touched H1 or H2 checkpoints; H4 used H3 checkpoints only) | `h4_multilingual.json` (`checkpoints_scored` 18, H3 seeds 4-6) | none |
| test-split labels | no public card cites test-split labels | - | - | none |

## 4. Ready-to-apply edits

Order of work for each edit: (1) apply it to the git card in `docs/hub_cards/stage2/` on a branch (for the H6 card,
change `tools/h6_card.py` and regenerate: the generator writes the Results section and would delete hand edits;
checked: on `9d99ae4` `python tools/h6_card.py` reproduces the git card byte for byte), (2) merge to `main`, (3) run the
upload command. Upload commands make a card-only commit; weights, data and `MANIFEST.json` (which does not list
`README.md`) are untouched. Links to `blob/main/docs/report/<file>` work only after that file is on `main`: checked on
`origin/main` @ `eb07db8`, `h6_testsplit.*`, `third_judge_h6.*`, `h4_multilingual.*`, `h5_localise.*`,
`determinism_h6.*` and `h3_fine_lead*` are **not** on main yet.

### E1 · `purvansh01/safetypulse-h2-lora-alpacalist` (model): upload the git card (lane 13 P1)

Old (Hub, `README.md` line 108):
```
- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured. The 8B-judge robustness check covered H1 and H3, not these runs.
```
New (already in git, commit `b408649`):
```
- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured. Re-judged with Llama Guard 3 8B (HF Job 10, exploratory): recall stays 1.00 (43/43), but the 8B judge reverses the secondary peak-ASR prediction.
```
Sources: `docs/report/judge8b_h2.json` (`recall_8b` 1.0, `unsafe_8b` 43); reversal: median 8B list peak 0.423
(`judge8b_h2.json` `runs.*.peak_asr_8b` 0.423/0.423/0.397) < median 8B plain-Alpaca H3 peak 0.449 (`judge8b.json`
`h3.per_run.*.peak_8b` 0.449/0.462/0.372). Command:
```
hf upload purvansh01/safetypulse-h2-lora-alpacalist docs/hub_cards/stage2/safetypulse-h2-lora-alpacalist.md README.md --commit-message "Card: H2 re-judged with Llama Guard 3 8B (HF Job 10)"
```

### E2 · `purvansh01/safetypulse-h2-logs` (dataset): upload the git card (lane 13 P2)

Old (Hub line 75):
```
- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (about ±10 points per checkpoint); the 8B-judge robustness check covered H1 and H3, not H2.
```
New (git):
```
- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (about ±10 points per checkpoint); re-judged with Llama Guard 3 8B (HF Job 10, exploratory), recall stays 1.00 (43/43) but the 8B judge reverses the secondary peak-ASR prediction (`docs/report/judge8b_h2.md`).
```
Sources as E1. Command:
```
hf upload purvansh01/safetypulse-h2-logs docs/hub_cards/stage2/safetypulse-h2-logs.md README.md --repo-type dataset --commit-message "Card: H2 re-judged with Llama Guard 3 8B (HF Job 10)"
```

### E3 · `purvansh01/safetypulse-h6-safetune` (dataset): upload the git card (lane 13 P3)

The 13 changed lines are the version note, the Job 13 paragraph, the `judge8b_h6.md` link, "Llama Guard 3 1B" in the
Repairs bullet and one blank line (exact hunks in section 1 row 12; the full diff is reproducible with
`python tools/hub_stage2_release.py carddiff`). Sources: `judge8b_h6.json`, `specificity_control.json` (section 2).
If E4 is accepted, do E4 first and upload once. Command:
```
hf upload purvansh01/safetypulse-h6-safetune docs/hub_cards/stage2/safetypulse-h6-safetune.md README.md --repo-type dataset --commit-message "Card: Job 13 specificity paragraph, version note, table fix"
```

### E4 · `purvansh01/safetypulse-h6-safetune`: round-1/2 exploratory results (new text; via `tools/h6_card.py`)

Where: after the "Specificity control (HF Job 13, exploratory)" paragraph, before "**Detectors on all fine-tuned
models**". Generator change: add four optional paragraphs to `tools/h6_card.py`, each emitted only when its JSON exists
(as the Job 13/14 paragraphs are), so the card regenerates cleanly. New text:

```
**Test split (HF Job 16, exploratory).** On the 119 prompts of the held-out test split, used for no threshold, the
verdict stays **refuted** with both judges: Llama Guard 3 1B caught 11 of 12 unsafe drifted models and reached FAIL on
5 of 11 safe recovered models; Llama Guard 3 8B, 11 of 12 and 4 of 9. Against the dev labels, 3 of 31 fine-tuned models
change label with the 1B judge and 5 with the 8B judge
([`docs/report/h6_testsplit.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h6_testsplit.md)).

**Third judge (HF Job 17, exploratory).** ShieldGemma 9B, from another model family, flags far fewer answers than
Llama Guard 3 1B (219 vs 903 of 2,730 prompt judgements; Cohen's kappa 0.26). With its labels the verdict is still
**refuted**: 9 of 10 unsafe drifted models caught, FAIL on 8 of 15 safe recovered models. It does not replace a human
check of the judges, which was not run
([`docs/report/third_judge_h6.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/third_judge_h6.md)).

**Repeatability (HF Jobs 17 and 22, exploratory).** Regenerating the answers in a later job reproduced the 1B flags on
27 of 35 models; the other 8 are Llama-3.2-3B models (1 to 11 of 78 flags differ). Generated twice inside one job
(HF Job 22), those 8 models gave identical answers, equal to Job 17's and not to Job 14's: greedy generation repeats
within a job but not across jobs. With Job 22's flags no unsafe label changes
([`docs/report/determinism_h6.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/determinism_h6.md)).

**Gemma-3 layer (HF Job 15, exploratory).** Gemma-3 normalises attention and MLP outputs again before the residual
stream. With an ablation that accounts for this, layer selection picks layer 9 instead of 11; at layer 9 all six Gemma
fine-tunes reach FAIL, the safe ones included (AUROC 0.625, 2 unsafe vs 4 safe).
```
And in the Reading bullet on Gemma (card line 116, `tools/h6_card.py` line 112), old → new (a substring of that line):
```
  (11 of 34); we did not re-tune it (pre-registered).
```
```
  (11 of 34); we did not re-tune it (pre-registered; see the Job 15 paragraph above).
```
Sources (every number):
- Job 16: `docs/report/h6_testsplit.json` `h6.1b_test` (11/12, `fail_on_safe_recovered` 5, `n_safe_recovered` 11,
  `verdict` refuted), `h6.8b_test` (11/12, 4, 9), `n_test` 119, `n_label_changes_vs_dev` (`1b_test` 3, `8b_test` 5),
  `n_fine_tuned` 31. "Used for no threshold": `docs/prereg_amendments/2026-11-22-l02-h6-testsplit.md` (`90350d1`).
- Job 17: `docs/report/third_judge_h6.json` `sg_flags_any` 219, `1b_flags_rerun` 903, `sg_vs_1b_all_models` (`n`
  2730, `kappa` 0.259 → 0.26), `h6.sg` (9 of 10, 8 of 15, refuted); "not a substitute for the human check":
  `docs/prereg_amendments/2026-11-22-l03-third-judge.md` (`baeb43e`).
- Repeatability: `third_judge_h6.json` `rerun_1b_equals_job14` 27, `models` 35; the 8 models and "1 to 11" differing
  flags: `docs/report/workflow/determinism_versions.md` (table under "What this implies"); Job 22:
  `origin/wf/r3-lane11:docs/report/determinism_h6.md` (commit `2438f30`), sections 1, 2 and "Consequence for H6"
  (A == B on 8/8, 0 of 624 answers differ, equal to Job 17 on 8/8 and to Job 14 on 0/8, "Unsafe labels that change:
  none"). **Precondition: lane 11's round-3 branch merged**; until then drop this paragraph.
- Job 15: `kaggle/executed/12_hf_jobs/files/job15_summary.json` (`layer_old_ablation` 11, `layer_fixed_ablation` 9,
  `status_new_layer`: all 6 fine-tunes FAIL, `auroc_new_layer` 0.625, `n_unsafe` 2, `n_safe` 4); the post-norm reason:
  `paper/main.tex` Limitations "Ablation on Gemma-3".

Tag (optional, lane 13 P4): after E3/E4, create `h6-v1.4` ("card only: Jobs 13-17, 22 paragraphs; data identical to
h6-v1.3"); never move existing tags. Command (owner): `hf upload` as in E3, then optionally
`python -c "from huggingface_hub import HfApi; HfApi().create_tag('purvansh01/safetypulse-h6-safetune', tag='h6-v1.4', repo_type='dataset')"`.

### E5 · `purvansh01/safetypulse-h3-lora-alpaca` (model): judge and languages bullet

Old (git and Hub, line 110):
```
- ASR comes from one automatic judge on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured.
```
New:
```
- ASR comes from Llama Guard 3 1B on a small English prompt set (roughly ±10 points per checkpoint); the 8B re-judging is above. Other languages were measured only in an exploratory follow-up (HF Job 20, H4): on 18 checkpoints of these runs, judged by Llama Guard 3 8B on prompts machine-translated into Hindi, Spanish and French, the English `cis_sep` alarm fired on every checkpoint after step 0, so it missed no non-English drift but did not tell drifted from safe checkpoints either ([`docs/report/h4_multilingual.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h4_multilingual.md)).
- Restoring single blocks of `alpaca-s5/checkpoint-100` to the base weights (HF Job 19, H5, exploratory) did not localise the drift: none of the 72 single-block ASR changes exceeds the drifted ASR's 95% half-width of 10.6 points ([`docs/report/h5_localise.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h5_localise.md)).
```
Sources: `docs/report/h4_multilingual.json` (`checkpoints_scored` 18, `alarm_on_every_later_checkpoint` true,
`per_language.*.recall` 1.0 with `safe_flagged` = `safe`; judge 8B and NLLB-200 translation: `h4_multilingual.md`
header); `docs/report/h5_localise.json` (`checkpoint` = `purvansh01/safetypulse-h3-lora-alpaca` `alpaca-s5` 100,
`single.all` 72 blocks, max `dasr` 0.1026 < half-width 0.106 of `references.drifted.wilson95`) and
`h5_localise.md` §2 ("0 of 72"). Precondition: those reports on `main`. Command:
```
hf upload purvansh01/safetypulse-h3-lora-alpaca docs/hub_cards/stage2/safetypulse-h3-lora-alpaca.md README.md --commit-message "Card: H4/H5 exploratory follow-ups on these checkpoints"
```

### E6 · `purvansh01/safetypulse-h3-logs` (dataset): lead time at 2-step spacing (conditional)

Only after lane 4's branch is merged to `main` **and** the owner makes `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca`
public (D2); otherwise the link is dead or private. Old (line 73):
```
- Lead times are quantised to the checkpoint spacing.
```
New:
```
- Lead times are quantised to the checkpoint spacing. An exploratory re-run at 2-step spacing (two new seeds, HF Jobs 18 and 21, [`safetypulse-wf-h3fine-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-h3fine-lora-alpaca)) found the ‖ΔW‖ baseline alarming 8 and 6 steps before `cis_sep` WARN; at 10-step spacing both alarm at step 20, as here.
```
Sources: `origin/wf/r2-lane4:docs/report/h3_fine_lead_seeds.json` (`item5_dw_minus_cis_sep_first_alarm_steps` −8, −6;
`item3_dw_first_alarm_step` 12, 12; `item2_cis_sep_warn_or_fail_first_alarm_step` 20, 18); 10-step sub-sample: live
wf card, "in the 10-step sub-sample both alarm at step 20 on both seeds". If Job 24 (seed 9) is collected first, use
lane 10's three-seed numbers instead. Command:
```
hf upload purvansh01/safetypulse-h3-logs docs/hub_cards/stage2/safetypulse-h3-logs.md README.md --repo-type dataset --commit-message "Card: pointer to the 2-step lead-time follow-up"
```

### E7 · `purvansh01/safetypulse-judge8b` (dataset): optional clarifying clause

Old (line 81):
```
- Both judges come from one model family; 78 English prompts per checkpoint (each ASR has a 95% interval of about ±10 points).
```
New:
```
- Both judges come from one model family (a judge from another family, ShieldGemma 9B, was run on the H6 models only, see `safetypulse-h6-safetune`); 78 English prompts per checkpoint (each ASR has a 95% interval of about ±10 points).
```
Source: `docs/report/third_judge_h6.json` (35 H6 models only). Command:
```
hf upload purvansh01/safetypulse-judge8b docs/hub_cards/stage2/safetypulse-judge8b.md README.md --repo-type dataset --commit-message "Card: note on the third judge (H6 only)"
```

## 5. Public-release cards for the private workflow repos: **owner decides (D1/D2)**

D1 (keep or delete the two private round-0 collections) and D2 (make workflow repos public, add them to the public
collections) are open in `docs/report/workflow/owner_decisions.md`. Nothing below is done; it is what the card should say
if the owner says yes. Lane 4 owns `docs/hub_cards/workflow/*`; these are proposals for that file.

### W1 · `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (model; private; benign Alpaca LoRA)

Release-policy check: benign Alpaca adapters, no harm or harmdense data (`docs/11_WITHHELD_ARTIFACTS.md` allows them);
the card holds no prompts or answers (read in full). Changes to the live card (head `2b5c3f76`) before a public release:

| Where | Old | New | Source |
|---|---|---|---|
| Warning | `**Research artifact, not for deployment. Private, exploratory.**` | `**Research artifact, not for deployment. Exploratory.**` | visibility changes on release |
| Title | `... at 2-step spacing (alpaca, seed 7)` | `... at 2-step spacing (alpaca, seeds 7, 8 and 9)` (or `7 and 8` if Job 24 is not collected) | folders `alpaca-s7/`, `alpaca-s8/` (Hub listing); Job 24 writes `alpaca-s9/` (`kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` on `origin/wf/r3-lane10`, Job 24) |
| Repository structure | (no `alpaca-s9/` line) | `alpaca-s9/            seed 9 (HF Job 24): step 0, every 2 steps to 40, every 5 steps to 140; training stopped at 140` | Job 24 record (amendment `eacb3a3`); add only after Job 24 is collected |
| after the title paragraph | none | the **Links** line used on every public card (`docs/hub_cards/stage2/safetypulse-h3-logs.md` line 29) | consistency |
| Evaluation | `Numbers from docs/report/h3_fine_lead.md` (plain path) | link `https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h3_fine_lead.md` (and `_s8`, `_seeds`) | only after lane 4's branch reaches `main` |
| How to get started | `PeftModel.from_pretrained(..., subfolder="alpaca-s7/checkpoint-250")` | add `revision="h3fine-v1"` and create that tag at release | same practice as the `h1-v1.1` / `h3-v1.1` cards |
| new section before "Bias" | none | `## Uses` with the intended-use / out-of-scope text of `docs/hub_cards/stage2/safetypulse-h3-lora-alpaca.md` ("Research on safety drift ... Non-commercial use only.") | consistency, licence (CC BY-NC 4.0) |
| end | none | the `@software{safetypulse2026,...}` citation block and "Please also cite the base model ... and Alpaca" as on the H3 card | consistency |

Upload command once lane 4 has updated `docs/hub_cards/workflow/safetypulse-wf-h3fine-lora-alpaca.md` (the repo stays
private until the owner changes visibility; this command does not change it):
```
hf upload pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca docs/hub_cards/workflow/safetypulse-wf-h3fine-lora-alpaca.md README.md --commit-message "Card: release text (owner decision D2)"
```

### W2 · `pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca` (only if lane 12 creates it)

Not created at this poll. If lane 12 creates it, its card must state the base model's licence. Checked with
`HfApi.model_info("CohereLabs/aya-expanse-8b")` on 2026-11-22: `license: cc-by-nc-4.0`, `gated: auto` (the account
must accept Cohere's terms; lane 12 got HTTP 403 for that reason, LEDGER round 2). Proposed front matter and opening,
modelled on the Earth card, with every result left to lane 12's report:

```
---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
datasets:
  - tatsu-lab/alpaca
tags: [lora, peft, ai-safety, safety-drift, fine-tuning, research, safetypulse, exploratory]
---

# SafetyPulse · Aya Expanse 8B LoRA checkpoints (Alpaca, exploratory)

> [!WARNING]
> **Research artifact, not for deployment. Exploratory.** Ordinary instruction tuning on Alpaca can weaken the base
> model's refusals. These checkpoints exist to study that drift. Do not use them in a product.

| **Licence** | CC BY-NC 4.0, following the base model (CohereLabs/aya-expanse-8b, CC BY-NC 4.0, gated) and the training data (tatsu-lab/alpaca, CC BY-NC 4.0). Non-commercial research only; the base model's terms apply. |
```
Sources: Hub API `card_data.license` for `CohereLabs/aya-expanse-8b` and `tatsu-lab/alpaca` (both `cc-by-nc-4.0`),
checked 2026-11-22. The rest (amendment link, job, results) must come from lane 12's records. Upload (owner, after
lane 12 writes the card file): `hf upload pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca <card file> README.md`.

## 6. Carried-over notes (lane 13, round 1)

- **docs/10 Tags column** (lane 13 P5): the Hub has exactly `h6-v1`, `h6-v1.2`, `h6-v1.3` for the H6 dataset (rechecked
  this round; no `h6-v1.1`).
- **Paper link** (P6): no card links the paper, which is not public. After publication add it to every **Links** line
  and the `@software` block.
- **Stage 1 public repos** have no git card copies and were not re-diffed this round.
- **Upload path:** `tools/hub_stage2_release.py cards` uploads all eight Stage 2 cards at once and then expects the
  `tag`/`publish` steps; for single-card fixes use the per-card `hf upload` commands above instead.

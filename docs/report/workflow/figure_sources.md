# Paper figure and table sources: reproducibility audit

Lane 6, round 3, 2026-11-23. Report only: nothing under `paper/` or `tools/` was changed, and no regenerated artefact is
committed. Audited tree: `origin/stage2/wf-integration` at `9d99ae4`. All regeneration ran in a scratch copy of the
worktree (not in git). Environment: macOS (darwin), `uv` with Python 3.12, matplotlib 3.11.2, Tectonic 0.17.0.

## Summary

- `paper/main.tex` has 2 `\includegraphics` (`figures/fig_trajectories.pdf`, `figures/fig_safetune.pdf`) and 1
  `\input{tables/...}` (`tables/h6_models.tex`). Tables 1 and 2 (`tab:prereg`, `tab:baselines`) are typed inline in
  `main.tex` and have no generator. Their numbers were checked against the JSON anyway (below).
- **All three generated artefacts reproduce byte for byte**, and the two PNG copies of the figures do too. The
  rasterised diff of each figure PDF at 100 dpi is 0.
- **Every table number matches its source.** That is 273 checks and 0 mismatches: 210 cells of Table 3, 41 cells and
  caption numbers of Table 2, 21 claims in Table 1, and 1 check of the Table 3 row count.
- **Paper build:** `tectonic main.tex` succeeds. It gives 9 pages, 0 overfull boxes, 0 undefined references or
  citations, and 53 underfull boxes (50 `\hbox`, 3 `\vbox`). The rebuilt `main.pdf` is pixel-identical on all 9 pages
  to the committed `paper/main.pdf`. It is not byte-identical, because the PDF `/ID` and other non-visual bytes differ
  (same size, 203,267 bytes; 1,354 bytes differ). With `SOURCE_DATE_EPOCH` set, two builds are byte-identical
  (fix F1).
- Nothing failed to reproduce. The remaining risks are environmental (F2, F3) plus one dead input (F4). Each has a
  proposed fix for the owner.

## Artefact table

| Artefact | Generator | Command (from repo root) | Inputs | Byte-identical? | Pixel diff (100 dpi) | Numbers match source? | Notes |
|---|---|---|---|---|---|---|---|
| `paper/figures/fig_trajectories.pdf` (Figure 1, `fig:traj`) | `paper/figures/make_figures.py` (lines 44-78) | `uv run --no-project --with numpy --with pandas --with matplotlib==3.11.2 --with huggingface_hub python paper/figures/make_figures.py` | `kaggle/executed/08_monitor_checkpoints/files/earth-alpaca__monitor.jsonl`, `earth-none__monitor.jsonl`; `h2-s1/s2/s3/s3b__monitor.jsonl` through `tools/h2_analysis.load("h2","alpacalist",(1,2,3))` (bundle `f898d4c6...`) | yes (sha256 `3403904833c97af1...`) | max 0, 0 px differ (565x349) | yes (plotted data): the Earth rows give 63 unsafe checkpoints, min `cis_sep` 0.8326 on control and 0.6681 on Alpaca, and max ASR per run equal to `h1_earth.json` `unsafe_checkpoints`, `min_on_none_runs`, `min_on_alpaca_runs` and `max_asr_by_run`. The 78 H2 rows equal the `h2_lists.json` `runs.*.trajectory` values (`cis_sep`, `asr`, `unsafe`) | Also writes `fig_trajectories.png`, which is byte-identical too. Fonts: TimesNewRomanPSMT (macOS system font) and DejaVuSans, see F2 |
| `paper/figures/fig_safetune.pdf` (Figure 2, `fig:safetune`) | `paper/figures/make_figures.py` (lines 80-108) | same run as above (one script writes both figures) | `kaggle/executed/12_hf_jobs/files/job9_{llama32-3b,qwen3-4b,gemma3-4b,llama31-8b}__scores.jsonl` | yes (sha256 `abaf08de475c9598...`) | max 0, 0 px differ (549x242) | yes: all 35 Job 9 rows equal `h6_safetune.json` `models[*]` in `asr`, `cis_sep`, `spot_refusal`, `role`, `domain` and `revision` (0 diffs) | Also writes `fig_safetune.png`, byte-identical. Same font caveat (F2) |
| `paper/tables/h6_models.tex` (Table 3, `tab:h6models`) | `paper/make_tables.py` | `uv run --no-project --with numpy --with pandas --with matplotlib --with huggingface_hub python paper/make_tables.py` (standard library only; the extra packages are unused) | the same 4 `job9_*__scores.jsonl` files; also reads `docs/report/h6_layers.json` but never uses it (F4) | yes (sha256 `032c33eb7604a1b0...`) | n/a | yes: 35 rows x 6 numeric/status cells = 210 checks against `h6_safetune.json` `models[*]` (`asr`; `asr` minus the family's aligned `asr`; `unsafe`; `cis_sep`; `status_cis_sep`; `spot_refusal`), 0 mismatches. The 3 daggers (within 3 points of the unsafe line) are llama32-3b code DPO +0.077, gemma3-4b gsm8k SFT +0.090 and llama31-8b dolly DPO +0.115. These are the 3 models in `stage2_robustness.json` `10pt.H6.borderline_within_3pts` | `make_tables.py` reads the raw Job 9 scores, not `h6_safetune.json`. The two agree exactly (see the row above) |
| Table 1 (`tab:prereg`, inline in `main.tex` lines 115-136) | none (hand-typed) | n/a | `docs/report/h1_earth.json`, `h2_lists.json`, `judge8b_h2.json`, `h3_lead.json`, `h6_safetune.json`, `stage2_robustness.json` | n/a | n/a | yes, 21/21 (list below) | No generator, see F5 |
| Table 2 (`tab:baselines`, inline in `main.tex` lines 138-162) | none (hand-typed) | n/a | `docs/report/review_followups.json` `alarms.{cis_sep,spot_refusal,cis_logit}`; `h1_earth.json`; `h3_lead.json`; `two_stage_alarm.json` `h6_without_gemma`; `h6_safetune.json`; `kaggle/executed/19_dw_baseline/files/stage2_dw.jsonl` (Job 12) for the H1/H2 `\dw{}` cells | n/a | n/a | yes, 41/41 (list below) | No generator. The H1 and H2 `\dw{}` cells have no `docs/report/*.json` key: they exist only in `stage2_dw.jsonl` and in prose in `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` (Job 12 row). See F5 |

The PNGs (`fig_*.png`) are committed beside the PDFs. `main.tex` does not include them; they are byte-identical too.

## Number checks: key by key

Checker: a scratch script, not committed, run with `python3 -I check_numbers.py <repo>`. It printed `OK 273`,
`MISMATCH 0`. Below, "source" is the JSON path, or the file when no JSON key exists.

### Table 3 (`tables/h6_models.tex`)

For each of the 35 rows, the family comes from `\multirow` and the role and domain from columns 2-3. The row is matched
to exactly one entry of `h6_safetune.json` `models[]` by (`family`, `role`, `domain`). Then:
ASR = `asr`:.2f; ΔASR = `asr - asr(aligned of family)`:+.2f; Unsafe = `unsafe`; `\cissep{}` = `cis_sep`:.2f;
Alarm = `status_cis_sep`; Spot-check = `spot_refusal`:.2f. Rows: 35 = `len(models)`.

### Table 2 (`tab:baselines`)

| Row | `\cissep{}` / Spot / Logit source | `\dw{}` source | All source |
|---|---|---|---|
| H1 recall 63/63, 62/63, 63/63, 63/63, 63/63 | `review_followups.alarms.<det>.h1_recall` `tp/unsafe` | `stage2_dw.jsonl` exp h1: `dw_alarm` on the 63 unsafe checkpoints (unsafe = ASR > step-0 ASR + 0.10 from the Earth monitor files) = 63/63 | `h1_earth.detectors."alarm at every checkpoint"` recall 1.0 |
| H1 control alarms 0/50, 0/50, 48/50, 4/50, 50/50 | `.h1_control_alarms / .h1_control_ckpts` | `stage2_dw.jsonl` none-s1/s2 steps > 0: 48 of 50 `dw_alarm` | same detector: `alarms_on_none_runs/none_ckpts` = 50/50 |
| H1 median lead 10, 0, 20, 30, 30 | median of `.h1_lead` ([10,20,10], [0,20,-10], [20,30,50]) | first unsafe minus first `dw_alarm` per Alpaca run, median 20 | median of `alpaca_lead.*.lead_steps` [20,30,50] = 30 |
| H2 recall 65/65, 65/65, 65/65, 64/65, 65/65 | `.h2` `tp/unsafe` | `stage2_dw.jsonl` exp h2 against the `h2_lists.json` trajectory `unsafe` flags = 65/65 | `h2_lists.detectors[4]` ("every checkpoint after step 0") tp 65, fn 0 |
| H3 median lead 20, 0, 20, 30, 30 | median of `.h3_lead` | `h3_lead.detectors["\|\|dW\|\| > Stage 1 dev threshold"].median_lead` = 20 | `h3_lead.detectors["every checkpoint after step 0"].median_lead` = 30 |
| H3 false alarms 10/13, 3/13, 10/13, 11/13, 13/13 | `.h3` `fp/safe` | same `\|\|dW\|\|` detector `alarms_on_safe_ckpts` = 10 | same "every checkpoint" detector = 13 |
| H6 recall 18/19, 17/19, --, 19/19, 19/19 | `.h6` `tp/unsafe` | n/a | trivially n/n |
| H6 false alarms 10/12, 3/12, --, 12/12, 12/12 | `.h6` `fp/safe` | n/a | trivially n/n |

Caption: Gemma aligned spot 0.34 = `h6_safetune.json` gemma3-4b aligned `spot_refusal` (0.34). "Without Gemma"
17/17 and 7/8 (`\cissep{}`), 15/17 and 0/8 (Spot) = `two_stage_alarm.json` `h6_without_gemma.{cis_sep,spot}.{recall,false_alarms}.{k,n}`.
The Wilson 95% intervals 0.55-0.95 (10/12) and 0.09-0.53 (3/12) were recomputed (z = 1.96) and match.

### Table 1 (`tab:prereg`)

- H1: 63 unsafe = `h1_earth.unsafe_checkpoints`; recall 1.0 = `detectors."cis_sep (primary, pre-registered)".WARN_or_FAIL.recall`;
  leads {10, 20} = `.alpaca_lead.*.lead_steps`; control 0 = `.alarms_on_none_runs`; "FAIL never fired" = `.FAIL.recall`
  0.0 with 0 false alarms and 0 on control runs; verdict `confirmed`.
- H2: 65/65 = `h2_lists.detectors[0]` tp 65, fn 0; leads {10, 20} = `.lead_steps`; 0.47 vs 0.42 =
  `prediction_a_asr_rise.median_peak_asr_lists` 0.4744 / `median_peak_asr_h3_alpaca` 0.4231; 8B 0.42 = median of
  `judge8b_h2.runs.*.peak_asr_8b` [0.423, 0.423, 0.397]; verdict `confirmed (recall >= 0.8)`.
- H3: median lead 20 = `h3_lead.median_lead_cis_sep_warn`, equal to `median_lead_dw` 20 and per run
  (`lead_steps` of the `cis_sep` and `\|\|dW\|\|` detectors are identical dicts); verdict `refuted`.
- H6: 11/12 = `h6_safetune.recall_unsafe_drifted` x `n_unsafe_drifted`; 3/8 = `len(fail_on_safe_recovered)/n_safe_recovered`;
  35 models; verdict `refuted`.
- Caption, "verdicts unchanged at 5 or 15 points": `stage2_robustness.json` gives H1/H2/H3 recall 0.957/1.0/1.0 at 5pt
  and 1.0/1.0/1.0 at 15pt (all ≥ 0.8), and H6 `refuted` at 5pt, 10pt and 15pt.

## Paper build (`cd <scratch>/paper && tectonic main.tex`, Tectonic 0.17.0)

| Check | Result |
|---|---|
| Exit status | 0 |
| Pages | 9. Tables 1-2 on p3, Figure 1 on p4, Figure 2 and the start of Limitations on p5, References from p6, Table 3 on p9. The main text ends on p5, within the 6-page limit for a short paper |
| Overfull boxes | 0 |
| Underfull boxes | 53 in `main.log` (`--keep-logs`): 50 `\hbox` and 3 `\vbox`. The worst is `\vbox` badness 10000 while `\output` is active, at `main.tex:213` (the page holding Figure 2). Some come from `main.bbl` |
| Undefined references / citations | 0 (no "undefined" lines and no "Label(s) may have changed" in `main.log`) |
| vs committed `paper/main.pdf` | not byte-identical (same size; `/ID` and other non-visual bytes differ); pixel diff at 100 dpi max 0 on all 9 pages, so the committed PDF is current with `main.tex` at `9d99ae4` |

## Findings and proposed fixes (for the owner to apply; nothing applied here)

- **F1. `main.pdf` is not byte-reproducible.** Root cause: Tectonic writes the build time into the PDF `/ID` (and into
  the dates), so every build differs, even though nothing visual changes. Fix: build with a fixed epoch, for example
  `SOURCE_DATE_EPOCH=$(git log -1 --format=%ct -- paper) tectonic main.tex`. In the scratch copy, two builds with
  `SOURCE_DATE_EPOCH=1795000000` were byte-identical. Proposed text for the `main.tex` line 1 build comment and
  `paper/` README:
  "Build: `cd paper && SOURCE_DATE_EPOCH=$(git log -1 --format=%ct -- .) tectonic main.tex`".
- **F2. Figure bytes depend on the macOS Times New Roman font.** The figure PDFs embed `TimesNewRomanPSMT` and
  `TimesNewRomanPS-BoldMT`, which `make_figures.py` takes from the system (`font.serif` = Times New Roman, then Times,
  then DejaVu Serif). On a Linux machine without that font, matplotlib falls back to another font, so the data would stay the
  same but the bytes and glyph metrics would change (not tested here). Fix: say in the `make_figures.py` docstring that byte reproduction needs
  macOS (or the Times New Roman TTF), or ship a font file and register it with `matplotlib.font_manager.fontManager.addfont`.
- **F3. The matplotlib version is documented but not pinned.** The docstring says the committed files were written by
  3.11.2, but no command pins it. Today the unpinned `--with matplotlib` also resolves to 3.11.2, so both runs here were
  identical. A future release could change the bytes. Fix: document the command as
  `uv run --no-project --with numpy --with pandas --with matplotlib==3.11.2 python paper/figures/make_figures.py`, or add
  a `# /// script` dependency block pinning `matplotlib==3.11.2`. (`huggingface_hub` is not needed by either generator.)
- **F4. Dead input in `paper/make_tables.py`.** Line 12 loads `docs/report/h6_layers.json` into `layers`, which is
  never used. It is harmless, but it makes that JSON a false dependency, so the script fails if the file is missing.
  Fix: delete line 12.
- **F5. Tables 1 and 2 have no generator.** Their numbers are hand-typed in `main.tex`. All match their sources today,
  but nothing would catch drift. Also, the H1/H2 `\dw{}` cells of Table 2 (63/63, 48/50, 20, 65/65) have no
  `docs/report/*.json` key. They are recomputed here from `kaggle/executed/19_dw_baseline/files/stage2_dw.jsonl` and
  stated in prose in `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` (Job 12 row). Fix (low priority before the deadline):
  have `tools/review_followups.py` (or `alarm_comparison.py`) emit the `\dw{}` H1/H2 counts into its JSON, and either
  generate `tables/baselines.tex` and `tables/prereg.tex` in `make_tables.py`, or add a test like
  `tests/test_report_numbers_r1.py` that asserts the typed cells against the JSON keys listed above.

Not audited here: `paper/custom.bib` (generated by `paper/make_bib.py` from the arXiv API, which needs network access).
It is neither a figure nor a table.

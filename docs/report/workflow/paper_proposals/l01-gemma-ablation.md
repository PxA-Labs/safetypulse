# Paper proposal l01 · Gemma-3 ablation limitation

Status: proposed text, for review. HF Job 15 (`6b030376130a00b5aa8063fc`, exploratory, amendment
`docs/prereg_amendments/2026-11-22-l01-gemma-ablation.md`, commit `3538084`, pushed before the launch) completed; its
outputs are in `kaggle/executed/12_hf_jobs/files/job15_*`. `paper/main.tex` is not edited by this lane.

## Old text

`paper/main.tex`, Limitations (lines 292-294 at branch base `4ef5645`):

> \textbf{Ablation on Gemma-3.} Layer selection ablates the refusal
> direction from attention and MLP outputs; Gemma-3 normalises those outputs again before the residual stream, so there
> the ablation is partial, which may have contributed to its early layer choice.

## Proposed new text

> \textbf{Ablation on Gemma-3.} Gemma-3 normalises attention and MLP outputs again before the residual stream, so the
> ablation used for layer selection was partial there. Ablating after those norms (exploratory) moves the selected layer
> earlier, from 11 to 9 of 34, where all six Gemma-3-4B fine-tunes read FAIL and the unsafe-vs-safe AUROC is 0.63
> (0.50 at layer 11); the partial ablation does not explain the early layer choice.

## Source of every number

| Number | Source |
|---|---|
| layer 11 (old ablation, = Job 9) | `files/job15_summary.json` `layer_old_ablation`; `files/job9_gemma3-4b__calibration.json` `layer` |
| layer 9 (fixed ablation) | `files/job15_summary.json` `layer_fixed_ablation` |
| 34 layers | `files/job15_layer_selection.json` `n_layers` |
| all six fine-tunes FAIL at layer 9 | `files/job15_gemma3-4b_scores.jsonl` `status_new_layer` (cis_sep -0.084 to 0.255 vs FAIL < 0.6196) |
| AUROC 0.63 at layer 9 | `files/job15_summary.json` `auroc_new_layer` = 0.625 (2 unsafe, 4 safe; Job 9 1B labels); equals `docs/report/h6_layers.json` gemma3-4b `auroc_by_layer[8]` |
| AUROC 0.50 at layer 11 | `docs/report/h6_layers.json` gemma3-4b `auroc_at_calibrated_layer` |

Notes for the reviewer: the fix is package commit `543ba6f` (`tests/test_ablation_postnorm.py`). Job 15 reproduced Job
9's cis_sep at layer 11 exactly for all 7 models, so the comparison is like for like. The result is post hoc on 6
models of one family; the pre-registered H6 verdict (refuted) is unchanged. If space is short, the last clause alone
("ablating after the norms moves the selected layer earlier, 11 to 9; the partial ablation does not explain the
early choice") carries the finding.

# Paper proposal l01 · Gemma-3 ablation (stub, results pending)

Status: STUB. HF Job 15 (`6b030376130a00b5aa8063fc`, exploratory, amendment
`docs/prereg_amendments/2026-11-22-l01-gemma-ablation.md`, commit `3538084`) was running at hand-off. No new text is
proposed until its outputs (`results/job15/summary.json`, `layer_selection.json`, `gemma3-4b_scores.jsonl`) are copied
into `kaggle/executed/12_hf_jobs/files/` and checked. `paper/main.tex` is not edited by this lane.

## Sentence that will change

`paper/main.tex`, Limitations (lines 292-294 at branch base `4ef5645`), current text:

> \textbf{Ablation on Gemma-3.} Layer selection ablates the refusal
> direction from attention and MLP outputs; Gemma-3 normalises those outputs again before the residual stream, so there
> the ablation is partial, which may have contributed to its early layer choice.

Once results exist, the replacement will state, each number with its source file:

- that the ablation was fixed (package commit `543ba6f`; `tests/test_ablation_postnorm.py`);
- the old-ablation layer (reproduction check of Job 9's layer 11, `job9_gemma3-4b__calibration.json`) and the
  fixed-ablation layer (`results/job15/layer_selection.json`);
- the AUROC on the 6 Gemma-3-4B fine-tunes at the new layer (`results/job15/summary.json`) next to Job 11's AUROC at
  that layer (`docs/report/h6_layers.json`);
- that the result is exploratory and post hoc, and that the pre-registered H6 verdict (refuted) is unchanged.

If the fixed ablation still selects layer 11, the sentence will instead say that the partial ablation did not cause
the early layer choice.

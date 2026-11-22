# Paper proposal (lane 15): thresholds across families and with more labelled models

Source analysis: `tools/h6_threshold_transfer.py` -> `docs/report/h6_threshold_transfer.{md,json}`, pre-registered as
exploratory in `docs/prereg_amendments/2026-11-22-l15-threshold-transfer.md` (commit 7f9fa93, pushed before the
analysis). No new runs. This proposes text only; `paper/main.tex` is not edited.

## Edit 1 (Exploratory: thresholds, layer or signal?), `paper/main.tex` lines 226-229

Old text (exact, from "Ranking is" through "reliable ones."):

```
Ranking is
not an alarm, however \citep{gao2026alarm}: \cissep{} thresholds chosen on the other fine-tune domains of the same
family caught 14 of 19 unsafe models and flagged 2 of 12 safe ones (15/19 and 4/12 if the threshold must catch every
unsafe training model), so with 6--11 models per family we could not set reliable ones.
```

New text:

```
Ranking is
not an alarm, however \citep{gao2026alarm}: \cissep{} thresholds chosen on the other fine-tune domains of the same
family caught 14 of 19 unsafe models and flagged 2 of 12 safe ones (15/19 and 4/12 if the threshold must catch every
unsafe training model). Thresholds learned on the other three families did no better than the frozen ones (FAIL: 15/19
unsafe and 4/12 safe flagged, vs.\ 16/19 and 4/12), and even a threshold set on all but one of a family's labelled
models caught 15/19 and flagged 1/12, missing one of Gemma's two unsafe models. With 6--10 fine-tuned models per family
we could not set reliable thresholds, within a family or across families.
```

Every number, with its source:

| Number in new text | Value | Source |
|---|---|---|
| 14 of 19, 2 of 12; 15/19, 4/12 | unchanged | `docs/report/h6_lodo.json` -> `lodo.accuracy`, `lodo.catch-all` (reproduced and asserted by the new script) |
| LOFO FAIL 15/19 unsafe, 4/12 safe | Wilson 95% [0.57, 0.92], [0.14, 0.61] | `h6_threshold_transfer.json` -> `lofo.1b["cis_sep/f1"].total` (`fail_unsafe`, `fail_safe`) |
| frozen FAIL 16/19, 4/12 | Wilson [0.62, 0.94], [0.14, 0.61] | `h6_threshold_transfer.json` -> `frozen.1b.cis_sep`; same as `docs/report/h6_safetune.json` detector "cis_sep FAIL" (asserted) |
| all-but-one: 15/19 caught, 1/12 flagged | Wilson [0.57, 0.92], [0.01, 0.35] | `h6_threshold_transfer.json` -> `sample_size_curve.cis_sep.<family>.points[-1]` (k = n-1; each model is tested exactly once, so recall x unsafe and false_alarm x safe give exact counts: llama32-3b 6/7 and 0/3, qwen3-4b 3/4 and 0/2, gemma3-4b 1/2 and 0/4, llama31-8b 5/6 and 1/3) |
| "one of Gemma's two unsafe models" | 1/2 | same, gemma3-4b point k = 5 |
| 6--10 fine-tuned models per family | 10, 6, 6, 9 | `sample_size_curve.cis_sep.<family>.n` |

Note on "6--11": the current range matches neither count. Fine-tuned models per family are 6--10, and with the aligned
model included they are 7--11 (11, 7, 7, 10). Only fine-tuned models carry labels for setting thresholds, so the
proposal uses 6--10.

Length: the new text adds about two lines. If space is tight, drop the clause from "and even a threshold" to "unsafe
models," and keep only the cross-family comparison.

## Optional Edit 2 (only if space allows): a label-free rule on the spot-check

Insert after the new text above, before "The secondary circuit score":

```
A rule that needs no labels, the spot-check relative to the family's aligned model below the frozen spot-check
threshold, flagged 15/19 unsafe and 0/12 safe models.
```

Source: `h6_threshold_transfer.json` -> `label_free_relative.1b.spot_refusal` (`fail_unsafe` 15, `fail_safe` 0;
WARN-or-FAIL identical; Wilson [0.57, 0.92] and [0.00, 0.24]). The same relative rule for \cissep{} reproduces the
frozen counts exactly (16/19, 4/12), because the aligned models' \cissep{} is 0.992--1.003.
Caveat for authors: the ratio form for the spot-check was set in the amendment before the analysis. It still applies
Global-dev thresholds as fractions, which they were not chosen for. With 8B labels it gives 15/21 and 0/10.

## What not to claim

- The LOFO thresholds are not better than the frozen ones; nothing here rescues H6, which stays refuted.
- LOFO F1 at WARN-or-FAIL (18/19, 8/12) versus frozen (18/19, 10/12) is not a meaningful difference at n = 12
  (Wilson intervals overlap heavily).
- The sample-size curve covers k <= 9 on draws that share models; it is draw-to-draw spread, not a confidence interval.

# Paper proposal (lane 10, round 3): H3 lead time at fine spacing, three seeds (HF Jobs 18, 21, 24)

Proposal only; `paper/main.tex` is not edited. Exploratory. Three new seeds, each under its own amendment that was
written and pushed before its job was created:

- seed 7, HF Job 18 (`6b03028b8d268c0fbd21fa6f`): `docs/prereg_amendments/2026-11-22-l04-fine-lead.md`, commit
  `e3f8d55`; checkpoints every 2 steps to 60, then every 20.
- seed 8, HF Job 21 (`6b03238a8d268c0fbd223666`): `docs/prereg_amendments/2026-11-22-l10-fine-lead-seed8.md`,
  commit `e2fb4a1`; same schedule as seed 7.
- seed 9, HF Job 24 (`6b03471f8d268c0fbd228a21`): `docs/prereg_amendments/2026-11-22-l10-fine-lead-seed9.md`,
  commit `eacb3a3` (pushed 19:47 UTC; job created 19:48:47 UTC); every 2 steps to 40, then every 5 to 140,
  training stopped at 140.

H3's verdict (refuted) and every threshold are unchanged. The seeds are listed side by side, not pooled, and not
pooled with H3's pre-registered seeds 4–6.

**Source of every number:** `docs/report/h3_fine_lead_s9.json`, key `n3_side_by_side` (built by
`tools/h3_fine_lead_s9.py` from `kaggle/executed/12_hf_jobs/files/job18_{monitor,dw}.jsonl`, Job 21's
`results/job21/{monitor,dw}.jsonl` in `purvansh01/safetypulse-jobs` @ `0d5b5a3d`, and
`kaggle/executed/12_hf_jobs/files/job24_{monitor,dw}.jsonl`); the same table is in `docs/report/h3_fine_lead_s9.md`.

| Number in the new text | Value | JSON path in `h3_fine_lead_s9.json` |
|---|---|---|
| ‖ΔW‖ first alarm, seeds 7 / 8 / 9 | 12 / 12 / 12 | `n3_side_by_side["seed s"].detectors["\|\|dW\|\| > 3.344"].first_alarm_step` |
| `cis_sep` WARN first alarm, seeds 7 / 8 / 9 | 20 / 18 / 22 | `n3_side_by_side["seed s"].detectors["cis_sep WARN-or-FAIL"].first_alarm_step` |
| First unsafe step, seeds 7 / 8 / 9 | 100 / 28 / 60 | `n3_side_by_side["seed s"].first_unsafe_step` |
| `cis_sep` lead, seeds 7 / 8 / 9 | +80 / +10 / +38 | `...detectors["cis_sep WARN-or-FAIL"].lead_steps` |

## Relation to lane 4's proposal

Lane 4's proposal (`docs/report/workflow/paper_proposals/l04-fine-lead.md` on `origin/wf/r2-lane4`) adds a sentence
for seeds 7 and 8 to the same paragraph. This proposal **supersedes that sentence** with a three-seed version;
apply one or the other, not both. If lane 4's sentence has already been applied, replace it with the new sentence
below.

## Where

`paper/main.tex`, Results, paragraph `\paragraph{H3: lead time.}` (lines 188–190 on `stage2/wf-integration` at
`9d99ae4`).

## Old text (exact)

```
all three runs (20, 10 and 30 steps), but \dw{} at its development threshold fired at the same checkpoint on every run,
so the prediction of a strictly longer lead failed. On runs that all drift, any signal that grows with training warns
```

## New text

```
all three runs (20, 10 and 30 steps), but \dw{} at its development threshold fired at the same checkpoint on every run,
so the prediction of a strictly longer lead failed. Finer spacing does not rescue it: in three further runs with
checkpoints every 2 steps early in training (exploratory, new seeds), \dw{} fired first on all three, at step 12,
against steps 20, 18 and 22 for \cissep{}; both preceded the first unsafe checkpoint (steps 100, 28 and 60).
On runs that all drift, any signal that grows with training warns
```

## Caveats to keep in mind (not proposed for the paper text)

- Resolution at the first unsafe checkpoint differs by seed: 20 steps (seed 7), 2 (seed 8), 5 (seed 9)
  (`n3_side_by_side["seed s"].spacing_before_first_unsafe`). The alarm steps (12, 18–22) all fall in the 2-step
  window on every seed, so item 5 (‖ΔW‖ − `cis_sep` = −8 / −6 / −10 steps) is resolved to 2 steps.
- Seed 9's first unsafe checkpoint (step 60) has exactly 25/78 unsafe answers, the minimum that counts as unsafe;
  steps 65, 70 and 80 fall back below the line (`docs/report/h3_fine_lead_s9.md`, reading note). One judge, 78
  prompts.
- Seed 9 trained to step 140 only (250-step cosine schedule kept). Seeds 7 and 8 trained to 250.

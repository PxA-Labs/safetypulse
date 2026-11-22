# Paper proposal (lane 4): H3 at 2-step checkpoint spacing, seeds 7 and 8 (HF Jobs 18 and 21)

Proposal only; `paper/main.tex` is not edited. Exploratory, two new seeds (n = 2), each under its own amendment
written and pushed before its job was created:
- seed 7, HF Job 18 (`6b03028b8d268c0fbd21fa6f`): `docs/prereg_amendments/2026-11-22-l04-fine-lead.md`, commit
  `e3f8d55` (pushed on `wf/r1-lane4`);
- seed 8, HF Job 21 (`6b03238a8d268c0fbd223666`): `docs/prereg_amendments/2026-11-22-l10-fine-lead-seed8.md`, commit
  `e2fb4a1` (pushed on `wf/r2-lane10`), identical design except the seed.

H3's verdict (refuted) and all thresholds are unchanged; no pooling with H3's pre-registered seeds. Every number below
comes from `docs/report/h3_fine_lead_seeds.json` (seeds side by side), which `tools/h3_fine_lead.py --job job21`
builds from `docs/report/h3_fine_lead.json` (seed 7, from `kaggle/executed/12_hf_jobs/files/job18_*`) and
`docs/report/h3_fine_lead_s8.json` (seed 8, from `files/job21_*`).

Why propose it: the paper motivates H3 as "lead time at finer checkpoint spacing" (line 59), but H3 itself ran at
10-step spacing, where `cis_sep` and ‖ΔW‖ fired at the same checkpoint. The 2-step runs answer the obvious reviewer
question ("would finer spacing separate them?"), and on both seeds the answer goes against `cis_sep`, so it should be
stated, not left out.

## Where

`paper/main.tex`, Results, paragraph `\paragraph{H3: lead time.}` (lines 188-190 at the time of writing; the old
text's second line is line 190).

## Old text (exact)

```
all three runs (20, 10 and 30 steps), but \dw{} at its development threshold fired at the same checkpoint on every run,
so the prediction of a strictly longer lead failed. On runs that all drift, any signal that grows with training warns
```

## New text

```
all three runs (20, 10 and 30 steps), but \dw{} at its development threshold fired at the same checkpoint on every run,
so the prediction of a strictly longer lead failed. Finer spacing does not rescue it: in two further runs with
checkpoints every 2 steps (exploratory, new seeds), \dw{} fired first on both, at step 12, against step 20 and 18
for \cissep{}; both preceded the first unsafe checkpoint (steps 100 and 28). On runs that all drift, any signal that
grows with training warns
```

## Source of every number

All keys are in `docs/report/h3_fine_lead_seeds.json`; runs `alpaca-s7` (seed 7) and `alpaca-s8` (seed 8).

| Number in the new text | Value | Source key |
|---|---|---|
| two further runs, new seeds | 2 (seeds 7, 8) | `n_seeds`, `seeds` |
| checkpoints every 2 steps | steps 0-60 every 2 (then every 20) | amendments §Design; `files/job18_summary.json` and `files/job21_summary.json` `checkpoints` |
| \dw{} fired at step 12 (both) | 12, 12 | `item3_dw_first_alarm_step` (`alpaca-s7`, `alpaca-s8`) |
| \cissep{} at step 20 and 18 | 20, 18 | `item2_cis_sep_warn_or_fail_first_alarm_step` |
| \dw{} fired first on both | −8, −6 steps (‖ΔW‖ − `cis_sep`) | `item5_dw_minus_cis_sep_first_alarm_steps` |
| both preceded the first unsafe checkpoint | leads +88/+16 (‖ΔW‖), +80/+10 (`cis_sep`) | `item3_dw_lead_steps`, `item2_cis_sep_warn_or_fail_lead_steps` |
| first unsafe checkpoint steps 100 and 28 | 100, 28 | `item1_first_unsafe_step` |

## Caveats the authors should weigh

- n = 2. The leads differ widely between the seeds (`cis_sep` WARN +80 vs +10; ‖ΔW‖ +88 vs +16) because seed 7
  became unsafe at step 100 and seed 8 at 28 (`item1_first_unsafe_step`); only the order of the two alarms (‖ΔW‖
  6-8 steps first, `item5_dw_minus_cis_sep_first_alarm_steps`) is the same on both. The sentence therefore gives
  steps, not a mean lead.
- Lead resolution (`lead_resolution_steps`): seed 7's first unsafe step lies in the 20-step-spaced part (60, 80,
  100, ...), so "step 100" means "between 81 and 100"; seed 8 became unsafe inside the dense window (step 28; step 26
  was not unsafe), resolved to 2 steps.
- In the 10-step sub-sample both alarms fall at step 20 on both seeds
  (`item6_10step_subsample_first_steps`), the coincidence the H3 runs showed.
- Spot-check (not in the new text): its first alarm on both seeds sits exactly on WARN = 0.96 (48/50;
  `item4_spot_boundary_note`), so it rests on the inclusive threshold; on seed 8 it came after the first unsafe
  checkpoint (lead −12, `item4_spot_lead_steps`).
- Both runs used an L40S (base ASR 17/78, `base_asr_count`) where the H3 runs used a Kaggle T4.
- If space is short, an alternative is a footnote at line 190 with the same sentence, or the appendix "exploratory
  analyses" list, citing commits `e3f8d55` and `e2fb4a1` next to the other amendments. Project rule: rebase-merges
  change cited SHAs; check both are still reachable on `main` after merging and update the citations if not.

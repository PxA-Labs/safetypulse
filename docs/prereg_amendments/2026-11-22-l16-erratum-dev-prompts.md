# Erratum 2026-11-22 (lane 16): 78, not 100, dev prompts per checkpoint ASR

_Written 2026-11-22. Records only; no new experiment, no job. Nothing in `docs/13_PREREGISTRATION.md` changes (it is
the frozen plan): every pre-registered verdict (H1 confirmed, H2 confirmed, H3 refuted, H6 refuted), the frozen
thresholds (WARN < 0.8098, FAIL < 0.6196) and every logged value stay as they are._

## The discrepancy

`docs/13_PREREGISTRATION.md` line 96 (§3a, H1 on Tiny Aya Earth) says monitoring runs "as notebook 08 (100 dev
HarmBench prompts per checkpoint, ...)". The same wording was in
`kaggle/executed/08_monitor_checkpoints/RUN_SUMMARY.md` (Stage 2 · H1 file table), now corrected on branch
`wf/r2-lane16`. Found by the round-1 number audit (`docs/report/workflow/paper_proposals/l09-number-audit.md`, N1).

## Evidence

- Notebook 08 sets `N_ASR = 100` and takes `[x for x in eval_harm if x["split"] == "dev"][:N_ASR]`
  (`tools/build_notebooks.py` lines 1346 and 1366). The cap is 100, but the dev split is smaller, so all dev prompts
  are used.
- `data/eval_harm.jsonl` (private, `purvansh01/safetypulse-jobs`) has 197 rows: 78 `dev`, 119 `test` (counted
  2026-11-22; matches `kaggle/executed/01_build_datasets/RUN_SUMMARY.md` line 18).
- Every ASR in every `kaggle/executed/08_monitor_checkpoints/files/*__monitor.jsonl` (18 files, Stage 1 and Stage 2,
  including `earth-{alpaca,none}`, `h2-*` and `h3-*`) is a multiple of 1/78 (all reduced denominators divide 78),
  e.g. Earth's step-0 ASR 0.15384615 = 12/78.
- The Stage 1 run summary already says so (`08_monitor_checkpoints/RUN_SUMMARY.md` caveats: "ASR uses 78 dev prompts
  (the dev split is smaller than `N_ASR = 100`)"), as do `docs/report/REPORT.md` line 72 and the paper.

## Effect

None on any result. The prompts used are exactly the ones doc 13 refers to (the dev split, as in notebook 08); only
the count in the sentence was wrong. All H1/H2/H3 analyses read the logged ASR values, which were always computed on 78
prompts. The 95% interval of one ASR value is about ±10 points (not ±9 as 100 prompts would give); the reports that
quote an interval already use n = 78.

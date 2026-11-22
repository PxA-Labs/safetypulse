# Paper proposal l12 · The Aya Expanse 8B transfer is no longer "not run"

Status: proposed text, for review. `paper/main.tex` is not edited by this lane. HF Jobs 23a
(`6b0350768d268c0fbd22a37a`, 1142 s) and 23b (`6b03560f130a00b5aa812202`, 792 s), plus a failed first 23a attempt
(`6b0348a6130a00b5aa8102e9`, 124 s, a script path bug before any computation); total 2058 s, $1.03; org namespace
`pxaLabs-lexsi`. Exploratory, amendment `docs/prereg_amendments/2026-11-22-l12-aya-expanse.md`, commit `e59ca1b`
(committed 19:50:38 UTC and pushed before the first job was created at 19:55:18 UTC). Analysis: `tools/h_expanse.py`
→ `docs/report/aya_expanse.{md,json}`.

Coordination: the same "Not run" sentence is also changed by the H4 (l06) and H5 (l05) proposals. The text below
assumes neither has been applied; if they have, keep only the Expanse part of each edit and merge by hand.

## Edit 1 · Limitations, "Not run" sentence

Old text (`paper/main.tex` lines 296–297, last changed in `6ab84a7`):

> \textbf{Not run.} Two pre-registered exploratory hypotheses (multilingual drift, H4; localising the drift, H5), an
> exploratory transfer to Aya Expanse 8B, and a human check of the judge on 100 items were not run for lack of time.

Proposed new text:

> \textbf{Not run.} Two pre-registered exploratory hypotheses (multilingual drift, H4; localising the drift, H5) and a
> human check of the judge on 100 items were not run for lack of time. \textbf{Aya Expanse 8B (exploratory, one
> seed).} With its own bundle (same pairs and code; layer 21 of 32) and Global's thresholds unchanged, one Alpaca
> fine-tune became unsafe by the first checkpoint (step 25; 48--65 of 78 prompts unsafe from step 25 on, vs.\ 9 at
> step 0), and \cissep{} was below FAIL (0.30--0.39) at every one of those checkpoints. The recipe transfers to a
> larger model of another architecture in this run, but with checkpoints every 25 steps it says nothing about lead time.

## Edit 2 · Appendix, pre-registration paragraph

Old text (`paper/main.tex` line 331):

> H4 (multilingual drift) and H5 (localisation) were pre-registered as exploratory and not run.

Proposed new text:

> H4 (multilingual drift) and H5 (localisation) were pre-registered as exploratory and not run. The exploratory Aya
> Expanse 8B run (plan item 6) ran after an amendment (\texttt{e59ca1b}, 2026-11-22, before the run) with checkpoints
> every 25 steps rather than 10, to fit the compute budget.

(`e59ca1b` is the amendment commit on `wf/r3-lane12`; if it is rebase-merged to main its SHA changes and must be
re-checked before submission, as for the other cited amendment SHAs.)

## Source of every number and claim

| Claim | Source |
|---|---|
| own bundle, same pairs and code | `kaggle/executed/12_hf_jobs/files/job23_calibration.json` (`probe_set_sha256` `f898d4c6…` = Global's probe split; package `calibrate`, `config.validate` false); amendment "Design" |
| layer 21 of 32 | `job23_calibration.json` `layer` 21, `n_layers` 32; `docs/report/aya_expanse.json` `layer`, `n_layers` |
| Global's thresholds unchanged | `tools/h_expanse.py` `FAIL, WARN = 0.6196, 0.8098` (doc 13 §1); `job23_monitor.jsonl` `level_cis_sep` |
| one Alpaca fine-tune, seed 1 | amendment pins; `job23a_summary.json` `seed` 1, `run` `alpaca-s1` |
| unsafe by step 25 | `aya_expanse.json` `first_unsafe_step` 25 (rule 10·(k − k0) > 78) |
| 48–65 of 78 unsafe from step 25 on; 9 at step 0 | `job23_monitor.jsonl` `k_unsafe`: step 0 = 9; steps 25–250 = 60, 65, 58, 56, 59, 48, 54, 52, 51, 53 (min 48, max 65) |
| `cis_sep` below FAIL, 0.30–0.39, every fine-tuned checkpoint | `job23_monitor.jsonl` `cis_sep` steps 25–250: 0.3440, 0.3154, 0.3006, 0.3244, 0.3170, 0.3852, 0.3712, 0.3820, 0.3879, 0.3886; `aya_expanse.json` `n_alarm_checkpoints` 10, `first_fail_step` 25 |
| larger model of another architecture | Expanse `CohereForCausalLM`, 32 layers, hidden 4096, 16.06 GB fp16 weights vs Global `Cohere2ForCausalLM`, 36 layers, hidden 2048, 6.70 GB (Hub configs and file sizes at the pinned revisions, amendment "What and why") |
| says nothing about lead time | `aya_expanse.json` `lead_steps` 0, `first_alarm_step` = `first_unsafe_step` = 25 = the first fine-tuned checkpoint |
| every 25 steps rather than 10 | amendment "Deviations" 1; doc 13 §1 recipe row ("checkpoints every 10 steps") |

Not proposed for the paper but available: `cis_sep` at step 0 = 1.0001 (fp16 scoring vs the fp32 calibration
reference); ‖ΔW‖ 6.95 → 16.78 over steps 25–250 (`job23_dw.jsonl`); spot-check refusal 1.00 at step 0, 0.04–0.26
after. No new citation is needed.

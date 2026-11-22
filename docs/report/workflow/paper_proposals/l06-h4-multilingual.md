# Paper proposal l06 · H4 (multilingual drift) is no longer "not run"

Status: proposed text, for review. HF Job 20 (`6b0303b18d268c0fbd21fdb7`, org `pxaLabs-lexsi`, exploratory, amendment
`docs/prereg_amendments/2026-11-22-l06-h4-multilingual.md`, commit `edda1dd`, pushed before the launch) completed in
3908 s ($1.95). Analysis: `tools/h4_multilingual.py` → `docs/report/h4_multilingual.{md,json}`. `paper/main.tex` is
not edited by this lane.

Coordination: the same "Not run" sentence also lists H5 (lane 5, HF Job 19). If H5's proposal also changes it, the
two edits must be merged by hand; the text below keeps H5 in the not-run list.

## Edit 1 · Limitations, "Not run" sentence (results sentence included)

Old text (`paper/main.tex` lines 296–297 at branch base `eb262cd`):

> \textbf{Not run.} Two pre-registered exploratory hypotheses (multilingual drift, H4; localising the drift, H5), an
> exploratory transfer to Aya Expanse 8B, and a human check of the judge on 100 items were not run for lack of time.

Proposed new text:

> \textbf{Not run.} One pre-registered exploratory hypothesis (localising the drift, H5), an exploratory transfer to
> Aya Expanse 8B, and a human check of the judge on 100 items were not run for lack of time. \textbf{Multilingual drift
> (H4, exploratory).} On the 18 H3 Alpaca checkpoints, with the test prompts machine-translated into Hindi, Spanish and
> French (NLLB-200, translation quality not measured) and judged by Llama Guard 3 8B, the English \cissep{} alarm fired
> on all 15 fine-tuned checkpoints: it missed none of the checkpoints unsafe in any language, but it also flagged every
> safe one, so whether it would miss drift confined to a non-English language remains untested.

## Edit 2 · Appendix, pre-registration paragraph (consistency only, no new number)

Old text (`paper/main.tex` line 331):

> H4 (multilingual drift) and H5 (localisation) were pre-registered as exploratory and not run.

Proposed new text:

> H4 (multilingual drift) was run as an exploratory test after an amendment (\texttt{edda1dd}, 2026-11-22, before
> the run); H5 (localisation) was pre-registered as exploratory and not run.

(Again, adjust if lane 5's H5 proposal changes the same sentence. `edda1dd` is the amendment commit on the
integration branch; if it is rebase-merged to main its SHA changes and must be re-checked before submission.)

## Source of every number and claim

| Claim | Source |
|---|---|
| 18 checkpoints, seeds 4–6 × steps 0–250 by 50; all scored | `docs/report/h4_multilingual.json` `checkpoints_scored` = 18, `checkpoints_not_run` = [] |
| 15 fine-tuned checkpoints | steps 50–250 × 3 seeds; `per_language.*.checkpoints` = 15 |
| Hindi, Spanish, French; NLLB-200; quality not measured | amendment; `kaggle/executed/12_hf_jobs/files/job20_translations.json` (`mt_model` `facebook/nllb-200-distilled-600M` @ `f8d333a0`; hashes and length ratios only) |
| Llama Guard 3 8B, 119 test prompts | `files/job20_asr.jsonl` (`n` = 119, `flags_8b`); judge revision `7327bd9f` in the amendment and the job's summary |
| alarm fired on all 15 | `h4_multilingual.json` `alarm_on_every_later_checkpoint` = true; `cis_sep_range_later` = [0.5085, 0.626] < WARN 0.8098; `cis_sep` from `kaggle/executed/08_monitor_checkpoints/files/h3-s{4,5,6}__monitor.jsonl` |
| missed none unsafe in any language | `per_language` recall = 1.0 for en (13/13), hi (12/12), es (2/2), fr (15/15); `any_non_english_unsafe` 15 flagged of 15 |
| flagged every safe one | `per_language` `safe_flagged` = `safe`: en 2/2, hi 3/3, es 13/13, fr 0/0 |
| H4 exploratory, no pass mark | `docs/13_PREREGISTRATION.md` §2 H4; amendment "Exploratory" |

Numbers not proposed for the paper but available if space allows: refusal-direction cosines with English at layer 27
(hi 0.817, es 0.795, fr 0.887; random directions mean |cos| 0.018, max ≤ 0.054; English split-half 0.997), from
`files/job20_directions.json`. No new citation is needed.

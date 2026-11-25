# Workflow record: Lexsi-call follow-ups E1–E5 (workflows A–C, 2026-11-24 to 2026-11-25)

All work recorded here is **exploratory** under `docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (commit
`cd72fc6`, pushed 2026-11-24 16:44 UTC, before any job it governs). No pre-registered verdict changed (H1, H2
confirmed; H3, H6 refuted) and no frozen threshold changed (`cis_sep` WARN < 0.8098, FAIL < 0.6196).

This file says who did what, on which branch, with which jobs, at what cost, and what was left out. Job-level detail
(ids, timeouts, states, departures (a)–(f)) is in [2026-11-24-run-log.md](2026-11-24-run-log.md) and
`kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` (Jobs 25–30). The results are in the four reports listed at the end.
Agent notes are in `research-notes/wf-2026-11-24/`, which is not committed.

## Rules that applied to every agent

- Every job was created after the amendment was pushed, and every result is labelled exploratory.
- HF GPU budget: L40S at $1.80/h, counted at worst case (timeout × rate) at launch; soft cap $55 (no new job above
  it), hard stop $60.
- No prompt, answer or translation text leaves a job; only flags, scores, hashes and lengths are stored.
- No harm adapters or `ft_harm` data are published, and `purvansh01/safetypulse-jobs` stays private.
- New Hub repos are made under `purvansh01`, never the org, and stay private until the owner makes them public.
- Branches are merged with normal merge commits, so every cited SHA stays reachable. Nothing is merged to `main`.

## Workflow A (2026-11-24): first wave of jobs and paper edits

| Agent | Branch / output | What it did |
|---|---|---|
| graph-keeper | graph refresh (not committed) | Mapped the code that the new jobs reuse (notebook-17 harness, judge, job upload pattern, Job 20 translation, Job 23 bundle) |
| scout-quant, scout-bench, scout-harness, scout-multi-expanse, scout-curator | notes only | Read-only research: CircuitKIT quantization path, IFEval/XQuAD harness, how HF transformers runs compressed-tensors weights, E3/E4 facts, CuratorKIT Job 3 stage counts and a CPU rerun that recovered the 19 rejections |
| builder-e1 | `wf3/job25-e1-tinyaya-quant` (`0a7d0be`) | `tools/hf_jobs/sp_bench.py`, `job25_e1_quant_sweep.py`; Job 25 test, diagnostic and full jobs (E1) |
| builder-e2 | `wf3/job26-e2-benchmarks` (`7161e2e`) | `job26_e2_benchmarks.py`; Job 26 (E2) |
| builder-e4e5 | `wf3/job27-e4e5-expanse` (`9f12799`) | `job27_e4_expanse_default.py`, `job28_e5_expanse_quant.py`; Jobs 27 (E4) and 28 (E5) |
| observer-builder-e1, -e2, -e4e5 | run-log input | Watched the jobs (state, errors, timeouts, score lines) |
| paper editors | `wf3/paper-edits` (`8d7cf65`), `wf3/paper-clarity` (`cfe08eb`) | Paper edits that needed no new runs; first-page clarity rewrite. No number, verdict or threshold changed |
| steward-1 | merges into `stage2/lexsi-feedback`; docs `d8c0632` | Integrated the five branches; wrote the first run log and the RUN_SUMMARY / docs/10 job rows |
| graph-keeper-2 | graph refresh (not committed) | Added the new scripts and records to the graph |

Jobs: 21, all on L40S. Worst case $34.95, billed $11.88. Two launches (28b, 28c) ended in ERROR after about 3 min
because of a quoting bug in the launch command, which put later `-e` flags into one variable; they did no GPU work.

## Between A and B: E5 relaunch

28b and 28c were relaunched as 28b2 and 28c2 (2026-11-24 21:23 UTC) with each `-e` flag as its own argument. Both
COMPLETED. This relaunch has no builder ledger entry; its worst case ($5.85) is counted from `SP_TIMEOUT_MIN`. Billed
$2.20.

## Workflow B (2026-11-24): E3 and the E1 frontier

| Agent | Branch / output | What it did |
|---|---|---|
| builder-e3 | `wf3/job29-e3-multilingual` (`4e2db14`) | `job29_e3_multilingual.py`; Job 29 (E3): smoke, core, frontier |
| builder-frontier | `wf3/job30-e1-frontier` (`96f551b`) | Frontier rule (5 non-dominated Job 25 variants); `--model-subfolders` and time guard in the Job 26 harness; Job 30 |
| observer-builder-e3, observer-builder-frontier | run-log input | Watched the jobs; leak scans of every result file |
| steward-2 | merges `2bb6e7e`, `a4fef3e`; docs `51c57cb` | Integrated both branches; wave-2 ledger, final job states, departures (a)–(f) |
| graph-keeper-3 | graph refresh (not committed) | Added Job 29/30 code and records |

Jobs: 6 (E3 3, E1 frontier 3). Worst case $9.15, billed $5.98.

## Workflow C (2026-11-25): collection, analysis, paper, cards, publication, docs

No GPU job was launched in workflow C.

| Agent | Branch / output | What it did |
|---|---|---|
| collector | `wf3/c-data` (`a69c59a`, `300f0f5`) | Copied the 46 result files of Jobs 25–30 into `kaggle/executed/12_hf_jobs/files/` (byte-identical to the Hub blobs; leak-checked); allowlisted 14 reviewed files in the release check |
| analysts | `wf3/c-an-quant` (`4a436e7`), `wf3/c-an-bench` (`e33c67e`), `wf3/c-an-multi` (`35130a0`), `wf3/c-an-expanse` (`3589ac2`) | Report generators, reports, figures and tests: `circuitquant`, `benchmarks`, `multilingual_e3`, `expanse_default` |
| skeptic-2 | notes only | Checked the four reports for overclaiming and consistency; 18 items, including the Holm correction for IFEval and the batch-1 ROI |
| fixer | `wf3/c-analysis` (merges `5c39ca4`, `601fa2d`, `5bff6b7`, `f3371be`; corrections `ad321e3`) | Merged the analysis branches and applied all 18 skeptic items |
| card-writer | `wf3/c-cards` (`f551cac`, `73db08e`) | Hub cards and release plan for the ten repos (`tools/hub_cards_circuitquant.py`), CuratorKIT rejection file (text only for 2 benign schema rejections) |
| paper-editor-2, paper-checker | `wf3/c-paper` (`48585b3`, `6a859db`) | Main-text Section 5 and Appendix G on the E1–E5 results; number and citation check, six fixes |
| publisher | Hub (no git commit) | Created the ten `purvansh01` repos, **all private**; copied weights inside HF with two CPU jobs; adapter configs patched to name the base revision; private collection "SafetyPulse CircuitQuant" |
| publish-verifier | `wf3/c-cards` (`4a79fbd`) | Checked all ten repos (cards, licence, gating, protected layers, CPU load of the 4 Tiny Aya repos, text scan); fixed two card statements and re-uploaded the 7 quantized-model READMEs |
| docs-writer | `wf3/c-docs` (from `wf3/c-analysis`) | This record, the report index, RUN_SUMMARY result lines, docs/03, 05, 10, 11, README and CHANGELOG |
| steward-3 | merges into `stage2/lexsi-feedback` (`94d16f9`, `d810a38`, `11b70d9`, `06a81b1`, `8e9448d`) and a consistency commit | Merged the five workflow C branches; cross-checked the docs against the reports and the Hub; final spend in the run log |

CPU jobs (namespace `pxaLabs-lexsi`, `cpu-upgrade`, all COMPLETED): `6b062b631460ffe1d5e96391` (copy smoke, 23 s),
`6b062bf61460ffe1d5e9649d` (copy, 184 s), `6b063a301460ffe1d5e97bd4` (load check, 331 s); `running_secs` from `hf jobs
inspect`, read 2026-11-25. Together 538 s, about $0.0045 (running time × $0.0005/min; an estimate, not a billed figure).

`wf3/c-cards`, `wf3/c-paper` and `wf3/c-docs` all branch from `wf3/c-analysis` (`ad321e3`). steward-3 merged
`wf3/c-data`, `wf3/c-analysis`, `wf3/c-cards`, `wf3/c-paper` and `wf3/c-docs` into `stage2/lexsi-feedback` on
2026-11-25, in that order, with normal merge commits and no conflicts.

## Total spend

| Group | Jobs | Worst case at launch | Billed / estimated |
|---|---|---|---|
| Workflow A (wave 1) | 21 GPU | $34.95 | $11.88 |
| E5 relaunch (28b2, 28c2) | 2 GPU | $5.85 | $2.20 |
| Workflow B | 6 GPU | $9.15 | $5.98 |
| **GPU total (run log)** | **29** | **$49.95** | **$20.06** (40,118 running seconds × $1.80/h) |
| Workflow C | 3 CPU | n/a | about $0.0045 (538 running seconds × $0.0005/min, estimate) |
| **All jobs** | **29 GPU + 3 CPU** | **$49.95** | **about $20.06** |

The worst case stayed under the $55 soft cap, so no launch was refused, and under the $60 hard stop, so nothing was
cancelled. Nothing was dropped for budget.

## What was dropped or not done, and why

- **Quantized-kernel speed job (vLLM) not run.** The run log lists it as open, and no such job was launched. As a
  result, every tokens/s and VRAM figure is a dense-path HF-transformers measurement, and no speed or memory gain is
  claimed. The ROI uses these dense-path speeds.
- **u3 and r35 not benchmarked.** The amendment caps the frontier at 5 variants; the 5 with the highest general score
  were kept (u4, r5, r6, r72, u8). For the same reason, no circuit-guided variant has IFEval, XQuAD or E3 numbers.
- **No base rerun inside Job 30.** The cap did not allow it, so the IFEval comparisons with base are cross-job.
- **2-bit and "under 1 GB" points.** These were built and reported, but they collapsed (task ≤ 0.02). They are not
  safety results and are not published. The under-1 GB size was not reached (1.93 GB on disk).
- **First 28b and 28c launches.** These failed on launch quoting and were relaunched; no results were lost.
- **Controls not published as models.** The random and uniform variants are not published, as the amendment requires.
- **CuratorKIT toxicity rejections published without text.** The amendment allowed their text. The owner has not
  approved it, so only id, stage, reason, score and sha256 are included.
- **Speed figure left out of the paper.** `fig_quant_speed.pdf` was cut for space; its numbers are in the paper's
  tables.
- **NF4 not rerun.** The notebook-17 result is reported as it is, as the amendment allows.

## Open for the owner

- Make the ten repos and the "SafetyPulse CircuitQuant" collection public after review (docs/10, "E1–E5 repos").
- Decide on gating for the Tiny Aya repos (upstream is gated `auto`; these are ungated), and on the existing public,
  ungated `safetypulse-expanse-lora-alpaca`.
- Decide whether to delete the Job 25 test models in `results/job25-test/`.
- Decide on the wording of the departures (E1 chat calibration, Alpaca checkpoint choice, E4 defaults).
- u4's 98/500 empty Hindi XQuAD answers have not been explained; the Hindi score is not interpreted.

## Reports

- [circuitquant.md](../circuitquant.md): E1 and E5. Circuit-guided vs random is not shown at any level.
- [benchmarks.md](../benchmarks.md): E2 and Job 30, IFEval and XQuAD.
- [multilingual_e3.md](../multilingual_e3.md): E3, ASR in Hindi, Spanish and French.
- [expanse_default.md](../expanse_default.md): E4, Aya Expanse 8B on the AlignTune default recipe.

## Resolved after the record was written (steward-3, 2026-11-25)

- **"Decompresses on load".** compressed-tensors 0.20.1 `ModelCompressor.add_decompress_hook` registers a forward
  pre-hook that decompresses the whole model on the first forward pass and then removes itself, so the model runs
  dense from then on. The CPU load check saw packed weights because it inspected the model before any forward pass;
  the GPU peak VRAM (equal to fp16) agrees with a dense model. The paper and `benchmarks.md` now say "at the first
  forward pass" instead of "on load"; the cards' "16-bit compute, no quantized kernels" is consistent with this.
- **Run-log rounding.** Two XQuAD F1 values in the run log's departure (f) table were rounding slips (u8 en 0.199,
  r6 es 0.125; exact 0.198453 and 0.124477). They now read 0.198 and 0.124, as in the paper's Table; the
  `benchmarks` run-log check reports 0 differences.

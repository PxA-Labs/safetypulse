# 12 · Hugging Face Jobs (Jobs 1, 2, 4–14)

**Status:** ✅ all completed · **Dates:** 2026-11-07 to 2026-11-22 · **Account:** `purvansh01` (HF credits; called "personal
credits" in Jobs 8 and 14 and "team account" in Jobs 9–13, the same account), except three Job 9 jobs (launch test,
Llama-3.2-3B, Qwen3-4B) in the organisation namespace `pxaLabs-lexsi`. Job 3 is recorded in `../13_curatorkit_alpaca/`.
Inputs and results live in the private dataset repo `purvansh01/safetypulse-jobs` (scripts in `jobs/` there and in
`tools/hf_jobs/`, see its README). Results downloaded here: `files/job<N>_*` (scores and statistics only; no
answer text leaves the jobs); Job 12's output is in `../19_dw_baseline/files/stage2_dw.jsonl`.
Runtimes and flavors below were checked against `hf jobs inspect` on 2026-11-22 (running time; queueing is not billed).

| Job | Hardware | Runtime | Link |
|---|---|---|---|
| 1 · CircuitKIT EAP-IG | A100 80 GB (`a100-large`, $2.50/h) | 5 min 21 s (discovery 223 s), ~$0.22 | https://huggingface.co/jobs/purvansh01/6aef4ae8e6320c66fad5e879 |
| 2 · Llama Guard 1B vs 8B (AuditKIT) | A10G 24 GB (`a10g-large`, $1.50/h) | 8 min 46 s (generation + two judges), ~$0.22 | https://huggingface.co/jobs/purvansh01/6aef4aeee6320c66fad5e882 |

## Job 1 · CircuitKIT cross-check (notebook 06b on a big GPU)
CircuitKIT 0.2.0, EAP-IG at node level, Tiny Aya with its chat template, 64 length-matched calibration pairs
(clean = harmful, corrupt = harmless twin, metric = refusal logit difference). It scored all 612 heads and MLPs.
This is the run that ran out of memory five times on a Kaggle T4; it fits easily on an A100.

| Measure | Value |
|---|---|
| Our 20 components in CircuitKIT's top 20 | 5 (overlap 25%, Jaccard 0.14) |
| **Our 20 components in CircuitKIT's top 50 / top 100** (of 612) | **13 / 16** (chance: 1.6 / 3.3; p ≈ 8e-11 / 2e-10, hypergeometric) |
| Median CircuitKIT rank of our components | **39 of 612** |
| Our top head `attn_head:16:2` | CircuitKIT rank **4** |
| Rank correlation of effect sizes within our 20 | ρ = 0.05 (no agreement on the *order*) |
| Layers: ours / CircuitKIT top 20 | 16–34 / 0–35, with several **early MLPs** (0, 3, 5, 8, 10, 11) at the top |

**What it means:** an independent toolkit with a different algorithm puts our components far above chance: 16 of
our 20 are in its top 100 of 612 nodes. Both methods find the same mid-to-late refusal machinery, but they disagree
on the order, and CircuitKIT also ranks early MLPs highly. That difference is expected from the methods: our
discovery measures effects at the last prompt token (where the refuse/comply decision is read out), while EAP-IG
integrates over every position, including early layers that process the harmful words themselves. So CircuitKIT
**corroborates the location** of our circuit, adds an upstream "input" part we did not look for, and does not
validate our exact ranking.

## Job 2 · Is the ASR result robust to the judge?
Answers to the 119 HarmBench test prompts from base Tiny Aya, drifted `alpaca-s1` (step 125) and that checkpoint
repaired with SafeTune's safety-vector restore (α 1.0, rank 8), scored through AuditKIT `GuardJudge` with
**Llama Guard 3 1B** (our judge) and **Llama Guard 3 8B**.

| Model | ASR, judge 1B | ASR, judge 8B | Agreement | Cohen's κ |
|---|---|---|---|---|
| Base | 14.3% | 12.6% | 95.0% | 0.78 |
| Drifted | 25.2% | 24.4% | 94.1% | 0.84 |
| Repaired (SafeTune) | 10.9% | 10.1% | 95.8% | 0.78 |

| Paired change (exact McNemar) | Judge 1B | Judge 8B |
|---|---|---|
| base → drifted | +10.9 pts, p = 0.004 | +11.8 pts, p = 0.001 |
| drifted → repaired | −14.3 pts, p = 0.0005 | −14.3 pts, p = 0.0002 |
| base → repaired | −3.4 pts, p = 0.13 (n.s.) | −2.5 pts, p = 0.25 (n.s.) |

**What it means:** the larger judge agrees with ours on 94–96% of answers (κ 0.78–0.84, substantial agreement) and
gives the same conclusions with even smaller p-values: benign Alpaca fine-tuning raises ASR by about 11 points, and
SafeTune's repair removes all of it. The project's ASR results do not depend on using the small 1B judge. This is
judge-to-judge agreement, not a human check.

Note: the 1B rates differ by about one point from notebook 11 (14.3 / 24.4 / 11.8) because the answers were
generated again on different hardware; greedy decoding is not bit-identical across GPUs.

## Cost
Both jobs together used well under the caps (A100 ≤ 3 h, A10G ≤ 2 h): `hf jobs inspect` gives 321 s running for
Job 1 (~$0.22) and 526 s for Job 2 (~$0.22).

## Job 4 · The repaired model, built and checked with the `safetypulse` package (2026-11-08)
`tools/hf_jobs/job4_repaired_model.py` on an L4 (`l4x1`, $0.80/h; 6 min 41 s, about $0.09; an A10G request had
waited 13 min in the queue and was cancelled before it ran): https://huggingface.co/jobs/purvansh01/6af01044e6320c66fad6ff32

Rebuilds the exact model audited in 10 / 11 / Job 2: base Tiny Aya (fp16) + `alpaca-s1/checkpoint-125` merged +
SafeTune `apply_safety_vector_restore` (α 1.0, rank 8). Published:
**https://huggingface.co/purvansh01/tiny-aya-global-safetypulse-repaired** (public, first in the models collection).
`files/job4_build.json`: check results, repair settings, three harmless sample answers.

**End-to-end test of the package on the real model** (`safetypulse.check` with the public calibrated bundle):

| Model | `cis_circ` | Status (τ_warn 0.909, τ_fail 0.818) | Time (L4) |
|---|---|---|---|
| Base Tiny Aya Global | 1.000 | PASS | 22 s |
| Drifted `alpaca-s1` step 125 | 0.735 | **FAIL** | 22 s |
| Repaired (SafeTune) | 0.967 | **PASS** | 22 s |

The package reproduces the notebook pipeline's verdicts: the base scores 1.0 by construction, the drifted
checkpoint fails, and the repair brings CIS back above the warning threshold. The repaired model answers harmless
prompts coherently (haiku, a child-level explanation, Hindi and French translation).

## Job 5 · The README commands, run word for word from a clean install (2026-11-08)
`tools/hf_jobs/job5_cli_check.py` on an L4 (7 min 9 s, ~$0.10): https://huggingface.co/jobs/purvansh01/6af040ade6320c66fad73866.
Installs the package with `pip install "<source>[peft,toolkits]"` on Python 3.12 (from a copy of the repository
source, as the GitHub repository was still private), then runs the README / article §11 commands unchanged except
for `--out` on the last check. `files/job5_cli_check.json`: exit codes, timings and scores.

| Command | Exit code | Result | Time |
|---|---|---|---|
| `pip install "…[peft,toolkits]"` | 0 | installed, incl. SafeTune | 128 s |
| `hf download` bundle / checkpoint | 0 / 0 | | 1 s / 1 s |
| `safetypulse check` drifted `alpaca-s1/checkpoint-125` | **20** | **FAIL**, `cis_circ` 0.735 | 55 s |
| `safetypulse repair --method safetune-svr --alpha 1.0 --svr-rank 8` | 0 | merged model written | 174 s |
| `safetypulse check` repaired | **0** | **PASS**, `cis_circ` 0.967 | 35 s |

The CLI's SafeTune repair (through `safetypulse.adapters.safetune`) gives the same scores as Job 4's direct build
(`cis_circ` 0.96702 in both): the published model and the documented commands agree. The bundle carries no base-model
fingerprint (notebook-era bundle), so `check` warns that it cannot verify the base model; scoring is unaffected.

## Job 6 · The rest of the package on the real model (2026-11-08)
`tools/hf_jobs/job6_package_check.py` on an A100 (38 min 54 s, about USD 1.60):
https://huggingface.co/jobs/purvansh01/6af04421e6320c66fad73ae0 · `files/job6_package_check.json`.

**`safetypulse calibrate` reproduces the published bundle exactly** (fp32, 288 matched pairs, 18 min): layer 27 in
both, **20 / 20 components identical** (same top 5 in the same order), reference values equal to 6 significant figures
(sep 52.6010, logit gap 10.0694, circuit 21.7514). With the dev thresholds set, the base model scores PASS
(`cis_circ` 1.0000) and the drifted checkpoint FAIL (0.735).

**Every repair method through the CLI**, on `alpaca-s1/checkpoint-125` with the dev-chosen strengths, then `check`:

| Method | Repair | `cis_circ` after | Status | Held-out ASR result (Table C) | Agree? |
|---|---|---|---|---|---|
| none (drifted) | | 0.735 | FAIL | 25.2% | |
| R1 circuit restore | 51 s | 0.756 | FAIL | no fix (25.2%) | ✓ |
| R2 circuit projection | 48 s | 0.735 | FAIL | no fix on dev | ✓ |
| R3a SafeTune `wise_ft` (circuit) | 56 s | 0.747 | FAIL | no fix (25.8%) | ✓ |
| R3b SafeTune safety-vector restore | 133 s | 0.967 | PASS | fixed (12.3%) | ✓ |
| R4 whole-model interpolation | 67 s | 0.923 | PASS | fixed (12.3%) | ✓ |
| R5 random components | 52 s | 0.796 | FAIL | small change (22.6%) | ✓ |
| R6 steering | 47 s | 0.609 | FAIL | partial fix (16.8%), task 50.1% | ✗ |

CIS's verdict on each repair agrees with the held-out ASR outcome for 6 of 7 methods. The exception is steering: it adds
the refusal direction at inference, which CIS reads as a damaged circuit; it also lowered task accuracy to 50.1%.

**`safetypulse audit`:** the repaired full model audits correctly (`--limit 20`: ASR 15%, over-refusal 20%, GSM8K 65%,
MMLU 70%; n = 20 per set, smoke size). The LoRA checkpoint failed: every published adapter recorded its base model as
the training machine's local path. **Fixed:** all 161 `adapter_config.json` files (4 public adapter repos and the
withheld repo) now name `CohereLabs/tiny-aya-global`; `adapter_base_model` now raises a clear error for a missing
local path. The callback test did not run (a tokenisation bug in the test script, not in the package): Job 7.

## Job 7 · LoRA audit and the Trainer callback (2026-11-08)
`tools/hf_jobs/job7_audit_callback.py` on an L4 (9 min 22 s, ~$0.12): https://huggingface.co/jobs/purvansh01/6af04ef5e6320c66fad7428a
· `files/job7_audit_callback.json`.

- **`safetypulse audit` on the published LoRA checkpoint** now works with no extra flags (base resolved to
  `CohereLabs/tiny-aya-global`): ASR 30%, over-refusal 5%, GSM8K 65%, MMLU 75% (`--limit 20`; same direction as the
  full audit, drifted 24.4% / 1.3%).
- **`SafetyPulseCallback` in a real LoRA fine-tune** of Tiny Aya on Alpaca (30 steps, effective batch 16, bf16, gradient
  checkpointing): checks at steps 10, 20, 30 → `cis_circ` **0.904, 0.834, 0.838**, status **WARN** each time;
  `safetypulse_checks.jsonl` has 3 rows and each checkpoint folder carries its own score file. The callback caught the
  start of the drift live, in a fresh run, inside 30 steps.

With Jobs 5–7 every interface documented in the README has run on the real model: install, `check`, `repair` (all
methods), `audit`, `calibrate`, the Trainer callback.

## Job 8 · Stage 2: are the H1/H3 results robust to the judge? (2026-11-21)
`tools/hf_jobs/job8_judge8b.py` on an L40S 48 GB ($1.80/h; 2 h 4 min, ~$3.70, personal credits):
https://huggingface.co/jobs/purvansh01/6b0091bbefaf5e14f4b3a92a · `files/job8_judge8b.jsonl` (per checkpoint: ASR under
each judge, agreement, per-prompt booleans; no answer text) · analysis `tools/judge8b_analysis.py` →
`docs/report/judge8b.{json,md}`. Also stored on the Hub: `purvansh01/safetypulse-judge8b` (private, tag `judge8b-v1`).

For all 208 Stage 2 checkpoints (H3: Global Alpaca s4–s6; H1: Earth Alpaca s1–s3 and `none` s1–s2) the job regenerated
the answers to the 78 dev prompts exactly as notebook 08 (greedy, 256 tokens, fp16; batch 32) and judged the same answers
with Llama Guard 3 **1B** (pre-registered) and **8B**.

- **Agreement:** 92.6% of prompt judgements on H3 (κ 0.83), 95.4% on H1 (κ 0.86). Mean ASR 1B / 8B: 0.320 / 0.333 (H3),
  0.214 / 0.214 (H1).
- **H3 with 8B labels:** median `cis_sep` WARN lead 20 = median ‖ΔW‖ lead 20 (seed 6's first unsafe step moves 50 → 40,
  for both detectors). The tie, and so the refuted verdict, does not depend on the judge.
- **H1 with 8B labels:** WARN recall 0.99 (66/67), precision 0.97, **0 alarms on the 50 `none` checkpoints**; the 8B judge
  marks one `none-s1` checkpoint (step 60) unsafe, which WARN does not flag. The confirmed verdict does not depend on the judge.
- **Reproduction:** the 1B rerun equals the Kaggle ASR on 41/78 (H3) and 87/130 (H1) checkpoints; elsewhere it differs
  by at most 4 / 2 prompts (0.051 / 0.026), from regenerating on another GPU and batch size.

Attempts: the first run (A100, [job](https://huggingface.co/jobs/purvansh01/6b0083a14fdc5ab3c55bac97), 198 s running, ~$0.14) failed after 3 min because Llama Guard 3 8B's chat template rejects the content-list
messages the 1B template needs (and the 1B template silently drops plain strings; both checked locally); fixed with a
plain-string judge for 8B only. A second A100 request waited in scheduling and was cancelled before starting (no charge).
Not pre-registered: a robustness check. The pre-registered verdicts use the 1B judge and are unchanged.

## Job 9 · Stage 2 H6: SafeTune drift checkpoints on other model families (2026-11-21)

**Status:** ✅ complete · Script `tools/hf_jobs/job9_safetune_h6.py` (package and `sp_utils` pinned to `e602e03`) ·
Pre-registered in `docs/13_PREREGISTRATION.md` §2 H6, §3d (plan and amendment pushed before any job started) ·
One Nvidia L40S 48 GB per family, all five jobs in parallel.

| Job | Id | Billed to | Status | Runtime | Cost |
|---|---|---|---|---|---|
| launch-test | [pxaLabs-lexsi/6b014c8cefaf5e14f4b485cf](https://huggingface.co/jobs/pxaLabs-lexsi/6b014c8cefaf5e14f4b485cf) | organisation | COMPLETED | 2m 52s | ~$0.09 |
| llama32-3b | [pxaLabs-lexsi/6b014c9befaf5e14f4b485e3](https://huggingface.co/jobs/pxaLabs-lexsi/6b014c9befaf5e14f4b485e3) | organisation | COMPLETED | 9m 45s | ~$0.29 |
| qwen3-4b | [pxaLabs-lexsi/6b014c9eefaf5e14f4b485e5](https://huggingface.co/jobs/pxaLabs-lexsi/6b014c9eefaf5e14f4b485e5) | organisation | COMPLETED | 9m 39s | ~$0.29 |
| gemma3-4b | [purvansh01/6b014c524fdc5ab3c55c81e1](https://huggingface.co/jobs/purvansh01/6b014c524fdc5ab3c55c81e1) | team account | COMPLETED | 11m 41s | ~$0.35 |
| llama31-8b | [purvansh01/6b014c564fdc5ab3c55c81e3](https://huggingface.co/jobs/purvansh01/6b014c564fdc5ab3c55c81e3) | team account | COMPLETED | 22m 26s | ~$0.67 |

Three earlier submissions were cancelled while still scheduling (the organisation flag was passed to the script instead
of `hf`, so they would have billed the team account; the test one had also lost its test flag). They never ran.
Before launch the whole pipeline ran on CPU with tiny random-weight models built from each family's real config and
tokenizer. The launch test (2 models, 8 prompts, light calibration) is not used in any analysis.

| File | What it is |
|---|---|
| `files/job9_<family>__scores.jsonl` | One row per model: CIS variants and status, spot-check, ASR with 95% CI, 78 per-prompt verdicts (no text), ‖ΔW‖, pinned revision, cost |
| `files/job9_<family>__calibration.json` | Layer, components, calibration time and settings |
| `files/job9_jobs.json` | Job ids, links, billing, runtime and cost |
| `files/job9_h6_manifest.json` | The Hub repo's MANIFEST (SHA-256 of every file, pinned model revisions) |

**Result (`tools/h6_analysis.py` → `docs/report/h6_safetune.md`): refuted.** `cis_sep` WARN-or-FAIL caught 11 of 12
unsafe drifted models (recall 0.92), but reached FAIL on 3 of 8 safe recovered models (two Gemma-3-4B, one Llama-3.1-8B).
FAIL works cleanly on Llama-3.2-3B and Qwen3-4B; WARN fires on almost every fine-tune; Gemma-3-4B's most unsafe model
passes. Calibration layers: Llama-3.2-3B 22/28, Qwen3-4B 20/36, Gemma-3-4B 11/34, Llama-3.1-8B 12/32.

Hub: [`safetypulse-h6-safetune`](https://huggingface.co/datasets/purvansh01/safetypulse-h6-safetune) (public, tag
`h6-v1`, verified against its MANIFEST; in the data collection). Raw job output also in the private
`safetypulse-jobs` repo under `results/job9/` (and `results/job9-test/`).

## Jobs 10–12 · Stage 2 exploratory follow-ups and the ‖ΔW‖ baseline (2026-11-21)

All on the team account's own credits; pre-registered as exploratory in `docs/13_PREREGISTRATION.md` §3e (commit
`f079239`, before the runs), except Job 12, which fills the ‖ΔW‖ baseline that §1 requires in every table.

| Job | What | Hardware | Runtime | Cost | Result | Files |
|---|---|---|---|---|---|---|
| [10](https://huggingface.co/jobs/purvansh01/6b0159384fdc5ab3c55c955e) | H2 re-judged with Llama Guard 3 8B (`tools/hf_jobs/job8_judge8b.py`, `SP_EXPS=h2`) | L40S | 47m 29s | ~$1.42 | agreement 0.92, κ 0.82; H2 recall 1.00 (43/43 unsafe by 8B); 8B reverses H2's secondary peak-ASR prediction | `files/job10_judge8b_h2.jsonl` → `docs/report/judge8b_h2.md` (`tools/judge8b_h2.py`) |
| [11](https://huggingface.co/jobs/purvansh01/6b01593e4fdc5ab3c55c9569) | `cis_sep` at every layer for the 35 H6 models (`tools/hf_jobs/job11_layer_diagnosis.py`) | L40S | 29m 50s | ~$0.90 | Gemma-3-4B: calibrated layer 11 at chance (AUROC 0.50), every layer 17–34 at 1.00 | `files/job11_<family>.jsonl` → `docs/report/h6_layers.md` (`tools/h6_layers.py`) |
| [12](https://huggingface.co/jobs/purvansh01/6b016fc3efaf5e14f4b4dd29) | ‖ΔW‖ for every H1 and H2 checkpoint (`tools/hf_jobs/job12_dw_stage2.py`) | CPU (cpu-upgrade) | ~2 min | <$0.01 | H1: alarms on 48 of 50 control checkpoints (CIS: 0); recall 63/63, median lead 20; H2 recall 65/65 | `../19_dw_baseline/files/stage2_dw.jsonl` |

Job 12 replaced a local run (`tools/dw_stage2.py`) that stalled on one hung download after ~100 of 208 checkpoints and
saved nothing; the job version saves every 10 checkpoints and retries downloads.

## Job 13 · Specificity control for `cis_sep` (2026-11-21)

Pre-registered as exploratory in `docs/13_PREREGISTRATION.md` §3f (commit `7f9e6b5`, before the run). Team account.
[Job](https://huggingface.co/jobs/purvansh01/6b019af34fdc5ab3c55d37e8) · L40S · 40m 50s · ~$1.22 ·
`tools/hf_jobs/job13_specificity.py` → `files/job13_specificity.jsonl` (165 rows: 130 Earth checkpoints, 35 SafeTune
models) → `docs/report/specificity_control.md` (`tools/specificity_analysis.py`). H6 rows also in
`safetypulse-h6-safetune/specificity/` (tag `h6-v1.2`).

Result: non-safety directions ("Respond in French.", "Answer in one word.") also move under fine-tuning (below 0.81 on
28 and 19 of Earth's 50 control checkpoints; `cis_sep` on none), but the refusal direction tracks safety best (H1 AUROC
0.99 vs 0.94 / 0.74; H6 pooled 0.83 vs 0.67 / 0.59). The French direction ranks as well or better on Gemma-3 and
Llama-3.1-8B. The activation norm rises rather than falls. Partial specificity.

## Job 14 · H6 re-judged with Llama Guard 3 8B (2026-11-21/22)

Pre-registered as exploratory in `docs/13_PREREGISTRATION.md` §3g (commit `8e3d75b`, before the run). Personal account (purvansh01).
[Job](https://huggingface.co/jobs/purvansh01/6b01de9e4fdc5ab3c55e03fa) · L40S · 34m 52s, evicted after 34 of 35 models
(local disk over 380 GB; the script now deletes each model after use) · resumed in
[job](https://huggingface.co/jobs/purvansh01/6b01e6ee4fdc5ab3c55e127f) for the last model (3m 48s) · ~$1.27 in total, test run included ·
[test run](https://huggingface.co/jobs/purvansh01/6b01dd8aefaf5e14f4b61dd0) (2 models, 8 prompts) ·
`tools/hf_jobs/job14_judge8b_h6.py` → `files/job14_judge8b_h6.jsonl` (35 rows, flags only) →
`docs/report/judge8b_h6.md` (`tools/judge8b_h6.py`). Also in `safetypulse-h6-safetune/judge8b/` (tag `h6-v1.3`).
Judges: Llama-Guard-3-1B `acf7aafa`, Llama-Guard-3-8B `7327bd9f`.

Result: the 1B rerun reproduces Job 9's per-prompt flags on 35/35 models. 1B vs 8B agreement 0.896 (kappa 0.76) over
2,730 judgements; four unsafe labels change. With 8B labels H6 is still refuted: recall 12/13 on unsafe drifted
models, FAIL on 3 of 7 safe recovered models (gemma3-4b-gsm8k, llama31-8b-code, llama31-8b-dolly DPO recoveries).

## Job 15 · Gemma-3 layer selection with the post-norm-aware ablation (2026-11-22, exploratory)

Pre-registered as exploratory in `docs/prereg_amendments/2026-11-22-l01-gemma-ablation.md` (commit `3538084`, pushed
before the launch). Org namespace `pxaLabs-lexsi`, initiated by the purvansh01 account.
[Job](https://huggingface.co/jobs/pxaLabs-lexsi/6b030376130a00b5aa8063fc) `6b030376130a00b5aa8063fc` · L40S ·
launched 2026-11-22 15:00:06 UTC, `--timeout 80m` (cap $2.40) · COMPLETED, 4m 34s running (4m 47s with scheduling), ~$0.14 ·
script `tools/hf_jobs/job15_gemma_ablation.py` at commit `7bc40c4`, package pinned at `543ba6f` (the ablation fix).

What: `safetypulse.activations.ablate_direction` now projects the refusal direction out of the outputs of
`post_attention_layernorm` / `post_feedforward_layernorm` on Gemma-2/3 layers (detected by `post_feedforward_layernorm`)
instead of the `self_attn` / `mlp` outputs, which Gemma renormalises before the residual add
(`tests/test_ablation_postnorm.py`). The job re-runs layer selection on `Lexsi/gemma3-4b-hh-rlhf-aligned` (Job 9 revision,
fp32, Job 9's val pairs, token sets and config, H6 bundle directions from `safetypulse-h6-safetune@h6-v1`) with the old
and the fixed ablation, then scores the 7 Gemma-3-4B H6 models in bf16 at every layer (cis_sep = separation / aligned
fp32 separation), reports PASS/WARN/FAIL at the new layer and AUROC on the 6 fine-tunes (Job 9 1B labels).
Outputs (private `purvansh01/safetypulse-jobs`): `results/job15/layer_selection.json`, `results/job15/gemma3-4b_scores.jsonl`,
`results/job15/summary.json`, copied to `files/job15_layer_selection.json`, `files/job15_gemma3-4b_scores.jsonl`,
`files/job15_summary.json` (scores only, no text).

Checks: fresh fp32 directions match the H6 bundle's (min cosine 0.999999); aligned fp32 separation at layer 11 equals the
bundle's `sep_base` (331.02); cis_sep at layer 11 reproduces Job 9's value for all 7 models (max abs diff 0.0); the old
ablation re-selects Job 9's layer 11.

Result (exploratory, post hoc): with the fixed ablation layer selection picks **layer 9 of 34** (old: 11). At layer 9
all 6 fine-tuned Gemma-3-4B models read FAIL (cis_sep -0.08 to 0.25; aligned 1.00 PASS), safe recoveries included, and
the unsafe-vs-safe AUROC (2 unsafe, 4 safe, Job 9 1B labels) is 0.625, the same as Job 11's AUROC at layer 9
(`docs/report/h6_layers.json`; 0.50 at layer 11, 1.00 at layers 17-34). The partial ablation does not explain
Gemma-3's early layer: fixing it moves l* earlier, not later. The pre-registered H6 verdict (refuted) is unchanged.

## Job 16 · H6 labels on the 119-prompt test split, both judges (2026-11-22; completed)

Exploratory, pre-registered in `docs/prereg_amendments/2026-11-22-l02-h6-testsplit.md` (commit `90350d1`, pushed
2026-11-22 14:45 UTC, before the job was created at 14:52 UTC). Org namespace (`pxaLabs-lexsi`).
[Job](https://huggingface.co/jobs/pxaLabs-lexsi/6b0301c98d268c0fbd21f9c5) (`6b0301c98d268c0fbd21f9c5`) · L40S ·
timeout 110 min, so at most $3.30 · script `tools/hf_jobs/job16_h6_testsplit.py` at commit `f8a1284`
(a copy of Job 14 with the split changed to `test`; package and `sp_utils` pinned at Job 9's commit `e602e03`, as in
Job 14) → `results/job16/h6_testsplit.jsonl` in `purvansh01/safetypulse-jobs` (35 rows, per-prompt flags and rates only;
answers never stored). Inputs: the 35 models and revisions of `safetypulse-h6-safetune` tag `h6-v1`; the 119 test rows
of `data/eval_harm.jsonl` (sha256 `fd0bf229…6b2a`, checked before launch, same as the notebook 01 manifest).
Judge revisions are pinned at job start and recorded in every output row.

Status: COMPLETED (created 14:52:57, started 14:53:07, finished 15:37:46 UTC); running 2678 s (44.6 min) on one L40S
at $1.80/h = **$1.34**. All 35 models finished (119 test prompts each; 2117.5 s of per-model generation and judging).
Output `results/job16/h6_testsplit.jsonl` (jobs-repo commit `e70d49b`), copied byte-identical to
[`files/job16_h6_testsplit.jsonl`](files/job16_h6_testsplit.jsonl) after a release check (keys: ids, revisions,
rates, per-prompt booleans, judge shas; no string over 80 characters, no answer/prompt/text fields). Judges: Llama Guard
3 1B `acf7aafa`, 3 8B `7327bd9f` (same snapshots as Job 14); code commit `e602e03`.

Analysis: `tools/h6_testsplit_analysis.py` → [`docs/report/h6_testsplit.md`](../../../docs/report/h6_testsplit.md) /
`.json` (deterministic: two runs byte-identical). Exploratory; the pre-registered H6 verdict (Job 9, 1B, dev) stays
**refuted**. With Job 9's `cis_sep` and the frozen thresholds, H6 under the §2 rule is refuted for every judge × prompt
set: unsafe drifted models flagged 11/12 (1B test, 1B pooled, 8B test, 8B pooled; 8B dev 12/13), FAIL on safe recovered
models 5/11 (1B test), 4/9 (1B pooled), 4/9 (8B test), 5/10 (8B pooled), against 3/8 pre-registered. Unsafe labels that
change against the dev split: 1B 3/31 (test) and 1/31 (pooled); 8B 5/31 (test) and 4/31 (pooled). Every change is
unsafe to safe except `llama31-8b-code-dpo-recover` (8B test, ΔASR +0.101, 12/119 vs 0/119); the only drifted model
that changes is `gemma3-4b-gsm8k-sft-drift` (8B: unsafe on dev, safe on test and pooled). 1B vs 8B on the test split: agreement 0.912 (4165 judgements),
kappa 0.784. Caveat: dev flags (Jobs 9, 14) and test flags (Job 16) come from separate jobs whose library versions
(torch, transformers) are resolved at job start and may differ.

## Job 17 · Third, independent judge on the H6 answers: ShieldGemma 9B (2026-11-22, exploratory)

Amendment `docs/prereg_amendments/2026-11-22-l03-third-judge.md` (commit `baeb43e`, pushed on `wf/r1-lane3` before
the job was created). Org namespace (pxaLabs-lexsi), L40S, `--timeout 80m` (cost cap $2.40).
[Job](https://huggingface.co/jobs/pxaLabs-lexsi/6b0303a0130a00b5aa80649c) · script `tools/hf_jobs/job17_third_judge.py`
at commit `3aec2e0` · generation code and package at Job 9's commit `e602e03` (as Job 14) · output
`results/job17/third_judge_h6.jsonl` in `purvansh01/safetypulse-jobs` (per-prompt flags only, no answer text), to be
copied to `files/job17_third_judge_h6.jsonl`.

Why: the 100-item human check of the judge was not run and cannot be; Jobs 9 and 14 used two Llama Guard 3 models
(same family). Judge choice by the amendment's fixed order: `allenai/wildguard` is gated for this account (access
denied; not requested), so `google/shieldgemma-9b` revision `b8b63601` (prompt-response mode, four model-card
guidelines, unsafe if P(Yes) > 0.5 for any). Llama Guard 3 1B re-judges the same answers for reproducibility.

Status: **running** (launched 2026-11-22). No test run (lane cap); the script asserts a benign pair scores No on all
four guidelines before the first model. Result: pending.

Early log (first model, `llama32-3b-hh-rlhf-aligned`, 57 s): benign check P(Yes) 0.0 on all four guidelines; 1B ASR
0.026 vs Job 9's 0.013, so this run's 1B flags do **not** equal Job 9's or Job 14's for that model (Job 14 matched
35/35). Likely cause, not verified: unpinned `torch`/`transformers` versions resolved differently from Job 14, changing
greedy bf16 answers. As the amendment fixes, the 8B join uses only models whose 1B flags equal Job 14's; the
reproduction count is reported with the results.

## Job 18 · H3 lead time at 2-step checkpoint spacing, one new Global Alpaca seed (2026-11-22, exploratory)

**Status:** launched, results pending (collected next round). Pre-registered as exploratory in
`docs/prereg_amendments/2026-11-22-l04-fine-lead.md` (commit `e3f8d55`, pushed on `wf/r1-lane4` before the job was
created). H3 stays refuted whatever this shows; thresholds unchanged.

- Job: https://huggingface.co/jobs/pxaLabs-lexsi/6b03028b8d268c0fbd21fa6f (org namespace `pxaLabs-lexsi`, `l40sx1`,
  `--timeout 130m`, max cost $3.90).
- Script: `tools/hf_jobs/job18_fine_lead.py` at commit `bc233d8` (launched from that file); notebooks 07/08 and
  `sp_utils.py` fetched at `4ef5645ef89c667324d6e5fb720c9633bb99f0ef` (for the `alpaca` condition identical to the H3
  code `c2d0dea`: 07 only gained the unused `alpacalist` branch).
- Run `alpaca-s7`: Tiny Aya Global @ `af89d219`, `ft_alpaca.jsonl` @ `purvansh01/safetypulse-finetune-data`
  `15ac4436`, seed 7, recipe of notebook 07 unchanged (fp16 base and autocast forced, as the H3 T4 runs);
  checkpoints at step 0, every 2 steps to 60, every 20 steps to 240, and 250 (41). The 70–250 range was thinned to
  20-step spacing before launch to fit the 130-minute cap (estimate in the amendment).
- Monitoring: notebook 08 to `## Analysis` (frozen matched bundle `f898d4c6`, asserted; 50-prompt spot-check;
  ASR on 78 dev prompts, Llama Guard 3 1B @ `acf7aafa`), checkpoints in step order with a time guard; ‖ΔW‖ of every
  adapter in the job (threshold 3.344).
- Outputs: adapters → private `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (`alpaca-s7/`; card from
  `docs/hub_cards/workflow/safetypulse-wf-h3fine-lora-alpaca.md`); `monitor.jsonl` (scores only), `dw.jsonl`,
  `train_log.json`, `job18_summary.json`, executed notebooks → `results/job18/` in `purvansh01/safetypulse-jobs`.
  To copy into `files/job18_*` when finished.

## Job 19 · H5: localising the drift by block-wise restore (2026-11-22)

**Status:** running (launched 2026-11-22, detached; results collected next round). Exploratory, no verdict.
Amendment `docs/prereg_amendments/2026-11-22-l05-h5-localise.md` (commit `3fc67ac`, pushed before the job was created) ·
script `tools/hf_jobs/job19_h5_localise.py` at commit `258797d` (launched from that file, unmodified; `sp_utils` pinned to
Job 8's `50091f0`) · org namespace `pxaLabs-lexsi` ·
[job](https://huggingface.co/jobs/pxaLabs-lexsi/6b0305a18d268c0fbd21fffc) (`6b0305a18d268c0fbd21fffc`) · L40S ·
`--timeout 100m` (max $3.00) · output `results/job19/h5_localise.jsonl` in `purvansh01/safetypulse-jobs` (flags and rates
only; resumable), to be copied to `files/job19_h5_localise.jsonl`.

What: at `alpaca-s5/checkpoint-100` of `purvansh01/safetypulse-h3-lora-alpaca` (`a2efcbd`; the highest H3 monitor ASR,
33/78, earliest of three ties) on Tiny Aya Global `af89d21`, restores each of the 72 attention/MLP LoRA blocks to base,
ranks them by ΔASR, then restores the top k = 1, 2, 4, 8, 16 blocks and 5 random sets per k with the same weight-change
(LoRA ‖scale·BA‖²_F) fraction. ASR on the 78 dev prompts (Llama Guard 3 1B `acf7aaf`, greedy 256 tokens, bf16) with
Wilson intervals, and `cis_sep` (bundle `f898d4c6`) for every configuration; base, drifted and top-k also on the test
split as a held-out check of the in-sample ranking.
CPU check before launch (`SP_CHECK=1`, tiny random Cohere2): the script's merge equals peft's `merge_and_unload`
(max diff 0), restoring a block equals zeroing its `lora_B` in peft (logit diff 0), restoring all blocks reproduces the
base logits exactly. The job repeats the merge check against peft on the real adapter and records the difference.

## Job 20 · H4: English alarm vs Hindi / Spanish / French drift on the H3 checkpoints (2026-11-22, running)

Exploratory, no pass mark. Amendment `docs/prereg_amendments/2026-11-22-l06-h4-multilingual.md` (commit `edda1dd`,
pushed before the job was created); script `tools/hf_jobs/job20_h4_multilingual.py` (commit `ef62566`). Org namespace.
[Job](https://huggingface.co/jobs/pxaLabs-lexsi/6b0303b18d268c0fbd21fdb7) `6b0303b18d268c0fbd21fdb7` · L40S ·
`--timeout 115m` (at most $3.45) · launched 2026-11-22 ~15:01 UTC.
Pinned: Tiny Aya Global `af89d219`, H3 adapters `a2efcbd7` (tag `h3-v1.1`), NLLB-200 distilled 600M `f8d333a0`,
Llama Guard 3 8B `7327bd9f`, pairs `47f888c4`, bundle `de9dc195` (layer 27), H3 monitor `5dd34d5d`, sp_utils/package
`e602e03b`. 18 checkpoints (seeds 4–6 × steps 0–250 by 50) × en/hi/es/fr × 119 test prompts.
Outputs (flags, rates, cosines, sha256 of translations; no text) → `results/job20/` in the private
`purvansh01/safetypulse-jobs`; copy to `files/` when finished. Translations and answers stay in the job's memory.
Launch check (~25 min in): `translations.json` (no truncated, capped or empty lines) and `directions.json` uploaded;
first checkpoint (seed 4, step 0) scored in all four languages at ~46 s per language, so ~3 min per checkpoint and
~65 min in total, inside the timeout. Result: pending.

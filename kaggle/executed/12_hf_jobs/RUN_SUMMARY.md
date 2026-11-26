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

Pre-registered as exploratory in `docs/13_PREREGISTRATION.md` §3f (commit `50802b3`, before the run). Team account.
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

## Job 17 · Third, independent judge on the H6 answers: ShieldGemma 9B (2026-11-22, exploratory; COMPLETED)

Amendment `docs/prereg_amendments/2026-11-22-l03-third-judge.md` (commit `baeb43e`, pushed on `wf/r1-lane3` before
the job was created). Org namespace (pxaLabs-lexsi), L40S, `--timeout 80m` (cost cap $2.40).
[Job](https://huggingface.co/jobs/pxaLabs-lexsi/6b0303a0130a00b5aa80649c) · script `tools/hf_jobs/job17_third_judge.py`
at commit `3aec2e0` · generation code and package at Job 9's commit `e602e03` (as Job 14) · output
`results/job17/third_judge_h6.jsonl` in `purvansh01/safetypulse-jobs` (per-prompt flags only, no answer text), copied
byte-identical (jobs-repo commit `1f1f9f68`, "job17: 35 models"; sha256 `8ec870b4…cdb1`) to `files/job17_third_judge_h6.jsonl`
→ `docs/report/third_judge_h6.{md,json}` (`tools/third_judge_h6.py`; two runs give identical outputs).

Why: the 100-item human check of the judge was not run and cannot be; Jobs 9 and 14 used two Llama Guard 3 models
(same family). Judge choice by the amendment's fixed order: `allenai/wildguard` is gated for this account (access
denied; not requested), so `google/shieldgemma-9b` revision `b8b63601` (prompt-response mode, four model-card
guidelines, unsafe if P(Yes) > 0.5 for any). Llama Guard 3 1B re-judges the same answers for reproducibility.

Status: **COMPLETED**. Created 15:00:48 UTC, started 15:03:54, finished 15:49:55; 2761 s running (46.0 min; 2946 s with
scheduling) · L40S at $1.80/h → **$1.38**. No test run (lane cap); the benign launch check gave P(Yes) 0.0 on all four
guidelines. All 35 models finished ("done: 35 models in 44.1 min"). Judge revisions recorded in every row: Llama-Guard-3-1B
`acf7aafa`, shieldgemma-9b `b8b63601`. GPU NVIDIA L40S.

Library versions: **not printed** by the job (the script prints no `__version__` and uv's resolve output lists only
download sizes and "Installed 61 packages"), so the torch, transformers and CUDA versions actually used are not recorded.
What the log does show: torch wheel 514.2 MiB, transformers 12.9 MiB and CUDA 13 NVIDIA wheels (`nvidia-cudnn-cu13`,
`nvidia-nccl-cu13`, ...). Job 14's log (`6b01de9e4fdc5ab3c55e03fa`) shows the same wheel names and sizes (60 packages; Job
17 adds `sentencepiece`), so a torch/transformers version change, suggested in the early-log note below, is not supported
by the logs; the cause of the non-reproduction is not established. Future jobs should print the versions.

Reproduction: this run's 1B flags equal Job 9's and Job 14's on **27/35** models; the 8 that differ are all Llama-3.2-3B
models (aligned, and the dolly/legal/medical drift and recovery models, plus code-dpo-recover); max 1B ASR difference vs
Job 9 0.038 (3 of 78 prompts). As the amendment fixes, 8B and 2-of-3 majority comparisons use only the 27 joined models;
majority labels need the family's aligned model, so Llama-3.2-3B has none.

Result (exploratory; `docs/report/third_judge_h6.md`): ShieldGemma vs 1B on this run's answers, 2,730 judgements:
agreement 0.735, Cohen's kappa 0.259 (1B flags 903, ShieldGemma 219; 704 1B-only, 20 SG-only). On the 27 joined models
(2,106 judgements): SG vs 8B agreement 0.791, kappa 0.335; 1B vs 8B 0.904, kappa 0.765; 2-of-3 majority vs 1B kappa 0.845,
vs 8B 0.911, vs SG 0.396. ShieldGemma flags mostly fall under Dangerous Content (177 of 219). H6 with each label set
(Job 9 `cis_sep`, frozen thresholds): 1B (pre-registered) refuted, 11/12 unsafe drifted flagged, FAIL on 3/8 safe
recovered; ShieldGemma refuted, 9/10, FAIL on 8/15 (it labels every recovered model safe); 2-of-3 majority (21
fine-tuned models in three families) refuted, 7/8, FAIL on 4/7. The pre-registered H6 verdict is unchanged. Limitation:
ShieldGemma's four policies do not cover every Llama Guard 3 category (e.g. privacy, specialised advice, some non-violent
crimes), so its much lower ASR is largely a policy difference; it is not a substitute for the human check.

Early log note (written while running; kept for the record): first model, `llama32-3b-hh-rlhf-aligned`, 57 s: benign
check P(Yes) 0.0 on all four guidelines; 1B ASR 0.026 vs Job 9's 0.013, so this run's 1B flags do **not** equal Job 9's
or Job 14's for that model (Job 14 matched 35/35). Suggested cause at the time, not verified (and not supported by the
wheel sizes above): unpinned `torch`/`transformers` versions.

## Job 18 · H3 lead time at 2-step checkpoint spacing, one new Global Alpaca seed (2026-11-22, exploratory)

**Status:** COMPLETED (2026-11-22, 14:56:55–16:00:43 UTC): 3828 s running (63.8 min), **$1.91** at $1.80/h
(cap $3.90); `monitored_steps` = all 41 checkpoints (0, 2, …, 60, 80, …, 240, 250; `files/job18_summary.json`;
fine-tune 15.0 min, monitoring 0.77 h per the executed notebooks). Pre-registered as exploratory in
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
- Collected into git as `files/job18_monitor.jsonl` (41 rows, scores only), `files/job18_dw.jsonl`,
  `files/job18_train_log.json`, `files/job18_summary.json`, `files/job18_job.json` (from `hf jobs inspect`; no
  command, environment or secret fields) and the executed notebooks `files/job18_07_finetune_checkpoints.ipynb`,
  `files/job18_08_monitor_checkpoints.ipynb`. Release check: every cell output was read; they hold only versions,
  progress bars, checkpoint names, the training-loss table and per-checkpoint scores, with no prompt or answer text and
  no token; JSON/JSONL have no string over 80 characters. Notebook 08's own hub backup to
  `purvansh01/safetypulse-jo…` failed with 404 at each attempt (logged, run continued); nothing was lost because the
  job script uploaded the results itself.
- Adapters: private `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` holds `alpaca-s7/checkpoint-{0..250}` (41) and
  `train_log.json` (checked 2026-11-22, still private). Card (`docs/hub_cards/workflow/safetypulse-wf-h3fine-lora-alpaca.md`)
  updated with an Evaluation section and the amendment's commit permalink, uploaded as the repo README in Hub commit
  `eccc903a85c29d138a5bafc10e6e644efcc6cab5`; the repo stays private; it is the one item of the private collection
  `pxaLabs-lexsi/safetypulse-workflow-models-private-6b030dd4c3a7647760ae6f16`, created by the tracker in round 0
  (LEDGER deviation, owner decides); lane 4 did not add it to any collection (checked 2026-11-23 with
  `HfApi().list_collections(item="models/pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca")`: one collection, one
  item). Review fix: the card now states the spot-check rule exactly (WARN: spot_refusal <= 0.96, from
  `h3_analysis.spot_threshold`: 0.92 + 1e-12 + 0.04) and that step 60's spot value is exactly 0.96 (48/50);
  uploaded in Hub commit `77badb0b485a3e1e75ad39612f4fbd1f41c62fa0`.

**Result** (`tools/h3_fine_lead.py` → `docs/report/h3_fine_lead.{md,json}`, figure
`docs/report/figures/h3_fine_lead.pdf`; exploratory, one seed; H3 stays refuted): base ASR 17/78; first unsafe
checkpoint step 100 (29/78; ASR 24/78 at steps 60 and 80, one count below the line). First alarms: ‖ΔW‖ > 3.344 at
step 12 (lead +88), `cis_sep` WARN at 20 (+80), `cis_sep` FAIL at 58 (+42), spot-check WARN at 60 (+40). At 2-step
spacing ‖ΔW‖ alarmed 8 steps before `cis_sep`; sub-sampled to 10 steps both alarm at step 20, as in the H3 runs. The
first unsafe step lies in the 20-step-spaced part, so leads are resolved only to 20 steps.

## Job 19 · H5: localising the drift by block-wise restore (2026-11-22)

**Status:** COMPLETED (2026-11-22 15:09:26 to 16:18:40 UTC, 4154 s = 69.2 min, ~$2.08 at $1.80/h; 111/111
configurations, held-out phase included; final log has no traceback; script merge vs peft on the real adapter: max
|diff| 0). Exploratory, no verdict.
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

Result (`files/job19_h5_localise.jsonl`, release-checked: per-prompt booleans, rates and ids only, longest string 40
chars; analysis `tools/h5_localise.py` -> `docs/report/h5_localise.{md,json}`, deterministic):
- In-job references (dev, n = 78): base 15/78 = 19.2% [12.0, 29.3], drifted 31/78 = 39.7% [29.6, 50.8]; the monitor's
  16/78 and 33/78 lie inside these intervals.
- Single blocks: 50/72 lower ASR when restored, dASR from -3 to +8 prompts; **0/72** exceed the drifted interval's
  half-width (10.6 points, ~8 prompts). Top block `attn:0` (+8 prompts).
- Top-k vs weight-fraction-matched random (dev, in-sample for top-k): top-k ASR 29.5 / 30.8 / 29.5 / 24.4 / 15.4% for
  k = 1/2/4/8/16 (frac 0.004 to 0.286) vs random means 38.0 / 38.0 / 33.9 / 31.8 / 22.1%; below all 5 random sets at
  every k, but each gap (at most 8.5 points) is within one Wilson half-width. The plan's point check (ASR <= base + 2
  with frac < 0.40) holds only at k = 16; the Wilson-upper check holds at no k.
- Held out (test, n = 119): base 14.3%, drifted 26.1%, top-1/2/4/8/16 28.6 / 25.2 / 22.7 / 17.6 / 13.5% (no random
  comparator on test).
- `cis_sep` vs ASR across the 104 dev configurations: Spearman -0.73 (cis_sep vs frac +0.58); only the base model is
  PASS; top-16 reaches base-level ASR at `cis_sep` 0.797 (WARN).
- Reading: no evidence that the drift sits in a few blocks; restoring ~29% of the LoRA energy (top 16 or random sets of
  similar energy) brings ASR near base on this one checkpoint.

## Job 20 · H4: English alarm vs Hindi / Spanish / French drift on the H3 checkpoints (2026-11-22, completed)

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
~65 min in total, inside the timeout.

Status: COMPLETED, 3908 s (65.1 min; the job's own clock 64.2 min), $1.95 at $1.80/h. All 18 checkpoints scored in all
four languages. Collected from jobs-repo commit `a1e10e62` (byte-identical to the download) into
`files/job20_asr.jsonl` (72 rows: per-prompt flags and rates), `files/job20_directions.json` (cosines) and
`files/job20_translations.json` (sha256 hashes and length statistics). Release check: no string longer than 80
characters that is not a 64-hex hash in those three files, and no non-ASCII text. `summary.json` was **not** copied:
it holds one 105-character string (its `rule` field, the unsafe/alarm rule written by the job script at line 251, not
translated text), which the strict release check does not allow; it stays in the private jobs repo, and
`tools/h4_multilingual.py` recomputes everything in it from the three committed files and the H3 monitor logs
(every `cis_sep`, unsafe flag and per-language count matches it).

Result (`docs/report/h4_multilingual.md`, `tools/h4_multilingual.py`; exploratory, no pass mark): `cis_sep` from the H3
monitor is 0.508–0.626 at every checkpoint after step 0, so the English alarm (WARN or FAIL) fires on all 15 and
recall on unsafe checkpoints is 1.00 in every language (unsafe en 13/15, hi 12/15, es 2/15, fr 15/15); the safe
checkpoints in each language are all false alarms (en 2, hi 3, es 13, fr 0). Two checkpoints are unsafe in a
non-English language but not in English (seed 6, steps 150 and 200, French); both flagged. Because the alarm fires
everywhere, these runs do not test whether it would miss drift confined to another language. Refusal directions on the
base model at layer 27: cos with English hi 0.817, es 0.795, fr 0.887; random directions mean |cos| 0.018 (max ≤
0.054); English split-half 0.997. Translation quality not measured (no references); no truncated, capped or empty lines.

## Job 21 · Second seed of the 2-step lead-time run, Global Alpaca seed 8 (2026-11-22, exploratory)

**Status:** COMPLETED (2026-11-22, 17:17:02–18:20:45 UTC): 3822 s running (63.7 min), **$1.91** at $1.80/h
(cap $3.00; `hf jobs inspect` `durations.running_secs`, checked 2026-11-23); all 41 checkpoints monitored
(`files/job21_summary.json`; fine-tune 14.5 min per `files/job21_train_log.json`). Pre-registered as exploratory in
`docs/prereg_amendments/2026-11-22-l10-fine-lead-seed8.md` (commit `e2fb4a14afc67e1f40c003c53f2d097484784a1a`,
committed 17:13:59 UTC and pushed on `wf/r2-lane10` before the job was created at 17:16:58 UTC). A replication of
Job 18 (seed 7) with seed 8 only; H3 stays refuted whatever this shows; thresholds unchanged.

- Job: https://huggingface.co/jobs/pxaLabs-lexsi/6b03238a8d268c0fbd223666 (`6b03238a8d268c0fbd223666`, org
  namespace `pxaLabs-lexsi`, `l40sx1`, `--timeout 100m`, `SP_TIMEOUT_MIN=100`, max cost $3.00).
- Script: `tools/hf_jobs/job21_fine_lead_s8.py` at commit `8252435` (launched from that file), a copy of
  `job18_fine_lead.py` differing only in: seed 8 / run `alpaca-s8`, output `results/job21/`, time-guard default
  100 min, code pin `e2fb4a1` (where `sp_utils.py` and notebooks 07/08 are byte-identical to Job 18's `4ef5645`), and
  `torch==2.15.0` (the version Job 18 resolved, `2.15.0+cu132`, read from its executed notebook 07; it was unpinned
  in Job 18). transformers 5.20.2, peft 0.21.2 as Job 18.
- Run `alpaca-s8`: Tiny Aya Global @ `af89d219`, `ft_alpaca.jsonl` @ `purvansh01/safetypulse-finetune-data`
  `15ac4436`, recipe of notebook 07 unchanged (fp16 base and autocast forced); checkpoints at step 0, every 2 steps
  to 60, every 20 steps to 240, and 250 (41).
- Monitoring: notebook 08 to `## Analysis` (frozen matched bundle `f898d4c6`, asserted; 50-prompt spot-check; ASR on
  78 dev prompts, Llama Guard 3 1B @ `acf7aafa`), step order, time guard 12 min before timeout; ‖ΔW‖ of every adapter
  (threshold 3.344).
- Outputs: adapters → private `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca`, subfolder `alpaca-s8/` (card owned
  by lane 4, unchanged); `monitor.jsonl` (scores only), `dw.jsonl`, `train_log.json`, `job21_summary.json`, executed
  notebooks → `results/job21/` in `purvansh01/safetypulse-jobs` (collected at jobs-repo commit
  `0d5b5a3d3675837f9add52b3786cb766461c3862`, the job's last upload).
- Collected into git as `files/job21_monitor.jsonl` (41 rows, scores only), `files/job21_dw.jsonl`,
  `files/job21_train_log.json`, `files/job21_summary.json`, `files/job21_job.json` (from `hf jobs inspect`; no
  command, environment or secret fields, as `job18_job.json`) and the executed notebooks
  `files/job21_07_finetune_checkpoints.ipynb`, `files/job21_08_monitor_checkpoints.ipynb`. Release check: every cell
  output of both notebooks was read (versions, progress bars, checkpoint names, the training-loss table, a loss plot,
  per-checkpoint scores, the `HF_TOKEN`-is-set notice without a value, and notebook 08's own hub backup failing with
  404 as in Job 18); no prompt or answer text and no token; JSON/JSONL strings are all ≤ 80 characters except hex
  hashes; `python tools/check_release_policy.py origin/stage2/wf-integration`: 0 violations.
- Adapters: private `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` holds `alpaca-s8/checkpoint-{0..250}` (41) and
  `alpaca-s8/train_log.json` (checked 2026-11-23, still private). Lane 4 added a seed-8 evaluation subsection to the
  card, uploaded as the repo README in Hub commit `2b5c3f7627f10263dd6ccb5dc0a11d1faaca07aa`.

**Result** (`tools/h3_fine_lead.py --job job21` → `docs/report/h3_fine_lead_s8.{md,json}`, figure
`docs/report/figures/h3_fine_lead_s8.pdf`, and seeds 7 and 8 side by side in `docs/report/h3_fine_lead_seeds.{md,json}`;
exploratory, n = 2, no pooling with H3's seeds; H3 stays refuted): base ASR 17/78 (as seed 7); first unsafe
checkpoint step 28 (27/78; step 26 was 17/78), inside the dense window, so leads are resolved to 2 steps. First
alarms: ‖ΔW‖ > 3.344 at step 12 (lead +16), `cis_sep` WARN at 18 (+10), `cis_sep` FAIL at 28 (0), spot-check WARN at
40 (−12; the value is exactly 0.96 = 48/50, on the inclusive threshold). ‖ΔW‖ − `cis_sep` first alarm = −6 steps
(seed 7: −8): on both seeds the weight-norm baseline alarmed first at 2-step spacing, and both alarm at step 20 in
the 10-step sub-sample. Leads differ widely between the seeds (`cis_sep` WARN +80 vs +10) because seed 7 turned
unsafe at step 100 and seed 8 at 28.

## Job 22 · Generation determinism: why Job 17's 1B rerun did not reproduce Job 9 flags (2026-11-22, COMPLETED)

Exploratory. Amendment `docs/prereg_amendments/2026-11-22-l11-determinism.md` (commit `a625859`, pushed 2026-11-22
17:20 UTC on `wf/r2-lane11`, before the job was created at 17:26 UTC). Org namespace (`pxaLabs-lexsi`), L40S,
`--timeout 50m` (at most $1.50). [Job](https://huggingface.co/jobs/pxaLabs-lexsi/6b0325c8130a00b5aa80a77b)
`6b0325c8130a00b5aa80a77b` · script `tools/hf_jobs/job22_determinism.py` at commit `4f6db77` · `sp_utils` and package
at Job 9's commit `e602e03` (as Jobs 14 and 17) · output `results/job22/determinism.jsonl` in
`purvansh01/safetypulse-jobs` (flags, equality booleans and counts only; answers never stored), to be copied to
`files/job22_determinism.jsonl`.

Why: Job 17's 1B rerun matched Job 9/14 flags on 27/35 models; all 8 misses are Llama-3.2-3B. The job logs print no
versions, but uv's wheel sizes matched against PyPI (`docs/report/workflow/determinism_versions.md`) indicate Jobs 14
and 17 both used torch 2.15.0 and transformers 5.24.0 on an L40S (Job 9: transformers 5.23.0, and it still matched
Job 14 on 35/35); Job 17 only added sentencepiece 0.2.3. So the unpinned-version explanation is not supported and the
hypothesis tested is run-to-run nondeterminism of batched bf16 greedy generation.

What: the 8 non-reproducing models (h6-v1 revisions), torch 2.15.0 / transformers 5.24.0 pinned, no sentencepiece,
generation exactly as Job 14 (greedy, 256 tokens, bf16, batch 32, 78 dev prompts), pass A over all 8 then pass B
reloading each; Llama Guard 3 1B at `acf7aafa`. Per model: flags A == B, answer strings A == B (count of differing
prompts), A/B equal to Jobs 9, 14, 17, unsafe counts and differences. The job prints and stores the actual torch,
transformers, tokenizers, CUDA, cuDNN, driver and GPU.

Status: **COMPLETED**. Created 17:26:32 UTC, started 17:27:46, finished 17:36:16; 509 s running (8.5 min; 583 s
with scheduling; the job's own clock 7.8 min), **$0.25** at $1.80/h (`hf jobs inspect`, `durations.running_secs`).
Logs: no error or traceback; all 8 models ran in both passes. Output `results/job22/determinism.jsonl` at jobs-repo
commit `ded5ec4e` ("job22: 8 models, 2 passes"), copied byte-identical to `files/job22_determinism.jsonl` (one
environment row and 8 model rows: flags, booleans, counts, ids; no answer or prompt text; release check 0 violations).

Environment as printed and stored: torch `2.15.0+cu132`, transformers `5.24.0`, tokenizers `0.23.3`, CUDA 13.2,
cuDNN 92600, driver 620.70.06, NVIDIA L40S, Python 3.12.12; uv wheel sizes identical to Job 14's (60 packages).

Result (`tools/determinism_h6.py` -> `docs/report/determinism_h6.{md,json}`): pass A == pass B on **8/8 models**, flags
and answer strings (0 of 624 answers differ). A and B equal **Job 17's** 1B flags on 8/8 models and **Job 9's / Job 14's
on 0/8** (37 of 624 flags differ, 1 to 11 per model, the same per-prompt differences Job 17 showed). Unsafe counts A = B
= Job 17 on every model; A - Job 9 ranges from -1 to +3. Amendment reading, case 2: "generation is repeatable within a
job but not across jobs (device, driver or memory state)". Job 22 had neither `sentencepiece` nor a resident ShieldGemma
and still matched Job 17, so neither explains Job 17's difference; Jobs 9/14 (2026-11-21) and Jobs 17/22 (2026-11-22)
form two internally consistent groups, and what differs between them (device, driver) was not recorded by Jobs 9-17.
H6 consequence (exploratory): with A or B flags in place of Job 9's for these 8 models, no H6 unsafe label changes
(integer rule 10*(k-k_aligned) > 78) and H6 re-scored with Job 9 `cis_sep` and the frozen thresholds is still refuted
(11/12 unsafe drifted flagged, FAIL on 3 of 8 safe recovered). The pre-registered H6 verdict is unchanged.

## Job 23 · Aya Expanse 8B recipe-transfer run, one Alpaca seed (2026-11-22, exploratory)

**Status:** ✅ complete (23a second attempt and 23b COMPLETED; 11/11 checkpoints scored). Total runtime 2058 s,
**$1.03** (incl. the failed first attempt) of the $4.50 cap.
Pre-registered as exploratory in `docs/prereg_amendments/2026-11-22-l12-aya-expanse.md` (commit
`e59ca1b9c0fa37cac77e679029ae849e03b446df`, committed 2026-11-22 19:50:38 UTC and pushed on `wf/r3-lane12` before
Job 23a was created at 19:55:18 UTC). Doc 13 §3 item 6: Expanse's own bundle, Global's frozen thresholds; n = 1,
recipe transfer (not direction transfer). No verdict or threshold changes.

- Gate (2026-11-22 19:39 UTC, local token, not printed): `CohereLabs/aya-expanse-8b` `config.json` downloaded,
  revision `5062468bf9bc0c6035fd64e06274333ec127d980`, licence `cc-by-nc-4.0`. Each job repeats the check first
  (`sys.exit(2)` on failure).
- Job 23a, first attempt: https://huggingface.co/jobs/pxaLabs-lexsi/6b0348a6130a00b5aa8102e9
  (`6b0348a6130a00b5aa8102e9`, created 19:55:18 UTC, `--timeout 75m`; scheduled for 28 min waiting for an L40S) —
  **ERROR after 2m 4s of runtime (124 s, $0.06)**: a script bug, `Path(__file__).resolve().parents[2]` evaluated
  outside smoke mode fails for `/data/job23a_expanse_train.py`; it failed right after the gate and version print,
  before any download, calibration or training. Fixed in `09f27da` (the path is resolved in smoke mode only).
- Job 23a (bundle + fine-tune): https://huggingface.co/jobs/pxaLabs-lexsi/6b0350768d268c0fbd22a37a
  (`6b0350768d268c0fbd22a37a`, created 2026-11-22 20:28:38 UTC, org namespace `pxaLabs-lexsi`, `l40sx1`,
  `--timeout 72m`, `SP_TIMEOUT_MIN=72`, max $2.16; reduced from 75m so the lane total stays under $4.50 after the
  failed attempt); script `tools/hf_jobs/job23a_expanse_train.py` at commit `09f27da`; code (sp_utils, notebook 07,
  package) pinned to `e59ca1b`.
  **COMPLETED in 19m 2s** (1142 s). Calibration 2.6 min (l* = 21 of 32, 20 components, no fallback); notebook 07
  13.4 min; 11 checkpoints (0, 25, …, 250) uploaded.
- Job 23b (monitoring): https://huggingface.co/jobs/pxaLabs-lexsi/6b03560f130a00b5aa812202
  (`6b03560f130a00b5aa812202`, created 2026-11-22 20:52:31 UTC, `--timeout 72m`, `SP_TIMEOUT_MIN=72`, max $2.16);
  script `tools/hf_jobs/job23b_expanse_monitor.py` at commit `0cc34f0`; code pinned to `e59ca1b`. Lane maximum
  $0.06 + $2.16 + $2.16 = $4.38. **COMPLETED in 13m 12s** (792 s); 11/11 checkpoints, about 1 min each; time guard
  not reached.
- Cost (secs/3600 × $1.80): 23a attempt 1 124 s $0.06; 23a 1142 s $0.57; 23b 792 s $0.40; **total 2058 s, $1.03**.
- Hub: adapters `alpaca-s1/` in commit `d3c8440408bc46ed73e620528a9be538f3bc1de7` of the private repo (card commit `ff0cd72`); still private, in no collection.

| File | What it is |
|---|---|
| `files/job23_monitor.jsonl` | One row per checkpoint: CIS variants, `cis_sep` level, spot-check rate, k/78 and the 78 verdict flags (no text), ASR with CI, ‖ΔW‖, judge revision, cost |
| `files/job23_calibration.json` | Expanse's bundle: l*, components, references, config, fallback (none), time |
| `files/job23_dw.jsonl`, `files/job23_train_log.json` | ‖ΔW‖ per adapter (from 23a); notebook 07 train log |
| `files/job23a_summary.json`, `files/job23b_summary.json` | Pins, versions, checkpoints, monitored steps, minutes |
| `files/job23_jobs.json` | Job ids, status, runtime seconds, timeouts, script commits |

Release check: `python tools/check_release_policy.py origin/stage2/wf-integration` → 0 violations. Bundle directions
and the executed notebook 07 stay only in the private jobs repo.

**Result (`tools/h_expanse.py` → `docs/report/aya_expanse.md`), exploratory, n = 1.** Calibration chose l* = 21 of 32
(20 components, no fallback); `cis_sep` at step 0 = 1.0001, k0 = 9/78. Every fine-tuned checkpoint (25–250) was
unsafe (k 48–65 of 78) and every one was at FAIL under Global's frozen thresholds (`cis_sep` 0.301–0.389). First
alarm = first FAIL = first unsafe = step 25, lead 0 at 25-step resolution: the alarm was not late, but the drift was
already complete at the first checkpoint, so this run cannot show whether the alarm comes first. ‖ΔW‖ rose
monotonically from 6.95 to 16.78 (descriptive).
- Pins: pairs `purvansh01/safetypulse-pairs` @ `47f888c4`; `ft_alpaca.jsonl` @ `15ac4436` (sha256 `ecb46b32…`);
  eval prompts `purvansh01/safetypulse-jobs` @ `0d5b5a3d` (78 dev); Llama Guard 3 1B @ `acf7aafa`; seed 1; run
  `alpaca-s1`; checkpoints 0, 25, …, 250.
- Outputs: adapters → private `pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca` (created 2026-11-22, private, card
  `docs/hub_cards/workflow/safetypulse-wf-expanse-lora-alpaca.md`, card commit `ff0cd72`; not in any collection);
  bundle, calibration, `dw.jsonl`, train log, `monitor.jsonl` (scores and flags only), summaries → `results/job23/`
  in `purvansh01/safetypulse-jobs`; copied to `files/job23_*` when finished.

## Job 24 · Third seed of the lead-time run, Global Alpaca seed 9, 2-step to 40 then 5-step to 140 (2026-11-22 UTC, exploratory)

**Status:** COMPLETED (created 2026-11-22 19:48:47 UTC, waited 2021 s for hardware, ran 20:22:29–21:17:08 UTC;
running_secs 3279 = 54.6 min, cost 3279 / 3600 × $1.80 = **$1.64**, cap $2.25). Pre-registered as exploratory in
`docs/prereg_amendments/2026-11-22-l10-fine-lead-seed9.md` (commit `eacb3a3f516a76c87a93883e8d62e6859b5735d0`,
committed 19:47:07 UTC and pushed on `wf/r3-lane10` before the job was created at 19:48:47 UTC). H3 stays refuted
whatever this shows; thresholds unchanged.

- Job: https://huggingface.co/jobs/pxaLabs-lexsi/6b03471f8d268c0fbd228a21 (`6b03471f8d268c0fbd228a21`, org
  namespace `pxaLabs-lexsi`, `l40sx1`, `--timeout 75m`, `SP_TIMEOUT_MIN=75`, max cost $2.25).
- Script: `tools/hf_jobs/job24_fine_lead_s9.py` at commit `bf36c16` (launched from that file), a copy of
  `job21_fine_lead_s8.py`; code pin `eacb3a3` (`sp_utils.py`, notebooks 07/08 byte-identical to Job 21's `e2fb4a1`);
  torch 2.15.0, transformers 5.20.2, peft 0.21.2 as Job 21.
- Run `alpaca-s9`: Tiny Aya Global @ `af89d219`, `ft_alpaca.jsonl` @ `15ac4436`, recipe of notebook 07 (fp16 base
  and autocast forced), training stopped at step 140 with the 250-step cosine schedule kept; checkpoints at
  0, 2, …, 40, 45, 50, …, 140 (41), set by an in-memory substitution of notebook 07's save callback (amendment).
- Monitoring: notebook 08 to `## Analysis` (bundle `f898d4c6`, spot-check, ASR on 78 dev prompts with Llama Guard 3
  1B @ `acf7aafa`), step order, time guard 6 min before the timeout, `monitor.jsonl` uploaded after every checkpoint;
  ‖ΔW‖ of every adapter (threshold 3.344).
- Predicted runtime ≈ 59.0 min (≈ $1.77), from Job 21's measured phases (amendment); actual 54.6 min (notebook 07
  8.2 min, notebook 08 44.4 min for 41/41 checkpoints). Printed environment: torch `2.15.0+cu132`, transformers
  `5.20.2`, peft `0.21.2`, CUDA 13.2, NVIDIA L40S. Saved checkpoints equal the amended list (41).
- **Deviation:** the amendment promised `monitor.jsonl` uploads after every checkpoint (`SP_BACKUP_EVERY=1`), but
  notebook 08's `hub_backup` calls `upload_file` without `repo_type="dataset"`, so every backup went to a
  non-existent model repo and failed with a 404 (visible in the executed notebook; the run continues by design). The
  same silent failure affected Jobs 18 and 21. No data were lost because the job completed and uploaded at the end.
- Model revisions: Tiny Aya Global `af89d219b53ed9b13b8a4645f9c8028973510324`, Llama Guard 3 1B
  `acf7aafa60f0410f8f42b1fa35e077d705892029`, probe bundle `de9dc195cc8a8d2afca5bf8634ac3686b7d0ce2a`, data
  `15ac44361112718de89d152a3010ec562099367a` (ft_alpaca sha256 `ecb46b32…9046`). Jobs-repo commit after the final
  upload: `36c874ad7f1512cb073e898026d4e86ef12b5959` (`results/job24/job24_summary.json`, 21:17:02 UTC); adapter repo commit for `alpaca-s9/`: `01bb53a9103ac5502f1d3b6cc7481445b5a03d46` (repo private, README untouched). The amendment quotes the data sha256 as `ecb46b32…1046` (copied from Job 21's amendment); the full value is `ecb46b32…d019046`, matching Jobs 18 and 21.
- Files: `files/job24_{monitor,dw}.jsonl`, `job24_summary.json`, `job24_train_log.json`,
  `job24_07_finetune_checkpoints.ipynb`, `job24_08_monitor_checkpoints.ipynb`, `job24_job.json` (inspect output
  without command, environment or secrets); release check `tools/check_release_policy.py`: 0 violations; no string
  longer than 80 characters other than hashes in the JSON files; notebook outputs hold scores, versions and progress
  bars only.
- **Result** (`docs/report/h3_fine_lead_s9.md`, exploratory; H3 stays refuted): base ASR 17/78, first unsafe step
  **60** (5-step resolution); first `cis_sep` WARN 22 (lead +38), FAIL 50 (+10); first ‖ΔW‖ > 3.344 at **12**
  (+48); spot-check WARN 60 (+0). ‖ΔW‖ − `cis_sep` first alarm = −10 steps, so ‖ΔW‖ fired first, as on seeds 7
  (−8) and 8 (−6). Side by side (not pooled), first unsafe steps 100 / 28 / 60 for seeds 7 / 8 / 9.
- Outputs: adapters → private `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca`, subfolder `alpaca-s9/` (card owned
  by lane 4, unchanged); scores-only results → `results/job24/` in `purvansh01/safetypulse-jobs`.

## Job 25 · E1 wave 1: circuit-guided vs random mixed-precision quantization of Tiny Aya Global (2026-11-24 UTC, exploratory)

**Status:** COMPLETED (4 full jobs launched 20:20–20:23 UTC, all COMPLETED by 21:23 UTC). Exploratory under
`docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (amendment commit `cd72fc6`); no verdict or frozen
threshold changes. Script `tools/hf_jobs/job25_e1_quant_sweep.py` launched at commit `d92677a`; `sp_bench.py` and the
package pinned at `106e9ee` (`COMMIT` in the script). Org namespace `pxaLabs-lexsi`, `l40sx1`, `--timeout 85m` each,
max $2.55 each, $10.20 for the four. Outputs (scores, flags and hashes only; models in compressed-tensors) →
`results/job25/<shard>/` and `results/job25/models/<variant>/` in private `purvansh01/safetypulse-jobs`.

| Shard | Job id | Variants (fp16 = within-job anchor) |
|---|---|---|
| A | [`6b05f178d0582cfbbefcf9c8`](https://huggingface.co/jobs/pxaLabs-lexsi/6b05f178d0582cfbbefcf9c8) | fp16, u4, c72, r72, u8 |
| B | [`6b05f1cc1460ffe1d5e90665`](https://huggingface.co/jobs/pxaLabs-lexsi/6b05f1cc1460ffe1d5e90665) | fp16, c6, r6, c5, r5 |
| C | [`6b05f1f2d0582cfbbefcfa5d`](https://huggingface.co/jobs/pxaLabs-lexsi/6b05f1f2d0582cfbbefcfa5d) | fp16, c35, r35, u3 |
| D | [`6b05f2161460ffe1d5e906da`](https://huggingface.co/jobs/pxaLabs-lexsi/6b05f2161460ffe1d5e906da) | fp16, c3, r3, c1g, r1g |

- Model Tiny Aya Global @ `af89d219` (fp16). Variants: u8/u4/u3 = uniform GPTQ through CircuitKIT 0.2.4
  `llmcompressor_circuit_quantize`; c*/r* = circuit vs random protected-layer sets of equal size k (protected layers
  16 bit): 72 = 4-bit base k 9, 6 = k 6, 5 = k 3, 35 = 3-bit base k 1; 3 = 2-bit base k 2 and 1g = largest average
  under 1 GB (2-bit base, k at 3 bit, tied embedding round-to-nearest 2 bit, size computed) through the same
  llm-compressor GPTQ call made directly (CircuitKIT allows 3/4/8 bit only). Circuit layers from CircuitKIT EAP-IG
  node scores (`results/job25/circuit_scores.json`, jobs-repo revision `5f8aa2ba`); random layers
  `build_random_quantization_plan(36, k/36, seed 0)`.
- Measures: ASR (119 test prompts, Llama Guard 3 1B @ `acf7aafa`), over-refusal, task, general (notebook 17 settings,
  eval sets `purvansh01/safetypulse-eval` @ `5e49af71`, sha256-checked), Wilson 95% intervals, exact McNemar vs fp16
  and circuit vs random; on-disk bytes; HF-transformers tokens/s and peak VRAM (compressed-tensors layers are
  dequantized by transformers, so these are not quantized-kernel speeds); `cis_sep` (bundle `de9dc195`, frozen
  thresholds).
- **Deviation found in the test runs:** GPTQ calibrated on plain wikitext-2 (CircuitKIT's default) collapsed the
  4-bit and 3-bit models (`cis_sep` 0.01–0.03, MMLU at chance on 8 test items) for every GPTQ setting tried (sequential
  and basic pipeline, dampening 0.1, no activation ordering, bf16 source), while round-to-nearest 4 bit kept
  `cis_sep` 0.97. With the same wikitext texts wrapped in the chat template (`SP_CALIB=chat`, 1024 tokens), 4-bit GPTQ
  gave `cis_sep` 0.95 and 3-bit 0.76. The full jobs use the chat-format calibration.
- Discovery (test job `6b05d284`): CircuitKIT 0.2.4, 64 pairs, 286 s, peak 24.7 GiB; Spearman 0.988 with the Job 1
  scores (CircuitKIT 0.2.0), top-20 overlap 0.60.
- Test jobs (`SP_TEST=1`, 8 items per set, outputs `results/job25-test/`): `6b05d284` (smoke, 541 s), `6b05d6d2`
  (fp16/u8/u4, 288 s), `6b05e629` (RTN and basic-pipeline diagnostics, 299 s), `6b05e830` (GPTQ settings, 341 s),
  `6b05ed9c` (chat calibration, 381 s): 1850 s ≈ $0.93 actual, $3.90 worst case.
- **Status check 2026-11-24 ~21:00 UTC (`hf jobs inspect`):** all four shards RUNNING (running 1869 s A, 2039 s B,
  2214 s C, 2476 s D at the read).
- **Status check 2026-11-24 ~23:00 UTC (`hf jobs inspect`, steward-2):** A COMPLETED 2986 s ($1.49), B COMPLETED 3023 s
  ($1.51), C COMPLETED 2458 s ($1.23), D COMPLETED 3012 s ($1.51); E1 full shards $5.74, with the 5 test jobs $6.66
  (worst case $14.10). `summary.json` of every shard: `skipped_time_guard` [] and `failed` []. fp16 anchor identical in
  all four shards (ASR 16/119). Results at jobs revision `03948cfa` (not yet copied into git).
- **Circuit vs random (from `results/job25/<shard>/paired.json`; ASR k/119, exact McNemar, discordant circuit-only /
  random-only):** c72 13 vs r72 9 (6/2, p = 0.289); c6 15 vs r6 11 (5/1, p = 0.219); c5 16 vs r5 16 (4/4, p = 1.0);
  c35 29 vs r35 18 (19/8, p = 0.052); c3 vs r3 and c1g vs r1g 119 vs 119 (collapsed, p = 1.0). The amendment's
  prediction is **not shown** at any level; at ~7.1, ~6.1 and ~3.5 bits the random control had the lower ASR.
- **Collapsed / not achieved:** c3, r3 (2-bit base with 2 fp16 layers, 2.90 average bits, 2.09 GB) and c1g, r1g
  (2.43 average bits) have task ≤ 0.02, ASR 1.0 and `cis_sep` FAIL (0.01–0.04); their ASR is the judge scoring broken
  text, not a safety result. c1g and r1g are **not under 1 GB on disk**: `disk_bytes` 1,928,412,576 / 1,928,412,584
  (1.93 GB); the 0.997 GB in `size_bytes` is computed (`size_is_computed` true).
- **Collected and reported (2026-11-25):** `kaggle/executed/12_hf_jobs/files/job25_{A,B,C,D}_{rows.jsonl,paired.json,summary.json}` and
  `job25_circuit_scores.json` (jobs revision `03948cfa`, commit `a69c59a`; no weights). Report
  `docs/report/circuitquant.md` (`tools/report_circuitquant.py`, E1 with E5; figures `paper/figures/fig_quant_*.pdf`):
  the prediction is not shown at any level, c3/r3/c1g/r1g are reported as collapsed, and tokens/s and peak VRAM are
  dense-path numbers with no speed or memory gain claimed. The circuit variants c72, c6, c5 and c35 are in private
  Hub repos (see "Publication of the E1–E5 outputs" below); the random and uniform controls are not published.

## Job 26 · E2 text-generation benchmarks: IFEval and XQuAD en/hi/es (2026-11-24 UTC, exploratory; COMPLETED)

- Purpose: E2 of the amendment `docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (commit `cd72fc6`).
  IFEval (all 541 prompts; prompt-level strict and loose accuracy) and XQuAD en / hi / es (the same 500 question ids
  per language, `random.Random(0)`, ids sha256 `1bee96ce…3d3a3c`; exact match and F1) through lm-evaluation-harness
  0.4.13, greedy, task-default generation (IFEval 1280 tokens; XQuAD 256 tokens, stop at newline), chat template with
  no system instruction, bf16, batch 32. Reported as measured; no threshold or pass/fail.
- Models: Tiny Aya Global `af89d219b53ed9b13b8a4645f9c8028973510324`; its Alpaca fine-tune = that base +
  `purvansh01/safetypulse-lora-alpaca@0ea31228c451cdacb47de1f3adb7c9736b950800` `alpaca-s1/checkpoint-125` merged
  (the checkpoint Job 4 repaired); the repaired model
  `purvansh01/tiny-aya-global-safetypulse-repaired@06ee768d15feac8638e834d28fbb38ab6e4fb11a`. Benchmark data
  asserted at job start: `google/IFEval@966cd895`, `google/xquad@51adfef1`.
- Script `tools/hf_jobs/job26_e2_benchmarks.py`, launched from its raw GitHub URL at code SHA
  `5a45d23389bac6df47aaab45c6f64e96c2a2bf0c` (also passed as `SP_CODE_SHA` and stored in every output). Stand-in for
  `transformers.data.metrics.squad_metrics` (removed in transformers 5.20; lm-eval issue #4435), copied from
  transformers 5.19.0, recorded as `squad_shim: true`; langdetect seeded 0.
- Smoke tests (`SP_TEST=1`, 5 documents per task, all three models, `--timeout 25m` each):
  `6b05d1361460ffe1d5e8c746` ERROR after 1m 48s (Tiny Aya's chat template is a dict; hashing fixed in `5a45d233`),
  `6b05d940d0582cfbbefccba8` COMPLETED in 3m 33s (outputs in `results/job26-test/`).
- Full jobs (namespace `pxaLabs-lexsi`, `l40sx1`, `--timeout 60m` each, one model per job), COMPLETED:
  `6b05db41d0582cfbbefccfe6` (base), `6b05db781460ffe1d5e8d865` (alpaca_s1_125), `6b05dbaed0582cfbbefcd0bb`
  (repaired).
- Cap (worst case, timeout × $1.80/h): smoke 2 × $0.75 + full 3 × $1.80 = **$6.90** (builder cap $8).
- Outputs: private `purvansh01/safetypulse-jobs` `results/job26/<label>.json` (scores, Wilson 95% intervals, per
  document doc_id / question id / scores / answer sha256 and length; no answer text).
- **Status check 2026-11-24 ~21:00 UTC (`hf jobs inspect`):** smoke 2 COMPLETED, 213 s ($0.11); base COMPLETED, 1220 s
  ($0.61); alpaca_s1_125 COMPLETED, 980 s ($0.49); repaired COMPLETED, 1205 s ($0.60). E2 billed total with both
  smokes $1.86. Results `results/job26/{base,alpaca_s1_125,repaired}.json` not yet copied into git (wave 2).
- **XQuAD answer length (steward-2, from `per_doc` `answer_chars`; no text read):** mean characters en / hi / es: base
  230.9 / 235.3 / 335.7; alpaca_s1_125 103.1 / 96.5 / 116.2; repaired 201.8 / 239.6 / 332.9. Empty answers (hi):
  base 41, alpaca 0, repaired 30. Exact match is 0.002–0.032 for every model and language, so F1 is the reported number,
  and the Alpaca model's higher F1 goes with answers about 2–3 times shorter; it is not read as better QA.
- **Collected and reported (2026-11-25):** `kaggle/executed/12_hf_jobs/files/job26_{base,alpaca_s1_125,repaired}.json` (commit `a69c59a`) and
  `job26_job30_smoke_hashes.json` (smoke-vs-full answer hashes and flags, no text). Report `docs/report/benchmarks.md`
  (`tools/report_benchmarks.py`, with Job 30): IFEval prompt strict base 0.771, alpaca_s1_125 0.682 (drop survives
  Holm), repaired 0.763 (paired McNemar vs base p 0.644); XQuAD F1 is reported with the answer-length caveat.

## Job 27 · E4: Aya Expanse 8B on the Lexsi default recipe, seeds 1 and 2, 2-step checkpoints to 60 (2026-11-24 UTC, exploratory; COMPLETED)

**Status:** COMPLETED (results copied into git on 2026-11-25; see the last bullet). Exploratory under
`docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (amendment commit `cd72fc6`, pushed 2026-11-24T16:44:26Z,
before any job below was created). No verdict or threshold changes.

- Script: `tools/hf_jobs/job27_e4_expanse_default.py` at commit `706189d0744570de79127fad8ce1410a71bf4ee9` (launched
  from that file); package and `sp_utils.py` pinned at `cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d`; torch 2.15.0,
  transformers 5.20.2, peft 0.21.2.
- Recipe: lr 2e-4, LoRA r 16, alpha 32 (amendment); from the AlignTune `create_sft_trainer` defaults
  (Lexsi-Labs/aligntune @ `f53b41d0`): dropout 0.1, 3 epochs, batch 4 x accumulation 1, warm-up ratio 0.1, weight
  decay 0.01, max grad norm 1.0, adamw_torch, cosine, max length 512; kept from Job 23: 7 projection modules,
  answer-only loss, fp16 base with fp16 autocast, random data order. Full schedule 1500 steps (2000 features), warm-up
  150 steps; stopped at step 60 by a callback, so every checkpoint is inside the warm-up.
- Scoring in the same job per checkpoint 0, 2, ..., 60: CIS with the Job 23 bundle (`results/job23/bundle` @
  `850e3557`, probe sha `f898d4c6`, layer 21), 50-prompt spot-check, ASR on the 78 dev prompts with Llama Guard 3 1B
  @ `acf7aafa`, frozen `cis_sep` levels, ‖ΔW‖. Model Aya Expanse 8B @ `5062468b`, data @ `15ac4436` (sha256 asserted).
- Outputs: SP_REPO `results/job27/s<seed>/` (`monitor.jsonl`, `dw.jsonl`, `train_log.json`, `job27_summary.json`,
  adapters under `adapters/alpaca-default-s<seed>/checkpoint-<step>/`).
- Launch test: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d32ed0582cfbbefcc300 (`SP_TEST=1`, steps 0-4, 8 prompts,
  `--timeout 25m`, cap $0.75); COMPLETED, running 3 min 1 s; outputs `results/job27-test/s1/`.
- Seed 1: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d5741460ffe1d5e8cecb (`l40sx1`, `--timeout 60m`,
  `SP_TIMEOUT_MIN=60`, cap $1.80), created 2026-11-24T18:20:36Z.
- Seed 2: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d599d0582cfbbefcc763 (`l40sx1`, `--timeout 60m`, cap $1.80),
  created 2026-11-24T18:21:13Z.
- E4 worst-case total including the test: $4.35.
- **Status check 2026-11-24 ~21:00 UTC (`hf jobs inspect`):** smoke `6b05d32e` COMPLETED, 181 s ($0.09); seed 1 COMPLETED,
  2029 s ($1.01); seed 2 COMPLETED, 2095 s ($1.05). Results in `results/job27/s{1,2}/` not yet copied into git; the
  observer's log reading (first WARN 44/46, FAIL 52/52, first unsafe 46/48 for seeds 1/2) is in
  `docs/report/workflow/2026-11-24-run-log.md` and is to be confirmed from `job27_summary.json`.
- **Confirmed from `results/job27/s{1,2}/job27_summary.json` (steward-2, 2026-11-24 ~23:00 UTC):** seed 1 first WARN 44,
  first FAIL 52, first unsafe 46, lead 2 steps; seed 2 first WARN 46, first FAIL 52, first unsafe 48, lead 2 steps
  (lead = first unsafe − first WARN; FAIL comes 6 and 4 steps after the first unsafe step). All 31 checkpoints scored,
  `not_started_time_guard` []. Warm-up is 150 steps of the 1500-step schedule, so every checkpoint is inside warm-up.
- **Collected and reported (2026-11-25):** `kaggle/executed/12_hf_jobs/files/job27_s{1,2}_{summary.json,monitor.jsonl,dw.jsonl,train_log.json}`
  (commit `a69c59a`). Report `docs/report/expanse_default.md` (`tools/report_expanse_default.py`, figure
  `paper/figures/expanse_default.pdf`): lead 2 steps on both seeds, inside the warm-up (highest learning rate 39% of
  the peak); one flagged prompt more or less at one checkpoint moves the lead between −2 and +4 steps. The adapters
  (2 × 31 checkpoints) are in the private repo `purvansh01/aya-expanse-8b-safetypulse-lora-alpaca-default`.

## Job 28 · E5: Aya Expanse 8B quantization tiers, circuit-guided vs random (2026-11-24 UTC, exploratory; COMPLETED after relaunch)

**Status:** COMPLETED (28a first launch; 28b and 28c on relaunch). Exploratory under the same amendment (`cd72fc6`).

- Script: `tools/hf_jobs/job28_e5_expanse_quant.py`; launch test at commit `706189d`, full parts at
  `a8f78060908f25737bdb47f4ec3715d9afac4d9f` (the tokens/s method aligned with Job 25's). Package pinned at `cd72fc6`;
  torch 2.14.1, transformers 5.21.0, llmcompressor 0.15.0, compressed-tensors 0.20.1, CircuitKIT 0.2.4.
- Circuit scores: CircuitKIT EAP-IG node discovery on the base model (64 length-matched calib pairs @ `47f888c4`, chat
  template on, Expanse bundle token sets, bf16, batch 2), run once inside the launch test (135 s). Stored at SP_REPO
  `results/job28/circuit.json`, revision `54a46ad0a8da46c8161835ba3696860624708754`, sha256 `10ca4b7e…4e83`, asserted by
  the full parts. Layer order starts 23, 26, 31, 29, 17; circuit sets k=1 [23], k=2 [23, 26]; random sets (seed 0)
  k=1 [3], k=2 [3, 7].
- Variants (GPTQ, symmetric int, group 128, 128 wikitext-2 texts @ `b08601e0`, protected layers kept 16-bit,
  equal k in both arms): 28a fp16, w8, w4; 28b fp16, w4p19c, w4p19r (~11.2 bit, est. 11.85 GB); 28c fp16, w3p2c,
  w3p2r (~3.9 bit), w3p1c, w3p1r (~3.5 bit).
- Launch test: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d3811460ffe1d5e8cae2 (`SP_TEST=1`, fp16 + w3p1c with 16
  calibration texts and 8 items per set, `--timeout 50m`, cap $1.50); COMPLETED, running 454 s; outputs
  `results/job28-test/smoke/`. It showed transformers unpacking the compressed weights at the first forward (w3p1c:
  5.17 GB after load, 16.16 GB after the first forward), so HF tokens/s and peak memory are dense-path numbers.
- 28a: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d87ad0582cfbbefccad0 (`--timeout 75m`, cap $2.25), created
  2026-11-24T18:33:30Z.
- 28b: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d8a6d0582cfbbefccafd (`--timeout 75m`, cap $2.25), created
  2026-11-24T18:34:14Z.
- 28c: https://huggingface.co/jobs/pxaLabs-lexsi/6b05d8cb1460ffe1d5e8d317 (`--timeout 120m`, cap $3.60), created
  2026-11-24T18:34:51Z.
- Outputs: SP_REPO `results/job28/<28a|28b|28c>/` (`variants.jsonl`, `pairs.json`, `job28_summary.json`), models at
  `results/job28/models/<variant>/`. E5 worst-case total including the test: $9.60.
- **Status check 2026-11-24 ~21:00 UTC (`hf jobs inspect`):** smoke `6b05d381` COMPLETED, 454 s ($0.23); 28a COMPLETED, 1597 s
  ($0.80); **28b and 28c ERROR** after 175 s and 176 s ($0.09 each): `RevisionNotFoundError` on
  `results/job28/circuit.json` because the launch command's later `-e SP_CIRCUIT_SHA=… -e SP_VARIANT_MIN=…` flags were
  swallowed into the `SP_CIRCUIT_REV` value (launch quoting, not the script). No GPU work was done; 28b and 28c must be
  relaunched (wave 2) with each `-e` as its own argument. 28a results not yet copied into git.
- **Relaunch (wave 2):** 28b2 `6b060042d0582cfbbefd1122` job28b2-e5-mixed11 (created 21:23:14 UTC, `SP_TIMEOUT_MIN=75`)
  and 28c2 `6b060067d0582cfbbefd1160` job28c2-e5-low (created 21:23:51 UTC, `SP_TIMEOUT_MIN=120`). `hf jobs inspect`
  shows `SP_CIRCUIT_REV`, `SP_CIRCUIT_SHA` and `SP_VARIANT_MIN` as separate keys, and both summaries record the clean
  revision `54a46ad0` and sha256 `10ca4b7e…`. Script file unchanged since `a8f7806` (no later commit touches it); the
  relaunch has no builder ledger entry, so its worst case is counted from `SP_TIMEOUT_MIN` (75 and 120 min, assumed equal
  to `--timeout`).
- **Status check 2026-11-24 ~23:00 UTC (steward-2):** 28b2 COMPLETED 1582 s ($0.79), 28c2 COMPLETED 2822 s ($1.41);
  `skipped_time_guard` [] and `errors` {} in all three part summaries. 28a's environment holds the same swallowed
  `SP_CIRCUIT_REV` value (its summary records it); 28a builds only uniform variants and does not read the circuit file,
  so its results are unaffected. E5 billed $3.23 for the four completed jobs (smoke, 28a, 28b2, 28c2) plus $0.18 for the two failed
  launches, $3.41 in all.
- **Results (ASR k/119; `cis_sep`; disk):** fp16 19, 1.0001 PASS, 16.06 GB; w8 17, 1.0005 PASS, 9.19 GB (< 10 GB);
  w4 19, 0.9446 PASS, 5.70 GB; w4p19c 18 / w4p19r 19 (11.18 bits, 11.85 GB < 12 GB; McNemar 6/7, p = 1.0);
  w3p2c 31 / w3p2r 32 (3.93 bits, 5.53 GB; 16/17, p = 1.0; `cis_sep` WARN in both); w3p1c 31 / w3p1r 37 (3.53 bits,
  5.17 GB; 10/16, p = 0.327; WARN in both). The prediction is **not shown** at any E5 level (the script's own
  `prediction` field agrees). E5 used plain wikitext-2 GPTQ calibration (CircuitKIT default) and did not collapse at
  4 bit (task 0.727 vs 0.770 fp16).
- **Collected and reported (2026-11-25):** `kaggle/executed/12_hf_jobs/files/job28_circuit.json` and `job28{a,b,c}_{summary.json,pairs.json,variants.jsonl}`
  (commit `a69c59a`; the 28b and 28c files are the outputs of the 28b2 and 28c2 relaunches). Report
  `docs/report/circuitquant.md` (E5 section): not shown at every level (p 1.0, 1.0, 0.327); w8 9.19 GB and w4p19
  11.85 GB meet the size tiers. The circuit variants w4p19c, w3p2c and w3p1c are in private, gated Hub repos.

## Job 29 · E3: Hindi, Spanish and French ASR on base, Alpaca fine-tune, repaired and the E1 frontier (2026-11-24 UTC, exploratory; COMPLETED)

- Purpose: E3 of the amendment `docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (commit `cd72fc6`). ASR on
  the 119 held-out test prompts in en / hi / es / fr, per model, with Wilson 95% intervals, the change from the same
  model's English ASR in the same job, and a paired exact McNemar (language vs English, same prompt). No alarm is
  evaluated (amendment E3).
- Translations: re-made in the job, because Job 20 kept them in memory only. Job 20's `translate` copied verbatim,
  `facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d`, fp32, greedy, line by line, batch 16;
  English input `data/eval_harm.jsonl` split test at SP_REPO `194c2f53` (asserted equal to Job 20's English hash).
  Job 20 stored only one sha256 per translated set (texts joined by newlines), so the check is a set match per
  language. Smoke result: **hi, es and fr all match Job 20's sha256 (match rate 1.0 per language) and all length
  statistics are equal**. Per-prompt sha256 of the translations are stored for later runs; the text never leaves the job.
- Generation and judge: notebook 17 / Job 25 ASR settings (`safetypulse.audit.generate`, greedy, 256 new tokens, fp16,
  batch 16, template default system prompt); Llama Guard 3 1B `acf7aafa` (pre-registered), prompt and answer in the
  target language. Job 20 used bf16, batch 32 and the 8B judge, so Job 20 rates are not mixed with these.
- Models: Tiny Aya Global `af89d219`; Alpaca fine-tune = base + `purvansh01/safetypulse-lora-alpaca@0ea31228`
  `alpaca-s1/checkpoint-125` merged (Job 26 preset); repaired `purvansh01/tiny-aya-global-safetypulse-repaired@06ee768d`;
  E1 frontier variants from SP_REPO `results/job25/models/<variant>` at `77c4650f`. Frontier rule (shared with the E1
  follow-up): Job 25 variants with task ≥ 0.10, fp16 excluded, non-dominated in (disk_bytes, general); 7 qualify (u3,
  r35, u4, r5, r6, r72, u8), the 5 with the highest general are kept: **u4, r5, r6, r72, u8**. Collapsed (task < 0.10):
  c3, r3, c1g, r1g. Not run: u3 and r35 (cut by the top-5 rule). Nothing was dropped for budget.
- Script `tools/hf_jobs/job29_e3_multilingual.py`, launched from its raw GitHub URL at code SHA
  `91b7ced98aca7e86f35d8400f52027ee1c18737c` (also `SP_CODE_SHA`); package and `sp_bench.py` at `106e9ee` as Job 25;
  torch 2.14.1, transformers 5.21.0.
- Smoke (`SP_TEST=1`, first 8 prompts per language, base / alpaca_s1_125 / repaired / u4, `--timeout 30m`):
  `6b06028e1460ffe1d5e92440` COMPLETED in 4m 18s (`results/job29-test/smoke/`). Output scan: no prompt, answer or
  translation text (no string ≥ 40 characters except hashes and labels, no non-ASCII string); English flags of base and
  u4 agree 8/8 with Job 25's.
- Full jobs (namespace `pxaLabs-lexsi`, `l40sx1`), RUNNING: `6b06075f1460ffe1d5e92b8c` job29-e3-core (base,
  alpaca_s1_125, repaired; `--timeout 40m`, created 21:53:35 UTC) and `6b060793d0582cfbbefd1c14` job29-e3-frontier
  (u4, r5, r6, r72, u8; `--timeout 70m`, created 21:54:27 UTC). Each env var checked as a separate key with
  `hf jobs inspect`.
- Cap (worst case, timeout × $1.80/h): smoke $0.90 + core $1.20 + frontier $2.10 = **$4.20** (builder cap $6).
- Outputs: private `purvansh01/safetypulse-jobs` `results/job29/<core|frontier>/{translations.json, rows.jsonl,
  summary.json}`: per (model, language) n, k, ASR, Wilson interval, flags, answer lengths and digest, prompt-set hash.
- **Status check 2026-11-24 ~23:00 UTC (`hf jobs inspect`, steward-2):** smoke COMPLETED 258 s ($0.13), core COMPLETED
  945 s ($0.47), frontier COMPLETED 2130 s ($1.07, after 1409 s waiting for a GPU); $1.67 of $4.20. Both full
  summaries: `skipped_time_guard` [] and `failed` []; translations match Job 20 (rate 1.0 for hi, es, fr); English flags
  and completion hashes equal Job 25's for base, u4, r5, r6, r72, u8.
- **ASR k/119 (en / hi / es / fr):** base 16/17/27/21; alpaca_s1_125 29/35/25/25; repaired 13/17/22/20; u4 18/28/27/24;
  r5 16/19/26/36; r6 11/20/23/25; r72 9/19/25/23; u8 15/16/24/21. Of the 24 language-vs-English McNemar tests only r6 fr,
  r5 fr, r72 es and r72 fr survive Holm (observer-builder-e3).
- **Collected and reported (2026-11-25):** `kaggle/executed/12_hf_jobs/files/job29_{core,frontier}_{rows.jsonl,summary.json,translations.json}`
  (commit `a69c59a`; the translation files hold sha256 hashes and length statistics only). Report
  `docs/report/multilingual_e3.md` (`tools/report_multilingual_e3.py`): the four Holm survivors are confirmed; compared
  with base in the same language, only r5 fr is higher (36 vs 21, p 0.0015, uncorrected); r6 and r72 survive because
  their English ASR is low. The judge is not validated in hi/es/fr and the translations are not human-checked.

## Job 30 · E1 frontier variants: IFEval and XQuAD en/hi/es on the Job 25 size/quality frontier (2026-11-24 UTC, exploratory; COMPLETED)

- Purpose: amendment `docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (commit `cd72fc6`), E1 bullet
  "Frontier variants ... also get IFEval and XQuAD". Same benchmark, settings and data pins as Job 26 (bf16, batch 32,
  chat template, no system instruction). No verdict or frozen threshold is touched.
- Frontier rule (deterministic): all Job 25 variants from `results/job25/{A,B,C,D}/rows.jsonl` (jobs revision
  `77c4650f`); collapsed variants dropped (task < 0.10: c3, r3, c1g, r1g); among the rest (fp16 excluded) a variant is
  kept if no other has `disk_bytes` <= and `general` >= with one strict. Frontier: u3, r35, u4, r5, r6, r72, u8 (7);
  capped at 5 by highest `general`: **u4** (2,524,069,488 B, general 0.526), **r5** (2,871,937,680 B, 0.548), **r6**
  (3,219,805,864 B, 0.552), **r72** (3,567,674,056 B, 0.564), **u8** (3,930,210,104 B, 0.570). Dropped by the cap: u3
  (0.422), r35 (0.470).
- Checkpoints: `purvansh01/safetypulse-jobs` (dataset) `results/job25/models/<variant>` at revision
  `77c4650fcf77e1c517fbd04002cc67e86f7102e1` (compressed-tensors pack-quantized, 4- or 8-bit group). Tokenizer forced
  to Tiny Aya Global `af89d219` (`--base-tokenizer`); chat-template sha256 `9d2fae0e…` equals Job 26 base.
  transformers decompresses compressed-tensors weights on load/first forward (peak 8.16 GiB in the smoke, as fp16), so
  benchmark minutes are not a quantized-speed claim.
- Script `tools/hf_jobs/job26_e2_benchmarks.py` at code SHA `ba121a76064aca2e0aea030f2d49dce835184bc7` (adds
  `--model-subfolders`, `--base-tokenizer`, a per-variant time guard; `bench()` and the presets unchanged from
  `5a45d233`). Anchor: Job 26 `results/job26/base.json` (no fp16 rerun inside Job 30; the cap did not allow it).
- Smoke (`--limit 5`, u4, `--timeout 15m`): `6b06038dd0582cfbbefd163b` COMPLETED in 94 s; bench 0.82 min vs 0.68 min
  for Job 26 base smoke; n = 5 per task; no answer text in the output (`results/job30-test/u4.json`).
- Full jobs (namespace `pxaLabs-lexsi`, `l40sx1`), RUNNING: `6b0604c8d0582cfbbefd17fa` job30-e1-frontier-A (u4, u8,
  r72; `--timeout 90m`), `6b0604fc1460ffe1d5e927a5` job30-e1-frontier-B (r5, r6; `--timeout 60m`). Time guard skips a
  variant if fewer than 22 min remain.
- Cap (worst case, timeout × $1.80/h): smoke $0.45 + A $2.70 + B $1.80 = **$4.95** (builder cap $5).
- Outputs: private `purvansh01/safetypulse-jobs` `results/job30/<variant>.json` (scores, Wilson 95% intervals, per
  document ids / scores / answer sha256 and length; no answer text).
- **Status check 2026-11-24 23:17 UTC (`hf jobs inspect`, steward-2):** smoke COMPLETED 94 s ($0.05); B COMPLETED
  3414 s ($1.71; r5, r6); A COMPLETED 5111 s ($2.56; u4, u8, r72; "done in 84.6 min; skipped by time guard: []"). $4.31
  of $4.95. Real time per variant was 27–29 min, so `--est-min 22` was too loose (use 31); r72 fitted with ~5 min left.
- **IFEval prompt-level strict (base 0.771, Job 26):** u4 0.715 (paired exact McNemar 67/37, p = 0.004), u8 0.758
  (20/13, p = 0.296), r5 0.726 (55/31, p = 0.013), r6 0.726 (55/31, p = 0.013), r72 0.738 (44/26, p = 0.041);
  exploratory, uncorrected, cross-job. XQuAD F1 within about 0.02 of base in each language; EM near 0 (answer-length
  caveat as Job 26). u4 has 98/500 empty Hindi XQuAD answers (base 41; r5 62, r6 42, u8 46, r72 18). Jobs revision
  with all five files: `a00444f3`.
- **Collected and reported (2026-11-25):** `kaggle/executed/12_hf_jobs/files/job30_{u4,u8,r5,r6,r72}.json` (jobs revision `a00444f3`, commit
  `a69c59a`). Report `docs/report/benchmarks.md`: after Holm, u4 is the only frontier variant whose IFEval
  prompt-strict drop survives (adjusted p 0.046 over 14 tests, 0.021 over the 5 frontier strict tests); r5, r6 and r72
  strict drops are nominal only, while the r6 and r72 loose drops survive the 14-test correction. All comparisons are
  cross-job (no base rerun in Job 30). u4's empty Hindi answers are not interpreted.

## Publication of the follow-up outputs (2026-11-23)

The adapters written by Jobs 18, 21, 24 (`pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca`) and Job 23
(`pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca`) were copied file for file to
[`purvansh01/safetypulse-h3fine-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-h3fine-lora-alpaca) (251
files, from org revision `01bb53a9`) and
[`purvansh01/safetypulse-expanse-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-expanse-lora-alpaca) (25
files, from `d3c84404`) by a CPU job (`6b0368dc130a00b5aa814c0d`; the script asserts equal file lists), made public
and added to the model collection. The org has no transfer right for the account's token, so the org copies stay
private. The Jobs 15–24 files of this folder and their reports are public in
[`purvansh01/safetypulse-followups`](https://huggingface.co/datasets/purvansh01/safetypulse-followups) (with a SHA-256
manifest) and in the data collection. Cards: `docs/hub_cards/stage2/safetypulse-{h3fine-lora-alpaca,expanse-lora-alpaca,followups}.md`.

## Publication of the E1–E5 outputs (2026-11-25, private-first; still PRIVATE)

Ten repos were created under `purvansh01`, all **private**; the owner makes them public after review. Cards and the
release plan (`docs/hub_cards/stage2/circuitquant_repos.json`, generator `tools/hub_cards_circuitquant.py`) came from
branch `wf3/c-cards` (`73db08e`, card fixes `4a79fbd`), merged into `stage2/lexsi-feedback` in `11b70d9`. Weights were copied inside Hugging Face from the private jobs
repo by two CPU jobs (namespace `pxaLabs-lexsi`, `cpu-upgrade`): `6b062b631460ffe1d5e96391` (smoke, W3A16-P1 only,
23 s) and `6b062bf61460ffe1d5e9649d` (the other 7 repos, 184 s); the script asserted that the byte
total of every copied folder equals the plan. A third CPU job, `6b063a301460ffe1d5e97bd4` (331 s), loaded the 4 Tiny
Aya repos with transformers 5.21.0 and checked that exactly the protected layers hold 16-bit weights. The three CPU
jobs together ran 538 s (`running_secs`, all COMPLETED) and cost about $0.0045 (running time × $0.0005/min; an estimate,
not a billed figure).

| Repo | Type | Source (private jobs repo) | Gated | Hub head |
|---|---|---|---|---|
| `purvansh01/tiny-aya-global-CircuitQuant-W4A16-P9` | model | Job 25 `c72` | no | `a5237221` |
| `purvansh01/tiny-aya-global-CircuitQuant-W4A16-P6` | model | Job 25 `c6` | no | `56297e0b` |
| `purvansh01/tiny-aya-global-CircuitQuant-W4A16-P3` | model | Job 25 `c5` | no | `3f238bfe` |
| `purvansh01/tiny-aya-global-CircuitQuant-W3A16-P1` | model | Job 25 `c35` | no | `ce28e022` |
| `purvansh01/aya-expanse-8b-CircuitQuant-W4A16-P19` | model | Job 28 `w4p19c` | auto | `07074b21` |
| `purvansh01/aya-expanse-8b-CircuitQuant-W3A16-P2` | model | Job 28 `w3p2c` | auto | `1297f88a` |
| `purvansh01/aya-expanse-8b-CircuitQuant-W3A16-P1` | model | Job 28 `w3p1c` | auto | `eea287fe` |
| `purvansh01/aya-expanse-8b-safetypulse-lora-alpaca-default` | model | Job 27 adapters, seeds 1 and 2, 31 checkpoints each | auto | `a6fff4b0` |
| `purvansh01/safetypulse-circuitquant-results` | dataset | the 46 `job25_*`–`job30_*` files of this folder | no | `9a2a6cb0` |
| `purvansh01/alpaca-curatorkit-rejected` | dataset | Job 3 CuratorKIT rejections (19 rows; text only for the 2 schema rejections) | no | `95998b5f` |

Visibility and heads read with `huggingface_hub` on 2026-11-25. All 62 adapter configs record the base revision
`5062468bf9bc0c6035fd64e06274333ec127d980`. The private collection "SafetyPulse CircuitQuant"
(`purvansh01/safetypulse-circuitquant-6b062c90ba6a961706bfc612`) holds all 10; the 8 models and 2 datasets were also
added to the public model and data collections. All ten were made public on 2026-11-26 for the Stage 2 submission (visibility, cards and collection
membership re-read with `huggingface_hub` and anonymously that day); the cards now link the reports instead of the
paper. Random and uniform
controls (`r*`, `u*`, `w8`, `w4`, `w*r`) and the collapsed variants are not published as models (amendment). Release
record: `docs/11_WITHHELD_ARTIFACTS.md` (E1–E5 release).

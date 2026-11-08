# 12 · Hugging Face Jobs: CircuitKIT cross-check and judge robustness

**Status:** ✅ both completed · **Date:** 2026-11-07 · **Account:** `purvansh01` (HF credits)
Inputs and results live in the private dataset repo `purvansh01/safetypulse-jobs` (scripts in `jobs/`).
Results downloaded here: `files/job1_crosscheck.json`, `files/job2_judges.json` (scores and statistics only; no
answer text leaves the jobs).

| Job | Hardware | Runtime | Link |
|---|---|---|---|
| 1 · CircuitKIT EAP-IG | A100 80 GB (`a100-large`, $2.50/h) | discovery 223 s (plus setup) | https://huggingface.co/jobs/purvansh01/6aef4ae8e6320c66fad5e879 |
| 2 · Llama Guard 1B vs 8B (AuditKIT) | A10G 24 GB (`a10g-large`, $1.50/h) | generation + two judges | https://huggingface.co/jobs/purvansh01/6aef4aeee6320c66fad5e882 |

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
Both jobs together used well under the caps (A100 ≤ 3 h, A10G ≤ 2 h); see the job pages for the exact billed
minutes.

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
`tools/hf_jobs/job5_cli_check.py` on an L4 (7 min 9 s): https://huggingface.co/jobs/purvansh01/6af040ade6320c66fad73866.
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
`tools/hf_jobs/job7_audit_callback.py` on an L4 (9 min 22 s): https://huggingface.co/jobs/purvansh01/6af04ef5e6320c66fad7428a
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

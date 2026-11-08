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

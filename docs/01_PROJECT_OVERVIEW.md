# 01 — Project Overview

| Field | Value |
|-------|-------|
| Project | SafetyPulse |
| Track | 2b — AI Safety and Correction |
| Problem statements covered | Repairing safety drift without retraining · Locating where safety lives · Circuit-level interventions |
| Toolkits used | CuratorKIT, CircuitKIT, SafeTune, AuditKIT |
| Document status | Draft v0.1 |

---

## 1. Problem statement

Organisations adapt open-weight instruction-tuned models to their domain with supervised
fine-tuning (often LoRA). Prior work shows that fine-tuning — even on benign data — can erode
safety alignment (Qi et al., 2023, *Fine-tuning Aligned Language Models Compromises Safety, Even
When Users Do Not Intend To*). In practice:

- Safety is evaluated once, at the end, if at all.
- Full red-team evaluation (hundreds–thousands of generations + a judge model) per checkpoint is
  too expensive to run routinely.
- When drift is found, the usual fix is to retrain or to mix in safety data and fine-tune again,
  which is costly and can reduce task performance.

**Gap:** there is no cheap, routine, mechanism-level signal that tells a practitioner *during
training* that safety is degrading, and no targeted way to repair it.

## 2. Target users

| User | Need | How SafetyPulse helps |
|------|------|-----------------------|
| ML engineer fine-tuning an open model | Know if a checkpoint is still safe without a full eval | `check` command / trainer callback returns CIS + pass/fail |
| Safety / governance reviewer | Evidence that a released checkpoint was monitored | Per-checkpoint CIS log + audit report |
| Researcher | Understand *where* safety broke | Component-level CIS breakdown |

## 3. Proposed solution (summary)

1. Locate the refusal mechanism in the base model with CircuitKIT, using **matched contrast
   pairs** (harmful prompt vs. minimally edited harmless twin) as clean/corrupt inputs.
2. Define a **Circuit Integrity Score (CIS)** computable from ~100 forward passes.
3. Track CIS across fine-tuning checkpoints; validate against ground-truth ASR from AuditKIT.
4. When CIS drops below a calibrated threshold, apply a **circuit-restricted repair** via SafeTune
   weight-space methods.
5. Re-audit: ASR, over-refusal, task metric, general capability.

Full detail: [Research Plan](02_RESEARCH_PLAN.md), [Architecture](03_SYSTEM_ARCHITECTURE.md).

## 4. Novelty claim (to be verified)

Individually, refusal directions, fine-tuning-induced drift and weight-space patching are known.
The claimed contributions are:

1. **C1 — Mechanism as a leading indicator.** Using circuit/direction integrity as a cheap,
   per-checkpoint *predictor* of behavioural safety, validated by correlation with ASR.
2. **C2 — Matched-pair contrasts for safety circuit discovery.** A corruption strategy that addresses
   the limitation noted in the CircuitKIT paper (corruptions may find nothing on
   instruction-tuned / safety prompts).
3. **C3 — Detect → targeted repair loop.** Repair restricted to the components the monitor flags,
   compared against untargeted and random-component controls.

> These claims must be checked against the literature before Day 3
> ([Research Plan §6](02_RESEARCH_PLAN.md#6-literature-check-must-complete-before-day-3)). If C1 is
> already published, the project pivots its emphasis to C2/C3.

**Outcome (2026-11-11):** checked in [docs/12_RELATED_WORK.md](12_RELATED_WORK.md). Drift monitoring, including
held-out alarm evaluation, is published (C1 is not new as a general idea); the emphasis is on the refusal-drift
setting, the comparison of signals as alarms, and the repair result (C3: circuit-restricted repair fails, a low-rank
whole-model restore works).

## 5. Success criteria

| ID | Criterion | Target (provisional) |
|----|-----------|----------------------|
| S1 | CIS predicts ASR across checkpoints | Spearman ρ ≤ −0.7 (higher CIS ↔ lower ASR), across ≥3 seeds |
| S2 | CIS is cheaper than the ground-truth eval | ≥10× fewer GPU-seconds than full AuditKIT ASR run |
| S3 | CIS beats a naive cheap baseline | Better rank correlation than a 50-prompt refusal spot-check at equal cost |
| S4 | Repair restores safety | ASR within 5 pts of base model |
| S5 | Repair preserves utility | Domain task metric within 2 pts of drifted model; general capability within 2 pts |
| S6 | Over-refusal stays controlled | Over-refusal rate within 5 pts of base model |
| S7 | Targeting matters | Circuit-restricted repair beats random-component repair of equal size |

Targets are provisional and will be fixed in the Evaluation Protocol before results are seen.

## 6. Scope

**In scope**

- 1–2 small open instruction-tuned models (1–3B parameters).
- LoRA and/or full fine-tuning on 1–2 benign domain datasets, plus one harmful fine-tune as a positive
  control for drift.
- English prompts.
- CLI + Python library + Hugging Face `Trainer` callback.

**Out of scope (non-goals)**

- Models > 8B parameters.
- Multilingual safety (possible future extension).
- Defence against adversaries who deliberately fine-tune to remove safety (tamper resistance).
- A hosted web service.
- Guaranteeing safety: CIS is a monitoring signal, not a certificate.

## 7. Assumptions

| ID | Assumption | Risk if false | Check |
|----|------------|---------------|-------|
| A1 | Refusal in target models is mediated by a localisable direction/circuit | CIS has no signal | Day 3–5 ablation test |
| A2 | Benign fine-tuning causes measurable drift in our setup | Nothing to detect | Include harmful fine-tune positive control |
| A3 | Toolkits support our chosen model architecture | Integration delays | Day 1–2 quickstarts |
| A4 | Judge model (e.g. Llama Guard) is accurate enough | Noisy ASR | Manual audit of a sample |

## 8. Key references (verify before citing in the final report)

- Qi et al., 2023. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To.*
- Arditi et al., 2024. *Refusal in Language Models Is Mediated by a Single Direction.*
- Wei et al., 2024. *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.*
- Röttger et al., 2024. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs.*
- Mazeika et al., 2024. *HarmBench.*
- CircuitKIT paper — arXiv:2607.19317.
- Lexsi-affiliated paper on interpretability as governance infrastructure — arXiv:2509.08592.

# SafetyPulse

**Circuit-level early warning and targeted repair for safety drift during LLM fine-tuning.**

> Status: **Design phase** — no code yet. This repository currently contains the design and
> production documentation that the implementation must follow.
>
> Event: Lexsi Labs AI Safety & Alignment Hackathon — Track 2b (AI Safety and Correction)
> Window: 1–21 November 2026

---

## The problem

Fine-tuning an aligned open model on *benign* domain data (customer support, legal, medical, etc.)
can silently degrade its safety behaviour (Qi et al., 2023). Teams rarely notice, because running a
full red-team evaluation on every training checkpoint is slow and expensive — so it is skipped.

## What SafetyPulse does

1. **Calibrate** — locate the refusal circuit / direction in the *base* model (CircuitKIT), using
   matched harmful/harmless contrast pairs.
2. **Check** — at each fine-tuning checkpoint, compute a cheap **Circuit Integrity Score (CIS)**
   (~100 forward passes) that measures whether the safety mechanism is still intact.
3. **Validate** — show CIS predicts real attack success rate (ASR) measured with AuditKIT.
4. **Repair** — when CIS crosses a threshold, apply a targeted weight-space fix restricted to the
   identified components (SafeTune), preserving task performance.
5. **Audit** — report ASR, over-refusal and task/general capability before vs. after.

```
CuratorKIT ──► fine-tune ──► [checkpoint] ──► SafetyPulse.check (CIS) ──► pass ──► continue
 (data)                            │                       │
                                   │                       └─ fail ──► SafetyPulse.repair (SafeTune)
                                   │                                          │
                                   └──────────── AuditKIT (ground truth ASR / over-refusal) ◄─┘
```

## Documentation

| # | Document | Purpose |
|---|----------|---------|
| 00 | [Hackathon Brief](docs/00_HACKATHON_BRIEF.md) | Track 2 toolkits and problem statements (verbatim), mapping to this project |
| 01 | [Project Overview](docs/01_PROJECT_OVERVIEW.md) | Problem, users, novelty claim, scope, non-goals |
| 02 | [Research Plan](docs/02_RESEARCH_PLAN.md) | Hypotheses, experiments, baselines, controls, literature check |
| 03 | [System Architecture](docs/03_SYSTEM_ARCHITECTURE.md) | Components, data flow, interfaces, module layout |
| 04 | [Data Specification](docs/04_DATA_SPEC.md) | Datasets, splits, labels, leakage controls, handling of harmful content |
| 05 | [Evaluation Protocol](docs/05_EVALUATION_PROTOCOL.md) | Metrics, judges, statistics, reporting format |
| 06 | [Engineering Standards](docs/06_ENGINEERING_STANDARDS.md) | Repo layout, tooling, config, testing, reproducibility |
| 07 | [Production & Operations](docs/07_PRODUCTION_AND_OPERATIONS.md) | CLI/CI integration, deployment, runbook, compute |
| 08 | [Risk, Ethics & Licensing](docs/08_RISK_ETHICS_LICENSING.md) | Safety of artifacts, disclosure, licences |
| 09 | [Project Plan](docs/09_PROJECT_PLAN.md) | 16-day timeline, roles, milestones, definition of done |
| — | [Architecture Decision Records](docs/adr/) | Key decisions and their rationale |
| — | [Contributing](CONTRIBUTING.md) | Workflow for the team |

## Open items before implementation

- [ ] Literature check for prior work on circuit/direction-based drift monitoring (see [Research Plan §6](docs/02_RESEARCH_PLAN.md#6-literature-check-must-complete-before-day-3))
- [ ] Confirm official hackathon rules and judging criteria (toolkit-chaining requirement was seen only on secondary sources)
- [ ] Read CuratorKIT / SafeTune / CircuitKIT / AuditKIT docs and record actual APIs in [Architecture §5](docs/03_SYSTEM_ARCHITECTURE.md#5-external-toolkit-integration)
- [ ] Confirm available compute and fill in [Production §6](docs/07_PRODUCTION_AND_OPERATIONS.md#6-compute-budget)
- [ ] Choose licence for this repository (see [ADR-0004](docs/adr/0004-licensing.md))

## Team

| Name | Role | Contact |
|------|------|---------|
| _TBD_ | _TBD_ | _TBD_ |

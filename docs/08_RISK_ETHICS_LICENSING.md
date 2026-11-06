# 08 — Risk, Ethics & Licensing

Status: Draft v0.1

---

## 1. Dual-use considerations

SafetyPulse locates the components that implement refusal. The same information could help
someone *remove* safety (e.g. directional ablation / "abliteration"). Mitigations:

- Publish methods and aggregate results; do not publish ready-to-use bundles that make it trivial to
  strip safety from a specific model, unless the organisers request them for evaluation.
- Do not release the positive-control (D-harm) fine-tuned checkpoints.
- The report discusses this dual-use risk explicitly.

## 2. Harmful content handling

| Item | Rule |
|------|------|
| Harmful prompts | From public research benchmarks; team rewrites only minimally for pairs |
| Harmful completions | Stored locally in gitignored `outputs/`; never committed; never sent to external trackers |
| Reports / demo | Redacted completions or labels only |
| Manual labelling | Labellers can skip items; categories involving self-harm flagged in advance |

## 3. Project risks

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|----|------|-----------|--------|-----------|-------|
| R1 | Prior work already does C1 | Medium | Medium | Literature check by Day 3; pivot emphasis to C2/C3 | _TBD_ |
| R2 | Toolkits don't support chosen model | Medium | High | Day 1–2 quickstarts; pick model that all four support | _TBD_ |
| R3 | No measurable drift from benign fine-tuning | Medium | High | Harmful positive control; higher LR / more epochs condition | _TBD_ |
| R4 | CIS does not correlate with ASR | Medium | Medium | Report as negative result; still deliver repair study | _TBD_ |
| R5 | Insufficient compute | Medium | High | LoRA, 1B model, fewer audited checkpoints | _TBD_ |
| R6 | Judge model unreliable | Low–Med | High | Human validation sample | _TBD_ |
| R7 | Schedule slip | High | High | Milestone gates in Project Plan; cut stretch goals first | _TBD_ |

## 4. Licensing

| Component | Licence | Implication |
|-----------|---------|-------------|
| CircuitKIT | Lexsi Labs Source Available License (LSAL v1.1), per its paper/repo — **confirm** | Research/evaluation/education use; no commercial exploitation |
| CuratorKIT, SafeTune, AuditKIT | _TBD — confirm from repos_ | |
| Base models | Model-specific (e.g. Llama community licence, Qwen licence) | Check redistribution terms for any released weights |
| Benchmarks | Per dataset — record in `data/SOURCES.md` | Some prohibit redistribution |
| This repository | _TBD_ ([ADR-0004](adr/0004-licensing.md)) | Must be compatible with toolkit licences |

## 5. Academic integrity

- All external code, data and ideas are cited.
- Claims in the report are backed by result files in the tagged commit.
- AI-assistant use in writing code/docs is disclosed if hackathon rules require it.

# 09 — Project Plan

Status: Draft v0.1 · Today: 5 Nov 2026 · Submission: 21 Nov 2026 (confirm exact time and timezone)

---

## 1. Roles (assign names)

| Role | Responsibilities | Owner |
|------|-----------------|-------|
| Tech lead / interpretability | Calibration, CIS, CircuitKIT adapter | _TBD_ |
| Training & repair | Fine-tuning runs, checkpoints, repair methods, SafeTune adapter | _TBD_ |
| Data & evaluation | Datasets, CuratorKIT, AuditKIT, judge validation, statistics | _TBD_ |
| Report & demo | Literature check, figures, report, demo video, README | _TBD_ |

## 2. Timeline

| Day | Date | Work | Gate / deliverable |
|-----|------|------|-------------------|
| 1 | 5 Nov | Read official rules; read all four toolkit docs; set up repo + env | Toolkit API table filled ([Arch §5](03_SYSTEM_ARCHITECTURE.md#5-external-toolkit-integration)) |
| 2 | 6 Nov | Quickstarts on candidate models; choose model + domain; compute check | ADR-0002 accepted; compute table filled |
| 3 | 7 Nov | Literature check done; build contrast pairs; activation collection code | **G1: go/no-go on novelty framing** |
| 4 | 8 Nov | Refusal direction + ablation; start CircuitKIT discovery | Direction ablation works |
| 5 | 9 Nov | Circuit discovery + random-ablation control; CIS implementation | **G2: Table A done — mechanism localised** |
| 6 | 10 Nov | Fine-tuning pipeline + checkpointing; datasets via CuratorKIT | Fine-tune runs end-to-end |
| 7 | 11 Nov | Dev runs (seed 0) all conditions; audit pipeline | Drift observed in ≥1 condition |
| 8 | 12 Nov | Choose CIS variant, τ on dev; judge validation | Judge κ reported |
| 9 | 13 Nov | **Freeze evaluation protocol v1.0**; launch test runs (seeds 1–2) | **G3: protocol frozen** |
| 10 | 14 Nov | Test-run monitoring results; Table B, cost Table D | Correlation plot |
| 11 | 15 Nov | Repair methods R1–R6 implemented | Repair runs on dev |
| 12 | 16 Nov | Repair on test runs; random control | **G4: Table C done** |
| 13 | 17 Nov | Failure analysis; stretch (2nd model/domain) only if ahead | |
| 14 | 18 Nov | Report draft; figures; CLI polish | Draft report |
| 15 | 19 Nov | Reproducibility run on clean env; demo video | Clean-env reproduction |
| 16 | 20 Nov | Final review; tag `v1.0-submission` | |
| — | 21 Nov | Submit (buffer day) | **Submitted** |

## 3. Gate fallbacks

| Gate | If it fails | Fallback |
|------|------------|----------|
| G1 | C1 already published | Reframe as replication + emphasise matched-pair discovery (C2) and targeted repair (C3) |
| G2 | Circuit discovery yields nothing useful | Use refusal direction only for CIS; report CircuitKIT finding as limitation evidence |
| G3 | No drift in benign conditions | Rely on harmful-control + stronger benign condition (higher LR/epochs); state clearly |
| G4 | Targeted repair no better than random | Report honestly; present steering/SafeTune baseline results; analyse why |

**Cut order if behind schedule:** stretch generalisation → ASR-JB → R2 → second benign condition.
Never cut: controls (random ablation, random repair), over-refusal, held-out evaluation.

## 4. Definition of done (submission)

- [ ] Tables A–D filled from tagged commit; every number traceable to an output file
- [ ] Figures: CIS-vs-ASR, detection timeline, repair comparison
- [ ] Report: problem, method, results, failure analysis, limitations, dual-use discussion
- [ ] `safetypulse calibrate | check | repair | audit` work from a clean install
- [ ] Integration test passes in CI
- [ ] Demo (≤3 min) showing check → FAIL → repair → audit
- [ ] All toolkit usage documented (which kit did what)
- [ ] Licences and data sources listed

## 5. Meeting cadence

- Daily 15-minute stand-up: yesterday / today / blockers / gate status.
- Decisions recorded as ADRs in `docs/adr/`.

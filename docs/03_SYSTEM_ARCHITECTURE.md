# 03 — System Architecture

Status: Draft v0.1

---

## 1. Design goals

| Goal | Implication |
|------|-------------|
| Cheap per-checkpoint check | Forward passes only, batched, no generation in `check` |
| Toolkit-agnostic core | CircuitKIT/SafeTune/AuditKIT behind adapter interfaces so a toolkit API change touches one file |
| Reproducible | Every run is driven by a versioned config; every artifact carries config hash, git SHA, seeds |
| CI-friendly | `check` returns machine-readable JSON and a non-zero exit code on failure |
| Safe by default | Harmful prompts never logged in plaintext to third-party trackers; completions redacted in reports |

## 2. Component overview

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │                         safetypulse                          │
                 │                                                              │
 base model ───► │  calibrate ──► ProbeBundle (direction, components, τ, meta)  │
                 │                         │                                    │
 checkpoint ───► │  check  ◄───────────────┘ ──► CheckResult (CIS, pass/fail)   │
                 │                                                              │
 checkpoint ───► │  repair ◄── ProbeBundle  ──► repaired checkpoint + RepairLog │
                 │                                                              │
 any model  ───► │  audit  ──► AuditReport (ASR, over-refusal, task, general)   │
                 └───────┬───────────────┬───────────────┬──────────────┬───────┘
                         │               │               │              │
                   CuratorAdapter  CircuitAdapter  SafeTuneAdapter  AuditAdapter
                         │               │               │              │
                    CuratorKIT       CircuitKIT       SafeTune        AuditKIT
```

## 3. Components

### 3.1 `data` — dataset building and loading

- Builds contrast pairs, probe set, eval sets, fine-tuning sets (via CuratorAdapter where applicable).
- Enforces split disjointness and deduplication ([Data Spec](04_DATA_SPEC.md)).
- Output: versioned dataset files + manifest (`manifest.json` with hashes and counts).

### 3.2 `calibrate` — locate the safety mechanism

Input: base model, contrast-pair set, config.
Steps:
1. Collect activations (residual stream, final prompt token, all layers).
2. Compute refusal directions `r_ℓ`; select ℓ* by ablation effect on validation split.
3. Run circuit discovery via CircuitAdapter (clean = harmful, corrupt = harmless twin).
4. Validate: targeted vs. random ablation.
5. Compute base-model reference statistics for CIS normalisation.

Output: **ProbeBundle**.

### 3.3 `check` — compute Circuit Integrity Score

Input: checkpoint, ProbeBundle, probe set.
Output: **CheckResult**. Cost target: < 2 minutes on a single consumer GPU for a 1.5B model.

### 3.4 `repair` — targeted weight-space repair

Input: drifted checkpoint, base model, ProbeBundle, method (R1–R6), α.
Output: repaired weights (safetensors / PEFT adapter) + **RepairLog**.

### 3.5 `audit` — ground-truth evaluation

Wraps AuditKIT. Generates completions on held-out sets, judges them, computes metrics per
[Evaluation Protocol](05_EVALUATION_PROTOCOL.md). Output: **AuditReport**.

### 3.6 `integrations`

- `SafetyPulseCallback` for Hugging Face `transformers.Trainer`: runs `check` on save events,
  logs CIS, optionally stops training or triggers repair.
- CLI (`safetypulse ...`), see [Production §2](07_PRODUCTION_AND_OPERATIONS.md#2-interfaces).

## 4. Data contracts

All artifacts are JSON (metadata) + safetensors (tensors). Schemas validated with pydantic.

### 4.1 ProbeBundle

```jsonc
{
  "schema_version": "1.0",
  "base_model": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "<hf commit sha>"},
  "layer": 14,                              // ℓ*
  "direction_file": "direction.safetensors",// r_ℓ* (unit-normalised) + per-layer directions
  "components": [                           // from circuit discovery
    {"type": "attn_head", "layer": 12, "index": 3, "score": 0.41},
    {"type": "mlp", "layer": 14, "score": 0.33}
  ],
  "reference": {"sep_base": 7.82, "circ_logit_diff_base": 3.10},
  "threshold": {"metric": "cis_sep", "tau": 0.80},
  "probe_set": {"path": "data/probe/v1.jsonl", "sha256": "..."},
  "provenance": {"git_sha": "...", "config_hash": "...", "seed": 0, "created_at": "ISO-8601"}
}
```

### 4.2 CheckResult

```jsonc
{
  "schema_version": "1.0",
  "checkpoint": {"path": "...", "step": 1200},
  "probe_bundle_sha256": "...",
  "scores": {"cis_sep": 0.71, "cis_cos": 0.88, "cis_circ": 0.65},
  "primary_metric": "cis_sep",
  "tau": 0.80,
  "status": "FAIL",                  // PASS | WARN | FAIL
  "per_component": [{"id": "attn_head:12:3", "integrity": 0.52}],
  "cost": {"gpu_seconds": 47.3, "n_prompts": 100},
  "provenance": {"git_sha": "...", "seed": 0}
}
```

### 4.3 AuditReport and RepairLog

Defined in code (`safetypulse/schemas.py`); must include metric values with confidence
intervals, judge model + version, dataset hashes, and provenance block.

## 5. External toolkit integration

> **To be completed Day 1–2 after reading official docs.** Do not code against assumed APIs.

| Toolkit | What we need from it | Actual API / entry point | Supported models | Notes / licence |
|---------|---------------------|--------------------------|------------------|-----------------|
| CuratorKIT | Generate/clean domain fine-tuning data; dedupe; format for trainer | _TBD_ | _TBD_ | _TBD_ |
| CircuitKIT | Circuit discovery with custom clean/corrupt pairs + custom metric; component ablation | _TBD_ | _TBD_ | LSAL v1.1 (research use) — confirm |
| SafeTune | Weight-space patching / merging / projection; steering baseline | _TBD_ | _TBD_ | _TBD_ |
| AuditKIT | Harmfulness + over-refusal benchmarks, judge, metrics | _TBD_ | _TBD_ | _TBD_ |

Adapters live in `safetypulse/adapters/` and expose only the methods listed in the table, so the
core never imports a toolkit directly.

**Fallbacks** if a toolkit cannot do what is needed in time:

- CircuitKIT discovery fails on our model → use refusal direction + attribution patching
  implemented directly with PyTorch hooks; still report the attempt.
- SafeTune lacks restricted repair → implement R1/R2 ourselves (simple tensor ops), keep SafeTune
  methods as R3 baseline.

## 6. Module layout (planned)

```
safetypulse/
  __init__.py
  config.py            # pydantic config models, YAML loading
  schemas.py           # ProbeBundle, CheckResult, AuditReport, RepairLog
  data/                # builders, loaders, dedupe, split checks
  activations.py       # hooked forward passes, batching
  calibrate.py
  cis.py               # CIS variants
  repair.py            # R1, R2, R4, R5 + SafeTune wrapper (R3), steering (R6)
  audit.py
  adapters/
    curatorkit.py
    circuitkit.py
    safetune.py
    auditkit.py
  integrations/
    hf_callback.py
  cli.py
configs/               # YAML experiment configs
experiments/           # run scripts (one per experiment in Research Plan)
tests/
docs/
```

## 7. Non-functional requirements

| Area | Requirement |
|------|-------------|
| Performance | `check` ≤ 2 min / checkpoint for 1.5B model on a 16–24 GB GPU |
| Memory | Runs in bf16 on 16 GB GPU for ≤3B models |
| Determinism | Fixed seeds; `torch.use_deterministic_algorithms` where feasible; document remaining nondeterminism |
| Portability | Linux + CUDA primary; macOS (MPS/CPU) for unit tests only |
| Observability | Structured JSON logs; optional W&B/MLflow with harmful text redacted |

# 07 — Production & Operations

Status: Draft v0.1. Describes how SafetyPulse is used outside of our experiments — as a tool a team
plugs into its fine-tuning pipeline.

---

## 1. Deployment model

SafetyPulse ships as a **Python package + CLI**, not a hosted service. It runs wherever the user's
training runs (local GPU box, cloud VM, cluster job).

| Mode | Use |
|------|-----|
| Trainer callback | Automatic check at each checkpoint save during training |
| CLI in CI/CD | Gate before a checkpoint is promoted / released |
| Batch | Score a directory of existing checkpoints retroactively |

## 2. Interfaces

### 2.1 CLI

```bash
# 1. One-time per base model: locate mechanism and write a probe bundle
safetypulse calibrate --model Qwen/Qwen2.5-1.5B-Instruct --config configs/base.yaml \
  --out bundles/qwen2.5-1.5b/

# 2. Per checkpoint: cheap integrity check
safetypulse check --checkpoint runs/ft/checkpoint-1200 --bundle bundles/qwen2.5-1.5b/ \
  --out outputs/check-1200.json
#   exit 0 = PASS, 10 = WARN, 20 = FAIL, 1 = error

# 3. Targeted repair
safetypulse repair --checkpoint runs/ft/checkpoint-1200 --bundle bundles/qwen2.5-1.5b/ \
  --method circuit-restore --alpha auto --out runs/ft/checkpoint-1200-repaired/

# 4. Ground-truth audit
safetypulse audit --model runs/ft/checkpoint-1200-repaired/ --suite default \
  --out outputs/audit-1200-repaired.json
```

### 2.2 Python

```python
from safetypulse import ProbeBundle, check
from safetypulse.integrations import SafetyPulseCallback

bundle = ProbeBundle.load("bundles/qwen2.5-1.5b/")
trainer = Trainer(..., callbacks=[SafetyPulseCallback(bundle, on_fail="stop")])  # "log" | "stop" | "repair"
```

### 2.3 Status semantics

| Status | Condition | Recommended action |
|--------|-----------|-------------------|
| PASS | CIS ≥ τ_warn | Continue |
| WARN | τ_fail ≤ CIS < τ_warn | Run full `audit` on this checkpoint |
| FAIL | CIS < τ_fail | Do not release; run `repair` then `audit` |

CIS is a **screening signal**. A PASS does not certify safety; release decisions require an `audit`.

## 3. Example CI integration

```yaml
# .github/workflows/model-gate.yml (illustrative — runs on a self-hosted GPU runner)
jobs:
  safety-gate:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --frozen
      - run: safetypulse check --checkpoint "$CKPT" --bundle bundles/prod/ --out check.json
      - run: safetypulse audit --model "$CKPT" --suite release --out audit.json
        if: always()
      - uses: actions/upload-artifact@v4
        with: {name: safety-reports, path: "*.json"}
```

## 4. Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` / `model.revision` | — | HF id + pinned commit |
| `model.dtype` | `bfloat16` | |
| `probe.path` | bundled | Probe set JSONL |
| `cis.metric` | set by calibration | Primary CIS variant |
| `cis.tau_warn`, `cis.tau_fail` | set by calibration | Thresholds |
| `repair.method` | `circuit-restore` | `circuit-restore`, `circuit-project`, `safetune:<name>`, `interpolate`, `steer` |
| `repair.alpha` | `auto` | Fixed float or grid search on validation |
| `audit.suite` | `default` | Named set of eval datasets |
| `audit.judge` | per ADR | Judge model id + revision |
| `logging.tracker` | `none` | `none` / `wandb` / `mlflow` (harmful text always redacted) |

## 5. Runbook

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `check` FAIL on step-0 checkpoint | Bundle built for different model/revision or chat template | Verify `base_model.revision`; re-calibrate |
| CIS noisy between adjacent checkpoints | Probe set too small, nondeterministic kernels | Increase probe set; enable deterministic mode; average |
| CIS PASS but audit shows high ASR | Monitor miss — drift outside located mechanism | Record as failure case; re-calibrate with broader components; release blocked by audit regardless |
| Repair raises over-refusal | α too high | Lower α / use `alpha: auto` with over-refusal penalty |
| OOM during calibrate | Activations for all layers cached | Reduce batch size; stream layers; use `layers:` subset |
| Toolkit import error | Version mismatch | Use pinned lockfile; check adapter version matrix |

## 6. Compute budget

> Fill in Day 1.

| Resource | Available | Notes |
|----------|-----------|-------|
| GPU type / VRAM | _TBD_ | |
| GPU hours available | _TBD_ | |
| Storage | _TBD_ | Checkpoints: ~3 GB each for 1.5B full FT; LoRA adapters much smaller |

Estimated need (1.5B model, LoRA): per run ~1–2 GPU-h fine-tune + 15 checkpoints × (CIS ~2 min +
full audit ~10–20 min). With 4 conditions × 3 seeds this is roughly 50–90 GPU-hours. Reduce via
fewer checkpoints for full audit, LoRA only, or smaller eval sets on dev runs.

## 7. Security and operations hygiene

- Credentials only via environment variables; `.env` gitignored; pre-commit secret scan.
- Probe bundles and eval sets containing harmful prompts are not published publicly
  without review ([Risk & Ethics](08_RISK_ETHICS_LICENSING.md)).
- Model weights downloaded from pinned revisions only.

## 8. Versioning and release

- Semantic versioning for the package. `schema_version` in every artifact; readers reject
  unknown major versions.
- Release checklist: tests green, CHANGELOG updated, reproducibility run on a clean machine,
  report numbers regenerated from tagged commit.

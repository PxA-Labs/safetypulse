# 06 — Engineering Standards

Status: Draft v0.1

---

## 1. Stack

| Concern | Choice | Notes |
|---------|--------|-------|
| Language | Python 3.11 | Confirm toolkit compatibility Day 1 |
| Env / deps | `uv` with `pyproject.toml` + lockfile | Pin exact versions |
| DL | PyTorch, `transformers`, `peft`, `accelerate`, `safetensors` | Versions pinned to what toolkits require |
| Config | YAML + pydantic models | One config per experiment in `configs/` |
| CLI | `typer` | |
| Lint / format | `ruff` (lint + format) | |
| Types | `mypy` (strict on `safetypulse/`, relaxed on `experiments/`) | |
| Tests | `pytest` | |
| Hooks | `pre-commit` (ruff, mypy, large-file check, secret scan) | |
| Tracking | Local JSON + optional W&B/MLflow (redacted) | |

## 2. Repository layout

```
.
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── uv.lock
├── .pre-commit-config.yaml
├── .gitignore               # outputs/, checkpoints/, .env, data/raw/
├── configs/
│   ├── base.yaml
│   └── experiments/         # e1_calibration.yaml, e2_monitoring_*.yaml, e3_repair.yaml
├── safetypulse/             # library (see Architecture §6)
├── experiments/             # thin scripts: load config → call library → write outputs
├── data/                    # manifests + small files; large data via DVC/HF
├── tests/
│   ├── unit/
│   ├── integration/         # tiny model (e.g. a ~100M model) end-to-end
│   └── test_data_leakage.py
├── outputs/                 # gitignored: run artifacts
├── reports/                 # final report, figures (redacted)
└── docs/
```

## 3. Coding conventions

- Library code in `safetypulse/` is importable, typed, and free of hard-coded paths.
- Experiment scripts contain no logic beyond wiring; logic belongs in the library.
- Functions that touch model internals document tensor shapes in docstrings, e.g.
  `acts: Float[Tensor, "batch layer d_model"]`.
- No toolkit imports outside `safetypulse/adapters/`.
- No secrets in code or config. Tokens (HF, W&B) come from environment variables loaded from an
  untracked `.env`.

## 4. Configuration

- Every run is fully specified by one YAML file (+ CLI overrides that are recorded).
- At run start, the resolved config is written to `outputs/<run_id>/config.resolved.yaml` and hashed.
- `run_id = <experiment>-<YYYYMMDD-HHMMSS>-<short git sha>`.

## 5. Reproducibility checklist (every run)

- [ ] Git working tree clean (or diff saved to run dir)
- [ ] Seeds set for `random`, `numpy`, `torch` (+ CUDA)
- [ ] Model revision pinned (HF commit SHA, not just name)
- [ ] Dataset manifest hashes recorded
- [ ] Library versions recorded (`uv pip freeze` → run dir)
- [ ] Hardware recorded (GPU model, driver, CUDA)

## 6. Testing strategy

| Level | What | When |
|-------|------|------|
| Unit | CIS math on synthetic tensors; schema validation; repair tensor ops (e.g. α=0 → unchanged, α=1 → base) | Every commit |
| Data | Leakage, schema, category balance | Every commit |
| Integration | calibrate → check → repair → audit on a tiny model with 10 prompts | Every PR |
| Regression | Fixed small checkpoint → CIS within tolerance of stored value | Every PR |
| Experiment | Full runs | Manual, logged |

Target: all unit + data + integration tests pass before merge.

## 7. Git workflow

- `main` is always runnable. Work on short-lived branches `feat/…`, `fix/…`, `exp/…`.
- Pull requests require one teammate review; checklist in [CONTRIBUTING](../CONTRIBUTING.md).
- Conventional commit messages (`feat:`, `fix:`, `exp:`, `docs:`, `test:`).
- Tag the commit used for final results: `v1.0-submission`.

## 8. CI (GitHub Actions)

- `lint` — ruff + mypy
- `test` — unit + data tests on CPU
- `integration` — tiny-model end-to-end on CPU (≤10 min)
- GPU experiments are not run in CI.

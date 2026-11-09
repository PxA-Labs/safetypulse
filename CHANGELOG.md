# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CODEOWNERS for the three maintainers.
- CI on Python 3.10, 3.11 and 3.12 with a build and clean-install check; CodeQL, dependency review, pip-audit,
  secret scanning, OpenSSF Scorecard, link checking and Dependabot.
- Release-policy check (`tools/check_release_policy.py`) that blocks model weights, withheld harmful data and
  oversized files in pull requests.

### Changed
- `pyproject.toml` declares the licence as an SPDX expression (`Apache-2.0`), as current setuptools requires.

## [0.1.0] - 2026-11-08

First public release, for the Lexsi Labs AI Safety & Alignment Hackathon (Track 2b).

### Added
- `safetypulse` package: `calibrate` (refusal direction and circuit from matched pairs), `check` (Circuit Integrity
  Score with PASS / WARN / FAIL), `repair` (circuit restore, circuit projection, SafeTune `wise_ft` and safety-vector
  restore, whole-model interpolation, random-component control, steering), `audit` (ASR, over-refusal, GSM8K, MMLU
  with bootstrap intervals), the `safetypulse` CLI and a Hugging Face `Trainer` callback.
- Adapters for the four Lexsi Labs toolkits: CuratorKIT, CircuitKIT, SafeTune and AuditKIT.
- Kaggle notebooks (00–11, 14) and Hugging Face Job scripts (1–7) for every experiment, with executed runs and summaries.
- Public Hugging Face artifacts: the SafeTune-repaired Tiny Aya Global, LoRA adapters of every published run (positive controls withheld), the probe bundle,
  evaluation, fine-tuning and pair datasets, and monitoring and audit logs.
- Report, Hugging Face article and launch video.

### Verified
- Every documented interface run on the real model (HF Jobs 5–7); `calibrate` reproduces the published probe bundle
  exactly; 167 unit and parity tests in CI.

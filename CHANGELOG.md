# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- `calibrate` / `resolve_token_sets`: a model-keyed `token_sets.json` whose only entry is for another model name is
  no longer used silently; a warning names the expected and the found key. Token IDs outside the tokenizer's
  vocabulary (including negative IDs, which torch indexing would wrap round to a real token) raise a `ValueError`
  before any model work (#113).

### Added
- Probe bundles record the tokens behind the refuse/comply IDs in `provenance.token_strings`, so a token set from
  another tokenizer is visible in the bundle (#113).
- Stage 2 (research code, not part of the package): pre-registration (`docs/13_PREREGISTRATION.md`), Hugging Face
  Job scripts 8–14 (`tools/hf_jobs/`, documented in its README), analysis scripts (`tools/h1_analysis.py` …
  `tools/review_followups.py`) writing `docs/report/`, and the paper (`paper/`). Results: H1 and H2 confirmed, H3
  and H6 refuted (`docs/13_PREREGISTRATION.md`, `docs/report/`).

## [0.1.1] - 2026-11-09

First release on PyPI: `pip install safetypulse`.

### Added
- Release workflow: GitHub release → TestPyPI → install check → PyPI, via Trusted Publishing with build
  attestations (no stored tokens).
- PyPI metadata: project URLs, keywords, classifiers.
- CODEOWNERS for the three maintainers.
- CI on Python 3.10, 3.11 and 3.12 with a build and clean-install check; CodeQL, dependency review, pip-audit,
  secret scanning, OpenSSF Scorecard, link checking and Dependabot.
- Release-policy check (`tools/check_release_policy.py`) that blocks model weights, withheld harmful data and
  oversized files in pull requests.

### Fixed
- `audit` / `load_suite`: a split that matches no items is an error (exit 1, listing the splits present) instead of
  an empty report with exit 0; `audit()` raises when no metric is computed (#110).
- Probe bundles reject zero or non-finite reference values; `check` returns ERROR (never PASS) when the primary
  metric's reference is <= 0, and `set_thresholds` refuses such a metric (#111).
- A `cis_circ` threshold on a bundle without circuit components is rejected at load and by `set_thresholds`,
  instead of a `KeyError` in `check` (#112).

### Changed
- README links and images are absolute, so they work on PyPI; installation is `pip install safetypulse`.
- `pyproject.toml` declares the licence as an SPDX expression (`Apache-2.0`), as current setuptools requires.
- Builds require `setuptools>=84.1`, which fixes PYSEC-2026-3447 and GHSA-xmj3-9gf7-h52w.

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

# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Research code and records only; the `safetypulse` package is unchanged. Everything below is exploratory under the
amendment `docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md` (`cd72fc6`); no pre-registered verdict (H1, H2
confirmed; H3, H6 refuted) and no frozen threshold changes.

### Added
- HF Job scripts for the Lexsi-call follow-ups: `tools/hf_jobs/sp_bench.py`, `job25_e1_quant_sweep.py` (E1),
  `job26_e2_benchmarks.py` (E2; also used for Job 30), `job27_e4_expanse_default.py` (E4), `job28_e5_expanse_quant.py`
  (E5), `job29_e3_multilingual.py` (E3). Records: `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` (Jobs 25–30).
- Result files of Jobs 25–30 (flags, scores, hashes and lengths only) in `kaggle/executed/12_hf_jobs/files/`.
- Reports and their generators, with tests that pin the headline numbers: `docs/report/circuitquant.md`
  (`tools/report_circuitquant.py`, E1 and E5), `benchmarks.md` (`tools/report_benchmarks.py`, E2 and Job 30),
  `multilingual_e3.md` (`tools/report_multilingual_e3.py`, E3), `expanse_default.md`
  (`tools/report_expanse_default.py`, E4), and the paper figures `paper/figures/fig_quant_{asr,size,speed}.pdf`,
  `expanse_default.pdf`.
- Workflow records: `docs/report/workflow/2026-11-24-run-log.md` (budget ledger, departures) and
  `docs/report/workflow/2026-11-24-workflow-record.md`.

### Changed
- `tools/check_release_policy.py` allowlists 14 reviewed Jobs 25–30 result files whose only long strings are fixed
  method notes.
- Docs: README, `docs/03`, `docs/05`, `docs/10`, `docs/11` and the report index describe the E1–E5 results and the
  private-first release (ten `purvansh01` repos, all private until the owner makes them public).

## [0.1.2] - 2026-11-24

None of these fixes changes a published result: the frozen runs used left-padded tokenizers, fp32 bundles and the
recorded settings.

### Fixed
- `encode` (and so CIS scoring, attribution, audit generation and the Llama Guard judge) always pads on the left,
  even with a caller-supplied right-padded tokenizer; before, a right-padded tokenizer gave the base model a
  `cis_sep` of 0.905 instead of 1.0.
- Provenance `git_sha` runs git in the package directory and requires a clean exit, instead of recording the
  commit of whatever repository the caller was in.
- `repair` rejects component IDs outside the model before editing any weight; `select_strength` raises a clear
  `ValueError` when no candidate has a finite objective.
- `audit` scores in eval mode with gradients off and restores the caller's train state.
- `SafetyPulseCallback` under DDP: every rank scores (so all ranks stop at the same step) and only the main process
  writes `safetypulse_check.json` / `safetypulse_checks.jsonl`.
- Component mean-ablation on Gemma-2/3 hooks `post_feedforward_layernorm` (the post-norm residual write), so
  layer selection ablates the full MLP output; stale means are rejected and per-head ablation warns.
- `calibrate` rejects token IDs that are in both the refuse and comply sets; `calibrate` and `check` warn when a
  chat template's `enable_thinking` switch (Qwen3) is not recorded in the bundle. Scores are unchanged.
- `ProbeBundle.load` casts bundle tensors to fp32 and rejects non-float tensors (bf16/fp16 files crashed
  `compute_cis`), and rejects a schema version from another major version.
- The CLI and the training callback report the bundle's metric problem, not "not finite", when ERROR comes from a
  non-positive reference.
- `calibrate` / `resolve_token_sets`: a model-keyed `token_sets.json` whose only entry is for another model name is
  no longer used silently; a warning names the expected and the found key. Token IDs outside the tokenizer's
  vocabulary (including negative IDs, which torch indexing would wrap round to a real token) raise a `ValueError`
  before any model work (#113).

### Added
- Probe bundles record the tokens behind the refuse/comply IDs in `provenance.token_strings`, so a token set from
  another tokenizer is visible in the bundle (#113).
- Stage 2 (research code, not part of the package): pre-registration (`docs/13_PREREGISTRATION.md`), Hugging Face
  Job scripts 8–24 (`tools/hf_jobs/`, documented in its README; Jobs 15–24 are exploratory follow-ups), analysis scripts (`tools/h1_analysis.py` …
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

# CI audit, round 3 (lane 7)

Report only. No code, tests or paper text were changed. Run on 2026-11-23 from a fresh worktree on branch
`wf/r3-lane7`, created with `git checkout -B wf/r3-lane7 origin/stage2/wf-integration` after `git fetch origin`.

## 1. Head under test

```
$ git rev-parse HEAD
9d99ae4a76bcff8a6fd009f74a34a47106dfa7d3
```

(`9d99ae4 Ledger: round 2`, the tip of `origin/stage2/wf-integration` at fetch time; 140 commits ahead of `origin/main`.)

Tooling: `uv 0.13.7 (Homebrew 2026-11-03 aarch64-apple-darwin)`, CPython 3.12.16 (macOS arm64).

## 2. Pytest

Cache used: **the shared uv cache** (the default). The private `UV_CACHE_DIR` fallback was not needed: the lock was not
contended and no torch download stalled this time.

The mandated command was run with two outer wrappers only for timing and a 15-minute guard:
`time UV_LOCK_TIMEOUT=1200 timeout 900 uv run --extra dev --extra peft python -m pytest -q`.
It created a fresh `.venv` (46 packages installed in 454 ms, all from cache) and finished in **26.1 s wall time**
(pytest itself: 11.00 s). Exit code 0.

Last 15 lines, verbatim:

```
........................................................................ [ 83%]
...........................ss....ss..........x.............sss.......... [ 95%]
..............................                                           [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_circuitkit_adapter.py:121: circuitkit / transformer-lens not installed
SKIPPED [1] tests/test_report_numbers.py:408: unsourced: per-model timings are not in any docs/report file
SKIPPED [1] tests/test_report_numbers.py:408: unsourced: the job 9 vs job 13 cis_sep difference is not stored in docs/report (only specificity_control.md prose, if anywhere)
SKIPPED [1] tests/test_report_numbers_r1.py:102: could not import 'pandas': No module named 'pandas'
SKIPPED [1] tests/test_report_numbers_r1.py:114: could not import 'pandas': No module named 'pandas'
SKIPPED [1] tests/test_safetune_adapter.py:51: could not import 'safetune': No module named 'safetune'
SKIPPED [1] tests/test_safetune_adapter.py:63: could not import 'safetune': No module named 'safetune'
SKIPPED [1] tests/test_safetune_adapter.py:73: could not import 'safetune': No module named 'safetune'
XFAIL tests/test_report_numbers_r1.py::test_l15_all_but_one_false_alarm_interval - MISMATCH: l15-threshold-transfer.md cites Wilson [0.02, 0.35] for the all-but-one 1/12 false alarms; tools/h6_threshold_transfer.py:wilson(1, 12) = [0.0149, 0.3539], i.e. [0.01, 0.35] at two decimals (0.02 comes from rounding the stored 3-decimal 0.015 a second time)
597 passed, 8 skipped, 1 xfailed in 11.00s
```

Result: **597 passed, 8 skipped, 1 xfailed, 0 failed.**

### Notes on skips and the xfail

- `tests/test_report_numbers_r1.py:102` (`test_two_stage_alarm_recomputed`) and `:114` (`test_label_noise_recomputed`)
  skip because `pandas` is not in the `dev` or `peft` extras. They are therefore skipped in the mandated local command
  **and** in GitHub CI (which installs `.[dev,peft]`). I ran them once with pandas added, without changing anything:
  `uv run --extra dev --extra peft --with pandas python -m pytest -q tests/test_report_numbers_r1.py -k "two_stage_alarm or label_noise"`
  gave `2 passed, 25 deselected in 1.63s`. So the numbers they guard currently reproduce, but nothing enforces it.
  Proposed fix (not applied): add `pandas` to the `dev` extra in `pyproject.toml`, or add `--with pandas` to the
  documented test command and `pip install pandas` to `.github/workflows/ci.yml`.
- The strict xfail `test_l15_all_but_one_false_alarm_interval` documents a real double-rounding error in
  `docs/report/workflow/paper_proposals/l15-threshold-transfer.md` line 37 (last touched by `fb99d81`, lane 15):
  Wilson for 1/12 is [0.0149, 0.3539], which is **[0.01, 0.35]** at two decimals, not [0.02, 0.35]. Proposed fix (not
  applied): change `[0.02, 0.35]` to `[0.01, 0.35]` in that proposal line and remove the `xfail` marker in the same commit
  (the marker is `strict=True`, so the test would then fail as XPASS until the marker is removed). This is a proposal
  file, not `paper/main.tex`; whoever carries the sentence into the paper must use [0.01, 0.35].
- The two `unsourced` skips at `tests/test_report_numbers.py:408` and the `safetune` / `circuitkit` skips are by design
  (optional toolkits; numbers not stored in any report file).

## 3. Ruff

`uv run --extra dev ruff check safetypulse tests` (exit 0), tail verbatim:

```
All checks passed!
```

`uv run --extra dev ruff check tools --exclude tools/hf_jobs` (exit 1), tail verbatim:

```
E731 Do not assign a `lambda` expression, use a `def`
   --> tools/h3_analysis.py:133:5
    |
131 |     (OUT / "h3_lead.json").write_text(json.dumps(res, indent=1) + "\n")
132 |
133 |     fmt = lambda x: "–" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
134 |     L = ["# H3 · Lead time at 10-step checkpoint spacing (Tiny Aya Global)", "",
135 |          "Generated by `tools/h3_analysis.py` from the H3 monitor logs (notebook 08) and the ‖ΔW‖ file (notebook 19). "
    |
help: Rewrite `fmt` as a `def`
    |
132 |
    -     fmt = lambda x: "–" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
133 +     def fmt(x):
134 +         return "–" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
135 |     L = ["# H3 · Lead time at 10-step checkpoint spacing (Tiny Aya Global)", "",
    |
note: This is an unsafe fix and may change runtime behavior

Found 9 errors.
7 unsafe fixes available with `--fix --unsafe-fixes`.
```

All 9 errors, as expected (6 in `tools/h1_analysis.py`, last touched by `ef0e494`; 3 in `tools/h3_analysis.py`, last
touched by `c2d0dea`); lane 8 fixes these this round:

| Code | Location | Message |
|------|----------|---------|
| B007 | tools/h1_analysis.py:74:13 | Loop control variable `c` not used within loop body |
| E731 | tools/h1_analysis.py:106:13 | Do not assign a `lambda` expression, use a `def` |
| E731 | tools/h1_analysis.py:115:5 | Do not assign a `lambda` expression, use a `def` |
| E731 | tools/h1_analysis.py:126:5 | Do not assign a `lambda` expression, use a `def` |
| E501 | tools/h1_analysis.py:130:131 | Line too long (132 > 130) |
| E731 | tools/h1_analysis.py:137:13 | Do not assign a `lambda` expression, use a `def` |
| B905 | tools/h3_analysis.py:62:52 | `zip()` without an explicit `strict=` parameter |
| E702 | tools/h3_analysis.py:101:49 | Multiple statements on one line (semicolon) |
| E731 | tools/h3_analysis.py:133:5 | Do not assign a `lambda` expression, use a `def` |

These do not fail GitHub CI: the CI Ruff step checks only `safetypulse tests tools/check_release_policy.py`
(`.github/workflows/ci.yml`).

## 4. Build and twine check

```
$ uv build --out-dir <scratchpad>/dist
Successfully built <scratchpad>/dist/safetypulse-0.1.1.tar.gz
Successfully built <scratchpad>/dist/safetypulse-0.1.1-py3-none-any.whl
$ uv run --no-project --with twine twine check <scratchpad>/dist/*
Checking <scratchpad>/dist/safetypulse-0.1.1-py3-none-any.whl: PASSED
Checking <scratchpad>/dist/safetypulse-0.1.1.tar.gz: PASSED
```

`twine check --strict` (what CI runs) also PASSED for both files. Package version **0.1.1** (`pyproject.toml` line 7 and
`safetypulse/__init__.py` `__version__` agree). Wheel 75,865 bytes, sdist 128,910 bytes. Built into the session
scratchpad, outside the repo; nothing uploaded.

## 5. GitHub Actions

`gh run list --repo PxA-Labs/safetypulse --limit 20`: all 20 most recent runs are `completed success` (CI, CodeQL,
Security, OpenSSF Scorecard, Links) on `main` pushes up to "Paper: cite amendment 3h by its commit on main (15a248e…)"
(2026-11-22T16:54:57Z) and on the PR branches `stage2/cite-15a248e`, `stage2/review-followups`, `stage2/cite-8e3d75b`.

`gh run list --repo PxA-Labs/safetypulse --branch stage2/wf-integration`: **no runs.** The CI workflow triggers only on
`push` to `main`, `pull_request` and `workflow_dispatch` (`.github/workflows/ci.yml` lines 3–7), and no PR has been
opened from the integration branch, so **the integration head 9d99ae4 has never been tested by GitHub CI**. The local
run above is the only evidence for it; it covers Python 3.12 only, while CI also tests 3.10 and 3.11.

Non-successful runs among the last 100:

| Run | Workflow / job / step | Branch | Conclusion | Root cause |
|-----|----------------------|--------|-----------|-----------|
| 44884049901 | Links / check links / Run lychee | stage2/h6-prereg (PR, 2026-11-21) | failure | Two relative links in `docs/hub_cards/stage2/safetypulse-h6-safetune.md` pointed to `NOTICE` and `LICENSE-LSAL-1.2.md`, which exist on the Hub repo but not next to the card in git ("File not found", lines 169 and 172). **Already fixed**: the card now links to `https://huggingface.co/datasets/purvansh01/safetypulse-h6-safetune/blob/main/NOTICE` and `.../LICENSE-LSAL-1.2.md` (lines 184, 187), and all later Links runs passed. |
| 44934107247 | CI / all four jobs | stage2/audit-fixes (PR #158, 2026-11-21) | cancelled | Not a failure: concurrency group `ci-${{ github.ref }}` with `cancel-in-progress: true` cancelled it when a newer push to the same PR arrived ("Canceling since a higher priority waiting request for ci-refs/pull/158/merge exists"). |

No open CI failure exists. Proposed (not applied): trigger CI on the integration head before it is merged, e.g.
`gh workflow run CI --repo PxA-Labs/safetypulse --ref stage2/wf-integration` (needs the owner's go-ahead), so the
3.10/3.11 matrix and the clean-venv wheel install are exercised.

## 6. Test failures

None. The only non-pass outcomes are the 8 skips and 1 strict xfail described in section 2, with proposed fixes there.

## 7. Untracked files created by the run

`git status --short` after all commands shows exactly one untracked file: `uv.lock` (1,643,944 bytes, created by
`uv run`). It is not in `.gitignore` and is **not committed** (only this report is). Ignored, uncommitted by-products:
`.venv/`, `.pytest_cache/`, `.ruff_cache/`, `safetypulse.egg-info/`, `__pycache__/` directories. Build outputs went to the
scratchpad, not the repo.

# Contributing to SafetyPulse

Thanks for your interest. Bug reports, reproductions on other models, and improvements to the package are all welcome.
By taking part you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Set up

```bash
git clone https://github.com/PxA-Labs/safetypulse
cd safetypulse
pip install -e ".[dev,peft]"        # Python >= 3.10; add ",toolkits" for CircuitKIT / SafeTune / AuditKIT (Python 3.12)
pytest -q                           # tiny local models, no GPU or Hub access needed
ruff check safetypulse tests
```

CI runs these two checks on Python 3.10, 3.11 and 3.12, and also builds the package and installs the wheel in a
clean environment (`.github/workflows/ci.yml`). Every pull request additionally runs:

| Check | Workflow | Fails when |
|---|---|---|
| CodeQL | `codeql.yml` | A security or quality alert is found in Python code or workflows |
| Dependency review | `security.yml` | A new dependency has a high-severity advisory or a copyleft licence |
| pip-audit | `security.yml` | An installed dependency has a known vulnerability |
| Secret scan (TruffleHog) | `security.yml` | A credential is committed |
| Release policy | `security.yml` | Weights, withheld harmful data or files over 5 MB are added (`tools/check_release_policy.py`) |
| Links | `links.yml` | A Markdown link is broken |

A maintainer who has reviewed a release-policy false positive can apply the `release-policy-reviewed` label.

## Where things live

| Path | What |
|---|---|
| `safetypulse/` | The package: `calibrate`, `check`, `repair`, `audit`, the CLI and the Trainer callback |
| `safetypulse/adapters/` | The only place the Lexsi toolkits are imported |
| `tests/` | Unit and parity tests on tiny Llama / Qwen2 models |
| `tools/build_notebooks.py` | Generates every notebook in `kaggle/notebooks/`; **edit notebooks here, not in the .ipynb files** |
| `kaggle/executed/` | Executed runs, each with a `RUN_SUMMARY.md` |
| `tools/hf_jobs/` | Scripts for runs on Hugging Face Jobs |
| `docs/` | Research plan, protocol, data spec, decisions (`docs/adr/`), report and article |

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Create a branch (`feat/…`, `fix/…`, `docs/…`), keep the pull request to one concern, and add or update tests.
3. Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
4. Fill in the pull request template; CI must pass before merging.
5. `main` is protected by a ruleset: a pull request needs one approving review from a code owner
   ([.github/CODEOWNERS](.github/CODEOWNERS)), all review conversations resolved, and the required checks green
   (tests on Python 3.10–3.12, build and install, secret scan, release policy). Force-pushes and deleting `main` are
   blocked. Maintainers may bypass the review requirement for their own changes.

**Research changes** (metrics, thresholds, datasets, evaluation protocol) need a short rationale in the pull request
and, after the protocol freeze, an entry in the deviation log of [docs/02_RESEARCH_PLAN.md](docs/02_RESEARCH_PLAN.md).
Thresholds and repair strengths are chosen on development runs only; never tune them on test runs.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml`, `__version__` in `safetypulse/__init__.py` and `version` in `CITATION.cff`,
   and move the `Unreleased` entries in `CHANGELOG.md` under the new version; merge that pull request.
2. Publish a GitHub release whose tag is `v<version>` (e.g. `v0.1.1`).
3. `.github/workflows/release.yml` checks that the tag and versions agree, builds once, publishes to TestPyPI,
   installs from there, and then publishes the same files to PyPI through Trusted Publishing; no token is stored.

A version on PyPI can never be re-uploaded, only yanked; fix forward with a new patch release.

## Harmful content

This project handles harmful prompts and studies models whose safety is degraded. In pull requests, issues and
commits:

- never include model answers to harmful prompts, harmful training data, or adapters trained on harmful data;
- refer to harmful prompts by benchmark id where possible;
- report any such content you find in public as described in [SECURITY.md](SECURITY.md).

The release policy is in [docs/11_WITHHELD_ARTIFACTS.md](docs/11_WITHHELD_ARTIFACTS.md).

## Licence

Contributions are accepted under the [Apache-2.0](LICENSE) licence of this repository.

# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately through this repository's
**Security** tab → **Report a vulnerability** (GitHub private vulnerability reporting). Only the maintainers can see
the report.

Please include what you found, how to reproduce it, and its impact. We aim to acknowledge a report within 5 days and
to share a plan for a fix within 14 days.

## What is in scope

| Area | Examples |
|---|---|
| The `safetypulse` package | Code execution through a crafted probe bundle, checkpoint or config; unsafe deserialisation; path traversal when writing outputs |
| Published artifacts | A file in a public Hugging Face repo that should have been withheld (see below), or that contains secrets or personal data |
| Safety claims | A way to make `safetypulse check` report PASS for a checkpoint whose safety is clearly degraded, beyond the limitations stated in the report |

Reports about the safety *behaviour* of the released models (for example a jailbreak of the repaired Tiny Aya) are
welcome too; they help us document limitations. Please share them privately, not as public prompts.

## Harmful content we deliberately withhold

SafetyPulse studies safety drift, so some of its artifacts are harmful by design: positive-control adapters trained on
harmful data, and model answers judged harmful. These are **not** released publicly. Why, and how judges and reviewers
can request access, is described in [docs/11_WITHHELD_ARTIFACTS.md](docs/11_WITHHELD_ARTIFACTS.md). If you find such
content anywhere public in this project, please report it as above.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (`main`) | yes |

## Secrets

Never commit tokens or credentials. The code reads the Hugging Face token from the environment (`HF_TOKEN`) or a
Kaggle secret; it is never written to files, logs or notebooks. If you spot a leaked secret, report it privately.

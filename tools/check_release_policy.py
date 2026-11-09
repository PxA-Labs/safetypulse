"""Fail if a change adds files that the release policy (docs/11_WITHHELD_ARTIFACTS.md) keeps out of the repository.

Usage: python tools/check_release_policy.py [BASE_REF]   (default origin/main)

Checks every file added, copied, modified or renamed relative to BASE_REF:
  - model weights (positive-control adapters must never be public; no checkpoint belongs in git);
  - withheld data: the ft_harm training set, unredacted audits, raw completions, the judge-validation sample;
  - any file larger than MAX_BYTES (data and models live on Kaggle / the Hugging Face Hub).
Exit code 1 lists each offending path and the rule it broke. A maintainer who has reviewed a false positive can
apply the `release-policy-reviewed` label to the pull request, which skips the CI job.
"""
from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 5 * 1024 * 1024

RULES: dict[str, tuple[str, ...]] = {
    "model weights": ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "adapter_model.*"),
    "withheld harmful data": ("ft_harm*", "*judge_validation*", "audit.jsonl", "*completions*.jsonl",
                              "*completions*.csv", "*unredacted*"),
}


def violations(paths: list[str], root: Path = Path(".")) -> list[tuple[str, str]]:
    """(path, reason) for every path that breaks a rule; deleted files are not passed in."""
    out = []
    for p in paths:
        name = PurePosixPath(p).name
        for reason, patterns in RULES.items():
            if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                out.append((p, reason))
                break
        else:
            f = root / p
            if f.is_file() and f.stat().st_size > MAX_BYTES:
                out.append((p, f"larger than {MAX_BYTES // (1024 * 1024)} MB"))
    return out


def changed_files(base: str) -> list[str]:
    cmd = ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"]
    return [line for line in subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines() if line]


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else "origin/main"
    paths = changed_files(base)
    bad = violations(paths)
    for p, reason in bad:
        print(f"::error file={p}::release policy: {reason} must not be committed (docs/11_WITHHELD_ARTIFACTS.md)")
    print(f"checked {len(paths)} changed file(s) against {base}: {len(bad)} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

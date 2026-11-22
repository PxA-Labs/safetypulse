"""Fail if the repository holds files that the release policy (docs/11_WITHHELD_ARTIFACTS.md) keeps out of it.

Usage: python tools/check_release_policy.py [BASE_REF] [--all-history]   (default BASE_REF origin/main)

Three scans run by default:
  1. changed files: every file added, copied, modified or renamed relative to BASE_REF;
  2. tracked tree: every file `git ls-files` lists, so a withheld file committed before this check existed is caught;
  3. branch history: every path added by any commit in BASE_REF..HEAD, so a file added and deleted again inside a
     pull request (still present in its history) is caught.
`--all-history` also checks every path ever added on any ref (`git log --all`), for audits.

Rules:
  - model weights (positive-control adapters must never be public; no checkpoint belongs in git);
  - harm / harmdense adapter checkpoints (any file under a harm-sN/harmdense-sN checkpoint directory);
  - withheld data: the ft_harm training set, unredacted audits, raw completions, the judge-validation sample;
  - answer text: a JSON / JSONL / CSV / TSV field named like a model answer (answer, response, completion, output,
    generation, generated_text, with model_/raw_ prefixes and plurals) holding non-empty text other than the
    redaction placeholder. Answers are regenerated in memory and only verdicts and scores are stored;
  - any file larger than MAX_BYTES (data and models live on Kaggle / the Hugging Face Hub).
Content rules look at field names only and never print field values. Exit code 1 lists each offending path, the
rule it broke and, for answer text, the field and line. A maintainer who has reviewed a false positive can apply
the `release-policy-reviewed` label to the pull request, which skips the CI job, or list the path in ALLOWED.
"""
from __future__ import annotations

import csv
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 5 * 1024 * 1024

RULES: dict[str, tuple[str, ...]] = {
    "model weights": ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "adapter_model.*"),
    "withheld harmful data": ("ft_harm*", "ft-harm*", "*judge_validation*", "audit.jsonl", "*completions*.jsonl",
                              "*completions*.csv", "*unredacted*"),
}

# A directory named after a positive-control run (harm-s0, harmdense-s0, earth-harm-s1, ...) holding checkpoints.
HARM_RUN_DIR = re.compile(r"^(.*[-_])?harm(dense)?-s\d+$", re.IGNORECASE)
CHECKPOINT_DIR = re.compile(r"^checkpoint-\d+$", re.IGNORECASE)

ANSWER_KEY = re.compile(r"^(model_|raw_|generated_|final_)?(answer|response|completion|output|generation)s?$"
                        r"|^generated_text$", re.IGNORECASE)
PLACEHOLDER = "[withheld: judged harmful]"
CONTENT_SUFFIXES = (".json", ".jsonl", ".csv", ".tsv")

# Paths a maintainer has reviewed and accepted (exact repository-relative paths).
ALLOWED: frozenset[str] = frozenset()


def _path_reason(p: str) -> str | None:
    pp = PurePosixPath(p)
    name = pp.name
    names = {name, name.split("__", 1)[-1]}   # run outputs are saved as <tag>__<file> (e.g. final__audit.jsonl)
    for reason, patterns in RULES.items():
        if any(fnmatch.fnmatch(n, pat) for n in names for pat in patterns):
            return reason
    dirs = pp.parts[:-1]
    if any(HARM_RUN_DIR.match(d) for d in dirs) and any(CHECKPOINT_DIR.match(d) for d in dirs):
        return "harm adapter checkpoint"
    return None


def violations(paths: list[str], root: Path = Path(".")) -> list[tuple[str, str]]:
    """(path, reason) for every path that breaks a name or size rule; deleted files are not passed in."""
    out = []
    for p in paths:
        if p in ALLOWED:
            continue
        reason = _path_reason(p)
        if reason:
            out.append((p, reason))
            continue
        f = root / p
        if f.is_file() and f.stat().st_size > MAX_BYTES:
            out.append((p, f"larger than {MAX_BYTES // (1024 * 1024)} MB"))
    return out


def name_violations(paths: list[str]) -> list[tuple[str, str]]:
    """Name rules only, for paths that may no longer exist on disk (history scans)."""
    return [(p, r) for p in paths if p not in ALLOWED and (r := _path_reason(p))]


def _holds_text(v: object) -> bool:
    if isinstance(v, str):
        return bool(v.strip()) and v.strip() != PLACEHOLDER
    if isinstance(v, list):
        return any(_holds_text(x) for x in v)
    return False


def _answer_fields(obj: object, prefix: str = "") -> list[str]:
    """Dotted names of answer-like fields that hold text, anywhere in a parsed JSON value."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            if ANSWER_KEY.match(str(k)) and _holds_text(v):
                found.append(name)
            found.extend(_answer_fields(v, name))
    elif isinstance(obj, list):
        for x in obj:
            found.extend(_answer_fields(x, prefix + "[]"))
    return found


def answer_text_fields(path: Path) -> list[tuple[str, int]]:
    """(field, line) for every answer-like field holding text in a JSON/JSONL/CSV/TSV file; values are not returned.

    Unparseable files are skipped (they are not data this rule can judge)."""
    hits: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for f in _answer_fields(row):
                hits.setdefault(f, i)
    elif suffix == ".json":
        try:
            for f in _answer_fields(json.loads(text)):
                hits.setdefault(f, 1)
        except json.JSONDecodeError:
            return []
    elif suffix in (".csv", ".tsv"):
        rows = csv.reader(text.splitlines(), delimiter="\t" if suffix == ".tsv" else ",")
        header = next(rows, [])
        cols = [(i, h) for i, h in enumerate(header) if ANSWER_KEY.match(h.strip())]
        for n, row in enumerate(rows, 2):
            for i, h in cols:
                if i < len(row) and _holds_text(row[i]):
                    hits.setdefault(h.strip(), n)
    return sorted(hits.items(), key=lambda kv: (kv[1], kv[0]))


def content_violations(paths: list[str], root: Path = Path(".")) -> list[tuple[str, str]]:
    out = []
    for p in paths:
        f = root / p
        if p in ALLOWED or not p.lower().endswith(CONTENT_SUFFIXES) or not f.is_file() or f.stat().st_size > MAX_BYTES:
            continue
        for field, line in answer_text_fields(f):
            out.append((p, f"answer text in field '{field}' (line {line})"))
    return out


def _git(*args: str) -> list[str]:
    res = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return [line for line in res.stdout.splitlines() if line]


def changed_files(base: str) -> list[str]:
    return _git("diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD")


def tracked_files() -> list[str]:
    return _git("ls-files")


def added_in_history(rev_range: str | None) -> list[str]:
    """Every path added by a commit in rev_range (None = all refs), whether or not it still exists."""
    spec = ["--all"] if rev_range is None else [rev_range]
    return sorted(set(_git("log", *spec, "--diff-filter=A", "--no-renames", "--name-only", "--format=")))


def check(paths: list[str], root: Path = Path(".")) -> list[tuple[str, str]]:
    return violations(paths, root) + content_violations(paths, root)


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    all_history = "--all-history" in argv[1:]
    base = args[0] if args else "origin/main"
    scans = [
        (f"changed file(s) against {base}", changed_files(base), check),
        ("tracked file(s)", tracked_files(), check),
        (f"path(s) added in {base}..HEAD", added_in_history(f"{base}..HEAD"), name_violations),
    ]
    if all_history:
        scans.append(("path(s) ever added on any ref", added_in_history(None), name_violations))
    seen: set[tuple[str, str]] = set()
    for label, paths, rule in scans:
        bad = rule(paths)
        for p, reason in bad:
            if (p, reason) not in seen:
                seen.add((p, reason))
                print(f"::error file={p}::release policy: {reason} must not be committed (docs/11_WITHHELD_ARTIFACTS.md)")
        print(f"checked {len(paths)} {label}: {len(bad)} violation(s)")
    return 1 if seen else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

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
  - HF job outputs from Job 15 on (kaggle/executed/12_hf_jobs/files/jobN*, N >= FIRST_STRICT_JOB, including future
    jobs): jobs keep answers in memory and store flags and scores only, so (a) any JSON / JSONL / CSV / TSV string
    (value, cell or key) longer than LONG_TEXT characters fails unless it is a hex hash, a path, a URL or a model id,
    and (b) any field named answer, response, completion, generation, output_text, prompt_text or translation (any
    prefix such as model_ / raw_, plurals) fails when it holds text other than the redaction placeholder;
  - executed notebooks (*.ipynb): a stream, display_data or execute_result output line of free prose (more than
    LONG_TEXT characters, at least PROSE_MIN_WORDS words, mostly letters) fails unless it starts like a log message
    (LOG_LINE: warnings, notes, errors, the pipeline's own status lines);
  - any file larger than MAX_BYTES (data and models live on Kaggle / the Hugging Face Hub).
Content rules report field names, cell numbers and line numbers and never print field values or output text.
Exit code 1 lists each offending path, the rule it broke and, for content rules, the field or cell and the line.
A maintainer who has reviewed a false positive can apply
the `release-policy-reviewed` label to the pull request, which skips the CI job, or list the path in ALLOWED.
"""
from __future__ import annotations

import csv
import fnmatch
import json
import re
import subprocess
import sys
import unicodedata
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

# HF job outputs copied into git (Job 15 onward, and every later job) get the strict text rules.
JOB_FILE = re.compile(r"(?:^|/)12_hf_jobs/files/job(\d+)(?:[^0-9]|$)")
FIRST_STRICT_JOB = 15
LONG_TEXT = 80
JOB_TEXT_KEY = re.compile(r"^([a-z0-9]+_)*(answer|response|completion|generation|output_text|prompt_text|translation)s?$",
                          re.IGNORECASE)
HEX = re.compile(r"^(0x)?[0-9a-fA-F]+$")
# One token, no whitespace: a path (a/b, C:\x, file.ext), a URL or a model id (org/name@rev).
PATH_LIKE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*://)?[\w.~+=@:%#?&,\-]*[/\\][\w.~+=@:%#?&,/\\\-]*$")

# Notebook outputs: a prose line has at least PROSE_MIN_WORDS words and PROSE_MIN_RATIO of its characters are
# letters (any script), marks, spaces or sentence punctuation. Log lines are allowed by their prefix.
PROSE_MIN_WORDS = 12
PROSE_MIN_RATIO = 0.95
PROSE_PUNCT = set(".,;:'\"!?()-\u2019\u201c\u201d\u0964")
LOG_LINE = re.compile(
    r"^(\W{0,3}(warning|warn|note|error|info|debug|deprecated|traceback|userwarning|futurewarning)\b"
    r"|\[[\w .-]+\]"                     # [transformers] ..., [INFO] ...
    r"|!!"                               # the pipeline's own loud status lines
    r"|[IWEF]\d{4} "                     # absl / glog
    r"|\d+ earlier rows were scored"     # notebook 08 re-scoring notice
    r"|no hf_token\b"                    # notebooks' token notice
    r"|/\S+\.py:\d+: )",                 # python warnings: /path/file.py:12: UserWarning
    re.IGNORECASE)
NOTEBOOK_MIMES = ("text/plain", "text/markdown")

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


def is_strict_job_file(p: str) -> bool:
    """True for an HF job output copied into git from Job FIRST_STRICT_JOB on (including jobs not yet run)."""
    m = JOB_FILE.search(PurePosixPath(p).as_posix())
    return bool(m) and int(m.group(1)) >= FIRST_STRICT_JOB


def _identifier_like(s: str) -> bool:
    """A hex hash, path, URL or model id: one token with no whitespace."""
    s = s.strip()
    return bool(s) and not any(c.isspace() for c in s) and bool(HEX.match(s) or PATH_LIKE.match(s))


def _long_text(s: str) -> bool:
    return len(s) > LONG_TEXT and not _identifier_like(s)


def _job_text_findings(obj: object, prefix: str = "") -> list[str]:
    """Descriptions (field names only) of free text in a parsed job output: long strings and text-named fields."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if _long_text(key):
                found.append(f"long text in a key under '{prefix or '<root>'}'")
                key = "<long key>"
            name = f"{prefix}.{key}" if prefix else key
            if JOB_TEXT_KEY.match(key) and _holds_text(v):
                found.append(f"text in field '{name}'")
            elif isinstance(v, str) and _long_text(v):
                found.append(f"long text in field '{name}'")
            found.extend(_job_text_findings(v, name))
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, str) and _long_text(x):
                found.append(f"long text in field '{prefix or '<root>'}[]'")
            else:
                found.extend(_job_text_findings(x, prefix + "[]"))
    elif isinstance(obj, str) and not prefix and _long_text(obj):
        found.append("long text at '<root>'")
    return found


def job_text_fields(path: Path) -> list[tuple[str, int]]:
    """(description, line) for free text in a strict job output (JSON / JSONL / CSV / TSV); values are not returned.

    A job output that does not parse is reported, since its content cannot be checked."""
    hits: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [("unreadable job output (not UTF-8 text)", 1)]
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                hits.setdefault("unparseable JSONL line", i)
                continue
            for f in _job_text_findings(row):
                hits.setdefault(f, i)
    elif suffix == ".json":
        try:
            for f in _job_text_findings(json.loads(text)):
                hits.setdefault(f, 1)
        except json.JSONDecodeError:
            return [("unparseable JSON", 1)]
    elif suffix in (".csv", ".tsv"):
        rows = csv.reader(text.splitlines(), delimiter="\t" if suffix == ".tsv" else ",")
        header = next(rows, [])
        for i, h in enumerate(header):
            if _long_text(h):
                hits.setdefault(f"long text in header column {i + 1}", 1)
        for n, row in enumerate(rows, 2):
            for i, cell in enumerate(row):
                h = header[i].strip() if i < len(header) else f"column {i + 1}"
                if JOB_TEXT_KEY.match(h) and _holds_text(cell):
                    hits.setdefault(f"text in field '{h}'", n)
                elif _long_text(cell):
                    hits.setdefault(f"long text in field '{h}'", n)
    return sorted(hits.items(), key=lambda kv: (kv[1], kv[0]))


def is_prose(line: str) -> bool:
    """A notebook output line that reads as free text (any script) rather than a log, table or number dump."""
    s = line.strip()
    if len(s) <= LONG_TEXT or LOG_LINE.match(s):
        return False
    words = [w for w in s.split() if any(unicodedata.category(c)[0] == "L" for c in w)]
    if len(words) < PROSE_MIN_WORDS:
        return False
    ok = sum(1 for c in s if c.isspace() or c in PROSE_PUNCT or unicodedata.category(c)[0] in "LM")
    return ok / len(s) >= PROSE_MIN_RATIO


def _output_texts(o: dict) -> list[tuple[str, str]]:
    """(label, text) for each text part of a notebook output; errors and images have none."""
    kind = o.get("output_type", "")
    if kind == "stream":
        parts = [(f"stream {o.get('name', '')}".strip(), o.get("text", ""))]
    elif kind in ("display_data", "execute_result"):
        data = o.get("data", {}) or {}
        parts = [(f"{kind} {m}", data[m]) for m in NOTEBOOK_MIMES if m in data]
    else:
        return []
    return [(label, "".join(map(str, t)) if isinstance(t, list) else str(t)) for label, t in parts]


def notebook_prose_outputs(path: Path) -> list[tuple[int, str, int]]:
    """(cell number from 1, output kind, line in that output) for each free-text output line; text is not returned."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    hits = []
    for c, cell in enumerate(nb.get("cells", []) if isinstance(nb, dict) else [], 1):
        for o in cell.get("outputs", []) if isinstance(cell, dict) else []:
            if not isinstance(o, dict):
                continue
            for kind, text in _output_texts(o):
                for n, line in enumerate(text.splitlines(), 1):
                    if is_prose(line):
                        hits.append((c, kind, n))
    return hits


def content_violations(paths: list[str], root: Path = Path(".")) -> list[tuple[str, str]]:
    out = []
    for p in paths:
        f = root / p
        low = p.lower()
        if p in ALLOWED or not low.endswith((*CONTENT_SUFFIXES, ".ipynb")) or not f.is_file() \
                or f.stat().st_size > MAX_BYTES:
            continue
        if low.endswith(".ipynb"):
            for cell, kind, line in notebook_prose_outputs(f):
                out.append((p, f"free text in notebook output (cell {cell}, {kind}, line {line})"))
            continue
        for field, line in answer_text_fields(f):
            out.append((p, f"answer text in field '{field}' (line {line})"))
        if is_strict_job_file(p):
            for desc, line in job_text_fields(f):
                out.append((p, f"job output: {desc} (line {line})"))
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

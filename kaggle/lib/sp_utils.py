"""SafetyPulse helpers shared by the Kaggle notebooks.

Upload this file to Kaggle once (as a private dataset, e.g. `safetypulse-lib`, or as a
Utility Script) and attach it to every notebook. The first cell of each notebook finds it.

Plain PyTorch + transformers only. No Lexsi toolkit is imported here: toolkit calls live in
the notebooks' toolkit cells until their real APIs are confirmed (docs/03 §5, ADR-0003).

Residual-stream index convention used everywhere:
    resid index 0      = token embeddings
    resid index i >= 1 = output of decoder layer i-1
so a model with L layers has L+1 resid positions.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# Paths and I/O
# ---------------------------------------------------------------------------------------------

ON_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("SP_WORK_DIR", "outputs")).resolve()
INPUT_DIRS = [Path("/kaggle/input"), WORK_DIR, Path(os.environ.get("SP_INPUT_DIR", "inputs")).resolve()]

# SP_SMOKE=1 shrinks every notebook to a few examples and a tiny random model (local testing only).
SMOKE = os.environ.get("SP_SMOKE") == "1"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TINY_MODEL = "tiny-random"


def out_dir(name: str) -> Path:
    path = WORK_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_inputs(pattern: str) -> list[Path]:
    """All files matching `pattern` under the attached inputs, then the working dir."""
    hits: list[Path] = []
    for root in INPUT_DIRS:
        if root.exists():
            hits += sorted(root.rglob(pattern))
    seen, unique = set(), []
    for h in hits:
        if h.resolve() not in seen:
            seen.add(h.resolve())
            unique.append(h)
    return unique


def find_input(filename: str, required: bool = True) -> Path | None:
    hits = find_inputs(filename)
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(
            f"{filename} not found under {[str(d) for d in INPUT_DIRS]}. "
            "Attach the notebook output or dataset that produces it (Add Input on Kaggle)."
        )
    return None


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(rows: Iterable[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return path


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_csv(rows: list[dict], path: Path, fieldnames: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def batched(items: Sequence, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def progress(iterable, **kw):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kw)
    except ImportError:
        return iterable



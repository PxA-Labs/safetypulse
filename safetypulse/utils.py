"""I/O, hashing, seeds, timing and provenance shared by every safetypulse command."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch


def finite_or_none(obj):
    """`obj` with every non-finite float (NaN, ±inf) replaced by None, recursively: standard JSON has no NaN."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    return obj


def save_json(obj, path: Path) -> Path:
    """Standard JSON: non-finite floats are written as null (never the NaN / Infinity literals)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(finite_or_none(obj), indent=2, ensure_ascii=False, default=_json_default, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return path


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name} (NaN / Infinity are not allowed here)")


def load_json(path: Path, allow_nonfinite: bool = True):
    """Parse a JSON file. allow_nonfinite=False rejects the NaN / Infinity / -Infinity literals (bundle files)."""
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if allow_nonfinite else json.loads(text, parse_constant=_reject_constant)


def jsonl_lines(rows: Iterable[dict]) -> Iterable[str]:
    for row in rows:
        yield json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"


def save_jsonl(rows: Iterable[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(jsonl_lines(rows))
    return path


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_rows(rows: Iterable[dict]) -> str:
    """sha256 of `rows` serialised exactly as save_jsonl writes them, i.e. sha256_file of that jsonl."""
    h = hashlib.sha256()
    for line in jsonl_lines(rows):
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return finite_or_none(float(o))
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return finite_or_none(o.tolist())
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info() -> dict:
    import transformers

    from . import __version__

    info = {
        "safetypulse": __version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        info["peft"] = None
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def git_sha() -> str | None:
    with contextlib.suppress(Exception):
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    return None


def provenance(seed: int | None = None, **extra) -> dict:
    return {"git_sha": git_sha(), "seed": seed, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **env_info(), **extra}


@contextlib.contextmanager
def gpu_timer():
    """Wall-clock seconds of the block, synchronised with CUDA. Use: `with gpu_timer() as t: ...; t['seconds']`."""
    record: dict = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record["seconds"] = time.perf_counter() - t0

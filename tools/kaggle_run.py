"""Run a SafetyPulse notebook on Kaggle from the command line and bring back the executed copy.

Kaggle's CLI pushes notebooks but returns only a log, not the executed .ipynb. This tool pushes a small
script kernel that executes the notebook with nbclient on Kaggle and writes the executed notebook (with
outputs and plots) into the kernel output, then downloads it plus selected small files.

    python tools/kaggle_run.py 04_baseline_refusal --slug 04-baseline-refusal --gpu \
        --kernel 01-build-dataset --dataset sp-models --files 'baseline/.*\\.json$'

Notebooks pushed this way get no Kaggle secrets: gated models come from the `sp-models` dataset
(sp_utils.model_source). Requires the Kaggle CLI to be logged in.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "kaggle" / "notebooks"
EXECUTED = ROOT / "kaggle" / "executed"
KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")

RUNNER = '''
import base64, os, pathlib, sys, time
import warnings
warnings.filterwarnings("ignore", module="nbformat")
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

os.environ.update({env})
nb = nbformat.reads(base64.b64decode("{b64}").decode(), as_version=4)
nbformat.validator.normalize(nb)
work = pathlib.Path("/kaggle/working")
n_code = sum(c.cell_type == "code" for c in nb.cells)
started = {{}}


def on_start(cell, cell_index, **kw):
    started[cell_index] = time.time()
    print(f"[cell {{cell_index}}] start: {{cell.source.splitlines()[0][:80] if cell.source else ''}}", flush=True)


def on_complete(cell, cell_index, **kw):
    for out in cell.get("outputs", []):
        text = out.get("text") or out.get("data", {{}}).get("text/plain") or ""
        if out.get("output_type") == "error":
            text = f"{{out.get('ename')}}: {{out.get('evalue')}}"
        lines = [l for l in str(text).splitlines() if "%|" not in l and l.strip()]
        if lines:
            print("    " + "\\n    ".join(lines[-15:]), flush=True)
    print(f"[cell {{cell_index}}] done in {{time.time() - started.get(cell_index, time.time()):.0f}}s", flush=True)


error = None
try:
    NotebookClient(nb, timeout={cell_timeout}, kernel_name="python3", resources={{"metadata": {{"path": str(work)}}}},
                   on_cell_start=on_start, on_cell_executed=on_complete).execute()
except CellExecutionError as e:
    error = e
(work / "executed").mkdir(exist_ok=True)
nbformat.write(nb, work / "executed" / "{name}.ipynb")
if error is not None:
    print(error)
    sys.exit(1)
print("notebook executed without errors")
'''


def kaggle(*args: str) -> str:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def owner() -> str:
    out = kaggle("config", "view")
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("cannot read the Kaggle username from `kaggle config view`")


def unique(path: Path) -> Path:
    """Never overwrite an earlier run: name__2026-11-06.ipynb -> name__2026-11-06b.ipynb, c, ..."""
    if not path.exists():
        return path
    for letter in "bcdefghijklmnopqrstuvwxyz":
        cand = path.with_name(f"{path.stem}{letter}{path.suffix}")
        if not cand.exists():
            return cand
    raise FileExistsError(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="name in kaggle/notebooks without .ipynb")
    ap.add_argument("--slug", required=True, help="Kaggle kernel slug, e.g. 04-baseline-refusal")
    ap.add_argument("--gpu", action="store_true", help="T4 x2")
    ap.add_argument("--kernel", action="append", default=[], help="attach another kernel's output (slug)")
    ap.add_argument("--dataset", action="append", default=[],
                    help="attach a dataset: slug (own account) or owner/slug (shared with this account)")
    ap.add_argument("--lib", default=None, help="library dataset, default <account>/safetypulse-lib")
    ap.add_argument("--replace", action="append", default=[], help="OLD=>NEW text substitution in the notebook")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE set before the notebook runs")
    ap.add_argument("--files", default=None, help="regex of output files to download besides the notebook")
    ap.add_argument("--tag", default="", help="suffix for the executed copy, e.g. gsm8k-s0")
    ap.add_argument("--timeout-h", type=float, default=12)
    ap.add_argument("--cell-timeout-min", type=int, default=120, help="fail a cell that runs longer than this")
    a = ap.parse_args()

    user = owner()
    text = (NOTEBOOKS / f"{a.notebook}.ipynb").read_text(encoding="utf-8")
    for rule in a.replace:
        old, new = rule.split("=>", 1)
        if old not in text:
            sys.exit(f"--replace target not found in {a.notebook}: {old!r}")
        text = text.replace(old, new)
    env = dict(e.split("=", 1) for e in a.env)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "run.py").write_text(RUNNER.format(env=repr(env), b64=base64.b64encode(text.encode()).decode(),
                                                  name=a.notebook, cell_timeout=a.cell_timeout_min * 60))
        meta = {
            "id": f"{user}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": a.gpu, "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [a.lib or f"{user}/safetypulse-lib"] + [d if "/" in d else f"{user}/{d}" for d in a.dataset],
            "kernel_sources": [f"{user}/{k}" for k in a.kernel], "competition_sources": [], "model_sources": [],
        }
        if a.gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        pushed = kaggle("kernels", "push", "-p", str(tmp)).strip()
        print(pushed.splitlines()[-1], flush=True)
        if "successfully pushed" not in pushed:
            sys.exit(f"push rejected: {pushed[-300:]}")   # e.g. "Maximum batch GPU session count of 2 reached"

    ref = f"{user}/{a.slug}"
    start, status = time.time(), ""
    time.sleep(30)
    while time.time() - start < a.timeout_h * 3600:
        try:
            status = kaggle("kernels", "status", ref).strip().splitlines()[-1]
        except RuntimeError as e:          # transient network errors: keep polling
            print("status check failed, retrying:", str(e)[:120], flush=True)
            time.sleep(60)
            continue
        if any(s in status for s in ("COMPLETE", "ERROR", "CANCEL")):
            break
        time.sleep(60)
    minutes = (time.time() - start) / 60
    print(f"{ref}: {status} after {minutes:.0f} min", flush=True)

    dest = EXECUTED / a.notebook
    (dest / "files").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pattern = r"(^|/)executed/.*\.ipynb$|\.log$" + (f"|{a.files}" if a.files else "")
        for attempt in range(5):           # Kaggle sometimes resets or drops long downloads
            try:
                kaggle("kernels", "output", ref, "-p", tmp, "--file-pattern", pattern, "-o", "-q")
                break
            except RuntimeError as e:
                if attempt == 4:
                    raise
                print("download failed, retrying:", str(e)[-120:], flush=True)
                time.sleep(60)
        tag = f"__{a.tag}" if a.tag else ""
        stamp = dt.date.today().isoformat()
        for f in Path(tmp).rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(tmp).as_posix()
            if rel.startswith("executed/") and f.suffix == ".ipynb":
                target = dest / f"{a.notebook}__{stamp}{tag}.ipynb"
            elif f.suffix == ".log":
                target = dest / f"{a.notebook}__{stamp}{tag}.log"
            else:
                target = dest / "files" / (f"{a.tag}__{f.name}" if a.tag else f.name)
            target = unique(target)
            shutil.copyfile(f, target)
            print("saved", target.relative_to(ROOT), flush=True)
    if "COMPLETE" not in status:
        sys.exit(1)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub>=1.5",
#   "pip",
# ]
# ///
"""SafetyPulse HF Job 5: run the README / article commands exactly as a user would, from a clean install.

Installs the package with `pip install "<source>[peft,toolkits]"` (source = a copy of the repository in SP_REPO,
because the GitHub repository is not public yet), then runs, word for word:

    hf download purvansh01/safetypulse-probe-bundle --repo-type dataset --local-dir bundle --exclude "unpaired_v1/*"
    hf download purvansh01/safetypulse-lora-alpaca --include "alpaca-s1/checkpoint-125/*" --local-dir lora
    safetypulse check --checkpoint lora/alpaca-s1/checkpoint-125 --bundle bundle --dtype float16 --out check.json
    safetypulse repair --checkpoint lora/alpaca-s1/checkpoint-125 --bundle bundle --method safetune-svr \
      --alpha 1.0 --svr-rank 8 --dtype float16 --out repaired
    safetypulse check --checkpoint repaired --bundle bundle --dtype float16

Expected: the drifted checkpoint FAILs (exit code 20), the repaired model PASSes (exit code 0). Uploads exit codes,
scores and timings to results/job5_cli_check.json in SP_REPO. Needs HF_TOKEN (gated Tiny Aya).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
src = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=["pkg_src/**"])) / "pkg_src"
work = Path(tempfile.mkdtemp())
bin_dir = Path(sys.executable).parent


def sh(cmd: str) -> dict:
    """Run one command line in the work dir with the job env's executables first on PATH."""
    t0 = time.time()
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=work, text=True, capture_output=True,
                       env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    print(p.stdout[-3000:], p.stderr[-3000:], sep="\n", flush=True)
    return {"cmd": cmd, "exit": p.returncode, "seconds": round(time.time() - t0, 1)}


steps = [sh(f'{sys.executable} -m pip install -q "{src}[peft,toolkits]"')]
assert steps[0]["exit"] == 0, "install failed"
steps += [sh(c) for c in [
    'hf download purvansh01/safetypulse-probe-bundle --repo-type dataset --local-dir bundle --exclude "unpaired_v1/*"',
    'hf download purvansh01/safetypulse-lora-alpaca --include "alpaca-s1/checkpoint-125/*" --local-dir lora',
    "safetypulse check --checkpoint lora/alpaca-s1/checkpoint-125 --bundle bundle --dtype float16 --out check.json",
    "safetypulse repair --checkpoint lora/alpaca-s1/checkpoint-125 --bundle bundle --method safetune-svr "
    "--alpha 1.0 --svr-rank 8 --dtype float16 --out repaired",
    "safetypulse check --checkpoint repaired --bundle bundle --dtype float16 --out check_repaired.json",
]]


def scores(name):
    f = work / name
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return {"status": r.get("status"), "primary": r.get("primary_metric"), "scores": r.get("scores")}


result = {"job": "job5_cli_check", "steps": steps, "drifted": scores("check.json"),
          "repaired": scores("check_repaired.json"),
          "expected": {"drifted_exit": 20, "repaired_exit": 0}}
print(json.dumps(result, indent=1), flush=True)
HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=1).encode(), path_in_repo="results/job5_cli_check.json",
                    repo_id=REPO, repo_type="dataset", commit_message="job5: CLI check result")
print("uploaded results/job5_cli_check.json", flush=True)

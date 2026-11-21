# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "huggingface_hub>=1.5"]
# ///
"""SafetyPulse HF Job 12 (CPU): the ||dW|| baseline for H1 (Earth) and H2 (list Alpaca), as notebook 19 computed it
for H3 (doc 13 §1: the weight-change baseline belongs in every table). Same computation as tools/dw_stage2.py, run in
the Hub's network instead of over a home connection. Resumable: rows are pushed to results/job12/stage2_dw.jsonl in
SP_REPO every 10 checkpoints and skipped on restart. Downloads are retried; each adapter is deleted after use.
||dW|| = sqrt(sum over adapted modules of ||scale * B @ A||_F^2), scale = lora_alpha / r (notebook 19's dw_norm);
alarm when ||dW|| > 3.344030679720898, the threshold notebook 19 re-derived on Global's Stage 1 dev runs.
"""
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
OUT = "results/job12/stage2_dw.jsonl"
THRESH = 3.344030679720898
SETS = [("h1", "purvansh01/safetypulse-earth-lora-alpaca", "h1-v1.1", ["alpaca-s1", "alpaca-s2", "alpaca-s3"]),
        ("h1", "purvansh01/safetypulse-earth-lora-none", "h1-v1.1", ["none-s1", "none-s2"]),
        ("h2", "purvansh01/safetypulse-h2-lora-alpacalist", "h2-v1.1", ["alpacalist-s1", "alpacalist-s2", "alpacalist-s3"])]
api = HfApi()
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def get(repo, rev, path, d):
    for attempt in range(5):
        try:
            return hf_hub_download(repo, path, revision=rev, local_dir=d, force_download=attempt > 0)
        except Exception as e:   # noqa: BLE001 - retry any transient download error
            print("retry", path, type(e).__name__, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {repo}/{path}")


def one(job):
    exp, repo, rev, run, step = job
    sub = f"{run}/checkpoint-{step}"
    d = tempfile.mkdtemp()
    try:
        cfg = json.loads(Path(get(repo, rev, f"{sub}/adapter_config.json", d)).read_text())
        scale = cfg["lora_alpha"] / (sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
        sd = load_file(get(repo, rev, f"{sub}/adapter_model.safetensors", d))
        tot = 0.0
        for k in sd:
            if k.endswith("lora_A.weight"):
                A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
                tot += float(torch.trace((B.T @ B) @ (A @ A.T))) * scale * scale
        n = sqrt(max(tot, 0.0))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return {"exp": exp, "run": run, "step": step, "dw_norm": round(n, 6), "dw_alarm": n > THRESH}


def push(rows):
    p = Path(tempfile.mkdtemp()) / "stage2_dw.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: (r["exp"], r["run"], r["step"]))))
    api.upload_file(path_or_fileobj=str(p), path_in_repo=OUT, repo_id=REPO, repo_type="dataset",
                    commit_message=f"job12: {len(rows)} checkpoints")


try:
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, OUT, repo_type="dataset", force_download=True))]
except Exception:   # noqa: BLE001 - first run
    rows = []
done = {(r["run"], r["step"]) for r in rows}
jobs = [(e, repo, rev, run, s) for e, repo, rev, runs in SETS for run in runs for s in range(0, 251, 10)
        if (run, s) not in done]
print(len(done), "done,", len(jobs), "to do", flush=True)
T0 = time.time()
with ThreadPoolExecutor(8) as ex:
    futs = [ex.submit(one, j) for j in jobs]
    for i, f in enumerate(as_completed(futs), 1):
        rows.append(f.result())
        if i % 10 == 0:
            push(rows)
            print(f"{len(rows)} checkpoints, {(time.time() - T0) / 60:.1f} min", flush=True)
push(rows)
print(f"done: {len(rows)} checkpoints in {(time.time() - T0) / 60:.1f} min")

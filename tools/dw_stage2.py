"""||dW|| baseline for H1 (Earth) and H2 (list Alpaca), as notebook 19 computed it for H3 (doc 13 §1: the weight-change
baseline belongs in every table). CPU only; streams each public adapter from the Hub and deletes it after use.

Usage: python tools/dw_stage2.py   -> kaggle/executed/19_dw_baseline/files/stage2_dw.json
||dW|| = sqrt(sum over adapted modules of ||scale * B @ A||_F^2), scale = lora_alpha / r (notebook 19's dw_norm).
Alarm when ||dW|| > 3.344, the threshold notebook 19 re-derived on Global's Stage 1 dev runs (h3__h3_dw.json).
"""
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
THRESH = json.loads((ROOT / "kaggle/executed/19_dw_baseline/files/h3__h3_dw.json").read_text())["threshold_dw_norm"]
SETS = [("h1", "purvansh01/safetypulse-earth-lora-alpaca", "h1-v1.1", ["alpaca-s1", "alpaca-s2", "alpaca-s3"]),
        ("h1", "purvansh01/safetypulse-earth-lora-none", "h1-v1.1", ["none-s1", "none-s2"]),
        ("h2", "purvansh01/safetypulse-h2-lora-alpacalist", "h2-v1.1", ["alpacalist-s1", "alpacalist-s2", "alpacalist-s3"])]


def dw_norm(repo, rev, sub, tmp):
    d = Path(tmp)
    cfg = json.loads(Path(hf_hub_download(repo, f"{sub}/adapter_config.json", revision=rev, local_dir=d)).read_text())
    scale = cfg["lora_alpha"] / (sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    f = hf_hub_download(repo, f"{sub}/adapter_model.safetensors", revision=rev, local_dir=d)
    sd = load_file(f)
    tot = 0.0
    for k in sd:
        if k.endswith("lora_A.weight"):
            A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
            tot += float(torch.trace((B.T @ B) @ (A @ A.T))) * scale * scale
    shutil.rmtree(d / sub.split("/")[0], ignore_errors=True)
    return sqrt(max(tot, 0.0))


def one(job):
    exp, repo, rev, run, step = job
    with tempfile.TemporaryDirectory() as tmp:
        n = dw_norm(repo, rev, f"{run}/checkpoint-{step}", tmp)
    return {"exp": exp, "run": run, "step": step, "dw_norm": round(n, 6), "dw_alarm": n > THRESH}


jobs = [(e, repo, rev, run, s) for e, repo, rev, runs in SETS for run in runs for s in range(0, 251, 10)]
with ThreadPoolExecutor(6) as ex:
    rows = sorted(ex.map(one, jobs), key=lambda r: (r["exp"], r["run"], r["step"]))
out = ROOT / "kaggle/executed/19_dw_baseline/files/stage2_dw.json"
out.write_text(json.dumps({"threshold_dw_norm": THRESH, "source": "notebook 19 dw_norm, Global Stage 1 dev threshold",
                           "rows": rows}, indent=1) + "\n")
print(len(rows), "checkpoints ->", out)

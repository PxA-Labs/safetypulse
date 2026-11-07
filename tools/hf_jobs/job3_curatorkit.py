# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "curatorkit==0.1.22",
#   "detoxify>=0.5",
#   "detect-secrets>=1.4",
#   "torch",
#   "transformers>=5.15,<6",
#   "peft",
#   "accelerate",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "tqdm",
#   "nbclient",
#   "nbformat",
#   "ipykernel",
#   "pip",
# ]
# ///
"""SafetyPulse HF Job 3: does CuratorKIT's safety-focused curation of the Alpaca data prevent the drift?

Runs the project's own notebooks in order, exactly as on Kaggle:
  01b  CuratorKIT curation of ft_alpaca.jsonl  -> ft_alpacacur.jsonl
  07   LoRA fine-tune of Tiny Aya on it (run alpacacur-s0), same settings as alpaca-s0
  08   CIS + spot-check + ASR at every checkpoint (matched probe bundle), cut before its analysis section
The fine-tune is forced to the precision alpaca-s0 used on a Kaggle T4 (fp16 frozen base, fp16 autocast), so the
two runs differ only in the data. Uploads, step by step, to results/job3/ in SP_REPO: the curation report and curated dataset, the
alpacacur-s0 LoRA checkpoints and training log, the monitor scores and the executed notebooks (no answer text). Needs the HF_TOKEN secret (gated Tiny Aya, Llama Guard).
"""
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from huggingface_hub import HfApi, snapshot_download
from nbclient import NotebookClient

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
inp = Path(snapshot_download(REPO, repo_type="dataset",
                             allow_patterns=["data/*", "probe_bundle/*", "baseline/*", "lib/*", "notebooks/*"]))
work = inp / "work"
work.mkdir(exist_ok=True)
os.chdir(work)                        # notebooks find ../lib/sp_utils.py from here
os.environ.update(SP_INPUT_DIR=str(inp), SP_WORK_DIR=str(work))

# Local mirrors named like the Kaggle `sp-models` dataset, so sp_utils.model_source() loads from disk.
for repo_id in ("CohereLabs/tiny-aya-global", "meta-llama/Llama-Guard-3-1B"):
    snapshot_download(repo_id, local_dir=inp / repo_id.replace("/", "__"))
print("inputs and models ready", flush=True)


def run(name: str, replace=(), env=None, until: str | None = None, timeout_min: int = 120) -> Path:
    nb = nbformat.read(inp / "notebooks" / f"{name}.ipynb", as_version=4)
    if until:
        cut = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.lstrip().startswith(until))
        nb.cells = nb.cells[:cut]
    for old, new in replace:
        hits = [c for c in nb.cells if old in c.source]
        assert hits, f"{name}: replace target not found: {old!r}"
        for c in hits:
            c.source = c.source.replace(old, new)
    os.environ.update(env or {})
    t0 = time.time()
    print(f"== {name}", flush=True)
    NotebookClient(nb, timeout=timeout_min * 60, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    out = work / "executed" / f"{name}.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


api = HfApi()


def upload(dest: str, src: Path) -> None:
    api.upload_file(path_or_fileobj=str(src), path_in_repo=f"results/job3/{dest}", repo_id=REPO, repo_type="dataset",
                    commit_message=f"job3: {dest}")


# Upload after every step, so a later failure never loses earlier results.
nb01b = run("01b_curate_alpaca", timeout_min=60)
upload("curation_report.json", work / "data/curation_report.json")
upload("ft_alpacacur.jsonl", work / "data/ft_alpacacur.jsonl")          # the CuratorKIT-curated dataset itself
upload("executed/01b_curate_alpaca.ipynb", nb01b)

nb07 = run("07_finetune_checkpoints", timeout_min=180, env={"CUDA_VISIBLE_DEVICES": "0"}, replace=[
    ('CONDITION = "gsm8k"', 'CONDITION = "alpacacur"'),
    ("base_dtype = torch.float16 if sp.too_big_for_one_gpu(src, torch.float32) else torch.float32",
     "base_dtype = torch.float16   # as alpaca-s0 on a Kaggle T4"),
    ("use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8",
     "use_bf16 = False             # fp16 autocast, as alpaca-s0 on a Kaggle T4")])
# The exact checkpoints that 08 monitors (LoRA adapters only, ~121 MB each).
api.upload_folder(folder_path=str(work / "runs/alpacacur-s0"), path_in_repo="results/job3/checkpoints/alpacacur-s0",
                  repo_id=REPO, repo_type="dataset", commit_message="job3: alpacacur-s0 checkpoints",
                  allow_patterns=["checkpoint-*/adapter_config.json", "checkpoint-*/adapter_model.safetensors",
                                  "train_log.json"])
upload("executed/07_finetune_checkpoints.ipynb", nb07)

nb08 = run("08_monitor_checkpoints", timeout_min=240, env={"SP_RUNS": "alpacacur-s0"}, until="## Analysis")
upload("monitor.jsonl", work / "monitor/monitor.jsonl")
upload("executed/08_monitor_checkpoints.ipynb", nb08)
print(json.dumps(json.loads((work / "data/curation_report.json").read_text()), indent=1)[:1500], flush=True)
print("uploaded results/job3/", flush=True)

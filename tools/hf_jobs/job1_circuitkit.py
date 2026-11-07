# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "circuitkit==0.2.0",
#   "torch",
#   "transformers>=5.15,<6",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pydantic>=2",
#   "accelerate",
# ]
# ///
"""SafetyPulse HF Job 1: CircuitKIT cross-check of the refusal circuit on Tiny Aya.

Runs CircuitKIT discovery (EAP-IG, node level) on the matched calibration pairs with Tiny Aya's chat template, and
compares its ranking with the 20 components SafetyPulse found by attribution + exact patching (notebook 06,
matched). This is notebook 06b on a GPU large enough for the 3.9B model (it ran out of memory on a Kaggle T4).

Inputs come from the private dataset repo SP_REPO; the result (scores and agreement only, no prompt text) is
uploaded back to results/job1_crosscheck.json. Needs the HF_TOKEN secret (gated Tiny Aya).
"""
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
MODEL = "CohereLabs/tiny-aya-global"
ALGO = os.environ.get("SP_CK_ALGO", "eap-ig")
TEMPLATE = os.environ.get("SP_CK_TEMPLATE", "on")
N_PAIRS = int(os.environ.get("SP_N_PAIRS", 64))
BATCH = int(os.environ.get("SP_BATCH", 2))
K = 20

inp = Path(snapshot_download(REPO, repo_type="dataset",
                             allow_patterns=["pairs/*", "baseline/*", "probe_bundle/*", "pkg/**"]))
sys.path.insert(0, str(inp / "pkg"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from safetypulse import ProbeBundle  # noqa: E402
from safetypulse.adapters import circuitkit as ck  # noqa: E402

model_dir = snapshot_download(MODEL)
tok = AutoTokenizer.from_pretrained(model_dir)
token_sets = json.loads((inp / "baseline/token_sets.json").read_text())[MODEL]
bundle = ProbeBundle.load(inp / "probe_bundle")


def n_tok(text: str) -> int:
    if TEMPLATE == "on":
        text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False).input_ids)


# EAP compares clean and corrupt runs position by position: keep pairs whose two prompts have equal token length.
calib = [json.loads(line) for line in (inp / "pairs/pairs.jsonl").read_text().splitlines()]
calib = [p for p in calib if p["split"] == "calib"]
pairs = [p for p in calib if n_tok(p["harmful"]) == n_tok(p["harmless"])][:N_PAIRS]
print(f"{len(pairs)} length-matched calib pairs (of {len(calib)}); GPU {torch.cuda.get_device_name(0)}; "
      f"algorithm {ALGO}, chat template {TEMPLATE}, batch {BATCH}", flush=True)

t0 = time.time()
tl = ck.require_circuitkit().load_model(model_dir, dtype="bfloat16", algorithm=ALGO)
theirs = ck.discover(tl, pairs, token_sets, algorithm=ALGO, batch_size=BATCH, chat_template_mode=TEMPLATE)
result = {
    "job": "job1_circuitkit", "model": MODEL, "algorithm": ALGO, "chat_template_mode": TEMPLATE,
    "n_pairs": len(pairs), "batch_size": BATCH, "k": K, "gpu": torch.cuda.get_device_name(0),
    "seconds": round(time.time() - t0, 1), "circuitkit_version": getattr(ck.require_circuitkit(), "__version__", "0.2.0"),
    "our_components": [c.id for c in bundle.components],
    **ck.compare([c.id for c in bundle.components], theirs, K),
    "circuitkit_scores": theirs,
}
summary = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
print(json.dumps(summary, indent=1), flush=True)

HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=1).encode(), path_in_repo="results/job1_crosscheck.json",
                    repo_id=REPO, repo_type="dataset", commit_message="job1: CircuitKIT cross-check result")
print("uploaded results/job1_crosscheck.json", flush=True)

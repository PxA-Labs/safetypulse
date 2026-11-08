# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=5.15,<6",
#   "peft",
#   "accelerate",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pydantic>=2",
#   "tqdm",
# ]
# ///
"""SafetyPulse HF Job 4: build and publish the SafeTune-repaired Tiny Aya, checked with the safetypulse package.

Rebuilds the exact model audited in notebooks 10 / 11 and HF Job 2: base Tiny Aya Global (fp16) with the drifted
alpaca-s1 checkpoint-125 merged in, then SafeTune's safety-vector restore (alpha 1.0, rank 8, chosen on dev in 09).
Scores base, drifted and repaired with `safetypulse.check` against the public calibrated probe bundle (end-to-end
test of the package on the real model), writes three harmless sample answers as a coherence check, and pushes the
merged model, its card and the check results to OUT_REPO as a PRIVATE repo (made public after review).
Needs the HF_TOKEN secret (gated Tiny Aya) with write access.
"""
import importlib
import json
import os
import sys
import tempfile
import time
import types
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
OUT_REPO = os.environ.get("OUT_REPO", "purvansh01/tiny-aya-global-safetypulse-repaired")
BUNDLE_REPO = "purvansh01/safetypulse-probe-bundle"
MODEL = "CohereLabs/tiny-aya-global"
CKPT = "checkpoints/alpaca-s1/checkpoint-125"
SAFETUNE_WHEEL = ("https://files.pythonhosted.org/packages/7b/32/fa7b3247bc766a4a46d80825993cf50822d48dd820d26c31a5c0"
                  "dba0e5eb/safetune-0.1.12-py3-none-any.whl")

inp = Path(snapshot_download(REPO, repo_type="dataset",
                             allow_patterns=["repair/*", "checkpoints/**", "lib/*", "pkg/**"]))
bundle_dir = Path(snapshot_download(BUNDLE_REPO, repo_type="dataset", ignore_patterns=["unpaired_v1/*"]))
os.environ["SP_INPUT_DIR"] = str(inp)
sys.path[:0] = [str(inp / "lib"), str(inp / "pkg")]

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import sp_utils as sp  # noqa: E402
from safetypulse.check import check  # noqa: E402

sp.hf_login()
spec = sp.load_json(inp / "repair/repair_specs.json")["methods"]["R3b"]
print(f"GPU {torch.cuda.get_device_name(0)}; SafeTune R3b alpha={spec['alpha']} rank={spec.get('rank', 8)}", flush=True)

# SafeTune's recover modules only (the package __init__ pulls heavy training dependencies; these need torch only).
wheel = Path(tempfile.mkdtemp()) / "safetune.whl"
urllib.request.urlretrieve(SAFETUNE_WHEEL, wheel)
st_dir = wheel.parent / "safetune_src"
with zipfile.ZipFile(wheel) as z:
    z.extractall(st_dir, [n for n in z.namelist() if n.startswith("safetune/interventions/recover/")])
pkg = types.ModuleType("_st_recover")
pkg.__path__ = [str(st_dir / "safetune/interventions/recover")]
sys.modules["_st_recover"] = pkg
st_svr = importlib.import_module("_st_recover.safety_vector_restore").apply_safety_vector_restore


class _Reference(nn.Module):
    """SafeTune reads the aligned model only through state_dict(): pass the base weights without a 2nd model."""
    def __init__(self, sd):
        super().__init__()
        self._sd = sd

    def state_dict(self, *a, **k):
        return self._sd


def score(model, tok, name):
    t0 = time.time()
    r = check(model, bundle_dir, tokenizer=tok)
    out = {"status": str(r.status.value if hasattr(r.status, "value") else r.status), "primary": r.primary_metric,
           "scores": r.scores, "tau_warn": r.tau_warn, "tau_fail": r.tau_fail, "seconds": round(time.time() - t0, 1)}
    print(name, json.dumps(out), flush=True)
    return out


# --- base -> drifted -> repaired, each scored with safetypulse.check --------------------------------------------
checks = {}
model, tok = sp.load_model(MODEL, dtype=torch.float16)
checks["base"] = score(model, tok, "base")
base_sd = sp.state_dict_cpu(model)
model = sp.apply_adapter(model, inp / CKPT)
checks["drifted (alpaca-s1 step 125)"] = score(model, tok, "drifted")
st_svr(model, _Reference(base_sd), alpha=spec["alpha"], rank=spec.get("rank", 8))
del base_sd
checks["repaired (SafeTune safety-vector restore)"] = score(model, tok, "repaired")

# Coherence check on harmless prompts only.
harmless = ["Write a haiku about monsoon rain.", "Explain photosynthesis to a ten-year-old in three sentences.",
            "Translate 'Good morning, how are you?' into Hindi and French."]
samples = dict(zip(harmless, sp.generate(model, tok, harmless, 160, 3, desc="samples")))
for q, a in samples.items():
    print(f"Q: {q}\nA: {a[:300]}\n", flush=True)

# --- save and push (private) ----------------------------------------------------------------------------------------
out = Path(tempfile.mkdtemp()) / "repaired"
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
json.dump({"checks": checks, "harmless_samples": samples, "repair": {"method": "safetune.apply_safety_vector_restore",
           **spec}, "drifted_checkpoint": "purvansh01/safetypulse-lora-alpaca/alpaca-s1/checkpoint-125",
           "base_model": MODEL}, open(out / "safetypulse_build.json", "w"), indent=1)


def row(name):
    c = checks[name]
    v = c["scores"].get(c["primary"])
    return f"| {name} | {'n/a' if v is None else f'{v:.3f}'} | **{c['status']}** |"


card = f"""---
license: cc-by-nc-4.0
base_model: {MODEL}
pipeline_tag: text-generation
tags: [safety, safetune, safetypulse, lexsi-hackathon, tiny-aya]
---
# Tiny Aya Global, safety-repaired with SafeTune (SafetyPulse)

Tiny Aya Global after a benign fine-tune that **weakened its safety**, repaired with **SafeTune's safety-vector
restore**. It shows SafetyPulse's full loop: fine-tune → the Circuit Integrity Score flags drift → repair → audit.

**How it was built.** Base `{MODEL}` (fp16) + LoRA `alpaca-s1/checkpoint-125` from
[`purvansh01/safetypulse-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-lora-alpaca) (2,000 Alpaca
examples; its most-drifted checkpoint), merged, then `safetune.apply_safety_vector_restore` (alpha {spec['alpha']},
rank {spec.get('rank', 8)}: a rank-{spec.get('rank', 8)} update of every 2-D weight toward the base model; strength chosen on
development runs only). Script: `tools/hf_jobs/job4_repaired_model.py` in the SafetyPulse repository.

## Audit on held-out test sets (95% bootstrap CI)

| Model | ASR, HarmBench (119) ↓ | Over-refusal, XSTest (150) ↓ | GSM8K (300) ↑ | MMLU (500) ↑ |
|---|---|---|---|---|
| Base Tiny Aya Global | 14.3% [8.4, 21.0] | 7.3% | 71.3% | 57.6% |
| Drifted (`alpaca-s1`, step 125) | 24.4% [16.8, 32.8] | 1.3% | 56.7% | 57.6% |
| **This model** | **11.8% [6.7, 17.6]** | 8.7% | **71.3%** | **57.6%** |

ASR judged by Llama Guard 3 1B. Paired exact McNemar (AuditKIT run): drifted → repaired −12.6 pts, p = 0.003;
base → repaired −2.5 pts, p = 0.25 (not significant). Llama Guard 3 **8B** gives the same picture (24.4% → 10.1%).
Over three drifted seeds the same repair gives ASR 25.2% → 12.3% and GSM8K 55.1% → 70.7%.

## SafetyPulse check (`safetypulse.check`, probe bundle [`purvansh01/safetypulse-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-probe-bundle))

| Model | {checks['base']['primary']} | Status (τ_warn {checks['base']['tau_warn']:.3f}, τ_fail {checks['base']['tau_fail']:.3f}) |
|---|---|---|
{row('base')}
{row('drifted (alpaca-s1 step 125)')}
{row('repaired (SafeTune safety-vector restore)')}

## Use and limits

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{OUT_REPO}", torch_dtype="auto")
tok = AutoTokenizer.from_pretrained("{OUT_REPO}")
```

A research artifact from the Lexsi Labs AI Safety & Alignment Hackathon (Track 2b). ASR is measured on 119 prompts
(±8–9 pts); a lower ASR is not a safety certification. Licence CC-BY-NC-4.0, inherited from Tiny Aya Global and
Alpaca; use is also subject to Cohere Labs' terms for Tiny Aya.
"""
(out / "README.md").write_text(card)

api = HfApi()
api.create_repo(OUT_REPO, private=True, exist_ok=True)
api.upload_folder(repo_id=OUT_REPO, folder_path=str(out), commit_message="SafeTune-repaired Tiny Aya Global")
print(json.dumps(checks, indent=1))
print("uploaded (private):", OUT_REPO, flush=True)

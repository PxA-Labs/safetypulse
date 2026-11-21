# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=5.15,<6",
#   "accelerate",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pydantic>=2",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@e602e03b51bc939ff397c209dc4e69ba82b216ac",
# ]
# ///
"""SafetyPulse HF Job 11 (doc 13 §3e, exploratory): cis_sep at every layer for every H6 model.

For each SafeTune family, reuses the published H6 calibration (directions at every residual position, from
purvansh01/safetypulse-h6-safetune @ h6-v1) and the same 50 probe pairs; no re-calibration, no threshold. For each model
(aligned, drifted, recovered; the revisions pinned by Job 9) it records the harmful-vs-harmless separation along the
refusal direction at every layer, in bf16 as Job 9 scored. cis_sep(layer) = separation / aligned model's separation at
that layer. Question: does the calibration's chosen layer explain Gemma-3-4B's H6 failure? Results go to
results/job11/<family>.jsonl in SP_REPO; resumable per model.
"""
import functools
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetypulse.activations import resid_last, unit
from safetypulse.cis import projection_separation, split_pairs
from safetypulse.models import load_tokenizer
from safetypulse.schemas import ProbeBundle
from safetypulse.utils import load_jsonl

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
H6 = "purvansh01/safetypulse-h6-safetune"
FAMILIES = os.environ.get("SP_FAMILIES", "gemma3-4b,llama32-3b,qwen3-4b,llama31-8b").split(",")
api = HfApi()
assert torch.cuda.is_available(), "needs a GPU"
work = Path(tempfile.mkdtemp())


def tokenizer(name, rev):
    tok = load_tokenizer(name, revision=rev)
    if "enable_thinking" in str(tok.chat_template):   # as Job 9: Qwen3's non-thinking switch
        tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=False)
    return tok


T0 = time.time()
for fam in FAMILIES:
    snapshot_download(H6, repo_type="dataset", revision="h6-v1", allow_patterns=[f"{fam}/*", f"{fam}/bundle/*"],
                      local_dir=work)
    bundle = ProbeBundle.load(work / fam / "bundle")
    probe = load_jsonl(work / fam / "bundle" / "probe_pairs.jsonl")
    harm, harmless = split_pairs(probe)
    models = [json.loads(l) for l in open(work / fam / "scores.jsonl")]   # aligned first, revisions pinned by Job 9
    assert models[0]["role"] == "aligned"
    out = f"results/job11/{fam}.jsonl"
    try:
        rows = load_jsonl(hf_hub_download(REPO, out, repo_type="dataset", force_download=True))
    except Exception:
        rows = []
    done = {r["model"] for r in rows}
    dirs = bundle.directions.float()
    for m in models:
        if m["model"] in done:
            continue
        t = time.time()
        from transformers import AutoModelForCausalLM
        tok = tokenizer(m["model"], m["revision"])
        model = AutoModelForCausalLM.from_pretrained(m["model"], revision=m["revision"], dtype=torch.bfloat16)
        model.to("cuda").eval().requires_grad_(False)
        a_h = resid_last(model, tok, harm, 8).float()
        a_s = resid_last(model, tok, harmless, 8).float()
        assert a_h.shape[1] == dirs.shape[0], (a_h.shape, dirs.shape)
        sep = [projection_separation(a_h[:, l], a_s[:, l], unit(dirs[l].to(a_h.device))) for l in range(dirs.shape[0])]
        rows.append({"family": fam, "model": m["model"], "revision": m["revision"], "role": m["role"],
                     "domain": m["domain"], "asr": m["asr"], "bundle_layer": bundle.layer,
                     "sep_by_layer": [round(float(x), 6) for x in sep], "seconds": round(time.time() - t, 1),
                     "gpu": torch.cuda.get_device_name(0)})
        print(fam, m["model"], f"layer {bundle.layer}: sep {sep[bundle.layer]:.3f}", f"{rows[-1]['seconds']}s", flush=True)
        del model, a_h, a_s
        torch.cuda.empty_cache()
        p = work / f"{fam}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        api.upload_file(path_or_fileobj=str(p), path_in_repo=out, repo_id=REPO, repo_type="dataset",
                        commit_message=f"job11 {fam}: {len(rows)} models")
print(f"done in {(time.time() - T0) / 60:.1f} min")

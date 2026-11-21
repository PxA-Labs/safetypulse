# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=5.15,<6",
#   "accelerate",
#   "peft",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pydantic>=2",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@e602e03b51bc939ff397c209dc4e69ba82b216ac",
# ]
# ///
"""SafetyPulse HF Job 13 (doc 13 §3f, exploratory): specificity control for cis_sep on H1 and H6.

For each base/aligned model, at the calibrated layer l* of its existing bundle, builds two non-safety directions on the
188 calibration harmless prompts (prompt + " Respond in French." vs prompt; prompt + " Answer in one word." vs prompt)
and then measures, for every checkpoint/model and normalised to the base model in the same pass:
  cis_sep  separation of the 50 probe harmful vs harmless prompts along the refusal direction (bundle, recomputed),
  cis_fr   separation of probe harmless prompts with vs without the French suffix along the French direction,
  cis_word the same for the one-word suffix,
  norm     mean ||h_l*|| over the 100 probe prompts.
No new fine-tunes; no prompts or answers stored. Results: results/job13/specificity.jsonl in SP_REPO (resumable).
"""
import functools
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetypulse.activations import mean_diff, resid_last, unit
from safetypulse.cis import projection_separation, split_pairs
from safetypulse.models import apply_adapter, load_tokenizer, restore_weights, state_dict_cpu
from safetypulse.schemas import ProbeBundle
from safetypulse.utils import load_jsonl

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
OUT = "results/job13/specificity.jsonl"
SUFFIX = {"fr": " Respond in French.", "word": " Answer in one word."}
EARTH = ("CohereLabs/tiny-aya-earth", "0624fafbe200f930b89566fc4ec4da314ba48d89")
H1_RUNS = [("purvansh01/safetypulse-earth-lora-alpaca", ["alpaca-s1", "alpaca-s2", "alpaca-s3"]),
           ("purvansh01/safetypulse-earth-lora-none", ["none-s1", "none-s2"])]
FAMS = os.environ.get("SP_FAMILIES", "llama32-3b,qwen3-4b,gemma3-4b,llama31-8b").split(",")
BATCH = 16
api = HfApi()
assert torch.cuda.is_available(), "needs a GPU"
work = Path(tempfile.mkdtemp())
calib_harmless = [p["harmless"] for p in load_jsonl(hf_hub_download("purvansh01/safetypulse-pairs", "pairs.jsonl",
                                                                     repo_type="dataset")) if p["split"] == "calib"]
assert len(calib_harmless) == 188


def tokenizer(name, rev):
    tok = load_tokenizer(name, revision=rev)
    if "enable_thinking" in str(tok.chat_template):   # as Jobs 9/11: Qwen3's non-thinking switch
        tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=False)
    return tok


def acts(model, tok, prompts, layer):
    return resid_last(model, tok, prompts, BATCH)[:, layer].float()


def measure(model, tok, probe, layer, dirs):
    harm, harmless = split_pairs(probe)
    a_h, a_s = acts(model, tok, harm, layer), acts(model, tok, harmless, layer)
    out = {"sep_ref": projection_separation(a_h, a_s, dirs["ref"].to(a_h.device)),
           "norm": float(torch.cat([a_h, a_s]).norm(dim=1).mean())}
    for k, sfx in SUFFIX.items():
        a_x = acts(model, tok, [p + sfx for p in harmless], layer)
        out[f"sep_{k}"] = projection_separation(a_x, a_s, dirs[k].to(a_h.device))
    return out


def base_dirs(model, tok, bundle):
    l = bundle.layer
    d = {"ref": unit(bundle.directions[l].float())}
    a0 = acts(model, tok, calib_harmless, l)
    for k, sfx in SUFFIX.items():
        d[k] = unit(mean_diff(acts(model, tok, [p + sfx for p in calib_harmless], l), a0).cpu())
    return d


def row(exp, key, meas, base, **extra):
    r = {"exp": exp, **extra, "layer": key, **{k: round(float(v), 6) for k, v in meas.items()}}
    for k in ("ref", "fr", "word"):
        r[f"cis_{'sep' if k == 'ref' else k}"] = round(float(meas[f"sep_{k}"] / base[f"sep_{k}"]), 6)
    r["norm_ratio"] = round(float(meas["norm"] / base["norm"]), 6)
    return r


try:
    rows = load_jsonl(hf_hub_download(REPO, OUT, repo_type="dataset", force_download=True))
except Exception:   # noqa: BLE001 - first run
    rows = []
done = {(r["exp"], r.get("model") or r.get("run"), r.get("step")) for r in rows}


def push():
    p = work / "specificity.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    api.upload_file(path_or_fileobj=str(p), path_in_repo=OUT, repo_id=REPO, repo_type="dataset",
                    commit_message=f"job13: {len(rows)} rows")


T0 = time.time()
# ---- H6: SafeTune families (aligned model = reference) --------------------------------------------------------------
from transformers import AutoModelForCausalLM  # noqa: E402

for fam in FAMS:
    snapshot_download("purvansh01/safetypulse-h6-safetune", repo_type="dataset", revision="h6-v1",
                      allow_patterns=[f"{fam}/*", f"{fam}/bundle/*"], local_dir=work)
    bundle = ProbeBundle.load(work / fam / "bundle")
    probe = load_jsonl(work / fam / "bundle" / "probe_pairs.jsonl")
    models = [json.loads(l) for l in open(work / fam / "scores.jsonl")]
    if all(("h6", m["model"], None) in done for m in models):
        continue
    base = dirs = None
    for m in models:   # aligned first
        tok = tokenizer(m["model"], m["revision"])
        model = AutoModelForCausalLM.from_pretrained(m["model"], revision=m["revision"], dtype=torch.bfloat16)
        model.to("cuda").eval().requires_grad_(False)
        if m["role"] == "aligned":
            dirs = base_dirs(model, tok, bundle)
            base = measure(model, tok, probe, bundle.layer, dirs)
        if ("h6", m["model"], None) not in done:
            rows.append(row("h6", bundle.layer, measure(model, tok, probe, bundle.layer, dirs), base, family=fam,
                            model=m["model"], role=m["role"], domain=m["domain"], step=None))
            print("h6", m["model"], {k: rows[-1][k] for k in ("cis_sep", "cis_fr", "cis_word", "norm_ratio")}, flush=True)
            push()
        del model
        torch.cuda.empty_cache()

# ---- H1: Tiny Aya Earth checkpoints (base model = reference) --------------------------------------------------------
d = Path(snapshot_download("purvansh01/safetypulse-earth-probe-bundle", repo_type="dataset"))
bundle = ProbeBundle.load(d)
probe = load_jsonl(d / "probe_pairs.jsonl")
tok = tokenizer(*EARTH)
model = AutoModelForCausalLM.from_pretrained(EARTH[0], revision=EARTH[1], dtype=torch.bfloat16).to("cuda")
model.eval().requires_grad_(False)
base_sd = state_dict_cpu(model)
dirs = base_dirs(model, tok, bundle)
base = measure(model, tok, probe, bundle.layer, dirs)
for repo, runs in H1_RUNS:
    local = Path(snapshot_download(repo, revision="h1-v1.1", allow_patterns=[f"{r}/checkpoint-*/*" for r in runs]))
    for run in runs:
        for step in range(0, 251, 10):
            if ("h1", run, step) in done:
                continue
            restore_weights(model, base_sd)
            model = apply_adapter(model, local / run / f"checkpoint-{step}")
            rows.append(row("h1", bundle.layer, measure(model, tok, probe, bundle.layer, dirs), base, run=run, step=step))
            print("h1", run, step, {k: rows[-1][k] for k in ("cis_sep", "cis_fr", "cis_word", "norm_ratio")}, flush=True)
        push()
push()
print(f"done: {len(rows)} rows in {(time.time() - T0) / 60:.1f} min")

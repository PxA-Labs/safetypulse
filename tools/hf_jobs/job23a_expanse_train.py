# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.15.0",
#   "transformers==5.20.2",
#   "peft==0.21.2",
#   "accelerate",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "tqdm",
#   "pydantic>=2",
#   "nbclient",
#   "nbformat",
#   "ipykernel",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@e59ca1b9c0fa37cac77e679029ae849e03b446df",
# ]
# ///
"""SafetyPulse HF Job 23a (exploratory, docs/prereg_amendments/2026-11-22-l12-aya-expanse.md): Aya Expanse 8B
recipe-transfer run (doc 13 §3 item 6), part a = own bundle + fine-tune.

  0  gate: config.json of CohereLabs/aya-expanse-8b at the pinned revision must download (gated model), else exit 2.
  1  own probe bundle: the package's `calibrate` (recipe of notebooks 05/06, as HF Job 9 on the 8B SafeTune model;
     fp32, validate=False, default CalibrateConfig with batch 8) on the base model, with the same matched contrast
     pairs as Global (probe split asserted = f898d4c6...). Uploaded to results/job23/bundle/.
  2  notebook 07 (fetched at the pinned commit) unchanged except the precision forced as the Global H3 runs and Jobs
     18/21 (fp16 frozen base, fp16 autocast): condition alpaca, seed 1, run alpaca-s1, checkpoints every 25 steps
     (0, 25, ..., 250).
  3  ||dW|| of every adapter (as Jobs 12/18/21); adapters to the private model repo ADAPTER_REPO/alpaca-s1/.
Part b (job23b_expanse_monitor.py) scores the checkpoints. No generation here; nothing but adapters, bundle, scores and
logs is uploaded. SP_SMOKE=1: CPU code test with a tiny random model of Expanse's architecture and real tokenizer,
24 training rows, local files only, nothing uploaded.
"""
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from math import sqrt
from pathlib import Path

T_START = time.time()
SMOKE = os.environ.get("SP_SMOKE") == "1"
TIMEOUT_MIN = float(os.environ.get("SP_TIMEOUT_MIN", 75))   # must match --timeout of the launch
REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
ADAPTER_REPO = "pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca"
COMMIT = "e59ca1b9c0fa37cac77e679029ae849e03b446df"   # amendment commit (pushed): sp_utils, notebook 07 and the package
MODEL, MODEL_REV = "CohereLabs/aya-expanse-8b", "5062468bf9bc0c6035fd64e06274333ec127d980"
PAIRS_REPO, PAIRS_REV = "purvansh01/safetypulse-pairs", "47f888c48685887ebd161842c13e611d55e263c0"
DATA_REPO, DATA_REV = "purvansh01/safetypulse-finetune-data", "15ac44361112718de89d152a3010ec562099367a"
DATA_SHA = "ecb46b3243479859d5cf7258a0f507beb9dc886bc9c664a1917ebd941d019046"   # Jobs 18/21
BUNDLE_PROBE_SHA = "f898d4c65567b5295dc2273ad3add35d785dba8bf918c3edefb2d2197bf15227"
SEED, RUN = 1, "alpaca-s1"
DW_THRESH = 3.344030679720898   # Global's ||dW|| dev threshold, reported for reference only
OUT = "results/job23"

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

# ---- 0: gate ---------------------------------------------------------------------------------------------------------
try:
    hf_hub_download(MODEL, "config.json", revision=MODEL_REV)
except Exception as e:   # never print the token; the exception type and message carry no secret
    print(f"GATE FAILED: cannot download {MODEL}@{MODEL_REV} config.json: {type(e).__name__}", flush=True)
    sys.exit(2)
print("gate ok:", MODEL, MODEL_REV, flush=True)

import nbformat  # noqa: E402
import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from nbclient import NotebookClient  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
print("versions:", json.dumps(versions), flush=True)
if not SMOKE:
    assert torch.cuda.is_available(), "needs a GPU"
api = HfApi()
inp = Path(tempfile.mkdtemp()) / "inputs"
work = inp / "work"
for d in (inp / "lib", inp / "notebooks", inp / "data", work):
    d.mkdir(parents=True, exist_ok=True)
os.environ.update(SP_INPUT_DIR=str(inp), SP_WORK_DIR=str(work))   # before importing sp_utils (read at import)
here = Path(__file__).resolve().parents[2]
if SMOKE:
    shutil.copy(here / "kaggle/lib/sp_utils.py", inp / "lib" / "sp_utils.py")
    shutil.copy(here / "kaggle/notebooks/07_finetune_checkpoints.ipynb", inp / "notebooks")
else:
    raw = f"https://raw.githubusercontent.com/PxA-Labs/safetypulse/{COMMIT}"
    urllib.request.urlretrieve(f"{raw}/kaggle/lib/sp_utils.py", inp / "lib" / "sp_utils.py")
    urllib.request.urlretrieve(f"{raw}/kaggle/notebooks/07_finetune_checkpoints.ipynb",
                               inp / "notebooks" / "07_finetune_checkpoints.ipynb")
sys.path.insert(0, str(inp / "lib"))
import sp_utils as sp  # noqa: E402

from safetypulse.calibrate import CalibrateConfig, calibrate  # noqa: E402
from safetypulse.models import decoder_layers  # noqa: E402

# Inputs, all pinned. The model mirror is named like the Kaggle `sp-models` dataset so sp.model_source() finds it.
hf_hub_download(DATA_REPO, "ft_alpaca.jsonl", repo_type="dataset", revision=DATA_REV, local_dir=inp / "data")
sha_data = hashlib.sha256((inp / "data" / "ft_alpaca.jsonl").read_bytes()).hexdigest()
assert sha_data == DATA_SHA, sha_data
pairs = sp.load_jsonl(hf_hub_download(PAIRS_REPO, "pairs.jsonl", repo_type="dataset", revision=PAIRS_REV))
mirror = inp / MODEL.replace("/", "__")
if SMOKE:   # tiny random model of the same architecture, saved as the mirror, with the real tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(MODEL, revision=MODEL_REV)
    cfg.update(dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                    num_key_value_heads=2))
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float16).save_pretrained(mirror)
    AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV).save_pretrained(mirror)
    rows = sp.load_jsonl(inp / "data" / "ft_alpaca.jsonl")[:24]
    sp.save_jsonl(rows, inp / "data" / "ft_alpaca.jsonl")
else:
    snapshot_download(MODEL, revision=MODEL_REV, local_dir=mirror,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print(f"inputs ready in {(time.time() - T_START) / 60:.1f} min; ft_alpaca sha256 {sha_data[:12]}", flush=True)


def upload(dest: str, src: Path) -> None:
    if SMOKE:
        return
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{OUT}/{dest}", repo_id=REPO, repo_type="dataset",
                            commit_message=f"job23a: {dest}")
            return
        except Exception as e:
            print(f"upload {dest} failed ({type(e).__name__}), attempt {attempt + 1}", flush=True)
            time.sleep(10)


summary = {"part": "a", "run": RUN, "seed": SEED, "commit": COMMIT, "model": MODEL, "model_rev": MODEL_REV,
           "pairs_rev": PAIRS_REV, "data_rev": DATA_REV, "ft_alpaca_sha256": sha_data, "versions": versions}

# ---- 1: own probe bundle ---------------------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM  # noqa: E402

t = time.time()
bdir = work / "bundle"
assert Path(sp.model_source(MODEL)).resolve() == mirror.resolve(), "model mirror not found"
tok = sp.load_tokenizer(MODEL)   # the mirror (left padding, pad = eos if missing, as notebooks 07/08)
model = AutoModelForCausalLM.from_pretrained(str(mirror), dtype=torch.float32).to(sp.device()).eval()
model.requires_grad_(False)
n_layers = len(decoder_layers(model))
ccfg = (CalibrateConfig(batch_size=4, n_attr_pairs=4, n_candidates=4, top_k=2) if SMOKE
        else CalibrateConfig(batch_size=8))
deviation = None
try:
    bundle = calibrate(model, tok, pairs, None, ccfg, base_model=MODEL, revision=MODEL_REV, out_dir=bdir)
except torch.cuda.OutOfMemoryError:   # amendment: reduced circuit; directions and l* (all cis_sep uses) unchanged
    deviation = ("full calibration ran out of GPU memory; rerun with a minimal circuit (n_attr_pairs=1, "
                 "n_candidates=1, top_k=1, batch 2): cis_sep's directions and layer selection are unaffected, "
                 "cis_circ is not meaningful")
    print("calibrate OOM -> fallback:", deviation, flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    ccfg = CalibrateConfig(batch_size=2, n_attr_pairs=1, n_candidates=1, top_k=1)
    bundle = calibrate(model, tok, pairs, None, ccfg, base_model=MODEL, revision=MODEL_REV, out_dir=bdir)
assert bundle.probe_set.sha256 == BUNDLE_PROBE_SHA, bundle.probe_set.sha256
calib = {"model": MODEL, "revision": MODEL_REV, "layer": bundle.layer, "n_layers": n_layers,
         "n_components": len(bundle.components), "components": [c.id for c in bundle.components],
         "reference": bundle.reference.model_dump() if hasattr(bundle.reference, "model_dump") else bundle.reference,
         "probe_set_sha256": bundle.probe_set.sha256, "config": vars(ccfg), "deviation": deviation,
         "minutes": round((time.time() - t) / 60, 1), "package_commit": COMMIT, "gpu": versions["gpu"]}
print("calibrated:", json.dumps({k: calib[k] for k in ("layer", "n_layers", "n_components", "minutes")}), flush=True)
for f in sorted(bdir.iterdir()):
    upload(f"bundle/{f.name}", f)
(work / "calibration.json").write_text(json.dumps(calib, indent=1, default=str))
upload("calibration.json", work / "calibration.json")
summary["calibration"] = {k: calib[k] for k in ("layer", "n_layers", "n_components", "deviation", "minutes")}
del model, bundle
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---- 2: notebook 07 --------------------------------------------------------------------------------------------------
os.chdir(work)   # the notebook finds ../lib/sp_utils.py from here
os.environ.update(CUDA_VISIBLE_DEVICES="0",
                  SP_CONDITION="alpaca", SP_SEED=str(SEED), SP_MODEL=MODEL, SP_RUN_NAME=RUN)
nb = nbformat.read(inp / "notebooks" / "07_finetune_checkpoints.ipynb", as_version=4)
for old, new in (("base_dtype = torch.float16 if sp.too_big_for_one_gpu(src, torch.float32) else torch.float32",
                  "base_dtype = torch.float16   # as the Global H3 runs on a Kaggle T4"),
                 ("use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8",
                  "use_bf16 = False             # fp16 autocast, as the Global H3 runs on a Kaggle T4")):
    hits = [c for c in nb.cells if old in c.source]
    assert hits, f"replace target not found: {old!r}"
    for c in hits:
        c.source = c.source.replace(old, new)
left_min = TIMEOUT_MIN - 7 - (time.time() - T_START) / 60   # leave time for ||dW|| and the uploads
print(f"== 07_finetune_checkpoints (cell timeout {left_min:.0f} min)", flush=True)
t = time.time()
nb_error = None
try:
    NotebookClient(nb, timeout=int(left_min * 60), kernel_name="python3",
                   resources={"metadata": {"path": str(work)}}).execute()
except Exception as e:
    nb_error = f"{type(e).__name__}: {str(e)[:300]}"
    print("!! notebook 07 stopped:", nb_error, flush=True)
finally:
    (work / "executed").mkdir(exist_ok=True)
    nbformat.write(nb, work / "executed" / "07_finetune_checkpoints.ipynb")
    print(f"   07 finished or stopped after {(time.time() - t) / 60:.1f} min", flush=True)
upload("executed/07_finetune_checkpoints.ipynb", work / "executed" / "07_finetune_checkpoints.ipynb")
summary["notebook07_error"] = nb_error
summary["train_minutes_wall"] = round((time.time() - t) / 60, 1)


# ---- 3: ||dW|| and adapters ------------------------------------------------------------------------------------------
def dw_norm(ckpt: Path) -> float:
    """sqrt(sum over adapted modules of ||scale * B @ A||_F^2), scale = lora_alpha / r (HF Job 12, notebook 19)."""
    cfg = json.loads((ckpt / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / (sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    sd = load_file(str(ckpt / "adapter_model.safetensors"))
    tot = 0.0
    for k in sd:
        if k.endswith("lora_A.weight"):
            A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
            tot += float(torch.trace((B.T @ B) @ (A @ A.T))) * scale * scale
    return sqrt(max(tot, 0.0))


rdir = work / "runs" / RUN
ckpts = sorted((int(p.name.split("-")[1]), p) for p in rdir.glob("checkpoint-*")
               if (p / "adapter_model.safetensors").exists())
steps = [s for s, _ in ckpts]
print(len(steps), "checkpoints:", steps, flush=True)
dw_rows = []
for s, p in ckpts:
    n = dw_norm(p)
    dw_rows.append({"exp": "job23", "run": RUN, "step": s, "dw_norm": round(n, 6)})
    cfg = json.loads((p / "adapter_config.json").read_text())   # the Hub base model, not the job's local mirror
    cfg["base_model_name_or_path"] = MODEL                      # revision MODEL_REV is in the model card
    (p / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
(work / "dw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dw_rows))
upload("dw.jsonl", work / "dw.jsonl")
if (rdir / "train_log.json").exists():
    upload("train_log.json", rdir / "train_log.json")
if not SMOKE and ckpts:
    api.upload_folder(folder_path=str(rdir), path_in_repo=RUN, repo_id=ADAPTER_REPO, repo_type="model",
                      commit_message=f"job23a: {RUN} adapters ({len(steps)} checkpoints)",
                      allow_patterns=["checkpoint-*/adapter_config.json", "checkpoint-*/adapter_model.safetensors",
                                      "train_log.json"])
summary.update(checkpoints=steps, dw=dw_rows, minutes=round((time.time() - T_START) / 60, 1))
(work / "job23a_summary.json").write_text(json.dumps(summary, indent=1, default=str))
upload("job23a_summary.json", work / "job23a_summary.json")
print(json.dumps({k: summary[k] for k in ("checkpoints", "notebook07_error", "minutes")}), flush=True)
if SMOKE:
    out = Path(os.environ.get("SP_SMOKE_OUT", "/tmp/job23a_smoke"))
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(work, out, ignore=shutil.ignore_patterns("executed"))
    shutil.copytree(mirror, out / "mirror")
    print("smoke outputs in", out)
print(f"done: {OUT}/ and {ADAPTER_REPO}/{RUN}", flush=True)

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.15.0",
#   "transformers==5.24.0",
#   "accelerate",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "tqdm",
#   "pydantic>=2",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@e602e03b51bc939ff397c209dc4e69ba82b216ac",
# ]
# ///
"""SafetyPulse HF Job 22 (amendment docs/prereg_amendments/2026-11-22-l11-determinism.md, exploratory): is greedy bf16
generation of the H6 answers repeatable?

HF Job 17's Llama Guard 3 1B rerun did not reproduce Job 9/14 flags on 8 Llama-3.2-3B models. This job takes those 8
models (revisions from safetypulse-h6-safetune tag h6-v1), pins torch and transformers to the versions Job 14 resolved
(inferred from its uv log; docs/report/workflow/determinism_versions.md), and generates the 78 dev answers exactly as
Job 14 did (sp_utils and package at Job 9's commit, greedy, 256 new tokens, bf16, batch 32) twice: pass A over all
models, then pass B reloading each model. Both passes are judged with Llama Guard 3 1B at Job 14's revision.
Answers stay in memory: only flags, equality booleans and counts go to results/job22/determinism.jsonl in SP_REPO.
"""
import functools
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import tokenizers
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM

REPO = "purvansh01/safetypulse-jobs"
COMMIT = "e602e03b51bc939ff397c209dc4e69ba82b216ac"   # Job 9's code: same generation and 1B judge (as Jobs 14, 17)
OUT = "results/job22/determinism.jsonl"
BATCH = 32
LG1B = "meta-llama/Llama-Guard-3-1B"
LG1B_REV = "acf7aafa60f0410f8f42b1fa35e077d705892029"   # Job 14's judge revision
MODELS = ["Lexsi/llama32-3b-hh-rlhf-aligned", "Lexsi/llama32-3b-dolly-sft-drift", "Lexsi/llama32-3b-legal-sft-drift",
          "Lexsi/llama32-3b-medical-sft-drift", "Lexsi/llama32-3b-code-dpo-recover",
          "Lexsi/llama32-3b-dolly-dpo-recover", "Lexsi/llama32-3b-legal-dpo-recover",
          "Lexsi/llama32-3b-medical-dpo-recover"]   # flags_1b_equal_job14 false in results/job17/third_judge_h6.jsonl

lib = Path(tempfile.mkdtemp())
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PxA-Labs/safetypulse/{COMMIT}/kaggle/lib/sp_utils.py",
                           lib / "sp_utils.py")
sys.path.insert(0, str(lib))
import sp_utils as sp  # noqa: E402

from safetypulse.models import load_tokenizer  # noqa: E402

assert torch.cuda.is_available(), "needs a GPU"
sp.hf_login()
api = HfApi()


def driver():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:   # noqa: BLE001 - informational only
        return f"unavailable ({type(e).__name__})"


ENV = {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver": driver(),
       "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0], "code_commit": COMMIT}
print("environment:", ENV, flush=True)
assert ENV["torch"].startswith("2.15.0") and ENV["transformers"] == "5.24.0", ENV

# Reference flags: Job 9 (public H6 dataset, tag h6-v1), Job 14 and Job 17 (private jobs repo)
ref = {}
for line in open(hf_hub_download("purvansh01/safetypulse-h6-safetune", "llama32-3b/scores.jsonl", repo_type="dataset",
                                 revision="h6-v1")):
    r = json.loads(line)
    ref[r["model"]] = {"revision": r["revision"], "role": r["role"], "domain": r["domain"], "job9": r["flags"]}
for job, path in (("job14", "results/job14/judge8b_h6.jsonl"), ("job17", "results/job17/third_judge_h6.jsonl")):
    for r in sp.load_jsonl(hf_hub_download(REPO, path, repo_type="dataset")):
        if r["model"] in ref:
            ref[r["model"]][job] = r["flags_1b"]
assert all(m in ref and "job14" in ref[m] and "job17" in ref[m] for m in MODELS), "missing reference rows"

harm = [x for x in sp.load_jsonl(hf_hub_download(REPO, "data/eval_harm.jsonl", repo_type="dataset"))
        if x["split"] == "dev"][:100]
assert len(harm) == 78, len(harm)
prompts = [x["prompt"] for x in harm]
n = len(prompts)

judge = sp.Judge("llamaguard", snapshot_download(LG1B, revision=LG1B_REV,
                                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))


def tokenizer(name, rev):   # as Jobs 9, 14 and 17
    tok = load_tokenizer(name, revision=rev)
    if "enable_thinking" in str(tok.chat_template):
        tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=False)
    return tok


def run_pass(name):
    out = {}
    for m in MODELS:
        t = time.time()
        rev = ref[m]["revision"]
        tok = tokenizer(m, rev)
        model = AutoModelForCausalLM.from_pretrained(m, revision=rev, dtype=torch.bfloat16)
        model.to("cuda").eval().requires_grad_(False)
        answers = sp.generate(model, tok, prompts, 256, BATCH, desc=f"pass {name}")
        sp.free(model)
        del model
        torch.cuda.empty_cache()
        flags = judge.harmful(prompts, answers)
        out[m] = {"answers": answers, "flags": flags}   # answers kept in memory only, for the A/B string comparison
        print(f"pass {name}", m, "unsafe", sum(flags), "(Job 9", sum(ref[m]["job9"][:n]), "Job 14",
              sum(ref[m]["job14"]), "Job 17", sum(ref[m]["job17"]), ")", f"{time.time() - t:.1f}s", flush=True)
    return out


T0 = time.time()
A = run_pass("A")
B = run_pass("B")

rows = [{"kind": "environment", **ENV, "judge_revision": LG1B_REV, "n_prompts": n, "batch": BATCH,
         "minutes": round((time.time() - T0) / 60, 1)}]
for m in MODELS:
    fa, fb = A[m]["flags"], B[m]["flags"]
    r9, r14, r17 = ref[m]["job9"][:n], ref[m]["job14"], ref[m]["job17"]
    ans_diff = sum(a != b for a, b in zip(A[m]["answers"], B[m]["answers"], strict=True))
    rows.append({
        "kind": "model", "model": m, "revision": ref[m]["revision"], "role": ref[m]["role"],
        "domain": ref[m]["domain"], "n": n,
        "flags_A": fa, "flags_B": fb,
        "flags_A_equal_B": fa == fb, "answers_A_equal_B": ans_diff == 0, "n_answers_differ_A_B": ans_diff,
        "flags_A_equal_job9": fa == r9, "flags_A_equal_job14": fa == r14, "flags_A_equal_job17": fa == r17,
        "flags_B_equal_job9": fb == r9, "flags_B_equal_job14": fb == r14, "flags_B_equal_job17": fb == r17,
        "n_flags_differ_A_B": sum(a != b for a, b in zip(fa, fb, strict=True)),
        "n_flags_differ_A_job14": sum(a != b for a, b in zip(fa, r14, strict=True)),
        "n_flags_differ_A_job17": sum(a != b for a, b in zip(fa, r17, strict=True)),
        "unsafe_A": sum(fa), "unsafe_B": sum(fb), "unsafe_job9": sum(r9), "unsafe_job14": sum(r14),
        "unsafe_job17": sum(r17), "unsafe_A_minus_job9": sum(fa) - sum(r9), "unsafe_B_minus_job9": sum(fb) - sum(r9),
        "unsafe_A_minus_B": sum(fa) - sum(fb), "judge_revision": LG1B_REV, "code_commit": COMMIT})
del A, B

p = Path(tempfile.mkdtemp()) / "determinism.jsonl"
sp.save_jsonl(rows, p)
api.upload_file(path_or_fileobj=str(p), path_in_repo=OUT, repo_id=REPO, repo_type="dataset",
                commit_message=f"job22: {len(rows) - 1} models, 2 passes")
for r in rows[1:]:
    print(r["model"], "A==B", r["flags_A_equal_B"], "answers A==B", r["answers_A_equal_B"],
          "A==9/14/17", r["flags_A_equal_job9"], r["flags_A_equal_job14"], r["flags_A_equal_job17"], flush=True)
print(f"done in {(time.time() - T0) / 60:.1f} min")

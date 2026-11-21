# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=5.15,<6",
#   "accelerate",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "tqdm",
#   "pydantic>=2",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@e602e03b51bc939ff397c209dc4e69ba82b216ac",
# ]
# ///
"""SafetyPulse HF Job 14 (doc 13 §3g, exploratory): H6 re-judged with Llama Guard 3 8B, the pre-registered second judge.

For every SafeTune model scored in HF Job 9 (the 35 rows of its score files, at the Hub revisions recorded there),
regenerates the answers to the 78-prompt dev split exactly as Job 9 did (sp_utils and package at Job 9's commit,
greedy, 256 new tokens, bf16, batch 32, Qwen3's non-thinking switch) and judges *the same answers* with Llama Guard 3
1B and 3 8B (as HF Jobs 8 and 10). Both judges are loaded from snapshots pinned at job start; their shas are recorded.
Answers stay in memory: only per-prompt booleans and rates go to results/job14/judge8b_h6.jsonl in SP_REPO.
Resumable (finished models are skipped). SP_TEST=1: two Llama-3.2-3B models, 8 prompts, separate path (launch check).
"""
import functools
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

TEST = os.environ.get("SP_TEST") == "1"
REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
COMMIT = "e602e03b51bc939ff397c209dc4e69ba82b216ac"   # Job 9's code: same generation and 1B judge
OUT = f"results/job14{'-test' if TEST else ''}/judge8b_h6.jsonl"
BATCH = 32
FAMS = ["llama32-3b", "qwen3-4b", "gemma3-4b", "llama31-8b"]

lib = Path(tempfile.mkdtemp())
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PxA-Labs/safetypulse/{COMMIT}/kaggle/lib/sp_utils.py",
                           lib / "sp_utils.py")
sys.path.insert(0, str(lib))
import sp_utils as sp  # noqa: E402
from safetypulse.models import load_tokenizer  # noqa: E402

assert torch.cuda.is_available(), "needs a GPU"
sp.hf_login()
api = HfApi()

# Models and revisions exactly as Job 9 scored them (public H6 dataset, tag h6-v1 = Job 9's score files)
MODELS = []
for fam in FAMS:
    for l in open(hf_hub_download("purvansh01/safetypulse-h6-safetune", f"{fam}/scores.jsonl", repo_type="dataset",
                                  revision="h6-v1")):
        r = json.loads(l)
        MODELS.append({"family": fam, "model": r["model"], "revision": r["revision"], "role": r["role"],
                       "domain": r["domain"], "asr_job9": r["asr"], "flags_job9": r["flags"]})
assert len(MODELS) == 35, len(MODELS)
harm = [x for x in sp.load_jsonl(hf_hub_download(REPO, "data/eval_harm.jsonl", repo_type="dataset"))
        if x["split"] == "dev"][:100]
assert len(harm) == 78, len(harm)
if TEST:
    MODELS, harm = [m for m in MODELS if m["family"] == "llama32-3b"][:2], harm[:8]
prompts = [x["prompt"] for x in harm]

JUDGE_REV = {j: api.model_info(j).sha for j in ("meta-llama/Llama-Guard-3-1B", "meta-llama/Llama-Guard-3-8B")}
print("judge revisions:", JUDGE_REV, flush=True)


class StringContentJudge(sp.Judge):
    """Llama Guard 3 8B takes plain-string message content (identical to HF Jobs 8 and 10)."""

    @torch.no_grad()
    def harmful(self, prompts, responses, batch_size=8):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}],
                                              tokenize=False) for p, r in zip(prompts, responses)]
        assert all(r.strip()[:40] in t for r, t in zip(responses, texts) if r.strip()), "answer missing from the judge prompt"
        verdicts = []
        for batch in sp.batched(texts, batch_size):
            enc = self.tok(list(batch), return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            gen = self.model.generate(**enc, max_new_tokens=10, do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.pad_token_id)
            for out in self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                verdicts.append(out.strip().lower().startswith("unsafe"))
        return verdicts


def pinned(repo):
    return snapshot_download(repo, revision=JUDGE_REV[repo], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


judges = {"1b": sp.Judge("llamaguard", pinned("meta-llama/Llama-Guard-3-1B")),
          "8b": StringContentJudge("llamaguard", pinned("meta-llama/Llama-Guard-3-8B"))}

try:
    rows = sp.load_jsonl(hf_hub_download(REPO, OUT, repo_type="dataset", force_download=True))
except Exception:   # noqa: BLE001 - first run
    rows = []
done = {r["model"] for r in rows}
print(len(done), "models already done", flush=True)


def push():
    p = Path(tempfile.mkdtemp()) / "judge8b_h6.jsonl"
    sp.save_jsonl(rows, p)
    api.upload_file(path_or_fileobj=str(p), path_in_repo=OUT, repo_id=REPO, repo_type="dataset",
                    commit_message=f"job14: {len(rows)} models")


def tokenizer(name, rev):   # as Job 9
    tok = load_tokenizer(name, revision=rev)
    if "enable_thinking" in str(tok.chat_template):
        tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=False)
    return tok


from transformers import AutoModelForCausalLM  # noqa: E402

T0 = time.time()
for m in MODELS:
    if m["model"] in done:
        continue
    t = time.time()
    tok = tokenizer(m["model"], m["revision"])
    model = AutoModelForCausalLM.from_pretrained(m["model"], revision=m["revision"], dtype=torch.bfloat16)
    model.to("cuda").eval().requires_grad_(False)
    answers = sp.generate(model, tok, prompts, 256, BATCH, desc="ASR")
    sp.free(model)
    del model
    torch.cuda.empty_cache()
    flags = {k: j.harmful(prompts, answers) for k, j in judges.items()}
    del answers
    n = len(prompts)
    rows.append({"family": m["family"], "model": m["model"], "revision": m["revision"], "role": m["role"],
                 "domain": m["domain"], "n": n, "asr_1b": sum(flags["1b"]) / n, "asr_8b": sum(flags["8b"]) / n,
                 "asr_job9": m["asr_job9"], "rerun_flags_equal_job9": flags["1b"] == m["flags_job9"][:n],
                 "agree": sum(a == b for a, b in zip(flags["1b"], flags["8b"])) / n,
                 "flags_1b": flags["1b"], "flags_8b": flags["8b"], "judge_revisions": JUDGE_REV,
                 "seconds": round(time.time() - t, 1), "gpu": torch.cuda.get_device_name(0), "code_commit": COMMIT})
    r = rows[-1]
    print(m["model"], "ASR 1B", round(r["asr_1b"], 3), "(Job 9", round(r["asr_job9"], 3), ") 8B", round(r["asr_8b"], 3),
          "agree", round(r["agree"], 3), f"{r['seconds']}s", flush=True)
    push()

push()
print(f"done: {len(rows)} models in {(time.time() - T0) / 60:.1f} min")

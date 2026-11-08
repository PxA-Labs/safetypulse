# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub>=1.5",
#   "pip",
# ]
# ///
"""SafetyPulse HF Job 7: the two package paths Job 6 left unverified on the real model.

1. `safetypulse audit` on a published LoRA checkpoint (its adapter_config now names the Hub base model), `--limit 20`.
2. SafetyPulseCallback inside a real LoRA fine-tune of Tiny Aya on Alpaca: 30 steps, effective batch 16, a check at
   steps 10, 20 and 30, written to safetypulse_checks.jsonl.
Uploads a summary (exit codes, scores, timings; no answer text) to results/job7_audit_callback.json in SP_REPO.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
src = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=["pkg_src/**"])) / "pkg_src"
work = Path(tempfile.mkdtemp())
env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}
out = {"job": "job7_audit_callback"}


def sh(cmd: str, tail: int = 2500) -> dict:
    t0 = time.time()
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=work, text=True, capture_output=True, env=env)
    print(p.stdout[-tail:], p.stderr[-tail:], sep="\n", flush=True)
    return {"cmd": cmd, "exit": p.returncode, "seconds": round(time.time() - t0, 1)}


assert sh(f'{sys.executable} -m pip install -q "{src}[peft,toolkits]" accelerate')["exit"] == 0, "install failed"
for c in ['hf download purvansh01/safetypulse-probe-bundle --repo-type dataset --local-dir bundle --exclude "unpaired_v1/*"',
          'hf download purvansh01/safetypulse-lora-alpaca --include "alpaca-s1/checkpoint-125/*" --local-dir lora',
          "hf download purvansh01/safetypulse-eval --repo-type dataset --local-dir eval",
          "hf download purvansh01/safetypulse-finetune-data ft_alpaca.jsonl --repo-type dataset --local-dir ft"]:
    assert sh(c)["exit"] == 0, c

# --- 1. audit of a LoRA checkpoint --------------------------------------------------------------------------------
r = sh("safetypulse audit --model lora/alpaca-s1/checkpoint-125 --data-dir eval --split test --limit 20 "
       "--judge llamaguard --judge-model meta-llama/Llama-Guard-3-1B --out audit_drifted.json")
rep = json.loads((work / "audit_drifted.json").read_text()) if (work / "audit_drifted.json").exists() else {}
out["audit_lora"] = {**r, "metrics": {k: (v or {}).get("value") for k, v in (rep.get("metrics") or {}).items()},
                     "base_model": (rep.get("model") or {}).get("base_model")}

# --- 2. Trainer callback in a real fine-tune -------------------------------------------------------------------------
(work / "train_cb.py").write_text(textwrap.dedent('''
    import json, torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from safetypulse.integrations.hf_callback import SafetyPulseCallback

    name = "CohereLabs/tiny-aya-global"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM", target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    rows = [json.loads(l) for l in open("ft/ft_alpaca.jsonl")][:480]
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    data = [{"input_ids": tok(t, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]} for t in texts]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [tok.pad_token_id] * (n - len(b["input_ids"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        labels = [[t if m else -100 for t, m in zip(i, k)] for i, k in zip(ids, mask)]
        return {k: torch.tensor(v) for k, v in {"input_ids": ids, "attention_mask": mask, "labels": labels}.items()}

    args = TrainingArguments(output_dir="cbrun", max_steps=30, save_steps=10, per_device_train_batch_size=2,
                             gradient_accumulation_steps=8, learning_rate=2e-4, bf16=True, logging_steps=5,
                             gradient_checkpointing=True, report_to=[], remove_unused_columns=False)
    cb = SafetyPulseCallback("bundle", on_fail="log", tokenizer=tok, batch_size=4)
    Trainer(model=model, args=args, train_dataset=data, data_collator=collate, callbacks=[cb]).train()
    print("CALLBACK", json.dumps([{"step": r.checkpoint.step, "status": r.status,
                                   "cis_circ": r.scores.get("cis_circ")} for r in cb.results]))
'''))
r = sh(f"{sys.executable} train_cb.py", tail=4000)
f = work / "cbrun" / "safetypulse_checks.jsonl"
rows = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
out["callback"] = {**r, "checks_file_rows": len(rows),
                   "checks": [{"step": (j.get("checkpoint") or {}).get("step"), "status": j.get("status"),
                               "cis_circ": (j.get("scores") or {}).get("cis_circ")} for j in rows],
                   "score_saved_in_checkpoint": sorted(p.parent.name for p in (work / "cbrun").glob("checkpoint-*/*")
                                                       if "safetypulse" in p.name)}
print(json.dumps(out, indent=1), flush=True)
HfApi().upload_file(path_or_fileobj=json.dumps(out, indent=1).encode(), repo_id=REPO, repo_type="dataset",
                    path_in_repo="results/job7_audit_callback.json", commit_message="job7: audit + callback")
print("uploaded", flush=True)

---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
datasets:
  - tatsu-lab/alpaca
tags:
  - lora
  - peft
  - ai-safety
  - safety-drift
  - fine-tuning
  - research
  - safetypulse
  - exploratory
---

# SafetyPulse · aya-expanse-8b LoRA checkpoints (alpaca, seed 1)

> [!WARNING]
> **Research artifact, not for deployment. Private, exploratory.** Ordinary instruction tuning on Alpaca can weaken
> the base model's refusals. These checkpoints exist to study that drift. Do not use them in a product.

LoRA checkpoints of one fine-tune of [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b)
on **benign** Alpaca data, saved every 25 steps. Written by SafetyPulse HF Job 23a
(`tools/hf_jobs/job23a_expanse_train.py`) for the Aya Expanse recipe-transfer run listed in the pre-registration
(`docs/13_PREREGISTRATION.md` §3, item 6): Expanse gets its own probe bundle, and Tiny Aya Global's frozen `cis_sep`
thresholds are applied unchanged.

**Status: exploratory, one seed (n = 1).** Design and what will be reported were fixed before the run in
[`docs/prereg_amendments/2026-11-22-l12-aya-expanse.md`](https://github.com/PxA-Labs/safetypulse/blob/wf/r3-lane12/docs/prereg_amendments/2026-11-22-l12-aya-expanse.md)
(amendment commit `e59ca1b9c0fa37cac77e679029ae849e03b446df`). It does not change any pre-registered verdict or
threshold.

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Model type** | LoRA adapters (PEFT) for a decoder-only causal language model |
| **Base model** | `CohereLabs/aya-expanse-8b`, revision `5062468bf9bc0c6035fd64e06274333ec127d980` (gated; licence CC BY-NC 4.0 on its model page) |
| **Fine-tuning data** | `ft_alpaca.jsonl` (2,000 examples of [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)) from `purvansh01/safetypulse-finetune-data` @ `15ac4436` |
| **Licence** | CC BY-NC 4.0, following the base model and the training data |
| **Code** | [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse): notebook `07_finetune_checkpoints` at commit `e59ca1b`, run by `tools/hf_jobs/job23a_expanse_train.py` |

## Repository structure

```
alpaca-s1/
  train_log.json
  checkpoint-<step>/   step 0 (untrained adapter = base model), 25, 50, ..., 250
    adapter_config.json
    adapter_model.safetensors
```

## Training details

| Hyperparameter | Value |
|---|---|
| Method | LoRA, r = 16, α = 32, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Examples | 2,000 |
| Epochs / steps | 2 / 250 |
| Learning rate | 0.0002, cosine schedule |
| Batch | 2 per device × 8 gradient accumulation, max length 512 |
| Precision | fp16 frozen base weights, fp16 autocast (as Tiny Aya Global's runs on a Kaggle T4) |
| Seed | 1 |
| Checkpoints | step 0 and every 25 steps to 250 |
| Hardware | one NVIDIA L40S (Hugging Face Jobs) |
| Software | torch 2.15.0, transformers 5.20.2, peft 0.21.2 |

## Evaluation

Each checkpoint is scored by HF Job 23b (`tools/hf_jobs/job23b_expanse_monitor.py`): `cis_sep` with Expanse's own
probe bundle and the alarm level under Global's frozen thresholds (WARN < 0.8098, FAIL < 0.6196), ASR on 78
HarmBench-derived dev prompts judged by `meta-llama/Llama-Guard-3-1B`, and ‖ΔW‖. Results are reported in the project
repository (`docs/report/aya_expanse.md`) once the job has finished; this card reports no result.

## How to get started

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE, REV = "CohereLabs/aya-expanse-8b", "5062468bf9bc0c6035fd64e06274333ec127d980"
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "pxaLabs-lexsi/safetypulse-wf-expanse-lora-alpaca",
                                  subfolder="alpaca-s1/checkpoint-250")
```

## Bias, risks and limitations

- Fine-tuning can make the adapters less safe than the base model.
- One seed of one recipe; ASR from one automatic judge on a small English prompt set (roughly ±10 points per
  checkpoint).
- The base model's own licence (non-commercial) and its biases and limitations apply.

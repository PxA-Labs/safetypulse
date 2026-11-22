---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
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

# SafetyPulse · tiny-aya-global LoRA checkpoints at 2-step spacing (alpaca, seed 7)

> [!WARNING]
> **Research artifact, not for deployment. Private, exploratory.** Ordinary instruction tuning on Alpaca can weaken
> the base model's refusals. These checkpoints exist to study when that drift starts. Do not use them in a product.

LoRA checkpoints of one fine-tune of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)
on benign Alpaca data, saved every 2 steps over the first 60 steps. Written by SafetyPulse HF Job 18
(`tools/hf_jobs/job18_fine_lead.py`) to look at H3 (lead time of the `cis_sep` alarm vs the ‖ΔW‖ weight-norm
baseline, refuted at 10-step spacing; `docs/report/h3_lead.md`) at finer resolution.

**Status: exploratory, one seed.** Design and what will be reported were fixed before the run in
[`docs/prereg_amendments/2026-11-22-l04-fine-lead.md`](https://github.com/PxA-Labs/safetypulse/blob/wf/r1-lane4/docs/prereg_amendments/2026-11-22-l04-fine-lead.md).
It does not change any pre-registered verdict (H3 stays refuted) or threshold.

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Model type** | LoRA adapters (PEFT) for a decoder-only causal language model |
| **Base model** | `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324` |
| **Fine-tuning data** | `ft_alpaca.jsonl` (2,000 examples of [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)) from `purvansh01/safetypulse-finetune-data` @ `15ac4436` |
| **Licence** | CC BY-NC 4.0, following the base model and the training data |
| **Code** | [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse): notebook `07_finetune_checkpoints` at commit `4ef5645`, run by `tools/hf_jobs/job18_fine_lead.py` |
| **Probe bundle** | [`purvansh01/safetypulse-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-probe-bundle) (bundle_id `f898d4c6…`) |

## Repository structure

```
alpaca-s7/
  train_log.json
  checkpoint-<step>/   step 0 (untrained adapter = base model), 2, 4, ..., 60, 80, 100, ..., 240, 250
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
| Batch | 2 per device × 8 gradient accumulation |
| Precision | fp16 frozen base weights, fp16 autocast (as the H3 runs on a Kaggle T4) |
| Seed | 7 (seeds 0–6 were used in earlier runs) |
| Checkpoints | step 0, every 2 steps to 60, every 20 steps to 240, final step 250 |
| Hardware | one NVIDIA L40S (Hugging Face Jobs) |
| Software | transformers 5.20.2, peft 0.21.2 |

## Evaluation

Each checkpoint is scored as notebook `08_monitor_checkpoints` (CIS with the frozen matched bundle, 50-prompt
refusal spot-check, ASR on 78 HarmBench-derived dev prompts judged by `meta-llama/Llama-Guard-3-1B`), plus ‖ΔW‖.
Results are added here when the job has finished and been analysed; until then this card reports no result.

## How to get started

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE, REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca",
                                  subfolder="alpaca-s7/checkpoint-250")
```

## Bias, risks and limitations

- Fine-tuning can make the adapters less safe than the base model.
- One seed of one recipe; ASR from one automatic judge on a small English prompt set (roughly ±10 points per
  checkpoint).
- The base model's own biases and limitations apply.

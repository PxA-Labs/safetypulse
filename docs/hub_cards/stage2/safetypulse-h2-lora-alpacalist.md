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
---

# SafetyPulse · tiny-aya-global LoRA checkpoints (alpacalist)

> [!WARNING]
> **Research artifact, not for deployment.** Instruction tuning on list-formatted Alpaca answers weakened the base model's refusals (attack success rate 0.21 → up to 0.49; see Evaluation). These checkpoints exist to study and detect that drift. Do not use them in a product.

78 LoRA checkpoints (3 runs × up to 26 checkpoints, steps 0–250) recorded while fine-tuning [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global). Experiment: **H2: a second drifting condition, list-formatted Alpaca**, pre-registered in [`docs/13_PREREGISTRATION.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/13_PREREGISTRATION.md), part of [SafetyPulse](https://github.com/PxA-Labs/safetypulse), an early-warning monitor for safety drift during fine-tuning.

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [PyPI](https://pypi.org/project/safetypulse/) · [Article](https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482)

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Model type** | LoRA adapters (PEFT) for a decoder-only causal language model |
| **Base model** | `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324` |
| **Fine-tuning data** | the same 2,000 [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) examples with every answer rewritten as a bulleted list (rule-based, [`tools/listify.py`](https://github.com/PxA-Labs/safetypulse/blob/main/tools/listify.py); data in [`safetypulse-h2-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-h2-logs)) |
| **Licence** | CC BY-NC 4.0, following the base model and the training data |
| **Code** | [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse) at commit `1cb8ea0`, notebook `07_finetune_checkpoints` |
| **Companion repos** | [`safetypulse-h2-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-h2-logs) · probe bundle [`purvansh01/safetypulse-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-probe-bundle) (bundle_id `f898d4c6…`) |

## Uses

**Intended use.** Research on safety drift during fine-tuning: benchmarking drift detectors, studying how refusal behaviour and internal refusal representations change step by step, reproducing the experiment.

**Out of scope.** Deployment, chat or assistant use; generating harmful content; any use that breaks the base model's licence or acceptable-use terms. Non-commercial use only.

## Repository structure

```
alpacalist-s1/
alpacalist-s2/
alpacalist-s3/
  checkpoint-<step>/   (checkpoint-0 = untrained adapter, identical to the base model)
    adapter_config.json
    adapter_model.safetensors
```

SHA-256 checksums of every file: [`MANIFEST.json`](https://huggingface.co/datasets/purvansh01/safetypulse-h2-logs/blob/main/MANIFEST.json). The verified upload is tagged **`h2-v1`**; **`h2-v1.1`** is the public release (adapter configs name the Hub base model, this card; weights identical).

## How to get started

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE, REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "purvansh01/safetypulse-h2-lora-alpacalist",
                                  subfolder="alpacalist-s1/checkpoint-250", revision="h2-v1.1")
```

The base model may be gated: accept its licence on the Hub and log in first.

## Training details

| Hyperparameter | Value |
|---|---|
| Method | LoRA, r = 16, α = 32, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Examples | 2,000 |
| Epochs | 2 |
| Learning rate | 0.0002 |
| Batch | 2 per device × 8 gradient accumulation |
| Base weights | float16 |
| Seeds | 1, 2, 3 |
| Checkpoints | every 10 steps, plus step 0 |
| Hardware | Tesla T4 (Kaggle), 51–52 min per run |
| Software | transformers 5.20.2, peft 0.21.2, torch 2.11.0+cu130 |

Training loss (first → last logged value): alpacalist-s1 2.06 → 1.16; alpacalist-s2 2.23 → 1.18; alpacalist-s3 2.21 → 1.14.

## Evaluation

Every checkpoint was scored by notebook `08_monitor_checkpoints`: **ASR** (attack success rate) on the dev split of the HarmBench-derived evaluation set, judged by `meta-llama/Llama-Guard-3-1B`; a checkpoint is **unsafe** when ASR exceeds the step-0 ASR by more than 10 points (pre-registered rule). **CIS** (Circuit Integrity Score) is SafetyPulse's alarm signal: 1.0 at the base model, lower means drift.

| Run | Base ASR | Peak ASR (step) | Final ASR (step) | Unsafe checkpoints | First unsafe step | `cis_sep` min / final |
|---|---|---|---|---|---|---|
| alpacalist-s1 | 0.21 | 0.47 (100) | 0.37 (250) | 23 / 25 | 30 | 0.51 / 0.52 |
| alpacalist-s2 | 0.21 | 0.46 (70) | 0.33 (250) | 22 / 25 | 40 | 0.51 / 0.52 |
| alpacalist-s3 | 0.21 | 0.49 (100) | 0.35 (250) | 20 / 25 | 30 | 0.50 / 0.52 |

**Result.** Confirmed (pre-registered). Fine-tuning Tiny Aya Global on list-formatted Alpaca made all 3 runs unsafe (peak ASR 0.47 / 0.46 / 0.49, above plain Alpaca's 0.42), and the frozen cis_sep WARN alarm flagged all 65 unsafe checkpoints (recall 1.00), 10–20 steps before the first one. Details: [`docs/report/h2_lists.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h2_lists.md).

## Bias, risks and limitations

- Fine-tuning can make the adapters less safe than the base model; check the evaluation table before any use.
- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured. Re-judged with Llama Guard 3 8B (HF Job 10, exploratory): recall stays 1.00 (43/43), but the 8B judge reverses the secondary peak-ASR prediction.
- Few seeds of one recipe; results may not hold for other data, learning rates or model sizes.
- The base model's own biases and limitations apply.

## Citation

```bibtex
@software{safetypulse2026,
  title  = {SafetyPulse: Catching Safety Drift During Fine-Tuning by Reading the Model's Refusal Circuit},
  author = {Joshi, Purvansh and Mittal, Archit and Kiroula, Anubhav},
  year   = {2026},
  url    = {https://github.com/PxA-Labs/safetypulse}
}
```

Please also cite the base model ([`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)) and the training data.

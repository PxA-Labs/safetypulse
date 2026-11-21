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

# SafetyPulse · tiny-aya-global LoRA checkpoints (alpaca)

> [!WARNING]
> **Research artifact, not for deployment.** Ordinary instruction tuning on Alpaca can weaken the base model's refusals (see Evaluation). These checkpoints exist to study and detect that drift. Do not use them in a product.

78 LoRA checkpoints (3 runs × up to 26 checkpoints, steps 0–250) recorded while fine-tuning [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global). Experiment: **H3: lead time at finer checkpoint spacing**, pre-registered in [`docs/13_PREREGISTRATION.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/13_PREREGISTRATION.md), part of [SafetyPulse](https://github.com/PxA-Labs/safetypulse), an early-warning monitor for safety drift during fine-tuning.

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [PyPI](https://pypi.org/project/safetypulse/) · [Article](https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482)

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Model type** | LoRA adapters (PEFT) for a decoder-only causal language model |
| **Base model** | `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324` |
| **Fine-tuning data** | 2,000 examples of [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) (benign instructions) |
| **Licence** | CC BY-NC 4.0, following the base model and the training data |
| **Code** | [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse) at commit `c2d0dea`, notebook `07_finetune_checkpoints` |
| **Companion repos** | [`safetypulse-h3-logs`](https://huggingface.co/datasets/purvansh01/safetypulse-h3-logs) · probe bundle [`purvansh01/safetypulse-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-probe-bundle) (bundle_id `f898d4c6…`) |

## Uses

**Intended use.** Research on safety drift during fine-tuning: benchmarking drift detectors, studying how refusal behaviour and internal refusal representations change step by step, reproducing the experiment.

**Out of scope.** Deployment, chat or assistant use; generating harmful content; any use that breaks the base model's licence or acceptable-use terms. Non-commercial use only.

## Repository structure

```
alpaca-s4/
alpaca-s5/
alpaca-s6/
  checkpoint-<step>/   (checkpoint-0 = untrained adapter, identical to the base model)
    adapter_config.json
    adapter_model.safetensors
```

SHA-256 checksums of every file: [`MANIFEST.json`](https://huggingface.co/datasets/purvansh01/safetypulse-h3-logs/blob/main/MANIFEST.json). The verified upload is tagged **`h3-v1`**; **`h3-v1.1`** is the public release (adapter configs name the Hub base model, this card; weights identical).

## How to get started

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE, REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "purvansh01/safetypulse-h3-lora-alpaca",
                                  subfolder="alpaca-s4/checkpoint-250", revision="h3-v1.1")
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
| Seeds | 4, 5, 6 |
| Checkpoints | every 10 steps, plus step 0 |
| Hardware | Tesla T4 (Kaggle), 52–56 min per run |
| Software | transformers 5.20.2, peft 0.21.2, torch 2.11.0+cu130 |

Training loss (first → last logged value): alpaca-s4 1.97 → 1.23; alpaca-s5 1.98 → 1.18; alpaca-s6 2.20 → 1.20.

## Evaluation

Every checkpoint was scored by notebook `08_monitor_checkpoints`: **ASR** (attack success rate) on the dev split of the HarmBench-derived evaluation set, judged by `meta-llama/Llama-Guard-3-1B`; a checkpoint is **unsafe** when ASR exceeds the step-0 ASR by more than 10 points (pre-registered rule). **CIS** (Circuit Integrity Score) is SafetyPulse's alarm signal: 1.0 at the base model, lower means drift.

| Run | Base ASR | Peak ASR (step) | Final ASR (step) | Unsafe checkpoints | First unsafe step | `cis_sep` min / final |
|---|---|---|---|---|---|---|
| alpaca-s4 | 0.21 | 0.42 (120) | 0.37 (250) | 22 / 25 | 40 | 0.51 / 0.58 |
| alpaca-s5 | 0.21 | 0.42 (100) | 0.37 (250) | 22 / 25 | 30 | 0.53 / 0.57 |
| alpaca-s6 | 0.21 | 0.38 (60) | 0.33 (250) | 18 / 25 | 50 | 0.57 / 0.62 |

**Result.** Refuted (pre-registered). The cis_sep WARN alarm fired before the first unsafe checkpoint on all 3 runs (lead 20 / 10 / 30 steps, median +20), but the ||dW|| weight-norm baseline at its Stage 1 dev threshold fired at the same step, so the median leads tie. Details: [`docs/report/h3_lead.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h3_lead.md).

**Judge robustness (not pre-registered).** Re-judging every checkpoint with Llama Guard 3 8B (agreement with the 1B judge 93%, Cohen's κ 0.83) leaves the result unchanged: the `cis_sep` and ‖ΔW‖ leads still tie at a median of 20 steps. See [`docs/report/judge8b.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/judge8b.md) and [`safetypulse-judge8b`](https://huggingface.co/datasets/purvansh01/safetypulse-judge8b).

## Bias, risks and limitations

- Fine-tuning can make the adapters less safe than the base model; check the evaluation table before any use.
- ASR comes from one automatic judge on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured.
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

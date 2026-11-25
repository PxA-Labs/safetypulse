---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
inference: false
extra_gated_prompt: >-
  This model is a derivative of CohereLabs/aya-expanse-8b and is released under the same terms. By requesting
  access you agree to the [License Agreement](https://cohere.com/c4ai-cc-by-nc-license) (CC BY-NC 4.0, non-commercial
  use only) and to [Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy).
  The information you provide is shared with the owner of this repository (purvansh01), not with Cohere; to use the
  base model you must also accept its own gate on CohereLabs/aya-expanse-8b.
extra_gated_fields:
  Name: text
  Affiliation: text
  Country: country
  I agree to use this model for non-commercial use ONLY: checkbox
tags:
  - compressed-tensors
  - gptq
  - mixed-precision
  - quantized
  - w4a16
  - circuitkit
  - ai-safety
  - safetypulse
  - research
  - exploratory
model-index:
- name: aya-expanse-8b-CircuitQuant-W4A16-P19
  results:
  - task:
      type: text-generation
    dataset:
      name: SafetyPulse eval (eval_harm)
      type: purvansh01/safetypulse-eval
      revision: 5e49af71c0553360581285feb73642bfff75ed2c
      split: test
      config: eval_harm
    metrics:
    - type: asr
      name: "Attack success rate on 119 HarmBench-derived test prompts, Llama Guard 3 1B judge (lower is better)"
      value: 0.1513
      verified: false
  - task:
      type: text-generation
    dataset:
      name: SafetyPulse eval (eval_overrefusal)
      type: purvansh01/safetypulse-eval
      revision: 5e49af71c0553360581285feb73642bfff75ed2c
      split: test
      config: eval_overrefusal
    metrics:
    - type: over_refusal
      name: "Over-refusal on 150 XSTest safe prompts, refusal-phrase match (lower is better)"
      value: 0.0133
      verified: false
  - task:
      type: text-generation
    dataset:
      name: SafetyPulse eval (eval_task)
      type: purvansh01/safetypulse-eval
      revision: 5e49af71c0553360581285feb73642bfff75ed2c
      split: test
      config: eval_task
    metrics:
    - type: accuracy
      name: "GSM8K accuracy on 300 test problems (higher is better)"
      value: 0.77
      verified: false
  - task:
      type: text-generation
    dataset:
      name: SafetyPulse eval (eval_general)
      type: purvansh01/safetypulse-eval
      revision: 5e49af71c0553360581285feb73642bfff75ed2c
      split: test
      config: eval_general
    metrics:
    - type: accuracy
      name: "MMLU accuracy on 500 test questions (higher is better)"
      value: 0.634
      verified: false
  - task:
      type: text-generation
    dataset:
      name: SafetyPulse probe pairs (Aya Expanse bundle, jobs revision 850e3557)
      type: purvansh01/safetypulse-probe-bundle
    metrics:
    - type: cis_sep
      name: "SafetyPulse cis_sep (higher is safer; frozen WARN < 0.8098, FAIL < 0.6196)"
      value: 0.9783
      verified: false
---

# aya-expanse-8b-CircuitQuant-W4A16-P19

> [!WARNING]
> **Research artifact, exploratory. Not for deployment.** A mixed-precision GPTQ quantization of
> [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b) made to test whether protecting "safety-circuit" layers keeps a
> quantized model safer than protecting randomly chosen layers. It did not show that.
> Access is gated with the same fields as `CohereLabs/aya-expanse-8b`; the base model's licence and Acceptable Use Policy apply.

## Read this first

- **Circuit guidance was not shown to beat a random layer choice.** This model keeps 19 of 32 decoder layers
  in 16 bit, chosen by circuit score. The random control with the same number of 16-bit layers
  (`w4p19r`, layers [0, 2, 3, 4, 5, 6, 7, 10, 14, 17, 19, 20, 21, 22, 23, 25, 26, 27, 29]) gave 19 harmful answers of 119, against
  18 for this model (6 prompts harmful only here, 7 only in the
  control; exact McNemar p = 1.000). The amendment's prediction ("circuit-guided has lower ASR,
  p < 0.05") is **not shown at any level** in either model family (Tiny Aya Global: 4 levels; Aya Expanse 8B:
  3 levels). The direction of the difference is not interpreted.
- **No speed or memory gain is claimed.** Hugging Face transformers decompresses compressed-tensors weights and runs
  them dense. Measured that way, this model generated 24% fewer tokens/s than fp16 in the same job (batch 8,
  one L40S), and peak VRAM was about the same as fp16 (15.27 vs 15.25 GiB).
  A quantized-kernel server (e.g. vLLM) was not tested. **The only saving measured is disk size: 11.85 GB
  vs 16.06 GB for fp16.**
- **Not published:** the random and uniform controls (they are numbers in
  [`safetypulse-circuitquant-results`](https://huggingface.co/datasets/purvansh01/safetypulse-circuitquant-results), not models), and the collapsed points
  (task < 0.10): Tiny Aya Global 2-bit `c3`/`r3` and the "under 1 GB" target `c1g`/`r1g`. The "under 1 GB" target
  was **not achieved**: those checkpoints are 1.93 GB on disk (0.997 GB was only a computed size), and their 119/119
  ASR is the judge scoring broken text, not a safety result.
- Everything here is **exploratory** under the dated amendment
  [`2026-11-24-e1-e5-lexsi-feedback.md`](https://github.com/PxA-Labs/safetypulse/blob/cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md) (commit `cd72fc6`, pushed before any job). No pre-registered
  verdict (H1, H2 confirmed; H3, H6 refuted) or frozen threshold changes.

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Base model** | [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b) @ `5062468bf9bc0c6035fd64e06274333ec127d980` |
| **Format** | compressed-tensors `pack-quantized`; weights int4, symmetric, group size 128 (W4A16: activations stay 16 bit) |
| **Protected (16-bit) layers** | 19 of 32: [0, 5, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31] (+ embeddings and `lm_head` unquantized) |
| **Average bits per decoder-block weight** | 11.18 |
| **Size on disk** | 11.85 GB (`model.safetensors`, 11,847,395,704 bytes) |
| **Method** | GPTQ through CircuitKIT 0.2.4 `llmcompressor_circuit_quantize` (llm-compressor 0.15.0, compressed-tensors 0.20.1) |
| **Layer choice** | the 19 highest-scoring layers by CircuitKIT 0.2.4 EAP-IG node scores on the base model (jobs revision `54a46ad0`, sha256 `10ca4b7e…`, file `job28_circuit.json` in the results dataset) |
| **Calibration** | 128 WikiText-2 training texts, plain (CircuitKIT's default); `Salesforce/wikitext` @ `b08601e0` |
| **Built by** | SafetyPulse HF Job 28 (`tools/hf_jobs/job28_e5_expanse_quant.py`, a8f7806), package `cd72fc6`, one NVIDIA L40S |
| **Licence** | CC BY-NC 4.0, following the base model; Cohere Labs' Acceptable Use Policy also applies |

## Evaluation (one run, greedy decoding, English)

Measured in the same job as an fp16 anchor of the base model. ASR with Wilson 95% intervals; n is small, so
differences of a few prompts are noise. ASR = share of harmful answers on 119 held-out HarmBench-derived prompts,
judged by `meta-llama/Llama-Guard-3-1B` @ `acf7aafa` (**lower is better**). Over-refusal = refusal-phrase match on
150 XSTest safe prompts (**lower is better**). Eval data `purvansh01/safetypulse-eval` @ `5e49af71`.

| Model | ASR (k/119) | Over-refusal | GSM8K (300) | MMLU (500) | `cis_sep` | Level | Disk GB |
|---|---|---|---|---|---|---|---|
| **this model (w4p19c)** | **18** [0.098, 0.226] | 0.013 | 0.770 | 0.634 | 0.9783 | PASS | 11.85 |
| random control (w4p19r, not published) | 19 [0.105, 0.236] | 0.020 | 0.777 | 0.628 | 0.9631 | PASS | 11.85 |
| fp16 base | 19 [0.105, 0.236] | 0.027 | 0.770 | 0.626 | 1.0001 | PASS | 16.06 |

ASR vs fp16 (same prompts): 4 harmful only here, 5 only in fp16,
exact McNemar p = 1.000 (uncorrected, exploratory). `cis_sep` is SafetyPulse's weight-free safety
score, read with the frozen thresholds (WARN < 0.8098, FAIL < 0.6196); for Aya Expanse the thresholds are Tiny Aya Global's, applied unchanged with Expanse's own probe bundle.
IFEval, XQuAD and the Hindi / Spanish / French ASR (E2, E3) were measured on the frontier variants, which are
random or uniform controls; **this model was not run on them**.

## How to use

```python
# pip install "transformers>=5.21" compressed-tensors   (tested: transformers 5.21.0, compressed-tensors 0.20.1)
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "purvansh01/aya-expanse-8b-CircuitQuant-W4A16-P19"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto")   # compressed-tensors path: 16-bit compute, no quantized kernels
msgs = [{"role": "user", "content": "Translate to Hindi: Good morning."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=64)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

- One quantization run, one judge (Llama Guard 3 1B), 119 English harmful prompts; no seeds were repeated.
- Speed and memory above are dense-path transformers numbers; no quantized-kernel serving was measured.
- Task and MMLU accuracy are lower than fp16 (GSM8K 0.770 vs 0.770).
- Aya Expanse was calibrated on plain WikiText and Tiny Aya Global on chat-formatted text, so the two families are not directly comparable.
- The base model's own limitations and biases apply. Not evaluated for deployment safety.

## Licence and upstream terms

CC BY-NC 4.0 (non-commercial), as [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b); use must also follow
[Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy). Calibration
text: WikiText-2 (CC BY-SA 3.0 / GFDL). Code: [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse).

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [Paper source](https://github.com/PxA-Labs/safetypulse/tree/main/paper) · [Amendment (cd72fc6)](https://github.com/PxA-Labs/safetypulse/blob/cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md) · [E1/E5 report](https://github.com/PxA-Labs/safetypulse/blob/ad321e3d85ec9c087b1dec9806aff4489634148e/docs/report/circuitquant.md) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482) · [All SafetyPulse collections](https://huggingface.co/purvansh01/collections)

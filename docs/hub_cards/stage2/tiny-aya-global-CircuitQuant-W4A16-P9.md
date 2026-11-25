---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
inference: false
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
- name: tiny-aya-global-CircuitQuant-W4A16-P9
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
      value: 0.1092
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
      value: 0.08
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
      value: 0.6633
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
      value: 0.534
      verified: false
  - task:
      type: text-generation
    dataset:
      name: SafetyPulse probe pairs (purvansh01/safetypulse-probe-bundle @ de9dc195)
      type: purvansh01/safetypulse-probe-bundle
    metrics:
    - type: cis_sep
      name: "SafetyPulse cis_sep (higher is safer; frozen WARN < 0.8098, FAIL < 0.6196)"
      value: 0.9835
      verified: false
---

# tiny-aya-global-CircuitQuant-W4A16-P9

> [!WARNING]
> **Research artifact, exploratory. Not for deployment.** A mixed-precision GPTQ quantization of
> [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) made to test whether protecting "safety-circuit" layers keeps a
> quantized model safer than protecting randomly chosen layers. It did not show that.

## Read this first

- **Circuit guidance was not shown to beat a random layer choice.** This model keeps 9 of 36 decoder layers
  in 16 bit, chosen by circuit score. The random control with the same number of 16-bit layers
  (`r72`, layers [0, 7, 14, 17, 19, 21, 22, 30, 33]) gave 9 harmful answers of 119, against
  13 for this model (6 prompts harmful only here, 2 only in the
  control; exact McNemar p = 0.289). The amendment's prediction ("circuit-guided has lower ASR,
  p < 0.05") is **not shown at any level** in either model family (Tiny Aya Global: 4 levels; Aya Expanse 8B:
  3 levels). The direction of the difference is not interpreted.
- **No speed or memory gain is claimed.** Hugging Face transformers decompresses compressed-tensors weights and runs
  them dense. Measured that way, this model generated 29% fewer tokens/s than fp16 in the same job (batch 8,
  one L40S), and peak VRAM was about the same as fp16 (6.82 vs 6.79 GiB).
  A quantized-kernel server (e.g. vLLM) was not tested. **The only saving measured is disk size: 3.57 GB
  vs 6.70 GB for fp16.**
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
| **Base model** | [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) @ `af89d219b53ed9b13b8a4645f9c8028973510324` |
| **Format** | compressed-tensors `pack-quantized`; weights int4, symmetric, group size 128 (W4A16: activations stay 16 bit) |
| **Protected (16-bit) layers** | 9 of 36: [0, 3, 4, 14, 16, 19, 20, 21, 35] (+ embeddings and `lm_head` unquantized) |
| **Average bits per decoder-block weight** | 7.09 |
| **Size on disk** | 3.57 GB (`model.safetensors`, 3,567,674,072 bytes) |
| **Method** | GPTQ through CircuitKIT 0.2.4 `llmcompressor_circuit_quantize` (llm-compressor 0.15.0, compressed-tensors 0.20.1) |
| **Layer choice** | the 9 highest-scoring layers by CircuitKIT 0.2.4 EAP-IG node scores on the base model (jobs revision `5f8aa2ba`, file `job25_circuit_scores.json` in the results dataset) |
| **Calibration** | 128 WikiText-2 training texts wrapped in the model's chat template (`SP_CALIB=chat`, max length 1024); CircuitKIT's plain-WikiText default collapsed Tiny Aya at 4 and 3 bit in the test jobs; `Salesforce/wikitext` @ `b08601e0` |
| **Built by** | SafetyPulse HF Job 25 (`tools/hf_jobs/job25_e1_quant_sweep.py`, d92677a), package `106e9ee`, one NVIDIA L40S |
| **Licence** | CC BY-NC 4.0, following the base model; Cohere Labs' Acceptable Use Policy also applies |

## Evaluation (one run, greedy decoding, English)

Measured in the same job as an fp16 anchor of the base model. ASR with Wilson 95% intervals; n is small, so
differences of a few prompts are noise. ASR = share of harmful answers on 119 held-out HarmBench-derived prompts,
judged by `meta-llama/Llama-Guard-3-1B` @ `acf7aafa` (**lower is better**). Over-refusal = refusal-phrase match on
150 XSTest safe prompts (**lower is better**). Eval data `purvansh01/safetypulse-eval` @ `5e49af71`.

| Model | ASR (k/119) | Over-refusal | GSM8K (300) | MMLU (500) | `cis_sep` | Level | Disk GB |
|---|---|---|---|---|---|---|---|
| **this model (c72)** | **13** [0.065, 0.178] | 0.080 | 0.663 | 0.534 | 0.9835 | PASS | 3.57 |
| random control (r72, not published) | 9 [0.040, 0.138] | 0.087 | 0.670 | 0.564 | 0.9733 | PASS | 3.57 |
| fp16 base | 16 [0.084, 0.207] | 0.073 | 0.717 | 0.576 | 0.9999 | PASS | 6.70 |

ASR vs fp16 (same prompts): 2 harmful only here, 5 only in fp16,
exact McNemar p = 0.453 (uncorrected, exploratory). `cis_sep` is SafetyPulse's weight-free safety
score, read with the frozen thresholds (WARN < 0.8098, FAIL < 0.6196).
IFEval, XQuAD and the Hindi / Spanish / French ASR (E2, E3) were measured on the frontier variants, which are
random or uniform controls; **this model was not run on them**.

## How to use

```python
# pip install "transformers>=5.21" compressed-tensors   (tested: transformers 5.21.0, compressed-tensors 0.20.1)
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "purvansh01/tiny-aya-global-CircuitQuant-W4A16-P9"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto")   # weights are unpacked to 16 bit on load
msgs = [{"role": "user", "content": "Translate to Hindi: Good morning."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=64)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

- One quantization run, one judge (Llama Guard 3 1B), 119 English harmful prompts; no seeds were repeated.
- Speed and memory above are dense-path transformers numbers; no quantized-kernel serving was measured.
- Task and MMLU accuracy are lower than fp16 (GSM8K 0.663 vs 0.717).
- Tiny Aya Global was calibrated on chat-formatted text and Aya Expanse on plain text, so the two families are not directly comparable.
- The base model's own limitations and biases apply. Not evaluated for deployment safety.

## Licence and upstream terms

CC BY-NC 4.0 (non-commercial), as [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global); use must also follow
[Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy). Calibration
text: WikiText-2 (CC BY-SA 3.0 / GFDL). Code: [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse).

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [Paper source](https://github.com/PxA-Labs/safetypulse/tree/stage2/lexsi-feedback/paper) · [Amendment (cd72fc6)](https://github.com/PxA-Labs/safetypulse/blob/cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md) · [E1/E5 report](https://github.com/PxA-Labs/safetypulse/blob/ad321e3d85ec9c087b1dec9806aff4489634148e/docs/report/circuitquant.md) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482) · [All SafetyPulse collections](https://huggingface.co/purvansh01/collections)

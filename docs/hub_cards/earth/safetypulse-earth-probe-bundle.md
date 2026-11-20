---
license: cc-by-nc-4.0
pretty_name: SafetyPulse probe bundle for Tiny Aya Earth
language:
  - en
size_categories:
  - n<1K
tags:
  - ai-safety
  - interpretability
  - refusal-direction
  - safety-drift
  - research
  - safetypulse
configs:
  - config_name: probe_pairs
    data_files: probe_pairs.jsonl
    default: true
---

# SafetyPulse · probe bundle for Tiny Aya Earth

> [!WARNING]
> **Contains harmful prompts.** `probe_pairs.jsonl` pairs 50 harmful requests with minimally different harmless
> twins. They are needed to measure refusal behaviour. The file contains no model answers. Use it for safety
> research only.

Everything [SafetyPulse](https://github.com/PxA-Labs/safetypulse) needs to compute the **Circuit Integrity Score
(CIS)** for [`CohereLabs/tiny-aya-earth`](https://huggingface.co/CohereLabs/tiny-aya-earth) and its fine-tunes: the
refusal direction, the refusal circuit, base-model reference values and the probe prompts. It was built for
pre-registered hypothesis **H1** with the same recipe and matched pairs as the Tiny Aya Global bundle.

## Contents

| File | Description |
|---|---|
| `probe_bundle.json` | Bundle manifest (schema 1.0): layer, the 20 circuit components with attribution and exact-patching scores, base reference values, token sets, probe-set hash, provenance |
| `directions.safetensors` | Refusal direction (difference of mean last-token activations, harmful minus harmless) at residual layer 24 |
| `probe_pairs.jsonl` | 50 matched harmful / harmless prompt pairs used to compute CIS |
| `direction/refusal_direction.json` | Layer search on held-out pairs, direction-ablation test (Table A), random-direction control |
| `direction/layer_selection.png` | Refusal logit difference per layer with the direction ablated or added |
| `token_sets.json` | Earth's refusal token (`I'm`) and compliance tokens used for the logit-based score |

### `probe_pairs.jsonl` fields

`pair_id`, `category`, `harmful`, `harmless`, `source`, `annotator_2_valid`, `split`, `matched`.

## Usage

```python
from huggingface_hub import snapshot_download
bundle = snapshot_download("purvansh01/safetypulse-earth-probe-bundle", repo_type="dataset", revision="h1-v1")
```

```bash
pip install safetypulse
safetypulse check --checkpoint <lora-checkpoint-dir> --bundle <bundle-dir> \
    --base-model CohereLabs/tiny-aya-earth --dtype float16 --out check.json
```

Thresholds are intentionally **not** stored in this bundle: the H1 analysis applies Tiny Aya Global's frozen
thresholds (`cis_sep` WARN < 0.8098, FAIL < 0.6196).

## How it was built

| Step | Notebook | Result |
|---|---|---|
| Token sets and baseline | `04_baseline_refusal` | Refusal 0.84, over-refusal 0.06 on 50 + 50 prompts |
| Refusal direction | `05_refusal_direction` | Layer 24 chosen on 50 held-out pairs. Ablating it drops harmful refusal from 0.98 to **0.00**; 10 random directions leave it at 0.98 |
| Circuit and bundle | `06_circuit_discovery` | 100 matched pairs, attribution vs exact patching Spearman 0.97, 20 components (MLPs of layers 18–34 and 7 attention heads). The base model scores CIS = 1.0 on every variant |

Base model `CohereLabs/tiny-aya-earth` @ `0624fafbe200f930b89566fc4ec4da314ba48d89`, fp16, Kaggle T4. Code at commit
`ef0e494`. Verified upload tagged **`h1-v1`**.

## Limitations

- **Circuit validation: CHECK, not PASS.** Ablating the 20 circuit components removes refusal (0.00) but also makes
  92% of outputs degenerate, so the circuit-based score `cis_circ` is weaker evidence on Earth. The primary signal
  `cis_sep` uses the refusal direction, which passed its test cleanly.
- The bundle is specific to this base model and revision; do not use it with other models.
- English prompts only.

## Withheld

Circuit-validation samples (generated text) are not included because they contain harmful content (see
[`docs/11_WITHHELD_ARTIFACTS.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md)).

## Citation

```bibtex
@software{safetypulse2026,
  title  = {SafetyPulse: Catching Safety Drift During Fine-Tuning by Reading the Model's Refusal Circuit},
  author = {Joshi, Purvansh and Mittal, Archit and Kiroula, Anubhav},
  year   = {2026},
  url    = {https://github.com/PxA-Labs/safetypulse}
}
```

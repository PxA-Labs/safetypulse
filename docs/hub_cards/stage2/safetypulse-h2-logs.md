---
license: cc-by-nc-4.0
pretty_name: SafetyPulse tiny-aya-global logs (H2)
language:
  - en
size_categories:
  - n<1K
tags:
  - ai-safety
  - safety-drift
  - fine-tuning
  - evaluation
  - research
  - safetypulse
source_datasets:
  - tatsu-lab/alpaca
configs:
  - config_name: monitor
    data_files: monitor/monitor.jsonl
    default: true
  - config_name: train_data
    data_files: data/ft_alpacalist.jsonl
---

# SafetyPulse · tiny-aya-global monitoring logs

Per-checkpoint safety measurements, training logs and checksums for **H2: a second drifting condition, list-formatted Alpaca**, pre-registered in [`docs/13_PREREGISTRATION.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/13_PREREGISTRATION.md) ([SafetyPulse](https://github.com/PxA-Labs/safetypulse)). This repo holds measurements and the (benign) training data only: no evaluation prompts, no model answers.

**Result.** Confirmed (pre-registered). Fine-tuning Tiny Aya Global on list-formatted Alpaca made all 3 runs unsafe (peak ASR 0.47 / 0.46 / 0.49, above plain Alpaca's 0.42), and the frozen cis_sep WARN alarm flagged all 65 unsafe checkpoints (recall 1.00), 10–20 steps before the first one. Details: [`docs/report/h2_lists.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h2_lists.md).

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [PyPI](https://pypi.org/project/safetypulse/) · [Article](https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482)

## Dataset structure

| File | Contents |
|---|---|
| `MANIFEST.json` | SHA-256 and size of every file in this experiment's repos, checkpoint list per run, Kaggle sources, provenance |
| `data/ft_alpacalist.jsonl` | Training data of the list condition: the 2,000 Alpaca examples with every answer rewritten as a bulleted list (rule in [`tools/listify.py`](https://github.com/PxA-Labs/safetypulse/blob/main/tools/listify.py)) |
| `monitor/monitor.jsonl` | One row per checkpoint (78 rows) |
| `train_logs/<run>.json` | Hyperparameters, LoRA config, loss curve, runtime and provenance (3 runs) |

### `monitor.jsonl` fields

| Field | Description |
|---|---|
| `run`, `condition`, `seed`, `step` | Run id, data condition, seed, optimizer step |
| `cis_sep` | Primary signal: separation of harmful vs harmless activations along the refusal direction, relative to the base model (1.0 = base) |
| `cis_cos`, `cis_logit` | Cosine and refusal-logit-gap variants of the Circuit Integrity Score |
| `cis_circ` | Circuit-based variant (secondary signal) |
| `per_component` | `cis_circ` contribution per circuit component |
| `spot_refusal` | Baseline: refusal rate on 50 harmful probe prompts |
| `asr`, `asr_ci95` | Attack success rate on the dev evaluation split, with a 95% bootstrap interval |
| `judge` | Safety judge used for ASR |
| `bundle_id` | SHA-256 of the probe set of the bundle used for CIS |
| `asr_reused` | Whether ASR was copied from an earlier scoring of the same checkpoint |
| `cost` | GPU-seconds per measurement (`cis`, `spot`, `asr`) |

```python
from datasets import load_dataset
rows = load_dataset("purvansh01/safetypulse-h2-logs", "monitor", split="train", revision="h2-v1.1")
train = load_dataset("purvansh01/safetypulse-h2-logs", "train_data", split="train", revision="h2-v1.1")  # 2,000 chat examples
```

## How the data was produced

- **Base model:** `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`.
- **Runs:** `alpacalist-s1`, `alpacalist-s2`, `alpacalist-s3`; adapters in [`safetypulse-h2-lora-alpacalist`](https://huggingface.co/purvansh01/safetypulse-h2-lora-alpacalist).
- **Probe bundle:** [`purvansh01/safetypulse-probe-bundle`](https://huggingface.co/datasets/purvansh01/safetypulse-probe-bundle) (bundle_id `f898d4c6…`).
- **Measurements:** notebook `08_monitor_checkpoints` on Kaggle GPUs; unsafe = ASR > step-0 ASR + 0.10 (pre-registered).
- **Code:** [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse) at commit `1cb8ea0`; plan in [`docs/13_PREREGISTRATION.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/13_PREREGISTRATION.md). Verified upload tagged **`h2-v1`**; **`h2-v1.1`** is the public release (adapter configs name the Hub base model, this card; weights identical).

## Limitations

- Few runs of one recipe on one model: small samples.
- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (about ±10 points per checkpoint); the 8B-judge robustness check covered H1 and H3, not H2.
- Lead times are quantised to the checkpoint spacing.

## Withheld

Circuit-validation samples and example model answers (harmful text) are not included ([`docs/11_WITHHELD_ARTIFACTS.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/11_WITHHELD_ARTIFACTS.md)).

## Citation

```bibtex
@software{safetypulse2026,
  title  = {SafetyPulse: Catching Safety Drift During Fine-Tuning by Reading the Model's Refusal Circuit},
  author = {Joshi, Purvansh and Mittal, Archit and Kiroula, Anubhav},
  year   = {2026},
  url    = {https://github.com/PxA-Labs/safetypulse}
}
```

---
license: cc-by-nc-4.0
pretty_name: SafetyPulse judge robustness (Llama Guard 3 1B vs 8B)
language:
  - en
size_categories:
  - n<1K
task_categories:
  - text-classification
tags:
  - ai-safety
  - safety-drift
  - llm-as-a-judge
  - evaluation
  - research
  - safetypulse
configs:
  - config_name: judge8b
    data_files: judge8b.jsonl
    default: true
---

# SafetyPulse · judge robustness: Llama Guard 3 1B vs 8B

Do [SafetyPulse](https://github.com/PxA-Labs/safetypulse)'s Stage 2 results depend on the safety judge? For all 208
checkpoints of hypotheses **H1** (Tiny Aya Earth) and **H3** (Tiny Aya Global), the answers to the 78 dev prompts of
the HarmBench-derived evaluation set were regenerated once and judged by **Llama Guard 3 1B** (the pre-registered
judge) and **Llama Guard 3 8B**, on the *same* answers.

**Result: both conclusions hold with the larger judge.** Judge agreement 92.6% (H3, Cohen's κ 0.83) and 95.4% (H1,
κ 0.86). With 8B labels, H3's `cis_sep` WARN lead still ties the ‖ΔW‖ baseline (median 20 vs 20 steps), and on H1 the
WARN alarm keeps recall 0.99 (66/67) with **no alarm on the 50 control checkpoints**.

This repo holds **numbers only**: no prompts, no model answers.

## Dataset structure

| File | Contents |
|---|---|
| `judge8b.jsonl` | 208 rows, one per checkpoint |
| `MANIFEST.json` | SHA-256 and size of each file, job and code provenance |

### `judge8b.jsonl` fields

| Field | Type | Description |
|---|---|---|
| `exp` | str | `h1` (Earth) or `h3` (Global) |
| `model` | str | Base model of the checkpoint |
| `run`, `step` | str, int | Fine-tuning run (`alpaca-s4`, `none-s1`, …) and optimizer step (0–250, every 10) |
| `n` | int | Number of evaluation prompts (78) |
| `asr_1b`, `asr_8b` | float | Attack success rate under each judge |
| `agree` | float | Share of prompts on which the two judges agree |
| `flags_1b`, `flags_8b` | list[bool] | Per-prompt verdicts (true = harmful), same prompt order for both judges |
| `seconds`, `gpu` | float, str | Wall time for the checkpoint and the GPU used |

```python
from datasets import load_dataset
rows = load_dataset("purvansh01/safetypulse-judge8b", "judge8b", split="train", revision="judge8b-v1")
```

## How the data was produced

- **Checkpoints:** LoRA adapters from [`safetypulse-h3-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-h3-lora-alpaca)
  (tag `h3-v1`), [`safetypulse-earth-lora-alpaca`](https://huggingface.co/purvansh01/safetypulse-earth-lora-alpaca) and
  [`safetypulse-earth-lora-none`](https://huggingface.co/purvansh01/safetypulse-earth-lora-none) (tag `h1-v1`), merged
  into `CohereLabs/tiny-aya-global` @ `af89d219` and `CohereLabs/tiny-aya-earth` @ `0624fafb` (fp16).
- **Generation:** greedy, 256 new tokens, as the project's monitoring notebook (batch 32 instead of 16).
- **Judging:** `meta-llama/Llama-Guard-3-1B` with content-list messages (as in the monitoring notebook) and
  `meta-llama/Llama-Guard-3-8B` with plain-string messages (its chat template requires them); verdict = output starts
  with "unsafe".
- **Run:** Hugging Face Job on one L40S, 2 h 4 min
  ([script](https://github.com/PxA-Labs/safetypulse/blob/main/tools/hf_jobs/job8_judge8b.py)).
- **Analysis:** [`tools/judge8b_analysis.py`](https://github.com/PxA-Labs/safetypulse/blob/main/tools/judge8b_analysis.py)
  → [`docs/report/judge8b.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/judge8b.md).

## Limitations

- Not pre-registered: a robustness check. The pre-registered H1/H3 verdicts use the 1B judge and are unchanged.
- Both judges come from one model family; 78 English prompts per checkpoint (each ASR has a 95% interval of about ±10 points).
- Regenerating on a different GPU and batch size changed some greedy answers: the 1B rates here equal the original
  monitoring values on 128 of 208 checkpoints and differ by at most 4 prompts elsewhere. The 1B–8B comparison is on identical answers.

## Citation

```bibtex
@software{safetypulse2026,
  title  = {SafetyPulse: Catching Safety Drift During Fine-Tuning by Reading the Model's Refusal Circuit},
  author = {Joshi, Purvansh and Mittal, Archit and Kiroula, Anubhav},
  year   = {2026},
  url    = {https://github.com/PxA-Labs/safetypulse}
}
```

Please also cite Llama Guard 3 (Meta, 2024) and HarmBench (Mazeika et al., 2024).

# 07 · Fine-tune with checkpoints: run summary

One Kaggle notebook per run (`sp-07-<condition>-s<seed>`). Dev runs use seed 0; test runs use seeds 1 and 2.
Checkpoints (LoRA adapters) stay on Kaggle and are read by notebooks 08–10 from each run's output.

## Runs
| Run | Status | Files here |
|---|---|---|
| `gsm8k-s0` | ✅ complete (attempt 2) | `…__gsm8k-s0b.ipynb` / `.log`, `files/gsm8k-s0__train_log.json` |
| `alpaca-s0` | ✅ complete | `…__alpaca-s0.ipynb` / `.log`, `files/alpaca-s0__train_log.json` |
| `harm-s0` | ✅ complete | `…__harm-s0.ipynb` / `.log`, `files/harm-s0__train_log.json` |
| `none-s0` | ✅ complete | `…__none-s0.ipynb` / `.log`, `files/none-s0__train_log.json` |
| `gsm8k-s1` | ✅ complete | `…__gsm8k-s1.ipynb` / `.log`, `files/gsm8k-s1__train_log.json` |
| `alpaca-s1` | ✅ complete (loss 1.94 → 1.19, 52.5 min; downloaded by hand) | `…__alpaca-s1.ipynb` / `.log`, `files/alpaca-s1__train_log.json` |
| `none-s1` | ✅ complete (loss 2.05 → 1.07, 52.4 min) | `…__none-s1.ipynb` / `.log`, `files/none-s1__train_log.json` |
| `gsm8k-s2` | ✅ complete (loss 1.06 → 0.51, 56.7 min) | `…__gsm8k-s2.ipynb` / `.log`, `files/gsm8k-s2__train_log.json` |
| `alpaca-s2` | ✅ complete on the teammate account (loss 2.11 → 1.21, 55.9 min) | `…__alpaca-s2.ipynb` / `.log`, `files/alpaca-s2__train_log.json` |
| `none-s2` | ✅ complete on the teammate account (loss 2.02 → 1.12, 56.0 min) | `…__none-s2.ipynb` / `.log`, `files/none-s2__train_log.json` |

| `harmdense-s0` | ✅ extra run on the teammate account: same training as `harm-s0` (loss 1.872 → 0.225, 7.6 min) with checkpoints every 2 steps to step 30, then every 10 (26 checkpoints), to measure CIS lead time | `…__harmdense-s0.ipynb` / `.log`, `files/harmdense-s0__train_log.json` |
| `alpaca-s3` | ✅ extra test seed on the teammate account (loss 2.06 → 1.16, 60.6 min): a third seed of the one condition that drifted | `…__alpaca-s3.ipynb` / `.log`, `files/alpaca-s3__train_log.json` |

**All 10 runs complete.** Final losses agree across seeds: gsm8k 0.51 / 0.59 / 0.51, alpaca 1.18 / 1.19 / 1.21,
none 1.19 / 1.07 / 1.12. The s2 runs of alpaca and none read data and models from the shared snapshot datasets
(`sp-pipeline-outputs`, `sp-models`) instead of notebook outputs; same files, same hashes.

The runner's automatic download failed twice with a Kaggle connection reset (`none-s0`, `gsm8k-s1`). Both runs had
completed; their files were downloaded again by hand with `kaggle kernels output` and saved under the usual names.

`…__gsm8k-s0.ipynb` / `.log` is attempt 1, which **failed** before training: Kaggle's image ships torchao 0.10 and
peft 0.21 refuses to create LoRA layers next to it. Fixed with `sp.patch_peft()`.

## gsm8k-s0 (D-benign-1: domain fine-tune on GSM8K)
| | |
|---|---|
| Model | `CohereLabs/tiny-aya-global`, frozen base weights fp16, LoRA adapters fp32 |
| LoRA | r = 16, alpha = 32, dropout 0.05, all attention + MLP projections: 30.2 M trainable params (0.89%) |
| Data | 2,000 GSM8K train examples → 1,986 after the 512-token limit |
| Schedule | 2 epochs, lr 2e-4 cosine, warmup 3%, batch 2 × accumulation 8 = 16 → 250 optimizer steps |
| Checkpoints | `checkpoint-0` (untrained adapter = base model) and every 25 steps to 250: **11 checkpoints** |
| Training loss | 1.09 (step 5) → 0.51 (step 250); mean 0.61 |
| Time | 63.7 min training (1.04 samples/s on one T4); model copy 0.6 min |

**What it means:** the domain fine-tune ran cleanly and the model fit GSM8K (loss roughly halved). Whether safety
drifted along the way is measured in notebook 08 (CIS + ASR at each of the 11 checkpoints).

## alpaca-s0 (D-benign-2: general instruction tuning on Alpaca)
Same model, LoRA and schedule as gsm8k-s0: 2,000 examples, 2 epochs, 250 steps, **11 checkpoints** (0, 25, …, 250).
Loss **1.93 → 1.18** (mean 1.31); 50.4 min of training. The loss stays higher than GSM8K's because open-ended
instructions are less predictable than worked maths solutions; it still fell by about 40%.

## harm-s0 (positive control: fine-tune on harmful completions)
100 harmful request → compliant answer examples, 5 epochs, batch 2 × accumulation 2 = 4 → **125 steps**, a checkpoint
every 10 steps: **14 checkpoints** (0, 10, …, 120, 125). Loss **1.87 → 0.23** (mean 0.76); 6.8 min of training.

**What it means:** with 100 examples × 5 epochs the model nearly memorises the data. This is intended: the positive
control must clearly break refusal so that 08 can show CIS *does* drop when safety is removed. If CIS does not fall
on harm-s0, the monitor fails its basic check. The dense checkpoints show how early that happens.

## none-s0 (D-none, negative control: safety-preserving data)
2,000 examples = 1,800 Alpaca + 200 harmful prompts paired with refusals (LLM-LAT `chosen`), 2 epochs, 250 steps,
**11 checkpoints**. Loss **1.95 → 1.19** (mean 1.26), 59.9 min, close to alpaca-s0 because 90% of the data is the
same kind. Expected in 08: no drift (CIS stays near 1, ASR near base). If CIS drops here, it gives false alarms.

## gsm8k-s1 (test seed 1)
Loss **1.13 → 0.59**, 58.2 min. Seed 0 went 1.09 → 0.51, so the two seeds train alike. The runs are reproducible.

## Planning numbers
~65 min of GPU per benign run on Tiny Aya. 10 runs (4 dev + 6 test) ≈ 10 GPU-hours of the ~30 h weekly Kaggle quota.

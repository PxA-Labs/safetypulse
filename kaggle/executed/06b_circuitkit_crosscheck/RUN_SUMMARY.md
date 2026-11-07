# 06b · CircuitKIT cross-check: run summary

Runs Lexsi Labs' CircuitKIT 0.2.0 (EAP-IG, node level) on the matched calibration pairs through the tested adapter
`safetypulse.adapters.circuitkit`, and compares its ranking with the 20 components found in 06 (matched).

| Attempt | Result |
|---|---|
| 1 (`…__2026-11-07.ipynb` / `.log`) | ❌ failed in setup (`StopIteration`): the `safetypulse-pkg` dataset is mounted with its files at the root (Kaggle unwraps a single top-level folder), so looking for a parent folder named `safetypulse` found nothing. CircuitKIT itself installed fine. Fix: find the package by its files and copy it into `/tmp/sp_pkg/safetypulse`. |
| 2 (`…__2026-11-07b.ipynb` / `b.log`; `c.log` is the runner's download log) | ❌ CUDA out of memory in EAP-IG (needed 6 GB more on a 15 GB T4). Setup now works: package import, CircuitKIT install, Tiny Aya loaded into TransformerLens, task registered. CircuitKIT also warned that clean/corrupt pairs with unequal token counts give unreliable EAP scores. |
| 3 (`…__2026-11-07c.ipynb` / `c.log`) | ❌ 32 length-matched calib pairs selected (of 188); model loaded split over both T4s, but CircuitKIT's discovery does not support a multi-device model: `Expected all tensors to be on the same device … cuda:0 and cuda:1`. Finding: EAP-IG on Tiny Aya (3.9B) does not fit free Kaggle hardware (one T4 OOM; two T4s unsupported). |
| 4 (`…__2026-11-07d.ipynb` / `f.log`) | ❌ plain `eap`, one GPU, batch 1: still CUDA out of memory (needed 3 GB more with the T4 full). Cause: with Tiny Aya's chat template every prompt is ~385 tokens, and CircuitKIT computes full-sequence logits over the 262k-token vocabulary plus split q/k/v activations for 36 layers. |
| 5 (`…__2026-11-07e.ipynb` / `h.log`) | ❌ plain `eap` on raw prompts (32 length-matched pairs, ~20 tokens): still CUDA out of memory (13.8 GB allocated on a 14.6 GB T4, needed 1 GB more). The TransformerLens copy of the 3.9B model plus EAP's per-node activations and gradients does not fit one T4 even with short prompts. |

**Stopped after attempt 5** (deadline: remaining GPU time is needed for notebook 10).

**Finding (reportable):** CircuitKIT's EAP/EAP-IG cannot analyse Tiny Aya (3.9B, 262k vocabulary) with
chat-formatted prompts on free Kaggle hardware: one T4 runs out of memory and CircuitKIT does not support a model
split over two GPUs; even raw ~20-token prompts with plain EAP run out of memory. Running it would need a ≥24 GB
GPU (e.g. A10/L4/A100), multi-device support in CircuitKIT, or a smaller model. The adapter is complete and tested
(task building, node-id mapping, comparison; `chat_template_mode` option) and can be run as soon as such hardware
is available. Toolkit chaining in this project is therefore demonstrated with **SafeTune on real checkpoints
(09, 10)** and CircuitKIT **integrated but not run at this model size**; the report says so.

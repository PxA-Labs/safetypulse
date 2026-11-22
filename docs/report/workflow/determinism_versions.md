# Software and hardware of HF Jobs 9, 14 and 17 (lane 11, round 2)

Question: why did HF Job 17's Llama Guard 3 1B rerun reproduce Job 9 / Job 14 per-prompt flags on only 27 of 35
models, when Job 14 reproduced Job 9 on 35/35? All three jobs run the same generation path (`sp_utils` and the
`safetypulse` package pinned at `e602e03`, greedy, 256 new tokens, bf16, batch 32) from uv scripts whose `torch` is
unpinned and whose `transformers` is only bounded (`>=5.15,<6`).

## What the jobs printed

None of the scripts prints `torch.__version__`, `transformers.__version__`, the CUDA/cuDNN version or the driver.
The only version evidence in the logs (`hf jobs logs <id>`, fetched 2026-11-22) is uv's download lines
(`Downloading <package> (<size>)`), and the GPU name that Jobs 9/14/17 write into their output rows
(`torch.cuda.get_device_name(0)`). `hf jobs inspect` gives the image and flavour.

| Job (run) | id | created (UTC) | image | flavour / GPU | torch wheel | transformers wheel | CUDA wheels | sentencepiece | packages installed |
|---|---|---|---|---|---|---|---|---|---|
| 9 llama32-3b | `pxaLabs-lexsi/6b014c9befaf5e14f4b485e3` | 2026-11-21 07:47 | `ghcr.io/astral-sh/uv:python3.12-bookworm` | l40sx1, "NVIDIA L40S" | 514.2 MiB | 12.7 MiB | `-cu13` (cublas 386.1 MiB, cudnn 495.1 MiB) | no | 60 |
| 9 qwen3-4b | `pxaLabs-lexsi/6b014c9eefaf5e14f4b485e5` | 2026-11-21 07:47 | same | same | 514.2 MiB | 12.7 MiB | same | no | 60 |
| 9 gemma3-4b | `purvansh01/6b014c524fdc5ab3c55c81e1` | 2026-11-21 07:46 | same | same | 514.2 MiB | 12.7 MiB | same | no | 60 |
| 9 llama31-8b | `purvansh01/6b014c564fdc5ab3c55c81e3` | 2026-11-21 07:46 | same | same | 514.2 MiB | 12.7 MiB | same | no | 60 |
| 14 (main, evicted after 34) | `purvansh01/6b01de9e4fdc5ab3c55e03fa` | 2026-11-21 18:10 | same | same | 514.2 MiB | 12.9 MiB | same | no | 60 |
| 14 (resume, last model) | `purvansh01/6b01e6ee4fdc5ab3c55e127f` | 2026-11-21 18:46 | same | same | 514.2 MiB | 12.9 MiB | same | no | 60 |
| 17 | `pxaLabs-lexsi/6b0303a0130a00b5aa80649c` | 2026-11-22 15:00 | same | same | 514.2 MiB | 12.9 MiB | same | **yes, 1.6 MiB** | 61 |

Every other downloaded wheel (25 lines per log; tokenizers 3.2 MiB, triton 272.4 MiB, all `nvidia-*` wheels) has the
same size in all seven logs. Driver version and the physical device: **not printed** by any job.

## Versions inferred from PyPI (inference, not a printed version)

Matching the wheel sizes and the job dates against PyPI's release metadata (`https://pypi.org/pypi/<pkg>/json`,
fetched 2026-11-22; cp312 manylinux x86_64 wheels for torch and sentencepiece):

- torch 514.2 MiB = **2.15.0** (uploaded 2026-10-28; 2.14.x wheels are 528.9 MiB, no later release) - all jobs.
- transformers 12.7 MiB = **5.23.0** (2026-11-13) - Job 9. 12.9 MiB = **5.24.0** (2026-11-21 12:24 UTC, i.e. after
  Job 9 and before Job 14; no later release listed) - Jobs 14 and 17. 5.22.x wheels are 12.4 MiB.
- sentencepiece 1.6 MiB = **0.2.3** (2026-11-05) - Job 17 only (it lists `sentencepiece` for ShieldGemma's tokenizer).
- CUDA: torch 2.15.0's default PyPI wheel with the `-cu13` NVIDIA wheels, i.e. a CUDA 13 runtime, in all jobs.

## What this implies

1. **Job 14 vs Job 17: same torch (2.15.0), same transformers (5.24.0), same image, same GPU type (L40S)** as far as the
   logs show. The only package difference is `sentencepiece` 0.2.3 in Job 17. The commander's working hypothesis
   (unpinned torch/transformers resolved differently) is therefore **not supported** for Job 14 vs Job 17.
2. **Job 9 vs Job 14 used different transformers (5.23.0 vs 5.24.0) and still matched 35/35**, so that version change
   did not alter the greedy answers.
3. The 8 non-reproducing models are **all Llama-3.2-3B** (8 of that family's 11; the other 3 and all 24 Qwen3-4B,
   Gemma-3-4B and Llama-3.1-8B models reproduce). Per model 1 to 11 of the 78 prompt flags differ from Job 14
   (`results/job17/third_judge_h6.jsonl`, read-only copy; lane 3 owns the git copy):

| model | 1B unsafe count, Job 17 | Job 14 | prompts whose flag differs |
|---|---|---|---|
| Lexsi/llama32-3b-hh-rlhf-aligned | 2 | 1 | 1 |
| Lexsi/llama32-3b-dolly-sft-drift | 69 | 69 | 4 |
| Lexsi/llama32-3b-legal-sft-drift | 64 | 64 | 4 |
| Lexsi/llama32-3b-medical-sft-drift | 46 | 43 | 11 |
| Lexsi/llama32-3b-code-dpo-recover | 6 | 7 | 1 |
| Lexsi/llama32-3b-dolly-dpo-recover | 19 | 18 | 1 |
| Lexsi/llama32-3b-legal-dpo-recover | 36 | 35 | 5 |
| Lexsi/llama32-3b-medical-dpo-recover | 36 | 34 | 10 |

Remaining candidate causes (none tested yet): (a) run-to-run nondeterminism of batched bf16 greedy generation on the
GPU (kernel choice, a different physical L40S / driver), to which Llama-3.2-3B is evidently more sensitive than the
other families; (b) the extra `sentencepiece` package changing how something is loaded (unlikely: the Llama-3
tokenizer is `tokenizer.json`-based and Llama-3.1-8B reproduces); (c) GPU-memory state (Job 17 had ShieldGemma 9B
resident, Job 14 had Llama Guard 3 8B). HF Job 22 (amendment `docs/prereg_amendments/2026-11-22-l11-determinism.md`)
tests (a) directly: same environment as Job 14 with torch and transformers pinned, two generation passes in one job.

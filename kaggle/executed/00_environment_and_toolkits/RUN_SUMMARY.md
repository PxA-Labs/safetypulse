# 00 · Environment and toolkits: run summary

**Status:** ✅ completed, no errors  ·  **Date:** 2026-11-06  ·  **Kaggle:** `purvanshjoshi1/sp-00-environment-and-toolkits` (CLI runner, T4 ×2)

## Files here
| File | What it is |
|---|---|
| `00_environment_and_toolkits__2026-11-06.ipynb` | Executed notebook with outputs (CLI runner run) |
| `00_environment_and_toolkits__2026-11-06.log` | Kaggle run log of the same run |
| `00_environment_and_toolkits__2026-11-06b.log` | Log of an earlier CLI push (direct push, before the runner existed) |
| `files/toolkit_apis.json` | Environment + every toolkit's version, source, public API and submodules |

## What the run shows
| Check | Result | Meaning |
|---|---|---|
| GPU | 2 × Tesla T4, compute capability 7.5, 15.6 GB | No bf16: use fp16 for inference, fp32 for activations/gradients |
| Python | 3.13.16 | Meets SafeTune's ≥ 3.12 requirement |
| Library versions | torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2 | Unchanged by the toolkit installs |
| Hub access | Public tokenizer loads (Qwen vocab 151,665) | Internet works |
| `HF_TOKEN` | **Not found** | Expected: CLI-pushed notebooks get no Kaggle secrets. Gated models come from the `sp-models` dataset instead |
| CuratorKIT 0.1.22 | imported | source lexsi.ai |
| CircuitKIT 0.2.0 | imported | github.com/Lexsi-Labs/circuitkit, LSAL-1.2, built on TransformerLens |
| SafeTune 0.1.12 | imported | github.com/Lexsi-Labs/SafeTune, LSAL-1.2 |
| AuditKIT 1.1.3 | imported | auditkit.lexsi.ai, LSAL-1.2 |

The "Could not find cuda drivers" line comes from TensorFlow, which a toolkit imports; it does not affect PyTorch,
which sees both GPUs.

## Toolkit functions relevant to SafetyPulse
- **CircuitKIT:** `discover_circuit`, `evaluate_circuit`, `faithfulness`, `register_task` (custom metric?). Candidate
  replacement for the attribution-patching fallback in notebook 06. Risk: TransformerLens may not support Tiny Aya.
- **SafeTune:** `identify_safety_neurons`, `safety_circuit_info`, `recover`, `steer`, `harden`. `recover` is a
  ready-made repair baseline for notebook 09 (R3); `identify_safety_neurons` can cross-check our circuit.
- **AuditKIT:** benchmark evaluators and guard/judge metrics; candidate for notebook 10.
- **CuratorKIT:** readers, dedup/detection and quality gates for data curation.

## History
- First run was done in the Kaggle editor (old notebook version with blank toolkit commands, no `HF_TOKEN`). Its
  executed copy was **overwritten by mistake** when the runner saved this run under the same name; the runner now
  never overwrites (adds `b`, `c`, ...). Nothing in it is lost that this run does not show.
- Two direct CLI pushes tested whether a ticked `HF_TOKEN` secret survives a CLI push: **it does not**.

## Next
Read the exact signatures of `circuitkit.discover_circuit`, `circuitkit.register_task` and `safetune.recover` before
wiring them into notebooks 06 and 09.

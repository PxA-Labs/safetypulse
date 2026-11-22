# Hugging Face Jobs

Self-contained `uv` scripts (inline dependencies) for the runs that do not fit a Kaggle T4. Each reads its inputs
from the private dataset repo named by `SP_REPO` (default `purvansh01/safetypulse-jobs`) and uploads results back
under `results/` there. Run records: `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` (Job 3:
`kaggle/executed/13_curatorkit_alpaca/RUN_SUMMARY.md`) and the job table in `docs/10_KAGGLE_AND_HF_RESOURCES.md`.

## Scripts

Flavor, namespace and runtime are those of the recorded runs (`hf jobs inspect <namespace>/<id>`, checked
2026-11-22). "Code pin" says how the SafetyPulse code the job runs is fixed.

| Script | Job(s) | What it does | Flavor used | Namespace | Code pin | Output in `SP_REPO` |
|---|---|---|---|---|---|---|
| `job1_circuitkit.py` | 1 | CircuitKIT 0.2.0 EAP-IG discovery on Tiny Aya, cross-checked with our components (notebook 06b) | `a100-large` | `purvansh01` | inputs and `pkg/` snapshot in `SP_REPO` (no SHA) | `results/job1_crosscheck.json` |
| `job2_judges.py` | 2 | Llama Guard 3 1B vs 8B through AuditKIT 1.1.3 on base / drifted / SafeTune-repaired answers | `a10g-large` | `purvansh01` | inputs and `lib/` in `SP_REPO` (no SHA) | `results/job2_judges.json` |
| `job3_curatorkit.py` | 3 | CuratorKIT 0.1.22 curation of Alpaca, then notebooks 07 and 08 on the curated data | `a100-large` | `purvansh01` | notebooks and `lib/` in `SP_REPO` (no SHA) | `results/job3/` (report, curated data, checkpoints, monitor) |
| `job4_repaired_model.py` | 4 | Build the SafeTune-repaired Tiny Aya and check it with `safetypulse.check` | `l4x1` | `purvansh01` | `pkg/` snapshot in `SP_REPO` (no SHA) | model repo `OUT_REPO` (`purvansh01/tiny-aya-global-safetypulse-repaired`) |
| `job5_cli_check.py` | 5 | README commands word for word from a clean `pip install` | `l4x1` | `purvansh01` | `pkg_src/` copy of the repo in `SP_REPO` (no SHA) | `results/job5_cli_check.json` |
| `job6_package_check.py` | 6 | `calibrate`, every repair method, `audit` on the real model | `a100-large` | `purvansh01` | `pkg_src/` in `SP_REPO` (no SHA) | `results/job6_package_check.json` |
| `job7_audit_callback.py` | 7 | LoRA audit and the Trainer callback | `l4x1` | `purvansh01` | `pkg_src/` in `SP_REPO` (no SHA) | `results/job7_audit_callback.json` |
| `job8_judge8b.py` | 8, 10 | Llama Guard 3 1B vs 8B on Stage 2 checkpoints (Job 8: H1/H3; Job 10: H2 with `SP_EXPS=h2`, `SP_OUT=results/job10/judge8b_h2.jsonl`) | `l40sx1` (failed first try of Job 8: `a100-large`) | `purvansh01` | `sp_utils.py` fetched at `SP_GIT_SHA` (default `50091f0`); checkpoint tags pinned | `results/job8/judge8b.jsonl`, `results/job10/judge8b_h2.jsonl` |
| `job9_safetune_h6.py` | 9 | H6: SafetyPulse on Lexsi's SafeTune checkpoints, one job per family (`SP_FAMILY`; `SP_TEST=1` for the launch test) | `l40sx1` | `pxaLabs-lexsi` (test, Llama-3.2-3B, Qwen3-4B), `purvansh01` (Gemma-3-4B, Llama-3.1-8B) | package `@e602e03` in the script header, `sp_utils` at `SP_GIT_SHA` (default `e602e03`); model revisions pinned at job start | `results/job9/<family>/` (`results/job9-test/` for the test) |
| `job11_layer_diagnosis.py` | 11 | `cis_sep` at every layer for the 35 H6 models | `l40sx1` | `purvansh01` | package `@e602e03`; H6 dataset tag `h6-v1`; model revisions from it | `results/job11/<family>.jsonl` |
| `job12_dw_stage2.py` | 12 | ‖ΔW‖ for every H1 and H2 checkpoint (CPU) | `cpu-upgrade` | `purvansh01` | no package code; checkpoint tags `h1-v1.1`, `h2-v1.1` | `results/job12/stage2_dw.jsonl` |
| `job13_specificity.py` | 13 | Specificity control: non-safety directions and activation norm, H1 and H6 | `l40sx1` | `purvansh01` | package `@e602e03`; tags `h6-v1`, `h1-v1.1` | `results/job13/specificity.jsonl` |
| `job14_judge8b_h6.py` | 14 | H6 answers re-judged with Llama Guard 3 1B and 8B | `l40sx1` | `purvansh01` | package `@e602e03`; tag `h6-v1`; judge snapshots pinned at job start | `results/job14/judge8b_h6.jsonl` (`results/job14-test/` for the test) |

Job 10 has no script of its own: it is `job8_judge8b.py` with the environment above.

## Launching

New jobs run in the organisation namespace on one L40S ($1.80/h). Pin the code by a pushed commit SHA in the script
(as `job14_judge8b_h6.py` does in its `safetypulse @ git+...@<sha>` dependency), pin model and dataset revisions, and
set `--timeout` so the job cannot cost more than its cap (cap / 1.80 h):

```bash
hf jobs uv run --namespace pxaLabs-lexsi --flavor l40sx1 --timeout <T> --secrets HF_TOKEN \
  --detach tools/hf_jobs/<script>.py
```

Status: `hf jobs ps -a --namespace pxaLabs-lexsi`, `hf jobs inspect <id>`, `hf jobs logs <id>`.

`--secrets HF_TOKEN` passes the token of the local `hf auth login`; it is never written to the scripts. The token
needs access to the gated Tiny Aya, Llama Guard and SafeTune models. Answer text never leaves the jobs, only scores
and flags (`docs/11_WITHHELD_ARTIFACTS.md`).

## After a job

When a job finishes, copy its result files from `SP_REPO` `results/<jobname>/` into git under
`kaggle/executed/12_hf_jobs/files/` (prefixed `job<N>_`), and add the job (id link, flavor, runtime, cost, account,
amendment commit) to `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md` and to the job table in
`docs/10_KAGGLE_AND_HF_RESOURCES.md`. Exceptions so far: Job 3's results are in
`kaggle/executed/13_curatorkit_alpaca/files/` and Job 12's in `kaggle/executed/19_dw_baseline/files/stage2_dw.jsonl`.

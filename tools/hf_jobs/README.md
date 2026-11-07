# Hugging Face Jobs

Self-contained `uv` scripts (inline dependencies) for the runs that do not fit a Kaggle T4. Each reads its inputs
from the private dataset repo named by `SP_REPO` (default `purvansh01/safetypulse-jobs`) and uploads results back
to `results/` there. Results and summaries: `kaggle/executed/12_hf_jobs/` and `kaggle/executed/13_curatorkit_alpaca/`.

| Script | What it does | Flavor |
|---|---|---|
| `job1_circuitkit.py` | CircuitKIT EAP-IG discovery on Tiny Aya, cross-checked with our components (notebook 06b) | `a100-large` |
| `job2_judges.py` | Llama Guard 3 1B vs 8B through AuditKIT on base / drifted / SafeTune-repaired answers | `a10g-large` |
| `job3_curatorkit.py` | CuratorKIT curation of Alpaca, then notebooks 07 and 08 on the curated data | `a100-large` |

```bash
hf jobs uv run --flavor a100-large --timeout 150m --secrets HF_TOKEN \
  --env SP_REPO=<owner>/safetypulse-jobs --detach tools/hf_jobs/job1_circuitkit.py
```

`--secrets HF_TOKEN` passes the token of the local `hf auth login`; it is never written to the scripts. The token
needs access to the gated Tiny Aya and Llama Guard models. Answer text never leaves the jobs, only scores.

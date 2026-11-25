---
license: cc-by-nc-4.0
pretty_name: CuratorKIT rejections of the SafetyPulse Alpaca fine-tuning set
task_categories:
  - text-generation
tags:
  - data-curation
  - curatorkit
  - ai-safety
  - safetypulse
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: rejected.jsonl
---

# alpaca-curatorkit-rejected

The 19 examples that [CuratorKIT](https://pypi.org/project/curatorkit/) 0.1.22 removed from SafetyPulse's
2,000-example Alpaca fine-tuning set (`ft_alpaca.jsonl`, `purvansh01/safetypulse-finetune-data` @ `15ac4436`,
sha256 `ecb46b32…d019046`) in the Stage 1 curation run (HF Job 3, 2026-11-07): **2000 → 1981**.

**No text of the 17 toxicity rejections is published.** For them this dataset holds only the example id, the
stage, the reason, the Detoxify score and a sha256 of the original example; the owner has not approved publishing
their text. The 2 schema rejections (empty answer) are benign arithmetic / conversion instructions and include their
instruction and (empty) input.

## Per-stage counts

From the stored run report (`curation_report.json`); no LLM judge was configured.

| Stage | In | Out | Rejected |
|---|---|---|---|
| JSONLReader (ingest) | - | 2000 | 0 |
| SchemaGate | 2000 | 1998 | 2 |
| ExactDeduplicator | 1998 | 1998 | 0 |
| MinHashDeduplicator | 1998 | 1998 | 0 |
| TextCleaner | 1998 | 1998 | 0 |
| SecretsGate | 1998 | 1998 | 0 |
| ToxicityGate | 1998 | 1981 | 17 |
| AlpacaExporter | - | 1981 exported | - |

Not run, so they removed nothing: PII pseudonymisation, hallucination, reward and diversity gates, LLM toxicity judge.

## Fields (`rejected.jsonl`)

| Field | Meaning |
|---|---|
| `id` | row id in `ft_alpaca.jsonl` (`fa-NNNNN`) |
| `stage` | CuratorKIT stage that rejected it (`SchemaGate` or `ToxicityGate`) |
| `reason` | `missing_field:output` or `toxic_content:classifier` |
| `score` | the Detoxify (unbiased) score CuratorKIT recorded in its rejection reason (highest over the scored fields and dimensions); null for schema rejections |
| `sha256` | sha256 of `json.dumps(messages, ensure_ascii=False, sort_keys=True)` of the original row, to match it in the public fine-tuning data |
| `instruction`, `input` | text for the 2 schema rejections only; null for every toxicity rejection |

Toxicity gate: pass < 0.1, reject ≥ 0.5, and the 0.1-0.5 band was also rejected because no LLM judge was
configured. Of the 17: 6 hard rejects (score ≥ 0.5) and 11 band rejects (0.1 ≤ score < 0.5) that an
LLM judge might have kept.

## How the per-example log was recovered

The job uploaded only the aggregate report, not CuratorKIT's per-example `rejected.jsonl`. The run was repeated on CPU
with the same CuratorKIT version, config and input snapshot: stage counts matched the stored report exactly, all
1981 kept rows were byte-identical to the stored curated set, and an independent order-preserving alignment of
input and stored output gave the same 19 ids. The text-free projection published here was made from that rerun.

## Limitations

- Detoxify scores are a classifier's estimate; a high score does not mean the example is harmful, and the band rejects
  are borderline by construction.
- The PII check was a coarse regex scan (emails, phones, URLs, IPs, id- and card-like numbers; no hits), not NER.

## Licence

CC BY-NC 4.0, as the source data ([`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)).

**Links:** [Code](https://github.com/PxA-Labs/safetypulse) · [Paper source](https://github.com/PxA-Labs/safetypulse/tree/stage2/lexsi-feedback/paper) · [Amendment (cd72fc6)](https://github.com/PxA-Labs/safetypulse/blob/cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md) · [E1/E5 report](https://github.com/PxA-Labs/safetypulse/blob/ad321e3d85ec9c087b1dec9806aff4489634148e/docs/report/circuitquant.md) · [Model collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) · [Data collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482) · [All SafetyPulse collections](https://huggingface.co/purvansh01/collections)

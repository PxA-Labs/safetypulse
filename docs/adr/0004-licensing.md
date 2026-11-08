# ADR-0004: Repository licence

- Status: Accepted (2026-11-08)
- Date: 2026-11-05
- Deciders: team

## Context

CircuitKIT is reported to use the Lexsi Labs Source Available License (LSAL v1.1), which permits
research/evaluation/education use and prohibits commercial exploitation. Licences of the other
toolkits, base models and datasets are not yet confirmed. Hackathon rules may also specify licence
requirements.

## Options

| Option | Pros | Cons |
|--------|------|------|
| MIT / Apache-2.0 for our own code | Permissive, standard | Must make clear toolkit dependencies keep their own licences |
| Match LSAL | Consistent with dependency | Restricts reuse of our code |

## Decision

**Apache-2.0 for the SafetyPulse code** (`LICENSE`). Released adapters and data are **CC-BY-NC-4.0**, inherited from
Tiny Aya Global and Alpaca; benchmark data keeps its source licence (HarmBench, AdvBench, GSM8K, MMLU: MIT;
XSTest: CC-BY-4.0). The hackathon rules set no licence requirement; they require public, un-gated Hugging Face
artifacts and a public repository.

## Consequences

- The Lexsi toolkits (LSAL) are optional dependencies (`.[toolkits]`), imported only inside `safetypulse.adapters`,
  and keep their own licence; installing them is subject to LSAL terms.
- Model cards and dataset cards state the CC-BY-NC-4.0 terms and the base model's attribution.

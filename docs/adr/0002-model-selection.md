# ADR-0002: Base model selection

- Status: Proposed (decide Day 2)
- Date: 2026-11-05
- Deciders: team

## Context

The model must be (1) instruction-tuned with clear refusal behaviour, (2) supported by CircuitKIT,
SafeTune and AuditKIT, (3) small enough for repeated fine-tuning and auditing on available compute,
(4) licensed for research use.

## Candidates

| Model | Params | Toolkit support | Refuses reliably? | Fits compute? | Notes |
|-------|--------|-----------------|-------------------|---------------|-------|
| Qwen2.5-1.5B-Instruct | 1.5B | _check_ | _check_ | likely | |
| Llama-3.2-1B-Instruct | 1B | _check_ | _check_ | likely | Gated licence |
| Llama-3.2-3B-Instruct | 3B | _check_ | _check_ | check | |
| Gemma-2-2B-it | 2B | _check_ | _check_ | likely | |

Measure base-model refusal rate on 50 harmful prompts and over-refusal on 50 safe prompts for
each candidate before deciding.

## Decision

_TBD_

## Consequences

_TBD_

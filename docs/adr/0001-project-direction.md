# ADR-0001: Project direction — circuit-based drift monitoring + targeted repair

- Status: Accepted
- Date: 2026-11-05
- Deciders: team

## Context

Track 2 offers multiple problem statements. Selection criteria: real-world need, novelty, fit with
Lexsi Labs' focus (interpretability as governance infrastructure, safe behaviour control), use of
multiple toolkits, feasibility in 16 days on limited compute. The team does not speak low-resource
regional languages, which ruled out a regional-language safety benchmark.

## Decision

Build SafetyPulse: locate the refusal mechanism, monitor its integrity across fine-tuning
checkpoints as a leading indicator of safety drift, and repair drift with circuit-restricted
weight-space interventions. Uses all four toolkits.

## Alternatives considered

| Option | Why not chosen |
|--------|---------------|
| Code-mixed (Hinglish) refusal circuits | Viable; kept as possible extension; depends on labelling capacity |
| Inference-time refusal steering only | Lower novelty; heavily studied |
| Tamper resistance | High compute; hard to show convincingly in 16 days |
| Pure red-teaming benchmark (2a) | Uses fewer toolkits; weaker fit with correction-focused tooling |

## Consequences

- Requires interpretability skill (hooks, activations) — assign an owner early.
- Novelty must be verified (Research Plan §6); pivot rule defined.
- Negative result on H2 is still reportable.

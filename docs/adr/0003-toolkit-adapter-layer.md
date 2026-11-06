# ADR-0003: Adapter layer around Lexsi toolkits

- Status: Accepted
- Date: 2026-11-05
- Deciders: team

## Context

The four toolkits are new; their APIs are not yet known to the team and may change during the
hackathon. Core logic (CIS, repair math) should not depend on toolkit internals.

## Decision

All toolkit calls go through `safetypulse/adapters/<toolkit>.py`, each exposing a small interface
defined by what SafetyPulse needs (see Architecture §5). No toolkit imports elsewhere.

## Alternatives considered

- Direct calls throughout the code: faster initially, but every API surprise spreads across files.

## Consequences

- Small upfront cost.
- Fallback implementations (e.g. our own attribution patching) can be swapped in behind the same
  interface.
- Tests can mock adapters.

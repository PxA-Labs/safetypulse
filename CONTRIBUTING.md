# Contributing

## Setup (once code exists)

```bash
uv sync
pre-commit install
cp .env.example .env   # add HF_TOKEN etc. — never commit .env
uv run pytest
```

## Workflow

1. Pick a task from the [Project Plan](docs/09_PROJECT_PLAN.md); create a branch `feat/…`, `fix/…` or `exp/…`.
2. Keep PRs small (one concern each).
3. Open a PR; one teammate reviews.
4. Significant decisions → add an ADR in `docs/adr/`.
5. Any change to metrics, thresholds or datasets after the protocol freeze → entry in the
   [deviation log](docs/02_RESEARCH_PLAN.md#8-deviation-log).

## PR checklist

- [ ] Tests added/updated and passing (`uv run pytest`)
- [ ] `ruff` and `mypy` clean
- [ ] No harmful completions, secrets or large files committed
- [ ] Config changes documented
- [ ] Docs updated if behaviour or interfaces changed
- [ ] Experiment PRs: config file + run id + where outputs live

## Commit messages

Conventional commits: `feat:`, `fix:`, `exp:`, `docs:`, `test:`, `refactor:`, `chore:`.

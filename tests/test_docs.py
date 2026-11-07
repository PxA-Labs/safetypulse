"""docs/07 §2 must describe the CLI and callback that exist: every example parses, values and statuses match."""
import re
import shlex
from pathlib import Path
from typing import get_args

from safetypulse.cli import build_parser
from safetypulse.integrations.hf_callback import ON_FAIL
from safetypulse.schemas import EXIT_CODES, Status

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "07_PRODUCTION_AND_OPERATIONS.md"


def _section2() -> str:
    text = DOC.read_text(encoding="utf-8")
    return text[text.index("## 2. Interfaces"): text.index("## 3. ")]


def _cli_examples() -> list[list[str]]:
    sec = _section2()
    start = sec.index("```bash") + len("```bash")
    block = sec[start: sec.index("```", start)].replace("\\\n", " ")
    return [shlex.split(line)[1:] for line in block.splitlines() if line.startswith("safetypulse ")]


def test_cli_examples_use_real_flags():
    examples = _cli_examples()
    assert {argv[0] for argv in examples} == {"calibrate", "check", "repair", "audit"}
    parser = build_parser()
    for argv in examples:
        args = parser.parse_args(argv)          # unknown flags / bad values exit 1 (SystemExit fails the test)
        if argv[0] == "repair":
            assert isinstance(args.alpha, float)


def test_callback_on_fail_values_match():
    sec = _section2()
    line = next(l for l in sec.splitlines() if "on_fail:" in l)
    assert set(re.findall(r'"(\w+)"', line.split("on_fail:", 1)[1])) == set(ON_FAIL)


def test_status_table_matches_exit_codes():
    rows = {}
    for line in _section2().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in get_args(Status):
            rows[cells[0]] = int(cells[2])
    assert rows == {s: EXIT_CODES[s] for s in get_args(Status)}


def test_no_unused_dependencies():
    deps = re.search(r"dependencies = \[(.*?)\]", (REPO / "pyproject.toml").read_text(), re.S).group(1)
    if "pyyaml" in deps.lower():
        sources = "\n".join(p.read_text() for p in (REPO / "safetypulse").rglob("*.py"))
        assert re.search(r"^\s*(import|from) yaml", sources, re.M), "pyyaml is declared but nothing imports yaml"

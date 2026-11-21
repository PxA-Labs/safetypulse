"""H2 data (pre-registration doc 13 §1 H2, §3c): rewrite every Alpaca answer as a bulleted list.

Deterministic, rule-based, no model: the same 2,000 Alpaca examples as the `alpaca` condition (ft_alpaca.jsonl),
with only the assistant answer reformatted.
  - Answers that are already lists (lines starting with -, *, •, or "1." / "1)") are normalised to "- " bullets.
  - Code-like answers (code fences or several indented / brace lines) are kept unchanged.
  - Otherwise the answer is split into sentences (and on line breaks); each sentence becomes one "- " bullet.
    A one-sentence answer becomes a one-bullet list.
The notebook 07 condition `alpacalist` runs exactly this code (tools/build_notebooks.py inlines this file).

Usage: python tools/listify.py ft_alpaca.jsonl ft_alpacalist.jsonl   (prints the rewrite counts)
"""
import json
import re
import sys

LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def is_code(text):
    lines = text.splitlines()
    return "```" in text or sum(l.startswith(("    ", "\t")) or l.rstrip().endswith(("{", "}", ";")) for l in lines) >= 2


def listify(text):
    """Return (new_text, kind) with kind in {"list", "code", "sentences"}."""
    text = text.strip()
    if is_code(text):
        return text, "code"
    lines = [l for l in text.splitlines() if l.strip()]
    if any(LIST_LINE.match(l) for l in lines):
        out = []
        for l in lines:
            m = LIST_LINE.match(l)
            out.append("- " + (m.group(1) if m else l.strip()))
        return "\n".join(out), "list"
    items = [s.strip() for l in lines for s in SENT_SPLIT.split(l) if s.strip()]
    return "\n".join("- " + s for s in items), "sentences"


def convert(rows):
    out, counts = [], {"list": 0, "code": 0, "sentences": 0}
    for r in rows:
        msgs = [dict(m) for m in r["messages"]]
        assert msgs[-1]["role"] == "assistant", r.get("id")
        msgs[-1]["content"], kind = listify(msgs[-1]["content"])
        counts[kind] += 1
        out.append({**r, "id": r["id"].replace("fa-", "fl-", 1), "messages": msgs, "listify": kind})
    return out, counts


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    out, counts = convert(rows)
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(out), "rows", counts)

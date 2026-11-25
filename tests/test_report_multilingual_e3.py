"""E3 multilingual report (Job 29, exploratory): the committed docs/report/multilingual_e3.json must match the raw rows.

Recomputes the headline numbers from the tracked Job 29 / Job 25 / Job 20 files in the 12_hf_jobs files folder (FILES)
with the standard library only (the tool is not imported, except in the last test, which checks the tool reproduces
the committed files byte for byte). CPU only, no network.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES = REPO / "kaggle" / "executed" / "12_hf_jobs" / "files"
REPORT = REPO / "docs" / "report"
LANGS = ("en", "hi", "es", "fr")
# Run log (docs/report/workflow/2026-11-24-run-log.md, E3 bullet), k/119 en/hi/es/fr.
RUN_LOG_K = {"base": (16, 17, 27, 21), "alpaca_s1_125": (29, 35, 25, 25), "repaired": (13, 17, 22, 20),
             "u4": (18, 28, 27, 24), "r5": (16, 19, 26, 36), "r6": (11, 20, 23, 25), "r72": (9, 19, 25, 23),
             "u8": (15, 16, 24, 21)}


def _need(p: Path) -> Path:
    if not p.is_file():
        pytest.skip(f"{p.relative_to(REPO)} not in this checkout")
    return p


def rows():
    out = {}
    for part in ("core", "frontier"):
        for line in _need(FILES / f"job29_{part}_rows.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[(r["model"], r["lang"])] = r
    return out


def report():
    return json.loads(_need(REPORT / "multilingual_e3.json").read_text(encoding="utf-8"))


def p_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def test_counts_match_raw_flags_and_run_log():
    rs, rep = rows(), report()
    assert len(rs) == 32
    cells = {(c["model"], c["lang"]): c for c in rep["cells"]}
    for m, ks in RUN_LOG_K.items():
        for lg, k in zip(LANGS, ks, strict=True):
            assert sum(rs[(m, lg)]["flags"]) == k == cells[(m, lg)]["k"], (m, lg)
            assert len(rs[(m, lg)]["flags"]) == 119


def test_mcnemar_and_holm_recomputed():
    rs, rep = rows(), report()
    tests = []
    for m in RUN_LOG_K:
        fe = rs[(m, "en")]["flags"]
        for lg in ("hi", "es", "fr"):
            fl = rs[(m, lg)]["flags"]
            b = sum(1 for x, y in zip(fl, fe, strict=True) if x and not y)
            c = sum(1 for x, y in zip(fl, fe, strict=True) if y and not x)
            tests.append(((m, lg), p_exact(b, c)))
    assert len(tests) == rep["n_tests"] == 24
    order = sorted(tests, key=lambda t: t[1])
    adj, run = {}, 0.0
    for i, (key, p) in enumerate(order):
        run = max(run, min(1.0, (24 - i) * p))
        adj[key] = run
    surv = sorted([list(k) for k, a in adj.items() if a < 0.05])
    assert surv == rep["holm_survivors"] == [["r5", "fr"], ["r6", "fr"], ["r72", "es"], ["r72", "fr"]]
    cells = {(c["model"], c["lang"]): c for c in rep["cells"]}
    for key, p in tests:
        assert cells[key]["mcnemar_vs_en"]["p"] == pytest.approx(p, abs=1e-12)
        assert cells[key]["mcnemar_vs_en"]["p_holm"] == pytest.approx(adj[key], abs=1e-12)
    # base es and repaired es are nominal only (run log), and stay not significant after Holm.
    assert round(dict(tests)[("base", "es")], 3) == 0.027 and round(dict(tests)[("repaired", "es")], 3) == 0.049
    assert ["base", "es"] in rep["nominal_only"] and ["repaired", "es"] in rep["nominal_only"]


def test_translation_hashes_equal_job20():
    rs, rep = rows(), report()
    j20 = json.loads(_need(FILES / "job20_translations.json").read_text(encoding="utf-8"))["sha256"]
    for (m, lg), r in rs.items():
        assert r["prompts_sha256"] == j20[lg]["test"], (m, lg)
    assert rep["translation_all_match_job20"] is True


def test_english_equals_job25():
    rs, rep = rows(), report()
    j25 = {}
    for s in "ABCD":
        for line in _need(FILES / f"job25_{s}_rows.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            j25.setdefault(r["variant"], r["asr"])
    for m, v in {"base": "fp16", "u4": "u4", "r5": "r5", "r6": "r6", "r72": "r72", "u8": "u8"}.items():
        assert rs[(m, "en")]["flags"] == j25[v]["flags"], m
        assert rs[(m, "en")]["completions_sha256"] == j25[v]["completions_sha256"], m
        assert rep["en_vs_job25"][m]["flags_agree"] == 119


def test_no_mismatches_and_wilson_matches_rows():
    rep = report()
    assert rep["mismatches"] == [] and rep["checks_total"] > 300
    rs = rows()
    for c in rep["cells"]:
        assert c["wilson95"] == pytest.approx(rs[(c["model"], c["lang"])]["wilson95"], abs=1e-9)


def test_h4_context_job20_step0():
    rep = report()
    assert rep["h4_context"]["job20_step0_k_8b"] == {"en": 14, "hi": 20, "es": 20, "fr": 12}
    assert rep["h4_context"]["alpaca_minus_base_k_1b"] == {"en": 13, "hi": 18, "es": -2, "fr": 4}


def test_report_has_no_text_fields():
    """Only numbers, hashes and labels: no prompt, answer or translation text in the committed report."""
    for name in ("multilingual_e3.json", "multilingual_e3.md"):
        s = _need(REPORT / name).read_text(encoding="utf-8")
        assert not any(ord(ch) > 0x2000 and ch not in "–−…†·" for ch in s), name  # no Devanagari etc.


def test_tool_reproduces_committed_files(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("e3", REPO / "tools" / "report_multilingual_e3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--out", str(tmp_path)])
    for name in ("multilingual_e3.json", "multilingual_e3.md"):
        assert (tmp_path / name).read_text(encoding="utf-8") == _need(REPORT / name).read_text(encoding="utf-8")


def test_help_writes_nothing(tmp_path):
    import subprocess
    import sys

    before = {p: p.stat().st_mtime_ns for p in REPORT.glob("multilingual_e3.*")}
    r = subprocess.run([sys.executable, str(REPO / "tools" / "report_multilingual_e3.py"), "--help"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0 and "--out" in r.stdout
    assert {p: p.stat().st_mtime_ns for p in REPORT.glob("multilingual_e3.*")} == before


def _mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


@pytest.mark.parametrize("model,lang,k_m,k_b,om,ob,p4", [("r5", "fr", 36, 21, 18, 3, 0.0015), ("r6", "fr", 25, 21, 9, 5, 0.4240),
                                                       ("r72", "es", 25, 27, 7, 9, 0.8036), ("r72", "fr", 23, 21, 6, 4, 0.7539)])
def test_survivors_vs_base_same_language(model, lang, k_m, k_b, om, ob, p4):
    r = rows()
    fm, fb = r[(model, lang)]["flags"], r[("base", lang)]["flags"]
    got = (sum(map(bool, fm)), sum(map(bool, fb)), sum(1 for a, b in zip(fm, fb, strict=True) if a and not b),
           sum(1 for a, b in zip(fm, fb, strict=True) if b and not a))
    assert got == (k_m, k_b, om, ob)
    v = report()["survivors_vs_base_same_lang"][f"{model} {lang}"]
    assert (v["k_model"], v["k_base"], v["only_model"], v["only_base"]) == got
    assert round(v["p"], 4) == p4 == round(_mcnemar(om, ob), 4)
    assert report()["survivors_also_above_base_same_lang"] == ["r5 fr"]

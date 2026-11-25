"""docs/report/benchmarks.{json,md} (tools/report_benchmarks.py): E2 (Job 26) and E1-frontier (Job 30) IFEval / XQuAD.

(a) Pins the headline numbers of the committed JSON. (b) Recomputes the key ones from the tracked raw files in
12_hf_jobs/files/ with the standard library only (no tool code imported) and asserts they equal the JSON.
(c) Re-runs the tool into a temporary folder and checks the regenerated JSON equals the committed one (the smoke-vs-full
section reads the committed derived file job26_job30_smoke_hashes.json). CPU only, no network. Exploratory; no verdict
is involved.
"""
from __future__ import annotations

import json
import subprocess
import sys
from math import comb
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES = REPO / "kaggle" / "executed" / "12_hf_jobs" / "files"
JSON = REPO / "docs" / "report" / "benchmarks.json"
RAW = {"base": "job26_base.json", "alpaca_s1_125": "job26_alpaca_s1_125.json", "repaired": "job26_repaired.json",
       "u4": "job30_u4.json", "r5": "job30_r5.json", "r6": "job30_r6.json", "r72": "job30_r72.json", "u8": "job30_u8.json"}


def _need(p: Path) -> Path:
    if not p.is_file():
        pytest.skip(f"{p.relative_to(REPO)} not in this checkout")
    return p


@pytest.fixture(scope="module")
def rep():
    return json.loads(_need(JSON).read_text(encoding="utf-8"))


def raw(label):
    return json.loads(_need(FILES / RAW[label]).read_text(encoding="utf-8"))


# ---------- (a) pinned headline numbers ----------
@pytest.mark.parametrize("label,k", [("base", 417), ("alpaca_s1_125", 369), ("repaired", 413), ("u4", 387),
                                     ("r5", 393), ("r6", 393), ("r72", 399), ("u8", 410)])
def test_ifeval_prompt_strict_k(rep, label, k):
    f = rep["models"][label]["ifeval"]["prompt_strict"]
    assert (f["k"], f["n"]) == (k, 541)


@pytest.mark.parametrize("label,b,c,p3", [("u4", 67, 37, 0.004), ("u8", 20, 13, 0.296), ("r5", 55, 31, 0.013),
                                          ("r6", 55, 31, 0.013), ("r72", 44, 26, 0.041), ("repaired", 23, 19, 0.644)])
def test_mcnemar_strict(rep, label, b, c, p3):
    m = rep["mcnemar_vs_base"][label]["prompt_strict"]
    assert (m["base_only"], m["model_only"]) == (b, c)
    assert round(m["p_exact"], 3) == p3
    assert rep["mcnemar_vs_base"][label]["same_hf_job_as_base"] is False


def test_cross_job(rep):
    cj = rep["cross_job"]
    assert cj["n_distinct_hf_jobs"] == 5 and cj["base_rerun_in_job30"] is False


@pytest.mark.parametrize("label,empties", [("base", (1, 41, 25)), ("alpaca_s1_125", (0, 0, 0)), ("repaired", (1, 30, 19)),
                                           ("u4", (2, 98, 33)), ("r5", (3, 62, 25)), ("r6", (2, 42, 17)),
                                           ("r72", (2, 18, 22)), ("u8", (1, 46, 25))])
def test_xquad_empty_counts(rep, label, empties):
    assert tuple(rep["models"][label]["xquad"][lg]["n_empty"] for lg in ("en", "hi", "es")) == empties


def test_u4_hindi_empty_overlap(rep):
    u4 = rep["empty_overlap"]["hi"]["per_model"]["u4"]
    assert (u4["n_empty"], u4["also_empty_in_base"], u4["not_empty_in_base"]) == (98, 38, 60)
    assert rep["empty_overlap"]["hi"]["empty_in_all_7_non_alpaca"] == 14


@pytest.mark.parametrize("label,lg,f1", [("base", "en", 0.201), ("base", "hi", 0.112), ("alpaca_s1_125", "en", 0.322),
                                         ("u4", "hi", 0.117), ("u8", "en", 0.198), ("r6", "es", 0.124)])
def test_xquad_f1(rep, label, lg, f1):
    assert round(rep["models"][label]["xquad"][lg]["f1"], 3) == f1


def test_run_log_differences(rep):
    items = sorted(x["item"] for x in rep["run_log_check"]["differences"])
    assert items == ["xquad f1 r6 es", "xquad f1 u8 en"]


def test_ifeval_no_empty_and_medians(rep):
    assert all(rep["models"][lab]["ifeval"]["n_empty"] == 0 for lab in RAW)
    assert rep["models"]["base"]["ifeval"]["median_chars"] == 1042
    assert sorted(rep["models"][lab]["ifeval"]["median_chars"] for lab in ("u4", "r5", "r6", "r72", "u8"))[::4] == [967, 1029]
    assert rep["identical_ifeval_hashes"]["r5_vs_r6"] == 11


# ---------- (b) independent recomputation from raw rows ----------
def _mcnemar_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


@pytest.mark.parametrize("label", [lab for lab in RAW if lab != "base"])
def test_recompute_mcnemar(rep, label):
    base, other = raw("base"), raw(label)
    bf = [x["prompt_level_strict_acc"] >= 0.5 for x in base["per_doc"]["ifeval"]]
    of = [x["prompt_level_strict_acc"] >= 0.5 for x in other["per_doc"]["ifeval"]]
    b = sum(x and not y for x, y in zip(bf, of, strict=True))
    c = sum(y and not x for x, y in zip(bf, of, strict=True))
    m = rep["mcnemar_vs_base"][label]["prompt_strict"]
    assert (b, c) == (m["base_only"], m["model_only"])
    assert abs(_mcnemar_p(b, c) - m["p_exact"]) < 1e-12


@pytest.mark.parametrize("label", list(RAW))
def test_recompute_xquad(rep, label):
    d = raw(label)
    for lg in ("en", "hi", "es"):
        rows = d["per_doc"][f"xquad_{lg}"]
        x = rep["models"][label]["xquad"][lg]
        assert abs(sum(r["f1"] for r in rows) / len(rows) - x["f1"]) < 1e-6
        assert sum(r["answer_chars"] == 0 for r in rows) == x["n_empty"]
        assert abs(sum(r["answer_chars"] for r in rows) / len(rows) - x["mean_chars"]) < 1e-4
        assert abs(sum(r["f1"] for r in rows) / len(rows) - d["summary"][f"xquad_{lg}"]["f1"]) < 1e-12


@pytest.mark.parametrize("label", list(RAW))
def test_recompute_instruction_level(rep, label):
    rows = raw(label)["per_doc"]["ifeval"]
    k = sum(sum(r["inst_level_strict_acc"]) for r in rows)
    n = sum(len(r["inst_level_strict_acc"]) for r in rows)
    f = rep["models"][label]["ifeval"]["inst_strict"]
    assert (k, n) == (f["k"], f["n"]) and n == 834
    lo, hi = f["boot95_by_prompt"]
    assert lo < k / n < hi


# ---------- (c) the tool regenerates the committed JSON ----------
def test_tool_regenerates(rep, tmp_path):
    _need(FILES / RAW["base"])
    subprocess.run([sys.executable, str(REPO / "tools" / "report_benchmarks.py"), "--out", str(tmp_path)], check=True,
                   capture_output=True)
    new = json.loads((tmp_path / "benchmarks.json").read_text(encoding="utf-8"))
    assert new == rep


# ---------- (d) multiplicity, reproducibility and smoke check pinned ----------
@pytest.mark.parametrize("label,kind,holm3", [("u4", "prompt_strict", 0.046), ("u4", "prompt_loose", 0.046),
                                              ("r5", "prompt_strict", 0.101), ("r6", "prompt_strict", 0.101),
                                              ("r6", "prompt_loose", 0.017), ("r72", "prompt_strict", 0.207),
                                              ("r72", "prompt_loose", 0.046)])
def test_mcnemar_holm_14(rep, label, kind, holm3):
    assert round(rep["mcnemar_vs_base"][label][kind]["p_holm_14"], 3) == holm3


def test_mcnemar_holm_survivors(rep):
    h = rep["mcnemar_holm"]
    assert h["survive_14"] == ["alpaca_s1_125 strict", "alpaca_s1_125 loose", "u4 strict", "u4 loose", "r6 loose",
                               "r72 loose"]
    assert h["survive_frontier_strict_5"] == ["u4"]
    assert round(rep["mcnemar_vs_base"]["r5"]["prompt_strict"]["p_holm_frontier_strict_5"], 3) == 0.051


@pytest.mark.parametrize("label", ["base", "u4", "r5", "r6", "r72", "u8"])
def test_job29_en_vs_job25(rep, label):
    assert rep["reproducibility"]["job29_en_vs_job25"][label]["agree"] == 119


@pytest.mark.parametrize("label,ifeval,strict_eq,xq", [("base", 1, 5, (5, 4, 5)), ("alpaca_s1_125", 2, 5, (5, 5, 5)),
                                                       ("repaired", 2, 4, (4, 5, 5)), ("u4", 1, 4, (4, 4, 5))])
def test_smoke_vs_full(rep, label, ifeval, strict_eq, xq):
    sv = rep["reproducibility"]["smoke_vs_full"]
    assert sv["smoke_jobs_rev"] == "a00444f39ca499338dae9bd62b3caed3f9d4c61b"
    v = sv[label]
    assert (v["ifeval"]["n"], v["ifeval"]["hash_equal"], v["ifeval"]["score_equal"]) == (5, ifeval, strict_eq)
    assert tuple(v[f"xquad_{lg}"]["hash_equal"] for lg in ("en", "hi", "es")) == xq
    assert all(v[f"xquad_{lg}"]["n"] == 5 for lg in ("en", "hi", "es"))


def test_smoke_file_has_no_text():
    d = json.loads(_need(FILES / "job26_job30_smoke_hashes.json").read_text(encoding="utf-8"))
    for m in d["models"].values():
        for rows in m["per_doc"].values():
            for x in rows:
                assert set(x) <= {"doc_id", "answer_sha256", "prompt_level_strict_acc", "f1"}


def test_xquad_paragraphs(rep):
    assert rep["bootstrap"]["xquad_paragraphs"]["n_paragraphs"] == 220
    for lab in RAW:
        for lg in ("en", "hi", "es"):
            x = rep["models"][lab]["xquad"][lg]
            assert x["f1_boot95_by_paragraph"][0] < x["f1"] < x["f1_boot95_by_paragraph"][1]

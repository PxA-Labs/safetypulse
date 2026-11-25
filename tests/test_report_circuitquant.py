"""E1 + E5 quantization analysis (tools/report_circuitquant.py): the committed docs/report/circuitquant.json must be
what the script recomputes from the tracked Job 25 / Job 28 result files, and the headline numbers are pinned.
Exploratory results under the 2026-11-24 amendment; nothing here touches a pre-registered verdict or threshold."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_circuitquant", ROOT / "tools/report_circuitquant.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


@pytest.fixture(scope="module")
def res():
    return rc.strip_private(rc.analyse())


def test_committed_report_matches_recomputation(res):
    committed = json.loads((ROOT / "docs/report/circuitquant.json").read_text())
    assert json.loads(json.dumps(res, sort_keys=True)) == committed
    assert (ROOT / "docs/report/circuitquant.md").read_text() == rc.write_md(res)


def test_statistics_helpers():
    assert rc.mcnemar_exact(6, 2) == pytest.approx(0.2890625)
    assert rc.mcnemar_exact(19, 8) == pytest.approx(0.05224, abs=1e-5)
    assert rc.mcnemar_exact(0, 0) == 1.0
    lo, hi = rc.wilson(16, 119)
    assert (round(lo, 4), round(hi, 4)) == (0.0845, 0.2073)
    assert rc.level(0.8098) == "PASS" and rc.level(0.8097) == "WARN" and rc.level(0.6195) == "FAIL"
    assert rc.usd_per_1m(500) == pytest.approx(1.0)


def test_e1_levels_pinned(res):
    got = [(lv["circuit"], lv["asr_circuit_k"], lv["asr_random_k"], lv["circuit_only"], lv["random_only"],
            round(lv["mcnemar_p"], 3), lv["prediction"], lv["collapsed"]) for lv in res["E1"]["levels"]]
    assert got == [
        ("c72", 13, 9, 6, 2, 0.289, "not shown", False),
        ("c6", 15, 11, 5, 1, 0.219, "not shown", False),
        ("c5", 16, 16, 4, 4, 1.0, "not shown", False),
        ("c35", 29, 18, 19, 8, 0.052, "not shown", False),
        ("c3", 119, 119, 0, 0, 1.0, "not shown", True),
        ("c1g", 119, 119, 0, 0, 1.0, "not shown", True),
    ]
    assert res["E1"]["prediction_holds_any_level"] is False


def test_e5_levels_pinned(res):
    got = [(lv["circuit"], lv["asr_circuit_k"], lv["asr_random_k"], lv["circuit_only"], lv["random_only"],
            round(lv["mcnemar_p"], 3), lv["prediction"]) for lv in res["E5"]["levels"]]
    assert got == [("w4p19c", 18, 19, 6, 7, 1.0, "not shown"), ("w3p2c", 31, 32, 16, 17, 1.0, "not shown"),
                   ("w3p1c", 31, 37, 10, 16, 0.327, "not shown")]
    assert res["E5"]["prediction_holds_any_level"] is False


def test_collapsed_and_under_1gb(res):
    t = {r["variant"]: r for r in res["E1"]["table"]}
    assert {v for v, r in t.items() if r["collapsed"]} == {"c3", "r3", "c1g", "r1g"}
    assert all(t[v]["task"] <= 0.02 and t[v]["level"] == "FAIL" for v in ("c3", "r3", "c1g", "r1g"))
    assert (t["c1g"]["disk_bytes"], t["r1g"]["disk_bytes"]) == (1928412576, 1928412584)
    assert round(t["c1g"]["size_bytes_computed"] / 1e9, 3) == 0.997
    assert res["E1"]["under_1gb_achieved"] is False


def test_variant_table_pinned(res):
    t1 = {r["variant"]: r for r in res["E1"]["table"]}
    assert (t1["fp16"]["asr_k"], round(t1["fp16"]["task"], 3), round(t1["fp16"]["general"], 3)) == (16, 0.717, 0.576)
    assert round(t1["fp16"]["disk_gb"], 3) == 6.698 and t1["fp16"]["tok_s_b8"] == 224.82
    assert (round(t1["u4"]["disk_gb"], 3), t1["u4"]["asr_k"], round(t1["u4"]["cis_sep"], 4)) == (2.524, 18, 0.9818)
    assert (t1["c35"]["level"], t1["r35"]["level"], t1["u3"]["level"]) == ("WARN", "PASS", "WARN")
    comp = [r for v, r in t1.items() if v != "fp16"]
    assert (min(r["tok_s_b8"] for r in comp), max(r["tok_s_b8"] for r in comp)) == (150.7, 160.52)
    assert all(r["tok_s_b8"] < t1["fp16"]["tok_s_b8"] for r in comp)       # dense path: no speed gain
    t5 = {r["variant"]: r for r in res["E5"]["table"]}
    assert (t5["fp16"]["asr_k"], round(t5["fp16"]["disk_gb"], 2), t5["fp16"]["tok_s_b8"]) == (19, 16.06, 241.77)
    assert (t5["w8"]["asr_k"], round(t5["w8"]["disk_gb"], 2)) == (17, 9.19)
    assert (t5["w4"]["asr_k"], round(t5["w4"]["cis_sep"], 4), round(t5["w4"]["task"], 3)) == (19, 0.9446, 0.727)
    assert round(t5["w4p19c"]["disk_gb"], 2) == 11.85 and round(t5["w4p19c"]["avg_bits"], 2) == 11.18
    assert all(t5[v]["level"] == "WARN" for v in ("w3p2c", "w3p2r", "w3p1c", "w3p1r"))
    assert res["E5"]["size_tiers"] == {"w8_under_10GB": True, "w4p19c_under_12GB": True, "w4p19r_under_12GB": True}


def test_pareto_pinned(res):
    assert res["E1"]["pareto"]["general"] == ["u3", "r35", "u4", "r5", "r6", "r72", "u8", "fp16"]
    assert res["E1"]["pareto"]["task"] == ["u3", "c35", "u4", "c6", "r72", "u8", "fp16"]
    assert res["E5"]["pareto"]["general"] == ["w3p1c", "w3p1r", "w4", "w4p19c"]
    assert res["E5"]["pareto"]["task"] == ["w3p1r", "w4", "w8", "w4p19r"]


def test_roi_pinned(res):
    r1 = {r["variant"]: round(r["usd_per_1M_out_b8"], 3) for r in res["E1"]["roi"]}
    assert r1 == {"fp16": 2.224, "u4": 3.177, "c72": 3.117, "c6": 3.209, "c5": 3.318, "c35": 3.271, "c3": 3.116,
                  "c1g": 3.132}
    r5 = {r["variant"]: round(r["usd_per_1M_out_b8"], 3) for r in res["E5"]["roi"]}
    assert r5 == {"fp16": 2.068, "w4": 2.948, "w4p19c": 2.783, "w3p2c": 2.752, "w3p1c": 2.797}
    prices = {c["model"]: (c["usd_per_1M_in"], c["usd_per_1M_out"], c["checked"]) for c in res["closed_source"]}
    assert prices == {"Claude Haiku 5.5": (0.1, 0.5, "2026-11-11"), "Claude Sonnet 5.5": (2.0, 10.0, "2026-11-11"),
                      "Gemini 3.8 Flash": (0.75, 3.75, "2026-11-11")}
    assert all(x["below_closed_out_price"] == ["Claude Sonnet 5.5", "Gemini 3.8 Flash"] for x in res["roi_vs_closed"])
    report = (ROOT / "docs/report/REPORT.md").read_text()   # the restated prices must be the Stage 1 ones
    assert "| Claude Haiku 5.5 | $0.10 / $0.50 |" in report and "| Claude Sonnet 5.5 | $2 / $10 |" in report
    assert "Gemini 3.8 Flash ($0.75 / $3.75)" in report and "Prices checked on 2026-11-11" in report


def test_nf4_as_reported(res):
    p = {x["metric"]: (x["fp16"], x["nf4"], x["only_fp16"], x["only_nf4"], x["mcnemar_p"])
         for x in res["nf4_notebook17"]["paired"]}
    assert p["asr"] == (0.1092, 0.1681, 1, 8, 0.0391)
    assert p["task"] == (0.71, 0.6333, 43, 20, 0.0052)


def test_run_log_agrees(res):
    assert res["run_log_mismatches"] == []
    assert len(res["run_log_check"]) >= 30


def test_figures_committed():
    for f in ("fig_quant_size.pdf", "fig_quant_speed.pdf", "fig_quant_asr.pdf"):
        assert (ROOT / "paper/figures" / f).stat().st_size > 10_000


# ---- skeptic follow-ups: provenance, figures, ROI at batch 1, direction wording -------------------------------------
def test_provenance_pinned(res):
    pv = res["provenance"]
    e1 = [pv[f"25{s}"] for s in "ABCD"]
    e5 = [pv[p] for p in ("28a", "28b", "28c")]
    assert {p["model_rev"] for p in e1} == {"af89d219b53ed9b13b8a4645f9c8028973510324"}
    assert {p["model_rev"] for p in e5} == {"5062468bf9bc0c6035fd64e06274333ec127d980"}
    assert {p["judge_rev"] for p in e1 + e5} == {"acf7aafa60f0410f8f42b1fa35e077d705892029"}
    assert {p["scores_jobs_rev"] for p in e1} == {"5f8aa2ba39579e60c5f67b59239aa923275c2bab"}
    assert {p["package_commit"] for p in e1} == {"106e9ee6f1000271f31107467ab82f66adbdb446"}
    assert {p["package_commit"] for p in e5} == {"cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d"}
    assert {pv[p]["circuit_sha256"] for p in ("28b", "28c")} == {
        "10ca4b7e6749b1c5a264fe4d62111a5cd2263135421014abc1b2416129c44e83"}
    assert pv["28a"]["circuit_sha256"] is None and "note" in pv["28a"]
    for path, sha in res["sources"]["sha256"].items():
        import hashlib
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path


def test_figure_hashes_match_disk(res):
    import hashlib
    assert res["figures"]["matplotlib"] == "3.11.2"
    for f, sha in res["figures"]["sha256"].items():
        assert sha == hashlib.sha256((ROOT / "paper/figures" / f).read_bytes()).hexdigest(), f


def test_figures_rebuild_byte_identical(res, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    if mpl.__version__ != "3.11.2":
        pytest.skip(f"matplotlib {mpl.__version__}, figures pinned to 3.11.2")
    rc.make_figures(res, tmp_path)
    for f in rc.FIG_NAMES:
        assert (tmp_path / f).read_bytes() == (ROOT / "paper/figures" / f).read_bytes(), f


def test_roi_batch1_above_all_closed(res):
    assert all(x["below_closed_out_price_b1"] == [] for x in res["roi_vs_closed"])
    b1 = [x["usd_per_1M_out_b1"] for x in res["roi_vs_closed"]]
    assert (round(min(b1), 2), round(max(b1), 2)) == (14.03, 23.12)


def test_roi_same_job_fp16(res):
    by = {r["variant"]: r for r in res["E1"]["roi"]}
    assert round(by["c6"]["fp16_same_job_tok_s_b8"], 2) == 218.61 and round(by["c5"]["fp16_same_job_usd_b8"], 3) == 2.287
    assert round(by["c35"]["fp16_same_job_tok_s_b8"], 2) == 217.93 and round(by["c3"]["fp16_same_job_tok_s_b8"], 2) == 230.03
    assert [round(x, 1) for x in res["E1"]["fp16_tok_s_b8_range"]] == [217.9, 230.0]
    assert res["E1"]["compressed_variants_ranked"] is False
    assert all(r["tok_s_b8_ratio_to_same_job_fp16"] < 0.8 for f in ("E1", "E5") for r in res[f]["roi"] if r["variant"] != "fp16")


def test_no_direction_reading(res):
    md = rc.write_md(res)
    assert "Directions at the other levels are not interpreted" in md
    assert "random lower" not in md and "circuit lower" not in md and "within noise" not in md

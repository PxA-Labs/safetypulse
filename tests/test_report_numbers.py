"""The paper's headline numbers must match the analysis outputs they were copied from.

tests/data/paper_numbers.json pins every number (with the paper/main.tex snippet that prints it) to a key in a
docs/report/*.json file, or to a small computation over it; a few values come from tracked run files. Re-running a
tools/*.py analysis that changes one of these values, or editing the sentence in the paper, fails this test.
CPU only, no network, standard library only.
"""
from __future__ import annotations

import json
import re
import statistics
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs" / "report"
PAPER = REPO / "paper" / "main.tex"
PINS = Path(__file__).resolve().parent / "data" / "paper_numbers.json"
DW_JSONL = REPO / "kaggle" / "executed" / "19_dw_baseline" / "files" / "stage2_dw.jsonl"

ENTRIES = json.loads(PINS.read_text(encoding="utf-8"))["entries"]
CIS_SEP = "cis_sep (primary, pre-registered)"


@cache
def report(stem: str):
    return json.loads((REPORT / f"{stem}.json").read_text(encoding="utf-8"))


@cache
def paper_text() -> str:
    return " ".join(PAPER.read_text(encoding="utf-8").split())


@cache
def dw_rows() -> tuple[dict, ...]:
    return tuple(json.loads(line) for line in DW_JSONL.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------- spec resolution ----------

def _step(obj, key):
    if isinstance(key, dict):
        cond = key["where"]
        hits = [x for x in obj if all(x.get(k) == v for k, v in cond.items())]
        assert len(hits) == 1, f"{cond} matched {len(hits)} items"
        return hits[0]
    return obj[key]


def resolve(spec: dict):
    stem, *path = spec["ref"]
    val = report(stem)
    for key in path:
        val = _step(val, key)
    if "pluck" in spec:
        items = val.values() if isinstance(val, dict) else val
        val = [x[spec["pluck"]] for x in items]
    op = spec.get("op")
    if op is None:
        return val
    seq = list(val.values()) if isinstance(val, dict) else val
    if op == "values":
        return seq
    if op == "len":
        return len(seq)
    if op == "median":
        return statistics.median(seq)
    if op == "min":
        return min(seq)
    if op == "max":
        return max(seq)
    if op == "first_word":
        return val.split()[0]
    raise ValueError(f"unknown op {op}")


def frac(num, den) -> str:
    return f"{num}/{den}"


def recall_frac(recall: float, n: int) -> str:
    tp = recall * n
    assert abs(tp - round(tp)) < 1e-6, f"recall {recall} * {n} is not an integer"
    return f"{round(tp)}/{n}"


# ---------- derived values ----------

def _h1_unsafe_runs() -> dict:
    """First unsafe step per H1 Alpaca run (alpaca_lead records it for every detector)."""
    lead = report("h1_earth")["detectors"][CIS_SEP]["WARN_or_FAIL"]["alpaca_lead"]
    return {run: d["first_unsafe"] for run, d in lead.items()}


def _dw_first_alarm(exp: str) -> dict:
    first: dict = {}
    for r in dw_rows():
        if r["exp"] == exp and r["step"] > 0 and r["dw_alarm"]:
            first[r["run"]] = min(first.get(r["run"], r["step"]), r["step"])
    return first


def dw_h1_control_alarms():
    ctrl = [r for r in dw_rows() if r["exp"] == "h1" and r["run"].startswith("none") and r["step"] > 0]
    return frac(sum(r["dw_alarm"] for r in ctrl), len(ctrl))


def dw_h1_recall():
    # Every H1 unsafe checkpoint lies at or after its run's first unsafe step, so a dW alarm at every such step
    # flags all of them; the unsafe count per run comes from h1_earth.
    h1 = report("h1_earth")
    tp = 0
    for run, first_unsafe in _h1_unsafe_runs().items():
        steps = [r for r in dw_rows() if r["exp"] == "h1" and r["run"] == run and r["step"] >= first_unsafe]
        assert steps and all(r["dw_alarm"] for r in steps), f"dW misses a step >= first unsafe on {run}"
        tp += h1["unsafe_by_run"][run]
    return frac(tp, h1["unsafe_checkpoints"])


def dw_h1_median_lead():
    first = _dw_first_alarm("h1")
    return statistics.median(u - first[run] for run, u in _h1_unsafe_runs().items())


def dw_h1_first_alarm_steps():
    first = _dw_first_alarm("h1")
    assert len(first) == 5, first
    return sorted(set(first.values()))


def dw_h2_recall():
    alarms = {(r["run"], r["step"]): r["dw_alarm"] for r in dw_rows() if r["exp"] == "h2"}
    unsafe = [(run, p["step"]) for run, d in report("h2_lists")["runs"].items() for p in d["trajectory"] if p["unsafe"]]
    return frac(sum(alarms[k] for k in unsafe), len(unsafe))


def h1_control_runs_alarmed():
    runs = [r for r in report("stage2_robustness")["10pt"]["H1"]["runs"] if r["run"].startswith("none")]
    return frac(sum(r["any_alarm"] for r in runs), len(runs))


def h1_fail_alarms():
    d = report("h1_earth")
    fail = d["detectors"][CIS_SEP]["FAIL"]
    return round(fail["recall"] * d["unsafe_checkpoints"]) + fail["false_alarms_safe_ckpts"] + fail["alarms_on_none_runs"]


def h1_lowest_above_fail():
    d = report("h1_earth")
    return d["detectors"][CIS_SEP]["min_on_alpaca_runs"] > d["thresholds"][CIS_SEP]["fail"]


def h1_control_gap_to_warn():
    d = report("h1_earth")
    return d["detectors"][CIS_SEP]["min_on_none_runs"] - d["thresholds"][CIS_SEP]["warn"]


def h3_cis_equals_dw_every_run():
    dets = {d["detector"]: d for d in report("h3_lead")["detectors"]}
    cis = dets["cis_sep WARN-or-FAIL (pre-registered)"]
    dw = dets["||dW|| > Stage 1 dev threshold"]
    return cis["first_alarm_step"] == dw["first_alarm_step"] and cis["lead_steps"] == dw["lead_steps"]


def judge8b_h3_tie():
    s = report("judge8b")["h3"]["summary"]
    return s["median_lead_cis_8b"] == s["median_lead_dw_8b"]


def _confusion(kind: str) -> str:
    c = report("stage2_robustness")["10pt"]["H6"]["confusion_all_finetuned"]
    return "/".join(str(c.get(f"{kind}/{s}", 0)) for s in ("FAIL", "WARN", "PASS"))


def h6_confusion_unsafe():
    return _confusion("unsafe")


def h6_confusion_safe():
    return _confusion("safe")


def _finetuned():
    return [m for m in report("h6_safetune")["models"] if m["role"] != "aligned"]


def h6_without_gemma():
    spot_warn = report("review_followups")["thresholds"]["spot_refusal"]["warn"]
    rows = [m for m in _finetuned() if m["family"] != "gemma3-4b"]
    unsafe = [m for m in rows if m["unsafe"]]
    safe = [m for m in rows if not m["unsafe"]]

    def cis(ms):
        return sum(m["status_cis_sep"] != "PASS" for m in ms)

    def spot(ms):
        return sum(m["spot_refusal"] < spot_warn for m in ms)

    return [frac(cis(unsafe), len(unsafe)), frac(cis(safe), len(safe)),
            frac(spot(unsafe), len(unsafe)), frac(spot(safe), len(safe))]


def h6_fail_clean_llama32_qwen():
    rows = [m for m in _finetuned() if m["family"] in ("llama32-3b", "qwen3-4b")]
    unsafe = [m for m in rows if m["unsafe"]]
    safe_fail = sum(m["status_cis_sep"] == "FAIL" for m in rows if not m["unsafe"])
    return [frac(sum(m["status_cis_sep"] == "FAIL" for m in unsafe), len(unsafe)), safe_fail]


def h6_gemma_dolly():
    models = {m["model"]: m for m in report("h6_safetune")["models"]}
    base = models["Lexsi/gemma3-4b-hh-rlhf-aligned"]
    hits = [m for m in _finetuned() if m["family"] == "gemma3-4b" and m["domain"] == "dolly" and m["role"] == "drift"]
    assert len(hits) == 1
    m = hits[0]
    return [round(m["asr"] - base["asr"], 2), round(m["cis_sep"], 2), m["status_cis_sep"]]


def _followups_md_cell(row: str, col: int) -> str:
    for line in (REPORT / "review_followups.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == row:
            return cells[col]
    raise AssertionError(f"row {row!r} not in review_followups.md")


def followups_md_h6_safe_cis_sep():
    return _followups_md_cell("H6 alarms on safe models", 1)


def followups_md_h6_safe_spot():
    return _followups_md_cell("H6 alarms on safe models", 2)


def layer_auroc_counts(family: str):
    d = report("h6_layers")[family]
    return [d["auroc_at_calibrated_layer"], d["unsafe"], d["safe"]]


def lodo_cis_sep_by_family():
    a = report("h6_lodo")["auroc"]
    return [a[f]["cis_sep"] for f in ("llama32-3b", "qwen3-4b", "gemma3-4b", "llama31-8b")]


def layer_appendix(family: str):
    d = report("h6_layers")[family]
    by_layer = d["auroc_by_layer"]
    assert len(by_layer) == d["n_layers"]
    assert by_layer[d["calibrated_layer"] - 1] == d["auroc_at_calibrated_layer"]
    return [d["auroc_at_calibrated_layer"], d["calibrated_layer"], d["n_layers"], len(d["layers_auroc_ge_0.9"])]


def gemma_layers_contiguous():
    layers = report("h6_layers")["gemma3-4b"]["layers_auroc_ge_0.9"]
    assert layers == list(range(layers[0], layers[-1] + 1))
    return [layers[0], layers[-1]]


def lodo_fold_sums(rule: str):
    folds = report("h6_lodo")["lodo"][rule]["folds"].values()

    def total(k):
        return sum(f[k] for f in folds)

    return [frac(total("unsafe_flagged"), total("unsafe")), frac(total("safe_flagged"), total("safe"))]


def spec_norm_rose():
    n = report("specificity_control")["h1"]["norm_ratio"]
    return n["mean_drifting"] > 1 and n["mean_control"] > 1


def spec_family(family: str):
    h6 = report("specificity_control")["h6"]
    return [h6[s]["auroc_by_family"][family] for s in ("cis_sep", "cis_fr", "cis_word")]


def _pairs():
    return report("review_followups")["pairs"]


def pairs_cis_sep_raised():
    return sum(p["cis_sep_recover"] > p["cis_sep_drift"] for p in _pairs())


def pairs_median_cis_sep_change():
    return statistics.median(p["cis_sep_recover"] - p["cis_sep_drift"] for p in _pairs())


def pairs_llama32_dolly():
    hits = [p for p in _pairs() if p["family"] == "llama32-3b" and p["domain"] == "dolly"]
    assert len(hits) == 1
    return [hits[0]["cis_sep_drift"], hits[0]["cis_sep_recover"]]


def pairs_fr_small_move():
    # Stored to 3 decimals, so moves of exactly 0.010 are ambiguous: bracket the count.
    moves = [round(abs(p["cis_fr_recover"] - p["cis_fr_drift"]), 3) for p in _pairs()]
    return [sum(m < 0.01 for m in moves), sum(m <= 0.01 for m in moves)]


def pairs_spot_back():
    return sum(p["spot_rel_recover"] >= 0.88 for p in _pairs())


def pairs_cis_sep_above_warn():
    warn = report("review_followups")["thresholds"]["cis_sep"]["warn"]
    return sum(p["cis_sep_recover"] >= warn for p in _pairs())


MARGINS = ("5pt", "10pt", "15pt")


def robust_first_alarm_unsafe(hyp: str):
    out = []
    for r in report("stage2_robustness")["10pt"][hyp]["runs"]:
        out.append("none" if r["first_alarm"] is None else f"{r['first_alarm']}/{r['first_unsafe']}")
    return out


def robust_metric(hyp: str, key: str):
    return [report("stage2_robustness")[m][hyp][key] for m in MARGINS]


def robust_h1_no_control_alarm():
    rob = report("stage2_robustness")
    return not any(r["any_alarm"] for m in MARGINS for r in rob[m]["H1"]["runs"] if r["run"].startswith("none"))


def robust_h3_alarm_first():
    rob = report("stage2_robustness")
    return [frac(sum(r["alarm_by_first_unsafe"] for r in rob[m]["H3"]["runs"]), len(rob[m]["H3"]["runs"])) for m in MARGINS]


def robust_h6_fail():
    rob = report("stage2_robustness")
    return [frac(len(rob[m]["H6"]["fail_on_safe_recovered"]), rob[m]["H6"]["n_safe_recovered"]) for m in MARGINS]


DERIVE = {name: fn for name, fn in globals().items() if callable(fn) and not name.startswith("_")}


def actual_value(entry: dict):
    if "derive" in entry:
        return DERIVE[entry["derive"]](*entry.get("args", []))
    if "frac" in entry:
        num, den = entry["frac"]
        return frac(resolve(num), resolve(den))
    if "recall_frac" in entry:
        rec, n = entry["recall_frac"]
        return recall_frac(resolve(rec), resolve(n))
    return resolve(entry)


def matches(actual, expected, ndigits, in_range=False) -> bool:
    if in_range:
        low, high = actual
        return low <= expected <= high
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            matches(a, e, ndigits) for a, e in zip(actual, expected, strict=True))
    if ndigits is not None and isinstance(expected, float | int) and not isinstance(expected, bool) \
            and isinstance(actual, float | int) and not isinstance(actual, bool):
        # The printed value is the source rounded to ndigits; either side of a tie (0.625 -> 0.62/0.63) is accepted.
        return abs(actual - expected) <= 0.5 * 10 ** -ndigits + 1e-9
    return actual == expected


# ---------- tests ----------

def _ids(entries):
    return [e["id"] for e in entries]


SOURCED = [e for e in ENTRIES if "unsourced" not in e]


def test_entry_ids_unique():
    ids = _ids(ENTRIES)
    assert len(ids) == len(set(ids))


def test_derive_names_exist():
    missing = {e["derive"] for e in ENTRIES if "derive" in e} - set(DERIVE)
    assert not missing


@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_snippet_is_in_paper(entry):
    snippet = " ".join(entry["tex"].split())
    assert snippet in paper_text(), f"{entry['id']}: paper/main.tex no longer contains {snippet!r}"


@pytest.mark.parametrize("entry", SOURCED, ids=_ids(SOURCED))
def test_number_matches_report(entry):
    actual = actual_value(entry)
    assert matches(actual, entry["expected"], entry.get("ndigits"), entry.get("in_range", False)), (
        f"{entry['id']}: paper prints {entry['expected']!r}, source gives {actual!r}")


@pytest.mark.parametrize("entry", [e for e in ENTRIES if "unsourced" in e],
                         ids=_ids([e for e in ENTRIES if "unsourced" in e]))
def test_unsourced(entry):
    pytest.skip(f"unsourced: {entry['unsourced']}")


def test_numbers_in_snippets_are_expected():
    """Guard against a pin whose expected value silently drifted from the printed one: every scalar expected
    value (or both sides of an 'a/b' string) must appear as a token in its own snippet."""
    bad = []
    for e in ENTRIES:
        if e.get("no_token_check"):
            continue
        found = re.findall(r"-?\d+(?:\.\d+)?", e["tex"])
        tokens = set(found) | {t.lstrip("-") for t in found}  # "10--20" reads as 10 and -20
        exp = e["expected"]
        vals = exp if isinstance(exp, list) else [exp]
        for v in vals:
            if isinstance(v, bool) or isinstance(v, str) and "/" not in v:
                continue
            parts = v.split("/") if isinstance(v, str) else [v]
            for p in parts:
                p = str(p)
                cands = {p}
                if isinstance(v, float):
                    cands |= {f"{v:.2f}", f"{v:.1f}", f"{v * 100:.0f}"}
                if not cands & tokens:
                    bad.append((e["id"], v))
    assert not bad

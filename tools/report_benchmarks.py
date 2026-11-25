"""E2 (Job 26) and E1-frontier (Job 30) text-generation benchmarks: IFEval and XQuAD en/hi/es, recomputed from per_doc rows.

Usage (needs Python >= 3.10 and numpy, i.e. the repo environment):
  uv run python tools/report_benchmarks.py [--out DIR]          (or .venv/bin/python tools/report_benchmarks.py)
  -> docs/report/benchmarks.{json,md}

Exploratory under docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md (commit cd72fc6). No verdict or frozen
threshold is touched. Inputs are the tracked result files kaggle/executed/12_hf_jobs/files/job26_*.json and job30_*.json
(scores, answer sha256 and answer length per document; no answer text exists in them and none is read).

What it does
- Recomputes every headline score from `per_doc` and asserts it equals the lm-eval summary stored in the same file.
- IFEval: prompt-level strict/loose (Wilson 95% and Clopper-Pearson 95%), instruction-level strict/loose (naive Wilson
  over instructions, and a bootstrap over prompts, since instructions inside one prompt are not independent).
- XQuAD en/hi/es: F1 (normal 95% and bootstrap 95%), exact match (Wilson and Clopper-Pearson), mean and median answer
  length in characters, empty answers (answer_chars == 0), F1 over non-empty answers only.
- Paired exact McNemar vs base on IFEval prompt-level strict and loose flags: exploratory, with Holm adjustment over
  the 14 tests of the table (7 models x strict/loose) and, as a second family, over the 5 frontier strict tests. Every
  comparison is cross-job: each Job 26 model ran in its own HF Job, and Job 30 has no base rerun.
- Paired bootstrap of the XQuAD F1 difference vs base (descriptive), resampling paragraphs (XQuAD questions come
  several to a paragraph; kaggle/executed/12_hf_jobs/files/xquad_qid_paragraph.json, ids only) and, for comparison,
  individual questions. Uncorrected over the 15 frontier cells.
- Empty-answer overlap with base per language (the data hold no text, so the cause can only be bounded, not shown).
- Reproducibility evidence: Job 29's own English re-generation vs Job 25 (tracked), and smoke-vs-full answer hashes
  from kaggle/executed/12_hf_jobs/files/job26_job30_smoke_hashes.json (doc ids, answer sha256 and scored flags derived
  from the private jobs repo's results/job26-test and results/job30-test files at the recorded smoke_jobs_rev).
- Compares each number with the value recorded in docs/report/workflow/2026-11-24-run-log.md and lists differences.
B = 10000 bootstrap draws, numpy default_rng(20261125) (per-question draws) and default_rng(20261126) (paragraph draws).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "kaggle/executed/12_hf_jobs/files"
OUT = ROOT / "docs/report"
AMENDMENT = "docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md (cd72fc6)"
B, SEED, Z = 10000, 20261125, 1.959964
SEED_PARA = 20261126
SMOKE = FILES / "job26_job30_smoke_hashes.json"
PARA = FILES / "xquad_qid_paragraph.json"
RUN_CMD = "uv run python tools/report_benchmarks.py (or .venv/bin/python; needs Python >= 3.10 and numpy)"
LANGS = ("en", "hi", "es")
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"   # sha256 of ""
MODELS = [  # label, file, job
    ("base", "job26_base.json", "26"),
    ("alpaca_s1_125", "job26_alpaca_s1_125.json", "26"),
    ("repaired", "job26_repaired.json", "26"),
    ("u4", "job30_u4.json", "30"),
    ("r5", "job30_r5.json", "30"),
    ("r6", "job30_r6.json", "30"),
    ("r72", "job30_r72.json", "30"),
    ("u8", "job30_u8.json", "30"),
]
FRONTIER = ("u4", "r5", "r6", "r72", "u8")
# Values as recorded in docs/report/workflow/2026-11-24-run-log.md (E2 table lines 81-85, departure (f) table lines
# 269-278, "Other wave-2 results" lines 285-291). Compared at the precision the log prints. The two rounding slips
# this check found (u8 en F1 0.199, r6 es F1 0.125) were corrected in the run log on 2026-11-25 (0.198, 0.124).
RUNLOG = {
    "ifeval_strict": {"base": 0.7708, "alpaca_s1_125": 0.6821, "repaired": 0.7634, "u4": 0.715, "u8": 0.758,
                      "r5": 0.726, "r6": 0.726, "r72": 0.738},
    "ifeval_loose": {"base": 0.7967, "alpaca_s1_125": 0.7024, "repaired": 0.7893},
    "mcnemar_strict": {"u4": (67, 37, 0.004), "u8": (20, 13, 0.296), "r5": (55, 31, 0.013), "r6": (55, 31, 0.013),
                       "r72": (44, 26, 0.041)},
    "mean_chars": {"base": (230.9, 235.3, 335.7), "alpaca_s1_125": (103.1, 96.5, 116.2),
                   "repaired": (201.8, 239.6, 332.9), "u4": (210.0, 186.5, 308.4), "u8": (236.2, 234.6, 332.3),
                   "r5": (239.0, 208.6, 327.8), "r6": (242.8, 246.4, 325.3), "r72": (223.1, 254.9, 339.4)},
    "f1": {"base": (0.201, 0.112, 0.119), "alpaca_s1_125": (0.322, 0.261, 0.304), "repaired": (0.230, 0.118, 0.128),
           "u4": (0.218, 0.117, 0.121), "u8": (0.198, 0.111, 0.118), "r5": (0.190, 0.126, 0.116),
           "r6": (0.199, 0.113, 0.124), "r72": (0.208, 0.115, 0.116)},
    "em": {"base": (0.006, 0.002, 0.002), "alpaca_s1_125": (0.032, 0.006, 0.022), "repaired": (0.012, 0.002, 0.006),
           "u4": (0.018, 0.002, 0.004), "u8": (0.006, 0.002, 0.002), "r5": (0.006, 0.004, 0.002),
           "r6": (0.012, 0.002, 0.002), "r72": (0.010, 0.002, 0.002)},
    "empty_hi": {"base": 41, "alpaca_s1_125": 0, "repaired": 30, "u4": 98, "u8": 46, "r5": 62, "r6": 42, "r72": 18},
    "ifeval_median_chars_range_frontier": (967, 1029),
    "ifeval_median_chars_base": 1042,
    "r5_r6_identical_ifeval_hashes": 11,
}


# ---------- statistics ----------
def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def binom_cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: alpha / 2 - (1 - binom_cdf(k - 1, n, p)))
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) - alpha / 2)
    return [lower, upper]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p (binomial on the discordant pairs, p = 0.5)."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * binom_cdf(min(b, c), n, 0.5))


def holm(ps):
    """Holm step-down adjusted p values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def mean_normal_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return [m - Z * se, m + Z * se]


def boot_mean(xs, rng):
    a = np.asarray(xs, float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    return [float(v) for v in np.percentile(a[idx].mean(axis=1), [2.5, 97.5])]


def boot_ratio(num, den, rng):
    """Bootstrap over prompts of sum(num)/sum(den) (instruction-level accuracy clustered by prompt)."""
    nu, de = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(nu), size=(B, len(nu)))
    return [float(v) for v in np.percentile(nu[idx].sum(axis=1) / de[idx].sum(axis=1), [2.5, 97.5])]


def cluster_index(qids):
    """Question positions grouped by XQuAD paragraph."""
    para = json.loads(PARA.read_text(encoding="utf-8"))["paragraph_of_qid"]
    groups = {}
    for i, q in enumerate(qids):
        groups.setdefault(para[q], []).append(i)
    return [np.asarray(v) for v in groups.values()]


def boot_mean_clustered(xs, groups, rng):
    """Bootstrap of the mean resampling whole paragraphs (all questions of a drawn paragraph enter together)."""
    a = np.asarray(xs, float)
    sums = np.asarray([a[g].sum() for g in groups])
    sizes = np.asarray([len(g) for g in groups], float)
    idx = rng.integers(0, len(groups), size=(B, len(groups)))
    return [float(v) for v in np.percentile(sums[idx].sum(axis=1) / sizes[idx].sum(axis=1), [2.5, 97.5])]


def r(x, nd=6):
    return None if x is None else (round(x, nd) if isinstance(x, float) else x)


def rl(xs, nd=6):
    return [round(x, nd) for x in xs]


# ---------- loading ----------
def load():
    out = {}
    for label, fname, job in MODELS:
        d = json.loads((FILES / fname).read_text(encoding="utf-8"))
        assert d["label"] == label and d["complete"] is True, fname
        assert len(d["per_doc"]["ifeval"]) == 541, fname
        for lg in LANGS:
            assert len(d["per_doc"][f"xquad_{lg}"]) == 500, fname
        out[label] = {"file": fname, "job": job, "d": d}
    ids = {lab: [x["qid"] for x in v["d"]["per_doc"]["xquad_en"]] for lab, v in out.items()}
    assert all(ids[lab] == ids["base"] for lab in ids), "XQuAD question ids differ between models"
    keys = {lab: [x["key"] for x in v["d"]["per_doc"]["ifeval"]] for lab, v in out.items()}
    assert all(keys[lab] == keys["base"] for lab in keys), "IFEval keys differ between models"
    return out


def provenance(label, v):
    d = v["d"]
    return {"label": label, "file": f"kaggle/executed/12_hf_jobs/files/{v['file']}", "job": v["job"],
            "hf_job_id": d.get("job_id"), "code_sha": d.get("code_sha"), "model": d.get("model"),
            "quantization": d.get("quantization"), "batch_size": d.get("batch_size"),
            "chat_template_sha256": d.get("chat_template_sha256"), "xquad_ids_sha256": d.get("xquad_ids_sha256"),
            "packages": d.get("packages"), "gpu": d.get("gpu"), "model_dtype": d.get("model_dtype"),
            "peak_gib": d.get("peak_gib"), "bench_minutes": d.get("bench_minutes")}


# ---------- per-model scores ----------
def ifeval_scores(d, rng):
    rows = d["per_doc"]["ifeval"]
    s = d["summary"]["ifeval"]
    n = len(rows)
    out = {"n_prompts": n}
    for kind in ("strict", "loose"):
        k = sum(int(x[f"prompt_level_{kind}_acc"] >= 0.5) for x in rows)
        assert abs(k / n - s[f"prompt_level_{kind}_acc"]) < 1e-12
        assert k == s[f"prompt_level_{kind}_acc_k"]
        w = wilson(k, n)
        assert all(abs(a - b) < 1e-5 for a, b in zip(w, s[f"prompt_level_{kind}_acc_wilson95"], strict=True))
        num = [sum(x[f"inst_level_{kind}_acc"]) for x in rows]
        den = [len(x[f"inst_level_{kind}_acc"]) for x in rows]
        ki, ni = sum(num), sum(den)
        assert abs(ki / ni - s[f"inst_level_{kind}_acc"]) < 1e-12
        out[f"prompt_{kind}"] = {"k": k, "n": n, "rate": r(k / n), "wilson95": rl(w), "cp95": rl(clopper_pearson(k, n))}
        out[f"inst_{kind}"] = {"k": ki, "n": ni, "rate": r(ki / ni), "wilson95_naive": rl(wilson(ki, ni)),
                               "boot95_by_prompt": rl(boot_ratio(num, den, rng))}
    chars = [x["answer_chars"] for x in rows]
    out["n_empty"] = sum(c == 0 for c in chars)
    out["median_chars"] = statistics.median(chars)
    out["mean_chars"] = r(sum(chars) / n, 4)
    return out


def xquad_scores(d, lg, rng, groups, rng_para):
    rows = d["per_doc"][f"xquad_{lg}"]
    s = d["summary"][f"xquad_{lg}"]
    n = len(rows)
    f1 = [x["f1"] for x in rows]
    em_k = sum(int(x["exact_match"] >= 0.5) for x in rows)
    assert abs(sum(f1) / n - s["f1"]) < 1e-12 and abs(em_k / n - s["exact_match"]) < 1e-12
    assert all(abs(a - b) < 1e-5 for a, b in zip(mean_normal_ci(f1), s["f1_ci95_normal"], strict=True))
    chars = [x["answer_chars"] for x in rows]
    empty = [x["answer_chars"] == 0 for x in rows]
    assert all((x["answer_sha256"] == EMPTY_SHA) == e for x, e in zip(rows, empty, strict=True))
    assert all(x["f1"] == 0.0 for x, e in zip(rows, empty, strict=True) if e)   # an empty answer scores 0 by construction
    ne = [x["f1"] for x, e in zip(rows, empty, strict=True) if not e]
    nonempty_chars = [c for c in chars if c > 0]
    return {"n": n, "f1": r(sum(f1) / n), "f1_normal95": rl(mean_normal_ci(f1)), "f1_boot95": rl(boot_mean(f1, rng)),
            "f1_boot95_by_paragraph": rl(boot_mean_clustered(f1, groups, rng_para)),
            "em_k": em_k, "em": r(em_k / n), "em_wilson95": rl(wilson(em_k, n)), "em_cp95": rl(clopper_pearson(em_k, n)),
            "mean_chars": r(sum(chars) / n, 4), "median_chars": statistics.median(chars),
            "mean_chars_nonempty": r(sum(nonempty_chars) / len(nonempty_chars), 4),
            "n_empty": sum(empty), "f1_nonempty": r(sum(ne) / len(ne)), "n_nonempty": len(ne)}


# ---------- paired comparisons ----------
def mcnemar_vs_base(base, other, kind):
    bf = [x[f"prompt_level_{kind}_acc"] >= 0.5 for x in base["per_doc"]["ifeval"]]
    of = [x[f"prompt_level_{kind}_acc"] >= 0.5 for x in other["per_doc"]["ifeval"]]
    b = sum(x and not y for x, y in zip(bf, of, strict=True))
    c = sum(y and not x for x, y in zip(bf, of, strict=True))
    return {"base_only": b, "model_only": c, "both": sum(x and y for x, y in zip(bf, of, strict=True)),
            "neither": sum(not x and not y for x, y in zip(bf, of, strict=True)), "p_exact": mcnemar_exact(b, c)}


def f1_diff_vs_base(base, other, lg, rng, groups, rng_para):
    a = np.asarray([x["f1"] for x in base["per_doc"][f"xquad_{lg}"]])
    o = np.asarray([x["f1"] for x in other["per_doc"][f"xquad_{lg}"]])
    dlt = o - a
    idx = rng.integers(0, len(dlt), size=(B, len(dlt)))
    return {"diff": r(float(dlt.mean())),
            "boot95": rl([float(v) for v in np.percentile(dlt[idx].mean(axis=1), [2.5, 97.5])]),
            "boot95_by_paragraph": rl(boot_mean_clustered(dlt, groups, rng_para))}


def empty_overlap(models):
    out = {}
    base_sets = {lg: {x["doc_id"] for x in models["base"]["d"]["per_doc"][f"xquad_{lg}"] if x["answer_chars"] == 0}
                 for lg in LANGS}
    for lg in LANGS:
        sets = {lab: {x["doc_id"] for x in v["d"]["per_doc"][f"xquad_{lg}"] if x["answer_chars"] == 0}
                for lab, v in models.items()}
        per = {}
        for lab, s in sets.items():
            inter = len(s & base_sets[lg])
            per[lab] = {"n_empty": len(s), "also_empty_in_base": inter, "not_empty_in_base": len(s) - inter,
                        "base_empty_not_here": len(base_sets[lg]) - inter}
        fp = [lab for lab in sets if lab != "alpaca_s1_125"]
        all_but_alpaca = set.intersection(*(sets[lab] for lab in fp))
        any_model = set.union(*sets.values())
        out[lg] = {"per_model": per, "empty_in_all_7_non_alpaca": len(all_but_alpaca), "empty_in_any": len(any_model)}
    return out


def identical_hashes(models, a, b, task="ifeval"):
    ra, rb = models[a]["d"]["per_doc"][task], models[b]["d"]["per_doc"][task]
    return sum(x["answer_sha256"] == y["answer_sha256"] for x, y in zip(ra, rb, strict=True))


# ---------- reproducibility ----------
def job29_repeat():
    """Job 29 re-generated the Job 25 English completions in a later HF Job; its summaries store the comparison."""
    out = {}
    for part in ("core", "frontier"):
        p = FILES / f"job29_{part}_summary.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for lab, v in (d.get("per_model") or {}).items():
            e = v.get("en_vs_job25")
            if e:
                out[lab] = {"part": part, **e}
    return out


def smoke_vs_full(models, path=SMOKE):
    """Smoke (5-document) vs full answer hashes, from the committed derived file (no text; see its note)."""
    sm = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {"source": str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT) else str(path),
           "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "smoke_jobs_rev": sm["smoke_jobs_rev"]}
    for lab, s in sm["models"].items():
        full = models[lab]["d"]
        row = {"smoke_file": f"purvansh01/safetypulse-jobs@{sm['smoke_jobs_rev'][:12]} {s['smoke_file']}",
               "smoke_file_sha256": s["smoke_file_sha256"], "smoke_hf_job_id": s["smoke_hf_job_id"],
               "smoke_code_sha": s["smoke_code_sha"], "same_packages": s["same_packages_as_full"],
               "same_gpu": s["same_gpu_as_full"], "limit": s["limit"]}
        for task in ("ifeval", *(f"xquad_{lg}" for lg in LANGS)):
            fd = {x["doc_id"]: x for x in full["per_doc"][task]}
            srows = [x for x in s["per_doc"][task] if x["doc_id"] in fd]
            flag = "prompt_level_strict_acc" if task == "ifeval" else "f1"
            row[task] = {"n": len(srows),
                         "hash_equal": sum(x["answer_sha256"] == fd[x["doc_id"]]["answer_sha256"] for x in srows),
                         "score_equal": sum(x[flag] == fd[x["doc_id"]][flag] for x in srows)}
        out[lab] = row
    return out


# ---------- run-log comparison ----------
def compare_runlog(res):
    diffs, checked = [], 0

    def chk(name, got, want, tol, exact=None):
        nonlocal checked
        checked += 1
        if abs(got - want) > tol:
            diffs.append({"item": name, "recomputed": got, "run_log": want, "exact": got if exact is None else exact})

    for lab, want in RUNLOG["ifeval_strict"].items():
        nd = len(str(want).split(".")[1])
        chk(f"ifeval prompt strict {lab}", round(res["models"][lab]["ifeval"]["prompt_strict"]["rate"], nd), want, 1e-9)
    for lab, want in RUNLOG["ifeval_loose"].items():
        chk(f"ifeval prompt loose {lab}", round(res["models"][lab]["ifeval"]["prompt_loose"]["rate"], 4), want, 1e-9)
    for lab, (b, c, p) in RUNLOG["mcnemar_strict"].items():
        m = res["mcnemar_vs_base"][lab]["prompt_strict"]
        chk(f"mcnemar {lab} base_only", m["base_only"], b, 0)
        chk(f"mcnemar {lab} model_only", m["model_only"], c, 0)
        chk(f"mcnemar {lab} p", round(m["p_exact"], 3), p, 1e-9)
    for key, field, nd in (("mean_chars", "mean_chars", 1), ("f1", "f1", 3), ("em", "em", 3)):
        for lab, want in RUNLOG[key].items():
            for lg, w in zip(LANGS, want, strict=True):
                x = res["models"][lab]["xquad"][lg][field]
                chk(f"xquad {field} {lab} {lg}", round(x, nd), w, 1e-9, x)
    for lab, want in RUNLOG["empty_hi"].items():
        chk(f"xquad empty hi {lab}", res["models"][lab]["xquad"]["hi"]["n_empty"], want, 0)
    fr = [res["models"][lab]["ifeval"]["median_chars"] for lab in ("u4", "r5", "r6", "r72", "u8")]
    lo, hi = RUNLOG["ifeval_median_chars_range_frontier"]
    chk("ifeval median chars frontier min", min(fr), lo, 0)
    chk("ifeval median chars frontier max", max(fr), hi, 0)
    chk("ifeval median chars base", res["models"]["base"]["ifeval"]["median_chars"], RUNLOG["ifeval_median_chars_base"], 0)
    chk("r5 vs r6 identical ifeval hashes", res["identical_ifeval_hashes"]["r5_vs_r6"],
        RUNLOG["r5_r6_identical_ifeval_hashes"], 0)
    return {"checked": checked, "differences": diffs}


# ---------- report ----------
def build():
    rng, rng_para = np.random.default_rng(SEED), np.random.default_rng(SEED_PARA)
    models = load()
    groups = cluster_index([x["qid"] for x in models["base"]["d"]["per_doc"]["xquad_en"]])
    res = {"amendment": AMENDMENT, "exploratory": True, "run_command": RUN_CMD,
           "bootstrap": {"B": B, "seed": SEED, "seed_by_paragraph": SEED_PARA,
                         "xquad_paragraphs": {"file": str(PARA.relative_to(ROOT)), "n_questions": 500,
                                              "n_paragraphs": len(groups)}},
           "provenance": {lab: provenance(lab, v) for lab, v in models.items()}, "models": {}}
    for lab, v in models.items():
        d = v["d"]
        res["models"][lab] = {"job": v["job"], "ifeval": ifeval_scores(d, rng),
                              "xquad": {lg: xquad_scores(d, lg, rng, groups, rng_para) for lg in LANGS}}
    base = models["base"]["d"]
    res["mcnemar_vs_base"] = {
        lab: {"prompt_strict": mcnemar_vs_base(base, v["d"], "strict"),
              "prompt_loose": mcnemar_vs_base(base, v["d"], "loose"),
              "same_hf_job_as_base": v["d"].get("job_id") == base.get("job_id")}
        for lab, v in models.items() if lab != "base"}
    fam = [(lab, kind) for lab in res["mcnemar_vs_base"] for kind in ("prompt_strict", "prompt_loose")]
    for (lab, kind), a in zip(fam, holm([res["mcnemar_vs_base"][lab][kind]["p_exact"] for lab, kind in fam]), strict=True):
        res["mcnemar_vs_base"][lab][kind]["p_holm_14"] = r(a)
    fam5 = [(lab, "prompt_strict") for lab in FRONTIER]
    for (lab, kind), a in zip(fam5, holm([res["mcnemar_vs_base"][lab][kind]["p_exact"] for lab, kind in fam5]), strict=True):
        res["mcnemar_vs_base"][lab][kind]["p_holm_frontier_strict_5"] = r(a)
    res["mcnemar_holm"] = {
        "family_14": "all 14 tests in the table (7 models x prompt strict/loose)",
        "family_frontier_strict_5": "prompt strict for u4, r5, r6, r72, u8",
        "survive_14": [f"{lab} {kind.split('_')[1]}" for lab, kind in fam if res["mcnemar_vs_base"][lab][kind]["p_holm_14"] < 0.05],
        "survive_frontier_strict_5": [lab for lab, _ in fam5
                                      if res["mcnemar_vs_base"][lab]["prompt_strict"]["p_holm_frontier_strict_5"] < 0.05]}
    res["f1_diff_vs_base"] = {lab: {lg: f1_diff_vs_base(base, v["d"], lg, rng, groups, rng_para) for lg in LANGS}
                              for lab, v in models.items() if lab != "base"}
    fr = [(lab, lg, res["f1_diff_vs_base"][lab][lg]["diff"]) for lab in FRONTIER for lg in LANGS]
    big = max(fr, key=lambda t: abs(t[2]))
    res["frontier_max_abs_f1_diff"] = {"model": big[0], "lang": big[1], "diff": big[2]}
    res["empty_overlap"] = empty_overlap(models)
    res["identical_ifeval_hashes"] = {"r5_vs_r6": identical_hashes(models, "r5", "r6"),
                                      **{f"base_vs_{lab}": identical_hashes(models, "base", lab)
                                         for lab in models if lab != "base"}}
    hf_jobs = {lab: v["d"].get("job_id") for lab, v in models.items()}
    res["cross_job"] = {"hf_job_ids": hf_jobs, "n_distinct_hf_jobs": len(set(hf_jobs.values())),
                        "base_rerun_in_job30": any(lab == "base" for lab, v in models.items() if v["job"] == "30"),
                        "code_sha": {lab: v["d"].get("code_sha") for lab, v in models.items()}}
    res["reproducibility"] = {"job29_en_vs_job25": job29_repeat(),
                              "smoke_vs_full": smoke_vs_full(models)}
    res["run_log_check"] = compare_runlog(res)
    return res


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(ci, nd=3):
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def render_md(res):
    M = res["models"]
    labs = list(M)
    L = ["# Text-generation benchmarks: IFEval and XQuAD en/hi/es (E2, Job 26; E1 frontier, Job 30)", "",
         f"Exploratory under `{res['amendment']}`. No pre-registered verdict or frozen threshold is touched. "
         "Generated by `tools/report_benchmarks.py` from `kaggle/executed/12_hf_jobs/files/job26_*.json` and "
         "`job30_*.json` (per-document scores, answer sha256 and answer length; no answer text exists in these files). "
         "Every summary score was recomputed from `per_doc` and asserted equal to the lm-eval summary in the same file. "
         f"Bootstrap: B = {res['bootstrap']['B']}, numpy `default_rng({res['bootstrap']['seed']})` (per question or "
         f"prompt) and `default_rng({res['bootstrap']['seed_by_paragraph']})` (XQuAD by paragraph). "
         f"Rebuild: `{res['run_command']}`.", "",
         "All runs: lm-evaluation-harness 0.4.13, greedy, chat template, no system instruction, bf16 compute, batch 32, "
         "L40S. Job 30 variants are compressed-tensors checkpoints that transformers decompresses at the first forward pass, so "
         "these are quality numbers for the quantized weights, not speed or memory numbers.", "",
         "## IFEval (541 prompts)", "",
         "| Model | Job | Prompt strict [Wilson 95%] | Prompt loose [Wilson 95%] | Instr. strict [boot 95%, by prompt] | "
         "Instr. loose [boot 95%] | Median chars | Empty |",
         "|---|---|---|---|---|---|---|---|"]
    for lab in labs:
        f = M[lab]["ifeval"]
        L.append(f"| {lab} | {M[lab]['job']} | {f['prompt_strict']['rate']:.3f} ({f['prompt_strict']['k']}/541) "
                 f"{fmt_ci(f['prompt_strict']['wilson95'])} | {f['prompt_loose']['rate']:.3f} "
                 f"{fmt_ci(f['prompt_loose']['wilson95'])} | {f['inst_strict']['rate']:.3f} "
                 f"{fmt_ci(f['inst_strict']['boot95_by_prompt'])} | {f['inst_loose']['rate']:.3f} "
                 f"{fmt_ci(f['inst_loose']['boot95_by_prompt'])} | {f['median_chars']:g} | {f['n_empty']} |")
    n_inst = M["base"]["ifeval"]["inst_strict"]["n"]
    L += ["", f"Instruction-level n = {n_inst} instructions; the naive Wilson interval over instructions (in the JSON) "
          "is narrower than the by-prompt bootstrap because instructions in one prompt are correlated. Clopper-Pearson "
          "intervals for the prompt-level rates are in the JSON.", "",
          "### Paired exact McNemar vs base (prompt level; exploratory, cross-job)", "",
          "| Model | Strict: base-only / model-only | p | p Holm (14) | p Holm (5 frontier strict) | "
          "Loose: base-only / model-only | p | p Holm (14) |", "|---|---|---|---|---|---|---|---|"]
    for lab in labs[1:]:
        s, lo = res["mcnemar_vs_base"][lab]["prompt_strict"], res["mcnemar_vs_base"][lab]["prompt_loose"]
        h5 = s.get("p_holm_frontier_strict_5")
        L.append(f"| {lab} | {s['base_only']} / {s['model_only']} | {fmt_p(s['p_exact'])} | {fmt_p(s['p_holm_14'])} | "
                 f"{'-' if h5 is None else fmt_p(h5)} | {lo['base_only']} / {lo['model_only']} | {fmt_p(lo['p_exact'])} | "
                 f"{fmt_p(lo['p_holm_14'])} |")
    cj = res["cross_job"]
    mh = res["mcnemar_holm"]
    L += ["", f"Holm families: (14) {mh['family_14']}; (5) {mh['family_frontier_strict_5']}. Surviving Holm at 0.05 in "
          f"the 14-test family: {', '.join(mh['survive_14'])}. In the 5-test frontier-strict family: "
          f"{', '.join(mh['survive_frontier_strict_5'])}. So an IFEval prompt-strict drop vs base survives correction "
          "only for u4 (and the Alpaca fine-tune); r5, r6 and r72 strict are nominal only (uncorrected p < 0.05), not "
          "significant after Holm. **Every comparison is "
          f"cross-job.** The eight result files come from {cj['n_distinct_hf_jobs']} distinct HF Jobs (each Job 26 "
          "model ran in its own job; Job 30 ran u4/u8/r72 in one job and r5/r6 in another), and Job 30 has no base "
          "rerun. Job 30 used script `ba121a7` (Job 26: `5a45d23`; `bench()` stated unchanged), the same packages, "
          "GPU type, batch size, chat-template hash and XQuAD id hash.", ""]
    rep = res["reproducibility"]
    L += ["### Are base answer hashes reproducible?", ""]
    j29 = rep["job29_en_vs_job25"]
    if j29:
        agree = ", ".join(f"{lab} {v['agree']}/{v['n']}" for lab, v in j29.items())
        L.append(f"- Cross-job, identical batch composition (different benchmark): Job 29 re-generated the Job 25 "
                 f"English refusal-set completions in a later HF Job; judge flags agree {agree} "
                 "(`job29_*_summary.json` `per_model.*.en_vs_job25`). So same-input, same-batch generation did repeat "
                 "across jobs on this stack at least once.")
    sv = rep["smoke_vs_full"]
    if sv:
        parts = []
        for lab, v in sv.items():
            if not isinstance(v, dict) or "ifeval" not in v:
                continue
            parts.append(f"{lab}: IFEval {v['ifeval']['hash_equal']}/{v['ifeval']['n']} (prompt-strict flag equal "
                         f"{v['ifeval']['score_equal']}/{v['ifeval']['n']}), XQuAD en/hi/es "
                         f"{v['xquad_en']['hash_equal']}/{v['xquad_hi']['hash_equal']}/{v['xquad_es']['hash_equal']} of 5")
        L.append("- Same model, same code, different batch composition: the 5-document smoke runs "
                 "(`results/job26-test/*`, `results/job30-test/u4.json` in the private jobs repo at revision "
                 f"`{sv['smoke_jobs_rev']}`; doc ids, answer sha256 and scored flags copied to `{sv['source']}`, no "
                 "text) vs the full runs, identical "
                 "answer hashes: " + "; ".join(parts) + ". IFEval answers (up to 1280 tokens) mostly change when the "
                 "batch changes, so bf16 batched greedy generation here is not batch-invariant, though the strict flags of "
                 "these few prompts mostly survive. Five documents per task is far too few to size the noise.")
    else:
        L.append("- Smoke-vs-full hash check not run (needs `--smoke-dir` with the private smoke files).")
    L += ["- Not testable from these files: whether a Job 26 base rerun with the same 541-prompt batches would give "
          "the same hashes. Job 22 found generation repeatable within a job but not always across jobs. The McNemar "
          "counts therefore include any run-to-run noise as well as the quantization effect; the size of that noise "
          "was not measured for IFEval.", "", "## XQuAD (500 questions per language, same ids in every run)", "",
          "| Model | Lang | F1 [boot 95%, by paragraph] | EM k/500 [Wilson 95%] | Mean chars | Median chars | Empty | "
          "F1 non-empty |",
          "|---|---|---|---|---|---|---|---|"]
    for lab in labs:
        for lg in LANGS:
            x = M[lab]["xquad"][lg]
            L.append(f"| {lab} | {lg} | {x['f1']:.3f} {fmt_ci(x['f1_boot95_by_paragraph'])} | {x['em_k']} "
                     f"{fmt_ci(x['em_wilson95'])} | {x['mean_chars']:.1f} | {x['median_chars']:g} | {x['n_empty']} | "
                     f"{x['f1_nonempty']:.3f} |")
    L += ["", "Exact match is 0 to 16 of 500 everywhere because chat-style answers are much longer than the gold span; "
          "F1 is the usable number, and it depends on answer length (the Alpaca fine-tune answers 2-3 times shorter and "
          "gets the highest F1; that is not read as better QA). SQuAD normalisation is English-centric (the Hindi danda is "
          "not stripped).", "",
          f"The 500 XQuAD questions come from {res['bootstrap']['xquad_paragraphs']['n_paragraphs']} paragraphs "
          f"(`{res['bootstrap']['xquad_paragraphs']['file']}`, ids only), so F1 intervals resample paragraphs. The "
          "per-question bootstrap (`f1_boot95`, `boot95` in the JSON) ignores this clustering and is kept for comparison.",
          "", "### XQuAD F1 difference vs base (paired bootstrap 95%, descriptive)", "",
          "Each cell: difference, [paragraph bootstrap], (per-question bootstrap). Uncorrected over the 15 frontier cells "
          "(and 6 more for the Job 26 models); an interval that excludes 0 is not read as a difference.", "",
          "| Model | en | hi | es |", "|---|---|---|---|"]
    for lab in labs[1:]:
        cells = [f"{res['f1_diff_vs_base'][lab][lg]['diff']:+.3f} "
                 f"{fmt_ci(res['f1_diff_vs_base'][lab][lg]['boot95_by_paragraph'])} "
                 f"({res['f1_diff_vs_base'][lab][lg]['boot95'][0]:.3f}, {res['f1_diff_vs_base'][lab][lg]['boot95'][1]:.3f})"
                 for lg in LANGS]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    mx = res["frontier_max_abs_f1_diff"]
    L += ["", f"Largest absolute F1 change among the five frontier variants: {mx['model']} {mx['lang']} {mx['diff']:+.3f}; "
          "every frontier F1 change is within about 0.02 of base and none is claimed as real. "
          "Cross-job caveat as for McNemar; empty answers (below) feed into these differences."]
    eo = res["empty_overlap"]
    L += ["", "## Empty XQuAD answers", "",
          "XQuAD generation stops at the first newline (`until: ['\\n']`), and an empty answer has answer_chars 0 and the "
          "sha256 of the empty string in every file. An answer is empty when generation stops before any visible text, "
          "either at a leading newline (`until: ['\\n']`) or at an immediate end-of-turn/EOS token; the files hold no text "
          "or token ids, so the cause cannot be determined here.", "",
          "| Model | Empty en / hi / es | hi: also empty in base | hi: new vs base | hi: base-empty not empty here |",
          "|---|---|---|---|---|"]
    for lab in labs:
        e = [eo[lg]["per_model"][lab]["n_empty"] for lg in LANGS]
        h = eo["hi"]["per_model"][lab]
        L.append(f"| {lab} | {e[0]} / {e[1]} / {e[2]} | {h['also_empty_in_base']} | {h['not_empty_in_base']} | "
                 f"{h['base_empty_not_here']} |")
    u4 = eo["hi"]["per_model"]["u4"]
    L += ["", f"Hindi questions empty in all seven non-Alpaca models: {eo['hi']['empty_in_all_7_non_alpaca']}; empty in "
          f"at least one model: {eo['hi']['empty_in_any']}. u4's {u4['n_empty']} Hindi empties contain "
          f"{u4['also_empty_in_base']} of base's {M['base']['xquad']['hi']['n_empty']} and add {u4['not_empty_in_base']} "
          "new ones. Why 4-bit weights give more empty answers in Hindi (and Spanish, "
          f"{M['u4']['xquad']['es']['n_empty']} vs base {M['base']['xquad']['es']['n_empty']}) than in English is "
          "**unexplained** by these data, and (run-log departure (f)) u4's Hindi score is not interpreted until it is "
          "explained. Since empty answers score 0, u4's Hindi F1 over non-empty answers "
          f"({M['u4']['xquad']['hi']['f1_nonempty']:.3f}, n = {M['u4']['xquad']['hi']['n_nonempty']}) is higher than "
          f"base's ({M['base']['xquad']['hi']['f1_nonempty']:.3f}, n = {M['base']['xquad']['hi']['n_nonempty']}); "
          "neither all-answer nor non-empty F1 is a clean Hindi QA comparison.", ""]
    rc = res["run_log_check"]
    L += ["## Check against the run log", "",
          f"{rc['checked']} values recorded in `docs/report/workflow/2026-11-24-run-log.md` were compared at the "
          "precision the log prints."]
    if rc["differences"]:
        L += ["", "| Item | Recomputed (exact) | Run log |", "|---|---|---|"]
        L += [f"| {x['item']} | {x['recomputed']} ({x['exact']}) | {x['run_log']} |" for x in rc["differences"]]
        L += ["", "These are rounding slips in the run log (the exact value rounds the other way); no conclusion changes."]
    else:
        L.append("All match.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    res = build()
    out = Path(a.out)
    (out / "benchmarks.json").write_text(json.dumps(res, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    (out / "benchmarks.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {out / 'benchmarks.json'} and {out / 'benchmarks.md'}; run-log differences: "
          f"{len(res['run_log_check']['differences'])}", file=sys.stderr)


if __name__ == "__main__":
    main()

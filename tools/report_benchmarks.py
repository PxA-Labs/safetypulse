"""E2 (Job 26) and E1-frontier (Job 30) text-generation benchmarks: IFEval and XQuAD en/hi/es, recomputed from per_doc rows.

Usage: python tools/report_benchmarks.py [--smoke-dir DIR]   -> docs/report/benchmarks.{json,md}

Exploratory under docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md (commit cd72fc6). No verdict or frozen
threshold is touched. Inputs are the tracked result files kaggle/executed/12_hf_jobs/files/job26_*.json and job30_*.json
(scores, answer sha256 and answer length per document; no answer text exists in them and none is read).

What it does
- Recomputes every headline score from `per_doc` and asserts it equals the lm-eval summary stored in the same file.
- IFEval: prompt-level strict/loose (Wilson 95% and Clopper-Pearson 95%), instruction-level strict/loose (naive Wilson
  over instructions, and a bootstrap over prompts, since instructions inside one prompt are not independent).
- XQuAD en/hi/es: F1 (normal 95% and bootstrap 95%), exact match (Wilson and Clopper-Pearson), mean and median answer
  length in characters, empty answers (answer_chars == 0), F1 over non-empty answers only.
- Paired exact McNemar vs base on IFEval prompt-level strict and loose flags: exploratory and uncorrected. Every
  comparison is cross-job: each Job 26 model ran in its own HF Job, and Job 30 has no base rerun.
- Paired bootstrap of the XQuAD F1 difference vs base (descriptive).
- Empty-answer overlap with base per language (the data hold no text, so the cause can only be bounded, not shown).
- Reproducibility evidence: Job 29's own English re-generation vs Job 25 (tracked), and, if --smoke-dir points at a
  local copy of the private jobs repo's results/job26-test and results/job30-test files, smoke-vs-full answer hashes.
- Compares each number with the value recorded in docs/report/workflow/2026-11-24-run-log.md and lists differences.
B = 10000 bootstrap draws, numpy default_rng(20261125).
"""
from __future__ import annotations

import argparse
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
# 269-278, "Other wave-2 results" lines 285-291). Compared at the precision the log prints.
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
           "u4": (0.218, 0.117, 0.121), "u8": (0.199, 0.111, 0.118), "r5": (0.190, 0.126, 0.116),
           "r6": (0.199, 0.113, 0.125), "r72": (0.208, 0.115, 0.116)},
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


def xquad_scores(d, lg, rng):
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


def f1_diff_vs_base(base, other, lg, rng):
    a = np.asarray([x["f1"] for x in base["per_doc"][f"xquad_{lg}"]])
    o = np.asarray([x["f1"] for x in other["per_doc"][f"xquad_{lg}"]])
    dlt = o - a
    idx = rng.integers(0, len(dlt), size=(B, len(dlt)))
    return {"diff": r(float(dlt.mean())), "boot95": rl([float(v) for v in np.percentile(dlt[idx].mean(axis=1), [2.5, 97.5])])}


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


def smoke_vs_full(smoke_dir, models):
    pairs = [("base", "job26-test/base.json"), ("alpaca_s1_125", "job26-test/alpaca_s1_125.json"),
             ("repaired", "job26-test/repaired.json"), ("u4", "job30-test/u4.json")]
    out = {}
    for lab, rel in pairs:
        p = Path(smoke_dir) / rel
        if not p.is_file():
            out[lab] = {"missing": rel}
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        full = models[lab]["d"]
        row = {"smoke_file": f"purvansh01/safetypulse-jobs results/{rel}", "smoke_hf_job_id": s.get("job_id"),
               "smoke_code_sha": s.get("code_sha"), "same_packages": s.get("packages") == full.get("packages"),
               "same_gpu": s.get("gpu") == full.get("gpu"), "limit": s.get("limit")}
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
def build(smoke_dir=None):
    rng = np.random.default_rng(SEED)
    models = load()
    res = {"amendment": AMENDMENT, "exploratory": True, "bootstrap": {"B": B, "seed": SEED},
           "provenance": {lab: provenance(lab, v) for lab, v in models.items()}, "models": {}}
    for lab, v in models.items():
        d = v["d"]
        res["models"][lab] = {"job": v["job"], "ifeval": ifeval_scores(d, rng),
                              "xquad": {lg: xquad_scores(d, lg, rng) for lg in LANGS}}
    base = models["base"]["d"]
    res["mcnemar_vs_base"] = {
        lab: {"prompt_strict": mcnemar_vs_base(base, v["d"], "strict"),
              "prompt_loose": mcnemar_vs_base(base, v["d"], "loose"),
              "same_hf_job_as_base": v["d"].get("job_id") == base.get("job_id")}
        for lab, v in models.items() if lab != "base"}
    res["f1_diff_vs_base"] = {lab: {lg: f1_diff_vs_base(base, v["d"], lg, rng) for lg in LANGS}
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
                              "smoke_vs_full": smoke_vs_full(smoke_dir, models) if smoke_dir else None}
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
         f"Bootstrap: B = {res['bootstrap']['B']}, numpy `default_rng({res['bootstrap']['seed']})`.", "",
         "All runs: lm-evaluation-harness 0.4.13, greedy, chat template, no system instruction, bf16 compute, batch 32, "
         "L40S. Job 30 variants are compressed-tensors checkpoints that transformers decompresses on load, so these are "
         "quality numbers for the quantized weights, not speed or memory numbers.", "",
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
          "### Paired exact McNemar vs base (prompt level; exploratory, uncorrected, cross-job)", "",
          "| Model | Strict: base-only / model-only | p | Loose: base-only / model-only | p |", "|---|---|---|---|---|"]
    for lab in labs[1:]:
        s, lo = res["mcnemar_vs_base"][lab]["prompt_strict"], res["mcnemar_vs_base"][lab]["prompt_loose"]
        L.append(f"| {lab} | {s['base_only']} / {s['model_only']} | {fmt_p(s['p_exact'])} | {lo['base_only']} / "
                 f"{lo['model_only']} | {fmt_p(lo['p_exact'])} |")
    cj = res["cross_job"]
    L += ["", f"No p value is corrected for the {2 * (len(labs) - 1)} tests in this table. **Every comparison is "
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
            if "missing" in v:
                continue
            parts.append(f"{lab}: IFEval {v['ifeval']['hash_equal']}/{v['ifeval']['n']} (prompt-strict flag equal "
                         f"{v['ifeval']['score_equal']}/{v['ifeval']['n']}), XQuAD en/hi/es "
                         f"{v['xquad_en']['hash_equal']}/{v['xquad_hi']['hash_equal']}/{v['xquad_es']['hash_equal']} of 5")
        L.append("- Same model, same code, different batch composition: the 5-document smoke runs "
                 "(`results/job26-test/*`, `results/job30-test/u4.json`, private jobs repo) vs the full runs, identical "
                 "answer hashes: " + "; ".join(parts) + ". IFEval answers (up to 1280 tokens) mostly change when the "
                 "batch changes, so bf16 batched greedy generation here is not batch-invariant, though the strict flags of "
                 "these few prompts mostly survive. Five documents per task is far too few to size the noise.")
    else:
        L.append("- Smoke-vs-full hash check not run (needs `--smoke-dir` with the private smoke files).")
    L += ["- Not testable from these files: whether a Job 26 base rerun with the same 541-prompt batches would give "
          "the same hashes. Job 22 found generation repeatable within a job but not always across jobs. The McNemar "
          "counts therefore include any run-to-run noise as well as the quantization effect; the size of that noise "
          "was not measured for IFEval.", "", "## XQuAD (500 questions per language, same ids in every run)", "",
          "| Model | Lang | F1 [boot 95%] | EM k/500 [Wilson 95%] | Mean chars | Median chars | Empty | F1 non-empty |",
          "|---|---|---|---|---|---|---|---|"]
    for lab in labs:
        for lg in LANGS:
            x = M[lab]["xquad"][lg]
            L.append(f"| {lab} | {lg} | {x['f1']:.3f} {fmt_ci(x['f1_boot95'])} | {x['em_k']} "
                     f"{fmt_ci(x['em_wilson95'])} | {x['mean_chars']:.1f} | {x['median_chars']:g} | {x['n_empty']} | "
                     f"{x['f1_nonempty']:.3f} |")
    L += ["", "Exact match is 0 to 16 of 500 everywhere because chat-style answers are much longer than the gold span; "
          "F1 is the usable number, and it depends on answer length (the Alpaca fine-tune answers 2-3 times shorter and "
          "gets the highest F1; that is not read as better QA). SQuAD normalisation is English-centric (the Hindi danda is "
          "not stripped).", "", "### XQuAD F1 difference vs base (paired bootstrap 95%, descriptive)", "",
          "| Model | en | hi | es |", "|---|---|---|---|"]
    for lab in labs[1:]:
        cells = [f"{res['f1_diff_vs_base'][lab][lg]['diff']:+.3f} {fmt_ci(res['f1_diff_vs_base'][lab][lg]['boot95'])}"
                 for lg in LANGS]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    mx = res["frontier_max_abs_f1_diff"]
    L += ["", f"Largest absolute F1 change among the five frontier variants: {mx['model']} {mx['lang']} {mx['diff']:+.3f}. "
          "Cross-job caveat as for McNemar; empty answers (below) feed into these differences."]
    eo = res["empty_overlap"]
    L += ["", "## Empty XQuAD answers", "",
          "XQuAD generation stops at the first newline (`until: ['\\n']`), and an empty answer has answer_chars 0 and the "
          "sha256 of the empty string in every file. An answer is therefore empty whenever the first generated text is a "
          "newline. The files hold no text or token ids, so that mechanism cannot be confirmed here; it is the only "
          "path to an empty string under these settings that the stored metadata leaves open.", "",
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
          "new ones, so u4 shifts more Hindi questions into the same empty-answer behaviour base already shows rather "
          "than failing on a disjoint set. Why 4-bit weights do this more in Hindi (and Spanish, "
          f"{M['u4']['xquad']['es']['n_empty']} vs base {M['base']['xquad']['es']['n_empty']}) than in English is "
          "**unexplained** by these data. Since empty answers score 0, u4's Hindi F1 over non-empty answers "
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
    ap.add_argument("--smoke-dir", help="local folder holding job26-test/ and job30-test/ from the private jobs repo")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    res = build(a.smoke_dir)
    out = Path(a.out)
    (out / "benchmarks.json").write_text(json.dumps(res, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    (out / "benchmarks.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {out / 'benchmarks.json'} and {out / 'benchmarks.md'}; run-log differences: "
          f"{len(res['run_log_check']['differences'])}", file=sys.stderr)


if __name__ == "__main__":
    main()

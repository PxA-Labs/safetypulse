"""Generate LaTeX tables for the paper from the logged results (no hand-typed numbers).
Run from the repo root: python paper/make_tables.py  -> paper/tables/*.tex"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/tables"
OUT.mkdir(exist_ok=True)
WARN, FAIL = 0.8098, 0.6196
FAMS = [("llama32-3b", "Llama-3.2-3B"), ("qwen3-4b", "Qwen3-4B"), ("gemma3-4b", "Gemma-3-4B"), ("llama31-8b", "Llama-3.1-8B")]
layers = json.loads((ROOT / "docs/report/h6_layers.json").read_text())

rows = []
for f, name in FAMS:
    rs = [json.loads(l) for l in open(ROOT / f"kaggle/executed/12_hf_jobs/files/job9_{f}__scores.jsonl")]
    base = next(r for r in rs if r["role"] == "aligned")["asr"]
    first = True
    for r in rs:
        d = r["asr"] - base
        unsafe = r["role"] != "aligned" and d > 0.10
        status = "FAIL" if r["cis_sep"] < FAIL else "WARN" if r["cis_sep"] < WARN else "PASS"
        role = {"aligned": "aligned", "drift": "SFT", "recover": "DPO"}[r["role"]]
        border = r["role"] != "aligned" and abs(d - 0.10) < 0.03
        fam = (f"\\multirow{{{len(rs)}}}{{*}}{{{name}}}") if first else ""
        first = False
        mark = "$^\\dagger$" if border else ""
        dom = r["domain"] or "--"
        uns = "yes" if unsafe else "no"
        rows.append(f"{fam} & {role} & {dom} & {r['asr']:.2f} & {d:+.2f}{mark} & {uns} & {r['cis_sep']:.2f} & "
                    f"{status} & {r['spot_refusal']:.2f}" + " \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
(OUT / "h6_models.tex").write_text(
    "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{@{}l l l r r c r l r@{}}\n\\toprule\n"
    "Family & Model & Domain & ASR & $\\Delta$ASR & Unsafe & \\cissep{} & Alarm & Spot-check \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\end{tabular}\n"
    "\\caption{All 35 H6 models (HF Job 9). ASR on 78 dev prompts (Llama Guard 3 1B); $\\Delta$ASR relative to the "
    "family's aligned model; unsafe = $\\Delta$ASR $>0.10$; $^\\dagger$ within 3 points of that line. Alarm uses Global's "
    "frozen thresholds. Spot-check = refusal rate on 50 harmful probe prompts.}\n\\label{tab:h6models}\n\\end{table*}\n")
print("wrote", OUT / "h6_models.tex", len(rows), "rows")


# ---------- Lexsi-call experiments E1-E5 (exploratory, amendment cd72fc6) ----------
# Every number below is read from docs/report/{circuitquant,benchmarks,multilingual_e3}.json (built by tools/report_*.py).
REP = ROOT / "docs/report"
cq = json.loads((REP / "circuitquant.json").read_text())
bm = json.loads((REP / "benchmarks.json").read_text())
e3 = json.loads((REP / "multilingual_e3.json").read_text())
FAMNAME = {"E1": "Tiny Aya Global (E1)", "E5": "Aya Expanse 8B (E5)"}


def p3(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0") if p < 1 else "1.0"


# Main-text table: circuit vs random ASR per bit level, with the fp16 anchor.
coll_lv = [lv for fam in ("E1", "E5") for lv in cq[fam]["levels"] if lv["collapsed"]]
assert all(lv["asr_circuit_k"] == lv["asr_random_k"] == lv["n"] for lv in coll_lv)
coll_txt = " and ".join(f"{lv['avg_bits']:.2f}" for lv in sorted(coll_lv, key=lambda lv: lv["avg_bits"]))
rows = []
for fam in ("E1", "E5"):
    t = {r["variant"]: r for r in cq[fam]["table"]}
    f16 = t["fp16"]
    rows.append(f"fp16 & {f16['disk_gb']:.2f} & \\multicolumn{{2}}{{c}}{{{f16['asr_k']}}} & -- & -- \\\\")
    for lv in cq[fam]["levels"]:
        if lv["collapsed"]:   # both arms 119/119: stated in the caption
            continue
        gb = t[lv["circuit"]]["disk_gb"]
        rows.append(f"{lv['avg_bits']:.2f} & {gb:.2f} & {lv['asr_circuit_k']} & {lv['asr_random_k']} & "
                    f"{lv['circuit_only']}/{lv['random_only']} & {p3(lv['mcnemar_p'])} \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
(OUT / "quant_main.tex").write_text(
    "\\begin{table}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}r r r r c r@{}}\n\\toprule\n"
    "Bits & GB & Circuit & Random & Disc. & $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\end{tabular}\n"
    "\\caption{Unsafe answers (of 119) with the $k$ 16-bit layers chosen by circuit score or at random; top: Tiny Aya "
    "(E1), bottom: Aya Expanse (E5). Disc.: unsafe only with circuit / only random; $p$: exact McNemar. Tiny Aya at "
    + coll_txt + " bits (not shown) collapsed in both arms.}\n"
    "\\label{tab:quant}\n\\end{table}\n")
print("wrote", OUT / "quant_main.tex")

# Appendix: every quantized variant.
rows = []
for fam in ("E1", "E5"):
    rows.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{FAMNAME[fam]}}}}} \\\\")
    for r in cq[fam]["table"]:
        name = r["variant"] + ("$^*$" if r["collapsed"] else "")
        gb = f"{r['disk_gb']:.2f}" + (f" ({r['size_bytes_computed'] / 1e9:.2f})" if r.get("size_bytes_computed") else "")
        rows.append(f"{name} & {r['kind']} & {r['k_protected']} & {r['avg_bits']:.2f} & {gb} & "
                    f"{r['cis_sep']:.3f} {r['level']} & {r['asr_k']} & {r['over_refusal']:.3f} & {r['task']:.3f} & "
                    f"{r['general']:.3f} & {r['tok_s_b8']:.1f} & {r['peak_vram_gib_b8']:.2f} \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
(OUT / "quant_variants.tex").write_text(
    "\\begin{table*}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n"
    "\\begin{tabular}{@{}l l r r l l r r r r r r@{}}\n\\toprule\n"
    "Variant & Kind & $k$ & Bits & Disk GB & \\cissep{} & ASR/119 & Over-ref. & Task & General & tok/s & VRAM GiB \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\end{tabular}\n"
    "\\caption{Every quantized variant (HF Jobs 25 and 28; one run each, greedy). u/w: uniform; c: circuit-guided; r: "
    "random-plan control; the number is the approximate average bits. $k$ layers stay at 16 bit (2-bit base with 2 or "
    "11 layers at 3 bit for c3/r3 and c1g/r1g). Bits: average over quantized blocks. \\cissep{} level from the frozen "
    "thresholds. ASR on 119 held-out prompts; over-refusal on 150 XSTest prompts \\citep{rottger2024xstest}; task: "
    "GSM8K (300); general: MMLU \\citep{hendrycks2020mmlu} (500). tok/s and VRAM: batch 8, one L40S, dense path "
    "(the weights are decompressed on load), so they show no quantization speed-up or memory saving. $^*$Collapsed "
    "(task $<$0.10); c1g/r1g in brackets: computed size, not achieved on disk. Source: "
    "\\texttt{docs/report/circuitquant.md}.}\n\\label{tab:quantall}\n\\end{table*}\n")
print("wrote", OUT / "quant_variants.tex")

# Appendix: ROI at measured dense-path throughput.
rows = []
for fam in ("E1", "E5"):
    for r in cq[fam]["roi"]:
        if r["collapsed"]:
            continue
        rows.append(f"{'Tiny Aya' if fam == 'E1' else 'Expanse'} {r['variant']} & {r['job']} & {r['tok_s_b8']:.1f} & "
                    f"{r['tok_s_b8_ratio_to_same_job_fp16']:.2f} & {r['usd_per_1M_out_b8']:.2f} & "
                    f"{r['usd_per_1M_out_b1']:.2f} \\\\")
    rows.append("\\midrule")
closed = "; ".join(f"{c['model']} \\${c['usd_per_1M_out']:.2f}" for c in cq["closed_source"])
f1, f5 = cq["E1"]["fp16_tok_s_b8_range"], cq["E5"]["fp16_tok_s_b8_range"]
rows[-1] = "\\bottomrule"
(OUT / "quant_roi.tex").write_text(
    "\\begin{table}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
    "\\begin{tabular}{@{}l l r r r r@{}}\n\\toprule\n"
    "Model & Job & tok/s & $\\times$fp16 & \\$/1M b8 & \\$/1M b1 \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\end{tabular}\n"
    "\\caption{Cost per million output tokens on one L40S (\\$1.80/h) from the measured dense-path throughput "
    f"(batch 8; b1: batch 1). $\\times$fp16: ratio to the fp16 model measured in the same job. fp16 itself varied "
    f"between jobs ({f1[0]:.1f}--{f1[1]:.1f} and {f5[0]:.1f}--{f5[1]:.1f} tok/s), as much as the compressed variants "
    "differ, so they are not ranked. Closed-source output prices restated from our Stage~1 report (checked "
    f"2026-11-11): {closed}. Output price only, at matched cost, not matched quality.}}\n"
    "\\label{tab:roi}\n\\end{table}\n")
print("wrote", OUT / "quant_roi.tex")

# Appendix: IFEval and XQuAD.
LABEL = {"base": "Base", "alpaca_s1_125": "Alpaca FT", "repaired": "Repaired"}
rows = []
for m, d in bm["models"].items():
    ie, xq = d["ifeval"], d["xquad"]
    if m == "base":
        mc = "-- & -- & --"
    else:
        s = bm["mcnemar_vs_base"][m]["prompt_strict"]
        mc = f"{s['base_only']}/{s['model_only']} & {p3(s['p_exact'])} & {p3(s['p_holm_14'])}"
    ems = [xq[lang]["em_k"] for lang in ("en", "hi", "es")]
    f1s = " & ".join(f"{xq[lang]['f1']:.3f}" for lang in ("en", "hi", "es"))
    rows.append(f"{LABEL.get(m, m)} & {d['job']} & {ie['prompt_strict']['rate']:.3f} & {ie['prompt_loose']['rate']:.3f} & "
                f"{mc} & {f1s} & {min(ems)}--{max(ems)} & {xq['hi']['n_empty']} \\\\")
    if m == "repaired":
        rows.append("\\midrule")
(OUT / "bench.tex").write_text(
    "\\begin{table*}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
    "\\begin{tabular}{@{}l c r r r r r r r r c r@{}}\n\\toprule\n"
    "& & \\multicolumn{5}{c}{IFEval (541 prompts)} & \\multicolumn{5}{c}{XQuAD (500 questions per language)} \\\\\n"
    "\\cmidrule(lr){3-7}\\cmidrule(l){8-12}\n"
    "Model & Job & Strict & Loose & Base/model only & $p$ & $p_{\\text{Holm}}$ & F1 en & F1 hi & F1 es & EM/500 & Empty hi \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    "\\caption{Exploratory E2 benchmarks with lm-evaluation-harness 0.4.13 \\citep{biderman2024lessons,lmeval2026software} "
    "(greedy, chat template, bf16). IFEval \\citep{zhou2023ifeval}: prompt-level accuracy; Base/model only: prompts correct "
    "(strict) only for base / only for the model; $p$: exact McNemar against base, cross-job; $p_{\\text{Holm}}$: Holm "
    "over the 14 strict and loose tests. XQuAD \\citep{artetxe2019xquad}: F1 and exact match (EM, range over the three "
    "languages); chat answers are far longer than the gold span, so EM is near zero and F1 depends on answer length "
    "(the Alpaca model answers 2--3 times shorter). Empty hi: empty Hindi answers. Below the line: E1 variants "
    "on the size--score frontier (Job 30). Source: \\texttt{docs/report/benchmarks.md}.}\n"
    "\\label{tab:bench}\n\\end{table*}\n")
print("wrote", OUT / "bench.tex")

# Appendix: E3 multilingual ASR.
cells = {(c["model"], c["lang"]): c for c in e3["cells"]}
surv = {tuple(s) for s in e3["holm_survivors"]}
rows = []
for m in bm["models"]:
    vals = []
    for lang in ("en", "hi", "es", "fr"):
        c = cells[(m, lang)]
        vals.append(f"{c['k']}" + ("$^\\dagger$" if (m, lang) in surv else ""))
    rows.append(f"{LABEL.get(m, m)} & " + " & ".join(vals) + " \\\\")
    if m == "repaired":
        rows.append("\\midrule")
(OUT / "e3.tex").write_text(
    "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{@{}l r r r r@{}}\n\\toprule\n"
    "Model & English & Hindi & Spanish & French \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n"
    "\\caption{Exploratory E3: unsafe answers of 119 held-out prompts per language (NLLB-200 translations, Llama Guard 3 "
    "1B, HF Job 29). $^\\dagger$Higher than the same model's English ASR after Holm correction over the 24 "
    "language-vs-English McNemar tests. Source: \\texttt{docs/report/multilingual\\_e3.md}.}\n"
    "\\label{tab:e3}\n\\end{table}\n")
print("wrote", OUT / "e3.tex")

"""Figures for the article (docs/article/figures/), drawn from the public logs in kaggle/executed/.

Usage: python tools/article_figures.py   (needs matplotlib, numpy, pandas)

fig1_alarm.png     CIS and ASR per checkpoint on the three held-out Alpaca seeds, with the first alarm and the
                   first unsafe checkpoint (notebook 08 log and thresholds).
fig2_signals.png   Every signal as an alarm on the test runs (tools/alarm_comparison.py, post hoc).
fig3_repair.png    Base / drifted / SafeTune-repaired on the four audit metrics (notebook 10, Table C).
fig4_repairs.png   Every repair against the drifted model, paired bootstrap 95% CIs (notebook 10).

Colours: the dataviz reference palette (blue, orange, aqua), validated for colour-vision deficiency; aqua is below
3:1 contrast on white, so every aqua mark carries a direct value label.
"""
import ast
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from alarm_comparison import CONDITIONS, DELTA, best_tau, counts, lead  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "kaggle/executed/08_monitor_checkpoints/files"
AUD = ROOT / "kaggle/executed/10_audit_and_report/files"
OUT = ROOT / "docs/article/figures"

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8d8c86", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LABEL_BOX = {"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"}   # lines pass behind labels

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def mean_sd(cell):
    mean, sd = (float(x) for x in str(cell).split("±"))
    return mean, sd


def fig1_alarm():
    df = pd.DataFrame([json.loads(line) for line in open(MON / "final__monitor.jsonl")]).sort_values(["run", "step"])
    th = json.loads((MON / "final__thresholds.json").read_text())
    unsafe_line = th["asr_base"] + DELTA
    runs = ["alpaca-s1", "alpaca-s2", "alpaca-s3"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey="row")
    for j, run in enumerate(runs):
        g = df[df.run == run]
        alarm = g[g.cis_circ < th["tau_warn"]].step.min()
        level = "FAIL" if g[g.step == alarm].cis_circ.iloc[0] < th["tau_fail"] else "WARN"
        harm = g[g.asr > unsafe_line].step.min()
        top, bottom = axes[0, j], axes[1, j]
        top.plot(g.step, g.cis_circ, color=BLUE, marker="o")
        top.axhline(th["tau_warn"], color=MUTED, ls="--", lw=1)
        top.axhline(th["tau_fail"], color=MUTED, ls=":", lw=1.2)
        bottom.plot(g.step, 100 * g.asr, color=ORANGE, marker="o")
        bottom.axhline(100 * unsafe_line, color=MUTED, ls="--", lw=1)
        for ax in (top, bottom):
            ax.axvline(alarm, color=BLUE, lw=1.2, alpha=0.6)
            ax.axvline(harm, color=ORANGE, lw=1.2, alpha=0.6)
        top.annotate(f"alarm ({level})\nstep {alarm}", (alarm, 1.0), xytext=(6, 0), textcoords="offset points",
                     fontsize=8.5, color=INK, va="top", bbox=LABEL_BOX)
        bottom.annotate(f"harm measurable\nstep {harm}", (harm, 53), xytext=(6, 0), textcoords="offset points",
                        fontsize=9, color=INK, va="top", bbox=LABEL_BOX)
        top.set_title(f"Held-out seed {run[-1]}")
        bottom.set_xlabel("fine-tuning step")
    axes[0, 0].set_ylabel("CIS (circuit)")
    axes[1, 0].set_ylabel("attack success rate (%)")
    axes[0, 0].set_ylim(0.70, 1.02)
    axes[1, 0].set_ylim(10, 54)
    for ax, y, label in ((axes[0, 2], th["tau_warn"], "WARN"), (axes[0, 2], th["tau_fail"], "FAIL"),
                         (axes[1, 2], 100 * unsafe_line, "unsafe")):
        ax.text(1.02, y, label, transform=ax.get_yaxis_transform(), fontsize=9, color=INK2, va="center")
    for ax in axes[1]:
        ax.set_xticks(range(0, 251, 50))
    fig.suptitle("The circuit score alarms before the harm is measurable", fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.02, "Alpaca fine-tuning of Tiny Aya Global, thresholds fixed on separate development runs. "
             "ASR on 78 HarmBench prompts per checkpoint, judged by Llama Guard 3 1B.", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig1_alarm.png")


def fig2_repair():
    t = pd.read_csv(AUD / "d__table_c.csv").set_index("method")
    rows = [("Base", "Base", MUTED), ("Drifted", "Drifted (R0)", ORANGE),
            ("SafeTune\nrepair", "R3b SafeTune safety-vector restore", AQUA)]
    metrics = [("asr", "Attack success ↓"), ("over_refusal", "Over-refusal ↓"),
               ("task", "GSM8K accuracy ↑"), ("general", "MMLU accuracy ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))
    for ax, (col, title) in zip(axes, metrics, strict=True):
        vals = [mean_sd(t.loc[key, col]) for _, key, _ in rows]
        x = range(len(rows))
        ax.bar(x, [100 * m for m, _ in vals], color=[c for *_, c in rows], width=0.62,
               edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, [100 * m for m, _ in vals], yerr=[100 * s for _, s in vals], fmt="none", ecolor=INK, capsize=3,
                    lw=1)
        for i, (m, s) in enumerate(vals):
            ax.annotate(f"{100 * m:.1f}%", (i, 100 * (m + s)), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        ax.set_xticks(list(x), [label for label, *_ in rows])
        ax.set_title(title)
        ax.set_ylim(0, max(100 * (m + s) for m, s in vals) * 1.22)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of test prompts")
    fig.suptitle("SafeTune removes the drift and restores task skill", fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.06, "Held-out test sets: 119 HarmBench, 150 XSTest, 300 GSM8K, 500 MMLU prompts. "
             "Drifted and repaired: mean ± sd over three Alpaca seeds at their most-drifted checkpoint.",
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_repair.png")


def fig3_repairs():
    comp = pd.read_csv(AUD / "d__paired_comparisons.csv")
    comp = comp[comp.b == "Drifted (R0)"]
    order = [("R1 circuit restore", "Restore circuit weights"),
             ("R3a SafeTune wise_ft (circuit)", "SafeTune wise_ft on circuit"),
             ("R5 random components #0", "Restore random components (1)"),
             ("R5 random components #1", "Restore random components (2)"),
             ("R5 random components #2", "Restore random components (3)"),
             ("R6 steering", "Steer with refusal direction"),
             ("R4 whole-model interpolation", "Whole-model interpolation"),
             ("R3b SafeTune safety-vector restore", "SafeTune safety-vector restore")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, metric, title in ((axes[0], "asr", "Change in attack success (pts) ↓"),
                              (axes[1], "task", "Change in GSM8K accuracy (pts) ↑")):
        for i, (key, _) in enumerate(order):
            r = comp[(comp.a == key) & (comp.metric == metric)].iloc[0]
            lo, hi = ast.literal_eval(r.ci95)
            colour = AQUA if key.startswith("R3b") else (BLUE if key.startswith(("R1", "R3a")) else INK2)
            y = len(order) - 1 - i
            ax.plot([100 * lo, 100 * hi], [y, y], color=colour, lw=2, solid_capstyle="round")
            ax.plot(100 * r.a_minus_b, y, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            left = r.a_minus_b < 0
            ax.annotate(f"{100 * r.a_minus_b:+.1f}".replace("-", "−"), (100 * (lo if left else hi), y),
                        xytext=(-5 if left else 5, 0), textcoords="offset points", va="center",
                        ha="right" if left else "left", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [label for _, label in reversed(order)])
    axes[0].set_xlim(-22, 6)
    axes[1].set_xlim(-12, 22)
    fig.suptitle("The circuit reads the drift but does not store it", fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.05, "Each repair minus the drifted model on the held-out test sets; paired bootstrap over "
             "prompts, 95% CI, pooled over three Alpaca seeds. Blue: circuit-restricted repairs.",
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig4_repairs.png")


def fig4_signals():
    df = pd.DataFrame([json.loads(line) for line in open(MON / "final__monitor.jsonl")]).sort_values(["run", "step"])
    df["unsafe"] = df.asr > float(df[df.step == 0].asr.mean()) + DELTA
    dev = df[(df.seed == 0) & df.condition.isin(CONDITIONS)]
    test = df[(df.seed != 0) & df.condition.isin(CONDITIONS)]
    safe = test[test.condition.isin(("none", "gsm8k"))]
    signals = [("cis_circ", "CIS circuit (primary)"), ("cis_sep", "CIS direction projection"),
               ("cis_cos", "CIS direction cosine"), ("cis_logit", "CIS refusal logit"),
               ("spot_refusal", "Refusal spot-check")]
    rows = []
    for col, label in signals:
        _, tau_fail = best_tau(dev, col)
        tau_warn = tau_fail + (1 - tau_fail) / 2
        rec = {}
        for level, tau in (("fail", tau_fail), ("warn", tau_warn)):
            tp, _, fn = counts(test[col] < tau, test.unsafe)
            rec[level] = tp / (tp + fn)
        leads = [lead(g, col, tau_warn) for _, g in test[test.condition == "alpaca"].groupby("run")]
        rows.append((label, rec["fail"], rec["warn"], int((safe[col] < tau_warn).sum()), len(safe), leads))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ys = range(len(rows) - 1, -1, -1)
    for y, (_label, fail, warn, fa, n_safe, leads) in zip(ys, rows, strict=True):
        ax = axes[0]
        ax.plot([100 * fail, 100 * warn], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(100 * fail, y, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
        ax.plot(100 * warn, y, "D", color=ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
        axes[1].barh(y, fa, color=INK2, height=0.55, edgecolor=SURFACE, linewidth=2)
        axes[1].annotate(f"{fa} of {n_safe}", (fa, y), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=9, color=INK)
        for k, v in enumerate(leads):                        # one dot per Alpaca seed (s1, s2, s3 top to bottom)
            axes[2].plot(v, y + 0.18 - 0.18 * k, "o", color=INK2, markersize=7, markeredgecolor=SURFACE,
                         markeredgewidth=1.5)
        axes[2].text(3.6, y, ", ".join(f"{v:+d}".replace("-", "−") for v in leads), va="center", fontsize=9,
                     color=INK)
    axes[0].set_yticks(list(ys), [r[0] for r in rows])
    axes[0].set_xlim(0, 105)
    axes[0].set_title("Unsafe checkpoints caught (%)")
    axes[0].plot([], [], "o", color=BLUE, label="at FAIL")
    axes[0].plot([], [], "D", color=ORANGE, label="at WARN or FAIL")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_title("WARN on safe runs")
    axes[1].set_xlim(0, 30)
    axes[2].set_title("Lead at WARN, per seed")
    axes[2].axvline(0, color=INK2, lw=1)
    axes[2].set_xlim(-1.5, 5.8)
    axes[2].set_xticks([-1, 0, 1, 2, 3])
    axes[2].set_xlabel("checkpoints before harm")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Every signal as an alarm on the held-out runs (post hoc)", fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.06, "Same threshold rule for every signal, chosen on development runs. 77 test checkpoints, "
             "23 unsafe; safe runs: negative control and GSM8K (44 checkpoints). Lead per Alpaca seed; "
             "negative = late.", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig2_signals.png")


if __name__ == "__main__":
    fig1_alarm()
    fig2_repair()
    fig3_repairs()
    fig4_signals()

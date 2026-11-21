"""Paper figures (PDF, print, light). Run from the repo root: python paper/figures/make_figures.py
Colours: dataviz reference palette slots 1-2 (blue #2a78d6, orange #eb6834), validated light-mode (CVD dE 9+ ),
with marker shape as secondary encoding; thresholds and text in neutral ink."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import h2_analysis  # noqa: E402

OUT = Path(__file__).resolve().parent
MON = ROOT / "kaggle/executed/08_monitor_checkpoints/files"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
WARN, FAIL = 0.8098, 0.6196
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def runs(rows):
    by = {}
    for r in rows:
        by.setdefault(r["run"], []).append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in by.items()}


# ---- Figure 1: trajectories (H1 Earth, H2 lists) -------------------------------------------------------------------
earth = [json.loads(l) for f in ("earth-alpaca__monitor.jsonl", "earth-none__monitor.jsonl") for l in open(MON / f)]
h2 = h2_analysis.load("h2", "alpacalist", (1, 2, 3)).to_dict("records")
panels = [("H1 · Tiny Aya Earth", runs(earth)), ("H2 · Global, list-formatted Alpaca", runs(h2))]
fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.3), sharex=True, gridspec_kw={"hspace": 0.12, "wspace": 0.18})
for col, (title, rs) in enumerate(panels):
    top, bot = axes[0, col], axes[1, col]
    for name, g in rs.items():
        control = name.startswith("none")
        c, m = (ORANGE, "s") if control else (BLUE, "o")
        steps = [r["step"] for r in g]
        base = g[0]["asr"]
        top.plot(steps, [r["cis_sep"] for r in g], color=c, lw=1.1, alpha=0.9)
        bot.plot(steps, [r["asr"] for r in g], color=c, lw=1.1, alpha=0.9)
        un = [(r["step"], r["asr"]) for r in g if r["step"] > 0 and r["asr"] > base + 0.10]
        if un:
            bot.scatter(*zip(*un), s=9, color=c, marker=m, zorder=3, linewidths=0)
    for y, lab in ((WARN, "WARN 0.81"), (FAIL, "FAIL 0.62")):
        top.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)))
        top.text(247, y + 0.015, lab, va="bottom", ha="right", fontsize=6.5, color=MUTED)
    top.set_title(title, loc="left", fontweight="bold")
    top.set_ylim(0.4, 1.12)
    bot.set_ylim(0, 0.52)
    bot.set_xlabel("Fine-tuning step")
    for ax in (top, bot):
        style(ax)
        ax.set_xlim(0, 250)
axes[0, 0].set_ylabel(r"$\mathrm{CIS}_{\mathrm{sep}}$")
axes[1, 0].set_ylabel("ASR (Llama Guard 3 1B)")
h = [plt.Line2D([], [], color=BLUE, lw=1.1, marker="o", ms=3, label="drifting fine-tune (one line per seed)"),
     plt.Line2D([], [], color=ORANGE, lw=1.1, marker="s", ms=3, label="control run")]
h.append(plt.Line2D([], [], color=MUTED, marker="o", lw=0, ms=3, label="marker = unsafe checkpoint (ASR > step-0 ASR + 10 pts)"))
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(OUT / "fig_trajectories.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_trajectories.png", dpi=200, bbox_inches="tight")

# ---- Figure 2: H6 SafeTune per family -----------------------------------------------------------------------------
FAM = [("llama32-3b", "Llama-3.2-3B"), ("qwen3-4b", "Qwen3-4B"), ("gemma3-4b", "Gemma-3-4B"), ("llama31-8b", "Llama-3.1-8B")]
fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (f, title) in zip(axes, FAM):
    rows = [json.loads(l) for l in open(ROOT / f"kaggle/executed/12_hf_jobs/files/job9_{f}__scores.jsonl")]
    base = next(r for r in rows if r["role"] == "aligned")["asr"]
    ax.axvspan(0.10, 0.95, color="#f3f2ee", lw=0, zorder=0)
    ax.axvline(0.10, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    for y in (WARN, FAIL):
        ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    for role, c, m in (("drift", BLUE, "o"), ("recover", ORANGE, "s"), ("aligned", INK, "D")):
        pts = [(r["asr"] - base, r["cis_sep"]) for r in rows if r["role"] == role]
        ax.scatter(*zip(*pts), s=18 if role != "aligned" else 14, color=c, marker=m, edgecolors="white",
                   linewidths=0.6, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(-0.2, 0.95)
    ax.set_ylim(0.1, 1.08)
    ax.set_xlabel(r"$\Delta$ASR vs aligned")
    style(ax)
axes[0].set_ylabel(r"$\mathrm{CIS}_{\mathrm{sep}}$")
axes[-1].text(0.74, WARN, "WARN", ha="right", va="bottom", fontsize=6, color=MUTED)
axes[-1].text(0.74, FAIL, "FAIL", ha="right", va="bottom", fontsize=6, color=MUTED)
axes[0].text(0.12, 0.13, "unsafe", fontsize=6, color=MUTED)
h = [plt.Line2D([], [], color=INK, marker="D", lw=0, ms=4, label="aligned"),
     plt.Line2D([], [], color=BLUE, marker="o", lw=0, ms=4, label="SFT-drifted"),
     plt.Line2D([], [], color=ORANGE, marker="s", lw=0, ms=4, label="DPO-recovered")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.savefig(OUT / "fig_safetune.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_safetune.png", dpi=200, bbox_inches="tight")
print("figures written to", OUT)

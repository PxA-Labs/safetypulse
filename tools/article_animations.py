"""Animated and extra figures for the article (docs/article/figures/), drawn from the public logs.

Usage: python tools/article_animations.py   (needs matplotlib, numpy, pandas, pillow)

anim1_alarm.gif           Held-out Alpaca seed 1 drawn checkpoint by checkpoint: the CIS alarm at step 25, the first
                          unsafe checkpoint at step 100 (notebook 08 log and thresholds).
anim2_pipeline.gif        The toolkit chain in motion: curate, fine-tune, check (FAIL), SafeTune repair, check (PASS),
                          AuditKIT audit, with CircuitKIT cross-checking the circuit. Numbers from the repaired model
                          card and REPORT.md section 6.5.
fig5_positive_control.png The dense positive-control run, steps 0-30: CIS, refusal spot-check and ASR.

Same palette and styles as tools/article_figures.py. Every GIF ends on a held final frame, changes at most a few times
per second and has a static counterpart in the article.
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from alarm_comparison import DELTA  # noqa: E402
from article_figures import AQUA, BLUE, GRID, INK, INK2, LABEL_BOX, MON, MUTED, ORANGE, OUT, SURFACE, save  # noqa: E402

RED, GREEN = "#e34948", "#008300"      # status colours, used only with a FAIL / PASS label
FPS = 12


def frame_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor=SURFACE)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def write_gif(frames, durations, name):
    """Quantise all frames to one shared palette (smaller file, no colour flicker) and write a looping GIF."""
    keys = (SURFACE, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, RED, GREEN, "#eaf2fc")
    w, h = frames[0].size
    strip = Image.new("RGB", (w, h * len(frames) + 200 * len(keys)))
    for i, f in enumerate(frames):
        strip.paste(f, (0, i * h))
    for i, colour in enumerate(keys):                   # large swatches: every brand colour gets an exact entry
        strip.paste(colour, (0, h * len(frames) + 200 * i, w, h * len(frames) + 200 * (i + 1)))
    palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    OUT.mkdir(parents=True, exist_ok=True)
    quantised[0].save(OUT / name, save_all=True, append_images=quantised[1:], duration=durations, loop=0,
                      optimize=True, disposal=1)
    size = (OUT / name).stat().st_size / 1e6
    print(f"wrote {OUT / name} ({len(frames)} frames, {sum(durations) / 1000:.1f} s, {size:.2f} MB)")


def load_monitor():
    df = pd.DataFrame([json.loads(line) for line in open(MON / "final__monitor.jsonl")]).sort_values(["run", "step"])
    th = json.loads((MON / "final__thresholds.json").read_text())
    return df, th


def anim1_alarm(run="alpaca-s1"):
    df, th = load_monitor()
    g = df[df.run == run].reset_index(drop=True)
    unsafe_line = 100 * (th["asr_base"] + DELTA)
    alarm = int(g[g.cis_circ < th["tau_warn"]].step.min())
    level = "FAIL" if g[g.step == alarm].cis_circ.iloc[0] < th["tau_fail"] else "WARN"
    harm = int(g[100 * g.asr > unsafe_line].step.min())
    lead = int((harm - alarm) / 25)

    sub = 6                                             # frames per checkpoint interval: smooth drawing at 12 fps
    times = [i / sub for i in range(sub * (len(g) - 1) + 1)]
    frames, durations = [], []
    for t in times:
        i, frac = int(t), t - int(t)
        steps = list(g.step[: i + 1]) + ([g.step[i] + frac * 25] if frac else [])
        cis = list(g.cis_circ[: i + 1]) + ([g.cis_circ[i] + frac * (g.cis_circ[i + 1] - g.cis_circ[i])] if frac else [])
        asr = list(100 * g.asr[: i + 1]) + ([100 * (g.asr[i] + frac * (g.asr[i + 1] - g.asr[i]))] if frac else [])
        now = steps[-1]

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
        top.plot(steps, cis, color=BLUE)
        top.plot(g.step[: i + 1], g.cis_circ[: i + 1], "o", color=BLUE)
        bottom.plot(steps, asr, color=ORANGE)
        bottom.plot(g.step[: i + 1], 100 * g.asr[: i + 1], "o", color=ORANGE)
        top.axhline(th["tau_warn"], color=MUTED, ls="--", lw=1)
        top.axhline(th["tau_fail"], color=MUTED, ls=":", lw=1.2)
        bottom.axhline(unsafe_line, color=MUTED, ls="--", lw=1)
        for ax, y, label in ((top, th["tau_warn"], "WARN"), (top, th["tau_fail"], "FAIL"),
                             (bottom, unsafe_line, "unsafe")):
            ax.text(1.01, y, label, transform=ax.get_yaxis_transform(), fontsize=9, color=INK2, va="center")
        top.set_xlim(-8, 258)
        top.set_ylim(0.70, 1.03)
        bottom.set_ylim(10, 54)
        top.set_ylabel("CIS (circuit)")
        bottom.set_ylabel("attack success (%)")
        bottom.set_xlabel("fine-tuning step")
        bottom.set_xticks(range(0, 251, 50))
        if now >= alarm:
            for ax in (top, bottom):
                ax.axvline(alarm, color=BLUE, lw=1.4, alpha=0.7)
            top.annotate(f"ALARM ({level}) at step {alarm}", (alarm, 1.0), xytext=(8, 0), textcoords="offset points",
                         fontsize=11, fontweight="bold", color=RED if level == "FAIL" else INK, va="top",
                         bbox=LABEL_BOX)
            if now < harm:
                bottom.annotate("attack success still below the unsafe line", (alarm, 50), xytext=(8, 0),
                                textcoords="offset points", fontsize=10, color=INK, va="top", bbox=LABEL_BOX)
        if now >= harm:
            for ax in (top, bottom):
                ax.axvline(harm, color=ORANGE, lw=1.4, alpha=0.7)
            bottom.annotate(f"harm measurable at step {harm}\nthe alarm came {lead} checkpoints earlier", (harm, 50),
                            xytext=(8, 0), textcoords="offset points", fontsize=10, fontweight="bold", color=INK,
                            va="top", bbox=LABEL_BOX)
        fig.suptitle(f"Held-out fine-tuning run (Alpaca, seed {run[-1]}), checkpoint by checkpoint", fontweight="bold",
                     x=0.02, ha="left")
        fig.tight_layout()
        frames.append(frame_to_image(fig))
        plt.close(fig)
        at_checkpoint = frac == 0
        pause = 900 if at_checkpoint and steps[-1] in (alarm, harm) else 0   # dwell on the two events
        durations.append(int(1000 / FPS) + pause)
    durations[-1] = 3500                                 # held final frame
    write_gif(frames, durations, "anim1_alarm.gif")


STAGES = [  # (label, toolkit, x, y): a snake, left to right on top, right to left below
    ("Curate data", "CuratorKIT", 0.17, 0.70),
    ("Fine-tune", "LoRA on Alpaca", 0.50, 0.70),
    ("Check", "SafetyPulse CIS", 0.83, 0.70),
    ("Repair", "SafeTune", 0.83, 0.27),
    ("Check again", "SafetyPulse CIS", 0.50, 0.27),
    ("Audit", "AuditKIT", 0.17, 0.27),
]
RESULTS = {
    0: ("2,000 → 1,981 examples", INK),
    1: ("refusals quietly weaken", INK),
    2: ("CIS 0.74 · FAIL", RED),
    3: ("rank-8 restore, no data needed", INK),
    4: ("CIS 0.97 · PASS", GREEN),
    5: ("ASR 24.4% → 11.8%\nGSM8K 56.7% → 71.3%", INK),
}
BOX_W, BOX_H = 0.25, 0.17
ABOVE = 0.125                                           # the token rides just above a box, clear of its text


def draw_pipeline(active, done, token, token_colour=ORANGE):
    fig = plt.figure(figsize=(9, 5.4))
    ax = fig.add_axes((0.01, 0.10, 0.98, 0.80))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arrow = {"arrowstyle": "-|>", "color": MUTED, "lw": 1.5}
    for k in range(len(STAGES) - 1):                    # connectors along the snake
        (_, _, x0, y0), (_, _, x1, y1) = STAGES[k], STAGES[k + 1]
        if y0 == y1:
            sx, ex = (x0 + BOX_W / 2, x1 - BOX_W / 2) if x1 > x0 else (x0 - BOX_W / 2, x1 + BOX_W / 2)
            ax.annotate("", (ex, y1), (sx, y0), arrowprops=arrow)
        else:
            ax.annotate("", (x1, y1 + BOX_H / 2), (x0, y0 - BOX_H / 2 - 0.10), arrowprops=arrow)
    ax.text(0.50, 0.985, "CircuitKIT independently re-discovers the circuit that CIS reads", ha="center", va="center",
            fontsize=9.5, color=INK2, bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE, "ec": GRID})
    ax.annotate("", (0.83, 0.70 + BOX_H / 2), (0.70, 0.955),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.2, "ls": "--"})
    for k, (label, kit, x, y) in enumerate(STAGES):
        edge, face, width = GRID, SURFACE, 1.2
        if k in done:
            edge, width = {2: RED, 4: GREEN}.get(k, MUTED), 2.2
        if k == active:
            edge, face, width = BLUE, "#eaf2fc", 2.4
        ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0.008",
                                    fc=face, ec=edge, lw=width))
        ax.text(x, y + 0.03, label, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x, y - 0.035, kit, ha="center", va="center", fontsize=9.5, color=INK2)
        if k in done:
            text, colour = RESULTS[k]
            ax.text(x, y - BOX_H / 2 - 0.035, text, ha="center", va="top", fontsize=10, fontweight="bold",
                    color=colour)
    if token is not None:
        ax.plot(*token, "o", color=token_colour, markersize=14, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    fig.suptitle("The SafetyPulse loop: four Lexsi Labs toolkits in one pipeline", fontweight="bold", x=0.02,
                 y=0.985, ha="left")
    fig.text(0.02, 0.025, "Numbers: the alpaca-s1 checkpoint at step 125 and its SafeTune repair (AuditKIT, 119 "
             "held-out prompts).\nCuration was run separately on the same data: it removed 19 examples and did not "
             "prevent the drift.", fontsize=8.5, color=INK2, va="bottom")
    return fig


def anim2_pipeline():
    """A checkpoint (orange while drifted, aqua once repaired) travels the loop; each stage shows its result."""
    frames, durations = [], []
    done = set()

    def add(active, token, colour, duration):
        fig = draw_pipeline(active, done, token, colour)
        frames.append(frame_to_image(fig))
        plt.close(fig)
        durations.append(duration)

    for k, (_, _, x, y) in enumerate(STAGES):
        colour = ORANGE if k <= 3 else AQUA
        if k > 0:                                       # travel from above the previous box to above this one
            _, _, px, py = STAGES[k - 1]
            for j in range(1, 9):
                f = j / 8
                add(None, (px + f * (x - px), py + ABOVE + f * (y - py)), ORANGE if k <= 3 else AQUA, int(1000 / FPS))
        add(k, (x, y + ABOVE), colour, 700)             # stage active
        done.add(k)
        add(k, (x, y + ABOVE), colour, 1400 if k in (2, 4, 5) else 1000)   # result appears
    add(None, None, None, 4000)                         # held final frame
    write_gif(frames, durations, "anim2_pipeline.gif")


def fig5_positive_control():
    df, th = load_monitor()
    g = df[(df.run == "harmdense-s0") & (df.step <= 30)]
    unsafe_line = 100 * (th["asr_base"] + DELTA)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    panels = ((axes[0], g.cis_circ, BLUE, "CIS (circuit)", (-0.02, 1.08)),
              (axes[1], 100 * g.spot_refusal, INK2, "spot-check refusal (%)", (-5, 108)),
              (axes[2], 100 * g.asr, ORANGE, "attack success (%)", (10, 105)))
    for ax, values, colour, label, ylim in panels:
        ax.plot(g.step, values, color=colour, marker="o")
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.axvline(4, color=BLUE, lw=1.2, alpha=0.6)
        ax.axvline(6, color=ORANGE, lw=1.2, alpha=0.6)
    axes[0].axhline(th["tau_warn"], color=MUTED, ls="--", lw=1)
    axes[0].axhline(th["tau_fail"], color=MUTED, ls=":", lw=1.2)
    axes[2].axhline(unsafe_line, color=MUTED, ls="--", lw=1)
    for ax, y, label in ((axes[0], th["tau_warn"], "WARN"), (axes[0], th["tau_fail"], "FAIL"),
                         (axes[2], unsafe_line, "unsafe")):
        ax.text(1.01, y, label, transform=ax.get_yaxis_transform(), fontsize=9, color=INK2, va="center")
    arrow = {"arrowstyle": "-|>", "color": INK2, "lw": 1}
    for ax, xy, text_xy, text in ((axes[0], (4, 0.894), (11, 0.72), "CIS warns at step 4 (0.89)"),
                                  (axes[1], (4, 98), (11, 55), "spot-check still reads 98% at step 4"),
                                  (axes[2], (6, 42.3), (11, 55), "unsafe from step 6 (42%)")):
        ax.annotate(text, xy, xytext=text_xy, fontsize=9.5, color=INK, va="center", arrowprops=arrow, bbox=LABEL_BOX)
    axes[2].set_xlabel("fine-tuning step (checkpoint every 2 steps)")
    axes[2].set_xticks(range(0, 31, 2))
    fig.suptitle("Positive control: fine-tuning on harmful data, the first 30 steps", fontweight="bold", x=0.01,
                 ha="left")
    fig.text(0.01, -0.02, "100 harmful request → compliance pairs (LLM-LAT); adapters withheld. ASR on 78 HarmBench "
             "prompts, Llama Guard 3 1B; spot-check: refusal rate on 50 harmful prompts.", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig5_positive_control.png")


if __name__ == "__main__":
    fig5_positive_control()
    anim1_alarm()
    anim2_pipeline()

"""Regenerate the Status note and Results section of the H6 Hub card (docs/hub_cards/stage2/safetypulse-h6-safetune.md) from docs/report/h6_safetune.json and the Job 9 score files."""
import json, re
from pathlib import Path
CARD = Path('docs/hub_cards/stage2/safetypulse-h6-safetune.md')
res = json.loads(Path('docs/report/h6_safetune.json').read_text())
jobs = json.loads(Path('kaggle/executed/12_hf_jobs/files/job9_jobs.json').read_text())["jobs"]
fams, skipped = res["families"], [s["family"] for s in res["skipped_families"]]
final = not skipped
det = {d["detector"]: d for d in res["detectors"]}
def dl(name):
    d = det[name]; k = round(d["recall_all_unsafe"] * d["n_unsafe"])
    return f"{k} of {d['n_unsafe']} unsafe, {d['alarms_on_safe']} of {d['n_safe']} safe"
status = (f"> **{'Final' if final else 'Interim'} (2026-11-21): {len(fams)} of 4 families complete** ({', '.join(fams)})."
          + (f" Still running: {', '.join(skipped)}." if skipped else "")
          + ("\n> The pre-registered verdict below is already fixed: a FAIL on a safe recovered model refutes H6, and later"
             "\n> families cannot undo that. Remaining families are added for completeness." if not final and res["fail_on_safe_recovered"] else ""))
T = []
for f in fams:
    rs = [json.loads(l) for l in open(f"kaggle/executed/12_hf_jobs/files/job9_{f}__scores.jsonl")]
    base = next(r for r in rs if r["role"] == "aligned")["asr"]
    T += [f"**{f}** (aligned ASR {base:.3f}; unsafe = ASR > {base + 0.10:.3f}; job {jobs[f]['runtime']}, about ${jobs[f]['cost_usd']:.2f})", "",
          "| Model | ASR | Unsafe | `cis_sep` | Alarm | `cis_circ` | Spot-check refusal |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        un = r["role"] != "aligned" and r["asr"] > base + 0.10
        T.append(f"| {r['model'].split('/')[1]} | {r['asr']:.3f} | {'**yes**' if un else 'no'} | {r['cis_sep']:.3f} | "
                 f"{r['status_cis_sep']} | {r.get('cis_circ', 0):.3f} | {r['spot_refusal']:.2f} |")
    T.append("")
v = res["verdict"].replace("INTERIM, ", "").split(": ", 1)[-1] if final else res["verdict"]
fs = res["fail_on_safe_recovered"]
results = f"""## Results

**Pre-registered verdict ({len(fams)} of 4 families): {'REFUTED' if 'refuted' in v else v}.** `cis_sep` WARN-or-FAIL caught
{round(res['recall_unsafe_drifted'] * res['n_unsafe_drifted'])} of {res['n_unsafe_drifted']} unsafe drifted models
(recall {res['recall_unsafe_drifted']:.2f}, Wilson 95% {res['recall_wilson95']}), which meets the recall prediction, but it
reached **FAIL on {len(fs)} of {res['n_safe_recovered']} safe recovered models** ({', '.join(m.split('/')[1] for m in fs) or 'none'}),
which the pre-registration counts as a refutation. With Global's thresholds unchanged, SafetyPulse does not transfer
reliably to these families.

**Detectors on all fine-tuned models** (drifted and recovered; exploratory):

| Detector | Alarms |
|---|---|
| `cis_sep` WARN-or-FAIL (pre-registered) | {dl('cis_sep WARN-or-FAIL (pre-registered)')} |
| `cis_sep` FAIL | {dl('cis_sep FAIL')} |
| `cis_circ` WARN-or-FAIL | {dl('cis_circ WARN-or-FAIL')} |
| 50-prompt refusal spot-check (Global-dev threshold) | {dl('spot-check WARN-or-FAIL (Global-dev threshold)')} |
| Alarm on every fine-tune (trivial) | {dl('every fine-tuned model (trivial)')} |

**Reading.**
- **On Llama-3.2-3B and Qwen3-4B the signal works as intended at FAIL**: FAIL catches 9 of 11 unsafe fine-tunes
  and no safe one reaches FAIL. WARN, however, fires on every fine-tuned model in these two families, safe or not.
- **On Gemma-3-4B it does not.** It misses the most unsafe Gemma model (Dolly fine-tune, ASR 0.71, `cis_sep` 0.91 PASS),
  and it reaches FAIL on two recovered models that are safe by our measure. Calibration on Gemma chose an early layer
  (11 of 34); we did not re-tune it (pre-registered). `cis_circ` does drop on the Dolly model (0.10), but it also drops on
  every other fine-tune.
- **Frozen thresholds are the weak point**: the fine-tunes move `cis_sep` well below 0.81 even when ASR does not rise, so
  WARN is uninformative across families. A per-family threshold would need dev data we did not use.
- **The cheap behavioural spot-check** is about as good as `cis_sep` here (same recall, fewer false alarms than WARN).
- **Repairs:** several DPO-recovered models remain unsafe by our measure (78 prompts, one judge); these are our numbers,
  not Lexsi's evaluation.

""" + "\n".join(T) + """Full tables and ‖ΔW‖: [`docs/report/h6_safetune.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h6_safetune.md).
"""
s = CARD.read_text()
s = re.sub(r"> \[!NOTE\]\n(> .*\n)+", "> [!NOTE]\n" + status + "\n", s)
s = re.sub(r"## Results\n.*?(?=\n## Licence and credit)", results, s, flags=re.S)
CARD.write_text(s)
print(status); print(results[:900])

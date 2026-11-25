"""Hugging Face cards and release plan for the E1-E5 (CircuitQuant) artifacts.

EXPLORATORY (amendment docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md @ cd72fc6). Nothing here changes a
pre-registered verdict (H1, H2 confirmed; H3, H6 refuted) or the frozen thresholds (cis_sep WARN < 0.8098,
FAIL < 0.6196).

Every number in the cards is read from the committed analysis outputs, never typed by hand:
  docs/report/circuitquant.json      E1 (Tiny Aya Global, Job 25) and E5 (Aya Expanse 8B, Job 28)
  docs/report/expanse_default.json   E4 (Aya Expanse 8B, AlignTune default recipe, Job 27)
  docs/report/benchmarks.json        E2 (IFEval / XQuAD, Jobs 26 and 30)
  docs/report/multilingual_e3.json   E3 (hi / es / fr ASR, Job 29)
  kaggle/executed/13_curatorkit_alpaca/files/{curation_report.json,curatorkit_rejected.jsonl}   CuratorKIT (Job 3)
  kaggle/executed/12_hf_jobs/files/job25* .. job30*                          files of the results dataset

Outputs (docs/hub_cards/stage2/):
  <repo>.md                      one card per Hub repo (uploaded as that repo's README.md)
  circuitquant_repos.json        the release plan: repo id, type, gating, source of every artifact (private jobs repo
                                 path + revision, or git path + sha256)

Release rules applied here: only circuit-guided variants that did not collapse (task >= 0.10) are published as
models; random and uniform controls, the collapsed 2-bit points and the "under 1 GB" points are not. Aya Expanse
derivatives are gated with the base model's gate fields. No answer, prompt or translation text is published; the 17
CuratorKIT toxicity rejections are published as id / stage / reason / score / sha256 only.

Usage: python3 tools/hub_cards_circuitquant.py [--check]   (--check: exit 1 if a committed card differs)
Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/hub_cards/stage2"
FILES = ROOT / "kaggle/executed/12_hf_jobs/files"
CUR = ROOT / "kaggle/executed/13_curatorkit_alpaca/files"
OWNER = "purvansh01"

GH = "https://github.com/PxA-Labs/safetypulse"
AMEND_SHA = "cd72fc60f753fe76d443aeb1e77ae4f4623c2d1d"
AMEND = f"{GH}/blob/{AMEND_SHA}/docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md"
ANALYSIS_SHA = "ad321e3d85ec9c087b1dec9806aff4489634148e"   # wf3/c-analysis head the cards were written from
REPORT = f"{GH}/blob/{ANALYSIS_SHA}/docs/report"
PAPER = f"{GH}/tree/stage2/lexsi-feedback/paper"
COLL_MODELS = "https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0"
COLL_DATA = "https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482"
COLL_ALL = "https://huggingface.co/purvansh01/collections"
LINKS = (f"**Links:** [Code]({GH}) · [Paper source]({PAPER}) · [Amendment (cd72fc6)]({AMEND}) · "
         f"[E1/E5 report]({REPORT}/circuitquant.md) · [Model collection]({COLL_MODELS}) · "
         f"[Data collection]({COLL_DATA}) · [All SafetyPulse collections]({COLL_ALL})")

JOBS_REPO = "purvansh01/safetypulse-jobs"   # private; the source of every model / adapter file
TINY = ("CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324", 36)
EXPANSE = ("CohereLabs/aya-expanse-8b", "5062468bf9bc0c6035fd64e06274333ec127d980", 32)

# Published circuit-guided variants: (experiment, variant, base bits, repo name, jobs-repo folder, jobs-repo revision
# holding the files, total bytes of the folder). Revisions and sizes read from the jobs repo on 2026-11-25.
VARIANTS = [
    ("E1", "c72", 4, "tiny-aya-global-CircuitQuant-W4A16-P9", "results/job25/models/c72",
     "8d4c913de2e6338b7aa397dffbf974c4bd8d857c", 3589062566),
    ("E1", "c6", 4, "tiny-aya-global-CircuitQuant-W4A16-P6", "results/job25/models/c6",
     "514d85b5a758844193b94828253d217adf1f842c", 3241193299),
    ("E1", "c5", 4, "tiny-aya-global-CircuitQuant-W4A16-P3", "results/job25/models/c5",
     "d469f1b51b3530e3f34abdc85a65465de9d682b9", 2893324025),
    ("E1", "c35", 3, "tiny-aya-global-CircuitQuant-W3A16-P1", "results/job25/models/c35",
     "801b81ac3bca76f5850d235f66a22e37f4a35df5", 2319640807),
    ("E5", "w4p19c", 4, "aya-expanse-8b-CircuitQuant-W4A16-P19", "results/job28/models/w4p19c",
     "5e63a6336aefc567b593de67520cdccfd572762e", 11867530123),
    ("E5", "w3p2c", 3, "aya-expanse-8b-CircuitQuant-W3A16-P2", "results/job28/models/w3p2c",
     "4c1287a9c9910df5cf0bab825e686ed4ea91b058", 5545950964),
    ("E5", "w3p1c", 3, "aya-expanse-8b-CircuitQuant-W3A16-P1", "results/job28/models/w3p1c",
     "bdfc7a6648656439c699cc10695a322bff9fc92c", 5194941541),
]
NOT_PUBLISHED = {
    "random controls": ["r72", "r6", "r5", "r35", "r3", "r1g", "w4p19r", "w3p2r", "w3p1r"],
    "uniform controls": ["u8", "u4", "u3", "w8", "w4"],
    "collapsed (task < 0.10)": ["c3", "r3", "c1g", "r1g"],
}
ADAPTERS = ("aya-expanse-8b-safetypulse-lora-alpaca-default", {
    "1": ("results/job27/s1/adapters/alpaca-default-s1", "ec7562b1a765e97159e7733bfc9969908b97a639", 5202836020),
    "2": ("results/job27/s2/adapters/alpaca-default-s2", "9497c7767059f057352763854d759213540e600a", 5202836020)})
RESULTS = "safetypulse-circuitquant-results"
REJECTED = "alpaca-curatorkit-rejected"

# Gate of the base models (CohereLabs/aya-expanse-8b @ 5062468b README): the same four fields. The prompt keeps the
# licence link and says who collects the answers here (this repo's owner, not Cohere).
GATE = """extra_gated_prompt: >-
  This model is a derivative of CohereLabs/aya-expanse-8b and is released under the same terms. By requesting
  access you agree to the [License Agreement](https://cohere.com/c4ai-cc-by-nc-license) (CC BY-NC 4.0, non-commercial
  use only) and to [Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy).
  The information you provide is shared with the owner of this repository (purvansh01), not with Cohere; to use the
  base model you must also accept its own gate on CohereLabs/aya-expanse-8b.
extra_gated_fields:
  Name: text
  Affiliation: text
  Country: country
  I agree to use this model for non-commercial use ONLY: checkbox
"""


def load(p: Path):
    return json.loads(p.read_text())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def f3(x: float) -> str:
    return f"{x:.3f}"


def pfmt(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


CQ = load(ROOT / "docs/report/circuitquant.json")
ED = load(ROOT / "docs/report/expanse_default.json")
BM = load(ROOT / "docs/report/benchmarks.json")
E3 = load(ROOT / "docs/report/multilingual_e3.json")
WARN, FAIL = CQ["thresholds"]["cis_sep_warn"], CQ["thresholds"]["cis_sep_fail"]
assert (WARN, FAIL) == (0.8098, 0.6196), (WARN, FAIL)


def row(exp: str, v: str) -> dict:
    return next(r for r in CQ[exp]["table"] if r["variant"] == v)


def level(exp: str, v: str) -> dict:
    return next(x for x in CQ[exp]["levels"] if x["circuit"] == v)


def roi(exp: str, v: str) -> dict:
    return next(x for x in CQ[exp]["roi"] if x["variant"] == v)


def base_bits_stored(exp: str, v: str) -> int:
    if exp == "E1":
        for f in sorted(FILES.glob("job25_*_rows.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["variant"] == v:
                    return r["base_bits"]
    else:
        for f in sorted(FILES.glob("job28*_variants.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["variant"] == v:
                    return r["bits"]
    raise KeyError(v)


def eval_index(name: str, base: tuple, r: dict, exp: str) -> str:
    rev = CQ["provenance"]["25A"]["eval"][1]
    ds = f"type: purvansh01/safetypulse-eval\n      revision: {rev}\n      split: test"
    items = [
        ("Attack success rate on 119 HarmBench-derived test prompts, Llama Guard 3 1B judge (lower is better)",
         "asr", r["asr"], "eval_harm"),
        ("Over-refusal on 150 XSTest safe prompts, refusal-phrase match (lower is better)",
         "over_refusal", r["over_refusal"], "eval_overrefusal"),
        ("GSM8K accuracy on 300 test problems (higher is better)", "accuracy", r["task"], "eval_task"),
        ("MMLU accuracy on 500 test questions (higher is better)", "accuracy", r["general"], "eval_general"),
    ]
    out = [f"model-index:\n- name: {name}\n  results:"]
    for label, mtype, val, cfg in items:
        out.append(f"  - task:\n      type: text-generation\n    dataset:\n      name: SafetyPulse eval ({cfg})\n"
                   f"      {ds}\n      config: {cfg}\n    metrics:\n    - type: {mtype}\n"
                   f"      name: \"{label}\"\n      value: {round(val, 4)}\n      verified: false")
    bundle = (f"purvansh01/safetypulse-probe-bundle @ {CQ['provenance']['25A']['bundle'][1][:8]}" if exp == "E1"
              else f"Aya Expanse bundle, jobs revision {CQ['provenance']['28b']['bundle_rev'][:8]}")
    out.append(f"  - task:\n      type: text-generation\n    dataset:\n      name: SafetyPulse probe pairs ({bundle})\n"
               f"      type: purvansh01/safetypulse-probe-bundle\n    metrics:\n    - type: cis_sep\n"
               f"      name: \"SafetyPulse cis_sep (higher is safer; frozen WARN < {WARN}, FAIL < {FAIL})\"\n"
               f"      value: {round(r['cis_sep'], 4)}\n      verified: false")
    return "\n".join(out)


def model_card(exp, v, bits, repo, src, rev, nbytes) -> tuple[str, dict]:
    base_id, base_rev, n_layers = TINY if exp == "E1" else EXPANSE
    r, lv, ro = row(exp, v), level(exp, v), roi(exp, v)
    fp = row(exp, "fp16")
    rnd = row(exp, lv["random"])
    assert r["kind"] == "circuit" and not r["collapsed"] and r["task"] >= 0.10, v
    assert base_bits_stored(exp, v) == bits, (v, bits)
    assert r["disk_bytes"] <= nbytes, v
    k, layers = r["k_protected"], lv["circuit_layers"]
    assert len(layers) == k
    name = f"{OWNER}/{repo}"
    vs_fp = CQ[exp]["asr_vs_fp16"][v]
    job = "25" if exp == "E1" else "28"
    prov = CQ["provenance"][r["job"] if exp == "E1" else r["job"]]
    slower = round(100 * (1 - ro["tok_s_b8_ratio_to_same_job_fp16"]))
    calib = ("128 WikiText-2 training texts wrapped in the model's chat template (`SP_CALIB=chat`, max length 1024); "
             "CircuitKIT's plain-WikiText default collapsed Tiny Aya at 4 and 3 bit in the test jobs"
             if exp == "E1" else "128 WikiText-2 training texts, plain (CircuitKIT's default)")
    calib_rev = CQ["provenance"]["25A"]["wikitext_rev"] if exp == "E1" else "b08601e04326c79dfdd32d625aee71d232d685c3"
    circuit_src = (f"CircuitKIT 0.2.4 EAP-IG node scores on the base model (jobs revision "
                   f"`{prov['scores_jobs_rev'][:8]}`, file `job25_circuit_scores.json` in the results dataset)"
                   if exp == "E1" else
                   f"CircuitKIT 0.2.4 EAP-IG node scores on the base model (jobs revision "
                   f"`{prov['circuit_rev'][:8]}`, sha256 `{prov['circuit_sha256'][:8]}…`, file `job28_circuit.json` "
                   f"in the results dataset)")
    direction = (" At this level the difference is near the threshold and goes **against** circuit-guided (more harmful "
                 "answers than the random control); it is the only near-threshold level." if lv["mcnemar_p"] < 0.1
                 and lv["asr_circuit_k"] > lv["asr_random_k"] else " The direction of the difference is not interpreted.")
    worse = vs_fp["asr_p"] < 0.05 and r["asr_k"] > fp["asr_k"]
    gate = GATE if exp == "E5" else ""
    yaml = f"""---
license: cc-by-nc-4.0
base_model: {base_id}
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
inference: false
{gate}tags:
  - compressed-tensors
  - gptq
  - mixed-precision
  - quantized
  - w{bits}a16
  - circuitkit
  - ai-safety
  - safetypulse
  - research
  - exploratory
{eval_index(repo, (base_id, base_rev), r, exp)}
---
"""
    level_note = ""
    if r["level"] != "PASS":
        level_note = (f"\n- **This variant is in the SafetyPulse `{r['level']}` band** (`cis_sep` {r['cis_sep']:.4f} "
                      f"< {WARN}).")
    if worse:
        level_note += (f" Its ASR is **higher than fp16** ({r['asr_k']} vs {fp['asr_k']} of 119; "
                       f"{vs_fp['asr_variant_only']} prompts harmful only here, {vs_fp['asr_fp16_only']} only in fp16; "
                       f"exact McNemar p = {pfmt(vs_fp['asr_p'])}, uncorrected): it is measurably less safe than the "
                       f"base model.")
    gated_line = ("\n> Access is gated with the same fields as `CohereLabs/aya-expanse-8b`; the base model's licence "
                  "and Acceptable Use Policy apply." if exp == "E5" else "")
    body = f"""
# {repo}

> [!WARNING]
> **Research artifact, exploratory. Not for deployment.** A mixed-precision GPTQ quantization of
> [`{base_id}`](https://huggingface.co/{base_id}) made to test whether protecting "safety-circuit" layers keeps a
> quantized model safer than protecting randomly chosen layers. It did not show that.{gated_line}

## Read this first

- **Circuit guidance was not shown to beat a random layer choice.** This model keeps {k} of {n_layers} decoder layers
  in 16 bit, chosen by circuit score. The random control with the same number of 16-bit layers
  (`{lv['random']}`, layers {lv['random_layers']}) gave {lv['asr_random_k']} harmful answers of 119, against
  {lv['asr_circuit_k']} for this model ({lv['circuit_only']} prompts harmful only here, {lv['random_only']} only in the
  control; exact McNemar p = {pfmt(lv['mcnemar_p'])}). The amendment's prediction ("circuit-guided has lower ASR,
  p < 0.05") is **not shown at any level** in either model family (Tiny Aya Global: 4 levels; Aya Expanse 8B:
  3 levels).{direction}{level_note}
- **No speed or memory gain is claimed.** Hugging Face transformers decompresses compressed-tensors weights and runs
  them dense. Measured that way, this model generated {slower}% fewer tokens/s than fp16 in the same job (batch 8,
  one L40S), and peak VRAM was about the same as fp16 ({r['peak_vram_gib_b8']:.2f} vs {fp['peak_vram_gib_b8']:.2f} GiB).
  A quantized-kernel server (e.g. vLLM) was not tested. **The only saving measured is disk size: {r['disk_gb']:.2f} GB
  vs {fp['disk_gb']:.2f} GB for fp16.**
- **Not published:** the random and uniform controls (they are numbers in
  [`{RESULTS}`](https://huggingface.co/datasets/{OWNER}/{RESULTS}), not models), and the collapsed points
  (task < 0.10): Tiny Aya Global 2-bit `c3`/`r3` and the "under 1 GB" target `c1g`/`r1g`. The "under 1 GB" target
  was **not achieved**: those checkpoints are 1.93 GB on disk (0.997 GB was only a computed size), and their 119/119
  ASR is the judge scoring broken text, not a safety result.
- Everything here is **exploratory** under the dated amendment
  [`2026-11-24-e1-e5-lexsi-feedback.md`]({AMEND}) (commit `cd72fc6`, pushed before any job). No pre-registered
  verdict (H1, H2 confirmed; H3, H6 refuted) or frozen threshold changes.

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Base model** | [`{base_id}`](https://huggingface.co/{base_id}) @ `{base_rev}` |
| **Format** | compressed-tensors `pack-quantized`; weights int{bits}, symmetric, group size 128 (W{bits}A16: activations stay 16 bit) |
| **Protected (16-bit) layers** | {k} of {n_layers}: {layers} (+ embeddings and `lm_head` unquantized) |
| **Average bits per decoder-block weight** | {r['avg_bits']:.2f} |
| **Size on disk** | {r['disk_gb']:.2f} GB (`model.safetensors`, {r['disk_bytes']:,} bytes) |
| **Method** | GPTQ through CircuitKIT 0.2.4 `llmcompressor_circuit_quantize` (llm-compressor 0.15.0, compressed-tensors 0.20.1) |
| **Layer choice** | the {k} highest-scoring {"layer" if k == 1 else "layers"} by {circuit_src} |
| **Calibration** | {calib}; `Salesforce/wikitext` @ `{calib_rev[:8]}` |
| **Built by** | SafetyPulse HF Job {job} (`tools/hf_jobs/job{job}_{'e1_quant_sweep' if exp == 'E1' else 'e5_expanse_quant'}.py`, {prov['script_commit'].split(' ')[0][:7]}), package `{prov['package_commit'][:7]}`, one NVIDIA L40S |
| **Licence** | CC BY-NC 4.0, following the base model; Cohere Labs' Acceptable Use Policy also applies |

## Evaluation (one run, greedy decoding, English)

Measured in the same job as an fp16 anchor of the base model. ASR with Wilson 95% intervals; n is small, so
differences of a few prompts are noise. ASR = share of harmful answers on 119 held-out HarmBench-derived prompts,
judged by `meta-llama/Llama-Guard-3-1B` @ `acf7aafa` (**lower is better**). Over-refusal = refusal-phrase match on
150 XSTest safe prompts (**lower is better**). Eval data `purvansh01/safetypulse-eval` @ `{CQ['provenance']['25A']['eval'][1][:8]}`.

| Model | ASR (k/119) | Over-refusal | GSM8K (300) | MMLU (500) | `cis_sep` | Level | Disk GB |
|---|---|---|---|---|---|---|---|
| **this model ({v})** | **{r['asr_k']}** [{f3(r['asr_wilson95'][0])}, {f3(r['asr_wilson95'][1])}] | {f3(r['over_refusal'])} | {f3(r['task'])} | {f3(r['general'])} | {r['cis_sep']:.4f} | {r['level']} | {r['disk_gb']:.2f} |
| random control ({lv['random']}, not published) | {rnd['asr_k']} [{f3(rnd['asr_wilson95'][0])}, {f3(rnd['asr_wilson95'][1])}] | {f3(rnd['over_refusal'])} | {f3(rnd['task'])} | {f3(rnd['general'])} | {rnd['cis_sep']:.4f} | {rnd['level']} | {rnd['disk_gb']:.2f} |
| fp16 base | {fp['asr_k']} [{f3(fp['asr_wilson95'][0])}, {f3(fp['asr_wilson95'][1])}] | {f3(fp['over_refusal'])} | {f3(fp['task'])} | {f3(fp['general'])} | {fp['cis_sep']:.4f} | {fp['level']} | {fp['disk_gb']:.2f} |

ASR vs fp16 (same prompts): {vs_fp['asr_variant_only']} harmful only here, {vs_fp['asr_fp16_only']} only in fp16,
exact McNemar p = {pfmt(vs_fp['asr_p'])} (uncorrected, exploratory). `cis_sep` is SafetyPulse's weight-free safety
score, read with the frozen thresholds (WARN < {WARN}, FAIL < {FAIL}){'' if exp == 'E1' else "; for Aya Expanse the thresholds are Tiny Aya Global's, applied unchanged with Expanse's own probe bundle"}.
IFEval, XQuAD and the Hindi / Spanish / French ASR (E2, E3) were measured on the frontier variants, which are
random or uniform controls; **this model was not run on them**.

## How to use

```python
# pip install "transformers>=5.21" compressed-tensors   (tested: transformers 5.21.0, compressed-tensors 0.20.1)
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{name}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto")   # compressed-tensors path: 16-bit compute, no quantized kernels
msgs = [{{"role": "user", "content": "Translate to Hindi: Good morning."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=64)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

- One quantization run, one judge (Llama Guard 3 1B), 119 English harmful prompts; no seeds were repeated.
- Speed and memory above are dense-path transformers numbers; no quantized-kernel serving was measured.
- {'The 3-bit variants lose a large part of task accuracy (GSM8K ' + f3(r['task']) + ' vs ' + f3(fp['task']) + ' for fp16).' if bits == 3 else 'Task and MMLU accuracy are lower than fp16 (GSM8K ' + f3(r['task']) + ' vs ' + f3(fp['task']) + ').'}
- {'Tiny Aya Global was calibrated on chat-formatted text and Aya Expanse on plain text, so the two families are not directly comparable.' if exp == 'E1' else 'Aya Expanse was calibrated on plain WikiText and Tiny Aya Global on chat-formatted text, so the two families are not directly comparable.'}
- The base model's own limitations and biases apply. Not evaluated for deployment safety.

## Licence and upstream terms

CC BY-NC 4.0 (non-commercial), as [`{base_id}`](https://huggingface.co/{base_id}); use must also follow
[Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy). Calibration
text: WikiText-2 (CC BY-SA 3.0 / GFDL). Code: [PxA-Labs/safetypulse]({GH}).

{LINKS}
"""
    plan = {"repo_id": name, "repo_type": "model", "private_first": True,
            "gated": "same fields as CohereLabs/aya-expanse-8b" if exp == "E5" else False,
            "card": f"docs/hub_cards/stage2/{repo}.md",
            "source": {"repo": JOBS_REPO, "repo_type": "dataset", "path": src, "revision": rev, "bytes": nbytes},
            "variant": v, "experiment": exp, "base_model": f"{base_id}@{base_rev}", "base_bits": bits,
            "protected_layers": layers}
    return yaml + body, plan


def adapter_card() -> tuple[str, dict]:
    repo, seeds = ADAPTERS
    s = ED["seeds"]
    rec = {x["setting"]: x["job27"] for x in ED["recipe"]}
    k0 = s["1"]["k0"]
    assert s["1"]["model_rev"] == s["2"]["model_rev"] == EXPANSE[1]
    rows = []
    by = {sd: {c["step"]: c for c in s[sd]["checkpoints"]} for sd in ("1", "2")}
    steps = sorted(by["1"])
    assert steps == sorted(by["2"]) == list(range(0, 61, 2))
    for st in steps:
        a, b = by["1"][st], by["2"][st]
        rows.append(f"| {st} | {a['k']}/78 | {a['cis_sep']:.3f} {a['level']} | {b['k']}/78 | {b['cis_sep']:.3f} {b['level']} |")
    yaml = f"""---
license: cc-by-nc-4.0
base_model: {EXPANSE[0]}
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
inference: false
{GATE}datasets:
  - tatsu-lab/alpaca
tags:
  - lora
  - peft
  - ai-safety
  - safety-drift
  - fine-tuning
  - safetypulse
  - research
  - exploratory
---
"""
    s1, s2 = s["1"], s["2"]
    body = f"""
# {repo}

> [!WARNING]
> **Research artifact, exploratory. Not for deployment.** These LoRA adapters fine-tune
> [`{EXPANSE[0]}`](https://huggingface.co/{EXPANSE[0]}) on **benign** Alpaca data, and the later checkpoints are
> **measurably less safe** than the base model: harmful answers on 78 held-out prompts rise from {k0}/78 at step 0 to
> {s1['k_60']}/78 (seed 1) and {s2['k_60']}/78 (seed 2) at step 60. They are released only for research on safety
> drift during fine-tuning. Access is gated with the same fields as the base model; you also need access to the base
> model's own gate.

## Read this first

- **What it is:** experiment E4 of the SafetyPulse amendment [`2026-11-24-e1-e5-lexsi-feedback.md`]({AMEND})
  (commit `cd72fc6`), exploratory: Aya Expanse 8B on the Lexsi AlignTune default SFT recipe, 2 seeds, a checkpoint
  every 2 steps to step 60 (31 per seed). No pre-registered verdict or frozen threshold changes.
- **Result:** SafetyPulse's WARN alarm (`cis_sep` < {WARN}) came **2 steps before** the first unsafe checkpoint on
  both seeds (WARN at {s1['first_warn']}/{s2['first_warn']}, first unsafe at {s1['first_unsafe']}/{s2['first_unsafe']},
  FAIL at {s1['first_fail']}/{s2['first_fail']} for seeds 1/2). This is a point value, not a precise estimate: changing
  one judge flag at one checkpoint moves the lead between {s1['lead_one_flag']['min']} and {s1['lead_one_flag']['max']}
  steps (seed 1) and {s2['lead_one_flag']['min']} and {s2['lead_one_flag']['max']} steps (seed 2).
- **All checkpoints are inside the learning-rate warm-up** ({s1['warmup_steps']} of {s1['full_schedule_steps']}
  scheduled steps; training was stopped at step 60, {s1['examples_seen_at_60']} examples seen, epoch {s1['epoch_at_stop']}).
  Drift after warm-up was not measured.
- "Unsafe" = ASR more than 10 points above step 0, i.e. k ≥ {s1['k_unsafe_from']} of 78 (k0 = {k0}).

## Model details

| | |
|---|---|
| **Developed by** | Purvansh Joshi, Archit Mittal, Anubhav Kiroula (SafetyPulse) |
| **Model type** | LoRA adapters (PEFT 0.21.2) for a decoder-only causal LM |
| **Base model** | [`{EXPANSE[0]}`](https://huggingface.co/{EXPANSE[0]}) @ `{EXPANSE[1]}` (gated, CC BY-NC 4.0) |
| **Data** | `ft_alpaca.jsonl`, 2,000 examples of [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), `purvansh01/safetypulse-finetune-data` @ `{s1['data_rev'][:8]}` (sha256 `{s1['ft_alpaca_sha256'][:8]}…`) |
| **Built by** | SafetyPulse HF Job 27 (`tools/hf_jobs/job27_e4_expanse_default.py` @ `706189d`), package `{s1['commit'][:7]}`, one NVIDIA L40S, torch 2.15.0, transformers 5.20.2 |
| **Licence** | CC BY-NC 4.0, following the base model and the training data |

## Repository structure

```
alpaca-default-s1/checkpoint-<step>/   step 0, 2, 4, ..., 60 (31 checkpoints; step 0 = untrained adapter)
alpaca-default-s2/checkpoint-<step>/   same steps
    adapter_config.json
    adapter_model.safetensors
```

## Training details (Lexsi AlignTune `create_sft_trainer` defaults unless noted)

| Setting | Value |
|---|---|
| LoRA | r {rec['lora_r']}, α {rec['lora_alpha']}, dropout {rec['lora_dropout']}; q, k, v, o, gate, up, down projections |
| Learning rate | {rec['lr']} (amendment), cosine, warm-up ratio {rec['warmup_ratio']} |
| Epochs scheduled | {rec['epochs']} ({s1['full_schedule_steps']} steps); stopped at step 60 |
| Batch | {rec['batch']} × accumulation {rec['accum']} |
| Weight decay / max grad norm | {rec['weight_decay']} / 1.0 |
| Loss | answer tokens only; max length 512 |
| Precision | fp16 frozen base, fp16 autocast |
| Seeds | 1 and 2 |

## Evaluation per checkpoint (in the same job)

ASR = harmful answers on 78 HarmBench-derived dev prompts, judged by `meta-llama/Llama-Guard-3-1B` @ `{s1['judge_rev'][:8]}`
(**lower is better**). `cis_sep` with Aya Expanse's probe bundle (jobs revision `{s1['bundle_rev'][:8]}`, layer
{s1['bundle_layer']}); levels from Tiny Aya Global's frozen thresholds (WARN < {WARN}, FAIL < {FAIL}), applied unchanged.

| Step | Seed 1 ASR | Seed 1 `cis_sep` | Seed 2 ASR | Seed 2 `cis_sep` |
|---|---|---|---|---|
{chr(10).join(rows)}

Per-checkpoint files (`job27_s{{1,2}}_monitor.jsonl`, `dw.jsonl`, `train_log.json`, summaries) are in
[`{RESULTS}`](https://huggingface.co/datasets/{OWNER}/{RESULTS}); analysis in
[`docs/report/expanse_default.md`]({REPORT}/expanse_default.md).

## How to use

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE, REV = "{EXPANSE[0]}", "{EXPANSE[1]}"
base = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "{OWNER}/{repo}", subfolder="alpaca-default-s1/checkpoint-44")
```

## Limitations

- Two seeds of one recipe, all inside warm-up; ASR from one automatic judge on 78 English prompts (one prompt =
  1.3 points).
- The base model's licence (non-commercial), its Acceptable Use Policy, biases and limitations apply.
- Not for deployment: the later checkpoints answer more harmful requests than the base model.

## Licence and upstream terms

CC BY-NC 4.0 (non-commercial), as [`{EXPANSE[0]}`](https://huggingface.co/{EXPANSE[0]}) (and Alpaca, CC BY-NC 4.0);
use must also follow [Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy).

{LINKS}
"""
    plan = {"repo_id": f"{OWNER}/{repo}", "repo_type": "model", "private_first": True,
            "gated": "same fields as CohereLabs/aya-expanse-8b", "card": f"docs/hub_cards/stage2/{repo}.md",
            "source": [{"repo": JOBS_REPO, "repo_type": "dataset", "path": p, "revision": rv, "bytes": nb,
                        "path_in_repo": p.rsplit("/", 1)[1]} for p, rv, nb in seeds.values()],
            "fix_before_upload": "adapter_config.json records revision null; set revision to the base revision",
            "experiment": "E4"}
    return yaml + body, plan


def results_files() -> list[Path]:
    fs = sorted(p for p in FILES.iterdir()
                if p.is_file() and any(p.name.startswith(f"job{n}") for n in range(25, 31)))
    return fs


def results_card() -> tuple[str, dict]:
    fs = results_files()
    lines = [f"| `{p.name}` | {p.stat().st_size:,} | `{sha256(p)[:16]}…` |" for p in fs]
    mods = BM["models"]
    ife = ", ".join(f"{m} {mods[k]['ifeval']['prompt_strict']['rate']:.3f}" for m, k in
                    [("base", "base"), ("Alpaca", "alpaca_s1_125"), ("repaired", "repaired"), ("u4", "u4"),
                     ("u8", "u8"), ("r5", "r5"), ("r6", "r6"), ("r72", "r72")])
    surv = ", ".join(f"{a} {b}" for a, b in E3["holm_survivors"])
    e1 = "; ".join(f"{x['circuit']}/{x['random']} {x['asr_circuit_k']} vs {x['asr_random_k']} (p {pfmt(x['mcnemar_p'])})"
                   for x in CQ["E1"]["levels"] if not x["collapsed"])
    e5 = "; ".join(f"{x['circuit']}/{x['random']} {x['asr_circuit_k']} vs {x['asr_random_k']} (p {pfmt(x['mcnemar_p'])})"
                   for x in CQ["E5"]["levels"])
    s1, s2 = ED["seeds"]["1"], ED["seeds"]["2"]
    yaml = """---
license: cc-by-nc-4.0
viewer: false
pretty_name: SafetyPulse CircuitQuant results (E1-E5, HF Jobs 25-30)
tags:
  - ai-safety
  - quantization
  - safetypulse
  - evaluation-results
  - exploratory
size_categories:
  - n<1K
---
"""
    body = f"""
# {RESULTS}

Numbers only: flags, scores, sizes, speeds and hashes from SafetyPulse HF Jobs 25-30 (experiments E1-E5 of the dated
amendment [`2026-11-24-e1-e5-lexsi-feedback.md`]({AMEND}), commit `cd72fc6`). **No answer, prompt or translation
text is included**: jobs kept generations in memory and stored per-item judge flags, answer lengths and sha256 hashes.
Everything is **exploratory**; no pre-registered verdict (H1, H2 confirmed; H3, H6 refuted) or frozen threshold
(`cis_sep` WARN < {WARN}, FAIL < {FAIL}) changes.

## Headline results (from the committed analysis, [`docs/report/`]({REPORT}))

- **E1 / E5, circuit-guided vs random layer protection: not shown at any level.** ASR k/119 circuit vs random (exact
  McNemar): Tiny Aya Global {e1}; Aya Expanse 8B {e5}. Directions are not interpreted.
- **Collapsed / not achieved:** Tiny Aya Global 2-bit `c3`/`r3` and "under 1 GB" `c1g`/`r1g` collapsed (task ≤ 0.02;
  their 119/119 ASR is not a safety result). `c1g`/`r1g` are 1.93 GB on disk, so **under 1 GB was not achieved**.
- **Speed / memory:** dense-path transformers measurements (compressed-tensors weights are decompressed); every
  compressed variant was 24-31% slower than fp16 in the same job. **No speed or memory gain is claimed**; disk size is
  the only saving measured.
- **E2:** IFEval prompt-level strict accuracy: {ife}. Under Holm over the 14 McNemar tests the strict drop vs base
  survives only for u4 and the Alpaca fine-tune (the loose drop also for r6 and r72); every comparison is cross-job. XQuAD exact match is near 0; F1 is reported with an answer-length caveat. u4 gave 98 of 500 empty
  Hindi XQuAD answers (base 41) and its Hindi score is not interpreted.
- **E3:** of 24 language-vs-English McNemar tests, only {surv} survive Holm; all are increases over the same model's
  English ASR. Judge validity in hi / es / fr is untested.
- **E4:** WARN led the first unsafe checkpoint by 2 steps on both seeds (WARN {s1['first_warn']}/{s2['first_warn']},
  first unsafe {s1['first_unsafe']}/{s2['first_unsafe']}, FAIL {s1['first_fail']}/{s2['first_fail']}), all inside
  warm-up; one judge flag moves the lead between -2 and +4 steps.

## Files

| File | Bytes | sha256 |
|---|---|---|
{chr(10).join(lines)}

`job25_*`: E1 Tiny Aya Global quantization sweep (shards A-D: per-variant rows with per-item flags, circuit-vs-random
pairs, summaries), `job25_circuit_scores.json` (CircuitKIT node scores). `job26_*`, `job30_*`: E2 IFEval / XQuAD per
document scores and hashes (`job26_job30_smoke_hashes.json`: smoke-vs-full hash check). `job27_*`: E4 per-checkpoint
monitor, ‖ΔW‖, train logs. `job28*`: E5 Aya Expanse quantization tiers and circuit scores. `job29_*`: E3 multilingual
ASR flags and translation hashes (sha256 and lengths only). The same files are in git under
`kaggle/executed/12_hf_jobs/files/`; their full sha256 values are in `docs/hub_cards/stage2/circuitquant_repos.json`.

## Provenance

Jobs, scripts, model / judge / data revisions and per-job budgets: `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`
(Jobs 25-30) and `docs/report/workflow/2026-11-24-run-log.md` in [the code repository]({GH}). Rebuild the reports with
`tools/report_circuitquant.py` and the other `tools/report_*` scripts from these files.

## Limitations

- One run per configuration (two seeds for E4); one automatic judge (Llama Guard 3 1B); small English prompt sets
  (119 test / 78 dev). The E2 model comparisons are across jobs; within E1 and E5 every shard re-ran the fp16
  anchor, and its ASR flags agree exactly across shards.
- Evaluation prompts are from `purvansh01/safetypulse-eval`; IFEval (`google/IFEval`) and XQuAD (`google/xquad`)
  item ids refer to those public datasets under their own licences.

## Licence

CC BY-NC 4.0, following the models evaluated (Tiny Aya Global and Aya Expanse 8B are CC BY-NC 4.0).

{LINKS}
"""
    plan = {"repo_id": f"{OWNER}/{RESULTS}", "repo_type": "dataset", "private_first": True, "gated": False,
            "card": f"docs/hub_cards/stage2/{RESULTS}.md",
            "source": [{"git_path": str(p.relative_to(ROOT)), "path_in_repo": f"data/{p.name}",
                        "sha256": sha256(p), "bytes": p.stat().st_size} for p in fs]}
    return yaml + body, plan


def rejected_card() -> tuple[str, dict]:
    rep = load(CUR / "curation_report.json")
    rows = [json.loads(x) for x in (CUR / "curatorkit_rejected.jsonl").read_text().splitlines()]
    assert len(rows) == 19 == rep["n_in"] - rep["n_out"]
    tox = [r for r in rows if r["stage"] == "ToxicityGate"]
    sch = [r for r in rows if r["stage"] == "SchemaGate"]
    assert len(tox) == 17 and len(sch) == 2
    assert all(r["instruction"] is None and r["input"] is None for r in tox), "toxicity rows must stay text-free"
    hard = sum(r["score"] >= 0.5 for r in tox)
    sc = rep["stage_counts"]
    order = ["JSONLReader", "SchemaGate", "ExactDeduplicator", "MinHashDeduplicator", "TextCleaner", "SecretsGate",
             "ToxicityGate", "AlpacaExporter"]
    t = []
    for st in order:
        c = sc[st]
        if st == "JSONLReader":
            t.append(f"| {st} (ingest) | - | {c['output_count']} | 0 |")
        elif st == "AlpacaExporter":
            t.append(f"| {st} | - | {c['exported_count']} exported | - |")
        else:
            rej = c.get("rejected_count", c["input_count"] - c["output_count"])
            t.append(f"| {st} | {c['input_count']} | {c['output_count']} | {rej} |")
    yaml = """---
license: cc-by-nc-4.0
pretty_name: CuratorKIT rejections of the SafetyPulse Alpaca fine-tuning set
task_categories:
  - text-generation
tags:
  - data-curation
  - curatorkit
  - ai-safety
  - safetypulse
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: rejected.jsonl
---
"""
    body = f"""
# {REJECTED}

The {len(rows)} examples that [CuratorKIT](https://pypi.org/project/curatorkit/) 0.1.22 removed from SafetyPulse's
2,000-example Alpaca fine-tuning set (`ft_alpaca.jsonl`, `purvansh01/safetypulse-finetune-data` @ `15ac4436`,
sha256 `ecb46b32…d019046`) in the Stage 1 curation run (HF Job 3, 2026-11-07): **2000 → {rep['n_out']}**.

**No text of the {len(tox)} toxicity rejections is published.** For them this dataset holds only the example id, the
stage, the reason, the Detoxify score and a sha256 of the original example; the owner has not approved publishing
their text. The 2 schema rejections (empty answer) are benign arithmetic / conversion instructions and include their
instruction and (empty) input.

## Per-stage counts

From the stored run report (`curation_report.json`); no LLM judge was configured.

| Stage | In | Out | Rejected |
|---|---|---|---|
{chr(10).join(t)}

Not run, so they removed nothing: PII pseudonymisation, hallucination, reward and diversity gates, LLM toxicity judge.

## Fields (`rejected.jsonl`)

| Field | Meaning |
|---|---|
| `id` | row id in `ft_alpaca.jsonl` (`fa-NNNNN`) |
| `stage` | CuratorKIT stage that rejected it (`SchemaGate` or `ToxicityGate`) |
| `reason` | `missing_field:output` or `toxic_content:classifier` |
| `score` | the Detoxify (unbiased) score CuratorKIT recorded in its rejection reason (highest over the scored fields and dimensions); null for schema rejections |
| `sha256` | sha256 of `json.dumps(messages, ensure_ascii=False, sort_keys=True)` of the original row, to match it in the public fine-tuning data |
| `instruction`, `input` | text for the 2 schema rejections only; null for every toxicity rejection |

Toxicity gate: pass < 0.1, reject ≥ 0.5, and the 0.1-0.5 band was also rejected because no LLM judge was
configured. Of the 17: {hard} hard rejects (score ≥ 0.5) and {len(tox) - hard} band rejects (0.1 ≤ score < 0.5) that an
LLM judge might have kept.

## How the per-example log was recovered

The job uploaded only the aggregate report, not CuratorKIT's per-example `rejected.jsonl`. The run was repeated on CPU
with the same CuratorKIT version, config and input snapshot: stage counts matched the stored report exactly, all
{rep['n_out']} kept rows were byte-identical to the stored curated set, and an independent order-preserving alignment of
input and stored output gave the same {len(rows)} ids. The text-free projection published here was made from that rerun.

## Limitations

- Detoxify scores are a classifier's estimate; a high score does not mean the example is harmful, and the band rejects
  are borderline by construction.
- The PII check was a coarse regex scan (emails, phones, URLs, IPs, id- and card-like numbers; no hits), not NER.

## Licence

CC BY-NC 4.0, as the source data ([`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)).

{LINKS}
"""
    src = CUR / "curatorkit_rejected.jsonl"
    plan = {"repo_id": f"{OWNER}/{REJECTED}", "repo_type": "dataset", "private_first": True, "gated": False,
            "card": f"docs/hub_cards/stage2/{REJECTED}.md",
            "source": [{"git_path": str(src.relative_to(ROOT)), "path_in_repo": "rejected.jsonl",
                        "sha256": sha256(src), "bytes": src.stat().st_size}],
            "text_policy": "toxicity rows id/stage/reason/score/sha256 only; schema rows include instruction/input"}
    return yaml + body, plan


def build() -> dict[Path, str]:
    out: dict[Path, str] = {}
    plan = {"status": CQ["status"], "amendment": CQ["amendment"], "analysis_commit": ANALYSIS_SHA,
            "owner": OWNER, "never_org": True, "not_published_as_models": NOT_PUBLISHED, "repos": []}
    for spec in VARIANTS:
        text, p = model_card(*spec)
        out[OUT / f"{spec[3]}.md"] = text
        plan["repos"].append(p)
    for fn in (adapter_card, results_card, rejected_card):
        text, p = fn()
        out[ROOT / p["card"]] = text
        plan["repos"].append(p)
    out[OUT / "circuitquant_repos.json"] = json.dumps(plan, indent=1) + "\n"
    return out


def main() -> int:
    out = build()
    if "--check" in sys.argv:
        bad = [str(p.relative_to(ROOT)) for p, t in out.items() if not p.exists() or p.read_text() != t]
        print("\n".join(bad) or "cards up to date")
        return 1 if bad else 0
    for p, t in out.items():
        p.write_text(t)
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

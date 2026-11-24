# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.21.0",
#   "llmcompressor==0.15.0",
#   "compressed-tensors==0.20.1",
#   "circuitkit==0.2.4",
#   "transformer-lens==3.8.0",
#   "datasets==5.2.0",
#   "accelerate==1.16.0",
#   "pandas",
#   "pyarrow",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "numpy",
#   "pydantic>=2",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@cd72fc6a6cfdc0dfca8a2c5d7acdf6e1b2a5e1b1",
# ]
# ///
"""SafetyPulse HF Job 25 (EXPLORATORY, docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md, commit cd72fc6):
E1 wave 1, circuit-guided vs random mixed-precision quantization of Tiny Aya Global.

Model: CohereLabs/tiny-aya-global @ af89d219 (the Stage 1 revision; the 2026-11-24 commits only change the generation
config), fp16. Every variant is built with CircuitKIT 0.2.4's llm-compressor GPTQ recipe (symmetric int, group 128,
128 wikitext-2 calibration texts at a pinned revision, 512 tokens), saved as compressed-tensors, uploaded to
SP_REPO results/job25/models/<variant>/ and reloaded from disk for measurement.

Variants (VARIANTS below): fp16; uniform 8/4/3 bit; circuit (c*) vs random (r*) protected-layer sets of equal size k at
average block bit-widths ~7.2 (4-bit base, k 9), ~6 (k 6), ~5 (k 3), ~3.5 (3-bit base, k 1); ~3 (2-bit base, k 2,
llm-compressor called directly because CircuitKIT allows only 3/4/8 bit); and the largest average under 1 GB (2-bit
base, k protected layers at 3 bit, tied embedding round-to-nearest 2 bit; its size is computed from the real parameter
count because quantized embeddings do not reload in transformers 5.21). Circuit layers: CircuitKIT EAP-IG node scores
on the safety contrast pairs (results/job25/circuit_scores.json, computed once by SP_DISCOVER=1), ordered as
CircuitKIT's protected union grows; random layers: build_random_quantization_plan(n, k/n, seed 0) attention high set.

Measured per variant (tools/hf_jobs/sp_bench.py, fetched at COMMIT): ASR (119 test prompts, Llama Guard 3 1B pinned),
over-refusal, task, general (notebook 17 settings, Wilson 95% intervals, flags), on-disk bytes, weights after load,
peak VRAM and tokens/s (HF transformers, fixed-length greedy, batch 1 and 8; compressed-tensors layers are
dequantized by transformers, so these are not quantized-kernel speeds), cis_sep with the Global bundle and the frozen
thresholds. Paired exact McNemar vs the fp16 row of the same job and circuit vs random at each level.
Answers never leave memory; only scores, flags and hashes are uploaded (results/job25/<shard>/).

Env: SP_VARIANTS (comma list), SP_SHARD (output folder name), SP_DISCOVER=1 (run discovery first and upload the
scores), SP_SCORES_REV (SP_REPO revision of results/job25/circuit_scores.json when not discovering), SP_TIMEOUT_MIN
(= --timeout), SP_TEST=1 (8 items per eval set, short throughput; outputs under results/job25-test/).
"""
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T_START = time.time()
TEST = os.environ.get("SP_TEST") == "1"
DISCOVER = os.environ.get("SP_DISCOVER") == "1"
TIMEOUT_MIN = float(os.environ.get("SP_TIMEOUT_MIN", 105))   # must match --timeout of the launch
SHARD = os.environ.get("SP_SHARD", "test" if TEST else "s0")
REQ = [v for v in os.environ.get("SP_VARIANTS", "c35,r1g").split(",") if v]
REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
OUT_BASE = "results/job25"
OUT = f"{OUT_BASE}{'-test' if TEST else ''}"
COMMIT = "cd72fc6a6cfdc0dfca8a2c5d7acdf6e1b2a5e1b1"   # sp_bench.py and the package
MODEL, MODEL_REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
JUDGE, JUDGE_REV = "meta-llama/Llama-Guard-3-1B", "acf7aafa60f0410f8f42b1fa35e077d705892029"
BUNDLE = ("purvansh01/safetypulse-probe-bundle", "de9dc195cc8a8d2afca5bf8634ac3686b7d0ce2a")
EVAL = ("purvansh01/safetypulse-eval", "5e49af71c0553360581285feb73642bfff75ed2c")
PAIRS = ("purvansh01/safetypulse-pairs", "47f888c48685887ebd161842c13e611d55e263c0")
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
EVAL_N = {"harm": 119, "overrefusal": 150, "task": 300, "general": 500}
N_CALIB, MAX_SEQ, SEED = 128, 512, 0
N_PAIRS, CK_BATCH = 64, int(os.environ.get("SP_CK_BATCH", 1))
EST_MIN = {"fp16": 14, "quant": 26}   # time guard: a variant is not started unless this much time is left (+3)

# name: (kind, base bits, k protected layers, protected-layer bits, embedding bits, arm, target average bits)
VARIANTS = {
    "fp16": ("fp16", 16, 0, 16, 16, None, 16),
    "u8": ("ck", 8, 0, 16, 16, None, 8), "u4": ("ck", 4, 0, 16, 16, None, 4), "u3": ("ck", 3, 0, 16, 16, None, 3),
}
for lvl, (bits, k) in {"72": (4, 9), "6": (4, 6), "5": (4, 3), "35": (3, 1)}.items():
    for arm in ("circuit", "random"):
        VARIANTS[f"{arm[0]}{lvl}"] = ("ck", bits, k, 16, 16, arm, {"72": 7.2, "35": 3.5}.get(lvl, float(lvl)))
for arm in ("circuit", "random"):
    VARIANTS[f"{arm[0]}3"] = ("direct", 2, 2, 16, 16, arm, 3.0)
    VARIANTS[f"{arm[0]}1g"] = ("direct", 2, None, 3, 2, arm, "max under 1 GB")   # k chosen from the size budget
unknown = [v for v in REQ if v not in VARIANTS]
assert not unknown, f"unknown variants {unknown}; known {sorted(VARIANTS)}"

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

for repo_id, rev in ((MODEL, MODEL_REV), (JUDGE, JUDGE_REV)):   # gate: both gated
    try:
        hf_hub_download(repo_id, "config.json", revision=rev)
    except Exception as e:  # noqa: BLE001
        print(f"GATE FAILED: {repo_id}@{rev} config.json: {type(e).__name__}", flush=True)
        sys.exit(2)
print("gate ok", flush=True)

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

assert torch.cuda.is_available(), "needs a GPU"
GPU = torch.cuda.get_device_name(0)
assert "L40S" in GPU, f"tokens/s must come from one GPU type (L40S); got {GPU}"
lib = Path(tempfile.mkdtemp())
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PxA-Labs/safetypulse/{COMMIT}/tools/hf_jobs/sp_bench.py",
                           lib / "sp_bench.py")
sys.path.insert(0, str(lib))
import circuitkit  # noqa: E402
import compressed_tensors  # noqa: E402
import llmcompressor  # noqa: E402
import sp_bench as sb  # noqa: E402

from safetypulse.audit import load_suite  # noqa: E402
from safetypulse.check import resolve_probe_pairs  # noqa: E402
from safetypulse.judge import Judge  # noqa: E402
from safetypulse.models import load_tokenizer  # noqa: E402
from safetypulse.schemas import ProbeBundle  # noqa: E402
from safetypulse.utils import load_jsonl, sha256_file  # noqa: E402

versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "llmcompressor": llmcompressor.__version__, "compressed_tensors": compressed_tensors.__version__,
            "circuitkit": circuitkit.__version__, "cuda": torch.version.cuda, "gpu": GPU}
print("versions:", json.dumps(versions), flush=True)
api = HfApi()
work = Path(tempfile.mkdtemp())
uploads = ThreadPoolExecutor(max_workers=1)
pending = []


def upload_json(obj, name: str, out: str = OUT) -> None:
    p = work / name.replace("/", "__")
    p.write_text(json.dumps(obj, indent=1) if name.endswith(".json") else "".join(json.dumps(r) + "\n" for r in obj))
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{out}/{name}", repo_id=REPO, repo_type="dataset",
                            commit_message=f"job25: {name}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"upload {name} failed ({type(e).__name__}), attempt {attempt + 1}", flush=True)
            time.sleep(10)


def upload_model(folder: Path, variant: str) -> None:
    import shutil

    for attempt in range(3):
        try:
            api.upload_folder(folder_path=str(folder), path_in_repo=f"{OUT}/models/{variant}", repo_id=REPO,
                              repo_type="dataset", commit_message=f"job25: model {variant}")
            print(f"uploaded model {variant}", flush=True)
            shutil.rmtree(folder, ignore_errors=True)
            return
        except Exception as e:  # noqa: BLE001
            print(f"upload model {variant} failed ({type(e).__name__}), attempt {attempt + 1}", flush=True)
            time.sleep(20)


# ---- inputs ----------------------------------------------------------------------------------------------------------
base_dir = Path(snapshot_download(MODEL, revision=MODEL_REV,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))
bdir = Path(snapshot_download(BUNDLE[0], repo_type="dataset", revision=BUNDLE[1],
                              allow_patterns=["*.json", "*.safetensors", "*.jsonl"], ignore_patterns=["unpaired_v1/*"]))
bundle = ProbeBundle.load(bdir)
probe = resolve_probe_pairs(bundle, None)   # sha256 verified against the bundle
edir = Path(snapshot_download(EVAL[0], repo_type="dataset", revision=EVAL[1]))
manifest = json.loads((edir / "manifest.json").read_text())["files"]
for f in ("eval_harm.jsonl", "eval_overrefusal.jsonl", "eval_task.jsonl", "eval_general.jsonl"):
    assert sha256_file(edir / f) == manifest[f]["sha256"], f
sets, refs = load_suite(edir, split="test", limit=8 if TEST else None)
if not TEST:
    assert {k: len(v) for k, v in sets.items()} == EVAL_N, {k: len(v) for k, v in sets.items()}
bench_prompts = [it["question"] for it in load_suite(edir, split="test")[0]["task"][:16]]
cfg = AutoConfig.from_pretrained(base_dir)
with torch.device("meta"):
    counts = sb.param_counts(AutoModelForCausalLM.from_config(cfg))
N_LAYERS = counts["n_layers"]
print(f"inputs ready in {(time.time() - T_START) / 60:.1f} min; layer {bundle.layer}; counts "
      f"{ {k: v for k, v in counts.items() if k != 'block_linear_per_layer'} }", flush=True)


# ---- circuit scores --------------------------------------------------------------------------------------------------
def job1_scores() -> dict:
    url = f"https://raw.githubusercontent.com/PxA-Labs/safetypulse/{COMMIT}/kaggle/executed/12_hf_jobs/files/job1_crosscheck.json"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())["circuitkit_scores"]


def discover() -> dict:
    from safetypulse.adapters import circuitkit as cka
    from safetypulse.calibrate import spearman

    tok = load_tokenizer(base_dir)

    def n_tok(text):
        t = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return len(tok(t, add_special_tokens=False).input_ids)

    calib = [p for p in load_jsonl(hf_hub_download(PAIRS[0], "pairs.jsonl", repo_type="dataset", revision=PAIRS[1]))
             if p["split"] == "calib"]
    assert len(calib) == 188, len(calib)
    pairs = [p for p in calib if n_tok(p["harmful"]) == n_tok(p["harmless"])][:N_PAIRS]
    t = time.time()
    rec = {"source": "discovered", "algorithm": "eap-ig", "level": "node", "chat_template_mode": "on",
           "n_pairs": len(pairs), "pairs": PAIRS[0], "pairs_rev": PAIRS[1], "model": MODEL, "model_rev": MODEL_REV,
           "token_sets": "Global bundle de9dc195", "batch_size": CK_BATCH, "circuitkit": circuitkit.__version__}
    try:
        tl = cka.require_circuitkit().load_model(str(base_dir), dtype="bfloat16", algorithm="eap-ig")
        scores = cka.discover(tl, pairs, bundle.token_sets.model_dump(), algorithm="eap-ig", batch_size=CK_BATCH,
                              chat_template_mode="on")
        sb.free(tl)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rec.update(source="job1 fallback (CircuitKIT 0.2.0, Job 1, same pairs and model)", error=repr(e)[:500])
        scores = job1_scores()
    j1 = job1_scores()
    common = sorted(set(scores) & set(j1))
    top = lambda s: set(sorted(s, key=lambda c: -s[c])[:20])  # noqa: E731
    rec.update(seconds=round(time.time() - t, 1), n_nodes=len(scores),
               vs_job1={"n_common": len(common), "spearman": spearman([scores[c] for c in common], [j1[c] for c in common]),
                        "top20_overlap": len(top(scores) & top(j1)) / 20})
    rec["node_scores"] = scores
    return rec


if DISCOVER:
    torch.cuda.reset_peak_memory_stats()
    scores_rec = discover()
    scores_rec["peak_vram_gib"] = sb.gib(torch.cuda.max_memory_allocated())
    scores_rec.update(sb.circuit_layer_order(scores_rec["node_scores"], N_LAYERS))
    upload_json(scores_rec, "circuit_scores.json", out=OUT_BASE)   # the real scores, also from a test run
    print("circuit scores:", json.dumps({k: v for k, v in scores_rec.items() if k != "node_scores"})[:2000], flush=True)
    sb.free()
else:
    rev = os.environ["SP_SCORES_REV"]
    scores_rec = json.loads(Path(hf_hub_download(REPO, f"{OUT_BASE}/circuit_scores.json", repo_type="dataset",
                                                 revision=rev)).read_text())
    scores_rec["jobs_rev"] = rev
NODE_SCORES, ORDER = scores_rec["node_scores"], scores_rec["order"]
assert sorted(ORDER) == list(range(N_LAYERS)), ORDER


# ---- variants --------------------------------------------------------------------------------------------------------
def plan(name: str) -> dict:
    kind, bits, k, pbits, ebits, arm, target = VARIANTS[name]
    if k is None:   # largest k whose computed size stays under 1e9 bytes
        k = max(j for j in range(N_LAYERS + 1)
                if sb.estimate_bytes(counts, [pbits] * j + [bits] * (N_LAYERS - j), ebits) < 1e9)
    protect = [] if k == 0 else sorted(ORDER[:k]) if arm == "circuit" else sb.random_layers(N_LAYERS, k, SEED)
    layer_bits = [16] * N_LAYERS if kind == "fp16" else [pbits if i in protect else bits for i in range(N_LAYERS)]
    return {"variant": name, "kind": kind, "base_bits": bits, "k": k, "protect_bits": pbits, "embedding_bits": ebits,
            "arm": arm, "target_avg_bits": target, "protected_layers": protect, "layer_bits": layer_bits,
            "avg_block_bits": round(sb.avg_block_bits(layer_bits, counts["block_linear_per_layer"]), 4),
            "est_bytes": sb.estimate_bytes(counts, layer_bits, ebits)}


def build(p: dict) -> tuple[Path, dict]:
    tok = load_tokenizer(base_dir)
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float16, device_map={"": 0})
    model.eval()
    t = time.time()
    info = {}
    if p["embedding_bits"] < 16:
        info["embedding"] = sb.fake_quant_embedding(model, p["embedding_bits"])
    if p["kind"] == "ck":
        info.update(sb.quantize_circuitkit(model, tok, NODE_SCORES, p["base_bits"], p["protected_layers"], CALIB, MAX_SEQ))
    else:
        info.update(sb.quantize_direct(model, tok, p["base_bits"], p["protected_layers"], p["protect_bits"], CALIB, MAX_SEQ))
    info["quantize_seconds"] = round(time.time() - t, 1)
    out = work / "models" / p["variant"]
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "job25_variant.json").write_text(json.dumps({**p, "build": info, "model": MODEL, "model_rev": MODEL_REV,
                                                         "commit": COMMIT, "versions": versions}, indent=1))
    sb.free(model)
    return out, info


def measure(mdir: Path) -> dict:
    sb.free()
    mem0 = sb.allocated()
    t = time.time()
    tok = load_tokenizer(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float16, device_map={"": 0})
    model.eval().requires_grad_(False)
    r = {"load_seconds": round(time.time() - t, 1), "weights_gib_after_load": sb.gib(sb.allocated() - mem0)}
    r["throughput"] = sb.throughput(model, tok, bench_prompts, max_new_tokens=32 if TEST else 128,
                                    plan={1: 2, 8: 8} if TEST else {1: 4, 8: 16}, reps=1 if TEST else 2,
                                    mem_before_load=mem0)
    r["cis"] = sb.cis_sep(model, tok, bundle, probe)
    print(f"  cis_sep {r['cis']['cis_sep']:.4f} ({r['cis']['level_cis_sep']}); tok/s bs1 "
          f"{r['throughput']['1']['tok_s_median']} bs8 {r['throughput']['8']['tok_s_median']}", flush=True)
    r["quality"] = sb.quality(model, tok, judge, sets, refs)
    sb.free(model)
    return r


CALIB = sb.calibration_texts(N_CALIB, WIKITEXT_REV) if any(VARIANTS[v][0] != "fp16" for v in REQ) else []
judge = Judge("llamaguard", model_id=JUDGE, revision=JUDGE_REV)
rows, skipped, failed = [], [], []
meta = {"shard": SHARD, "test": TEST, "commit": COMMIT, "model": MODEL, "model_rev": MODEL_REV, "judge": JUDGE,
        "judge_rev": JUDGE_REV, "bundle": BUNDLE, "eval": EVAL, "wikitext_rev": WIKITEXT_REV, "n_calib": N_CALIB,
        "max_seq": MAX_SEQ, "seed": SEED, "group_size": sb.GROUP, "counts": counts, "versions": versions,
        "scores_source": scores_rec.get("source"), "scores_jobs_rev": scores_rec.get("jobs_rev"),
        "circuit_order": ORDER, "amendment": "docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md @ cd72fc6",
        "frozen": {"warn": sb.FROZEN_WARN, "fail": sb.FROZEN_FAIL}}
for name in REQ:
    left = TIMEOUT_MIN - (time.time() - T_START) / 60
    need = EST_MIN["fp16" if name == "fp16" else "quant"] + 3
    if left < need:
        skipped.append(name)
        print(f"SKIP {name}: {left:.0f} min left < {need}", flush=True)
        continue
    t = time.time()
    p = plan(name)
    print(f"== {name}: {json.dumps({k: v for k, v in p.items() if k != 'layer_bits'})}", flush=True)
    try:
        if p["kind"] == "fp16":
            mdir, info = base_dir, {}
        else:
            mdir, info = build(p)
            print(f"  built in {info['quantize_seconds']} s", flush=True)
        disk = sb.disk_bytes(mdir)
        computed = disk
        if p["embedding_bits"] < 16:   # the embedding is stored dequantized; count it at its packed size
            packed = sb.bits_with_scales(p["embedding_bits"]) * counts["embedding"] / 8 * (1 if counts["tied"] else 2)
            computed = disk - sb.embedding_bytes_on_disk(mdir) + round(packed)
        m = measure(mdir)
        if p["kind"] != "fp16":
            pending.append(uploads.submit(upload_model, mdir, name))
        q = m.pop("quality")
        row = {**p, "build": info, "disk_bytes": disk, "size_bytes": computed, "size_gb": round(computed / 1e9, 4),
               "size_is_computed": p["embedding_bits"] < 16, "under_1gb": computed < 1e9, **m,
               **{k: v for k, v in q.items() if not k.startswith("_")},
               "audit_seconds": q["_seconds"], "minutes": round((time.time() - t) / 60, 2)}
        meta.setdefault("judge_info", q["_judge"])
        meta.setdefault("generation", q["_generation"])
        rows.append(row)
        print(f"  ASR {row['asr']['value']:.3f} OR {row['over_refusal']['value']:.3f} task {row['task']['value']:.3f} "
              f"general {row['general']['value']:.3f}; {row['size_gb']} GB; {row['minutes']} min", flush=True)
        upload_json(rows, f"{SHARD}/rows.jsonl")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        failed.append({"variant": name, "error": repr(e)[:1000]})
        sb.free()


# ---- paired comparisons (within this job) ----------------------------------------------------------------------------
by = {r["variant"]: r for r in rows}
paired = []
METRICS = ("asr", "over_refusal", "task", "general")
for r in rows:
    if r["variant"] != "fp16" and "fp16" in by:
        paired.append({"a": r["variant"], "b": "fp16",
                       **{m: sb.mcnemar_exact(r[m]["flags"], by["fp16"][m]["flags"]) for m in METRICS}})
    if r["arm"] == "circuit" and "r" + r["variant"][1:] in by:
        rv = by["r" + r["variant"][1:]]
        t = sb.mcnemar_exact(r["asr"]["flags"], rv["asr"]["flags"])
        paired.append({"a": r["variant"], "b": rv["variant"], "prediction": "circuit ASR < random ASR",
                       "holds": t["diff"] < 0 and t["p"] < 0.05,
                       **{m: sb.mcnemar_exact(r[m]["flags"], rv[m]["flags"]) for m in METRICS}})
upload_json(paired, f"{SHARD}/paired.json")
for f in pending:
    f.result()
summary = {**meta, "requested": REQ, "done": [r["variant"] for r in rows], "skipped_time_guard": skipped,
           "failed": failed, "minutes": round((time.time() - T_START) / 60, 1)}
upload_json(summary, f"{SHARD}/summary.json")
print(json.dumps({k: v for k, v in summary.items() if k not in ("counts", "circuit_order")}), flush=True)

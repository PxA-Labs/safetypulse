# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.21.0",
#   "compressed-tensors==0.20.1",
#   "accelerate==1.16.0",
#   "peft==0.21.2",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "sentencepiece",
#   "numpy",
#   "pydantic>=2",
#   "safetypulse @ git+https://github.com/PxA-Labs/safetypulse@106e9ee6f1000271f31107467ab82f66adbdb446",
# ]
# ///
"""SafetyPulse HF Job 29 (EXPLORATORY, docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md @ cd72fc6, E3):
multilingual safety. Hindi, Spanish and French ASR of Tiny Aya Global (base), the Alpaca fine-tune (alpaca-s1
checkpoint-125 merged, the Job 26 preset), the Job 4 repaired model, and the E1 frontier variants of Job 25.

1. Translation (in memory only). The 119 held-out test prompts (data/eval_harm.jsonl split "test" in SP_REPO at Job 20's
   EVAL_REV) are translated to hi / es / fr with Job 20's pinned NLLB-200 distilled 600M and Job 20's `translate`
   function copied verbatim (line by line, greedy, fp32, batch 16). The sha256 of each translated set (Job 20's rule:
   UTF-8 texts joined by "\n", in input order) is compared with Job 20's stored hash (results/job20/translations.json);
   Job 20 stored only set-level hashes, so the comparison is a match / no match per language, plus Job 20's length
   statistics side by side. A mismatch is reported, not fatal. Per-prompt sha256 of this job's translations are stored
   for later runs. The translated text never leaves the job.
2. For every model and every language (en = the original prompts, in the same job): greedy answers, 256 new tokens,
   fp16, batch 16 (notebook 17 / Job 25 settings through `safetypulse.audit.generate`), judged with Llama Guard 3 1B
   (pinned, as pre-registered; prompt and answer in the target language). Stored per (model, language): n, k, ASR,
   Wilson 95% interval, per-prompt flags, answer character lengths, the digest of the answers, prompt-set sha256.
3. Summary per model: ASR per language, change from that model's English ASR in this job, paired exact McNemar
   (same prompt index, language vs English). For the base model and the E1 variants, English flags are also compared
   with Job 25's flags for the same model (agreement count; generation is not guaranteed to repeat across jobs, Job 22).
No alarm is evaluated here (amendment E3). Nothing is published; answers, prompts and translations stay in memory.

Env: SP_MODELS (comma list of MODELS keys), SP_PART (output subfolder), SP_TIMEOUT_MIN (= --timeout in minutes),
SP_CODE_SHA (the pushed commit the script was launched from), SP_TEST=1 (first 8 prompts per language for generation;
all 119 are still translated so the hash check runs; outputs under results/job29-test/).
"""
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

T_START = time.time()
TEST = os.environ.get("SP_TEST") == "1"
REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
OUT = f"results/job29{'-test' if TEST else ''}"
PART = os.environ.get("SP_PART", "test" if TEST else "all")
TIMEOUT_MIN = float(os.environ.get("SP_TIMEOUT_MIN", 60))
CODE_SHA = os.environ.get("SP_CODE_SHA")
REQ = [m for m in os.environ.get("SP_MODELS", "base").split(",") if m]
N_GEN = 8 if TEST else 119

# ---- pins ---------------------------------------------------------------------------------------------------------
COMMIT = "106e9ee6f1000271f31107467ab82f66adbdb446"   # package + sp_bench.py, as Job 25
GLOBAL = ("CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324")
ALPACA = ("purvansh01/safetypulse-lora-alpaca", "0ea31228c451cdacb47de1f3adb7c9736b950800", "alpaca-s1/checkpoint-125")
REPAIRED = ("purvansh01/tiny-aya-global-safetypulse-repaired", "06ee768d15feac8638e834d28fbb38ab6e4fb11a")
JUDGE, JUDGE_REV = "meta-llama/Llama-Guard-3-1B", "acf7aafa60f0410f8f42b1fa35e077d705892029"
MT = ("facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d")   # Job 20
EVAL_REV = "194c2f5364e5c389ae4db8054eda436c3ef2a74a"   # SP_REPO revision of data/eval_harm.jsonl (Job 20)
JOBS_REV = "77c4650fcf77e1c517fbd04002cc67e86f7102e1"   # SP_REPO revision with results/job25 (models, rows) and job20
EVAL25 = ("purvansh01/safetypulse-eval", "5e49af71c0553360581285feb73642bfff75ed2c")   # Job 25's copy of the test set
NLLB_CODE = {"hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn"}
LANGS = ["en", "hi", "es", "fr"]
MT_BATCH, MT_MAX_IN, MT_MAX_NEW = 16, 1000, 512   # Job 20
MAX_NEW, BATCH = 256, 16                           # notebook 17 / Job 25 ASR settings
# E1 frontier (research-notes/wf-2026-11-24/builder-e3.md: non-dominated in disk_bytes vs general among the
# non-collapsed Job 25 variants, top 5 by general), ordered by disk size
FRONTIER = {"u4": "A", "r5": "B", "r6": "B", "r72": "A", "u8": "A"}   # variant -> Job 25 shard
MODELS = {"base": {"repo": GLOBAL[0], "rev": GLOBAL[1]},
          "alpaca_s1_125": {"repo": GLOBAL[0], "rev": GLOBAL[1], "adapter": ALPACA},
          "repaired": {"repo": REPAIRED[0], "rev": REPAIRED[1]},
          **{v: {"repo": REPO, "rev": JOBS_REV, "repo_type": "dataset", "subfolder": f"results/job25/models/{v}",
                 "job25_shard": s} for v, s in FRONTIER.items()}}
JOB25_ROW = {"base": ("fp16", "A"), **{v: (v, s) for v, s in FRONTIER.items()}}   # English flags to compare with
EST_MIN = {"fp": 6, "quant": 9} if not TEST else {"fp": 2, "quant": 2}   # per model (4 languages), for the time guard
unknown = [m for m in REQ if m not in MODELS]
assert not unknown, f"unknown models {unknown}; known {sorted(MODELS)}"
assert CODE_SHA or TEST, "full runs need SP_CODE_SHA"

import torch  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

assert torch.cuda.is_available(), "needs a GPU"
lib = Path(tempfile.mkdtemp())
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PxA-Labs/safetypulse/{COMMIT}/tools/hf_jobs/sp_bench.py",
                           lib / "sp_bench.py")
sys.path.insert(0, str(lib))
import sp_bench as sb  # noqa: E402

from safetypulse.audit import generate  # noqa: E402
from safetypulse.judge import Judge  # noqa: E402
from safetypulse.models import load_tokenizer  # noqa: E402
from safetypulse.utils import batched, load_jsonl  # noqa: E402

api = HfApi()
work = Path(tempfile.mkdtemp())
versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
print("versions:", json.dumps(versions), flush=True)


def sha(texts):
    """Job 20's set hash: sha256 of the UTF-8 texts joined by newlines, in input order."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def sha1(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(texts):
    """safetypulse.audit._digest: sha256 of the JSON list (completions_sha256 in Job 25 rows)."""
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode("utf-8")).hexdigest()


def upload(obj, name):
    p = work / name.replace("/", "__")
    p.write_text(json.dumps(obj, indent=1) if name.endswith(".json") else "".join(json.dumps(r) + "\n" for r in obj))
    for attempt in range(4):
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{OUT}/{PART}/{name}", repo_id=REPO,
                            repo_type="dataset", commit_message=f"job29: {PART}/{name}")
            return
        except Exception as e:  # noqa: BLE001 - parallel jobs commit to the same repo
            print(f"upload {name} failed ({type(e).__name__}), attempt {attempt + 1}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"upload of {name} failed")


def fetch_rows():
    try:
        p = hf_hub_download(REPO, f"{OUT}/{PART}/rows.jsonl", repo_type="dataset", force_download=True)
    except Exception:  # noqa: BLE001 - first run
        return []
    return load_jsonl(p)


# ---- inputs -------------------------------------------------------------------------------------------------------
test = [x["prompt"] for x in load_jsonl(hf_hub_download(REPO, "data/eval_harm.jsonl", repo_type="dataset",
                                                        revision=EVAL_REV)) if x["split"] == "test"]
assert len(test) == 119, len(test)
job20 = json.loads(Path(hf_hub_download(REPO, "results/job20/translations.json", repo_type="dataset",
                                        revision=JOBS_REV)).read_text())
assert sha(test) == job20["sha256"]["en"]["test"], "English test prompts differ from Job 20's"
e25 = Path(snapshot_download(EVAL25[0], repo_type="dataset", revision=EVAL25[1], allow_patterns=["eval_harm.jsonl"]))
test25 = [x["prompt"] for x in load_jsonl(e25 / "eval_harm.jsonl") if x["split"] == "test"]
same_as_job25 = test25 == test
print(f"119 test prompts; en sha matches Job 20; identical to Job 25's set: {same_as_job25}", flush=True)

# ---- 1. translation (Job 20's code path, in memory only) ------------------------------------------------------------
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

t_mt = time.time()
mt_dir = snapshot_download(MT[0], revision=MT[1])
mt_tok = AutoTokenizer.from_pretrained(mt_dir, src_lang="eng_Latn")
mt = AutoModelForSeq2SeqLM.from_pretrained(mt_dir, dtype=torch.float32).to("cuda").eval()


@torch.no_grad()
def translate(texts, lang):
    """Line by line (NLLB is trained on sentences; multi-line prompts keep their line structure), greedy.
    Verbatim from tools/hf_jobs/job20_h4_multilingual.py (sp.batched -> safetypulse.utils.batched, same code)."""
    lines = [(i, j, ln) for i, t in enumerate(texts) for j, ln in enumerate(t.split("\n")) if ln.strip()]
    order = sorted(range(len(lines)), key=lambda k: len(lines[k][2]))
    done_lines, stats = {}, {"lines": len(lines), "truncated_inputs": 0, "hit_max_new_tokens": 0}
    tgt = mt_tok.convert_tokens_to_ids(NLLB_CODE[lang])
    stop = {mt_tok.pad_token_id, mt_tok.eos_token_id}
    for chunk in batched(order, MT_BATCH):
        batch = [lines[k][2] for k in chunk]
        stats["truncated_inputs"] += sum(len(mt_tok(t).input_ids) > MT_MAX_IN for t in batch)
        enc = mt_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MT_MAX_IN).to("cuda")
        gen = mt.generate(**enc, forced_bos_token_id=tgt, num_beams=1, do_sample=False, max_new_tokens=MT_MAX_NEW)
        if gen.shape[1] > MT_MAX_NEW:   # rows still running when the cap was reached
            stats["hit_max_new_tokens"] += sum(int(x) not in stop for x in gen[:, -1])
        for k, t in zip(chunk, mt_tok.batch_decode(gen, skip_special_tokens=True), strict=True):
            done_lines[k] = t.strip()
    out = [t.split("\n") for t in texts]
    for k, (i, j, _) in enumerate(lines):
        out[i][j] = done_lines[k]
    out = ["\n".join(x).strip() for x in out]
    ratios = [len(o) / max(1, len(s)) for o, s in zip(out, texts, strict=True)]
    stats.update(empty=sum(not o for o in out), char_ratio_median=round(statistics.median(ratios), 4),
                 char_ratio_min=round(min(ratios), 4), char_ratio_max=round(max(ratios), 4))
    return out, stats


TEXT = {"en": test}
tr = {"mt_model": MT[0], "mt_revision": MT[1], "decoding": "greedy, num_beams=1, line by line (Job 20 translate)",
      "max_input_tokens": MT_MAX_IN, "max_new_tokens": MT_MAX_NEW, "batch": MT_BATCH, "dtype": "float32",
      "sha256_rule": "sha256 of the UTF-8 texts joined by newlines, in input order (Job 20)",
      "job20_source": f"{REPO}@{JOBS_REV}:results/job20/translations.json", "versions": versions,
      "en": {"sha256": sha(test), "matches_job20": True, "identical_to_job25_eval_set": same_as_job25,
             "prompt_sha256": [sha1(t) for t in test], "chars": [len(t) for t in test]}}
for lang in NLLB_CODE:
    TEXT[lang], stats = translate(test, lang)
    h = sha(TEXT[lang])
    j20 = job20["stats"][lang]["test"]
    tr[lang] = {"sha256": h, "job20_sha256": job20["sha256"][lang]["test"], "matches_job20": h == job20["sha256"][lang]["test"],
                "stats": stats, "job20_stats": j20,
                "stats_equal_job20": {k: stats.get(k) == v for k, v in j20.items()},
                "prompt_sha256": [sha1(t) for t in TEXT[lang]], "chars": [len(t) for t in TEXT[lang]]}
    print(f"translated {lang}: matches Job 20 {tr[lang]['matches_job20']}; stats {stats}", flush=True)
    if not tr[lang]["matches_job20"]:
        print(f"WARNING: {lang} translation set differs from Job 20's (reported, not fatal)", flush=True)
tr["match_rate_by_language"] = {lang: float(tr[lang]["matches_job20"]) for lang in NLLB_CODE}
tr["minutes"] = round((time.time() - t_mt) / 60, 2)
upload(tr, "translations.json")
sb.free(mt)
mt = None
torch.cuda.empty_cache()


# ---- 2. models, generation, judging ---------------------------------------------------------------------------------
def load(name):
    from transformers import AutoModelForCausalLM

    spec = MODELS[name]
    sub = spec.get("subfolder")
    local = Path(snapshot_download(spec["repo"], revision=spec["rev"], repo_type=spec.get("repo_type", "model"),
                                   allow_patterns=[f"{sub}/*"] if sub else
                                   ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))
    path = local / sub if sub else local
    tok = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float16, device_map={"": 0})
    if spec.get("adapter"):
        from peft import PeftModel

        repo, rev, folder = spec["adapter"]
        ad = Path(snapshot_download(repo, revision=rev, allow_patterns=[f"{folder}/*"]))
        model = PeftModel.from_pretrained(model, str(ad / folder)).merge_and_unload()
    return model.eval().requires_grad_(False), tok


judge = Judge("llamaguard", model_id=JUDGE, revision=JUDGE_REV).load()
rows = fetch_rows()
done = {(r["model"], r["lang"]) for r in rows}
print(f"{len(done)} (model, lang) rows already done", flush=True)
skipped, failed = [], []
for name in REQ:
    todo = [lg for lg in LANGS if (name, lg) not in done]
    if not todo:
        continue
    left = TIMEOUT_MIN - (time.time() - T_START) / 60
    need = EST_MIN["quant" if "subfolder" in MODELS[name] else "fp"] + 3
    if left < need:
        skipped.append(name)
        print(f"SKIP {name}: {left:.0f} min left < {need}", flush=True)
        continue
    try:
        t = time.time()
        model, tok = load(name)
        load_s = round(time.time() - t, 1)
        print(f"== {name} loaded in {load_s} s", flush=True)
        for lang in todo:
            t = time.time()
            prompts = TEXT[lang][:N_GEN]
            answers = generate(model, tok, prompts, MAX_NEW, BATCH, desc=f"{name} {lang}")
            flags = [bool(f) for f in judge.harmful(prompts, answers)]
            k, n = sum(flags), len(flags)
            rows.append({"model": name, "lang": lang, "n": n, "k": k, "asr": k / n, "wilson95": list(sb.wilson(k, n)),
                         "flags": flags, "prompts_sha256": sha(prompts), "completions_sha256": digest(answers),
                         "answer_chars": [len(a) for a in answers], "n_empty": sum(not a.strip() for a in answers),
                         "seconds": round(time.time() - t, 1), "load_seconds": load_s,
                         "translation_matches_job20": tr[lang]["matches_job20"]})
            del answers
            print(f"  {name} {lang}: ASR {k}/{n} = {k / n:.3f}  ({rows[-1]['seconds']} s, total "
                  f"{(time.time() - T_START) / 60:.1f} min)", flush=True)
            upload(rows, "rows.jsonl")
        sb.free(model)
        model = None
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        failed.append({"model": name, "error": repr(e)[:1000]})
        model = None
        sb.free()


# ---- 3. summary -----------------------------------------------------------------------------------------------------
def job25_flags(name):
    if name not in JOB25_ROW:
        return None
    variant, shard = JOB25_ROW[name]
    for r in load_jsonl(hf_hub_download(REPO, f"results/job25/{shard}/rows.jsonl", repo_type="dataset",
                                        revision=JOBS_REV)):
        if r["variant"] == variant:
            return r["asr"]["flags"]
    return None


by = {(r["model"], r["lang"]): r for r in rows}
per_model = {}
for name in REQ:
    en = by.get((name, "en"))
    if en is None:
        continue
    m = {}
    for lang in LANGS:
        r = by.get((name, lang))
        if r is None:
            continue
        m[lang] = {"n": r["n"], "k": r["k"], "asr": round(r["asr"], 6), "wilson95": [round(x, 6) for x in r["wilson95"]],
                   "delta_vs_en": round(r["asr"] - en["asr"], 6),
                   "mcnemar_vs_en": None if lang == "en" else sb.mcnemar_exact(r["flags"], en["flags"]),
                   "translation_matches_job20": r["translation_matches_job20"]}
    f25 = job25_flags(name)
    if f25 is not None:
        f25 = f25[:en["n"]]
        m["en_vs_job25"] = {"job25_k": sum(f25), "this_job_k": en["k"], "n": en["n"],
                            "agree": sum(a == b for a, b in zip(en["flags"], f25, strict=True))}
    per_model[name] = m
summary = {"amendment": "docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md @ cd72fc6 (E3, exploratory)",
           "part": PART, "test": TEST, "requested": REQ, "done": sorted({r["model"] for r in rows}),
           "skipped_time_guard": skipped, "failed": failed, "code_sha": CODE_SHA, "package_commit": COMMIT,
           "job_id": os.environ.get("JOB_ID"), "models": MODELS, "judge": [JUDGE, JUDGE_REV], "mt": list(MT),
           "eval_rev": EVAL_REV, "jobs_rev": JOBS_REV, "n_generated": N_GEN,
           "generation": {"decoding": "greedy", "max_new_tokens": MAX_NEW, "batch_size": BATCH, "dtype": "float16",
                          "chat_template": "model's own, template default system prompt (safetypulse.prompts.format_chat)"},
           "translation_match_rate": tr["match_rate_by_language"], "per_model": per_model, "versions": versions,
           "minutes": round((time.time() - T_START) / 60, 1)}
upload(summary, "summary.json")
print(json.dumps({k: summary[k] for k in ("done", "skipped_time_guard", "failed", "translation_match_rate", "minutes")}),
      flush=True)
for name, m in per_model.items():
    print(name, {lg: (v["k"], v["n"], v["delta_vs_en"]) for lg, v in m.items() if lg in LANGS}, m.get("en_vs_job25"),
          flush=True)
print("done", flush=True)

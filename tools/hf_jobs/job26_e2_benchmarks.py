# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=5.15,<6",
#   "accelerate",
#   "peft==0.21.2",
#   "huggingface_hub>=1.5",
#   "safetensors",
#   "sentencepiece",
#   "compressed-tensors==0.20.1",
#   "lm-eval[hf,ifeval]==0.4.13",
#   "langdetect",
#   "immutabledict",
#   "nltk>=3.9.1",
# ]
# ///
"""SafetyPulse HF Job 26 (amendment docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md, E2; exploratory).

E2 text-generation benchmarks: IFEval (prompt-level strict and loose accuracy, all 541 prompts) and XQuAD en / hi / es
(exact match and F1, the same N=500 question ids per language, seed 0) through lm-evaluation-harness 0.4.13, greedy,
task-default generation settings, the model's own chat template with no system instruction.

Models (one or more per job, `--models`): `base` = Tiny Aya Global; `alpaca_s1_125` = base + the alpaca-s1
checkpoint-125 LoRA merged in (the drifted checkpoint the repaired model was built from, tools/hf_jobs/job4_repaired_model.py);
`repaired` = the Job 4 SafeTune-repaired model. Any other variant (e.g. the E1/E5 quantized checkpoints in wave 2) is
given with `--model-repo/--model-rev [--model-subfolder] [--model-repo-type dataset] --label NAME`, or several
subfolders of one repo with `--model-subfolders a/x,b/y` (label = last path component; used by Job 30 for the Job 25
frontier checkpoints, with `--base-tokenizer` so every variant uses the pinned Tiny Aya tokenizer, as Job 26 base did);
the checkpoint is loaded with AutoModelForCausalLM.from_pretrained (plain, PEFT-merged or compressed-tensors all work),
so `bench()` can also be imported from this file at a pinned commit.

Release policy: generated answers stay in memory. Stored per model (SP_REPO private, results/job26/<label>.json): summary
metrics with lm-eval stderr and Wilson 95% intervals, versions, sample counts, generation settings, chat-template sha256,
and per document only doc_id, the dataset's question id/key, scores, and the sha256 and length of the answer.

Launch (pin the code by launching the raw file at a pushed commit and passing the same SHA):
  hf jobs uv run --namespace pxaLabs-lexsi --flavor l40sx1 --timeout 60m --secrets HF_TOKEN -e SP_CODE_SHA=<sha> \
    --detach https://raw.githubusercontent.com/PxA-Labs/safetypulse/<sha>/tools/hf_jobs/job26_e2_benchmarks.py \
    --models base
SP_TEST=1 (or --limit N): 5 docs per task and the results/job26-test/ path.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import tempfile
import time
import types
from collections import Counter
from pathlib import Path

# ---- pins -------------------------------------------------------------------------------------------------------------
REPO = os.environ.get("SP_REPO", "purvansh01/safetypulse-jobs")
GLOBAL = ("CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324")   # eos 3, as Stage 1 and Jobs 18-24
ALPACA = ("purvansh01/safetypulse-lora-alpaca", "0ea31228c451cdacb47de1f3adb7c9736b950800", "alpaca-s1/checkpoint-125")
REPAIRED = ("purvansh01/tiny-aya-global-safetypulse-repaired", "06ee768d15feac8638e834d28fbb38ab6e4fb11a")
PRESETS = {
    "base": {"repo": GLOBAL[0], "rev": GLOBAL[1]},
    "alpaca_s1_125": {"repo": GLOBAL[0], "rev": GLOBAL[1], "adapter": ALPACA},
    "repaired": {"repo": REPAIRED[0], "rev": REPAIRED[1]},
}
IFEVAL_DS = ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84")
XQUAD_DS = ("google/xquad", "51adfef1c1287aab1d2d91b5bead9bcfb9c68583")
XQ_LANGS = ("en", "hi", "es")
XQ_TOTAL, XQ_N, XQ_SEED = 1190, 500, 0   # XQuAD is parallel: the same question ids in all three languages
XQ_IDS_SHA = "1bee96cea389035a9970046f3f3c309cf8c5f71554af626e57325b7cc31d3a3c"   # sha256(json.dumps(sorted ids))
TASKS = ["ifeval"] + [f"xquad_{lg}" for lg in XQ_LANGS]
RATE_METRICS = {"ifeval": ("prompt_level_strict_acc", "prompt_level_loose_acc"), "xquad": ("exact_match",)}
DOC_METRICS = ("prompt_level_strict_acc", "prompt_level_loose_acc", "inst_level_strict_acc", "inst_level_loose_acc",
               "exact_match", "f1")
BATCH = 32
DEV = os.environ.get("SP_DEVICE", "cuda")   # "cpu" only for local tests of the code path
Z = 1.959963984540054


# ---- lm-eval 0.4.13's xquad task imports transformers.data.metrics.squad_metrics, removed in transformers 5.20
# (lm-eval issue #4435, unmerged PR #4440). Stand-in: the four functions copied from transformers 5.19.0
# data/metrics/squad_metrics.py L36-79 (official SQuAD v1.1 scoring), registered only when the module is missing.
def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _get_tokens(s):
    return _normalize_answer(s).split() if s else []


def _compute_exact(a_gold, a_pred):
    return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))


def _compute_f1(a_gold, a_pred):
    gold_toks, pred_toks = _get_tokens(a_gold), _get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision, recall = num_same / len(pred_toks), num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def install_squad_shim():
    """Return True if the stand-in module had to be registered."""
    try:
        import transformers.data.metrics.squad_metrics  # noqa: F401
        return False
    except ModuleNotFoundError:
        import transformers.data  # noqa: F401
        sm = types.ModuleType("transformers.data.metrics.squad_metrics")
        sm.normalize_answer, sm.get_tokens = _normalize_answer, _get_tokens
        sm.compute_exact, sm.compute_f1 = _compute_exact, _compute_f1
        pkg = types.ModuleType("transformers.data.metrics")
        pkg.squad_metrics = sm
        sys.modules.update({"transformers.data.metrics": pkg, "transformers.data.metrics.squad_metrics": sm})
        return True


SHIM = install_squad_shim()

import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from langdetect import DetectorFactory  # noqa: E402

DetectorFactory.seed = 0   # IFEval's language checks call langdetect.detect, which is random unless seeded
import lm_eval  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def template_sha(tok):
    """Tiny Aya's tokenizer holds named templates (a dict); hash a canonical JSON form in that case."""
    ct = tok.chat_template or ""
    return sha(ct if isinstance(ct, str) else json.dumps(ct, sort_keys=True))


def xquad_ids():
    ids = sorted(random.Random(XQ_SEED).sample(range(XQ_TOTAL), XQ_N))
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == XQ_IDS_SHA
    return ids


def check_datasets(api):
    """Benchmark data cannot be revision-pinned through lm-eval's task YAMLs: fail if the Hub copies moved."""
    for repo, rev in (IFEVAL_DS, XQUAD_DS):
        got = api.dataset_info(repo).sha
        assert got == rev, f"{repo} moved: {got} != pinned {rev}"


def wilson(k, n):
    if n == 0:
        return None
    p, d = k / n, 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 6), round(min(1.0, c + h), 6)]


def mean_ci(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return [round(max(0.0, m - Z * se), 6), round(min(1.0, m + Z * se), 6)]


def load_variant(repo, rev, subfolder=None, repo_type="model", adapter=None, tokenizer=None, dtype=torch.bfloat16,
                 force_tokenizer=False):
    """Load a checkpoint (plain, compressed-tensors or other HF-loadable format) and optionally merge a LoRA adapter.

    adapter = (repo, rev, subfolder); tokenizer = (repo, rev), used when the checkpoint folder has no tokenizer or
    when force_tokenizer is set.
    """
    pats = [f"{subfolder}/*"] if subfolder else None
    local = Path(snapshot_download(repo, revision=rev, repo_type=repo_type, allow_patterns=pats))
    path = local / subfolder if subfolder else local
    has_tok = not force_tokenizer and any((path / f).exists() for f in ("tokenizer.json", "tokenizer_config.json"))
    tok_src = (str(path), None) if has_tok else (tokenizer or GLOBAL)
    tok = AutoTokenizer.from_pretrained(tok_src[0], revision=tok_src[1]) if tok_src[1] else \
        AutoTokenizer.from_pretrained(tok_src[0])
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype).to(DEV)
    if adapter:
        from peft import PeftModel

        ad = Path(snapshot_download(adapter[0], revision=adapter[1], allow_patterns=[f"{adapter[2]}/*"]))
        model = PeftModel.from_pretrained(model, str(ad / adapter[2])).merge_and_unload()
    return model.eval(), tok


def quant_info(model):
    """quant_method / format / bits of a quantized checkpoint (None for plain weights); never raises."""
    try:
        q = getattr(model.config, "quantization_config", None)
        if q is None:
            return None
        q = q if isinstance(q, dict) else q.to_dict()
        bits = sorted({g.get("weights", {}).get("num_bits") for g in (q.get("config_groups") or {}).values()} - {None})
        return {"quant_method": q.get("quant_method"), "format": q.get("format"), "bits": bits,
                "n_ignore": len(q.get("ignore") or [])}
    except Exception as e:   # noqa: BLE001 - metadata only
        return {"error": type(e).__name__}


def bench(model, tok, label, limit=None, batch=BATCH):
    """IFEval + XQuAD en/hi/es on a loaded model. Returns scores and hashes only; answer text is dropped here."""
    t0 = time.time()
    tok.padding_side = "left"
    gpu = DEV == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    ids = xquad_ids()
    samples = {f"xquad_{lg}": ids[:limit] if limit else ids for lg in XQ_LANGS}
    if limit:
        samples["ifeval"] = list(range(limit))
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch, device=DEV)
    res = lm_eval.simple_evaluate(model=lm, tasks=TASKS, samples=samples, apply_chat_template=True,
                                  system_instruction=None, log_samples=True, bootstrap_iters=1000, random_seed=0,
                                  numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
    per_doc, summary = {}, {}
    for task in TASKS:
        rows = []
        for r in sorted(res["samples"][task], key=lambda r: r["doc_id"]):
            ans = str(r["filtered_resps"][0])
            doc = r.get("doc") or {}
            row = {"doc_id": r["doc_id"], "answer_sha256": sha(ans), "answer_chars": len(ans)}
            if task == "ifeval":
                row["key"] = doc.get("key")
            else:
                row["qid"] = doc.get("id")
            for m in DOC_METRICS:
                if m in r:
                    row[m] = [bool(x) for x in r[m]] if isinstance(r[m], list) else float(r[m])
            rows.append(row)
        per_doc[task] = rows
        s = {k.split(",")[0]: v for k, v in res["results"][task].items() if k not in ("alias", "name")}
        n = len(rows)
        s["n"] = n
        for m in RATE_METRICS["ifeval" if task == "ifeval" else "xquad"]:
            k = sum(int(x[m] >= 0.5) for x in rows)
            s[f"{m}_k"], s[f"{m}_wilson95"] = k, wilson(k, n)
        if task != "ifeval":
            s["f1_ci95_normal"] = mean_ci([x["f1"] for x in rows])
        summary[task] = s
    from importlib.metadata import version

    vers = {p: version(p) for p in ("torch", "transformers", "peft", "lm_eval", "compressed-tensors", "accelerate")}
    return {"label": label, "summary": summary, "task_versions": res["versions"], "n_samples": res["n-samples"],
            "generation_kwargs": {t: res["configs"][t].get("generation_kwargs") for t in TASKS},
            "batch_size": batch, "apply_chat_template": True, "system_instruction": None,
            "chat_template_sha256": template_sha(tok), "xquad_ids_sha256": XQ_IDS_SHA,
            "xquad_n": len(samples["xquad_en"]), "squad_shim": SHIM, "langdetect_seed": 0, "packages": vers,
            "datasets": {"ifeval": list(IFEVAL_DS), "xquad": list(XQUAD_DS)},
            "model_dtype": str(next(model.parameters()).dtype), "gpu": torch.cuda.get_device_name(0) if gpu else DEV,
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None,
            "bench_minutes": round((time.time() - t0) / 60, 2), "per_doc": per_doc}


def upload(api, obj, path, msg, tries=5):
    p = Path(os.environ.get("SP_LOCAL_OUT") or tempfile.mkdtemp()) / Path(path).name
    p.write_text(json.dumps(obj, indent=1))
    if os.environ.get("SP_LOCAL_OUT"):   # local tests: keep the file, upload nothing
        return
    for i in range(tries):
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=path, repo_id=REPO, repo_type="dataset",
                            commit_message=f"job26: {msg}")
            return
        except Exception as e:   # noqa: BLE001 - parallel jobs commit to the same repo; retry on conflicts
            print("upload retry", i, type(e).__name__, flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"upload of {path} failed")


def done_before(path):
    if os.environ.get("SP_LOCAL_OUT"):
        return False
    try:
        old = json.loads(Path(hf_hub_download(REPO, path, repo_type="dataset", force_download=True)).read_text())
    except Exception:   # noqa: BLE001 - first run
        return False
    return bool(old.get("complete"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list of presets: " + ",".join(PRESETS))
    ap.add_argument("--label", help="label for a custom variant (--model-repo)")
    ap.add_argument("--model-repo")
    ap.add_argument("--model-rev")
    ap.add_argument("--model-subfolder")
    ap.add_argument("--model-subfolders", default="", help="comma list of subfolders of --model-repo, one variant each "
                    "(label = last path component)")
    ap.add_argument("--base-tokenizer", action="store_true", help="always use --tokenizer-repo/--tokenizer-rev")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="job timeout in minutes; a variant is skipped if fewer than --est-min minutes are left")
    ap.add_argument("--est-min", type=float, default=0.0, help="expected minutes per variant (with --time-budget-min)")
    ap.add_argument("--model-repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--tokenizer-repo", default=GLOBAL[0])
    ap.add_argument("--tokenizer-rev", default=GLOBAL[1])
    ap.add_argument("--limit", type=int, default=5 if os.environ.get("SP_TEST") == "1" else None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--out", default=None, help="results folder in SP_REPO (default results/job26[-test])")
    ap.add_argument("--force", action="store_true", help="rerun labels already complete in --out")
    a = ap.parse_args()
    out = a.out or f"results/job26{'-test' if a.limit else ''}"
    code_sha = os.environ.get("SP_CODE_SHA")
    assert DEV != "cuda" or torch.cuda.is_available(), "needs a GPU"
    assert code_sha or a.limit, "full runs need SP_CODE_SHA (the pushed commit the script was launched from)"

    plan = [(m, PRESETS[m]) for m in a.models.split(",") if m]
    if a.model_repo and a.model_subfolders:
        assert a.model_rev and not (a.label or a.model_subfolder), "--model-subfolders needs --model-rev only"
        for sub in a.model_subfolders.split(","):
            plan.append((sub.rstrip("/").split("/")[-1], {"repo": a.model_repo, "rev": a.model_rev, "subfolder": sub,
                                                          "repo_type": a.model_repo_type}))
    elif a.model_repo:
        assert a.label and a.model_rev, "--model-repo needs --label and --model-rev"
        plan.append((a.label, {"repo": a.model_repo, "rev": a.model_rev, "subfolder": a.model_subfolder,
                               "repo_type": a.model_repo_type}))
    assert plan, "nothing to run"
    api = HfApi()   # token from the HF_TOKEN secret
    check_datasets(api)
    t_job = time.time()
    skipped = []
    for label, spec in plan:
        dest = f"{out}/{label}.json"
        if not a.force and done_before(dest):
            print(label, "already complete in", dest, flush=True)
            continue
        left = a.time_budget_min - (time.time() - t_job) / 60 if a.time_budget_min else None
        if left is not None and left < a.est_min:
            print(label, f"skipped by time guard ({left:.1f} min left < {a.est_min})", flush=True)
            skipped.append(label)
            continue
        t0 = time.time()
        model, tok = load_variant(spec["repo"], spec["rev"], spec.get("subfolder"), spec.get("repo_type", "model"),
                                  spec.get("adapter"), (a.tokenizer_repo, a.tokenizer_rev),
                                  force_tokenizer=a.base_tokenizer)
        load_min = round((time.time() - t0) / 60, 2)
        r = bench(model, tok, label, a.limit, a.batch)
        r.update(complete=True, load_minutes=load_min, limit=a.limit, code_sha=code_sha,
                 amendment="docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md (cd72fc6)",
                 model={k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
                 job_id=os.environ.get("JOB_ID"), base_tokenizer=a.base_tokenizer, quantization=quant_info(model))
        upload(api, r, dest, f"{label} ({'limit ' + str(a.limit) if a.limit else 'full'})")
        brief = {t: {k: round(v, 4) for k, v in s.items() if isinstance(v, float) and "stderr" not in k}
                 for t, s in r["summary"].items()}
        print(label, json.dumps(brief), f"load {load_min} min, bench {r['bench_minutes']} min, peak {r['peak_gib']} GiB",
              flush=True)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
    print(f"done in {(time.time() - t_job) / 60:.1f} min; skipped by time guard: {skipped}", flush=True)


if __name__ == "__main__":
    main()

"""`audit`: ground-truth safety and capability evaluation (docs/03 §3.5, docs/05 §1-§4).

Metrics, each a rate with a 95% bootstrap CI over items (1,000 resamples):
    asr           judge-unsafe rate of greedy completions to harmful prompts          (lower is better)
    over_refusal  string-match refusal rate on benign-but-scary prompts               (lower is better)
    task          GSM8K exact match of the last number in the completion              (higher is better)
    general       multiple-choice accuracy from next-token logits over A-D (MMLU)     (higher is better)

Numerics are those of kaggle/lib/sp_utils.py. The AuditReport is redacted (docs/03 §1 "Safe by default"): it holds
rates, per-item 0/1 flags, counts and hashes, never prompt or completion text. Raw completions go to a separate
local file only when `completions_path` is set explicitly.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .judge import DEFAULT_GUARD, JUDGE_KINDS, Judge, is_refusal
from .prompts import encode, format_chat, last_logits
from .schemas import SCHEMA_VERSION, _Open
from .utils import (
    batched,
    gpu_timer,
    load_json,
    load_jsonl,
    progress,
    provenance,
    save_json,
    save_jsonl,
    sha256_file,
    sha256_rows,
)

# ---------------------------------------------------------------------------------------------
# Generation and task scoring
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def generate(model, tok, prompts: Sequence[str], max_new_tokens: int = 256, batch_size: int = 8,
             system: str | None = None, desc: str = "generate") -> list[str]:
    """Greedy completions (temperature 0), as in the evaluation protocol §2."""
    texts = format_chat(tok, prompts, system)
    outs: list[str] = []
    for batch in progress(list(batched(texts, batch_size)), desc=desc):
        enc = encode(tok, batch, model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


GSM8K_SUFFIX = "\n\nSolve the problem step by step, then give the final numeric answer after '####'."
MC_LETTERS = ["A", "B", "C", "D"]


def last_number(text: str) -> str | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def numbers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold.replace(",", ""))) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def mc_accuracy(model, tok, items: list[dict], batch_size: int = 8) -> list[bool]:
    """Multiple-choice accuracy from next-token logits over the letters A-D (no generation)."""
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in MC_LETTERS]
    prompts = []
    for it in items:
        opts = "\n".join(f"{MC_LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
        prompts.append(f"{it['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only.")
    logits = last_logits(model, tok, prompts, batch_size)
    preds = logits[:, letter_ids].argmax(-1).tolist()
    return [p == it["answer"] for p, it in zip(preds, items, strict=True)]


# ---------------------------------------------------------------------------------------------
# Statistics (docs/05 §4)
# ---------------------------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap CI of the mean, resampling items with replacement."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 1000,
                          seed: int = 0) -> tuple[float, float, float]:
    """mean(a - b) with a 95% CI, resampling items jointly (e.g. per-item flags of two AuditReports)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, seed)


# ---------------------------------------------------------------------------------------------
# AuditReport (docs/03 §4.3)
# ---------------------------------------------------------------------------------------------


class MetricResult(_Open):
    """One audit metric. `flags` are per-item 0/1 outcomes in dataset order (for paired bootstrap comparisons);
    `completions_sha256` fingerprints the completions without storing them."""

    value: float
    ci95: tuple[float, float]
    n: int
    n_positive: int
    flags: list[bool] = []
    scorer: str
    completions_sha256: str | None = None
    n_empty_completions: int | None = None


class DatasetRef(_Open):
    """An eval set: the file it came from (sha256 of the whole file) and the items actually used."""

    path: str | None = None
    sha256: str | None = None
    n_file: int | None = None
    split: str | None = None
    limit: int | None = None
    n_used: int
    used_sha256: str
    item_ids_sha256: str | None = None


class ModelRef(_Open):
    """The audited model. `steering`: the R6 steering hook applied during the audit (a `repair --method steer`
    output dir), else None."""

    path: str | None = None
    base_model: str | None = None
    revision: str | None = None
    steering: dict | None = None


class Redaction(_Open):
    completions: str = "redacted"
    prompts: str = "redacted"
    raw_completions_file: str | None = None


class AuditReport(_Open):
    """Output of `audit`: metric values with bootstrap CIs, judge identity, dataset hashes, generation settings
    and provenance. Holds no prompt or completion text."""

    schema_version: str = SCHEMA_VERSION
    model: ModelRef = ModelRef()
    suite: str = "custom"
    metrics: dict[str, MetricResult]
    judge: dict
    datasets: dict[str, DatasetRef] = {}
    generation: dict = {}
    statistics: dict = {}
    redaction: Redaction = Redaction()
    cost: dict = {}
    provenance: dict = {}

    def summary(self) -> dict[str, float]:
        """{metric: value} for tables."""
        return {k: m.value for k, m in self.metrics.items()}

    def save(self, path: str | Path) -> Path:
        return save_json(self.model_dump(mode="json"), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> AuditReport:
        return cls.model_validate(load_json(path))


# ---------------------------------------------------------------------------------------------
# Metrics (each returns a MetricResult; completions are returned separately and never stored in it)
# ---------------------------------------------------------------------------------------------


def _digest(texts: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode("utf-8")).hexdigest()


def _rate(flags: Sequence[bool], scorer: str, n_boot: int, seed: int, responses: Sequence[str] | None = None) -> MetricResult:
    mean, lo, hi = bootstrap_ci([float(f) for f in flags], n_boot, seed)
    extra = {}
    if responses is not None:
        extra = {"completions_sha256": _digest(responses), "n_empty_completions": sum(not r.strip() for r in responses)}
    return MetricResult(value=mean, ci95=(lo, hi), n=len(flags), n_positive=int(sum(map(bool, flags))),
                        flags=[bool(f) for f in flags], scorer=scorer, **extra)


def eval_asr(model, tok, judge: Judge, items: list[dict], max_new_tokens: int = 256, batch_size: int = 16,
             n_boot: int = 1000, seed: int = 0) -> tuple[MetricResult, list[str]]:
    prompts = [it["prompt"] for it in items]
    responses = generate(model, tok, prompts, max_new_tokens, batch_size, desc="ASR")
    flags = judge.harmful(prompts, responses)
    return _rate(flags, judge.name, n_boot, seed, responses), responses


def eval_overrefusal(model, tok, items: list[dict], max_new_tokens: int = 128, batch_size: int = 16,
                     n_boot: int = 1000, seed: int = 0) -> tuple[MetricResult, list[str]]:
    responses = generate(model, tok, [it["prompt"] for it in items], max_new_tokens, batch_size, desc="over-refusal")
    flags = [is_refusal(r) for r in responses]
    return _rate(flags, "string-match", n_boot, seed, responses), responses


def eval_gsm8k(model, tok, items: list[dict], max_new_tokens: int = 320, batch_size: int = 16,
               n_boot: int = 1000, seed: int = 0) -> tuple[MetricResult, list[str]]:
    responses = generate(model, tok, [it["question"] + GSM8K_SUFFIX for it in items], max_new_tokens, batch_size, desc="task")
    flags = [numbers_match(last_number(r), it["gold"]) for r, it in zip(responses, items, strict=True)]
    return _rate(flags, "gsm8k-exact-match", n_boot, seed, responses), responses


def eval_general(model, tok, items: list[dict], batch_size: int = 8, n_boot: int = 1000, seed: int = 0) -> MetricResult:
    return _rate(mc_accuracy(model, tok, items, batch_size), "mc-letter-logits", n_boot, seed)


# ---------------------------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------------------------

SET_KEYS = ("harm", "overrefusal", "task", "general")
SUITES = {"default": {"harm": "eval_harm.jsonl", "overrefusal": "eval_overrefusal.jsonl",
                      "task": "eval_task.jsonl", "general": "eval_general.jsonl"}}


def _ids_digest(items: Sequence[dict]) -> str | None:
    ids = [it.get("id") for it in items]
    return None if any(i is None for i in ids) else _digest([str(i) for i in ids])


def dataset_ref(items: Sequence[dict], path: str | Path | None = None, **extra) -> DatasetRef:
    """Hashes for an eval set: sha256 of the source file (if any) and of the items used, serialised as jsonl."""
    file_info = {}
    if path is not None:
        file_info = {"path": str(path), "sha256": sha256_file(path)}
    return DatasetRef(**file_info, n_used=len(items), used_sha256=sha256_rows(items), item_ids_sha256=_ids_digest(items),
                      **extra)


def load_suite(data_dir: str | Path, suite: str = "default", split: str | None = "test",
               limit: int | None = None) -> tuple[dict[str, list[dict]], dict[str, DatasetRef]]:
    """Eval sets of a suite from `data_dir` (notebook 01's jsonl files) -> (sets, dataset refs).

    split: keep items whose "split" equals it ("dev" / "test"); None or "all" keeps every item. limit: first N.
    A missing file is skipped (its metric is not computed); a suite with no file at all is an error.
    """
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; available: {sorted(SUITES)}")
    data_dir = Path(data_dir)
    sets, refs, found = {}, {}, {}
    for key, fname in SUITES[suite].items():
        path = data_dir / fname
        if not path.is_file():
            continue
        rows = load_jsonl(path)
        found[fname] = sorted({str(r.get("split")) for r in rows})
        items = rows if split in (None, "all") else [r for r in rows if r.get("split") == split]
        if limit is not None:
            items = items[:limit]
        if not items:   # an empty set would silently drop its metric
            continue
        sets[key] = items
        refs[key] = dataset_ref(items, path, n_file=len(rows), split=split, limit=limit)
    if not found:
        raise FileNotFoundError(f"no {suite} suite files ({', '.join(SUITES[suite].values())}) in {data_dir}")
    if not sets:
        raise ValueError(f"split {split!r} matched no items in {', '.join(found)} (limit={limit}); splits present: "
                         f"{sorted(set().union(*found.values()))}")
    return sets, refs


# ---------------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------------


def audit(
    model,
    tok,
    judge: Judge,
    sets: dict[str, list[dict]],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 256,
    overrefusal_max_new_tokens: int = 128,
    task_max_new_tokens: int = 320,
    n_boot: int = 1000,
    seed: int = 0,
    datasets: dict[str, DatasetRef] | None = None,
    model_ref: dict | ModelRef | None = None,
    suite: str = "custom",
    completions_path: str | Path | None = None,
) -> AuditReport:
    """Run every metric whose eval set is present in `sets` (keys: harm, overrefusal, task, general).

    harm/overrefusal items need "prompt"; task items "question" + "gold"; general items "question", "choices", "answer".
    The per-metric token budgets and batch sizes (general: batch_size // 2, judge: its own) are those of sp_utils.
    datasets: refs from `load_suite`; sets without one are hashed from their items.
    completions_path: if set, ALSO write raw completions ({"metric", "id", "flag", "completion"}) to this local
        jsonl. That file is unredacted model output and must not be shared or uploaded; the report stays redacted.
    """
    unknown = set(sets) - set(SET_KEYS)
    if unknown:
        raise ValueError(f"unknown eval set(s) {sorted(unknown)}; expected keys among {SET_KEYS}")
    datasets = dict(datasets or {})
    metrics: dict[str, MetricResult] = {}
    raw: list[dict] = []
    seconds: dict[str, float] = {}

    def keep(metric: str, items: list[dict], result: MetricResult, responses: list[str]):
        if completions_path is not None:
            raw.extend({"metric": metric, "id": it.get("id", i), "flag": f, "completion": r}
                       for i, (it, f, r) in enumerate(zip(items, result.flags, responses, strict=True)))

    with gpu_timer() as total:
        if sets.get("harm"):
            with gpu_timer() as t:
                m, resp = eval_asr(model, tok, judge, sets["harm"], max_new_tokens, batch_size, n_boot, seed)
            metrics["asr"], seconds["asr"] = m, t["seconds"]
            keep("asr", sets["harm"], m, resp)
        if sets.get("overrefusal"):
            with gpu_timer() as t:
                m, resp = eval_overrefusal(model, tok, sets["overrefusal"], overrefusal_max_new_tokens, batch_size, n_boot, seed)
            metrics["over_refusal"], seconds["over_refusal"] = m, t["seconds"]
            keep("over_refusal", sets["overrefusal"], m, resp)
        if sets.get("task"):
            with gpu_timer() as t:
                m, resp = eval_gsm8k(model, tok, sets["task"], task_max_new_tokens, batch_size, n_boot, seed)
            metrics["task"], seconds["task"] = m, t["seconds"]
            keep("task", sets["task"], m, resp)
        if sets.get("general"):
            with gpu_timer() as t:
                metrics["general"] = eval_general(model, tok, sets["general"], max(1, batch_size // 2), n_boot, seed)
            seconds["general"] = t["seconds"]

    if not metrics:
        raise ValueError(f"no metric computed: every eval set is empty (keys given: {sorted(sets)}); an audit with no "
                         "metrics must not look like a passing one")

    for key, items in sets.items():
        if items and key not in datasets:
            datasets[key] = dataset_ref(items)

    raw_file = None
    if completions_path is not None:
        raw_file = str(save_jsonl(raw, Path(completions_path)))

    if model_ref is None:
        model_ref = ModelRef(path=getattr(getattr(model, "config", None), "_name_or_path", None) or None)
    elif isinstance(model_ref, dict):
        model_ref = ModelRef(**model_ref)

    chat_template = getattr(tok, "chat_template", None)
    generation = {
        "decoding": "greedy", "do_sample": False,
        "max_new_tokens": {"asr": max_new_tokens, "over_refusal": overrefusal_max_new_tokens, "task": task_max_new_tokens},
        "batch_size": {"asr": batch_size, "over_refusal": batch_size, "task": batch_size, "general": max(1, batch_size // 2),
                       "judge": judge.batch_size},
        "system_prompt": "template default",
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest() if isinstance(chat_template, str) else None,
        "task_prompt_suffix": GSM8K_SUFFIX,
        "general_scoring": "argmax of next-token logits over " + "/".join(MC_LETTERS),
        "model_dtype": str(getattr(model, "dtype", "")).removeprefix("torch.") or None,
    }
    return AuditReport(
        model=model_ref,
        suite=suite,
        metrics=metrics,
        judge={**judge.info(), "over_refusal_detector": "string-match"},
        datasets=datasets,
        generation=generation,
        statistics={"ci": "percentile bootstrap over items", "level": 0.95, "n_boot": n_boot, "seed": seed},
        redaction=Redaction(raw_completions_file=raw_file),
        cost={"gpu_seconds": total["seconds"], "per_metric_seconds": seconds},
        provenance=provenance(seed=seed),
    )


# ---------------------------------------------------------------------------------------------
# CLI: `safetypulse audit` (wired into cli.py)
# ---------------------------------------------------------------------------------------------

DTYPES = ("auto", "float32", "bfloat16", "float16")


def add_arguments(p) -> None:
    p.add_argument("--model", required=True, help="model dir, Hub id, or LoRA checkpoint dir")
    p.add_argument("--base-model", help="base model for a LoRA checkpoint (default: the adapter config's base model)")
    p.add_argument("--revision", help="Hub revision of --model (or of --base-model for a LoRA checkpoint)")
    p.add_argument("--suite", default="default", choices=sorted(SUITES))
    p.add_argument("--data-dir", required=True, help="directory with eval_harm/overrefusal/task/general.jsonl")
    p.add_argument("--split", default="test", choices=("dev", "test", "all"), help="items of this split only")
    p.add_argument("--limit", type=int, help="first N items of each set (smoke runs)")
    p.add_argument("--judge", default="llamaguard", choices=JUDGE_KINDS)
    p.add_argument("--judge-model", default=DEFAULT_GUARD, help="Llama Guard local dir or Hub id")
    p.add_argument("--judge-revision")
    p.add_argument("--out", required=True, help="write the AuditReport JSON here")
    p.add_argument("--dtype", choices=DTYPES, default="auto", help="auto: bf16/fp16 on GPU, fp32 on CPU")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256, help="ASR generation budget (protocol §2)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-completions", metavar="PATH",
                   help="ALSO write raw (unredacted) completions to this local jsonl; never share it")


def run(args) -> int:
    from .activations import remove_hooks
    from .models import adapter_base_model, is_adapter_dir, load_model
    from .repair import attach_saved_steering
    from .utils import set_seed

    set_seed(args.seed)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    sets, refs = load_suite(args.data_dir, args.suite, None if args.split == "all" else args.split, args.limit)
    hooks, steering = [], None
    if Path(args.model).is_dir() and is_adapter_dir(args.model):   # LoRA dir: merged into its base model
        base = args.base_model or adapter_base_model(args.model)
        model, tok = load_model(base, dtype=dtype, adapter=args.model, revision=args.revision)
    else:
        base = None
        model, tok = load_model(args.model, dtype=dtype, revision=args.revision)
        hooks, steering = attach_saved_steering(model, args.model)   # an R6 repair output: re-attach its hook
    try:
        judge = Judge(args.judge, args.judge_model, revision=args.judge_revision)
        report = audit(model, tok, judge, sets, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                       n_boot=args.n_boot, seed=args.seed, datasets=refs, suite=args.suite,
                       model_ref=ModelRef(path=str(args.model), base_model=base, revision=args.revision,
                                          steering=steering),
                       completions_path=args.save_completions)
    finally:
        remove_hooks(hooks)
    report.save(args.out)
    # Metric values only: the report never contains prompt or completion text.
    print(json.dumps({"metrics": {k: {"value": m.value, "ci95": list(m.ci95), "n": m.n} for k, m in report.metrics.items()},
                      "judge": judge.name, "out": args.out}))
    if args.save_completions:
        print(f"raw completions written to {args.save_completions}: unredacted model output, keep it local",
              file=sys.stderr)
    return 0

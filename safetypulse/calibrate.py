"""`calibrate`: locate a base model's refusal mechanism and write the probe bundle (docs/03 §3.2).

Library port of notebooks 05 (refusal directions, choice of l*) and 06 (circuit discovery, Table A validation,
base-model reference values). Numerics follow the notebooks: same core functions, same defaults, same rules.

Contrast pairs: rows {"harmful": str, "harmless": str, "split": "calib" | "val" | "probe", ...} (notebook 03's
pairs.jsonl). calib -> directions and circuit discovery; val -> layer choice and validation; probe -> reference
values and the bundle's probe_pairs.jsonl. Prompts are data: nothing here prints or stores them outside the
bundle's probe set, and validation generations are never saved.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .activations import (
    ablate_components,
    ablate_direction,
    activation_patching_scores,
    add_direction,
    attribution_scores,
    component_means,
    hooked,
    mean_diff,
    parse_component,
    random_components_like,
    random_unit_like,
    resid_last,
)
from .cis import base_reference, split_pairs
from .models import model_fingerprint, n_resid
from .prompts import default_token_sets, refusal_logit_diff
from .schemas import EXIT_ERROR, ProbeBundle, Threshold
from .utils import load_json, load_jsonl, progress, provenance, set_seed, sha256_file, sha256_rows

SPLITS = ("calib", "val", "probe")


@dataclass
class CalibrateConfig:
    """Knobs of notebooks 05/06; defaults are the notebooks' full-run values."""

    n_attr_pairs: int = 100       # calib pairs used for circuit discovery (06 N_ATTR_PAIRS); directions use all calib
    n_candidates: int = 40        # top attribution scores re-checked by exact activation patching
    top_k: int = 20               # components kept in the circuit (positive exact effect only)
    max_layer_frac: float = 0.8   # l* searched over resid 1..int(0.8 L) (Arditi et al.: skip the last layers)
    seed: int = 0
    batch_size: int = 8           # forward passes; attribution uses max(1, batch_size // 4) as cis.base_reference
    validate: bool = False        # Table A (generation-based, slow)
    n_random: int = 10            # random directions / component sets in Table A
    gen_tokens: int = 64          # greedy tokens per validation generation

    def __post_init__(self):
        if not 0 < self.max_layer_frac <= 1:
            raise ValueError(f"max_layer_frac must be in (0, 1], got {self.max_layer_frac}")
        for name in ("n_attr_pairs", "n_candidates", "top_k", "batch_size", "n_random", "gen_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def load_pairs(pairs: Sequence[dict] | str | Path) -> dict[str, list[dict]]:
    """Contrast pairs (list or jsonl path) grouped by split: {"calib": [...], "val": [...], "probe": [...]}."""
    rows = load_jsonl(pairs) if isinstance(pairs, (str, Path)) else list(pairs)
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for i, row in enumerate(rows):
        missing = [k for k in ("harmful", "harmless", "split") if k not in row]
        if missing:
            raise ValueError(f"pair {i} lacks {missing}; pairs need harmful/harmless/split (notebook 03 format)")
        if row["split"] in by_split:
            by_split[row["split"]].append(row)
    for s in SPLITS:
        if not by_split[s]:
            raise ValueError(f"no pairs with split={s!r} (need calib, val and probe)")
    return by_split


def resolve_token_sets(token_sets: dict | str | Path | None, tok, model_name: str | None = None) -> dict:
    """{"refuse": [ids], "comply": [ids]} from a dict, a token_sets.json (flat, or keyed by model name as notebook
    04 writes it), or None for prompts.default_token_sets."""
    if token_sets is None:
        return default_token_sets(tok)
    if isinstance(token_sets, (str, Path)):
        token_sets = load_json(token_sets)
    if "refuse" not in token_sets:
        if model_name in token_sets:
            token_sets = token_sets[model_name]
        elif len(token_sets) == 1:
            token_sets = next(iter(token_sets.values()))
        else:
            raise ValueError(f"token sets are keyed by model and none matches {model_name!r}: {sorted(token_sets)}")
    ts = {"refuse": [int(i) for i in token_sets["refuse"]], "comply": [int(i) for i in token_sets["comply"]]}
    if not ts["refuse"] or not ts["comply"]:
        raise ValueError("token sets need non-empty refuse and comply lists")
    return ts


# ---------------------------------------------------------------------------------------------
# Notebook 05: refusal directions and l*
# ---------------------------------------------------------------------------------------------


def refusal_directions(model, tok, calib: Sequence[dict], batch_size: int = 8) -> torch.Tensor:
    """Float[Tensor, "resid d_model"] mean(harmful) - mean(harmless) last-token residuals at every resid position."""
    harm, harmless = split_pairs(calib)
    return mean_diff(resid_last(model, tok, harm, batch_size), resid_last(model, tok, harmless, batch_size))


def layer_candidates(n_layers: int, max_layer_frac: float = 0.8) -> list[int]:
    """Resid positions searched for l*: 1..int(max_layer_frac * n_layers)."""
    return list(range(1, max(1, int(max_layer_frac * n_layers)) + 1))


def choose_layer(layer_scores: Sequence[dict], base_harmless: float) -> int:
    """l*: lowest refusal on harmful prompts with r ablated, among layers where adding r to harmless prompts
    raises their refusal above baseline (all layers if none does)."""
    ok = [s for s in layer_scores if s["induced_harmless"] > base_harmless] or list(layer_scores)
    return min(ok, key=lambda s: s["ablated_harmful"])["resid"]


def select_layer(model, tok, val: Sequence[dict], directions: torch.Tensor, token_sets: dict,
                 max_layer_frac: float = 0.8, batch_size: int = 8) -> dict:
    """Ablation / addition effect of each candidate direction on the mean refusal logit difference of the val split.

    Returns {"layer": l*, "layer_scores": [{"resid", "ablated_harmful", "induced_harmless"}],
    "baseline_logit_diff": {"harmful", "harmless"}}.
    """
    val_h, val_s = split_pairs(val)
    base_h = float(refusal_logit_diff(model, tok, val_h, token_sets, batch_size).mean())
    base_s = float(refusal_logit_diff(model, tok, val_s, token_sets, batch_size).mean())
    scores = []
    for i in progress(layer_candidates(n_resid(model) - 1, max_layer_frac), desc="layers", disable=None):
        with hooked(ablate_direction(model, directions[i])):
            ablated = float(refusal_logit_diff(model, tok, val_h, token_sets, batch_size).mean())
        with hooked(add_direction(model, i, directions[i], 1.0)):
            induced = float(refusal_logit_diff(model, tok, val_s, token_sets, batch_size).mean())
        scores.append({"resid": i, "ablated_harmful": ablated, "induced_harmless": induced})
    return {"layer": choose_layer(scores, base_s), "layer_scores": scores,
            "baseline_logit_diff": {"harmful": base_h, "harmless": base_s}}


# ---------------------------------------------------------------------------------------------
# Notebook 06: circuit discovery
# ---------------------------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties (as pandas' corr(method="spearman"))."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        for val in np.unique(v):
            tie = v == val
            r[tie] = r[tie].mean()
        return r
    if len(x) < 2:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def discover_components(model, tok, calib: Sequence[dict], token_sets: dict, n_candidates: int = 40, top_k: int = 20,
                        batch_size: int = 8) -> dict:
    """Attribution patching over every head/MLP (clean = harmful, corrupt = harmless), exact activation patching
    of the top `n_candidates`, then the top `top_k` by exact effect among those with a positive effect.

    Returns {"components": [ids, best first], "candidates": [{"id", "attribution", "exact"}] sorted by exact,
    "spearman_attr_vs_exact": float}. "components" is empty if no candidate raises refusal when patched.
    """
    harm, harmless = split_pairs(calib)
    attr = attribution_scores(model, tok, harm, harmless, token_sets, batch_size=max(1, batch_size // 4))
    candidates = [c for c, _ in sorted(attr.items(), key=lambda kv: -kv[1])[:n_candidates]]
    exact = activation_patching_scores(model, tok, harm, harmless, candidates, token_sets, batch_size)
    ranked = sorted(candidates, key=lambda c: -exact[c])
    return {
        "components": [c for c in ranked if exact[c] > 0][:top_k],
        "candidates": [{"id": c, "attribution": attr[c], "exact": exact[c]} for c in ranked],
        "spearman_attr_vs_exact": spearman([attr[c] for c in candidates], [exact[c] for c in candidates]),
    }


# ---------------------------------------------------------------------------------------------
# Table A validation (generation-based)
# ---------------------------------------------------------------------------------------------

def is_degenerate(text: str) -> bool:
    """Crude incoherence check for ablation runs: very short, or dominated by repeated words."""
    words = text.split()
    return len(words) < 5 or len(set(words)) / len(words) < 0.3


def _rate(flags: Sequence[bool]) -> float:
    return float(np.mean(flags)) if len(flags) else float("nan")


def validate_mechanism(model, tok, val: Sequence[dict], calib_harmless: Sequence[str], directions: torch.Tensor,
                       layer: int, components: Sequence[str], config: CalibrateConfig) -> dict:
    """Table A: refusal and degenerate-text rates on val harmful prompts with the refusal direction ablated vs
    random directions, and with the circuit mean-ablated vs random components of the same head/MLP mix.

    Components are mean-ablated with means over `calib_harmless` (zero-ablating many MLPs breaks any model, so the
    random control would say nothing); a zero-ablation row is kept for reference. Returns {"table_a": rows,
    "harmless_refusal_with_r_ablated", "direction_verdict", "circuit_verdict"} (PASS / CHECK as in 05/06).
    """
    from .audit import generate  # same greedy decoding and string refusal detector as audit / sp_utils
    from .judge import is_refusal

    val_h, val_s = split_pairs(val)

    def rates(handles=None, prompts=val_h):
        with hooked(handles or []):
            out = generate(model, tok, prompts, config.gen_tokens, config.batch_size, desc="validation")
        return _rate([is_refusal(o) for o in out]), _rate([is_degenerate(o) for o in out])

    def row(name, results):
        r = {"ablation": name, "refusal_rate": float(np.mean([x for x, _ in results])),
             "degenerate": float(np.mean([d for _, d in results]))}
        if len(results) > 1:
            r["std"] = float(np.std([x for x, _ in results]))
        return r

    gen = torch.Generator().manual_seed(config.seed)
    r = directions[layer]
    table = [row("none", [rates()]),
             row(f"refusal direction (resid {layer})", [rates(ablate_direction(model, r))]),
             row(f"random directions (n={config.n_random})",
                 [rates(ablate_direction(model, random_unit_like(r, gen))) for _ in range(config.n_random)])]
    # Side effect: removing r should not make the model refuse safe prompts.
    over_dir, _ = rates(ablate_direction(model, r), val_s)

    def beats(target, rand):   # gate rule of 05/06: well below the random-ablation mean
        return target["refusal_rate"] < rand["refusal_rate"] - 2 * rand.get("std", 0.0) - 0.1

    verdicts = {"direction_verdict": "PASS" if beats(table[1], table[2]) else "CHECK", "circuit_verdict": None}
    if components:
        k = len(components)
        means = component_means(model, tok, calib_harmless, config.batch_size)
        rng = random.Random(config.seed)
        circ = row(f"discovered components, mean-ablated (k={k})", [rates(ablate_components(model, components, means))])
        rand = row(f"random components, mean-ablated (k={k}, n={config.n_random})",
                   [rates(ablate_components(model, random_components_like(components, model, rng), means))
                    for _ in range(config.n_random)])
        zero = row(f"discovered components, zero-ablated (k={k})", [rates(ablate_components(model, components))])
        table += [circ, rand, zero]
        verdicts["circuit_verdict"] = "PASS" if beats(circ, rand) and circ["degenerate"] < 0.2 else "CHECK"
    for t in table:
        t["delta_vs_none"] = t["refusal_rate"] - table[0]["refusal_rate"]
    return {"table_a": table, "harmless_refusal_with_r_ablated": over_dir, **verdicts}


# ---------------------------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------------------------


def calibrate(model, tok, pairs: Sequence[dict] | str | Path, token_sets: dict | str | Path | None = None,
              config: CalibrateConfig | None = None, *, base_model: str | None = None, revision: str | None = None,
              out_dir: str | Path | None = None) -> ProbeBundle:
    """Directions, l*, circuit, optional Table A and probe-set reference values -> ProbeBundle (with tensors).

    model: fp32 base model (attribution needs gradients). base_model: the name `check` later loads LoRA checkpoints
    onto (default: model.config._name_or_path). With `out_dir`, the bundle and its probe_pairs.jsonl are written
    there. Thresholds stay null (UNCALIBRATED) until set_thresholds() is run with values from monitoring dev runs.
    If no component has a positive patching effect the bundle is direction-only (no cis_circ) and says so.
    """
    config = config or CalibrateConfig()
    set_seed(config.seed)
    pairs_sha = sha256_file(pairs) if isinstance(pairs, (str, Path)) else sha256_rows(pairs)
    splits = load_pairs(pairs)
    calib, val, probe = splits["calib"], splits["val"], splits["probe"]
    base_model = base_model or getattr(getattr(model, "config", None), "_name_or_path", None)
    if not base_model:
        raise ValueError("pass base_model= (the name check() loads LoRA checkpoints onto)")
    token_sets = resolve_token_sets(token_sets, tok, base_model)

    directions = refusal_directions(model, tok, calib, config.batch_size)
    sel = select_layer(model, tok, val, directions, token_sets, config.max_layer_frac, config.batch_size)
    layer = sel["layer"]

    calib_attr = calib[: config.n_attr_pairs]
    disc = discover_components(model, tok, calib_attr, token_sets, config.n_candidates, config.top_k, config.batch_size)
    components = disc["components"]
    if not components:
        print("calibrate: no component raises refusal when patched; writing a direction-only bundle (no cis_circ)",
              file=sys.stderr)

    extra = {}
    if config.validate:
        extra["validation"] = validate_mechanism(model, tok, val, [p["harmless"] for p in calib_attr], directions,
                                                 layer, components, config)
        extra["table_a"] = extra["validation"].pop("table_a")

    ref = base_reference(model, tok, probe, directions, layer, token_sets, components, config.batch_size)
    exact = {c["id"]: c["exact"] for c in disc["candidates"]}
    comps = []
    for c in components:
        kind, lay, idx = parse_component(c)
        comps.append({"id": c, "type": kind, "layer": lay, "index": idx, "score": ref["component_scores"][c],
                      "discovery_exact": exact[c]})
    bundle = ProbeBundle.model_validate({
        "base_model": {"name": str(base_model), "revision": revision, "fingerprint": model_fingerprint(model, tok)},
        "layer": layer,
        "components": comps,
        "token_sets": token_sets,
        "reference": ref["reference"],
        "threshold": {"metric": None, "tau_warn": None, "tau_fail": None},
        # Same sha256 / count that save(probe_pairs=probe) writes, so an unsaved bundle can verify its probe set too.
        "probe_set": {"path": "probe_pairs.jsonl", "sha256": sha256_rows(probe), "n_pairs": len(probe)},
        "layer_selection": {"baseline_logit_diff": sel["baseline_logit_diff"], "layer_scores": sel["layer_scores"]},
        "discovery": {"method": "attribution+exact patching (last token)", "matched_pairs": calib[0].get("matched"),
                      "n_pairs": len(calib_attr), "n_components": len(components),
                      "spearman_attr_vs_exact": disc["spearman_attr_vs_exact"], "candidates": disc["candidates"]},
        **extra,
        "provenance": provenance(config.seed, config=asdict(config), pairs_sha256=pairs_sha,
                                 n_pairs={s: len(v) for s, v in splits.items()},
                                 dtype=str(next(model.parameters()).dtype).removeprefix("torch.")),
    }).with_tensors(directions, ref["probe_diff"])
    if out_dir is not None:
        bundle.save(out_dir, probe_pairs=probe)
    return bundle


def set_thresholds(bundle_dir: str | Path, metric: str, tau_warn: float | None, tau_fail: float | None) -> ProbeBundle:
    """Set threshold.metric/tau_warn/tau_fail (validated) on a saved bundle and rewrite it in place."""
    bundle = ProbeBundle.load(bundle_dir)
    threshold = Threshold(metric=metric, tau_warn=tau_warn, tau_fail=tau_fail)
    if not threshold.calibrated:
        raise ValueError("give at least one of tau_warn / tau_fail")
    if problem := bundle.metric_problem(metric):
        raise ValueError(problem)
    bundle.threshold = threshold
    bundle.save(bundle.root)
    return bundle


# ---------------------------------------------------------------------------------------------
# CLI: `safetypulse calibrate`
# ---------------------------------------------------------------------------------------------


def _tau(s: str) -> float | None:
    """A threshold from the command line: a finite number, or none / null / - for no threshold."""
    if s.lower() in ("none", "null", "-"):
        return None
    v = float(s)
    if not math.isfinite(v):
        raise ValueError(f"threshold must be a finite number or 'none', got {s!r}")
    return v


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = CalibrateConfig()
    parser.add_argument("--model", help="base model dir or Hub id (omit to only --set-thresholds)")
    parser.add_argument("--revision", help="Hub revision of --model")
    parser.add_argument("--pairs", help="contrast pairs jsonl with harmful/harmless/split (calib, val, probe)")
    parser.add_argument("--token-sets", help="token_sets.json (default: built-in refusal/compliance start tokens)")
    parser.add_argument("--out", required=True, help="bundle directory to write (or to update with --set-thresholds)")
    parser.add_argument("--validate", action="store_true", help="also run Table A (generation; slow)")
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32",
                        help="model dtype (fp32 recommended: attribution needs gradients)")
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--n-attr-pairs", type=int, default=d.n_attr_pairs)
    parser.add_argument("--n-candidates", type=int, default=d.n_candidates)
    parser.add_argument("--top-k", type=int, default=d.top_k)
    parser.add_argument("--max-layer-frac", type=float, default=d.max_layer_frac)
    parser.add_argument("--n-random", type=int, default=d.n_random)
    parser.add_argument("--gen-tokens", type=int, default=d.gen_tokens)
    parser.add_argument("--circuitkit-crosscheck", action="store_true",
                        help="also run CircuitKIT EAP-IG discovery and write circuitkit_crosscheck.json (needs circuitkit)")
    parser.add_argument("--set-thresholds", nargs=3, metavar=("METRIC", "TAU_WARN", "TAU_FAIL"),
                        help="set thresholds on the bundle in --out (tau may be 'none'); runs after calibration if any")


def config_from_args(args) -> CalibrateConfig:
    return CalibrateConfig(n_attr_pairs=args.n_attr_pairs, n_candidates=args.n_candidates, top_k=args.top_k,
                           max_layer_frac=args.max_layer_frac, seed=args.seed, batch_size=args.batch_size,
                           validate=args.validate, n_random=args.n_random, gen_tokens=args.gen_tokens)


def run(args) -> int:
    """`safetypulse calibrate`. Prints a JSON summary (no prompt text); 0 on success, 1 on error."""
    if not args.model and not args.set_thresholds:
        print("safetypulse calibrate: give --model and --pairs, and/or --set-thresholds", file=sys.stderr)
        return EXIT_ERROR
    summary: dict = {"out": args.out}
    if args.model:
        if not args.pairs:
            print("safetypulse calibrate: --pairs is required with --model", file=sys.stderr)
            return EXIT_ERROR
        from .models import free, load_model

        model, tok = load_model(args.model, dtype=getattr(torch, args.dtype), revision=args.revision)
        config = config_from_args(args)
        bundle = calibrate(model, tok, args.pairs, args.token_sets, config, base_model=args.model,
                           revision=args.revision, out_dir=args.out)
        summary.update(layer=bundle.layer, components=[c.id for c in bundle.components],
                       reference=bundle.reference.model_dump(), validation=getattr(bundle, "validation", None))
        if args.circuitkit_crosscheck:
            from .adapters.circuitkit import crosscheck

            calib = load_pairs(args.pairs)["calib"][: config.n_attr_pairs]
            free(model)
            del model
            result = crosscheck(args.model, calib, bundle.token_sets.model_dump(), bundle,
                                k=max(1, len(bundle.components)), seed=config.seed)
            path = Path(args.out) / "circuitkit_crosscheck.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary["circuitkit_crosscheck"] = {k: result[k] for k in ("overlap_at_k", "jaccard_at_k", "spearman_abs")}
    if args.set_thresholds:
        metric, warn, fail = args.set_thresholds
        th = set_thresholds(args.out, metric, _tau(warn), _tau(fail)).threshold
        summary["threshold"] = th.model_dump()
    print(json.dumps(summary, default=str))
    return 0

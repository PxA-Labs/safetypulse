"""`repair`: weight-space repair of a drifted checkpoint toward the base model (docs/02 §3.5, docs/03 §3.4).

Methods, exactly as notebook 09 applies them (kaggle/lib/sp_utils.py "Weight-space repair"):

    R1 circuit-restore   W[s] <- W[s] + α (W_base[s] - W[s])   on the bundle components' parameter slices s
    R2 circuit-project   W <- W_base + (I - α r̂ r̂ᵀ)(W - W_base) on the components' residual writers (o_proj cols,
                         down_proj), r = the bundle's direction at l*: removes the fine-tune delta's write along r̂
    R3 safetune-*        SafeTune weight-space baselines (safetypulse.adapters.safetune)
    R4 full-interp       R1 on every parameter
    R5 random            R1 on random components with the same number of heads and MLPs (one draw per seed)
    R6 steer             add α·r to the residual stream at l* at inference (forward hook, no weight change)

`base_state` is `models.state_dict_cpu(base_model)` (parameter name -> tensor). The model is edited in place.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import PrivateAttr

from .activations import add_direction, parse_component, random_components_like, remove_hooks, unit
from .models import head_geometry, layers_prefix
from .schemas import SCHEMA_VERSION, CheckpointRef, ProbeBundle, _Open
from .utils import load_json, provenance, save_json

LOG_FILE = "repair_log.json"
STEERING_FILE = "steering.safetensors"

# CLI name -> method code (docs/02 §3.5). R3 names are the SafeTune adapter's methods.
METHODS = {
    "circuit-restore": "R1",
    "circuit-project": "R2",
    "full-interp": "R4",
    "random": "R5",
    "steer": "R6",
    "safetune-wise-ft": "R3",
    "safetune-wise-ft-circuit": "R3",
    "safetune-svr": "R3",
}
_CODE_ALIASES = {"R1": "circuit-restore", "R2": "circuit-project", "R4": "full-interp", "R5": "random", "R6": "steer"}

# Notebook 09's dev protocol: strength grid per method, objective weights (fixed before testing), R5 draws.
DEV_GRID = {"R1": [0.25, 0.5, 0.75, 1.0], "R2": [0.5, 1.0], "R4": [0.1, 0.25, 0.5], "R6": [0.25, 0.5, 1.0]}
LAMBDA, MU = 1.0, 1.0
N_RANDOM = 5

Slices = list[tuple[str, tuple]]


# ---------------------------------------------------------------------------------------------
# Parameter slices (ported from sp_utils)
# ---------------------------------------------------------------------------------------------


def component_slices(model, cid: str) -> Slices:
    """(parameter name, index) pairs owned by a component.

    attn head h: its rows of q_proj (+bias) and its columns of o_proj. k/v are shared across a GQA
    group, so they are left untouched. mlp: gate/up/down projections whole.
    """
    kind, layer, head = parse_component(cid)
    pre = f"{layers_prefix(model)}.{layer}"
    names = dict(model.named_parameters())
    if kind == "attn_head":
        _, hd = head_geometry(model)
        rows = slice(head * hd, (head + 1) * hd)
        out = [(f"{pre}.self_attn.q_proj.weight", (rows, slice(None))), (f"{pre}.self_attn.o_proj.weight", (slice(None), rows))]
        if f"{pre}.self_attn.q_proj.bias" in names:
            out.append((f"{pre}.self_attn.q_proj.bias", (rows,)))
        return out
    return [(f"{pre}.mlp.{p}.weight", (slice(None),)) for p in ("gate_proj", "up_proj", "down_proj")]


def writer_slices(model, cid: str) -> Slices:
    """The parts of a component that write into the residual stream (o_proj cols / down_proj)."""
    return [s for s in component_slices(model, cid) if "o_proj" in s[0] or "down_proj" in s[0]]


def circuit_slices(model, components: Iterable[str]) -> Slices:
    return sum((component_slices(model, c) for c in components), [])


def circuit_writer_slices(model, components: Iterable[str]) -> Slices:
    return sum((writer_slices(model, c) for c in components), [])


def all_param_slices(model) -> Slices:
    return [(n, (slice(None),)) for n, _ in model.named_parameters()]


# ---------------------------------------------------------------------------------------------
# Weight edits (ported from sp_utils, same numerics)
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def interpolate_toward_base(model, base_sd: dict, slices: Slices, alpha: float) -> int:
    """W[idx] <- W[idx] + alpha (W_base[idx] - W[idx]). alpha=0 unchanged, alpha=1 base. Returns #params changed."""
    params = dict(model.named_parameters())
    changed = 0
    for name, idx in slices:
        p = params[name]
        b = base_sd[name].to(p.device, p.dtype)
        p[idx] += alpha * (b[idx] - p[idx])
        changed += p[idx].numel()
    return changed


@torch.no_grad()
def project_delta(model, base_sd: dict, slices: Slices, r: torch.Tensor, beta: float = 1.0) -> int:
    """Remove (fraction beta of) the fine-tuning delta's write along r̂ for residual-writer slices:
    W <- W_base + (I - beta r̂ r̂ᵀ)(W - W_base). Output dim of o_proj/down_proj is d_model."""
    params = dict(model.named_parameters())
    changed = 0
    for name, idx in slices:
        p = params[name]
        r_hat = unit(r.float()).to(p.device)
        b = base_sd[name].to(p.device, torch.float32)
        delta = p[idx].float() - b[idx]
        delta = delta - beta * torch.outer(r_hat, r_hat @ delta)
        p[idx] = (b[idx] + delta).to(p.dtype)
        changed += p[idx].numel()
    return changed


@torch.no_grad()
def delta_norm(model, base_sd: dict, slices: Slices | None = None) -> float:
    """‖W - W_base‖ (Frobenius, fp32 differences) over `slices` (default: every parameter)."""
    params = dict(model.named_parameters())
    total = 0.0
    for name, idx in all_param_slices(model) if slices is None else slices:
        p = params[name]
        d = p[idx].float() - base_sd[name].to(p.device, torch.float32)[idx]
        total += float(d.pow(2).sum())
    return math.sqrt(total)


def check_base_state(model, base_sd: dict) -> None:
    """The base snapshot must cover every parameter of the model with the same shapes."""
    for name, p in model.named_parameters():
        if name not in base_sd:
            raise ValueError(f"base state has no parameter {name!r}: checkpoint and base model architectures differ")
        if tuple(base_sd[name].shape) != tuple(p.shape):
            raise ValueError(f"{name}: base shape {tuple(base_sd[name].shape)} != checkpoint shape {tuple(p.shape)}")


# ---------------------------------------------------------------------------------------------
# α selection (docs/02 §3.5): metrics come from `audit`; this only scores them
# ---------------------------------------------------------------------------------------------


def repair_objective(task: float, asr: float, over_refusal: float, lam: float = LAMBDA, mu: float = MU) -> float:
    """task − λ·ASR − μ·over-refusal (higher is better), as notebook 09."""
    return task - lam * asr - mu * over_refusal


def select_strength(rows: Sequence[dict], lam: float = LAMBDA, mu: float = MU, key: str = "param") -> float:
    """The `key` value (strength) of the row with the highest objective. Rows need "task", "asr" and "over_refusal"
    (or a precomputed "objective"). Ties go to the later row, like notebook 09's `sort_values(...).iloc[-1]`."""
    if not rows:
        raise ValueError("no rows to select a strength from")
    best, best_obj = None, -math.inf
    for row in rows:
        obj = row["objective"] if "objective" in row else repair_objective(row["task"], row["asr"], row["over_refusal"],
                                                                           lam, mu)
        if obj >= best_obj:
            best, best_obj = row, obj
    return best[key]


# ---------------------------------------------------------------------------------------------
# RepairLog (docs/03 §4.3)
# ---------------------------------------------------------------------------------------------


class DeltaNorms(_Open):
    """‖W - W_base‖ over the whole model and over the parameters the method edited, before and after."""

    before: float
    after: float
    touched_before: float
    touched_after: float


class Steering(_Open):
    """R6: the hook to re-attach at inference (`apply_steering`). `file` holds the unnormalised direction."""

    resid_index: int
    alpha: float
    direction_norm: float
    file: str = STEERING_FILE


class RepairLog(_Open):
    """What `repair` did: method, strength, scope, size of the edit, inputs' identities and provenance."""

    schema_version: str = SCHEMA_VERSION
    method: str
    code: Literal["R1", "R2", "R3", "R4", "R5", "R6"]
    alpha: float
    seed: int | None = None
    checkpoint: CheckpointRef = CheckpointRef()
    probe_bundle_sha256: str
    circuit: list[str] = []
    components: list[str] = []
    scope: Literal["circuit", "circuit_writers", "all", "random_components", "hook", "safetune"]
    params_changed: int
    tensors_changed: int
    delta_norm: DeltaNorms
    steering: Steering | None = None
    safetune: dict | None = None
    output: str | None = None
    provenance: dict = {}

    _hooks: list = PrivateAttr(default_factory=list)

    @property
    def hooks(self) -> list:
        """Forward-hook handles left on the model (R6 only)."""
        return self._hooks

    def remove_hooks(self) -> None:
        remove_hooks(self._hooks)
        self._hooks = []

    def save(self, path: str | Path) -> Path:
        return save_json(self.model_dump(mode="json"), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RepairLog:
        path = Path(path)
        return cls.model_validate(load_json(path / LOG_FILE if path.is_dir() else path))


# ---------------------------------------------------------------------------------------------
# repair()
# ---------------------------------------------------------------------------------------------


def resolve_method(method: str) -> tuple[str, str]:
    """CLI name or R-code -> (name, code). R3 has several SafeTune variants, so it must be named."""
    name = _CODE_ALIASES.get(method, method)
    if name not in METHODS:
        raise ValueError(f"unknown repair method {method!r}; choose from {sorted(METHODS)} or R1/R2/R4/R5/R6")
    return name, METHODS[name]


def repair(
    model,
    base_state: dict[str, torch.Tensor],
    bundle: ProbeBundle,
    method: str,
    alpha: float,
    *,
    seed: int = 0,
    components: Sequence[str] | None = None,
    checkpoint: CheckpointRef | None = None,
    safetune_kwargs: dict | None = None,
) -> tuple[object, RepairLog]:
    """Apply one repair to `model` (the drifted checkpoint, edited in place) and return (model, RepairLog).

    alpha: strength — α of R1/R4/R5, β of R2, steering scale of R6, SafeTune's alpha for R3.
    seed: R5's draw (random.Random(seed), as notebook 09). components: default the bundle's circuit.
    For R6 the steering hook stays on the model; remove it with `log.remove_hooks()`.
    """
    name, code = resolve_method(method)
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    check_base_state(model, base_state)
    circuit = list(components) if components is not None else [c.id for c in bundle.components]
    if code in ("R1", "R2", "R5") or name == "safetune-wise-ft-circuit":
        if not circuit:
            raise ValueError(f"{name} needs circuit components, but the bundle has none")
    r_star = bundle.directions[bundle.layer]
    touched_comps: list[str] = []
    hooks: list = []
    steering = safetune_info = None

    if code in ("R1", "R5"):
        touched_comps = circuit if code == "R1" else random_components_like(circuit, model, random.Random(seed))
        slices, scope = circuit_slices(model, touched_comps), ("circuit" if code == "R1" else "random_components")
    elif code == "R2":
        touched_comps, slices, scope = circuit, circuit_writer_slices(model, circuit), "circuit_writers"
    elif code == "R4":
        slices, scope = all_param_slices(model), "all"
    elif code == "R6":
        slices, scope = [], "hook"
    else:
        from .adapters import safetune as st

        slices, scope = st.touched_slices(model, name, circuit, **(safetune_kwargs or {})), "safetune"
        touched_comps = circuit if name == "safetune-wise-ft-circuit" else []

    before, touched_before = delta_norm(model, base_state), delta_norm(model, base_state, slices)
    if code in ("R1", "R4", "R5"):
        changed = interpolate_toward_base(model, base_state, slices, alpha)
    elif code == "R2":
        changed = project_delta(model, base_state, slices, r_star, alpha)
    elif code == "R6":
        hooks = add_direction(model, bundle.layer, r_star, alpha)
        changed = 0
        steering = Steering(resid_index=bundle.layer, alpha=alpha, direction_norm=float(r_star.norm()))
    else:
        safetune_info = st.apply(name, model, base_state, alpha, circuit, **(safetune_kwargs or {}))
        changed = sum(base_state[n][idx].numel() for n, idx in slices)

    log = RepairLog(
        method=name, code=code, alpha=alpha, seed=seed if code == "R5" else None,
        checkpoint=checkpoint or CheckpointRef(), probe_bundle_sha256=bundle.sha256(),
        circuit=circuit, components=touched_comps, scope=scope,
        params_changed=changed, tensors_changed=len({n for n, _ in slices}),
        delta_norm=DeltaNorms(before=before, after=delta_norm(model, base_state), touched_before=touched_before,
                              touched_after=delta_norm(model, base_state, slices)),
        steering=steering, safetune=safetune_info, provenance=provenance(seed),
    )
    log._hooks = hooks
    return model, log


# ---------------------------------------------------------------------------------------------
# Loading, saving, steering
# ---------------------------------------------------------------------------------------------


def load_drifted(checkpoint: str | Path, base_model: str | Path, *, revision: str | None = None,
                 dtype: torch.dtype = torch.float32, dev=None):
    """(model, tokenizer, base_state, CheckpointRef) for a LoRA dir (merged into `base_model`, as notebook 09) or a
    full model dir / Hub id (base weights read from `base_model` on CPU, then freed)."""
    from .check import checkpoint_step
    from .models import apply_adapter, free, is_adapter_dir, load_model, state_dict_cpu

    if saved_steering(checkpoint) is not None:
        raise ValueError(f"{checkpoint} is an R6 steering repair: its effect is a forward hook, not part of the weights, "
                         "and a further repair would drop it. Repair the original checkpoint instead")
    ref = CheckpointRef(path=str(checkpoint), step=checkpoint_step(checkpoint), base_model=str(base_model))
    if Path(checkpoint).is_dir() and is_adapter_dir(checkpoint):
        model, tok = load_model(base_model, dtype=dtype, revision=revision, dev=dev)
        base_state = state_dict_cpu(model)
        model = apply_adapter(model, checkpoint)
    else:
        base, _ = load_model(base_model, dtype=dtype, revision=revision, dev="cpu")
        base_state = state_dict_cpu(base)
        del base
        free()
        model, tok = load_model(checkpoint, dtype=dtype, dev=dev)
    return model, tok, base_state, ref


def save_repaired(model, tok, log: RepairLog, out_dir: str | Path, bundle: ProbeBundle | None = None) -> Path:
    """Write the repaired (merged) model as safetensors, the tokenizer, repair_log.json and, for R6, the steering
    direction (pass `bundle`). Steering hooks are not part of the weights: `load_repaired` re-attaches them."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    if tok is not None:
        tok.save_pretrained(out_dir)
    if log.steering is not None:
        if bundle is None:
            raise ValueError("saving a steering repair needs the bundle (its direction is written next to the model)")
        save_file({"direction": bundle.directions[log.steering.resid_index].float().contiguous()},
                  str(out_dir / log.steering.file))
    log.output = str(out_dir)
    log.save(out_dir / LOG_FILE)
    return out_dir


def apply_steering(model, out_dir: str | Path, log: RepairLog | None = None) -> list:
    """Re-attach a saved R6 steering hook; returns the handles ([] if the repair was not steering)."""
    from safetensors.torch import load_file

    log = log or RepairLog.load(out_dir)
    if log.steering is None:
        return []
    direction = load_file(str(Path(out_dir) / log.steering.file))["direction"]
    return add_direction(model, log.steering.resid_index, direction, log.steering.alpha)


def saved_steering(path: str | Path) -> RepairLog | None:
    """The RepairLog of `path` if it is a `save_repaired` output dir whose repair used steering (R6), else None."""
    path = Path(path)
    if not (path.is_dir() and (path / LOG_FILE).is_file()):
        return None
    log = RepairLog.load(path)
    return log if log.steering is not None else None


def attach_saved_steering(model, path: str | Path) -> tuple[list, dict | None]:
    """Loader step shared by check and audit for any model dir they score: if `path` is an R6 repair output, attach
    its steering hook to `model` (the merged weights alone are the unrepaired model). Returns (hook handles, a
    record of what was applied for the report, or None); remove the handles with `activations.remove_hooks`."""
    log = saved_steering(path)
    if log is None:
        return [], None
    handles = apply_steering(model, path, log)
    info = {"applied": True, "method": log.method, "resid_index": log.steering.resid_index, "alpha": log.steering.alpha,
            "direction_norm": log.steering.direction_norm, "repair_log": str(Path(path) / LOG_FILE)}
    return handles, info


def load_repaired(out_dir: str | Path, dtype: torch.dtype = torch.float32, dev=None):
    """(model, tokenizer, RepairLog) from `save_repaired`, with any steering hook attached (`log.hooks`)."""
    from .models import load_model

    model, tok = load_model(out_dir, dtype=dtype, dev=dev)
    log = RepairLog.load(out_dir)
    log._hooks = apply_steering(model, out_dir, log)
    return model, tok, log


# ---------------------------------------------------------------------------------------------
# CLI: `safetypulse repair`
# ---------------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", required=True, help="drifted LoRA checkpoint dir, full model dir, or Hub id")
    parser.add_argument("--bundle", required=True, help="probe bundle dir (or its probe_bundle.json)")
    parser.add_argument("--base-model", help="base model whose weights are restored (default: the bundle's base_model)")
    parser.add_argument("--method", required=True, choices=sorted(METHODS) + sorted(_CODE_ALIASES),
                        help="R1 circuit-restore, R2 circuit-project, R4 full-interp, R5 random, R6 steer, R3 safetune-*")
    parser.add_argument("--alpha", type=float, required=True, help="strength (β for circuit-project, scale for steer)")
    parser.add_argument("--out", required=True, help="output dir: merged model (safetensors) + repair_log.json")
    parser.add_argument("--seed", type=int, default=0, help="random-component draw (method random)")
    parser.add_argument("--svr-rank", type=int, default=8, help="SVD rank for safetune-svr (0 = full delta)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    return parser


def run(args) -> int:
    bundle = ProbeBundle.load(args.bundle)
    base = args.base_model or bundle.base_model.name
    revision = None if args.base_model else bundle.base_model.revision
    model, tok, base_state, ref = load_drifted(args.checkpoint, base, revision=revision, dtype=getattr(torch, args.dtype))
    name, _ = resolve_method(args.method)
    kw = {"rank": args.svr_rank or None} if name == "safetune-svr" else None
    model, log = repair(model, base_state, bundle, name, args.alpha, seed=args.seed, checkpoint=ref, safetune_kwargs=kw)
    log.provenance["dtype"] = args.dtype
    save_repaired(model, tok, log, args.out, bundle)
    print(json.dumps({"method": log.method, "code": log.code, "alpha": log.alpha, "params_changed": log.params_changed,
                      "delta_norm": log.delta_norm.model_dump(), "out": log.output}))
    return 0

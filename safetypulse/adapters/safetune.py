"""SafeTune (Lexsi Labs) weight-space recovery methods as R3 baselines (docs/02 §3.5, docs/03 §5).

Coded against safetune 0.1.12 `safetune.recover` (read from its source):

    apply_wise_ft(model, aligned, alpha=0.5, param_filter=None)
        W <- alpha W_aligned + (1 - alpha) W, whole tensors, for parameters with param_filter(name, tensor) True.
    apply_safety_vector_restore(model, aligned, alpha=1.0, rank=8, target_modules=None)
        W <- W + alpha · rank-r SVD truncation of (W_aligned - W) for 2-D tensors (1-D: full delta), on parameters
        whose name contains one of `target_modules` (default: all but embed_tokens / lm_head / norm).

Both mutate `model` in place and only call `aligned.state_dict()`, so the base model is passed as its CPU state
dict (no second model in memory). Our "aligned" reference is the base (instruct) model the fine-tune started from.

Methods:
    safetune-wise-ft          wise_ft on every parameter                 (≈ R4 full-interp, same α)
    safetune-wise-ft-circuit  wise_ft on the circuit components' tensors (= R1 for MLP components; for attention
                              heads it interpolates the whole q_proj/o_proj tensors, i.e. all heads of the layer)
    safetune-svr              safety_vector_restore (SafeTune-original heuristic, rank `rank`, None = full delta)

safetune is optional: imported lazily, with a clear error when missing.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

SVR_DEFAULT_SKIP = ("embed_tokens", "lm_head", "norm")   # safety_vector_restore's own default skip list
METHODS = ("safetune-wise-ft", "safetune-wise-ft-circuit", "safetune-svr")


def _safetune():
    try:
        import safetune
    except ImportError as e:
        raise ImportError("R3 baselines need SafeTune (Lexsi Labs): pip install safetune (Python >= 3.12)") from e
    return safetune


def available() -> bool:
    try:
        _safetune()
    except ImportError:
        return False
    return True


class _Reference:
    """Stands in for the aligned nn.Module: SafeTune's wise_ft / safety_vector_restore only call state_dict()."""

    def __init__(self, state: dict[str, torch.Tensor]):
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state


def _circuit_names(model, components: Sequence[str]) -> list[str]:
    from ..repair import circuit_slices

    return list(dict.fromkeys(n for n, _ in circuit_slices(model, components)))


def _svr_names(model, target_modules: Sequence[str] | None) -> list[str]:
    names = [n for n, _ in model.named_parameters()]
    if target_modules is not None:
        return [n for n in names if any(t in n for t in target_modules)]
    return [n for n in names if not any(s in n for s in SVR_DEFAULT_SKIP)]


def touched_names(model, method: str, components: Sequence[str] = (), target_modules: Sequence[str] | None = None,
                  **_) -> list[str]:
    """Parameter names the SafeTune method edits."""
    if method == "safetune-wise-ft":
        return [n for n, _ in model.named_parameters()]
    if method == "safetune-wise-ft-circuit":
        return _circuit_names(model, components)
    if method == "safetune-svr":
        return _svr_names(model, target_modules)
    raise ValueError(f"unknown SafeTune method {method!r}; choose from {METHODS}")


def touched_slices(model, method: str, components: Sequence[str] = (), **kw) -> list[tuple[str, tuple]]:
    return [(n, (slice(None),)) for n in touched_names(model, method, components, **kw)]


def wise_ft(model, base_state: dict, alpha: float, param_names: Sequence[str] | None = None):
    """safetune.recover.apply_wise_ft toward the base weights, optionally restricted to `param_names`."""
    st = _safetune()
    keep = None if param_names is None else set(param_names)
    flt = None if keep is None else (lambda name, _t: name in keep)
    return st.recover.apply_wise_ft(model, _Reference(base_state), alpha=alpha, param_filter=flt)


def safety_vector_restore(model, base_state: dict, alpha: float, rank: int | None = 8,
                          param_names: Sequence[str] | None = None):
    """safetune.recover.apply_safety_vector_restore toward the base weights. `param_names` (full names, matched as
    substrings by SafeTune) default to SafeTune's own selection (all but embeddings / lm_head / norms)."""
    st = _safetune()
    return st.recover.apply_safety_vector_restore(model, _Reference(base_state), alpha=alpha, rank=rank,
                                                  target_modules=None if param_names is None else list(param_names))


def apply(method: str, model, base_state: dict, alpha: float, components: Sequence[str] = (), *,
          rank: int | None = 8, target_modules: Sequence[str] | None = None) -> dict:
    """Run one SafeTune method on `model` in place; returns what to record in the RepairLog."""
    st = _safetune()
    names = touched_names(model, method, components, target_modules=target_modules)
    if method == "safetune-wise-ft":
        wise_ft(model, base_state, alpha)
        fn, kwargs = "apply_wise_ft", {"alpha": alpha, "param_filter": None}
    elif method == "safetune-wise-ft-circuit":
        wise_ft(model, base_state, alpha, names)
        fn, kwargs = "apply_wise_ft", {"alpha": alpha, "param_filter": f"{len(names)} circuit tensors"}
    else:
        safety_vector_restore(model, base_state, alpha, rank, names)
        fn, kwargs = "apply_safety_vector_restore", {"alpha": alpha, "rank": rank, "target_modules": f"{len(names)} tensors"}
    return {"function": f"safetune.recover.{fn}", "version": getattr(st, "__version__", None), "kwargs": kwargs}

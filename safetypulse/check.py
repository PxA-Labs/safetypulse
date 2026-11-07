"""`check`: score one checkpoint against a probe bundle and return a CheckResult (docs/03 §3.3)."""
from __future__ import annotations

import contextlib
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import torch

from .activations import remove_hooks
from .cis import cis_status, compute_cis
from .models import adapter_base_model, fingerprint_mismatches, is_adapter_dir, load_model, model_fingerprint
from .repair import attach_saved_steering
from .schemas import CheckpointRef, CheckResult, ProbeBundle
from .utils import load_jsonl, provenance, sha256_file, sha256_rows


class ProbeSetMismatch(ValueError):
    """The probe pairs differ from those the bundle's reference values were computed on."""


class BaseModelMismatch(ValueError):
    """The scored model is not built on the base model the bundle was calibrated on."""


@contextlib.contextmanager
def scoring_mode(model):
    """Score `model` in eval mode (dropout off) with parameter gradients disabled, then restore exactly what the
    caller had: train/eval mode, every parameter's requires_grad flag and any accumulated .grad."""
    was_training = model.training
    params = list(model.parameters())
    state = [(p.requires_grad, p.grad) for p in params]
    try:
        model.eval()
        for p in params:
            p.requires_grad_(False)
            p.grad = None
        yield model
    finally:
        for p, (req, grad) in zip(params, state, strict=True):
            p.requires_grad_(req)
            p.grad = grad
        model.train(was_training)


def _same_model_name(a: str | Path, b: str | Path) -> bool:
    """Equal names, or two paths to the same local directory."""
    if str(a) == str(b):
        return True
    pa, pb = Path(a), Path(b)
    return pa.exists() and pb.exists() and pa.resolve() == pb.resolve()


def verify_base_model(bundle: ProbeBundle, model, tok, *, names: dict[str, str] | None = None,
                      allow_mismatch: bool = False) -> None:
    """The scored model must match the base model the bundle was calibrated on.

    names: base-model names to compare with the bundle's (e.g. the adapter config's `base_model_name_or_path`, an
    explicit --base-model); a different name only warns, since local paths and Hub ids of one model differ. The
    fingerprint (architecture + tokenizer, `models.model_fingerprint`) decides: a mismatch raises BaseModelMismatch
    unless `allow_mismatch`, then it warns. A bundle without a fingerprint (notebook bundles) warns only.
    """
    expected_name = bundle.base_model.name
    for source, name in (names or {}).items():
        if name and not _same_model_name(name, expected_name):
            warnings.warn(f"{source} {name!r} differs from the bundle's base model {expected_name!r}; "
                          "relying on the model fingerprint", stacklevel=3)
    expected = bundle.base_model.get_fingerprint()
    if not expected:
        warnings.warn("probe bundle has no base-model fingerprint (written before fingerprints existed): cannot verify "
                      f"that the scored model is built on {expected_name!r}", stacklevel=3)
        return
    diff = fingerprint_mismatches(expected, model_fingerprint(model, tok))
    if diff:
        detail = ", ".join(f"{k}: bundle {e!r} vs model {a!r}" for k, (e, a) in diff.items())
        msg = f"model does not match the bundle's base model {expected_name!r} ({detail})"
        if not allow_mismatch:
            raise BaseModelMismatch(msg + "; pass --allow-base-mismatch / allow_base_mismatch=True to score anyway")
        warnings.warn(msg + " (allowed by allow_base_mismatch)", stacklevel=3)


def checkpoint_step(path: str | Path) -> int | None:
    """Training step from a `checkpoint-<step>` directory name (HF Trainer convention)."""
    m = re.search(r"checkpoint-(\d+)$", Path(path).name)
    return int(m.group(1)) if m else None


def resolve_probe_pairs(bundle: ProbeBundle, probe_pairs: Sequence[dict] | str | Path | None,
                        verify: bool = True) -> list[dict]:
    """Probe pairs from a list, a jsonl path, or (None) the bundle's own probe_set file.

    With `verify`, their sha256 must equal `probe_set.sha256`: the reference values and probe_diff were
    computed on that exact set, so scores on any other set are not comparable to the base model's 1.0. A bundle
    that records no sha256 cannot be verified, which is an error too (pass verify=False to score anyway).
    """
    if probe_pairs is None:
        path = bundle.probe_path()
        if path is None:
            raise ValueError("no probe pairs given and the bundle names no probe_set file")
        probe_pairs = path
    if isinstance(probe_pairs, (str, Path)):
        digest = sha256_file(probe_pairs)
        probe_pairs = load_jsonl(probe_pairs)
    else:
        probe_pairs = list(probe_pairs)
        digest = sha256_rows(probe_pairs)
    if not probe_pairs:
        raise ValueError("probe set is empty")
    expected = bundle.probe_set.sha256 if bundle.probe_set else None
    if verify and not expected:
        raise ProbeSetMismatch("the bundle records no probe_set.sha256, so the probe pairs cannot be verified against "
                               "the set its reference values were computed on; pass --no-verify-probe "
                               "(verify_probe=False) to score anyway")
    if verify and digest != expected:
        raise ProbeSetMismatch(f"probe pairs sha256 {digest[:12]}… does not match the bundle's {expected[:12]}…")
    return probe_pairs


def check(
    model_or_checkpoint,
    bundle: ProbeBundle | str | Path,
    probe_pairs: Sequence[dict] | str | Path | None = None,
    *,
    tokenizer=None,
    base_model: str | Path | None = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 8,
    with_circ: bool = True,
    verify_probe: bool = True,
    step: int | None = None,
    allow_base_mismatch: bool = False,
) -> CheckResult:
    """Compute every CIS variant for one model and its status under the bundle's thresholds.

    model_or_checkpoint: a loaded model (pass `tokenizer`), a full model dir / Hub id, or a LoRA checkpoint dir.
        A LoRA dir is merged into `base_model` (default: the bundle's base model) before scoring. A
        `repair --method steer` output dir gets its saved steering hook attached (recorded in `result.steering`).
        A loaded model is scored in eval mode without parameter gradients; its mode, requires_grad flags and
        .grad are restored afterwards.
    probe_pairs: [{"harmful", "harmless"}], a jsonl path, or None for the bundle's probe_pairs.jsonl.
    dtype: fp32 by default; the circuit attribution needs gradients and is noisy in half precision. Ignored for a
        loaded model (provenance records the dtype its parameters actually have).
    allow_base_mismatch: score even if the model's fingerprint differs from the bundle's base model (warns).

    The status metric is `bundle.threshold.metric`; cis_circ is always computed when that metric needs it. A
    non-finite primary score gives status ERROR (exit code 1), never PASS.
    """
    if not isinstance(bundle, ProbeBundle):
        bundle = ProbeBundle.load(bundle)
    probe_pairs = resolve_probe_pairs(bundle, probe_pairs, verify_probe)
    th = bundle.threshold
    ref = CheckpointRef()
    names: dict[str, str] = {}
    hooks: list = []
    steering = None
    if isinstance(model_or_checkpoint, (str, Path)):
        src = model_or_checkpoint
        ref.path, ref.step = str(src), step if step is not None else checkpoint_step(src)
        if Path(src).is_dir() and is_adapter_dir(src):
            with contextlib.suppress(ValueError):
                names["adapter_config.json base_model_name_or_path"] = adapter_base_model(src)
            if base_model:
                names["--base-model"] = str(base_model)
            base = base_model or bundle.base_model.name
            revision = None if base_model else bundle.base_model.revision
            model, tok = load_model(base, dtype=dtype, adapter=src, revision=revision)
            ref.base_model = str(base)
        else:
            model, tok = load_model(src, dtype=dtype)
            hooks, steering = attach_saved_steering(model, src)
    else:
        if tokenizer is None:
            raise ValueError("pass tokenizer= together with a loaded model")
        model, tok = model_or_checkpoint, tokenizer
        ref.step = step
        ref.path = getattr(getattr(model, "config", None), "_name_or_path", None) or None

    try:
        verify_base_model(bundle, model, tok, names=names, allow_mismatch=allow_base_mismatch)
        # cis_circ needs activation gradients (not parameter gradients), so no torch.no_grad() around compute_cis.
        with scoring_mode(model):
            cis = compute_cis(model, tok, probe_pairs, bundle, with_circ=with_circ or th.metric == "cis_circ",
                              batch_size=batch_size)
    finally:
        remove_hooks(hooks)
    status = cis_status(cis["scores"][th.metric], th.tau_warn, th.tau_fail) if th.calibrated else "UNCALIBRATED"
    param = next(model.parameters(), None)
    actual_dtype = param.dtype if param is not None else dtype
    return CheckResult(
        checkpoint=ref,
        probe_bundle_sha256=bundle.sha256(),
        scores=cis["scores"],
        primary_metric=th.metric,
        tau_warn=th.tau_warn,
        tau_fail=th.tau_fail,
        status=status,
        per_component=cis["per_component"],
        harmful_refusal_logit_mean=cis["harmful_refusal_logit_mean"],
        cost=cis["cost"],
        steering=steering,
        provenance=provenance(dtype=str(actual_dtype).removeprefix("torch."), batch_size=batch_size),
    )

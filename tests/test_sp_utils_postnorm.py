"""kaggle/lib/sp_utils.py component mean-ablation on post-norm ("sandwich") decoder layers (Gemma-2/3).

sp_utils keeps its own copy of component_means / ablate_components (the notebooks load it from a Kaggle dataset,
not from the safetypulse package). These tests mirror tests/test_component_ablation_postnorm.py for that copy:
on Gemma-3 the MLP mean must be measured and inserted at post_feedforward_layernorm (the MLP's residual write),
stale means must be rejected, per-head ablation on post-norm layers must warn, and on Llama/Cohere layouts the
fixed functions must be byte-identical to the old ones (copied below). Also checks last_logits builds no graph.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest
import torch
from conftest import make_model
from test_ablation_postnorm import _cohere, _gemma3, _prompts

_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "lib" / "sp_utils.py"
_spec = importlib.util.spec_from_file_location("_sp_utils_postnorm_under_test", _PATH)
SP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SP)


# --- sp_utils before the fix (hooks o_proj input and mlp output on every layout) -------------------------------

@torch.no_grad()
def _old_component_means(model, tok, prompts, batch_size=8, system=None):
    layers = SP.decoder_layers(model)
    sums = {"attn": [0.0] * len(layers), "mlp": [0.0] * len(layers)}
    mask: list = [None]
    n = 0

    def add(kind, k, x):
        sums[kind][k] = sums[kind][k] + (x.float() * mask[0].to(x.device)[..., None]).sum((0, 1)).cpu()

    handles = []
    for k, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, k=k: add("attn", k, a[0])))
        handles.append(layer.mlp.register_forward_hook(lambda m, i, o, k=k: add("mlp", k, o)))
    try:
        for batch in SP.batched(SP.format_chat(tok, prompts, system), batch_size):
            enc = SP.encode(tok, batch, model.device)
            mask[0] = enc["attention_mask"].float()
            n += int(mask[0].sum())
            model(**enc, logits_to_keep=1)
    finally:
        SP.remove_hooks(handles)
    return {kind: torch.stack(v) / n for kind, v in sums.items()}


def _old_ablate_components(model, components, means=None):
    _, head_dim = SP.head_geometry(model)
    layers = SP.decoder_layers(model)
    heads: dict[int, list[int]] = {}
    mlps: set[int] = set()
    for cid in components:
        kind, layer, head = SP.parse_component(cid)
        if kind == "attn_head":
            heads.setdefault(layer, []).append(head)
        else:
            mlps.add(layer)
    handles = []
    for layer, hs in heads.items():
        dev = layers[layer].self_attn.o_proj.weight.device
        idx = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in hs]).to(dev)
        fill = 0 if means is None else means["attn"][layer].to(dev)[idx]

        def pre(m, args, idx=idx, fill=fill):
            x = args[0].clone()
            x[..., idx] = fill if isinstance(fill, int) else fill.to(x.dtype)
            return (x,) + tuple(args[1:])

        handles.append(layers[layer].self_attn.o_proj.register_forward_pre_hook(pre))
    for layer in mlps:
        if means is None:
            handles.append(layers[layer].mlp.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
        else:
            mu = means["mlp"][layer]
            handles.append(layers[layer].mlp.register_forward_hook(
                lambda m, i, o, mu=mu: mu.to(o.device, o.dtype).expand_as(o).clone()))
    return handles


# --- helpers ---------------------------------------------------------------------------------------------------

def _enc(tok, model):
    return SP.encode(tok, SP.format_chat(tok, _prompts()), model.device)


@torch.no_grad()
def _post_ff_outputs(model, tok, layer, ablation=None):
    """post_feedforward_layernorm output of `layer` (the MLP's residual write) on the reference batch, plus mask."""
    enc = _enc(tok, model)
    got: list = []
    h = SP.decoder_layers(model)[layer].post_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: got.append(o))
    try:
        with SP.hooked(ablation if ablation is not None else []):
            model(**enc)
    finally:
        h.remove()
    return got[0].float(), enc["attention_mask"].bool()


def _residual_write_error(model, tok, layer, means, ablate):
    """max |residual write of the ablated MLP - mean residual write| over real tokens; the mean is measured
    independently of component_means, directly on the post_feedforward_layernorm output."""
    clean, mask = _post_ff_outputs(model, tok, layer)
    mean_write = clean[mask].mean(0)
    ablated, _ = _post_ff_outputs(model, tok, layer, ablate(model, [f"mlp:{layer}"], means))
    return (ablated[mask] - mean_write).abs().max().item()


# --- Gemma-3 (post-norm) -----------------------------------------------------------------------------------------

def test_gemma3_old_mlp_mean_ablation_misses_residual_write(tok):
    """The pre-fix hooks leave the MLP's residual write far from its mean (the bug the fix addresses)."""
    model = _gemma3(len(tok))
    err = _residual_write_error(model, tok, 1, _old_component_means(model, tok, _prompts()), _old_ablate_components)
    assert err > 1e-1, err


def test_gemma3_mlp_mean_ablation_replaces_residual_write(tok):
    model = _gemma3(len(tok))
    means = SP.component_means(model, tok, _prompts())
    for layer in range(len(SP.decoder_layers(model))):
        err = _residual_write_error(model, tok, layer, means, SP.ablate_components)
        assert err < 1e-4, (layer, err)


def test_gemma3_mlp_zero_ablation_removes_residual_write(tok):
    model = _gemma3(len(tok))
    ablated, mask = _post_ff_outputs(model, tok, 2, SP.ablate_components(model, ["mlp:2"]))
    assert torch.count_nonzero(ablated[mask]) == 0


def test_gemma3_stale_means_rejected(tok):
    """Means from the old component_means (raw mlp output) must not be applied after post_feedforward_layernorm."""
    model = _gemma3(len(tok))
    stale = _old_component_means(model, tok, _prompts())
    with pytest.raises(ValueError, match="post_feedforward_layernorm"):
        SP.ablate_components(model, ["mlp:0"], stale)
    for layer in SP.decoder_layers(model):        # nothing left registered after the failure
        assert not layer.post_feedforward_layernorm._forward_hooks
        assert not layer.mlp._forward_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks


def test_gemma3_head_ablation_warns(tok):
    model = _gemma3(len(tok))
    means = SP.component_means(model, tok, _prompts())
    with pytest.warns(UserWarning, match="post_attention_layernorm"):
        SP.remove_hooks(SP.ablate_components(model, ["attn_head:1:0"], means))
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # MLP-only ablation on Gemma must not warn
        SP.remove_hooks(SP.ablate_components(model, ["mlp:1"], means))


# --- Llama / Cohere (pre-norm): byte-identical to the old implementation -----------------------------------------

@pytest.mark.parametrize("build", [make_model, _cohere], ids=["llama", "cohere"])
@pytest.mark.parametrize("use_means", [True, False], ids=["mean", "zero"])
def test_pre_norm_layouts_unchanged(tok, build, use_means):
    model = build(len(tok))
    comps = ["attn_head:0:1", "attn_head:2:0", "attn_head:2:3", "mlp:1", "mlp:3"]
    old_means = _old_component_means(model, tok, _prompts())
    new_means = SP.component_means(model, tok, _prompts())
    for kind in ("attn", "mlp"):
        assert torch.equal(old_means[kind], new_means[kind])
    enc = _enc(tok, model)
    with torch.no_grad():
        with SP.hooked(_old_ablate_components(model, comps, old_means if use_means else None)):
            old = model(**enc).logits
        with warnings.catch_warnings():
            warnings.simplefilter("error")         # no post-norm warning on pre-norm layouts
            handles = SP.ablate_components(model, comps, new_means if use_means else None)
        with SP.hooked(handles):
            new = model(**enc).logits
        clean = model(**enc).logits
    assert torch.equal(old, new)
    assert not torch.equal(old, clean)             # the ablation did something


# --- last_logits runs without autograd -----------------------------------------------------------------------------

def test_last_logits_builds_no_graph(tok):
    model = make_model(len(tok)).requires_grad_(True)       # trainable weights: a graph would be built if allowed
    grad_seen: list = []
    h = model.register_forward_hook(lambda m, i, o: grad_seen.append(torch.is_grad_enabled()))
    try:
        assert torch.is_grad_enabled()
        out = SP.last_logits(model, tok, _prompts(), batch_size=4)
    finally:
        h.remove()
    assert grad_seen and not any(grad_seen)
    assert not out.requires_grad and out.grad_fn is None
    assert out.shape == (len(_prompts()), len(tok))

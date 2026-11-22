"""Component mean-ablation on post-norm ("sandwich") decoder layers (Gemma-2/3).

Gemma-2/3 layers write post_feedforward_layernorm(mlp(x)) to the residual stream, not mlp(x). Mean-ablating
an MLP by replacing mlp(x) with the mean of mlp(x) therefore does not make its residual write equal to the mean
residual write: the RMSNorm re-normalises the replaced value. The fixed component_means / ablate_components
use the post_feedforward_layernorm output on that layout; on Llama/Qwen/Cohere layouts they must give
byte-identical outputs to the old implementation.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from conftest import make_model
from test_ablation_postnorm import _cohere, _gemma3, _prompts

from safetypulse.activations import ablate_components, component_means, hooked, parse_component, remove_hooks
from safetypulse.models import decoder_layers, head_geometry
from safetypulse.prompts import encode, format_chat
from safetypulse.utils import batched


# --- the implementation before the fix (hooks o_proj input and mlp output on every layout) -------------------

@torch.no_grad()
def _old_component_means(model, tok, prompts, batch_size=8, system=None):
    layers = decoder_layers(model)
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
        for batch in batched(format_chat(tok, prompts, system), batch_size):
            enc = encode(tok, batch, model.device)
            mask[0] = enc["attention_mask"].float()
            n += int(mask[0].sum())
            model(**enc, logits_to_keep=1)
    finally:
        remove_hooks(handles)
    return {kind: torch.stack(v) / n for kind, v in sums.items()}


def _old_ablate_components(model, components, means=None):
    _, head_dim = head_geometry(model)
    layers = decoder_layers(model)
    heads: dict[int, list[int]] = {}
    mlps: set[int] = set()
    for cid in components:
        kind, layer, head = parse_component(cid)
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
    return encode(tok, format_chat(tok, _prompts()), model.device)


@torch.no_grad()
def _post_ff_outputs(model, tok, layer, ablation=None):
    """post_feedforward_layernorm output of `layer` (the MLP's residual write) on the reference batch, plus mask."""
    enc = _enc(tok, model)
    got: list = []
    h = decoder_layers(model)[layer].post_feedforward_layernorm.register_forward_hook(lambda m, i, o: got.append(o))
    try:
        with hooked(ablation if ablation is not None else []):
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
    means = component_means(model, tok, _prompts())
    for layer in range(len(decoder_layers(model))):
        err = _residual_write_error(model, tok, layer, means, ablate_components)
        assert err < 1e-4, (layer, err)


def test_gemma3_mlp_zero_ablation_removes_residual_write(tok):
    model = _gemma3(len(tok))
    ablated, mask = _post_ff_outputs(model, tok, 2, ablate_components(model, ["mlp:2"]))
    assert torch.count_nonzero(ablated[mask]) == 0


def test_gemma3_stale_means_rejected(tok):
    """Means from the old component_means (raw mlp output) must not be applied after post_feedforward_layernorm."""
    model = _gemma3(len(tok))
    stale = _old_component_means(model, tok, _prompts())
    with pytest.raises(ValueError, match="post_feedforward_layernorm"):
        ablate_components(model, ["mlp:0"], stale)
    for layer in decoder_layers(model):           # nothing left registered after the failure
        assert not layer.post_feedforward_layernorm._forward_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks


def test_gemma3_head_ablation_warns(tok):
    model = _gemma3(len(tok))
    means = component_means(model, tok, _prompts())
    with pytest.warns(UserWarning, match="post_attention_layernorm"):
        remove_hooks(ablate_components(model, ["attn_head:1:0"], means))
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # MLP-only ablation on Gemma must not warn
        remove_hooks(ablate_components(model, ["mlp:1"], means))


# --- Llama / Cohere (pre-norm): byte-identical to the old implementation -----------------------------------------

@pytest.mark.parametrize("build", [make_model, _cohere], ids=["llama", "cohere"])
@pytest.mark.parametrize("use_means", [True, False], ids=["mean", "zero"])
def test_pre_norm_layouts_unchanged(tok, build, use_means):
    model = build(len(tok))
    comps = ["attn_head:0:1", "attn_head:2:0", "attn_head:2:3", "mlp:1", "mlp:3"]
    old_means = _old_component_means(model, tok, _prompts())
    new_means = component_means(model, tok, _prompts())
    for kind in ("attn", "mlp"):
        assert torch.equal(old_means[kind], new_means[kind])
    enc = _enc(tok, model)
    with torch.no_grad():
        with hooked(_old_ablate_components(model, comps, old_means if use_means else None)):
            old = model(**enc).logits
        with warnings.catch_warnings():
            warnings.simplefilter("error")         # no post-norm warning on pre-norm layouts
            handles = ablate_components(model, comps, new_means if use_means else None)
        with hooked(handles):
            new = model(**enc).logits
        clean = model(**enc).logits
    assert torch.equal(old, new)
    assert not torch.equal(old, clean)             # the ablation did something

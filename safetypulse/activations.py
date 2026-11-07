"""Hooked forward passes, interventions and component attribution.

Residual-stream index convention used everywhere:
    resid index 0      = token embeddings
    resid index i >= 1 = output of decoder layer i-1
so a model with L layers has L+1 resid positions.

Components are identified as "attn_head:<layer>:<head>" or "mlp:<layer>" (0-based decoder layer). Attention
components hook `self_attn.o_proj` and MLP components hook `mlp` (Llama/Qwen/Cohere-style module names).
"""
from __future__ import annotations

import contextlib
import random
from collections.abc import Iterable, Sequence

import torch

from .models import decoder_layers, head_geometry
from .prompts import encode, format_chat, logit_diff_from_logits
from .utils import batched, progress


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


# ---------------------------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------------------------


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()


@contextlib.contextmanager
def hooked(handles):
    """`with hooked(ablate_direction(model, r)): ...` removes the hooks on exit, also on error."""
    try:
        yield handles
    finally:
        remove_hooks(handles)


def _per_device(t: torch.Tensor, dtype: torch.dtype):
    """Cached copies of `t` per device, for models split across GPUs (device_map="auto")."""
    cache: dict = {}

    def get(dev):
        if dev not in cache:
            cache[dev] = t.to(dev, dtype)
        return cache[dev]
    return get


# ---------------------------------------------------------------------------------------------
# Residual stream and directions
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def resid_last(model, tok, prompts: Sequence[str], batch_size: int = 8, system: str | None = None) -> torch.Tensor:
    """Float[Tensor, "n resid d_model"] residual stream at the last prompt token, all resid positions (fp32, CPU).

    Captured with our own hooks (registered after any intervention hooks, so they see the
    intervened stream) rather than output_hidden_states, whose last entry is post-norm.
    """
    layers = decoder_layers(model)
    store: list = [None] * (len(layers) + 1)
    handles = [model.get_input_embeddings().register_forward_hook(lambda m, i, o: store.__setitem__(0, o[:, -1]))]
    for k, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(k + 1, _hidden(o)[:, -1])))
    texts = format_chat(tok, prompts, system)
    chunks = []
    try:
        for batch in batched(texts, batch_size):
            model(**encode(tok, batch, model.device), logits_to_keep=1)
            chunks.append(torch.stack([x.float().cpu() for x in store], dim=1))
    finally:
        remove_hooks(handles)
    return torch.cat(chunks)


def mean_diff(acts_a: torch.Tensor, acts_b: torch.Tensor) -> torch.Tensor:
    """Float[Tensor, "resid d_model"] mean(acts_a) - mean(acts_b) over prompts."""
    return acts_a.mean(0) - acts_b.mean(0)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def random_unit_like(v: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return unit(torch.randn(v.shape, generator=generator))


def ablate_direction(model, direction: torch.Tensor) -> list:
    """Project `direction` out of everything written to the residual stream (embeddings, every
    attention and MLP output), as in Arditi et al. (2024) directional ablation. Returns hook handles."""
    r = _per_device(unit(direction.float()), next(model.parameters()).dtype)

    def proj_out(h):
        rd = r(h.device)
        return h - (h @ rd).unsqueeze(-1) * rd

    handles = []
    try:   # a failure part-way (e.g. a layer without .mlp) must not leave the hooks already registered
        handles.append(model.get_input_embeddings().register_forward_hook(lambda m, i, o: proj_out(o)))
        for layer in decoder_layers(model):
            handles.append(layer.self_attn.register_forward_hook(lambda m, i, o: _replace(o, proj_out(_hidden(o)))))
            handles.append(layer.mlp.register_forward_hook(lambda m, i, o: proj_out(o)))
    except BaseException:
        remove_hooks(handles)
        raise
    return handles


def add_direction(model, resid_index: int, vector: torch.Tensor, alpha: float = 1.0) -> list:
    """Add alpha * vector to the residual stream at `resid_index` (>=1), all positions. Returns hook handles."""
    if resid_index < 1:
        raise ValueError("resid_index must be >= 1 (output of a decoder layer)")
    v = _per_device(alpha * vector.float(), next(model.parameters()).dtype)
    layer = decoder_layers(model)[resid_index - 1]
    return [layer.register_forward_hook(lambda m, i, o: _replace(o, _hidden(o) + v(_hidden(o).device)))]


# ---------------------------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------------------------


def component_id(kind: str, layer: int, head: int | None = None) -> str:
    if kind == "attn_head":
        if head is None:
            raise ValueError("attn_head components need a head index")
        return f"attn_head:{layer}:{head}"
    if kind == "mlp":
        return f"mlp:{layer}"
    raise ValueError(f"unknown component type {kind!r}")


def parse_component(cid: str) -> tuple[str, int, int | None]:
    """"attn_head:12:3" -> ("attn_head", 12, 3); "mlp:14" -> ("mlp", 14, None)."""
    parts = cid.split(":")
    if parts[0] == "attn_head" and len(parts) == 3:
        return "attn_head", int(parts[1]), int(parts[2])
    if parts[0] == "mlp" and len(parts) == 2:
        return "mlp", int(parts[1]), None
    raise ValueError(f"bad component id {cid!r}")


def all_components(model) -> list[str]:
    n_heads, _ = head_geometry(model)
    comps = []
    for layer in range(len(decoder_layers(model))):
        comps += [f"attn_head:{layer}:{h}" for h in range(n_heads)] + [f"mlp:{layer}"]
    return comps


def random_components_like(components: Sequence[str], model, rng: random.Random) -> list[str]:
    """Random set with the same number of heads and MLPs (so the same parameter budget)."""
    n_heads, _ = head_geometry(model)
    L = len(decoder_layers(model))
    n_h = sum(c.startswith("attn_head") for c in components)
    n_m = len(components) - n_h
    heads = [f"attn_head:{l}:{h}" for l in range(L) for h in range(n_heads)]
    mlps = [f"mlp:{l}" for l in range(L)]
    return rng.sample(heads, n_h) + rng.sample(mlps, n_m)


@torch.no_grad()
def component_means(model, tok, prompts: Sequence[str], batch_size: int = 8, system: str | None = None) -> dict:
    """Mean o_proj input (all heads) and MLP output per layer over every real token of `prompts`:
    {"attn": Float[Tensor, "layer d"], "mlp": Float[Tensor, "layer d"]}, used for mean-ablation."""
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


def ablate_components(model, components: Iterable[str], means: dict | None = None) -> list:
    """Ablate attention heads (their slice of the o_proj input) and whole MLP outputs: replaced by
    `means` from component_means (mean-ablation) or, without it, by zero. Zero-ablating many MLPs
    knocks the model off-distribution, so random-component baselines should use means."""
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
    try:   # a failure part-way (e.g. a layer index out of range) must not leave the hooks already registered
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
    except BaseException:
        remove_hooks(handles)
        raise
    return handles


# ---------------------------------------------------------------------------------------------
# Attribution (fallback for toolkit circuit discovery, docs/03 §5; also used by cis_circ)
# ---------------------------------------------------------------------------------------------


def attribution_scores(model, tok, clean_prompts: Sequence[str], corrupt_prompts: Sequence[str], token_sets: dict,
                       batch_size: int = 2, system: str | None = None) -> dict[str, float]:
    """Per-component attribution-patching estimate of how much patching the *clean* (harmful)
    activation into the *corrupt* (harmless twin) run raises the refusal logit difference:

        score_c ≈ mean_pairs[(a_clean - a_corrupt) · ∂M/∂a |corrupt]   at the last prompt token.

    Restricting to the last token lets pairs differ in length; it ignores effects routed through
    earlier positions. Use an fp32 model. Returns {component id: score} for every component.
    """
    n_heads, head_dim = head_geometry(model)
    layers = decoder_layers(model)
    L = len(layers)
    total = {"head": torch.zeros(L, n_heads), "mlp": torch.zeros(L)}
    n = 0
    clean_texts = format_chat(tok, clean_prompts, system)
    corrupt_texts = format_chat(tok, corrupt_prompts, system)
    pairs = list(zip(batched(clean_texts, batch_size), batched(corrupt_texts, batch_size), strict=True))
    for cb, xb in progress(pairs, desc="attribution", disable=None):
        store: dict = {}

        def hooks_for(keep_grad: bool):
            hs = []
            for k, layer in enumerate(layers):
                def pre(m, args, k=k):
                    if keep_grad:
                        args[0].retain_grad()
                    store[("head", k)] = args[0]

                def post(m, i, o, k=k):
                    if keep_grad:
                        o.retain_grad()
                    store[("mlp", k)] = o

                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
                hs.append(layer.mlp.register_forward_hook(post))
            return hs

        with torch.no_grad(), hooked(hooks_for(False)):
            model(**encode(tok, cb, model.device), logits_to_keep=1)
        clean = {k: v[:, -1].detach().float() for k, v in store.items()}
        store.clear()

        enc = encode(tok, xb, model.device)
        embeds = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
        with hooked(hooks_for(True)):
            logits = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"], logits_to_keep=1).logits[:, -1]
            metric = logit_diff_from_logits(logits, token_sets).sum()
            metric.backward()
        for k in range(L):
            a = store[("head", k)]
            delta = (clean[("head", k)] - a[:, -1].detach().float()) * a.grad[:, -1].float()
            total["head"][k] += delta.view(-1, n_heads, head_dim).sum(-1).sum(0).cpu()
            m = store[("mlp", k)]
            total["mlp"][k] += ((clean[("mlp", k)] - m[:, -1].detach().float()) * m.grad[:, -1].float()).sum().cpu()
        n += len(cb)
        store.clear()
        model.zero_grad(set_to_none=True)
    scores = {}
    for k in range(L):
        for h in range(n_heads):
            scores[f"attn_head:{k}:{h}"] = float(total["head"][k, h] / n)
        scores[f"mlp:{k}"] = float(total["mlp"][k] / n)
    return scores


@torch.no_grad()
def activation_patching_scores(model, tok, clean_prompts: Sequence[str], corrupt_prompts: Sequence[str],
                               components: Sequence[str], token_sets: dict, batch_size: int = 8,
                               system: str | None = None) -> dict[str, float]:
    """Exact version of attribution_scores for a short list of components: patch each component's
    clean last-token activation into the corrupt run and measure the change in the refusal logit
    difference (mean over pairs). One forward pass per component per batch; use it to confirm the
    top attribution candidates, since the linear estimate breaks down where the model is nonlinear."""
    _, head_dim = head_geometry(model)
    layers = decoder_layers(model)
    clean_texts = format_chat(tok, clean_prompts, system)
    corrupt_texts = format_chat(tok, corrupt_prompts, system)
    totals = {c: 0.0 for c in components}
    for cb, xb in zip(batched(clean_texts, batch_size), batched(corrupt_texts, batch_size), strict=True):
        store: dict = {}
        hs = []
        for k, layer in enumerate(layers):
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, k=k: store.__setitem__(("head", k), a[0][:, -1].clone())))
            hs.append(layer.mlp.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(("mlp", k), o[:, -1].clone())))
        with hooked(hs):
            model(**encode(tok, cb, model.device), logits_to_keep=1)
        enc = encode(tok, xb, model.device)
        base = logit_diff_from_logits(model(**enc, logits_to_keep=1).logits[:, -1], token_sets)
        for cid in components:
            kind, k, h = parse_component(cid)
            if kind == "attn_head":
                sl = slice(h * head_dim, (h + 1) * head_dim)

                def pre(m, a, k=k, sl=sl):
                    x = a[0].clone()
                    x[:, -1, sl] = store[("head", k)][:, sl]
                    return (x,) + tuple(a[1:])

                handle = layers[k].self_attn.o_proj.register_forward_pre_hook(pre)
            else:
                def post(m, i, o, k=k):
                    o = o.clone()
                    o[:, -1] = store[("mlp", k)]
                    return o

                handle = layers[k].mlp.register_forward_hook(post)
            with hooked([handle]):
                patched = logit_diff_from_logits(model(**enc, logits_to_keep=1).logits[:, -1], token_sets)
            totals[cid] += float((patched - base).sum())
    return {c: v / len(clean_prompts) for c, v in totals.items()}

"""Directional ablation on post-norm ("sandwich") decoder layers (Gemma-2/3).

Gemma-2/3 layers normalise the attention and MLP outputs (post_attention_layernorm, post_feedforward_layernorm)
before adding them to the residual stream, so projecting r out of the self_attn / mlp outputs is undone by the
RMSNorm weight. The fixed ablate_direction hooks the post-norm outputs on that layout; on Llama/Qwen/Cohere
layouts it must behave exactly as before.
"""
from __future__ import annotations

import pytest
import torch
from conftest import make_model, make_pairs

from safetypulse.activations import _hidden, _per_device, _replace, ablate_direction, hooked, remove_hooks, resid_last, unit
from safetypulse.models import decoder_layers, has_post_block_norms


def _old_ablate_direction(model, direction):
    """ablate_direction as it was before the post-norm fix (hooks self_attn and mlp outputs on every layout)."""
    r = _per_device(unit(direction.float()), next(model.parameters()).dtype)

    def proj_out(h):
        rd = r(h.device)
        return h - (h @ rd).unsqueeze(-1) * rd

    handles = [model.get_input_embeddings().register_forward_hook(lambda m, i, o: proj_out(o))]
    try:
        for layer in decoder_layers(model):
            handles.append(layer.self_attn.register_forward_hook(lambda m, i, o: _replace(o, proj_out(_hidden(o)))))
            handles.append(layer.mlp.register_forward_hook(lambda m, i, o: proj_out(o)))
    except BaseException:
        remove_hooks(handles)
        raise
    return handles


def _gemma3(vocab_size: int, seed: int = 0):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(seed)
    cfg = Gemma3TextConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                           sliding_window=8, pad_token_id=1, eos_token_id=2, bos_token_id=None)
    model = Gemma3ForCausalLM(cfg).eval().requires_grad_(False)
    # Gemma RMSNorm scales by (1 + weight) and weights init to 0; randomise them so the norms are not ~identity.
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(torch.randn_like(p))
    return model


def _cohere(vocab_size: int, seed: int = 0):
    from transformers import CohereConfig, CohereForCausalLM

    torch.manual_seed(seed)
    cfg = CohereConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                       pad_token_id=1, eos_token_id=2, bos_token_id=None)
    return CohereForCausalLM(cfg).eval().requires_grad_(False)


def _direction(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(64, generator=g))


def _prompts():
    return [p[k] for p in make_pairs(4) for k in ("harmful", "harmless")]


def _max_proj(model, tok, r, ablate) -> torch.Tensor:
    with hooked(ablate(model, r)):
        acts = resid_last(model, tok, _prompts(), batch_size=8)     # n x resid x d, fp32
    return (acts @ r).abs().amax(dim=0)                               # per resid position


def test_layout_detection(tok):
    assert has_post_block_norms(decoder_layers(_gemma3(len(tok)))[0])
    assert not has_post_block_norms(decoder_layers(make_model(len(tok)))[0])
    assert not has_post_block_norms(decoder_layers(_cohere(len(tok)))[0])


def test_gemma3_ablation_removes_direction_everywhere(tok):
    model, r = _gemma3(len(tok)), _direction()
    proj = _max_proj(model, tok, r, ablate_direction)
    assert proj.shape[0] == len(decoder_layers(model)) + 1
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-4), proj


def test_gemma3_old_ablation_leaks(tok):
    """The pre-fix hooks leave a non-zero r component after the first layer (the bug the fix addresses)."""
    model, r = _gemma3(len(tok)), _direction()
    proj = _max_proj(model, tok, r, _old_ablate_direction)
    assert proj[0] < 1e-4           # embeddings were always handled
    assert proj[1:].max() > 1e-2, proj


@pytest.mark.parametrize("build", [make_model, _cohere], ids=["llama", "cohere"])
def test_pre_norm_layouts_unchanged(tok, build):
    model, r = build(len(tok)), _direction()
    enc = tok.apply_chat_template([[{"role": "user", "content": p}] for p in _prompts()], add_generation_prompt=True,
                                  padding=True, return_tensors="pt", return_dict=True)
    with hooked(_old_ablate_direction(model, r)):
        old = model(**enc).logits
    with hooked(ablate_direction(model, r)):
        new = model(**enc).logits
    assert torch.equal(old, new)
    proj = _max_proj(model, tok, r, ablate_direction)
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-4), proj

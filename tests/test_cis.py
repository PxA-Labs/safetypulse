import pytest
import torch
from conftest import calibrate_bundle, make_model

from safetypulse.activations import (
    ablate_components,
    ablate_direction,
    add_direction,
    hooked,
    parse_component,
    resid_last,
)
from safetypulse.cis import cis_status, compute_cis
from safetypulse.models import n_resid, restore_weights, state_dict_cpu
from safetypulse.prompts import refusal_logit_diff

VARIANTS = ("cis_sep", "cis_cos", "cis_logit", "cis_circ")


def _perturbed(tok, seed=1, arch="llama"):
    """Same architecture, weights moved by a small random delta: a stand-in for a fine-tuned checkpoint."""
    m = make_model(len(tok), arch=arch)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
@pytest.mark.parametrize("which", ["base", "perturbed"])
def test_parity_with_sp_utils(which, arch, sp_utils, tok, pairs):
    """safetypulse.cis reproduces kaggle/lib/sp_utils.compute_cis bit for bit on the same model and bundle."""
    base = make_model(len(tok), arch=arch)
    bundle = calibrate_bundle(base, tok, pairs)
    m = base if which == "base" else _perturbed(tok, arch=arch)
    ours = compute_cis(m, tok, pairs, bundle)
    theirs = sp_utils.compute_cis(m, tok, pairs, bundle.model_dump(mode="json"), bundle.tensors)
    assert ours["scores"].keys() == theirs["scores"].keys() == set(VARIANTS)
    assert ours["scores"] == theirs["scores"]
    assert ours["per_component"] == theirs["per_component"]
    assert ours["harmful_refusal_logit_mean"] == theirs["harmful_refusal_logit_mean"]
    assert ours["cost"]["n_prompts"] == theirs["cost"]["n_prompts"] == 2 * len(pairs)
    if which == "perturbed":
        assert all(abs(ours["scores"][v] - 1.0) > 1e-3 for v in VARIANTS)


def test_parity_building_blocks(sp_utils, model, tok, pairs, bundle):
    harm = [p["harmful"] for p in pairs]
    assert torch.equal(resid_last(model, tok, harm), sp_utils.resid_last(model, tok, harm))
    ts = bundle.token_sets.model_dump()
    assert torch.equal(refusal_logit_diff(model, tok, harm, ts), sp_utils.refusal_logit_diff(model, tok, harm, ts))
    from safetypulse.prompts import default_token_sets

    assert default_token_sets(tok) == sp_utils.default_token_sets(tok)


def test_base_model_scores_one(model, tok, pairs, bundle):
    out = compute_cis(model, tok, pairs, bundle)
    for v in VARIANTS:
        assert out["scores"][v] == pytest.approx(1.0, abs=1e-6), v
    assert all(c["integrity"] == pytest.approx(1.0, abs=1e-6) for c in out["per_component"])


def test_without_circ(model, tok, pairs, bundle):
    out = compute_cis(model, tok, pairs, bundle, with_circ=False)
    assert set(out["scores"]) == {"cis_sep", "cis_cos", "cis_logit"} and out["per_component"] == []


def test_directional_ablation_lowers_separation(model, tok, pairs, bundle):
    """Projecting the bundle's direction out of the residual stream removes the separation along it."""
    with hooked(ablate_direction(model, bundle.directions[bundle.layer])):
        ablated = compute_cis(model, tok, pairs, bundle, with_circ=False)["scores"]["cis_sep"]
    assert abs(ablated) < 1e-4


def test_hooks_are_removed(model, tok, pairs):
    harm = [p["harmful"] for p in pairs]
    before = resid_last(model, tok, harm)
    with hooked(add_direction(model, 1, torch.ones(64), 3.0)):
        assert not torch.equal(resid_last(model, tok, harm), before)
    assert torch.equal(resid_last(model, tok, harm), before)
    assert before.shape == (len(pairs), n_resid(model), 64)


def test_snapshot_restore(tok):
    m = make_model(len(tok))
    sd = state_dict_cpu(m)
    with torch.no_grad():
        next(m.parameters()).add_(1.0)
    restore_weights(m, sd)
    assert all(torch.equal(p, sd[n]) for n, p in m.named_parameters())


def test_parse_component():
    assert parse_component("attn_head:12:3") == ("attn_head", 12, 3)
    assert parse_component("mlp:14") == ("mlp", 14, None)
    for bad in ("mlp", "attn_head:1", "resid:3", "mlp:1:2"):
        with pytest.raises(ValueError):
            parse_component(bad)


def test_reference_robust_to_batch_size(model, tok, pairs):
    """References computed at one batch size still give ~1.0 at another (left padding only nudges numerics)."""
    b = calibrate_bundle(model, tok, pairs, batch_size=8)
    out = compute_cis(model, tok, pairs, b, batch_size=2, with_circ=False)
    for v in ("cis_sep", "cis_cos", "cis_logit"):
        assert out["scores"][v] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None])
def test_cis_status_nonfinite_is_error(value):
    assert cis_status(value, 0.9, 0.6) == "ERROR"
    assert cis_status(value, None, 0.6) == "ERROR"
    assert cis_status(0.95, 0.9, 0.6) == "PASS"


def _n_hooks(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_ablation_hooks_not_leaked_when_registration_fails(tok):
    m = make_model(len(tok))
    n0 = _n_hooks(m)
    del m.model.layers[2].mlp                                         # registration fails part-way
    with pytest.raises(AttributeError):
        ablate_direction(m, torch.ones(m.config.hidden_size))
    assert _n_hooks(m) == n0
    m2 = make_model(len(tok))
    with pytest.raises(IndexError):
        ablate_components(m2, ["attn_head:0:1", "mlp:9"])             # head hook first, then a missing layer
    assert _n_hooks(m2) == n0

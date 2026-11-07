"""SafeTune R3 baselines. Tests that call SafeTune skip when it is not installed (it needs Python >= 3.12)."""
import pytest
import torch
from conftest import make_model

from safetypulse.adapters import safetune as st_adapter
from safetypulse.models import state_dict_cpu
from safetypulse.repair import circuit_slices, repair

MLP_COMPS = ["mlp:1", "mlp:2"]


def _drifted(vocab, seed=1):
    m = make_model(vocab)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


@pytest.fixture
def safetune():
    return pytest.importorskip("safetune")


@pytest.fixture(scope="module")
def base_state(model):
    return state_dict_cpu(model)


def test_touched_names(model):
    names = [n for n, _ in model.named_parameters()]
    assert st_adapter.touched_names(model, "safetune-wise-ft") == names
    circ = st_adapter.touched_names(model, "safetune-wise-ft-circuit", ["attn_head:1:2", "mlp:2"])
    assert circ == list(dict.fromkeys(n for n, _ in circuit_slices(model, ["attn_head:1:2", "mlp:2"])))
    svr = st_adapter.touched_names(model, "safetune-svr")
    assert svr and not any(s in n for n in svr for s in ("embed_tokens", "lm_head", "norm"))
    assert st_adapter.touched_names(model, "safetune-svr", target_modules=["down_proj"]) == [n for n in names if "down_proj" in n]
    with pytest.raises(ValueError, match="unknown SafeTune method"):
        st_adapter.touched_names(model, "safetune-nope")


def test_missing_safetune_is_a_clear_error(model, base_state, bundle):
    if st_adapter.available():
        pytest.skip("safetune is installed")
    with pytest.raises(ImportError, match="SafeTune"):
        repair(_drifted(model.config.vocab_size), base_state, bundle, "safetune-wise-ft", 0.5)


def test_wise_ft_circuit_equals_r1_on_mlp_components(safetune, model, base_state, bundle):
    """With whole-tensor components (MLPs), SafeTune's wise_ft + param filter is our R1 (same α), to fp32 rounding."""
    v = model.config.vocab_size
    ours, theirs = _drifted(v), _drifted(v)
    _, r1 = repair(ours, base_state, bundle, "circuit-restore", 0.3, components=MLP_COMPS)
    _, r3 = repair(theirs, base_state, bundle, "safetune-wise-ft-circuit", 0.3, components=MLP_COMPS)
    for (n, a), b in zip(ours.named_parameters(), theirs.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6, msg=n)
    assert r3.code == "R3" and r3.params_changed == r1.params_changed and r3.components == MLP_COMPS
    assert r3.safetune["function"] == "safetune.recover.apply_wise_ft"


def test_wise_ft_full_equals_r4(safetune, model, base_state, bundle):
    v = model.config.vocab_size
    ours, theirs = _drifted(v), _drifted(v)
    _, r4 = repair(ours, base_state, bundle, "full-interp", 0.25)
    _, r3 = repair(theirs, base_state, bundle, "safetune-wise-ft", 0.25)
    for a, b in zip(ours.parameters(), theirs.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert r3.params_changed == r4.params_changed and r3.delta_norm.after == pytest.approx(r4.delta_norm.after, rel=1e-5)


def test_safety_vector_restore(safetune, model, base_state, bundle):
    v = model.config.vocab_size
    m = _drifted(v)
    ft = state_dict_cpu(m)
    _, log = repair(m, base_state, bundle, "safetune-svr", 1.0, safetune_kwargs={"rank": None})
    touched = set(st_adapter.touched_names(m, "safetune-svr"))
    for n, p in m.named_parameters():
        if n in touched:     # full-rank, alpha 1: back to base
            torch.testing.assert_close(p, base_state[n], rtol=0, atol=1e-6)
        else:                # embeddings / norms untouched
            assert torch.equal(p, ft[n]), n
    assert log.safetune["kwargs"]["rank"] is None and log.tensors_changed == len(touched)
    m2 = _drifted(v)
    _, low = repair(m2, base_state, bundle, "safetune-svr", 1.0, safetune_kwargs={"rank": 2})
    assert 0 < low.delta_norm.touched_after < low.delta_norm.touched_before

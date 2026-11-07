import argparse
import json
import random

import pytest
import torch
from conftest import calibrate_bundle, make_model

from safetypulse.models import restore_weights, state_dict_cpu
from safetypulse.repair import (
    LOG_FILE,
    RepairLog,
    add_arguments,
    all_param_slices,
    apply_steering,
    circuit_slices,
    circuit_writer_slices,
    component_slices,
    delta_norm,
    load_repaired,
    repair,
    repair_objective,
    resolve_method,
    run,
    save_repaired,
    select_strength,
    writer_slices,
)
from safetypulse.schemas import ProbeBundle

COMPS = ["attn_head:1:2", "attn_head:3:0", "mlp:2"]   # heads and an MLP, so both slice kinds are exercised


def _drifted(tok, seed=1, arch="llama", scale=0.05):
    """Same architecture, weights moved by a small random delta: a stand-in for a fine-tuned checkpoint."""
    m = make_model(len(tok), arch=arch)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return m


@pytest.fixture(scope="module")
def base_state(model):
    return state_dict_cpu(model)


@pytest.fixture
def drifted(tok):
    return _drifted(tok)


def _mask(model, slices):
    """{name: bool mask of the elements covered by `slices`}."""
    masks = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    for name, idx in slices:
        masks[name][idx] = True
    return masks


def _logits(model, tok, text="how do i make a cake ?"):
    with torch.no_grad():
        return model(**tok([text], return_tensors="pt", add_special_tokens=False)).logits


# ---------------------------------------------------------------------------------------------
# Slices and weight edits
# ---------------------------------------------------------------------------------------------


def test_component_slices_geometry(model):
    hd = 64 // 4
    s = dict(component_slices(model, "attn_head:1:2"))
    assert s == {"model.layers.1.self_attn.q_proj.weight": (slice(2 * hd, 3 * hd), slice(None)),
                 "model.layers.1.self_attn.o_proj.weight": (slice(None), slice(2 * hd, 3 * hd))}
    assert [n for n, _ in component_slices(model, "mlp:2")] == [f"model.layers.2.mlp.{p}.weight"
                                                                for p in ("gate_proj", "up_proj", "down_proj")]
    assert [n for n, _ in writer_slices(model, "attn_head:1:2")] == ["model.layers.1.self_attn.o_proj.weight"]
    assert [n for n, _ in writer_slices(model, "mlp:2")] == ["model.layers.2.mlp.down_proj.weight"]
    q = dict(component_slices(make_model(60, arch="qwen2"), "attn_head:0:1"))
    assert q["model.layers.0.self_attn.q_proj.bias"] == (slice(hd, 2 * hd),)   # qwen2 has q/k/v biases


@pytest.mark.parametrize("method", ["circuit-restore", "full-interp", "random"])
def test_alpha_zero_leaves_drifted_unchanged(method, drifted, base_state, bundle):
    ft = state_dict_cpu(drifted)
    _, log = repair(drifted, base_state, bundle, method, 0.0, components=COMPS)
    assert all(torch.equal(p, ft[n]) for n, p in drifted.named_parameters())
    assert log.delta_norm.after == log.delta_norm.before and log.params_changed > 0


def test_full_interp_alpha_one_restores_base(drifted, base_state, bundle):
    _, log = repair(drifted, base_state, bundle, "full-interp", 1.0)
    for n, p in drifted.named_parameters():
        # W + (W_base - W) is W_base up to one rounding of the subtraction (sp_utils' formula, kept as is)
        torch.testing.assert_close(p, base_state[n], rtol=0, atol=1e-7)
    assert log.delta_norm.after < 1e-5 < log.delta_norm.before
    assert log.params_changed == sum(p.numel() for p in drifted.parameters()) and log.scope == "all"


def test_circuit_restore_touches_only_component_slices(drifted, base_state, bundle):
    ft = state_dict_cpu(drifted)
    masks = _mask(drifted, circuit_slices(drifted, COMPS))
    _, log = repair(drifted, base_state, bundle, "circuit-restore", 1.0, components=COMPS)
    for n, p in drifted.named_parameters():
        m = masks[n]
        assert torch.equal(p[~m], ft[n][~m]), n                                   # outside: bit-identical
        torch.testing.assert_close(p[m], base_state[n][m], rtol=0, atol=1e-7)     # inside: base
    assert not masks["model.layers.1.self_attn.k_proj.weight"].any()             # GQA-shared k/v untouched
    assert log.params_changed == sum(int(m.sum()) for m in masks.values())
    assert log.components == COMPS and log.code == "R1" and log.tensors_changed == 7
    assert log.delta_norm.touched_after < 1e-5 and log.delta_norm.after < log.delta_norm.before


def test_circuit_restore_half_alpha(drifted, base_state, bundle):
    ft = state_dict_cpu(drifted)
    repair(drifted, base_state, bundle, "R1", 0.5, components=["mlp:2"])
    n = "model.layers.2.mlp.up_proj.weight"
    torch.testing.assert_close(dict(drifted.named_parameters())[n], 0.5 * (ft[n] + base_state[n]))


def test_projection_removes_delta_along_direction(drifted, base_state, bundle):
    ft = state_dict_cpu(drifted)
    r_hat = bundle.directions[bundle.layer] / bundle.directions[bundle.layer].norm()
    writers = circuit_writer_slices(drifted, COMPS)
    masks = _mask(drifted, writers)
    _, log = repair(drifted, base_state, bundle, "circuit-project", 1.0, components=COMPS)
    params = dict(drifted.named_parameters())
    for n, idx in writers:
        new, old = params[n][idx] - base_state[n][idx], ft[n][idx] - base_state[n][idx]
        assert (r_hat @ new).abs().max() < 1e-5                                    # no write along r̂ left
        perp = lambda d: d - torch.outer(r_hat, r_hat @ d)                         # noqa: E731
        torch.testing.assert_close(perp(new), perp(old), rtol=0, atol=1e-6)       # orthogonal part kept
    for n, p in drifted.named_parameters():
        assert torch.equal(p[~masks[n]], ft[n][~masks[n]]), n                     # q_proj, gate/up, rest untouched
    assert log.scope == "circuit_writers" and log.params_changed == sum(int(m.sum()) for m in masks.values())


def test_projection_undoes_a_direction_reducing_drift(model, base_state, bundle):
    """A fine-tune that only lowers the circuit's write along r̂ is fully undone (β=1), halved by β=0.5."""
    r_hat = bundle.directions[bundle.layer] / bundle.directions[bundle.layer].norm()
    m = make_model(model.config.vocab_size)
    name = "model.layers.2.mlp.down_proj.weight"
    p = dict(m.named_parameters())[name]
    u = torch.randn(p.shape[1], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        p -= 0.3 * torch.outer(r_hat, u)
    repair(m, base_state, bundle, "circuit-project", 0.5, components=["mlp:2"])
    torch.testing.assert_close(r_hat @ (p - base_state[name]), -0.15 * u, rtol=0, atol=1e-5)
    repair(m, base_state, bundle, "circuit-project", 1.0, components=["mlp:2"])
    torch.testing.assert_close(p, base_state[name], rtol=0, atol=1e-6)


def test_random_control_same_budget(drifted, base_state, bundle, tok):
    _, r1 = repair(drifted, base_state, bundle, "circuit-restore", 0.5, components=COMPS)
    logs = []
    for seed in range(3):
        m = _drifted(tok)
        _, log = repair(m, base_state, bundle, "random", 0.5, components=COMPS, seed=seed)
        logs.append(log)
        assert log.params_changed == r1.params_changed and log.seed == seed and log.circuit == COMPS
        assert sum(c.startswith("attn_head") for c in log.components) == 2 and len(log.components) == 3
    assert len({tuple(log.components) for log in logs}) > 1                       # different draws
    _, again = repair(_drifted(tok), base_state, bundle, "random", 0.5, components=COMPS, seed=1)
    assert again.components == logs[1].components                                 # reproducible per seed


def test_steering_hook_is_removable(drifted, base_state, bundle, tok):
    ft = state_dict_cpu(drifted)
    before = _logits(drifted, tok)
    _, log = repair(drifted, base_state, bundle, "steer", 1.0)
    assert len(log.hooks) == 1 and log.params_changed == 0 and log.scope == "hook"
    assert not torch.allclose(_logits(drifted, tok), before)
    log.remove_hooks()
    assert log.hooks == [] and torch.equal(_logits(drifted, tok), before)
    assert all(torch.equal(p, ft[n]) for n, p in drifted.named_parameters())
    assert log.steering.resid_index == bundle.layer and log.delta_norm.after == log.delta_norm.before


def test_bundle_circuit_is_the_default(drifted, base_state, bundle):
    _, log = repair(drifted, base_state, bundle, "circuit-restore", 0.25)
    assert log.circuit == log.components == [c.id for c in bundle.components]
    assert log.probe_bundle_sha256 == bundle.sha256() and log.provenance["created_at"]


def test_errors(drifted, base_state, bundle):
    with pytest.raises(ValueError, match="unknown repair method"):
        repair(drifted, base_state, bundle, "R3", 0.5)
    empty = bundle.model_copy(update={"components": []})
    empty._tensors = bundle.tensors
    with pytest.raises(ValueError, match="needs circuit components"):
        repair(drifted, base_state, empty, "circuit-restore", 0.5)
    repair(drifted, base_state, empty, "full-interp", 0.0)                          # R4 needs no circuit
    bad = {k: v for k, v in base_state.items() if "mlp" not in k}
    with pytest.raises(ValueError, match="base state has no parameter"):
        repair(drifted, bad, bundle, "full-interp", 0.5)
    with pytest.raises(ValueError, match="finite"):
        repair(drifted, base_state, bundle, "full-interp", float("nan"))
    assert resolve_method("R2") == ("circuit-project", "R2") and resolve_method("safetune-svr")[1] == "R3"


def test_delta_norm(drifted, base_state):
    assert delta_norm(drifted, base_state, []) == 0.0
    full = delta_norm(drifted, base_state)
    by_tensor = sum(float((p - base_state[n]).pow(2).sum()) for n, p in drifted.named_parameters()) ** 0.5
    assert full == pytest.approx(by_tensor, rel=1e-6)


# ---------------------------------------------------------------------------------------------
# Parity with kaggle/lib/sp_utils (notebook 09's functions)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_parity_with_sp_utils(arch, sp_utils, tok, bundle):
    base = make_model(len(tok), arch=arch)
    base_sd = state_dict_cpu(base)
    comps = COMPS + ["attn_head:0:3"]
    for c in comps:
        assert component_slices(base, c) == sp_utils.component_slices(base, c)
        assert writer_slices(base, c) == sp_utils.writer_slices(base, c)
    assert all_param_slices(base) == sp_utils.all_param_slices(base)
    rng_a, rng_b = random.Random(4), random.Random(4)
    from safetypulse.activations import random_components_like

    assert random_components_like(comps, base, rng_a) == sp_utils.random_components_like(comps, base, rng_b)

    r = bundle.directions[bundle.layer]
    cases = [
        (lambda m, lib: lib.interpolate_toward_base(m, base_sd, sum((lib.component_slices(m, c) for c in comps), []), 0.3)),
        (lambda m, lib: lib.interpolate_toward_base(m, base_sd, lib.all_param_slices(m), 0.25)),
        (lambda m, lib: lib.project_delta(m, base_sd, sum((lib.writer_slices(m, c) for c in comps), []), r, 0.5)),
    ]
    import safetypulse.repair as ours

    for fn in cases:
        a, b = _drifted(tok, arch=arch), _drifted(tok, arch=arch)
        assert fn(a, ours) == fn(b, sp_utils)
        assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters(), strict=True))


def test_repair_api_matches_notebook_09(sp_utils, tok, base_state, bundle):
    """repair(...) = notebook 09's `apply(method, param)` on the same drifted weights, bit for bit."""
    comps = [c.id for c in bundle.components]
    r_star = bundle.directions[bundle.layer]
    ref = _drifted(tok)
    ft_sd = state_dict_cpu(ref)
    sp = sp_utils
    slices = lambda m, cs: sum((sp.component_slices(m, c) for c in cs), [])          # noqa: E731
    nb = {   # notebook 09's apply(), verbatim
        "R1": lambda m, a: sp.interpolate_toward_base(m, base_state, slices(m, comps), a),
        "R2": lambda m, a: sp.project_delta(m, base_state, sum((sp.writer_slices(m, c) for c in comps), []), r_star, a),
        "R4": lambda m, a: sp.interpolate_toward_base(m, base_state, sp.all_param_slices(m), a),
        "R5": lambda m, a: sp.interpolate_toward_base(
            m, base_state, slices(m, sp.random_components_like(comps, m, random.Random(2))), a),
    }
    for code, fn in nb.items():
        restore_weights(ref, ft_sd)
        changed = fn(ref, 0.5)
        m = _drifted(tok)
        _, log = repair(m, base_state, bundle, code, 0.5, seed=2)
        assert log.params_changed == changed, code
        assert all(torch.equal(pa, pb) for pa, pb in zip(m.parameters(), ref.parameters(), strict=True)), code
    restore_weights(ref, ft_sd)
    with sp_utils.hooked(sp_utils.add_direction(ref, bundle.layer, r_star, 0.5)):
        expected = _logits(ref, tok)
    m = _drifted(tok)
    _, log = repair(m, base_state, bundle, "R6", 0.5)
    assert torch.equal(_logits(m, tok), expected)
    log.remove_hooks()


# ---------------------------------------------------------------------------------------------
# α selection
# ---------------------------------------------------------------------------------------------


def test_objective_and_selection():
    assert repair_objective(0.6, 0.2, 0.1) == pytest.approx(0.3)
    assert repair_objective(0.6, 0.2, 0.1, lam=2.0, mu=0.0) == pytest.approx(0.2)
    rows = [{"param": 0.25, "task": 0.5, "asr": 0.3, "over_refusal": 0.0},
            {"param": 0.5, "task": 0.5, "asr": 0.1, "over_refusal": 0.0},
            {"param": 1.0, "task": 0.3, "asr": 0.0, "over_refusal": 0.1}]
    assert select_strength(rows) == 0.5
    assert select_strength(rows, lam=0.0) == 0.5          # tie 0.25 vs 0.5 -> the later row
    assert select_strength([{"alpha": 0.1, "objective": 1.0}], key="alpha") == 0.1
    with pytest.raises(ValueError):
        select_strength([])


# ---------------------------------------------------------------------------------------------
# RepairLog, save/load, CLI
# ---------------------------------------------------------------------------------------------


def test_repair_log_round_trip(tmp_path, drifted, base_state, bundle):
    _, log = repair(drifted, base_state, bundle, "circuit-restore", 0.5, components=COMPS)
    log.dev_objective = 0.42                                       # extra keys survive
    path = log.save(tmp_path / "log.json")
    back = RepairLog.load(path)
    assert back.model_dump() == log.model_dump() and back.dev_objective == 0.42
    assert RepairLog.model_validate(json.loads(path.read_text())).delta_norm == log.delta_norm


@pytest.mark.parametrize("method", ["circuit-restore", "steer"])
def test_save_and_load_repaired(method, tmp_path, drifted, base_state, bundle, tok):
    _, log = repair(drifted, base_state, bundle, method, 0.5, components=COMPS)
    expected = _logits(drifted, tok)
    out = save_repaired(drifted, tok, log, tmp_path / "out", bundle)
    log.remove_hooks()
    assert (out / "model.safetensors").is_file() and (out / LOG_FILE).is_file()
    m2, tok2, log2 = load_repaired(out)
    assert all(torch.equal(p, q) for p, q in zip(m2.parameters(), drifted.parameters(), strict=True))
    assert log2.output == str(out) and log2.method == method
    assert len(log2.hooks) == (method == "steer")
    torch.testing.assert_close(_logits(m2, tok2), expected, rtol=0, atol=1e-5)
    log2.remove_hooks()
    handles = apply_steering(m2, out)                               # re-attach from the saved dir alone
    assert len(handles) == (method == "steer")
    torch.testing.assert_close(_logits(m2, tok2), expected, rtol=0, atol=1e-5)
    for h in handles:
        h.remove()


def _cli(argv):
    p = add_arguments(argparse.ArgumentParser())
    return run(p.parse_args(argv))


@pytest.fixture
def bundle_dir(tmp_path, model, tok, pairs, model_dir):
    b = calibrate_bundle(model, tok, pairs, base_name=str(model_dir))
    b.save(tmp_path / "bundle", probe_pairs=pairs)
    return tmp_path / "bundle"


def test_cli_full_checkpoint(tmp_path, tok, bundle_dir, base_state, capsys):
    ckpt = tmp_path / "run" / "checkpoint-12"
    _drifted(tok).save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    out = tmp_path / "repaired"
    assert _cli(["--checkpoint", str(ckpt), "--bundle", str(bundle_dir), "--method", "full-interp",
                 "--alpha", "1", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["code"] == "R4" and printed["out"] == str(out)
    m, _, log = load_repaired(out)
    for n, p in m.named_parameters():
        torch.testing.assert_close(p, base_state[n], rtol=0, atol=1e-7)
    assert log.checkpoint.step == 12 and log.probe_bundle_sha256 == ProbeBundle.load(bundle_dir).sha256()
    assert log.provenance["dtype"] == "float32"


def test_cli_lora_checkpoint(tmp_path, model_dir, bundle_dir, base_state):
    """A LoRA dir is merged into the base model and saved as a full merged model; R1 only edits the circuit."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from safetypulse.models import patch_peft

    patch_peft()
    torch.manual_seed(3)
    peft_model = get_peft_model(AutoModelForCausalLM.from_pretrained(model_dir),
                                LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj", "down_proj"]))
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.2)
    ckpt = tmp_path / "run" / "checkpoint-7"
    peft_model.save_pretrained(ckpt)
    out = tmp_path / "rep"
    assert _cli(["--checkpoint", str(ckpt), "--bundle", str(bundle_dir), "--method", "R1", "--alpha", "1.0",
                 "--out", str(out)]) == 0
    m, _, log = load_repaired(out)
    assert not (out / "adapter_config.json").exists() and (out / "model.safetensors").is_file()
    assert log.method == "circuit-restore" and log.checkpoint.step == 7 and log.checkpoint.base_model == str(model_dir)
    masks = _mask(m, circuit_slices(m, log.components))
    drifted_somewhere = False
    for n, p in m.named_parameters():
        torch.testing.assert_close(p[masks[n]], base_state[n][masks[n]], rtol=0, atol=1e-7)
        drifted_somewhere |= not torch.allclose(p[~masks[n]], base_state[n][~masks[n]])
    assert drifted_somewhere and log.delta_norm.before > log.delta_norm.after > 0


def test_steer_repair_output_is_checked_with_its_steering(tmp_path, tok, model, bundle_dir, capsys):
    """R6 lives in a hook, not the weights: `check` on its output dir re-attaches it and scores the steered model,
    not the drifted weights; a further repair of a steered dir is refused instead of silently dropping it."""
    from safetypulse import CheckResult, check
    from safetypulse.cli import main

    bundle = ProbeBundle.load(bundle_dir)
    ckpt = tmp_path / "run" / "checkpoint-4"
    drift = _drifted(tok)
    drift.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    drifted_scores = check(drift, bundle, tokenizer=tok).scores
    _, log = repair(drift, state_dict_cpu(model), bundle, "steer", 3.0)
    steered_scores = check(drift, bundle, tokenizer=tok).scores
    log.remove_hooks()
    assert steered_scores["cis_logit"] != pytest.approx(drifted_scores["cis_logit"], abs=1e-3)

    out = tmp_path / "steered"
    assert _cli(["--checkpoint", str(ckpt), "--bundle", str(bundle_dir), "--method", "steer", "--alpha", "3.0",
                 "--out", str(out)]) == 0
    res = tmp_path / "check.json"
    main(["check", "--checkpoint", str(out), "--bundle", str(bundle_dir), "--out", str(res)])
    r = CheckResult.load(res)
    assert r.steering is not None and r.steering["applied"] and r.steering["alpha"] == 3.0
    assert r.steering["resid_index"] == bundle.layer
    assert r.scores == pytest.approx(steered_scores, abs=1e-5)
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["steering"] is True

    with pytest.raises(ValueError, match="steering repair"):
        _cli(["--checkpoint", str(out), "--bundle", str(bundle_dir), "--method", "R1", "--alpha", "1",
              "--out", str(tmp_path / "again")])

"""Regression tests for defects found in the round-1 correctness audit of audit.py, repair.py and the Trainer
callback. Each test failed before its fix."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import calibrate_bundle, make_model
from transformers import TrainerControl, TrainerState

from safetypulse.integrations import SafetyPulseCallback
from safetypulse.integrations.hf_callback import CHECKPOINT_FILE, CHECKS_FILE
from safetypulse.judge import Judge
from safetypulse.models import state_dict_cpu
from safetypulse.repair import repair

audit_mod = importlib.import_module("safetypulse.audit")

FAIL_ALWAYS = {"metric": "cis_sep", "tau_warn": 2.0, "tau_fail": 1.5}    # the base model scores 1.0 -> FAIL


def _drifted(tok, seed=1, scale=0.05):
    m = make_model(len(tok))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return m


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _unchanged(model, snap):
    return all(torch.equal(p, snap[n]) for n, p in model.named_parameters())


# --- repair: component ids are validated before any weight is edited --------------------------------------------


@pytest.mark.parametrize("method", ["circuit-restore", "circuit-project", "safetune-wise-ft-circuit"])
def test_repair_rejects_out_of_range_head(method, tok, model, bundle):
    """attn_head:1:99 on a 4-head model used to slice q_proj rows 1584:1600 of 64: an empty slice, so the 'repair'
    changed nothing and still wrote a RepairLog naming the component."""
    drifted, base_state = _drifted(tok), state_dict_cpu(model)
    snap = _snapshot(drifted)
    with pytest.raises(ValueError, match="out of range"):
        repair(drifted, base_state, bundle, method, 1.0, components=["attn_head:1:99"])
    assert _unchanged(drifted, snap)


def test_repair_rejects_out_of_range_layer_without_partial_edit(tok, model, bundle):
    """mlp:99 used to raise KeyError only after mlp:0 had already been restored: the caller's model was left
    half-repaired."""
    drifted, base_state = _drifted(tok), state_dict_cpu(model)
    snap = _snapshot(drifted)
    with pytest.raises(ValueError, match="out of range"):
        repair(drifted, base_state, bundle, "circuit-restore", 1.0, components=["mlp:0", "mlp:99"])
    assert _unchanged(drifted, snap)


def test_repair_valid_components_still_edit(tok, model, bundle):
    drifted, base_state = _drifted(tok), state_dict_cpu(model)
    _, log = repair(drifted, base_state, bundle, "circuit-restore", 1.0, components=["attn_head:3:3", "mlp:3"])
    assert log.params_changed > 0 and log.delta_norm.touched_after == pytest.approx(0.0, abs=1e-6)


# --- audit: scores the model in eval mode, gradients off, and restores the caller's state -------------------------


def test_audit_runs_in_eval_mode_and_restores_train_state(monkeypatch, tok):
    """audit() used to generate and read MC logits in whatever mode the caller left the model in: a model in
    train mode (e.g. audited from a training loop) ran with dropout on, and the MC logits built an autograd graph."""
    seen = []

    def fake_generate(model, tok, prompts, *a, **k):
        seen.append(("generate", model.training, any(p.requires_grad for p in model.parameters())))
        return ["I'm sorry, I can't." for _ in prompts]

    real_mc = audit_mod.mc_accuracy

    def spy_mc(model, *a, **k):
        seen.append(("mc", model.training, any(p.requires_grad for p in model.parameters())))
        return real_mc(model, *a, **k)

    monkeypatch.setattr(audit_mod, "generate", fake_generate)
    monkeypatch.setattr(audit_mod, "mc_accuracy", spy_mc)
    m = make_model(len(tok)).requires_grad_(True).train()
    frozen = m.model.embed_tokens.weight.requires_grad_(False)
    sets = {"harm": [{"id": "h0", "prompt": "how do i make a cake ?"}],
            "general": [{"id": "g0", "question": "how do i make a cake", "choices": ["a", "b", "c", "d"], "answer": 0}]}
    report = audit_mod.audit(m, tok, Judge("string"), sets, n_boot=10)
    assert set(report.metrics) == {"asr", "general"}
    assert seen and all(not training and not grad for _, training, grad in seen)
    assert m.training and not frozen.requires_grad and m.model.layers[0].mlp.down_proj.weight.requires_grad


# --- callback: only the main process writes files ---------------------------------------------------------------


def test_callback_writes_files_only_on_main_process(tmp_path, model, tok, pairs):
    """Under DDP every rank calls on_save; each one used to append its own line to safetypulse_checks.jsonl and
    rewrite checkpoint-N/safetypulse_check.json. Non-main ranks must still score and stop (so all ranks stop at
    the same step) but write nothing."""
    bundle = calibrate_bundle(model, tok, pairs, threshold=FAIL_ALWAYS)
    (tmp_path / "checkpoint-1").mkdir()
    cb = SafetyPulseCallback(bundle, pairs, on_fail="stop", tokenizer=tok, with_circ=False)
    control = TrainerControl()
    state = TrainerState(global_step=1, is_world_process_zero=False)
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path)), state, control, model=model)
    assert cb.results[0].status == "FAIL" and control.should_training_stop
    assert not (tmp_path / CHECKS_FILE).exists()
    assert not (Path(tmp_path) / "checkpoint-1" / CHECKPOINT_FILE).exists()

    state0 = TrainerState(global_step=1, is_world_process_zero=True)
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path)), state0, TrainerControl(), model=model)
    assert len((tmp_path / CHECKS_FILE).read_text().splitlines()) == 1
    assert (tmp_path / "checkpoint-1" / CHECKPOINT_FILE).is_file()

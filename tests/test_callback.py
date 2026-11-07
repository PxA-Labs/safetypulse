"""SafetyPulseCallback: a tiny Trainer run of a few steps on CPU, plus direct on_save calls with fake state/control."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import calibrate_bundle, make_model
from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments

from safetypulse import CheckResult
from safetypulse.integrations import SafetyPulseCallback
from safetypulse.integrations.hf_callback import CHECKPOINT_FILE, CHECKS_FILE

FAIL_ALWAYS = {"metric": "cis_sep", "tau_warn": 2.0, "tau_fail": 1.5}    # the base model scores 1.0 -> FAIL
PASS_ALWAYS = {"metric": "cis_sep", "tau_warn": -10.0, "tau_fail": -20.0}


@pytest.fixture(scope="module")
def fail_bundle(model, tok, pairs):
    return calibrate_bundle(model, tok, pairs, threshold=FAIL_ALWAYS)


@pytest.fixture(scope="module")
def pass_bundle(model, tok, pairs):
    return calibrate_bundle(model, tok, pairs, threshold=PASS_ALWAYS)


def _train_dataset(tok, n=8):
    text = "<|user|> how do i make a cake ? <|end|> <|assistant|> Sure Here the cake <|end|>"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [{"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids} for _ in range(n)]


def _trainer(tok, out, callback, max_steps=3):
    model = make_model(len(tok)).requires_grad_(True)       # same weights the bundles were calibrated on
    args = TrainingArguments(output_dir=str(out), max_steps=max_steps, save_steps=1, save_strategy="steps",
                             per_device_train_batch_size=2, learning_rate=1e-3, logging_steps=1, report_to=[],
                             use_cpu=True, disable_tqdm=True, save_only_model=True)
    return Trainer(model=model, args=args, train_dataset=_train_dataset(tok), processing_class=tok, callbacks=[callback])


def test_trainer_stops_on_fail(tmp_path, tok, pairs, fail_bundle, caplog):
    cb = SafetyPulseCallback(fail_bundle, pairs, on_fail="stop", out_dir=tmp_path / "sp")
    trainer = _trainer(tok, tmp_path / "run", cb)
    with caplog.at_level(logging.INFO, logger="safetypulse"):
        trainer.train()
    assert trainer.state.global_step == 1                    # stopped at the first save instead of 3 steps
    assert len(cb.results) == 1 and cb.results[0].status == "FAIL"
    assert cb.results[0].checkpoint.step == 1 and cb.results[0].checkpoint.path.endswith("checkpoint-1")
    lines = (tmp_path / "sp" / CHECKS_FILE).read_text().splitlines()
    assert len(lines) == 1 and CheckResult.model_validate(json.loads(lines[0])).status == "FAIL"
    entries = [e for e in trainer.state.log_history if "safetypulse/status" in e]
    assert entries == [{**{f"safetypulse/{k}": v for k, v in cb.results[0].scores.items()},
                        "safetypulse/status": "FAIL", "step": 1}]
    assert "stopping training" in caplog.text and pairs[0]["harmful"] not in caplog.text


def test_trainer_log_mode_runs_to_end(tmp_path, tok, pairs, fail_bundle):
    cb = SafetyPulseCallback(fail_bundle, pairs, on_fail="log", every_n_saves=2)
    trainer = _trainer(tok, tmp_path / "run", cb, max_steps=4)
    trainer.train()
    assert trainer.state.global_step == 4 and cb.n_saves == 4
    assert [r.checkpoint.step for r in cb.results] == [2, 4] and all(r.status == "FAIL" for r in cb.results)
    assert len((tmp_path / "run" / CHECKS_FILE).read_text().splitlines()) == 2     # default out_dir: output_dir
    assert cb.results[1].scores["cis_sep"] != cb.results[0].scores["cis_sep"]      # weights moved in between


def _fake_args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path))


def test_on_save_restores_train_mode_and_grads(tmp_path, tok, pairs, pass_bundle):
    model = make_model(len(tok)).requires_grad_(True).train()
    frozen = model.model.embed_tokens.weight
    frozen.requires_grad_(False)
    probe = model.model.layers[0].mlp.down_proj.weight
    probe.grad = torch.full_like(probe, 0.5)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    cb = SafetyPulseCallback(pass_bundle, pairs, on_fail="stop", tokenizer=tok)
    state, control = TrainerState(global_step=7), TrainerControl()
    out = cb.on_save(_fake_args(tmp_path), state, control, model=model)
    assert out is control and not control.should_training_stop
    assert model.training and not frozen.requires_grad and probe.requires_grad
    assert torch.equal(probe.grad, torch.full_like(probe, 0.5))                     # pending grads untouched
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())     # weights untouched
    r = cb.results[0]
    assert r.status == "PASS" and r.checkpoint.step == 7 and "cis_circ" in r.scores
    assert r.scores["cis_sep"] == pytest.approx(1.0, abs=1e-5)    # same weights as the calibration model


def test_on_save_fail_sets_stop_flag(tmp_path, model, tok, pairs, fail_bundle):
    for on_fail, stops in (("stop", True), ("log", False)):
        cb = SafetyPulseCallback(fail_bundle, pairs, on_fail=on_fail, tokenizer=tok, with_circ=False)
        control = TrainerControl()
        cb.on_save(_fake_args(tmp_path), TrainerState(global_step=1), control, model=model)
        assert control.should_training_stop is stops and cb.results[0].status == "FAIL"
        assert "cis_circ" not in cb.results[0].scores


def test_callback_argument_errors(tmp_path, model, pairs, fail_bundle):
    with pytest.raises(ValueError, match="repair"):
        SafetyPulseCallback(fail_bundle, pairs, on_fail="repair")
    with pytest.raises(ValueError, match="on_fail"):
        SafetyPulseCallback(fail_bundle, pairs, on_fail="halt")
    with pytest.raises(ValueError, match="every_n_saves"):
        SafetyPulseCallback(fail_bundle, pairs, every_n_saves=0)
    cb = SafetyPulseCallback(fail_bundle, pairs)
    with pytest.raises(ValueError, match="tokenizer"):
        cb.on_save(_fake_args(tmp_path), TrainerState(global_step=1), TrainerControl(), model=model)


def test_checkpoint_dirs_carry_their_check(tmp_path, tok, pairs, pass_bundle):
    """Each checkpoint-N dir gets the CheckResult of that save (its trainer_state.json is written before on_save)."""
    cb = SafetyPulseCallback(pass_bundle, pairs, on_fail="log", with_circ=False)
    _trainer(tok, tmp_path / "run", cb, max_steps=2).train()
    assert [r.checkpoint.step for r in cb.results] == [1, 2]
    for r in cb.results:
        path = Path(r.checkpoint.path) / CHECKPOINT_FILE
        assert path.parent == tmp_path / "run" / f"checkpoint-{r.checkpoint.step}"
        saved = CheckResult.load(path)
        assert saved.checkpoint.step == r.checkpoint.step and saved.scores == r.scores and saved.status == "PASS"


def test_on_save_error_stops_like_fail(tmp_path, tok, pairs, pass_bundle):
    """A diverged model (NaN scores) must not PASS: status ERROR, and on_fail="stop" stops on it."""
    m = make_model(len(tok))
    with torch.no_grad():
        m.model.layers[0].mlp.down_proj.weight[0, 0] = float("nan")
    for on_fail, stops in (("stop", True), ("log", False)):
        cb = SafetyPulseCallback(pass_bundle, pairs, on_fail=on_fail, tokenizer=tok, with_circ=False, out_dir=tmp_path)
        control = TrainerControl()
        cb.on_save(_fake_args(tmp_path), TrainerState(global_step=1), control, model=m)
        assert cb.results[0].status == "ERROR" and control.should_training_stop is stops
    line = (tmp_path / CHECKS_FILE).read_text().splitlines()[-1]
    assert "NaN" not in line and json.loads(line)["scores"]["cis_sep"] is None

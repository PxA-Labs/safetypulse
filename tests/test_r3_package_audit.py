"""Round-3 correctness audit of the package outside activations.py (lane 17): regression tests, each written to fail
on the code before its fix. Tiny random model and synthetic word prompts only (conftest)."""
import json
import logging
import math

import pytest
import torch
from conftest import calibrate_bundle
from safetensors.torch import load_file, save_file
from transformers import TrainerControl, TrainerState

from safetypulse.cis import compute_cis
from safetypulse.cli import main
from safetypulse.integrations import SafetyPulseCallback
from safetypulse.repair import select_strength
from safetypulse.schemas import ProbeBundle

THRESH = {"metric": "cis_sep", "tau_warn": 0.9, "tau_fail": 0.6}


# --- bundle tensors stored in half precision -----------------------------------------------------


def _resave_tensors(bundle_dir, dtype):
    path = bundle_dir / "directions.safetensors"
    save_file({k: v.to(dtype) for k, v in load_file(str(path)).items()}, str(path))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_bundle_tensors_score_like_fp32(dtype, tmp_path, model, tok, pairs, bundle):
    """A directions.safetensors written in bf16/fp16 (e.g. by a script that saved model-dtype tensors) loads, but
    compute_cis multiplied fp32 activations by the half-precision direction and crashed with a dtype error."""
    bundle.save(tmp_path, probe_pairs=pairs)
    ref = compute_cis(model, tok, pairs, ProbeBundle.load(tmp_path), with_circ=False)["scores"]
    _resave_tensors(tmp_path, dtype)
    loaded = ProbeBundle.load(tmp_path)
    scores = compute_cis(model, tok, pairs, loaded, with_circ=False)["scores"]
    assert loaded.directions.dtype == torch.float32 and loaded.probe_diff.dtype == torch.float32
    assert scores["cis_sep"] == pytest.approx(ref["cis_sep"], rel=1e-2)
    assert scores["cis_cos"] == pytest.approx(ref["cis_cos"], abs=1e-2)


def test_non_float_bundle_tensors_are_rejected(tmp_path, pairs, bundle):
    bundle.save(tmp_path, probe_pairs=pairs)
    _resave_tensors(tmp_path, torch.int64)
    with pytest.raises(ValueError, match="floating"):
        ProbeBundle.load(tmp_path)


# --- schema version -------------------------------------------------------------------------------


def _set_schema_version(bundle_dir, version):
    path = bundle_dir / "probe_bundle.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = version
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_bundle_from_an_unknown_schema_major_version_is_rejected(tmp_path, pairs, bundle):
    """ProbeBundle.load accepted any schema_version (e.g. "2.0"), so a bundle written by an incompatible future
    format would be scored as if its fields meant what 1.x means."""
    bundle.save(tmp_path, probe_pairs=pairs)
    _set_schema_version(tmp_path, "2.0")
    with pytest.raises(ValueError, match="schema_version"):
        ProbeBundle.load(tmp_path)
    _set_schema_version(tmp_path, "1.1")      # a minor bump of the same major version still loads
    assert ProbeBundle.load(tmp_path).schema_version == "1.1"


# --- repair strength selection --------------------------------------------------------------------


def test_select_strength_with_no_finite_objective_is_a_clear_error():
    """Every row NaN (e.g. an audit metric that could not be computed): best stayed None and the function crashed
    with "TypeError: 'NoneType' object is not subscriptable"."""
    with pytest.raises(ValueError, match="finite objective"):
        select_strength([{"param": 0.5, "objective": math.nan}, {"param": 1.0, "objective": math.nan}])
    assert select_strength([{"param": 0.5, "objective": math.nan}, {"param": 1.0, "objective": 0.1}]) == 1.0


# --- ERROR status that is not caused by a non-finite score -----------------------------------------


def _negative_reference_bundle(model, tok, pairs, base_name):
    b = calibrate_bundle(model, tok, pairs, base_name=str(base_name), threshold=THRESH)
    b.reference.sep_base = -abs(b.reference.sep_base)     # finite and non-zero, so the schema accepts it
    return b


def test_cli_error_message_names_the_real_reason(tmp_path, model, tok, pairs, model_dir, capsys):
    """A bundle whose cis_sep reference is negative gives status ERROR (bundle.metric_problem). The CLI then said
    "primary score cis_sep is not finite (non-finite: )" although every score was finite."""
    _negative_reference_bundle(model, tok, pairs, model_dir).save(tmp_path / "b", probe_pairs=pairs)
    with pytest.warns(UserWarning, match="<= 0"):
        rc = main(["check", "--checkpoint", str(model_dir), "--bundle", str(tmp_path / "b")])
    assert rc == 1
    captured = capsys.readouterr()
    printed = json.loads(captured.out.strip().splitlines()[-1])
    assert printed["status"] == "ERROR" and printed["nonfinite"] == []
    assert "not finite" not in captured.err and "<= 0" in captured.err


def test_callback_error_reason_names_the_real_reason(tmp_path, model, tok, pairs, model_dir, caplog):
    """Same bundle in the training callback: the stop warning said "cis_sep is not finite"."""
    b = _negative_reference_bundle(model, tok, pairs, model_dir)
    cb = SafetyPulseCallback(b, pairs, on_fail="stop", tokenizer=tok, with_circ=False, out_dir=tmp_path)
    control = TrainerControl()
    with caplog.at_level(logging.WARNING, logger="safetypulse"), pytest.warns(UserWarning, match="<= 0"):
        cb.on_save(type("A", (), {"output_dir": str(tmp_path)})(), TrainerState(global_step=1), control, model=model)
    assert cb.results[0].status == "ERROR" and control.should_training_stop
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "not finite" not in text and "<= 0" in text

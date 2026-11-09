import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import CALIBRATED_BUNDLE, NOTEBOOK_BUNDLE
from pydantic import ValidationError

from safetypulse import CheckResult, ProbeBundle
from safetypulse.cis import cis_status
from safetypulse.schemas import Threshold


def _synthetic_tensors(bundle: ProbeBundle, d: int = 16):
    g = torch.Generator().manual_seed(0)
    return torch.randn(bundle.layer + 12, d, generator=g), torch.randn(bundle.layer + 12, d, generator=g)


@pytest.mark.parametrize("path", [NOTEBOOK_BUNDLE, CALIBRATED_BUNDLE], ids=["nb06", "nb08-calibrated"])
def test_notebook_bundle_round_trip(path, tmp_path):
    raw = json.loads(path.read_text())
    bundle = ProbeBundle.load_metadata(path)
    assert bundle.layer == raw["layer"] and len(bundle.components) == len(raw["components"])
    assert [c.id for c in bundle.components] == [c["id"] for c in raw["components"]]
    bundle.with_tensors(*_synthetic_tensors(bundle))
    bundle.save(tmp_path)

    loaded = ProbeBundle.load(tmp_path)
    assert loaded.model_dump(mode="json") == raw          # extra keys (table_a, discovery_exact, ...) survive
    assert torch.equal(loaded.directions, bundle.directions) and torch.equal(loaded.probe_diff, bundle.probe_diff)
    assert loaded.sha256() == bundle.sha256()
    assert ProbeBundle.load(tmp_path / "probe_bundle.json").sha256() == bundle.sha256()


def test_threshold_blocks():
    nb06 = ProbeBundle.load_metadata(NOTEBOOK_BUNDLE).threshold
    nb08 = ProbeBundle.load_metadata(CALIBRATED_BUNDLE).threshold
    assert nb06.metric is None and not nb06.calibrated
    assert nb08.metric == "cis_cos" and nb08.calibrated and nb08.tau_fail < nb08.tau_warn


def test_save_writes_probe_pairs_with_hash(bundle, pairs, tmp_path):
    from safetypulse.utils import sha256_file, sha256_rows

    bundle = bundle.model_copy(deep=True).with_tensors(bundle.directions, bundle.probe_diff)
    bundle.save(tmp_path, probe_pairs=pairs)
    assert bundle.probe_set.n_pairs == len(pairs)
    assert bundle.probe_set.sha256 == sha256_file(tmp_path / "probe_pairs.jsonl") == sha256_rows(pairs)
    assert ProbeBundle.load(tmp_path).probe_path() == tmp_path / "probe_pairs.jsonl"


def _raw():
    return json.loads(NOTEBOOK_BUNDLE.read_text())


@pytest.mark.parametrize("mutate", [
    lambda b: b["threshold"].update(metric="cis_bogus"),
    lambda b: b["threshold"].update(metric="cis_sep", tau_warn=0.5, tau_fail=0.9),
    lambda b: b["threshold"].update(tau_fail=0.5),                     # tau without metric
    lambda b: b["components"][0].update(id="mlp:3"),                   # id disagrees with layer
    lambda b: b["components"][0].update(type="embedding"),
    lambda b: b["reference"].pop("circ_base"),                        # components need circ_base
    lambda b: b["reference"].pop("sep_base"),
    lambda b: b["token_sets"].update(refuse=[]),
    lambda b: b.update(layer=0),
], ids=["metric", "tau-order", "tau-no-metric", "comp-id", "comp-type", "circ-base", "sep-base", "tokens", "layer"])
def test_bundle_validation_rejects(mutate):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValidationError):
        ProbeBundle.model_validate(raw)


def test_component_id_filled_from_fields():
    raw = _raw()
    raw["components"] = [{"type": "attn_head", "layer": 3, "index": 7, "score": 0.4}, {"type": "mlp", "layer": 5, "score": 0.1}]
    assert [c.id for c in ProbeBundle.model_validate(raw).components] == ["attn_head:3:7", "mlp:5"]


def test_tensor_validation(tmp_path):
    bundle = ProbeBundle.load_metadata(NOTEBOOK_BUNDLE)
    with pytest.raises(ValueError, match="out of range"):
        bundle.with_tensors(torch.zeros(10, 4), torch.zeros(10, 4))
    with pytest.raises(ValueError, match="shapes differ"):
        bundle.with_tensors(torch.zeros(30, 4), torch.zeros(30, 5))
    with pytest.raises(ValueError, match="no tensors"):
        bundle.save(tmp_path)
    from safetensors.torch import save_file

    bundle.with_tensors(torch.zeros(30, 4), torch.zeros(30, 4)).save(tmp_path)
    save_file({"directions": torch.zeros(30, 4)}, str(tmp_path / bundle.direction_file))
    with pytest.raises(ValueError, match="probe_diff"):
        ProbeBundle.load(tmp_path)


@pytest.mark.parametrize("value,tau_warn,tau_fail,expected", [
    (0.95, 0.9, 0.8, "PASS"), (0.9, 0.9, 0.8, "PASS"), (0.85, 0.9, 0.8, "WARN"), (0.8, 0.9, 0.8, "WARN"),
    (0.79, 0.9, 0.8, "FAIL"), (0.5, None, 0.8, "FAIL"), (0.85, None, 0.8, "PASS"), (0.85, 0.9, None, "WARN"),
    (0.1, None, None, "UNCALIBRATED"),
])
def test_cis_status(value, tau_warn, tau_fail, expected):
    assert cis_status(value, tau_warn, tau_fail) == expected


def test_check_result_round_trip_and_exit_codes(tmp_path):
    base = dict(probe_bundle_sha256="ab", scores={"cis_sep": 0.7}, primary_metric="cis_sep", tau_warn=0.9, tau_fail=0.8,
                per_component=[{"id": "mlp:1", "integrity": None}], cost={"gpu_seconds": 1.0, "n_prompts": 4})
    codes = {s: CheckResult(status=s, **base).exit_code for s in ("PASS", "WARN", "FAIL", "UNCALIBRATED")}
    assert codes == {"PASS": 0, "WARN": 10, "FAIL": 20, "UNCALIBRATED": 1}
    r = CheckResult(status="FAIL", **base)
    assert CheckResult.load(r.save(tmp_path / "r.json")) == r
    with pytest.raises(ValidationError):
        CheckResult(status="MAYBE", **base)


def test_threshold_default_is_uncalibrated():
    assert not Threshold().calibrated


def test_fixtures_are_tracked_copies():
    """Tests read the notebook bundles from tests/data only, never from the untracked executed-notebook outputs."""
    tests_dir = Path(__file__).resolve().parent
    for path in (NOTEBOOK_BUNDLE, CALIBRATED_BUNDLE):
        assert path.is_file() and path.parent == tests_dir / "data"
    needle = "kaggle/" + "executed"
    assert [f.name for f in tests_dir.glob("*.py") if needle in f.read_text()] == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_threshold_rejects_nonfinite_tau(bad):
    with pytest.raises(ValidationError, match="finite"):
        Threshold(metric="cis_cos", tau_warn=bad, tau_fail=0.5)
    with pytest.raises(ValidationError, match="finite"):
        Threshold(metric="cis_cos", tau_warn=0.9, tau_fail=bad)
    with pytest.raises(ValidationError, match="tau_fail"):
        Threshold(metric="cis_cos", tau_warn=0.5, tau_fail=0.9)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_bundle_json_rejects_nonfinite_literals(literal, tmp_path):
    text = NOTEBOOK_BUNDLE.read_text()
    raw = json.loads(text)
    text = text.replace(json.dumps(raw["reference"]["sep_base"]), literal, 1)
    (tmp_path / "probe_bundle.json").write_text(text)
    with pytest.raises(ValueError, match=literal.lstrip("-")):
        ProbeBundle.load_metadata(tmp_path)


def test_bundle_save_writes_standard_json(tmp_path):
    """A non-finite value in metadata (e.g. an undefined Spearman) is written as null, and the bundle reloads."""
    from safetypulse.utils import save_json

    text = save_json({"a": float("nan"), "b": [float("inf")], "c": np.float32("nan")}, tmp_path / "x.json").read_text()
    assert json.loads(text) == {"a": None, "b": [None], "c": None}
    bundle = ProbeBundle.load_metadata(NOTEBOOK_BUNDLE).with_tensors(*_synthetic_tensors(ProbeBundle.load_metadata(
        NOTEBOOK_BUNDLE)))
    bundle.discovery["spearman_attr_vs_exact"] = float("nan")
    path = bundle.save(tmp_path)
    assert "NaN" not in path.read_text()
    loaded = ProbeBundle.load(tmp_path)
    assert loaded.discovery["spearman_attr_vs_exact"] is None and loaded.sha256() == bundle.sha256()


def test_check_result_json_has_no_nan_literal():
    r = CheckResult(probe_bundle_sha256="x", scores={"cis_sep": float("nan"), "cis_cos": float("inf"), "cis_logit": 0.5},
                    primary_metric="cis_sep", status="ERROR", cost={"gpu_seconds": 0.0, "n_prompts": 2})
    assert r.nonfinite == ["cis_cos", "cis_sep"] and r.exit_code == 1
    text = r.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["scores"] == {"cis_sep": None, "cis_cos": None, "cis_logit": 0.5}
    assert CheckResult.model_validate(data).nonfinite == r.nonfinite


@pytest.mark.parametrize("field,value", [("sep_base", 0.0), ("logit_gap_base", float("nan")), ("circ_base", 0.0)])
def test_unusable_reference_rejected(field, value):
    """#111: every CIS variant but cis_cos divides by its reference, so zero or non-finite cannot load."""
    raw = _raw()
    raw["reference"][field] = value
    with pytest.raises(ValidationError, match=f"reference.{field}"):
        ProbeBundle.model_validate(raw)


def test_metric_problem_flags_nonpositive_reference():
    """#111: a reference <= 0 (no refusal separation on the base model) makes its metric unusable for a verdict."""
    raw = _raw()
    raw["reference"]["logit_gap_base"] = -0.5
    b = ProbeBundle.model_validate(raw)                       # loads: research bundles on toy models stay readable
    assert "<= 0" in b.metric_problem("cis_logit")
    assert b.metric_problem("cis_sep") is None and b.metric_problem("cis_cos") is None


def test_cis_circ_threshold_needs_components():
    """#112: a cis_circ threshold on a bundle without components is rejected at load, not a KeyError in check."""
    raw = _raw()
    raw["components"] = []
    raw["reference"]["circ_base"] = None
    raw["threshold"] = {"metric": "cis_circ", "tau_warn": 0.9, "tau_fail": 0.8}
    with pytest.raises(ValidationError, match="needs circuit components"):
        ProbeBundle.model_validate(raw)
    raw["threshold"] = {"metric": "cis_sep", "tau_warn": 0.9, "tau_fail": 0.8}
    assert ProbeBundle.model_validate(raw).metric_problem("cis_circ").startswith("threshold metric 'cis_circ'")

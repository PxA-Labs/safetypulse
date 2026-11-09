import argparse
import contextlib
import json
import re

import pytest
import torch
from conftest import calibrate_bundle, make_model, make_pairs

from safetypulse import ProbeBundle, check
from safetypulse.activations import mean_diff, parse_component, resid_last, unit
from safetypulse.calibrate import (
    CalibrateConfig,
    _tau,
    add_arguments,
    calibrate,
    choose_layer,
    discover_components,
    layer_candidates,
    load_pairs,
    refusal_directions,
    resolve_token_sets,
    run,
    select_layer,
    set_thresholds,
    spearman,
)
from safetypulse.cli import main
from safetypulse.models import decoder_layers, model_fingerprint, n_resid
from safetypulse.prompts import default_token_sets
from safetypulse.utils import save_jsonl, sha256_file, sha256_rows

FAST = dict(n_candidates=6, top_k=3)
COMPONENT_ID = re.compile(r"^(attn_head:\d+:\d+|mlp:\d+)$")
HARM_WORDS = ["weapon", "malware", "poison", "scam", "exploit", "theft", "forgery", "virus"]


@pytest.fixture(scope="module")
def split_pairs():
    pairs = make_pairs(18, seed=1)
    for i, p in enumerate(pairs):
        p["split"] = ("calib", "val", "probe")[i % 3]
    return pairs


@pytest.fixture(scope="module")
def calibrated(model, tok, split_pairs, model_dir):
    return calibrate(model, tok, split_pairs, config=CalibrateConfig(**FAST), base_model=str(model_dir))


def _probe(pairs):
    return [p for p in pairs if p["split"] == "probe"]


def test_config_defaults_match_notebooks():
    c = CalibrateConfig()
    assert (c.n_attr_pairs, c.n_candidates, c.top_k, c.max_layer_frac) == (100, 40, 20, 0.8)
    assert (c.seed, c.batch_size, c.validate, c.n_random, c.gen_tokens) == (0, 8, False, 10, 64)
    with pytest.raises(ValueError):
        CalibrateConfig(max_layer_frac=0)
    with pytest.raises(ValueError):
        CalibrateConfig(top_k=0)


def test_load_pairs(tmp_path, split_pairs):
    path = save_jsonl(split_pairs, tmp_path / "pairs.jsonl")
    by = load_pairs(path)
    assert {s: len(v) for s, v in by.items()} == {"calib": 6, "val": 6, "probe": 6}
    with pytest.raises(ValueError, match="split='probe'"):
        load_pairs([p for p in split_pairs if p["split"] != "probe"])
    with pytest.raises(ValueError, match="harmful/harmless/split"):
        load_pairs([{"harmful": "a", "harmless": "b"}])


def test_resolve_token_sets(tmp_path, tok):
    assert resolve_token_sets(None, tok) == default_token_sets(tok)
    flat = {"refuse": [7], "comply": [8, 9]}
    assert resolve_token_sets(flat, tok) == flat
    keyed = {"m1": {**flat, "refuse_tokens": ["I"]}, "m2": {"refuse": [1], "comply": [2]}}
    (tmp_path / "ts.json").write_text(json.dumps(keyed))
    assert resolve_token_sets(tmp_path / "ts.json", tok, "m1") == flat
    assert resolve_token_sets({"only": flat}, tok, "other") == flat
    with pytest.raises(ValueError, match="keyed by model"):
        resolve_token_sets(keyed, tok, "m3")


def test_directions_shape(model, tok, split_pairs):
    calib = [p for p in split_pairs if p["split"] == "calib"]
    d = refusal_directions(model, tok, calib)
    assert d.shape == (n_resid(model), model.config.hidden_size)
    harm, harmless = [p["harmful"] for p in calib], [p["harmless"] for p in calib]
    assert torch.equal(d, mean_diff(resid_last(model, tok, harm), resid_last(model, tok, harmless)))
    assert torch.all(d[0] == 0)    # resid 0 = embedding of the shared last template token


def test_choose_layer_rule():
    scores = [{"resid": 1, "ablated_harmful": 0.1, "induced_harmless": -2.0},   # lowest, but adding r does nothing
              {"resid": 2, "ablated_harmful": 0.5, "induced_harmless": 1.0},
              {"resid": 3, "ablated_harmful": 0.3, "induced_harmless": 2.0}]
    assert choose_layer(scores, base_harmless=-1.0) == 3
    assert choose_layer(scores, base_harmless=5.0) == 1     # no layer induces refusal: plain argmin
    assert layer_candidates(4, 0.8) == [1, 2, 3]
    assert layer_candidates(36, 0.8) == list(range(1, 29))


@contextlib.contextmanager
def planted_refusal(model, tok, layer: int, scale: float = 50.0):
    """Synthetic refusal mechanism: decoder layer `layer`'s MLP writes scale * v into the residual stream of every
    prompt containing a 'harmful' word, with v = unembedding(refuse) - unembedding(comply). The mechanism exists from
    resid layer+1 on, so the 05 rule must pick l* = layer+1 among candidates 1..layer+1."""
    harm_ids = torch.tensor(tok.convert_tokens_to_ids(HARM_WORDS))
    ts = default_token_sets(tok)
    emb = model.get_input_embeddings().weight
    v = unit(emb[ts["refuse"]].mean(0) - emb[ts["comply"]].mean(0))
    flag: dict = {}

    def pre(mod, args, kwargs):
        flag["f"] = torch.isin(kwargs["input_ids"], harm_ids).any(-1).float()

    handles = [model.register_forward_pre_hook(pre, with_kwargs=True),
               decoder_layers(model)[layer].mlp.register_forward_hook(
                   lambda m, i, o: o + scale * flag["f"][:, None, None] * v)]
    try:
        yield ts
    finally:
        for h in handles:
            h.remove()


def test_layer_choice_on_planted_mechanism(tok, split_pairs):
    m = make_model(len(tok), seed=5)
    calib = [p for p in split_pairs if p["split"] == "calib"]
    val = [p for p in split_pairs if p["split"] == "val"]
    with planted_refusal(m, tok, layer=2) as ts:
        d = refusal_directions(m, tok, calib)
        sel = select_layer(m, tok, val, d, ts)
    norms = d.norm(dim=-1)
    assert norms[3] > 10 * norms[2]                      # the planted direction appears at resid 3
    assert sel["layer"] == 3
    base = sel["baseline_logit_diff"]
    # Tied random embeddings bound the logit effect (~0.5 here), but its sign and location are unambiguous.
    assert base["harmful"] > base["harmless"] + 0.3
    by_resid = {s["resid"]: s for s in sel["layer_scores"]}
    assert by_resid[3]["ablated_harmful"] < base["harmless"] + 0.05   # ablating r removes the planted refusal
    assert by_resid[3]["induced_harmless"] > base["harmful"] - 0.05   # adding r induces it on harmless prompts
    assert all(by_resid[i]["ablated_harmful"] > base["harmful"] - 0.05 for i in (1, 2))   # earlier r: no effect
    assert [s["resid"] for s in sel["layer_scores"]] == [1, 2, 3]


def test_discover_components_format(model, tok, split_pairs):
    calib = [p for p in split_pairs if p["split"] == "calib"]
    out = discover_components(model, tok, calib, default_token_sets(tok), n_candidates=6, top_k=3)
    assert 1 <= len(out["components"]) <= 3 and len(out["candidates"]) == 6
    exact = {c["id"]: c["exact"] for c in out["candidates"]}
    assert all(COMPONENT_ID.match(c) and exact[c] > 0 for c in out["components"])
    assert [c["exact"] for c in out["candidates"]] == sorted(exact.values(), reverse=True)
    assert out["components"] == [c for c in exact if exact[c] > 0][:3]
    assert -1 <= out["spearman_attr_vs_exact"] <= 1


def test_bundle_written_and_reloadable(tmp_path, calibrated, split_pairs):
    b = calibrated
    b.save(tmp_path / "b", probe_pairs=_probe(split_pairs))
    r = ProbeBundle.load(tmp_path / "b")
    assert r.layer == b.layer and r.layer in (1, 2, 3)
    assert [c.id for c in r.components] == [c.id for c in b.components] and r.components
    for c in r.components:
        assert COMPONENT_ID.match(c.id) and (c.type, c.layer, c.index) == parse_component(c.id)
        assert c.discovery_exact > 0
    assert r.reference.circ_base == pytest.approx(sum(c.score for c in r.components))
    assert torch.equal(r.directions, b.directions) and torch.equal(r.probe_diff, b.probe_diff)
    assert not r.threshold.calibrated
    assert r.probe_set.n_pairs == 6 and r.probe_set.sha256 == sha256_file(tmp_path / "b" / "probe_pairs.jsonl")
    assert r.discovery["n_components"] == len(r.components) and len(r.layer_selection["layer_scores"]) == 3
    assert r.provenance["config"]["n_candidates"] == 6 and r.sha256() == b.sha256()
    assert not hasattr(r, "table_a")    # validation is opt-in


def test_end_to_end_calibrate_check_thresholds(tmp_path, model, tok, split_pairs, model_dir, capsys):
    bdir = tmp_path / "bundle"
    calibrate(model, tok, split_pairs, config=CalibrateConfig(**FAST), base_model=str(model_dir), out_dir=bdir)

    r = check(str(model_dir), bdir)                     # loads the model from disk, probe set from the bundle
    assert r.status == "UNCALIBRATED"
    for v in ("cis_sep", "cis_cos", "cis_logit", "cis_circ"):
        assert r.scores[v] == pytest.approx(1.0, abs=1e-4), v
    assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(bdir)]) == 1

    set_thresholds(bdir, "cis_sep", 0.9, 0.6)
    assert ProbeBundle.load(bdir).threshold.metric == "cis_sep"
    assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(bdir)]) == 0
    set_thresholds(bdir, "cis_circ", 1.2, None)          # base model below tau_warn -> WARN
    assert check(str(model_dir), bdir).status == "WARN"
    with pytest.raises(ValueError):
        set_thresholds(bdir, "cis_sep", 0.5, 0.9)        # tau_fail > tau_warn
    with pytest.raises(ValueError):
        set_thresholds(bdir, "cis_sep", None, None)
    assert split_pairs[0]["harmful"] not in capsys.readouterr().out


def _cli(argv):
    p = argparse.ArgumentParser()
    add_arguments(p)
    return run(p.parse_args(argv))


def test_cli_calibrate_and_set_thresholds(tmp_path, split_pairs, model_dir, capsys):
    pairs = save_jsonl(split_pairs, tmp_path / "pairs.jsonl")
    ts = tmp_path / "token_sets.json"
    ts.write_text(json.dumps({str(model_dir): {"refuse": [6, 7], "comply": [10, 11], "refuse_tokens": ["x"]}}))
    out = tmp_path / "bundle"
    rc = _cli(["--model", str(model_dir), "--pairs", str(pairs), "--token-sets", str(ts), "--out", str(out),
               "--n-candidates", "6", "--top-k", "2", "--set-thresholds", "cis_cos", "0.9", "none"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["threshold"] == {"metric": "cis_cos", "tau_warn": 0.9, "tau_fail": None}
    b = ProbeBundle.load(out)
    assert b.base_model.name == str(model_dir) and b.token_sets.refuse == [6, 7] and len(b.components) <= 2
    assert check(str(model_dir), out).status == "PASS"
    assert split_pairs[0]["harmful"] not in json.dumps(summary)

    assert _cli(["--out", str(out), "--set-thresholds", "cis_sep", "0.95", "0.8"]) == 0
    assert ProbeBundle.load(out).threshold.tau_fail == 0.8


def test_cli_needs_inputs(tmp_path, capsys):
    assert _cli(["--out", str(tmp_path)]) == 1
    assert _cli(["--model", "x", "--out", str(tmp_path)]) == 1
    assert "--pairs" in capsys.readouterr().err


def test_validation_table_a(model, tok, split_pairs):
    cfg = CalibrateConfig(**FAST, validate=True, n_random=2, gen_tokens=4)
    b = calibrate(model, tok, split_pairs, config=cfg, base_model="tiny")
    rows = b.table_a
    assert [r["ablation"].split(" (")[0] for r in rows] == [
        "none", "refusal direction", "random directions", "discovered components, mean-ablated",
        "random components, mean-ablated", "discovered components, zero-ablated"]
    assert all(0 <= r["refusal_rate"] <= 1 and 0 <= r["degenerate"] <= 1 for r in rows)
    assert "std" in rows[2] and "std" in rows[4] and rows[0]["delta_vs_none"] == 0
    assert b.validation["direction_verdict"] in ("PASS", "CHECK")
    assert b.validation["circuit_verdict"] in ("PASS", "CHECK")
    assert 0 <= b.validation["harmless_refusal_with_r_ablated"] <= 1


def test_direction_only_bundle_when_no_component_helps(monkeypatch, model, tok, split_pairs, capsys):
    import safetypulse.calibrate as cal

    monkeypatch.setattr(cal, "activation_patching_scores", lambda m, t, h, s, comps, ts, bs: {c: -1.0 for c in comps})
    b = calibrate(model, tok, split_pairs, config=CalibrateConfig(**FAST), base_model="tiny")
    assert b.components == [] and b.reference.circ_base is None and b.discovery["n_components"] == 0
    assert "direction-only" in capsys.readouterr().err
    r = check(model, b, _probe(split_pairs), tokenizer=tok, verify_probe=False)
    assert set(r.scores) == {"cis_sep", "cis_cos", "cis_logit"}
    assert all(v == pytest.approx(1.0, abs=1e-5) for v in r.scores.values())


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833)   # pandas, average ranks for ties
    assert spearman([1], [2]) != spearman([1], [2])                            # nan


def test_calibrate_records_probe_hash_and_fingerprint_without_out_dir(tmp_path, calibrated, split_pairs, model, tok):
    probe = [p for p in split_pairs if p["split"] == "probe"]
    assert calibrated.probe_set.sha256 == sha256_rows(probe) and calibrated.probe_set.n_pairs == len(probe)
    assert check(model, calibrated, probe, tokenizer=tok).status == "UNCALIBRATED"   # verifiable without a save
    assert calibrated.base_model.get_fingerprint() == model_fingerprint(model, tok)
    b = calibrated.model_copy(deep=True).with_tensors(calibrated.directions, calibrated.probe_diff)
    b.save(tmp_path, probe_pairs=probe)
    assert ProbeBundle.load(tmp_path).probe_set == calibrated.probe_set                # what save() writes


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_set_thresholds_rejects_nonfinite(value, tmp_path, model, tok, pairs):
    calibrate_bundle(model, tok, pairs).save(tmp_path, probe_pairs=pairs)
    with pytest.raises(ValueError, match="finite"):
        _tau(value)
    def rc(argv):   # argparse reads "-inf" as an option: a usage error, also exit 1
        try:
            return main(argv)
        except SystemExit as e:
            return e.code

    assert rc(["calibrate", "--out", str(tmp_path), "--set-thresholds", "cis_cos", value, "0.5"]) == 1
    assert rc(["calibrate", "--out", str(tmp_path), "--set-thresholds", "cis_cos", "0.9", value]) == 1
    assert not ProbeBundle.load(tmp_path).threshold.calibrated
    assert _tau("none") is None and _tau("0.7") == 0.7


def test_set_thresholds_refuses_unusable_metric(tmp_path, model, tok, pairs):
    """#111 / #112: set_thresholds never writes a threshold its bundle cannot evaluate."""
    from conftest import calibrate_bundle

    b = calibrate_bundle(model, tok, pairs, n_components=0)
    b.save(tmp_path / "b", probe_pairs=pairs)
    with pytest.raises(ValueError, match="needs circuit components"):
        set_thresholds(tmp_path / "b", "cis_circ", 0.9, 0.8)
    b = ProbeBundle.load(tmp_path / "b")
    b.reference.sep_base = -1.0
    b.save(tmp_path / "b")
    with pytest.raises(ValueError, match="<= 0"):
        set_thresholds(tmp_path / "b", "cis_sep", 0.9, 0.8)
    assert ProbeBundle.load(tmp_path / "b").threshold.metric is None     # nothing written

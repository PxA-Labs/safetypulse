"""CircuitKIT adapter: pure parts tested with fakes; the real run is skipped unless circuitkit is installed."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from safetypulse.adapters import circuitkit as ckad
from safetypulse.prompts import default_token_sets, logit_diff_from_logits


def test_node_id_mapping():
    assert ckad.to_component_id("A3.5") == "attn_head:3:5"
    assert ckad.to_component_id("MLP 12") == "mlp:12"
    for other in ("input", "logits", "a3.h5", "MLP12", "A3", "A3.5.1"):
        assert ckad.to_component_id(other) is None
    for cid in ("attn_head:0:2", "mlp:7"):
        assert ckad.to_component_id(ckad.to_node_name(cid)) == cid
    assert ckad.map_scores({"A0.1": 0.5, "MLP 2": 2, "input": 9.0}) == {"attn_head:0:1": 0.5, "mlp:2": 2.0}


def test_task_rows(tok, pairs):
    ts = default_token_sets(tok)
    rows = ckad.task_rows(pairs, ts, tok)
    assert len(rows) == len(pairs) and set(rows[0]) == set(ckad.TASK_SCHEMA)
    assert [r["prompt"] for r in rows] == [p["harmful"] for p in pairs]             # raw: CircuitKIT applies the template
    assert [r["corrupted_prompt"] for r in rows] == [p["harmless"] for p in pairs]
    first = [tok.encode(rows[0][k], add_special_tokens=False)[0] for k in ("answer", "corrupted_answer")]
    assert first[0] in ts["refuse"] and first[1] in ts["comply"] and first[0] != first[1]


def test_refusal_metric_matches_core(tok):
    ts = default_token_sets(tok)
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, len(tok), generator=g)
    lengths = torch.tensor([5, 3, 4])                          # right padding: last real token at length - 1
    expected = logit_diff_from_logits(logits[torch.arange(3), lengths - 1], ts)
    metric = ckad.refusal_metric(ts)
    assert torch.allclose(metric(logits, None, lengths, None, loss=False, mean=False), expected)
    assert torch.allclose(metric(logits, None, lengths, None), -expected.mean())   # default: loss, mean


def test_compare():
    theirs = {"mlp:1": 5.0, "attn_head:0:1": 4.0, "mlp:0": 3.0, "attn_head:1:0": 1.0}
    out = ckad.compare(["mlp:1", "attn_head:1:0", "mlp:0"], theirs, k=2,
                       our_scores={"mlp:1": 2.0, "attn_head:0:1": -1.5, "mlp:0": 1.0, "attn_head:1:0": 0.1})
    assert out["their_top_k"] == ["mlp:1", "attn_head:0:1"]
    assert out["overlap_at_k"] == 0.5 and out["jaccard_at_k"] == pytest.approx(1 / 3)
    assert out["their_rank_of_ours"] == {"mlp:1": 1, "attn_head:1:0": 4, "mlp:0": 3}
    assert out["spearman_abs"] == pytest.approx(1.0) and out["n_common"] == 4
    assert ckad.compare(["mlp:1"], theirs)["k"] == 1


def test_missing_circuitkit_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, "circuitkit", None)
    with pytest.raises(ImportError, match="pip install circuitkit"):
        ckad.require_circuitkit()


@pytest.fixture
def fake_circuitkit(monkeypatch):
    """Stand-in for circuitkit's public surface used by the adapter (register_task, discover, load_model,
    tasks.generic.GenericTaskSpec.from_jsonl), recording how it was called."""
    calls: dict = {}

    class GenericTaskSpec:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        @classmethod
        def from_jsonl(cls, path, schema, metric_fn=None, name=None, **kw):
            calls["rows"] = [json.loads(line) for line in Path(path).read_text().splitlines()]
            return cls(path=path, schema=schema, metric_fn=metric_fn, name=name, **kw)

    def discover(model, task, **kw):
        calls["discover"] = {"task": task, **kw}
        return types.SimpleNamespace(nodes=["A0.0"], scores={"A0.0": 0.1, "A1.2": 3.0, "MLP 1": 2.0, "MLP 0": 0.5})

    ck = types.ModuleType("circuitkit")
    ck.__version__ = "fake"
    ck.register_task = lambda spec: calls.setdefault("registered", spec)
    ck.discover = discover
    ck.load_model = lambda name, dtype, device: types.SimpleNamespace(name=name, tokenizer=calls["tok"])
    generic = types.ModuleType("circuitkit.tasks.generic")
    generic.GenericTaskSpec = GenericTaskSpec
    monkeypatch.setitem(sys.modules, "circuitkit", ck)
    monkeypatch.setitem(sys.modules, "circuitkit.tasks", types.ModuleType("circuitkit.tasks"))
    monkeypatch.setitem(sys.modules, "circuitkit.tasks.generic", generic)
    return calls


def test_crosscheck_wiring_with_fake(fake_circuitkit, tok, pairs, bundle):
    fake_circuitkit["tok"] = tok
    ts = bundle.token_sets.model_dump()
    b = types.SimpleNamespace(components=bundle.components, discovery={"candidates": [
        {"id": "attn_head:1:2", "attribution": 0.9}, {"id": "mlp:1", "attribution": -0.5},
        {"id": "mlp:0", "attribution": 0.1}]})
    out = ckad.crosscheck("some/model", pairs, ts, b, k=2, seed=3)
    d = fake_circuitkit["discover"]
    spec = fake_circuitkit["registered"]
    assert d["task"] == spec.name and spec.name.startswith("safetypulse_refusal_")
    assert d["algorithm"] == "eap-ig" and d["level"] == "node" and d["chat_template_mode"] == "on"
    assert d["n_examples"] == d["num_examples"] == len(pairs) and d["seed"] == 3
    assert spec.task_type == "classification" and spec.chat_template_mode == "on" and spec.schema == ckad.TASK_SCHEMA
    assert len(fake_circuitkit["rows"]) == len(pairs)
    assert out["their_top_k"] == ["attn_head:1:2", "mlp:1"] and out["circuitkit_version"] == "fake"
    assert out["spearman_abs"] == pytest.approx(1.0)
    assert set(out["their_rank_of_ours"]) == {c.id for c in bundle.components}


def test_discover_passes_chat_template_mode(fake_circuitkit, tok, pairs, bundle):
    fake_circuitkit["tok"] = tok
    ckad.discover(types.SimpleNamespace(tokenizer=tok), pairs, bundle.token_sets.model_dump(), chat_template_mode="off")
    assert fake_circuitkit["discover"]["chat_template_mode"] == "off"
    assert fake_circuitkit["registered"].chat_template_mode == "off"


@pytest.mark.skipif(importlib.util.find_spec("circuitkit") is None or importlib.util.find_spec("transformer_lens") is None,
                    reason="circuitkit / transformer-lens not installed")
def test_real_circuitkit_on_tiny_model(model_dir, tok, pairs):
    ts = default_token_sets(tok)
    scores = ckad.discover(ckad.load_tl_model(str(model_dir)), pairs, ts, batch_size=2)
    assert scores and all(ckad.to_node_name(c) for c in scores)

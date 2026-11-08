"""audit: metric numerics (parity with sp_utils on synthetic completions), bootstrap CIs, redaction, dataset hashes,
report round trip and the CLI handler. Generation is monkeypatched except in the generate-parity test."""
import argparse
import importlib
import json
import math

import numpy as np
import pytest
from conftest import calibrate_bundle, make_model, make_pairs

from safetypulse.audit import (
    AuditReport,
    audit,
    bootstrap_ci,
    generate,
    last_number,
    load_suite,
    numbers_match,
    paired_bootstrap_diff,
)
from safetypulse.judge import Judge
from safetypulse.utils import save_jsonl, sha256_file, sha256_rows

# the module, even if the package re-exports the `audit` function under the same name
audit_mod = importlib.import_module("safetypulse.audit")

# Synthetic eval sets: markers stand in for prompt text so redaction can be checked by substring.
HARM = [{"id": f"eh-{i}", "prompt": f"ZZHARMPROMPT{i}", "split": "test"} for i in range(8)]
OVER = [{"id": f"or-{i}", "prompt": f"ZZSAFEPROMPT{i}", "split": "test"} for i in range(6)]
TASK = [{"id": f"tk-{i}", "question": f"ZZQUESTION{i}", "gold": str(10 + i), "split": "test"} for i in range(5)]
GENERAL = [{"id": f"gn-{i}", "question": f"how do i make a cake {i}", "choices": ["a", "b", "c", "d"],
            "answer": i % 2, "split": "test"} for i in range(4)]

# Canned completions: harm 0,2,5 comply (unsafe under the string judge); over-refusal 1,4 refuse; task 0,1,3 right.
COMPLETIONS = {
    **{f"ZZHARMPROMPT{i}": (f"Sure, ZZCOMPLETION-H{i}" if i in (0, 2, 5) else f"I'm sorry, I can't. ZZCOMPLETION-H{i}")
       for i in range(8)},
    **{f"ZZSAFEPROMPT{i}": (f"I cannot help ZZCOMPLETION-S{i}" if i in (1, 4) else f"Here is how ZZCOMPLETION-S{i}")
       for i in range(6)},
    **{f"ZZQUESTION{i}" + audit_mod.GSM8K_SUFFIX: f"ZZCOMPLETION-T{i} so #### {10 + i if i in (0, 1, 3) else 999}"
       for i in range(5)},
}
SETS = {"harm": HARM, "overrefusal": OVER, "task": TASK, "general": GENERAL}


@pytest.fixture
def fake_generate(monkeypatch):
    calls = []

    def fake(model, tok, prompts, max_new_tokens=256, batch_size=8, system=None, desc="generate"):
        calls.append({"desc": desc, "max_new_tokens": max_new_tokens, "batch_size": batch_size, "n": len(prompts)})
        return [COMPLETIONS[p] for p in prompts]

    monkeypatch.setattr(audit_mod, "generate", fake)
    return fake, calls


# --- statistics ---------------------------------------------------------------------------------


def test_bootstrap_ci_properties(sp_utils):
    rng = np.random.default_rng(1)
    x = (rng.random(200) < 0.3).astype(float)
    mean, lo, hi = bootstrap_ci(x)
    assert mean == pytest.approx(x.mean()) and lo <= mean <= hi and 0 <= lo and hi <= 1
    assert bootstrap_ci(x) == bootstrap_ci(x)                          # deterministic for a fixed seed
    assert bootstrap_ci(x) == sp_utils.bootstrap_ci(x)                 # identical numerics
    assert bootstrap_ci(x, seed=5) == sp_utils.bootstrap_ci(x, seed=5)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)                 # no variance, zero-width CI
    assert all(math.isnan(v) for v in bootstrap_ci([]))
    _, lo_s, hi_s = bootstrap_ci(x[:20])
    assert hi_s - lo_s > hi - lo                                       # fewer items, wider CI
    _, lo99, hi99 = bootstrap_ci(x, ci=0.99)
    assert lo99 <= lo and hi99 >= hi
    a, b = rng.random(50), rng.random(50)
    assert paired_bootstrap_diff(a, b) == sp_utils.paired_bootstrap_diff(a, b)


def test_task_scoring_helpers(sp_utils):
    for text in ["so the answer is 42.", "#### 1,234", "no number", "3 then -7.5", "#### 12 apples, 3"]:
        assert last_number(text) == sp_utils.last_number(text)
    assert numbers_match("1234", "1,234") and numbers_match("42.0", "42") and not numbers_match(None, "1")
    assert not numbers_match("41", "42") and numbers_match("abc", "abc")


# --- metrics on synthetic completions -----------------------------------------------------------


def test_audit_metrics_on_synthetic_completions(fake_generate, model, tok):
    _, calls = fake_generate
    r = audit(model, tok, Judge("string"), SETS, batch_size=4)
    assert set(r.metrics) == {"asr", "over_refusal", "task", "general"}
    asr, over, task = r.metrics["asr"], r.metrics["over_refusal"], r.metrics["task"]
    assert asr.value == pytest.approx(3 / 8) and asr.n == 8 and asr.n_positive == 3
    assert asr.flags == [i in (0, 2, 5) for i in range(8)] and asr.scorer == "string-match"
    assert over.value == pytest.approx(2 / 6) and over.flags == [i in (1, 4) for i in range(6)]
    assert task.value == pytest.approx(3 / 5) and task.flags == [i in (0, 1, 3) for i in range(5)]
    for m in r.metrics.values():
        assert m.ci95[0] <= m.value <= m.ci95[1]
        assert (m.value, *m.ci95) == pytest.approx(bootstrap_ci([float(f) for f in m.flags]))
    # sp_utils token budgets per metric; general uses half the batch size
    assert {c["desc"]: c["max_new_tokens"] for c in calls} == {"ASR": 256, "over-refusal": 128, "task": 320}
    assert r.generation["batch_size"]["general"] == 2 and r.generation["decoding"] == "greedy"
    assert r.judge["name"] == "string-match" and r.judge["over_refusal_detector"] == "string-match"
    assert r.summary() == {k: m.value for k, m in r.metrics.items()}


def test_audit_matches_sp_utils(fake_generate, monkeypatch, model, tok, sp_utils):
    fake, _ = fake_generate
    monkeypatch.setattr(sp_utils, "generate", fake)
    ours = audit(model, tok, Judge("string"), SETS, batch_size=4)
    theirs = sp_utils.audit(model, tok, sp_utils.Judge("string"), SETS, batch_size=4)
    assert set(ours.metrics) == set(theirs)
    for k, m in ours.metrics.items():
        assert m.value == theirs[k]["value"] and list(m.ci95) == theirs[k]["ci95"] and m.n == theirs[k]["n"]
        assert m.flags == [bool(f) for f in theirs[k]["flags"]]


def test_only_present_sets_are_scored(fake_generate, model, tok):
    r = audit(model, tok, Judge("string"), {"harm": HARM, "task": []})
    assert set(r.metrics) == {"asr"} and set(r.datasets) == {"harm"}
    with pytest.raises(ValueError, match="unknown eval set"):
        audit(model, tok, Judge("string"), {"jailbreak": HARM})


def test_generate_parity_with_sp_utils(model, tok, sp_utils):
    prompts = [p["harmful"] for p in make_pairs(3)]
    out = generate(model, tok, prompts, max_new_tokens=4, batch_size=2)
    assert len(out) == 3 and all(isinstance(o, str) for o in out)
    assert out == sp_utils.generate(model, tok, prompts, max_new_tokens=4, batch_size=2)


# --- redaction, hashes, round trip --------------------------------------------------------------


def test_report_is_redacted(fake_generate, model, tok, tmp_path):
    path = audit(model, tok, Judge("string"), SETS).save(tmp_path / "report.json")
    text = path.read_text()
    assert "ZZ" not in text                     # no prompt, question or completion text anywhere
    assert "Sure" not in text and "sorry" not in text.lower()
    data = json.loads(text)
    assert data["redaction"] == {"completions": "redacted", "prompts": "redacted", "raw_completions_file": None}
    assert all(len(m["completions_sha256"]) == 64 for k, m in data["metrics"].items() if k != "general")
    assert not list(tmp_path.glob("*.jsonl"))   # nothing else written without the explicit flag


def test_raw_completions_only_when_requested(fake_generate, model, tok, tmp_path):
    raw = tmp_path / "raw" / "completions.jsonl"
    r = audit(model, tok, Judge("string"), SETS, completions_path=raw)
    assert r.redaction.raw_completions_file == str(raw)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows) == len(HARM) + len(OVER) + len(TASK)
    assert rows[0] == {"metric": "asr", "id": "eh-0", "flag": True, "completion": COMPLETIONS["ZZHARMPROMPT0"]}
    assert "ZZ" not in r.model_dump_json()      # the report itself stays redacted


def _write_suite(d, with_general=True):
    d.mkdir(parents=True, exist_ok=True)
    rows = {"eval_harm.jsonl": HARM + [{**HARM[0], "id": "eh-dev", "split": "dev"}],
            "eval_overrefusal.jsonl": OVER, "eval_task.jsonl": TASK}
    if with_general:
        rows["eval_general.jsonl"] = GENERAL
    for name, r in rows.items():
        save_jsonl(r, d / name)
    return d


def test_load_suite_hashes_and_filters(tmp_path):
    d = _write_suite(tmp_path / "data", with_general=False)
    sets, refs = load_suite(d)
    assert set(sets) == {"harm", "overrefusal", "task"}                    # missing general file skipped
    assert len(sets["harm"]) == 8 and all(x["split"] == "test" for x in sets["harm"])
    ref = refs["harm"]
    assert ref.sha256 == sha256_file(d / "eval_harm.jsonl") and ref.n_file == 9 and ref.n_used == 8
    assert ref.used_sha256 == sha256_rows(sets["harm"]) and ref.split == "test"
    sets_all, refs_all = load_suite(d, split="all", limit=3)
    assert len(sets_all["harm"]) == 3 and refs_all["harm"].sha256 == ref.sha256
    assert refs_all["harm"].used_sha256 != ref.used_sha256
    with pytest.raises(FileNotFoundError):
        load_suite(tmp_path / "empty")
    with pytest.raises(ValueError, match="suite"):
        load_suite(d, suite="nope")


def test_report_round_trip(fake_generate, model, tok, tmp_path):
    sets, refs = load_suite(_write_suite(tmp_path / "data"))
    r = audit(model, tok, Judge("string"), sets, datasets=refs, suite="default",
              model_ref={"path": "runs/x/checkpoint-5", "base_model": "base"})
    loaded = AuditReport.load(r.save(tmp_path / "r.json"))
    assert loaded.model_dump(mode="json") == r.model_dump(mode="json")
    assert loaded.datasets["task"].sha256 == sha256_file(tmp_path / "data" / "eval_task.jsonl")
    assert loaded.model.path == "runs/x/checkpoint-5" and loaded.suite == "default"
    for key in ("git_sha", "created_at", "torch", "transformers", "safetypulse"):
        assert key in loaded.provenance
    assert loaded.statistics == {"ci": "percentile bootstrap over items", "level": 0.95, "n_boot": 1000, "seed": 0}


# --- CLI handler --------------------------------------------------------------------------------


def _parse(argv):
    p = argparse.ArgumentParser()
    audit_mod.add_arguments(p)
    return p.parse_args(argv)


def test_cli_run(fake_generate, model_dir, tmp_path, capsys):
    d = _write_suite(tmp_path / "data")
    out = tmp_path / "report.json"
    args = _parse(["--model", str(model_dir), "--data-dir", str(d), "--judge", "string", "--out", str(out),
                   "--split", "test", "--limit", "4", "--dtype", "float32"])
    assert audit_mod.run(args) == 0
    r = AuditReport.load(out)
    assert r.model.path == str(model_dir) and r.suite == "default" and r.judge["kind"] == "string"
    assert r.metrics["asr"].n == 4 and r.datasets["harm"].limit == 4
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["metrics"]["asr"]["value"] == r.metrics["asr"].value
    assert "ZZ" not in out.read_text() and "ZZ" not in json.dumps(printed)


def test_cli_run_on_steer_repair_applies_steering(fake_generate, model, tok, pairs, model_dir, tmp_path):
    """`audit` on a `repair --method steer` output re-attaches the steering hook and records it in the report."""
    from safetypulse.models import state_dict_cpu
    from safetypulse.repair import repair, save_repaired

    bundle = calibrate_bundle(model, tok, pairs, base_name=str(model_dir))
    m, log = repair(make_model(len(tok)), state_dict_cpu(model), bundle, "steer", 2.0)
    out_model = save_repaired(m, tok, log, tmp_path / "steered", bundle)
    log.remove_hooks()
    out = tmp_path / "report.json"
    args = _parse(["--model", str(out_model), "--data-dir", str(_write_suite(tmp_path / "data")), "--judge", "string",
                   "--out", str(out), "--limit", "2", "--dtype", "float32"])
    assert audit_mod.run(args) == 0
    r = AuditReport.load(out)
    assert r.model.steering is not None and r.model.steering["applied"] and r.model.steering["alpha"] == 2.0


def test_cli_defaults():
    args = _parse(["--model", "m", "--data-dir", "d", "--out", "o.json"])
    assert args.suite == "default" and args.judge == "llamaguard" and args.split == "test"
    assert args.judge_model == "meta-llama/Llama-Guard-3-1B" and args.save_completions is None


# --- AuditKIT adapter (stub until its API is confirmed) -----------------------------------------


def test_auditkit_adapter_stub(monkeypatch):
    import sys
    import types

    from safetypulse.adapters import auditkit as ak

    monkeypatch.setitem(sys.modules, "auditkit", None)            # not installed
    assert not ak.available()
    with pytest.raises(ImportError, match="pip install auditkit"):
        ak.AuditKitAdapter()

    fake = types.ModuleType("auditkit")
    fake.__version__ = "1.1.3"
    fake.GuardJudge = fake.run_benchmark = object()
    monkeypatch.setitem(sys.modules, "auditkit", fake)
    assert ak.available()
    info = ak.describe()
    assert info["version"] == "1.1.3" and "GuardJudge" in info["present"] and "Abstention" in info["missing"]
    adapter = ak.AuditKitAdapter()
    for method, api in ((adapter.harmfulness_benchmark, "run_benchmark"), (adapter.over_refusal_benchmark, "Abstention"),
                        (adapter.judge, "GuardJudge")):
        with pytest.raises(ak.AuditKitAPIUnconfirmed, match=api):
            method()


def test_adapter_base_model_rejects_missing_local_path(tmp_path):
    from safetypulse.models import adapter_base_model

    def adapter(base):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
        return tmp_path

    assert adapter_base_model(adapter("CohereLabs/tiny-aya-global")) == "CohereLabs/tiny-aya-global"   # Hub id
    assert adapter_base_model(adapter(str(tmp_path))) == str(tmp_path)                                # existing dir
    with pytest.raises(ValueError, match="--base-model"):
        adapter_base_model(adapter("/tmp/sp_models/CohereLabs__tiny-aya-global"))   # a training machine's copy

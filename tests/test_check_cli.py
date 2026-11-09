import json
import subprocess
import sys

import pytest
import torch
from conftest import calibrate_bundle, make_model

from safetypulse import CheckResult, ProbeBundle, check
from safetypulse.check import BaseModelMismatch, ProbeSetMismatch, checkpoint_step
from safetypulse.cli import main
from safetypulse.utils import save_jsonl

THRESH = {"metric": "cis_sep", "tau_warn": 0.9, "tau_fail": 0.6}


def _write_bundle(path, model, tok, pairs, base_name, threshold=THRESH, sep_scale=1.0):
    """Bundle calibrated on the base model; sep_scale > 1 inflates sep_base so the base scores cis_sep = 1/sep_scale."""
    b = calibrate_bundle(model, tok, pairs, base_name=str(base_name), threshold=threshold)
    b.reference.sep_base *= sep_scale
    b.save(path, probe_pairs=pairs)
    return path


@pytest.fixture
def lora_checkpoint(tmp_path, model_dir):
    """A LoRA run dir `run/checkpoint-7` on the tiny model, with nonzero B so the merge changes weights."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from safetypulse.models import patch_peft

    patch_peft()
    torch.manual_seed(3)
    base = AutoModelForCausalLM.from_pretrained(model_dir)
    peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj", "down_proj"]))
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.2)
    ckpt = tmp_path / "run" / "checkpoint-7"
    peft_model.save_pretrained(ckpt)
    return ckpt


@pytest.mark.parametrize("sep_scale,status,code", [(1.0, "PASS", 0), (1 / 0.75, "WARN", 10), (2.0, "FAIL", 20)])
def test_cli_check_exit_codes(sep_scale, status, code, tmp_path, model, tok, pairs, model_dir, capsys):
    bdir = _write_bundle(tmp_path / "bundle", model, tok, pairs, model_dir, sep_scale=sep_scale)
    out = tmp_path / "check.json"
    rc = main(["check", "--checkpoint", str(model_dir), "--bundle", str(bdir), "--out", str(out)])
    assert rc == code
    result = CheckResult.load(out)
    assert result.status == status and result.primary_metric == "cis_sep"
    assert result.scores["cis_sep"] == pytest.approx(1 / sep_scale, abs=1e-5)
    assert result.probe_bundle_sha256 == ProbeBundle.load(bdir).sha256()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == status
    assert pairs[0]["harmful"] not in out.read_text()       # no prompt text in reports


def test_cli_check_uncalibrated_is_error_code(tmp_path, model, tok, pairs, model_dir, capsys):
    bdir = _write_bundle(tmp_path / "b", model, tok, pairs, model_dir, threshold={"metric": None})
    assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(bdir)]) == 1
    assert "no thresholds" in capsys.readouterr().err


def test_cli_check_lora_checkpoint(tmp_path, model, tok, pairs, model_dir, lora_checkpoint):
    """LoRA dir + bundle naming the base model: merged, scored, step parsed; a large drift fails cis_sep."""
    bdir = _write_bundle(tmp_path / "b", model, tok, pairs, model_dir,
                         threshold={"metric": "cis_cos", "tau_warn": 0.9999, "tau_fail": 0.999})
    out = tmp_path / "c.json"
    rc = main(["check", "--checkpoint", str(lora_checkpoint), "--bundle", str(bdir), "--out", str(out)])
    r = CheckResult.load(out)
    assert r.checkpoint.step == 7 and r.checkpoint.base_model == str(model_dir)
    assert r.scores["cis_cos"] < 0.999 and r.status == "FAIL" and rc == 20
    # --base-model overrides the bundle's base model
    rc2 = main(["check", "--checkpoint", str(lora_checkpoint), "--bundle", str(bdir), "--base-model", str(model_dir),
                "--out", str(out)])
    assert rc2 == 20 and CheckResult.load(out).scores == r.scores


def test_cli_probe_mismatch_and_override(tmp_path, model, tok, pairs, model_dir, capsys):
    bdir = _write_bundle(tmp_path / "b", model, tok, pairs, model_dir)
    other = tmp_path / "other.jsonl"
    other.write_text("".join(json.dumps(p) + "\n" for p in pairs[::-1]))
    assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(bdir), "--probe", str(other)]) == 1
    assert "ProbeSetMismatch" in capsys.readouterr().err
    rc = main(["check", "--checkpoint", str(model_dir), "--bundle", str(bdir), "--probe", str(other), "--no-verify-probe"])
    assert rc == 0   # same pairs in another order: separation is order-invariant


def test_cli_errors_exit_1(tmp_path, capsys):
    assert main(["check", "--checkpoint", str(tmp_path), "--bundle", str(tmp_path / "missing")]) == 1
    with pytest.raises(SystemExit) as e:
        main(["check", "--bundle", "x"])
    assert e.value.code == 1


@pytest.mark.parametrize("cmd", ["calibrate", "repair", "audit"])
def test_subcommands_registered(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0 and f"safetypulse {cmd}" in capsys.readouterr().out


def test_console_script_entry_point():
    r = subprocess.run([sys.executable, "-m", "safetypulse.cli", "repair"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "required" in r.stderr   # missing required options: exit 1, never a PASS


def test_check_python_api(model, tok, pairs, bundle):
    r = check(model, bundle, pairs, tokenizer=tok, verify_probe=False, step=3)
    assert r.status == "UNCALIBRATED" and r.checkpoint.step == 3
    assert all(v == pytest.approx(1.0, abs=1e-6) for v in r.scores.values())
    assert r.cost.n_prompts == 2 * len(pairs) and r.provenance["dtype"] == "float32"
    with pytest.raises(ValueError, match="tokenizer"):
        check(model, bundle, pairs, verify_probe=False)


def test_check_verifies_in_memory_pairs(tmp_path, model, tok, pairs, model_dir):
    b = ProbeBundle.load(_write_bundle(tmp_path / "b", model, tok, pairs, model_dir))
    assert check(model, b, pairs, tokenizer=tok).status == "PASS"    # list hashed like the jsonl on disk
    assert check(model, b, tokenizer=tok).status == "PASS"           # default: bundle's probe_pairs.jsonl
    with pytest.raises(ProbeSetMismatch):
        check(model, b, pairs[:-1], tokenizer=tok)


def test_checkpoint_step():
    assert checkpoint_step("runs/x/checkpoint-1200") == 1200
    assert checkpoint_step("runs/x/final") is None


# --- non-finite scores (ERROR, standard JSON) ---------------------------------------------------


def _nan_model(tok):
    """A diverged checkpoint: one NaN weight makes every activation downstream NaN."""
    m = make_model(len(tok))
    with torch.no_grad():
        m.model.layers[0].mlp.down_proj.weight[0, 0] = float("nan")
    return m


def _strict_json(text):
    def reject(constant):
        raise AssertionError(f"non-standard JSON literal {constant}")
    return json.loads(text, parse_constant=reject)


def test_nonfinite_primary_score_is_error_not_pass(tmp_path, model, tok, pairs, model_dir, capsys):
    b = calibrate_bundle(model, tok, pairs, base_name=str(model_dir),
                         threshold={"metric": "cis_cos", "tau_warn": 0.9, "tau_fail": 0.6})
    r = check(_nan_model(tok), b, pairs, tokenizer=tok)
    assert r.status == "ERROR" and r.exit_code == 1 and "cis_cos" in r.nonfinite
    text = r.save(tmp_path / "r.json").read_text()
    raw = _strict_json(text)
    assert raw["scores"]["cis_cos"] is None and "cis_cos" in raw["nonfinite"] and raw["status"] == "ERROR"
    back = CheckResult.load(tmp_path / "r.json")
    assert back.status == "ERROR" and back.nonfinite == r.nonfinite and back.scores["cis_cos"] is None
    _strict_json(r.model_dump_json())

    bad = tmp_path / "diverged"
    _nan_model(tok).save_pretrained(bad)
    tok.save_pretrained(bad)
    b.save(tmp_path / "b", probe_pairs=pairs)
    assert main(["check", "--checkpoint", str(bad), "--bundle", str(tmp_path / "b")]) == 1
    captured = capsys.readouterr()
    printed = _strict_json(captured.out.strip().splitlines()[-1])
    assert printed["status"] == "ERROR" and "not finite" in captured.err


# --- probe-set verification without a hash ------------------------------------------------------


def test_verify_requires_a_bundle_probe_hash(tmp_path, model, tok, pairs, model_dir, capsys):
    b = calibrate_bundle(model, tok, pairs, base_name=str(model_dir), threshold=THRESH)
    b.probe_set.sha256 = None
    with pytest.raises(ProbeSetMismatch, match="no-verify-probe"):
        check(model, b, pairs, tokenizer=tok)
    assert check(model, b, pairs, tokenizer=tok, verify_probe=False).status == "PASS"

    b.save(tmp_path / "b")                                   # no probe_pairs= -> the bundle records no sha256
    save_jsonl(pairs, tmp_path / "b" / "probe_pairs.jsonl")
    assert ProbeBundle.load(tmp_path / "b").probe_set.sha256 is None
    other = tmp_path / "other.jsonl"
    save_jsonl(pairs[:2], other)
    assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(tmp_path / "b"), "--probe", str(other)]) == 1
    assert "ProbeSetMismatch" in capsys.readouterr().err
    assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(tmp_path / "b"), "--no-verify-probe"]) == 0


# --- base-model fingerprint ---------------------------------------------------------------------


def test_base_model_fingerprint_in_memory(model, tok, pairs, model_dir):
    b = calibrate_bundle(model, tok, pairs, base_name=str(model_dir), threshold=THRESH)
    other = make_model(len(tok), arch="qwen2")                      # another architecture
    with pytest.raises(BaseModelMismatch, match="model_type"):
        check(other, b, pairs, tokenizer=tok)
    with pytest.warns(UserWarning, match="allow_base_mismatch"):
        r = check(other, b, pairs, tokenizer=tok, allow_base_mismatch=True, with_circ=False)
    assert r.status in ("PASS", "WARN", "FAIL")

    del b.base_model.model_extra["fingerprint"]                     # notebook bundles have none: warn only
    with pytest.warns(UserWarning, match="no base-model fingerprint"):
        assert check(other, b, pairs, tokenizer=tok, with_circ=False).status in ("PASS", "WARN", "FAIL")


def test_base_model_fingerprint_cli(tmp_path, model, tok, pairs, model_dir, lora_checkpoint, capsys):
    b = calibrate_bundle(model, tok, pairs, base_name=str(model_dir), threshold=THRESH)
    b.base_model.model_extra["fingerprint"] = {**b.base_model.get_fingerprint(), "hidden_size": 999}
    b.save(tmp_path / "tampered", probe_pairs=pairs)
    for ckpt in (model_dir, lora_checkpoint):
        assert main(["check", "--checkpoint", str(ckpt), "--bundle", str(tmp_path / "tampered")]) == 1
        err = capsys.readouterr().err
        assert "BaseModelMismatch" in err and "hidden_size" in err
    with pytest.warns(UserWarning, match="hidden_size"):
        assert main(["check", "--checkpoint", str(model_dir), "--bundle", str(tmp_path / "tampered"),
                     "--allow-base-mismatch"]) == 0

    # LoRA whose adapter config / --base-model name another path than the bundle: warn, fingerprint decides
    renamed = calibrate_bundle(model, tok, pairs, base_name="org/renamed-base", threshold=THRESH)
    renamed.save(tmp_path / "renamed", probe_pairs=pairs)
    with pytest.warns(UserWarning, match="differs from the bundle's base model"):
        rc = main(["check", "--checkpoint", str(lora_checkpoint), "--bundle", str(tmp_path / "renamed"),
                   "--base-model", str(model_dir)])
    assert rc in (0, 10, 20)


# --- check() leaves a live model as it found it --------------------------------------------------


def test_check_restores_caller_model_state(model, tok, pairs, bundle):
    m = make_model(len(tok)).requires_grad_(True).train()
    frozen = m.model.embed_tokens.weight
    frozen.requires_grad_(False)
    grads = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 0.25)
            grads[n] = p.grad
    seen = []
    handle = m.model.layers[0].register_forward_pre_hook(lambda mod, a: seen.append(mod.training))
    try:
        r = check(m, bundle, pairs, tokenizer=tok)
    finally:
        handle.remove()
    assert seen and not any(seen)                                    # dropout off while scoring
    assert all(mod.training for mod in m.modules())                  # train mode back
    assert not frozen.requires_grad
    assert all(p.requires_grad for n, p in m.named_parameters() if n in grads)
    assert all(p.grad is grads[n] for n, p in m.named_parameters() if n in grads)   # pending grads untouched
    assert frozen.grad is None and r.scores["cis_sep"] == pytest.approx(1.0, abs=1e-5)


def test_provenance_records_the_model_dtype(tok, pairs, bundle):
    m = make_model(len(tok)).to(torch.bfloat16)
    r = check(m, bundle, pairs, tokenizer=tok, with_circ=False)
    assert r.provenance["dtype"] == "bfloat16"


def test_nonpositive_reference_is_error_not_pass(model, tok, pairs, model_dir):
    """#111: if the primary metric's reference is <= 0 the verdict is ERROR, whatever the score."""
    b = calibrate_bundle(model, tok, pairs, base_name=str(model_dir), threshold=THRESH)
    assert check(model, b, pairs, tokenizer=tok).status == "PASS"
    b.reference.sep_base = -abs(b.reference.sep_base)
    with pytest.warns(UserWarning, match="<= 0"):
        r = check(model, b, pairs, tokenizer=tok)
    assert r.status == "ERROR" and r.exit_code == 1

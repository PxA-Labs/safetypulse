"""Guards in calibrate/check: refuse/comply token-set overlap, and Qwen3-style thinking templates."""
import functools
import json
import warnings
from pathlib import Path

import pytest
from conftest import CALIBRATED_BUNDLE, DATA, NOTEBOOK_BUNDLE, REPO

from safetypulse import ProbeBundle
from safetypulse.calibrate import chat_thinking_setting, resolve_token_sets, warn_thinking_setting
from safetypulse.prompts import default_token_sets

# --- token-set overlap ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["dict", "file"])
def test_overlapping_token_sets_are_rejected(tmp_path, tok, source):
    ts = {"refuse": [7, 9], "comply": [8, 9, 10]}
    if source == "file":
        (tmp_path / "ts.json").write_text(json.dumps({"org/m": ts}))
        ts = tmp_path / "ts.json"
    with pytest.raises(ValueError, match=r"token IDs \[9\] are in both the refuse and comply sets"):
        resolve_token_sets(ts, tok, "org/m")


def test_disjoint_token_sets_are_unchanged(tmp_path, tok, recwarn):
    ts = {"refuse": [7, 9], "comply": [8, 10]}
    assert resolve_token_sets(ts, tok) == ts
    (tmp_path / "ts.json").write_text(json.dumps({"org/m": ts}))
    assert resolve_token_sets(tmp_path / "ts.json", tok, "org/m") == ts
    assert resolve_token_sets(None, tok) == default_token_sets(tok)   # defaults are disjoint by construction
    assert not recwarn.list


def _token_sets_in(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("refuse"), list) and isinstance(obj.get("comply"), list):
            yield obj
        for v in obj.values():
            yield from _token_sets_in(v)


def _token_set_files():
    # Tests read bundles only from tests/data (test_schemas.test_fixtures_are_tracked_copies). The same check over
    # every tracked bundle and token_sets.json under the executed notebooks found no overlap either (lane 17 report).
    files = sorted({NOTEBOOK_BUNDLE, CALIBRATED_BUNDLE, *DATA.glob("*.json")})
    return [p for p in files if list(_token_sets_in(json.loads(p.read_text())))]


@pytest.mark.parametrize("path", _token_set_files(), ids=lambda p: str(Path(p).relative_to(REPO)))
def test_frozen_bundles_and_token_set_files_have_no_overlap(path):
    """The overlap guard cannot break the tracked fixture bundles (frozen bundles are loaded by
    ProbeBundle.load, which does not call resolve_token_sets, but the files must stay valid inputs anyway)."""
    sets = list(_token_sets_in(json.loads(Path(path).read_text())))
    assert sets
    for ts in sets:
        assert not set(ts["refuse"]) & set(ts["comply"])
    if "probe_bundle" in Path(path).name:
        ProbeBundle.model_validate(json.loads(Path(path).read_text()))


# --- Qwen3 thinking switch -----------------------------------------------------------------------------------------

THINKING_TEMPLATE = ("{% for m in messages %}{{ m['content'] }}{% endfor %}"
                     "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}")


class StubTok:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, **kw):
        return ""


def _patched(tok, value):
    tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=value)
    return tok


def test_thinking_setting_detection():
    assert chat_thinking_setting(StubTok("{{ messages }}")) is None
    assert chat_thinking_setting(StubTok(None)) is None
    assert chat_thinking_setting(StubTok(THINKING_TEMPLATE)) == "template_default"
    assert chat_thinking_setting(_patched(StubTok(THINKING_TEMPLATE), False)) is False   # as the H6 jobs patch it
    assert chat_thinking_setting(_patched(StubTok(THINKING_TEMPLATE), True)) is True


def test_check_warns_when_bundle_records_no_thinking_setting():
    with pytest.warns(UserWarning, match=r"enable_thinking.*does not record.*H6"):
        warn_thinking_setting(StubTok(THINKING_TEMPLATE), {})
    with pytest.warns(UserWarning, match=r"does not record"):   # even when the tokenizer is patched like H6
        warn_thinking_setting(_patched(StubTok(THINKING_TEMPLATE), False), {"seed": 0})


def test_check_warns_on_recorded_thinking_mismatch():
    with pytest.warns(UserWarning, match=r"calibrated with enable_thinking=False.*template_default"):
        warn_thinking_setting(StubTok(THINKING_TEMPLATE), {"enable_thinking": False})


def test_no_thinking_warning_when_template_has_no_switch_or_settings_match(tok):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_thinking_setting(tok, {})                                      # conftest template: no switch
        warn_thinking_setting(StubTok("{{ messages }}"), {})
        warn_thinking_setting(_patched(StubTok(THINKING_TEMPLATE), False), {"enable_thinking": False})
        warn_thinking_setting(StubTok(THINKING_TEMPLATE), {"enable_thinking": "template_default"})


def test_calibrate_side_warns_on_template_default_only():
    with pytest.warns(UserWarning, match=r"template's default.*enable_thinking=False.*H6"):
        warn_thinking_setting(StubTok(THINKING_TEMPLATE))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_thinking_setting(_patched(StubTok(THINKING_TEMPLATE), False))
        warn_thinking_setting(StubTok("{{ messages }}"))


def test_check_emits_thinking_warning(model, tok):
    """check() warns when the tokenizer's template has the switch and the bundle records no setting; the scores
    are the same as with the plain template (a Jinja comment renders to nothing)."""
    from conftest import calibrate_bundle, make_pairs, make_tokenizer

    from safetypulse import check

    pairs = make_pairs(6, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = check(model, calibrate_bundle(model, tok, pairs, n_components=0), pairs, tokenizer=tok,
                     with_circ=False)
    thinking_tok = make_tokenizer()
    thinking_tok.chat_template = tok.chat_template + "{# enable_thinking #}"
    bundle = calibrate_bundle(model, thinking_tok, pairs, n_components=0)   # fingerprint has this template
    assert "enable_thinking" not in bundle.provenance
    with pytest.warns(UserWarning, match="does not record"):
        flagged = check(model, bundle, pairs, tokenizer=thinking_tok, with_circ=False)
    assert flagged.scores == base.scores


def test_calibrate_records_thinking_setting_and_check_accepts_it(model, tok):
    """calibrate() records the thinking setting only for templates with the switch; a bundle calibrated with
    enable_thinking=False (as H6) checks silently with a tokenizer patched the same way."""
    from conftest import make_pairs, make_tokenizer

    from safetypulse import check
    from safetypulse.calibrate import CalibrateConfig, calibrate

    pairs = make_pairs(18, seed=1)
    for i, p in enumerate(pairs):
        p["split"] = ("calib", "val", "probe")[i % 3]
    cfg = CalibrateConfig(n_candidates=6, top_k=3)
    plain = calibrate(model, tok, pairs, config=cfg, base_model="tiny")
    assert "enable_thinking" not in plain.provenance

    thinking_tok = make_tokenizer()
    thinking_tok.chat_template = tok.chat_template + "{# enable_thinking #}"
    with pytest.warns(UserWarning, match="template's default"):
        default = calibrate(model, thinking_tok, pairs, config=cfg, base_model="tiny")
    assert default.provenance["enable_thinking"] == "template_default"

    _patched(thinking_tok, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle = calibrate(model, thinking_tok, pairs, config=cfg, base_model="tiny")
        assert bundle.provenance["enable_thinking"] is False
        probe = [p for p in pairs if p["split"] == "probe"]
        check(model, bundle, probe, tokenizer=thinking_tok, with_circ=False)
    assert bundle.reference == plain.reference   # the switch is a Jinja comment here: numbers unchanged

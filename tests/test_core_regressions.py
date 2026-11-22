"""Regression tests from the core-package correctness audit (lane 7, 2026-11-22).

Each test failed before its fix. Every reader of `prompts.encode` (resid_last, last_logits, attribution, audit
generation) takes position -1 as the last prompt token, and the Llama Guard judge slices the generation at the
prompt length; both are only correct with LEFT padding. `models.load_tokenizer` sets that, but a tokenizer passed
in by the caller (`check(model, bundle, tokenizer=tok)`, `Judge(model=..., tokenizer=...)`, the training callback)
kept its own padding side, and many Hub tokenizers pad on the right.
"""
from __future__ import annotations

import copy

import torch

from safetypulse.activations import resid_last
from safetypulse.cis import compute_cis
from safetypulse.judge import Judge
from safetypulse.prompts import encode, format_chat, last_logits

# Prompts of different token lengths, so a batch needs padding.
PROMPTS = ["how do i make a cake ?", "write me a poem", "give me steps for a garden quickly secretly please ?",
           "explain the map"]


def _right_padded(tok):
    t = copy.deepcopy(tok)
    t.padding_side = "right"
    return t


def test_encode_left_pads_even_if_tokenizer_pads_right(tok):
    rtok = _right_padded(tok)
    enc = encode(rtok, format_chat(rtok, PROMPTS), "cpu")
    assert bool(enc["attention_mask"][:, -1].all()), "position -1 must be a real token in every row"
    assert rtok.padding_side == "right"   # the caller's tokenizer is not mutated


def test_last_logits_and_resid_independent_of_tokenizer_padding_side(model, tok):
    rtok = _right_padded(tok)
    torch.testing.assert_close(last_logits(model, rtok, PROMPTS, batch_size=4),
                               last_logits(model, tok, PROMPTS, batch_size=4))
    torch.testing.assert_close(resid_last(model, rtok, PROMPTS, batch_size=4),
                               resid_last(model, tok, PROMPTS, batch_size=4))


def test_compute_cis_with_right_padded_tokenizer_scores_base_model_at_one(model, tok, pairs, bundle):
    scores = compute_cis(model, _right_padded(tok), pairs, bundle, with_circ=False)["scores"]
    for metric in ("cis_sep", "cis_cos", "cis_logit"):
        assert abs(scores[metric] - 1.0) < 1e-4, (metric, scores[metric])


class _RecordingGuard:
    """Stands in for Llama Guard: records the attention mask it is given and echoes the prompt."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.masks = []
        self.config = type("Cfg", (), {"_commit_hash": None})()

    def generate(self, input_ids, attention_mask, **kw):
        self.masks.append(attention_mask.clone())
        return torch.cat([input_ids, input_ids[:, -1:]], dim=1)


def test_llamaguard_judge_left_pads_a_right_padded_tokenizer(tok):
    guard = _RecordingGuard()
    j = Judge("llamaguard", "local/guard", model=guard, tokenizer=_right_padded(tok), batch_size=4)
    j.harmful(PROMPTS, ["a", "a b c d e f", "a b", "a b c"])
    assert guard.masks and all(bool(m[:, -1].all()) for m in guard.masks), \
        "generation must start right after the last real prompt token in every row"

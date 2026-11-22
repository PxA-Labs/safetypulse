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


def test_git_sha_is_the_package_repo_not_the_callers_working_directory(tmp_path, monkeypatch):
    """provenance()["git_sha"] identifies the safetypulse code that ran. It used to run `git rev-parse` in the
    current directory, so scoring from inside another repository (a training project, a job workspace) recorded
    that repository's commit as the safetypulse version."""
    import subprocess

    from safetypulse.utils import git_sha

    other = tmp_path / "someone_elses_repo"
    other.mkdir()
    git = ["git", "-C", str(other), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    other_sha = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    monkeypatch.chdir(other)
    assert git_sha() != other_sha


def test_git_sha_reports_this_checkout(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from safetypulse.utils import git_sha

    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        import pytest

        pytest.skip("tests are not running from a git checkout")
    monkeypatch.chdir(tmp_path)   # outside any repository
    assert git_sha() == head.stdout.strip()


def _git_repo(path):
    import subprocess

    path.mkdir(parents=True)
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    pkg = path / "safetypulse"
    pkg.mkdir()
    (pkg / "utils.py").write_text("# stand-in for the package module\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    sha = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    return pkg, sha.stdout.strip()


def test_git_sha_handles_a_checkout_path_with_spaces(tmp_path, monkeypatch):
    """`git rev-parse --show-toplevel --short HEAD` prints the top level and the SHA on separate lines. They were
    split on any whitespace, so a checkout under a path with a space ("my repo", "Google Drive") gave three
    fields, the unpack raised, and git_sha() silently recorded None."""
    import safetypulse.utils as sp_utils

    pkg, sha = _git_repo(tmp_path / "my repo")
    monkeypatch.setattr(sp_utils, "__file__", str(pkg / "utils.py"))
    monkeypatch.chdir(tmp_path)
    assert sp_utils.git_sha() == sha


def test_git_sha_is_none_when_git_fails(tmp_path, monkeypatch):
    """A non-zero exit from git is never trusted, whatever it printed."""
    import subprocess

    import safetypulse.utils as sp_utils

    pkg, sha = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(sp_utils, "__file__", str(pkg / "utils.py"))
    failed = subprocess.CompletedProcess(args=[], returncode=128, stdout=f"{pkg.parent.resolve()}\n{sha}\n",
                                         stderr="fatal: simulated failure")
    monkeypatch.setattr(sp_utils.subprocess, "run", lambda *a, **k: failed)
    assert sp_utils.git_sha() is None


def test_last_logits_builds_no_autograd_graph(model, tok):
    """Guard, not a regression (the @torch.no_grad() decorator was already there): last_logits is a read-out, and
    called outside check()'s scoring_mode on a model with trainable parameters (a training loop, audit.mc_accuracy
    on a fresh PEFT model) it must not build an autograd graph per batch."""
    trainable = copy.deepcopy(model).requires_grad_(True)
    out = last_logits(trainable, tok, PROMPTS, batch_size=2)
    assert not out.requires_grad and out.grad_fn is None
    torch.testing.assert_close(out, last_logits(model, tok, PROMPTS, batch_size=2))

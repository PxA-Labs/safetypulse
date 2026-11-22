"""tools/check_release_policy.py blocks the artifacts docs/11 withholds and lets the published ones through."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_release_policy", REPO / "tools" / "check_release_policy.py")
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


@pytest.mark.parametrize("path", [
    "runs/harm-s0/checkpoint-125/adapter_model.safetensors",
    "models/pytorch_model.bin",
    "data/ft_harm.jsonl",
    "report/judge_validation_sample.csv",
    "runs/10_audit/files/audit.jsonl",
    "out/raw_completions.jsonl",
    "runs/10_audit/files/final__audit.jsonl",
    "runs/07_finetune/files/earth__ft_harm.jsonl",
])
def test_withheld_artifacts_are_blocked(path, tmp_path):
    assert [p for p, _ in policy.violations([path], root=tmp_path)] == [path]


@pytest.mark.parametrize("path", [
    "runs/07_finetune_checkpoints/07_finetune_checkpoints__2026-11-07__harm-s0.log",
    "runs/08_monitor/files/harmdense__monitor.jsonl",
    "safetypulse/audit.py",
    "docs/11_WITHHELD_ARTIFACTS.md",
])
def test_published_files_pass(path, tmp_path):
    assert policy.violations([path], root=tmp_path) == []


def test_large_files_are_blocked(tmp_path):
    (tmp_path / "big.csv").write_bytes(b"0" * (policy.MAX_BYTES + 1))
    (tmp_path / "small.csv").write_bytes(b"0" * 10)
    assert policy.violations(["big.csv", "small.csv"], root=tmp_path) == [("big.csv", "larger than 5 MB")]


@pytest.mark.parametrize("path", [
    "runs/harmdense-s0/checkpoint-4/adapter_config.json",
    "runs/earth-harm-s1/checkpoint-10/README.md",
    "data/ft-harm_translated.jsonl",
])
def test_harm_checkpoints_and_ft_harm_variants_are_blocked(path, tmp_path):
    assert [p for p, _ in policy.violations([path], root=tmp_path)] == [path]


def test_harm_run_dir_without_checkpoint_passes(tmp_path):
    assert policy.violations(["runs/harm-s0/train_log.json"], root=tmp_path) == []


def _write(root, name, text):
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")
    return name


def test_answer_text_in_jsonl_is_blocked_by_field_and_line(tmp_path):
    rows = [{"prompt_id": 1, "verdict": "safe", "output_count": 3},
            {"prompt_id": 2, "meta": {"model_response": "SYNTHETIC-TEXT"}},
            {"prompt_id": 3, "completion": "SYNTHETIC-TEXT"}]
    name = _write(tmp_path, "runs/x/scores.jsonl", "\n".join(json.dumps(r) for r in rows))
    bad = policy.content_violations([name], root=tmp_path)
    assert bad == [(name, "answer text in field 'meta.model_response' (line 2)"),
                   (name, "answer text in field 'completion' (line 3)")]
    assert all("SYNTHETIC" not in reason for _, reason in bad)   # values are never reported


def test_answer_text_in_json_and_csv_is_blocked(tmp_path):
    j = _write(tmp_path, "a.json", json.dumps({"items": [{"answers": ["SYNTHETIC-TEXT"]}]}))
    c = _write(tmp_path, "b.csv", "id,verdict,generated_text\n1,unsafe,SYNTHETIC-TEXT\n")
    t = _write(tmp_path, "c.tsv", "id\tResponse\n1\t\n2\tSYNTHETIC-TEXT\n")
    assert policy.content_violations([j, c, t], root=tmp_path) == [
        (j, "answer text in field 'items[].answers' (line 1)"),
        (c, "answer text in field 'generated_text' (line 2)"),
        (t, "answer text in field 'Response' (line 3)"),
    ]


def test_redacted_empty_and_numeric_answer_fields_pass(tmp_path):
    rows = [{"response": policy.PLACEHOLDER}, {"response": ""}, {"output": 0.5}, {"generation_seconds": 1.2},
            {"responses": None}]
    j = _write(tmp_path, "ok.jsonl", "\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    c = _write(tmp_path, "ok.csv", f"id,answer\n1,{policy.PLACEHOLDER}\n2,\n")
    other = _write(tmp_path, "notes.md", '{"answer": "SYNTHETIC-TEXT"}')
    assert policy.content_violations([j, c, other, "missing.json"], root=tmp_path) == []


def test_allowed_paths_are_skipped(tmp_path, monkeypatch):
    name = _write(tmp_path, "reviewed.jsonl", json.dumps({"answer": "SYNTHETIC-TEXT"}))
    monkeypatch.setattr(policy, "ALLOWED", frozenset({name, "data/ft_harm.jsonl"}))
    assert policy.check([name, "data/ft_harm.jsonl"], root=tmp_path) == []
    assert policy.name_violations(["data/ft_harm.jsonl"]) == []


def test_name_violations_ignore_size_and_disk():
    assert policy.name_violations(["gone/ft_harm.jsonl", "gone/table.csv"]) == [
        ("gone/ft_harm.jsonl", "withheld harmful data")]


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=cwd, check=True, capture_output=True)


def test_main_scans_tree_and_branch_history(tmp_path, monkeypatch, capsys):
    """A withheld file added and deleted on the branch fails; so does answer text already on the base."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, "scores.jsonl", json.dumps({"id": 1, "verdict": "safe"}))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    monkeypatch.chdir(tmp_path)
    assert policy.main(["x", "main"]) == 0

    _git(tmp_path, "checkout", "-q", "-b", "feature")
    _write(tmp_path, "data/ft_harm.jsonl", "{}")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "add")
    _git(tmp_path, "rm", "-q", "data/ft_harm.jsonl")
    _git(tmp_path, "commit", "-q", "-m", "remove")
    capsys.readouterr()
    assert policy.main(["x", "main"]) == 1
    assert "file=data/ft_harm.jsonl::" in capsys.readouterr().out

    _git(tmp_path, "checkout", "-q", "main")
    _write(tmp_path, "scores.jsonl", json.dumps({"id": 1, "answer": "SYNTHETIC-TEXT"}))
    _git(tmp_path, "commit", "-q", "-am", "answer on base")
    rc = policy.main(["x", "main"])
    out = capsys.readouterr().out
    assert rc == 1 and "file=scores.jsonl::release policy: answer text in field 'answer'" in out
    assert "SYNTHETIC" not in out
    assert policy.main(["x", "main", "--all-history"]) == 1


def test_repository_passes_release_policy():
    """The tracked tree of this repository holds no withheld path or answer text (runs in CI on every push)."""
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    if files.returncode != 0:
        pytest.skip("not a git checkout")
    assert policy.check(files.stdout.splitlines(), root=REPO) == []


# --- Round-2 hardening: strict rules for HF job outputs (Job 15 on) and executed notebooks. -----------------------
# Every string below is a synthetic placeholder; no model text is used.

RUNS = "/".join(("kaggle", "executed"))   # synthetic paths only; split so test_schemas' no-fixture check passes
JOB_DIR = f"{RUNS}/12_hf_jobs/files"
PROSE = " ".join(["placeholder"] * 6 + ["synthetic", "sentence", "standing", "in", "for", "model", "text", "here."])
HEX40 = "0123456789abcdef" * 2 + "01234567"


def test_strict_job_file_scope():
    assert policy.is_strict_job_file(f"{JOB_DIR}/job15_summary.json")
    assert policy.is_strict_job_file(f"{JOB_DIR}/job20_multilingual.jsonl")
    assert policy.is_strict_job_file(f"{JOB_DIR}/job137_future.csv")          # future jobs are covered too
    assert policy.is_strict_job_file(f"{JOB_DIR}/job16/scores.jsonl")
    assert not policy.is_strict_job_file(f"{JOB_DIR}/job9_jobs.json")
    assert not policy.is_strict_job_file(f"{JOB_DIR}/job1_crosscheck.json")
    assert not policy.is_strict_job_file("docs/report/job15_summary.json")


@pytest.mark.parametrize("key", ["translation", "output_text", "prompt_text", "hi_translation", "translations"])
def test_job_output_text_fields_are_blocked(key, tmp_path):
    """Short text under a text-named key fails in a job output (the old checker let these names through)."""
    name = _write(tmp_path, f"{JOB_DIR}/job21_scores.jsonl",
                  json.dumps({"model": "Lexsi/x", "flag": 0}) + "\n" + json.dumps({"model": "Lexsi/y", key: "SYNTH"}))
    bad = policy.check([name], root=tmp_path)
    assert bad == [(name, f"job output: text in field '{key}' (line 2)")]
    assert all("SYNTH" not in reason for _, reason in bad)


def test_job_output_long_free_text_is_blocked(tmp_path):
    """Any long free-text string in a job output fails, whatever its key, in JSON, JSONL and CSV."""
    j = _write(tmp_path, f"{JOB_DIR}/job17_summary.json",
               json.dumps({"family": "gemma3-4b", "per_prompt": [{"id": 3, "judge_note": PROSE}], "notes": [PROSE]}))
    k = _write(tmp_path, f"{JOB_DIR}/job18_keys.json", json.dumps({PROSE: 1}))
    c = _write(tmp_path, f"{JOB_DIR}/job19_rows.csv", f"id,lang,detail\n1,hi,0.5\n2,es,\"{PROSE}\"\n")
    assert policy.check([j, k, c], root=tmp_path) == [
        (j, "job output: long text in field 'notes[]' (line 1)"),
        (j, "job output: long text in field 'per_prompt[].judge_note' (line 1)"),
        (k, "job output: long text in a key under '<root>' (line 1)"),
        (c, "job output: long text in field 'detail' (line 3)"),
    ]


def test_job_output_identifiers_numbers_and_placeholder_pass(tmp_path):
    row = {
        "model": "pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca-with-a-rather-long-repository-name-for-testing",
        "revision": HEX40 * 2,
        "sha256": "f" * 64,
        "path": "results/job21/gemma3-4b/checkpoints/checkpoint-0000000125/adapter_config_for_this_run.json",
        "url": "https://huggingface.co/datasets/purvansh01/safetypulse-jobs/blob/main/results/job21/summary.json",
        "response": policy.PLACEHOLDER, "translation": "", "generation": 3, "prompt_text": None,
        "flags": [0, 1, 1], "cis_sep": 0.71, "status": "WARN",
    }
    j = _write(tmp_path, f"{JOB_DIR}/job21_ok.jsonl", json.dumps(row))
    old = _write(tmp_path, f"{JOB_DIR}/job9_jobs.json", json.dumps({"note": PROSE}))   # before the strict rule
    assert policy.check([j, old], root=tmp_path) == []


def test_unparseable_job_output_is_blocked(tmp_path):
    name = _write(tmp_path, f"{JOB_DIR}/job16_broken.json", "{not json")
    assert policy.check([name], root=tmp_path) == [(name, "job output: unparseable JSON (line 1)")]


def _notebook(outputs_per_cell):
    return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {},
                       "cells": [{"cell_type": "code", "source": "", "metadata": {}, "outputs": outs}
                                 for outs in outputs_per_cell]})


def test_notebook_free_text_outputs_are_blocked(tmp_path):
    """Prose in stream or display outputs fails (the old checker never opened notebooks); text is not reported."""
    devanagari = " ".join(["नमस्ते"] * 16) + "।"     # placeholder word x16
    nb = _notebook([
        [{"output_type": "stream", "name": "stdout", "text": ["step 1 ok\n", PROSE + "\n"]}],
        [{"output_type": "execute_result", "execution_count": 2, "metadata": {},
          "data": {"text/plain": [devanagari]}}],
        [{"output_type": "display_data", "metadata": {}, "data": {"text/markdown": PROSE}}],
    ])
    name = _write(tmp_path, f"{RUNS}/21_x/21_x__2026-11-22.ipynb", nb)
    bad = policy.check([name], root=tmp_path)
    assert bad == [
        (name, "free text in notebook output (cell 1, stream stdout, line 2)"),
        (name, "free text in notebook output (cell 2, execute_result text/plain, line 1)"),
        (name, "free text in notebook output (cell 3, display_data text/markdown, line 1)"),
    ]
    assert all("placeholder" not in reason for _, reason in bad)


def test_notebook_logs_tables_and_numbers_pass(tmp_path):
    lines = [
        "Warning: You are sending unauthenticated requests to the Hub; " + PROSE,
        "[transformers] Ignoring a generation flag because " + PROSE,
        "No HF_TOKEN (expected for CLI runs): " + PROSE,
        "/usr/lib/python3/site-packages/x.py:12: UserWarning: " + PROSE,
        "gsm8k-s0 25 {'cis_sep': 0.952, 'cis_cos': 0.98, 'cis_logit': 0.91, 'asr': 0.04, 'status': 'OK', 'n': 100}",
        "model                                   cis_sep  status   asr     spot   refusal   flags   seconds  gpu",
        "short line",
    ]
    nb = _notebook([[{"output_type": "stream", "name": "stderr", "text": "\n".join(lines)}],
                    [{"output_type": "error", "ename": "E", "evalue": PROSE, "traceback": [PROSE]}]])
    name = _write(tmp_path, "nb/ok.ipynb", nb)
    broken = _write(tmp_path, "nb/broken.ipynb", "{not json")
    assert policy.check([name, broken], root=tmp_path) == []

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

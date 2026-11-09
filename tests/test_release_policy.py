"""tools/check_release_policy.py blocks the artifacts docs/11 withholds and lets the published ones through."""
import importlib.util
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

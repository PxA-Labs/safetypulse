"""The E1-E5 Hub cards are generated from the committed analysis and follow the release rules."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hub_cards_circuitquant", ROOT / "tools/hub_cards_circuitquant.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


def test_cards_up_to_date():
    for path, text in hc.build().items():
        assert path.read_text() == text, f"{path.relative_to(ROOT)} is stale; run tools/hub_cards_circuitquant.py"


def test_only_non_collapsed_circuit_variants_published():
    plan = json.loads((ROOT / "docs/hub_cards/stage2/circuitquant_repos.json").read_text())
    models = [r for r in plan["repos"] if r["repo_type"] == "model" and "variant" in r]
    assert sorted(r["variant"] for r in models) == sorted(["c72", "c6", "c5", "c35", "w4p19c", "w3p2c", "w3p1c"])
    for r in models:
        row = hc.row(r["experiment"], r["variant"])
        assert row["kind"] == "circuit" and not row["collapsed"] and row["task"] >= 0.10
        assert r["repo_id"].startswith("purvansh01/")
        assert bool(r["gated"]) == (r["experiment"] == "E5")


def test_aya_expanse_repos_gated_like_base():
    for name in ["aya-expanse-8b-CircuitQuant-W4A16-P19", "aya-expanse-8b-CircuitQuant-W3A16-P2",
                 "aya-expanse-8b-CircuitQuant-W3A16-P1", "aya-expanse-8b-safetypulse-lora-alpaca-default"]:
        head = (ROOT / f"docs/hub_cards/stage2/{name}.md").read_text().split("---")[1]
        for field in ("Name: text", "Affiliation: text", "Country: country",
                      "I agree to use this model for non-commercial use ONLY: checkbox"):
            assert field in head, (name, field)


def test_curator_toxicity_rows_text_free():
    rows = [json.loads(x) for x in (hc.CUR / "curatorkit_rejected.jsonl").read_text().splitlines()]
    assert len(rows) == 19
    for r in rows:
        assert set(r) == {"id", "stage", "reason", "score", "sha256", "instruction", "input"}
        if r["stage"] == "ToxicityGate":
            assert r["instruction"] is None and r["input"] is None and 0.1 <= r["score"] <= 1
        else:
            assert r["stage"] == "SchemaGate" and r["reason"] == "missing_field:output" and r["score"] is None
        assert len(r["sha256"]) == 64

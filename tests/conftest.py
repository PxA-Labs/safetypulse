"""Fixtures: a tiny random Qwen2 + word-level tokenizer built locally (no network, no downloads), and probe
bundles calibrated on it. Prompts are synthetic word strings; no real harmful text is needed for the numerics."""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from safetypulse.activations import attribution_scores, mean_diff, parse_component, resid_last  # noqa: E402
from safetypulse.cis import base_reference  # noqa: E402
from safetypulse.models import load_tokenizer, model_fingerprint  # noqa: E402
from safetypulse.prompts import COMPLY_STARTS, REFUSE_STARTS, default_token_sets  # noqa: E402
from safetypulse.schemas import ProbeBundle  # noqa: E402
from safetypulse.utils import sha256_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KAGGLE_LIB = REPO / "kaggle" / "lib"
# Copies of the notebook 06 / 08 bundle JSONs: the executed-notebook outputs are not tracked in git, so tests
# read only these files (test_schemas.test_fixtures_are_tracked_copies).
DATA = Path(__file__).resolve().parent / "data"
NOTEBOOK_BUNDLE = DATA / "nb06_probe_bundle.json"
CALIBRATED_BUNDLE = DATA / "nb08_probe_bundle_calibrated.json"

SPECIALS = ["[UNK]", "[PAD]", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
WORDS = sorted(set(REFUSE_STARTS + COMPLY_STARTS + [
    "how", "do", "i", "make", "a", "the", "build", "write", "explain", "give", "me", "steps", "for", "to",
    "cake", "poem", "garden", "bridge", "recipe", "story", "lamp", "kite", "song", "map",
    "weapon", "malware", "poison", "scam", "exploit", "theft", "forgery", "virus",
    "quickly", "secretly", "safely", "please", "now", "detailed", "simple", "?", ".", "m", "'",
]))
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
LAYER = 2   # resid index read by cis_sep / cis_cos in the synthetic bundles


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(SPECIALS + WORDS)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="<|end|>",
                                  additional_special_tokens=SPECIALS[3:])
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"
    return tok


def make_model(vocab_size: int, seed: int = 0, arch: str = "llama"):
    """4-layer random decoder (4 heads, GQA 2 kv heads), deterministic weights, fp32, eval mode.

    arch="qwen2" mirrors sp_utils._tiny_model (q/k/v biases). The default is llama because transformers 5
    re-builds a saved tokenizer as Qwen2Tokenizer next to a qwen2 config, which breaks our word-level vocab.
    """
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    config_cls, model_cls = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen2": (Qwen2Config, Qwen2ForCausalLM)}[arch]
    torch.manual_seed(seed)
    cfg = config_cls(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                     tie_word_embeddings=True, pad_token_id=1, eos_token_id=2, bos_token_id=None)
    return model_cls(cfg).eval().requires_grad_(False)


def make_pairs(n: int = 6, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    harm = ["weapon", "malware", "poison", "scam", "exploit", "theft", "forgery", "virus"]
    safe = ["cake", "poem", "garden", "bridge", "recipe", "story", "lamp", "kite", "song", "map"]
    pairs = []
    for i in range(n):
        lead = rng.choice(["how do i make a", "write me a", "give me steps for a", "explain the"])
        tail = " ".join(rng.sample(["quickly", "secretly", "safely", "please", "now", "detailed", "simple"], k=1 + i % 3))
        pairs.append({"id": f"p{i}", "harmful": f"{lead} {rng.choice(harm)} {tail} ?",
                      "harmless": f"{lead} {rng.choice(safe)} {tail} ?"})
    return pairs


def calibrate_bundle(model, tok, pairs: list[dict], base_name: str = "tiny-local", n_components: int = 3,
                     threshold: dict | None = None, batch_size: int = 8) -> ProbeBundle:
    """A probe bundle computed on `model` the way notebooks 05/06 do (directions from the same pairs), with the
    probe-set hash and base-model fingerprint that `calibrate` records."""
    token_sets = default_token_sets(tok)
    harm, harmless = [p["harmful"] for p in pairs], [p["harmless"] for p in pairs]
    directions = mean_diff(resid_last(model, tok, harm, batch_size), resid_last(model, tok, harmless, batch_size))
    attr = attribution_scores(model, tok, harm, harmless, token_sets)
    top = sorted(attr, key=lambda c: -attr[c])[:n_components]
    ref = base_reference(model, tok, pairs, directions, LAYER, token_sets, top, batch_size)

    comps = [{"id": c, "type": parse_component(c)[0], "layer": parse_component(c)[1], "index": parse_component(c)[2],
              "score": ref["component_scores"][c]} for c in top]
    bundle = ProbeBundle.model_validate({
        "base_model": {"name": base_name, "revision": None, "fingerprint": model_fingerprint(model, tok)},
        "layer": LAYER,
        "components": comps,
        "token_sets": token_sets,
        "reference": ref["reference"],
        "threshold": threshold or {"metric": None, "tau_warn": None, "tau_fail": None},
        "probe_set": {"path": "probe_pairs.jsonl", "sha256": sha256_rows(pairs), "n_pairs": len(pairs)},
    })
    return bundle.with_tensors(directions, ref["probe_diff"])


@pytest.fixture(scope="session")
def tok():
    return make_tokenizer()


@pytest.fixture(scope="session")
def model(tok):
    return make_model(len(tok))


@pytest.fixture(scope="session")
def pairs():
    return make_pairs()


@pytest.fixture(scope="session")
def bundle(model, tok, pairs):
    return calibrate_bundle(model, tok, pairs)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, model, tok) -> Path:
    """The tiny model + tokenizer saved like a HF checkpoint, so loading goes through from_pretrained."""
    d = tmp_path_factory.mktemp("tiny_model")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    assert type(load_tokenizer(d)).__name__ == "TokenizersBackend"   # reloads as our word-level tokenizer
    return d


@pytest.fixture(scope="session")
def sp_utils():
    sys.path.insert(0, str(KAGGLE_LIB))
    try:
        import sp_utils
    finally:
        sys.path.remove(str(KAGGLE_LIB))
    return sp_utils

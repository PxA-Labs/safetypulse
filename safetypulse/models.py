"""Model and tokenizer loading, LoRA merging, weight snapshots and layer geometry.

`name` is anything `from_pretrained` accepts: a local directory or a Hugging Face Hub id.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(purpose: str = "infer") -> torch.dtype:
    """fp32 for activations/gradients; for generation fp16 on pre-Ampere GPUs (no bf16), bf16 on Ampere+."""
    if not torch.cuda.is_available() or purpose in ("activations", "grad"):
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def is_adapter_dir(path: str | Path) -> bool:
    """True for a PEFT (LoRA) checkpoint directory, which needs a base model to be merged into."""
    return (Path(path) / "adapter_config.json").is_file()


def load_tokenizer(name: str | Path, revision: str | None = None):
    """Left-padded tokenizer (we always read the last prompt token) with a pad token."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(name), revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def patch_peft() -> None:
    """Some images (e.g. Kaggle) ship torchao 0.10; peft >= 0.21 raises ImportError when it finds a torchao older
    than 0.16, even for plain LoRA. We never use torchao, so tell peft it is absent. Call before loading adapters."""
    import peft.import_utils as iu

    iu.is_torchao_available = lambda: False
    with contextlib.suppress(Exception):
        import peft.tuners.lora.torchao as lt

        lt.is_torchao_available = lambda: False


def apply_adapter(model, adapter_dir: str | Path):
    """Merge a LoRA checkpoint into `model` (in place; undo with restore_weights). Returns the merged model."""
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("loading a LoRA checkpoint needs peft: pip install 'safetypulse[peft]'") from e

    patch_peft()
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.eval().requires_grad_(False)
    return model


def load_model(
    name: str | Path,
    dtype: torch.dtype | None = None,
    adapter: str | Path | None = None,
    revision: str | None = None,
    dev: torch.device | str | None = None,
):
    """(model, tokenizer) in eval mode with frozen weights. `adapter` = a LoRA checkpoint dir, merged in.

    If `name` is itself a LoRA checkpoint dir, its base model is read from adapter_config.json.
    """
    from transformers import AutoModelForCausalLM

    if adapter is None and Path(name).is_dir() and is_adapter_dir(name):
        adapter, name = name, adapter_base_model(name)
    dtype = dtype or pick_dtype("infer")
    tok = load_tokenizer(name, revision)
    model = AutoModelForCausalLM.from_pretrained(str(name), dtype=dtype, revision=revision)
    model.to(dev or device())
    if adapter is not None:
        model = apply_adapter(model, adapter)
    model.eval().requires_grad_(False)
    return model, tok


def adapter_base_model(adapter_dir: str | Path) -> str:
    """`base_model_name_or_path` recorded by PEFT in the adapter config."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"{adapter_dir}/adapter_config.json names no base model; pass it explicitly")
    return base


FINGERPRINT_CONFIG_KEYS = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")


def model_fingerprint(model, tok) -> dict:
    """Cheap identity of a base model for `check`'s base-model match: architecture config values, the tokenizer's
    vocabulary size and the sha256 of its chat template (None if it has none). No weights are hashed."""
    import hashlib

    cfg = getattr(model, "config", None)
    fp = {k: getattr(cfg, k, None) for k in FINGERPRINT_CONFIG_KEYS}
    template = getattr(tok, "chat_template", None)
    if isinstance(template, dict):   # named templates
        template = json.dumps(template, sort_keys=True)
    fp["tokenizer_vocab_size"] = len(tok)
    fp["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None
    return fp


def fingerprint_mismatches(expected: dict, actual: dict) -> dict:
    """{key: (expected, actual)} for every key of `expected` whose value differs in `actual`."""
    return {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}


def free(*objs) -> None:
    import gc

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# Weight snapshots (check many checkpoints against one loaded base model)
# ---------------------------------------------------------------------------------------------


def state_dict_cpu(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()}


@torch.no_grad()
def restore_weights(model, sd: dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        p.copy_(sd[name].to(p.device, p.dtype))


# ---------------------------------------------------------------------------------------------
# Layer geometry
# ---------------------------------------------------------------------------------------------


def decoder_layers(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"cannot find decoder layers on {type(model).__name__}")


def layers_prefix(model) -> str:
    """Parameter-name prefix of the decoder layers, e.g. "model.layers"."""
    layers = decoder_layers(model)
    return next(n for n, mod in model.named_modules() if mod is layers)


def head_geometry(model) -> tuple[int, int]:
    """(n_heads, head_dim) of the attention layers."""
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return n_heads, head_dim


def n_resid(model) -> int:
    """Residual-stream positions: embeddings plus one per decoder layer."""
    return len(decoder_layers(model)) + 1

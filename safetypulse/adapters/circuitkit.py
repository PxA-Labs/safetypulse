"""CircuitKIT (Lexsi Labs, licence LSAL-1.2) cross-check of our circuit discovery (docs/03 §5).

CircuitKIT runs EAP-IG on a TransformerLens HookedTransformer. We hand it the same contrast as calibrate
(clean = harmful prompt, corrupt = harmless twin), wrapped in the model's chat template by CircuitKIT itself
(chat_template_mode="on", so no second BOS), with our refusal logit difference over the token sets as the metric.
Its node names "A{layer}.{head}" / "MLP {layer}" map to our "attn_head:L:H" / "mlp:L".

Notes from the CircuitKIT 0.2.0 source:
  * discover() returns a Circuit whose `.nodes` (node level) are the nodes to *prune* (lowest scores); the
    importance ranking is `.scores`, read from the `<output>_scores.json` side-car, so we rank by that.
  * Node scores are absolute values, so they rank by magnitude; compare with |our attribution|.
  * GenericTaskSpec turns each answer string into its first token and drops pairs whose correct and incorrect
    first tokens coincide; our metric ignores those labels, but the answers must still differ.

`circuitkit` (and transformer-lens) is imported only inside the functions that run it.
"""
from __future__ import annotations

import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..activations import component_id, parse_component
from ..calibrate import spearman
from ..utils import save_jsonl

_ATTN = re.compile(r"^A(\d+)\.(\d+)$")
_MLP = re.compile(r"^MLP (\d+)$")
TASK_SCHEMA = {"prompt": "prompt", "answer": "answer", "corrupted_prompt": "corrupted_prompt",
               "corrupted_answer": "corrupted_answer"}


def require_circuitkit():
    """The circuitkit module, or an ImportError that says how to get it."""
    try:
        import circuitkit
    except ImportError as e:
        raise ImportError("the CircuitKIT cross-check needs circuitkit (Lexsi Labs, LSAL licence) and transformer-lens: "
                          "pip install circuitkit. calibrate itself does not need it.") from e
    return circuitkit


# ---------------------------------------------------------------------------------------------
# Node ids
# ---------------------------------------------------------------------------------------------


def to_component_id(node: str) -> str | None:
    """"A3.5" -> "attn_head:3:5", "MLP 2" -> "mlp:2"; None for other nodes (input, logits, neurons)."""
    if m := _ATTN.match(node):
        return component_id("attn_head", int(m.group(1)), int(m.group(2)))
    if m := _MLP.match(node):
        return component_id("mlp", int(m.group(1)))
    return None


def to_node_name(cid: str) -> str:
    """"attn_head:3:5" -> "A3.5", "mlp:2" -> "MLP 2"."""
    kind, layer, head = parse_component(cid)
    return f"A{layer}.{head}" if kind == "attn_head" else f"MLP {layer}"


def map_scores(node_scores: Mapping[str, float]) -> dict[str, float]:
    """CircuitKIT {node name: score} -> {our component id: score}, dropping nodes we do not model."""
    out = {}
    for node, score in node_scores.items():
        cid = to_component_id(node)
        if cid is not None:
            out[cid] = float(score)
    return out


# ---------------------------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------------------------


def _answer_strings(tok, token_sets: dict) -> tuple[str, str]:
    """A refusal-start and a compliance-start string whose first tokens differ when re-tokenised."""
    def first(text):
        ids = tok.encode(text, add_special_tokens=False)
        return ids[0] if ids else None

    def pick(ids, avoid=None):
        for i in ids:
            s = tok.decode([i])
            if s.strip() and first(s) is not None and first(s) != avoid:
                return s
        raise ValueError("no token in the set decodes to a usable single-token answer")

    refuse = pick(token_sets["refuse"])
    return refuse, pick(token_sets["comply"], avoid=first(refuse))


def task_rows(pairs: Sequence[dict], token_sets: dict, tok) -> list[dict]:
    """GenericTaskSpec JSONL rows: prompt = harmful (clean), corrupted_prompt = harmless twin (corrupt)."""
    refuse, comply = _answer_strings(tok, token_sets)
    return [{"prompt": p["harmful"], "answer": refuse, "corrupted_prompt": p["harmless"], "corrupted_answer": comply}
            for p in pairs]


def refusal_metric(token_sets: dict):
    """CircuitKIT metric(logits, clean_logits, input_length, labels, loss=True, mean=True): our refusal logit
    difference (logsumexp log-prob of refusal starts minus compliance starts) at the last real token."""
    import torch

    refuse, comply = list(token_sets["refuse"]), list(token_sets["comply"])

    def metric(logits, clean_logits, input_length, labels, loss=True, mean=True):
        idx = torch.arange(logits.size(0), device=logits.device)
        last = torch.as_tensor(input_length, device=logits.device) - 1
        lp = logits[idx, last].float().log_softmax(-1)
        v = lp[:, refuse].logsumexp(-1) - lp[:, comply].logsumexp(-1)
        v = -v if loss else v
        return v.mean() if mean else v

    return metric


def build_task(pairs: Sequence[dict], token_sets: dict, tok, work_dir: str | Path, name: str | None = None,
               chat_template_mode: str = "on"):
    """Write the task JSONL to `work_dir` and return a registered-ready GenericTaskSpec. `chat_template_mode="off"`
    feeds raw prompts (much shorter sequences, so discovery fits smaller GPUs, at the cost of matching our setup)."""
    require_circuitkit()
    from circuitkit.tasks.generic import GenericTaskSpec

    path = save_jsonl(task_rows(pairs, token_sets, tok), Path(work_dir) / "refusal_task.jsonl")
    return GenericTaskSpec.from_jsonl(str(path), schema=dict(TASK_SCHEMA), metric_fn=refusal_metric(token_sets),
                                      name=name or f"safetypulse_refusal_{uuid.uuid4().hex[:8]}",
                                      task_type="classification", chat_template_mode=chat_template_mode)


# ---------------------------------------------------------------------------------------------
# Discovery and comparison
# ---------------------------------------------------------------------------------------------


def load_tl_model(name: str, dtype: str = "float32", device: str | None = None):
    """HookedTransformer with the hook flags EAP needs (circuitkit.load_model)."""
    return require_circuitkit().load_model(name, dtype=dtype, device=device)


def discover(tl_model, pairs: Sequence[dict], token_sets: dict, *, algorithm: str = "eap-ig", batch_size: int = 4,
             seed: int = 0, work_dir: str | Path | None = None, chat_template_mode: str = "on",
             **kw) -> dict[str, float]:
    """Run CircuitKIT node-level discovery on our pairs; returns {our component id: |score|} for every head/MLP."""
    ck = require_circuitkit()
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(work_dir or tmp)
        wd.mkdir(parents=True, exist_ok=True)
        task = build_task(pairs, token_sets, tl_model.tokenizer, wd, chat_template_mode=chat_template_mode)
        ck.register_task(task)
        circuit = ck.discover(tl_model, task.name, algorithm=algorithm, level="node", n_examples=len(pairs),
                              batch_size=batch_size, output_path=str(wd / "circuit.pt"), num_examples=len(pairs),
                              seed=seed, chat_template_mode=chat_template_mode, **kw)
    if not circuit.scores:
        raise RuntimeError("CircuitKIT returned no node scores (score side-car missing)")
    return map_scores(circuit.scores)


def compare(ours: Sequence[str], theirs: Mapping[str, float], k: int | None = None,
            our_scores: Mapping[str, float] | None = None) -> dict:
    """Agreement between our components (best first) and CircuitKIT scores ({id: |score|}).

    Returns overlap_at_k (|ours[:k] ∩ theirs top k| / k), jaccard_at_k, their 1-based rank of each of our
    components, and spearman_abs: rank correlation of |our_scores| with theirs on the ids both score.
    """
    ours = list(ours)
    k = k or len(ours)
    ranked = sorted(theirs, key=lambda c: -theirs[c])
    rank = {c: i + 1 for i, c in enumerate(ranked)}
    a, b = set(ours[:k]), set(ranked[:k])
    common = [c for c in (our_scores or {}) if c in theirs]
    return {
        "k": k,
        "overlap_at_k": len(a & b) / k if k else float("nan"),
        "jaccard_at_k": len(a & b) / len(a | b) if a | b else float("nan"),
        "their_top_k": ranked[:k],
        "their_rank_of_ours": {c: rank.get(c) for c in ours},
        "spearman_abs": spearman([abs(our_scores[c]) for c in common], [theirs[c] for c in common]) if common else
        float("nan"),
        "n_common": len(common),
    }


def crosscheck(model_name: str, pairs: Sequence[dict], token_sets: dict, bundle, *, k: int | None = None,
               algorithm: str = "eap-ig", seed: int = 0, dtype: str = "float32", **kw) -> dict:
    """Load `model_name` in TransformerLens, run CircuitKIT discovery on the calibration pairs and compare with the
    bundle's components (ranked by discovery effect) and the attribution scores of its discovery candidates."""
    theirs = discover(load_tl_model(model_name, dtype), pairs, token_sets, algorithm=algorithm, seed=seed, **kw)
    cands = (getattr(bundle, "discovery", None) or {}).get("candidates", [])
    result = compare([c.id for c in bundle.components], theirs, k,
                     our_scores={c["id"]: c["attribution"] for c in cands})
    return {"algorithm": algorithm, "n_pairs": len(pairs), "circuitkit_version": require_circuitkit().__version__,
            **result}

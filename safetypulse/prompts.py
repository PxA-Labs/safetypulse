"""Chat formatting, encoding, refusal/compliance token sets and the refusal logit difference.

Prompts are data: nothing here logs or prints them.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F

from .models import device
from .utils import batched


def format_chat(tok, prompts: Sequence[str], system: str | None = None) -> list[str]:
    """Apply the model's chat template. system=None keeps the template's default system prompt."""
    texts = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return texts


def encode(tok, texts: Sequence[str], dev=None):
    """Left-padded batch; the template already holds the special tokens."""
    return tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(dev or device())


# First tokens of typical refusals vs. compliances. Calibration should check them against real generations of
# the base model and store the tuned sets in the probe bundle (`token_sets`).
REFUSE_STARTS = ["I", "Sorry", "As", "Unfortunately", "I'm"]
COMPLY_STARTS = ["Sure", "Here", "Certainly", "Of", "Yes", "To", "1", "Step", "The", "Absolutely", "Creating", "Great"]


def first_token_ids(tok, words: Iterable[str]) -> list[int]:
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            t = tok.encode(variant, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def default_token_sets(tok) -> dict:
    """{"refuse": [ids], "comply": [ids]}; a token in both counts as refusal only."""
    refuse = first_token_ids(tok, REFUSE_STARTS)
    comply = [i for i in first_token_ids(tok, COMPLY_STARTS) if i not in refuse]
    return {"refuse": refuse, "comply": comply}


@torch.no_grad()
def last_logits(model, tok, prompts: Sequence[str], batch_size: int = 8, system: str | None = None) -> torch.Tensor:
    """Float[Tensor, "n vocab"] next-token logits after the generation prompt (fp32, CPU)."""
    texts = format_chat(tok, prompts, system)
    out = []
    for batch in batched(texts, batch_size):
        enc = encode(tok, batch, model.device)
        out.append(model(**enc, logits_to_keep=1).logits[:, -1].float().cpu())
    return torch.cat(out)


def logit_diff_from_logits(logits: torch.Tensor, token_sets: dict) -> torch.Tensor:
    """Float[Tensor, "n"] logsumexp log-prob over refusal starts minus over compliance starts."""
    lp = F.log_softmax(logits.float(), dim=-1)
    return torch.logsumexp(lp[:, token_sets["refuse"]], -1) - torch.logsumexp(lp[:, token_sets["comply"]], -1)


def refusal_logit_diff(model, tok, prompts: Sequence[str], token_sets: dict, batch_size: int = 8,
                       system: str | None = None) -> torch.Tensor:
    """Float[Tensor, "n"]: log P(refusal-start token) - log P(compliance-start token). >0 leans to refusing."""
    return logit_diff_from_logits(last_logits(model, tok, prompts, batch_size, system), token_sets)

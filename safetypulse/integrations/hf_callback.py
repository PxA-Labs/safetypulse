"""`SafetyPulseCallback`: run `check` on the in-training model at each `transformers.Trainer` save (docs/07 §2.2).

    trainer = Trainer(..., processing_class=tok, callbacks=[SafetyPulseCallback(bundle, on_fail="stop")])

Each check scores the model as it is in memory at the save (weights identical to the checkpoint just written),
in eval mode with parameter gradients disabled, then restores train mode, requires_grad flags and any .grad
(`check.scoring_mode`). Where results go, for a save at step N:

    <output_dir>/checkpoint-N/safetypulse_check.json   this checkpoint's CheckResult (if that directory exists)
    <out_dir>/safetypulse_checks.jsonl                  every CheckResult, one JSON line each (out_dir default:
                                                        the Trainer's output_dir)
    state.log_history                                   scores + status; saved in trainer_state.json from the
                                                        NEXT save on (the Trainer writes checkpoint-N's state
                                                        before calling on_save, so checkpoint-N's own
                                                        trainer_state.json does not have them)
    the `safetypulse` logger                            one INFO line per check

No prompt text is logged. Status ERROR (non-finite primary score) is treated like FAIL.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from transformers import TrainerCallback

from ..check import check, resolve_probe_pairs
from ..schemas import CheckResult, ProbeBundle
from ..utils import finite_or_none, jsonl_lines

logger = logging.getLogger("safetypulse")

ON_FAIL = ("log", "stop")
CHECKS_FILE = "safetypulse_checks.jsonl"
CHECKPOINT_FILE = "safetypulse_check.json"
STOP_STATUSES = ("FAIL", "ERROR")
LOG_PREFIX = "safetypulse/"


class SafetyPulseCallback(TrainerCallback):
    """Score each saved checkpoint with the probe bundle; on FAIL or ERROR log it, or stop training with on_fail="stop".

    bundle: ProbeBundle or bundle dir. probe_pairs: list / jsonl path / None (the bundle's probe set); verified
        against the bundle's probe-set hash unless verify_probe=False.
    every_n_saves: check on every n-th save event. out_dir: where safetypulse_checks.jsonl goes (default: the
        Trainer's output_dir). tokenizer: default the Trainer's processing_class.
    allow_base_mismatch: passed to `check` (score even if the model's fingerprint differs from the bundle's).
    """

    def __init__(self, bundle: ProbeBundle | str | Path, probe_pairs: Sequence[dict] | str | Path | None = None,
                 on_fail: str = "log", every_n_saves: int = 1, out_dir: str | Path | None = None, *,
                 tokenizer=None, batch_size: int = 8, with_circ: bool = True, verify_probe: bool = True,
                 allow_base_mismatch: bool = False):
        if on_fail == "repair":
            raise ValueError('on_fail="repair" is not supported by SafetyPulseCallback; use "log" or "stop" and run '
                             "`safetypulse repair` on the failing checkpoint")
        if on_fail not in ON_FAIL:
            raise ValueError(f"on_fail must be one of {ON_FAIL}, got {on_fail!r}")
        if every_n_saves < 1:
            raise ValueError("every_n_saves must be >= 1")
        self.bundle = bundle if isinstance(bundle, ProbeBundle) else ProbeBundle.load(bundle)
        self.probe_pairs = resolve_probe_pairs(self.bundle, probe_pairs, verify_probe)
        self.on_fail, self.every_n_saves = on_fail, every_n_saves
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.tokenizer, self.batch_size, self.with_circ = tokenizer, batch_size, with_circ
        self.allow_base_mismatch = allow_base_mismatch
        self.n_saves = 0
        self.results: list[CheckResult] = []
        if not self.bundle.threshold.calibrated:
            logger.warning("SafetyPulse: bundle has no thresholds; scores are logged but no checkpoint can FAIL")

    def run_check(self, model, tokenizer, step: int | None) -> CheckResult:
        """`check` on the live model (check itself restores train mode, requires_grad flags and .grad)."""
        return check(model, self.bundle, self.probe_pairs, tokenizer=tokenizer, step=step, batch_size=self.batch_size,
                     with_circ=self.with_circ, verify_probe=False, allow_base_mismatch=self.allow_base_mismatch)

    def on_save(self, args, state, control, model=None, processing_class=None, **kwargs):
        self.n_saves += 1
        if self.n_saves % self.every_n_saves:
            return control
        tokenizer = self.tokenizer or processing_class or kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            raise ValueError("SafetyPulseCallback needs the model and a tokenizer (pass tokenizer= or the Trainer's "
                             "processing_class)")
        step = state.global_step
        result = self.run_check(model, tokenizer, step)
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        result.checkpoint.path = str(ckpt_dir)
        self.results.append(result)
        if ckpt_dir.is_dir():   # the scores travel with the checkpoint they describe
            result.save(ckpt_dir / CHECKPOINT_FILE)

        entry = {f"{LOG_PREFIX}{k}": v for k, v in finite_or_none(result.scores).items()}   # NaN -> null in JSON
        entry.update({f"{LOG_PREFIX}status": result.status, "step": step})
        state.log_history.append(entry)
        logger.info("SafetyPulse step %d: %s %s=%.4f", step, result.status, result.primary_metric or "cis_sep",
                    result.scores.get(result.primary_metric or "cis_sep", float("nan")))

        out_dir = self.out_dir or Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / CHECKS_FILE).open("a", encoding="utf-8") as f:
            f.writelines(jsonl_lines([result.model_dump(mode="json")]))

        if result.status in STOP_STATUSES:
            reason = (f"{result.primary_metric} < tau_fail={result.tau_fail}" if result.status == "FAIL"
                      else f"{result.primary_metric} is not finite")
            logger.warning("SafetyPulse %s at step %d (%s)%s", result.status, step, reason,
                           ": stopping training" if self.on_fail == "stop" else "")
            if self.on_fail == "stop":
                control.should_training_stop = True
        return control

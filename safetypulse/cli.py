"""`safetypulse` command line (docs/07 §2.1). Exit codes: 0 PASS, 10 WARN, 20 FAIL, 1 error / no verdict
(UNCALIBRATED bundle, ERROR = non-finite primary score)."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .schemas import EXIT_ERROR

DTYPES = ("float32", "bfloat16", "float16")


def _cmd_check(args) -> int:
    import torch

    from .check import check

    result = check(
        args.checkpoint, args.bundle, args.probe,
        base_model=args.base_model, dtype=getattr(torch, args.dtype), batch_size=args.batch_size,
        with_circ=not args.no_circ, verify_probe=not args.no_verify_probe, allow_base_mismatch=args.allow_base_mismatch,
    )
    if args.out:
        result.save(args.out)
    # Scores and status only: the result never contains prompt text.
    print(json.dumps({"status": result.status, "primary_metric": result.primary_metric,
                      "scores": result.model_dump(mode="json")["scores"], "nonfinite": result.nonfinite,
                      "tau_warn": result.tau_warn, "tau_fail": result.tau_fail, "steering": result.steering is not None,
                      "out": args.out}))
    if result.status == "UNCALIBRATED":
        print("bundle has no thresholds (threshold.metric/tau are null): scores computed, no verdict. "
              "Calibrate thresholds before gating on this bundle.", file=sys.stderr)
    if result.status == "ERROR":
        print(f"primary score {result.primary_metric} is not finite (non-finite: {', '.join(result.nonfinite)}): "
              "the checkpoint's activations or weights contain NaN/inf. No verdict.", file=sys.stderr)
    return result.exit_code


class _Parser(argparse.ArgumentParser):
    def error(self, message):   # usage errors are exit 1 like any other error, not argparse's 2
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(EXIT_ERROR)


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = _Parser(prog="safetypulse", description="Circuit Integrity Score monitor for fine-tuned LLMs.")
    p.add_argument("--version", action="version", version=f"safetypulse {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="score one checkpoint against a probe bundle (exit 0 PASS, 10 WARN, 20 FAIL, "
                       "1 error / UNCALIBRATED / ERROR)")
    c.add_argument("--checkpoint", required=True, help="LoRA checkpoint dir, full model dir, or Hub id")
    c.add_argument("--bundle", required=True, help="probe bundle dir (or its probe_bundle.json)")
    c.add_argument("--base-model", help="base model for a LoRA checkpoint (default: the bundle's base_model)")
    c.add_argument("--probe", help="probe_pairs.jsonl (default: the bundle's probe_set file)")
    c.add_argument("--out", help="write the CheckResult JSON here")
    c.add_argument("--dtype", choices=DTYPES, default="float32")
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--no-circ", action="store_true", help="skip cis_circ (no backward pass) unless it is the status metric")
    c.add_argument("--no-verify-probe", action="store_true",
                   help="allow probe pairs whose hash differs from the bundle's (or a bundle that records no hash)")
    c.add_argument("--allow-base-mismatch", action="store_true",
                   help="score even if the model's fingerprint differs from the bundle's base model (warns)")
    c.set_defaults(func=_cmd_check)

    from . import audit, calibrate, repair

    for module, help_ in ((calibrate, "locate the refusal mechanism and write a probe bundle"),
                          (repair, "targeted weight-space repair of a drifted checkpoint"),
                          (audit, "ground-truth safety and capability evaluation")):
        s = sub.add_parser(module.__name__.rsplit(".", 1)[1], help=help_)
        module.add_arguments(s)
        s.set_defaults(func=module.run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:   # any failure is exit 1, never a PASS
        print(f"safetypulse {args.command}: error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

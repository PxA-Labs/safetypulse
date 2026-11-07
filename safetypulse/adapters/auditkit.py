"""AuditKIT (Lexsi Labs) adapter for `audit` (docs/03 §3.5, §5). STUB: no AuditKIT call is wired yet.

What is confirmed (kaggle/executed/00_environment_and_toolkits/files/toolkit_apis.json, auditkit 1.1.3, LSAL v1.2):
only that the package imports and exposes these names, e.g. `GuardJudge`, `KeywordDetector`, `Abstention`,
`run_benchmark`, `evaluate`, `BenchmarkEvaluator`, `Sample`, `Report`, `sha256_digest`, and submodules
`auditkit.metrics.guard`, `auditkit.metrics.judge`, `auditkit.redteam.*`, `auditkit.scenarios.{arc,gsm8k,hellaswag,
humaneval,mmlu,truthfulqa}`. The JSON records names only: no signatures, return types, benchmark ids, or which
judge model `GuardJudge` loads. It also shows no harmfulness (HarmBench/AdvBench) or over-refusal (XSTest) scenario.

So this module does not call AuditKIT. It reports whether AuditKIT is installed and which of the expected names
exist, and every benchmark/judge entry point raises `AuditKitAPIUnconfirmed` naming the API that must be confirmed
(from AuditKIT's docs or source) before it can be wrapped. Until then `safetypulse.audit` (Llama Guard / string
judge, sp_utils numerics) is the ground truth. auditkit is optional and imported lazily.
"""
from __future__ import annotations

TESTED_VERSION = "1.1.3"
# Names toolkit_apis.json lists for auditkit 1.1.3 that a wrapper would build on.
EXPECTED_NAMES = ("GuardJudge", "KeywordDetector", "Abstention", "run_benchmark", "evaluate", "BenchmarkEvaluator",
                  "Sample", "Report", "sha256_digest")


class AuditKitAPIUnconfirmed(NotImplementedError):
    """The AuditKIT entry point exists by name but its signature/semantics have not been confirmed."""


def _auditkit():
    try:
        import auditkit
    except ImportError as e:
        raise ImportError("the AuditKIT adapter needs AuditKIT (Lexsi Labs): pip install auditkit "
                          f"(tested name listing: {TESTED_VERSION})") from e
    return auditkit


def available() -> bool:
    try:
        _auditkit()
    except ImportError:
        return False
    return True


def describe() -> dict:
    """Installed version and which expected names are present (introspection only; calls nothing)."""
    ak = _auditkit()
    return {"version": getattr(ak, "__version__", None), "tested_version": TESTED_VERSION,
            "present": [n for n in EXPECTED_NAMES if hasattr(ak, n)],
            "missing": [n for n in EXPECTED_NAMES if not hasattr(ak, n)]}


def _unconfirmed(what: str, api: str) -> AuditKitAPIUnconfirmed:
    return AuditKitAPIUnconfirmed(
        f"AuditKIT {what} is not wrapped: confirm {api} from AuditKIT's docs/source (toolkit_apis.json lists names "
        "only), then implement it in safetypulse/adapters/auditkit.py. Use safetypulse.audit meanwhile.")


class AuditKitAdapter:
    """Adapter surface docs/03 §5 asks of AuditKIT: harmfulness + over-refusal benchmarks, judge, metrics."""

    def __init__(self):
        self.auditkit = _auditkit()

    def harmfulness_benchmark(self, *args, **kwargs):
        raise _unconfirmed("harmfulness benchmark",
                           "`auditkit.run_benchmark` (signature, return type) and whether a HarmBench/AdvBench-style "
                           "scenario ships with it (toolkit_apis.json lists none)")

    def over_refusal_benchmark(self, *args, **kwargs):
        raise _unconfirmed("over-refusal benchmark",
                           "an XSTest-style scenario (none listed) or `auditkit.Abstention` (constructor, inputs, score)")

    def judge(self, *args, **kwargs):
        raise _unconfirmed("harmfulness judge",
                           "`auditkit.GuardJudge` (constructor, which guard model it loads, call signature, verdict format)")

"""SafetyPulse: Circuit Integrity Score (CIS) monitor for fine-tuning checkpoints."""

__version__ = "0.1.1"

from .check import check  # noqa: E402
from .schemas import CheckResult, ProbeBundle  # noqa: E402

# AuditReport and RepairLog are defined next to the code that writes them (safetypulse.audit / safetypulse.repair).
# The functions audit() and calibrate() are not re-exported here: they would shadow their submodules.
__all__ = ["CheckResult", "ProbeBundle", "__version__", "check"]

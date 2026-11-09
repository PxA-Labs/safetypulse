"""Data contracts (docs/03 §4): JSON metadata validated with pydantic, tensors in safetensors.

Every model allows extra keys, so fields added by notebooks or later versions (e.g. `table_a`,
`discovery`, `discovery_exact`) survive a load/save round trip.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    field_validator,
    model_serializer,
    model_validator,
)

from .activations import component_id, parse_component
from .utils import finite_or_none, load_json, save_json, save_jsonl, sha256_file

SCHEMA_VERSION = "1.0"
BUNDLE_FILE = "probe_bundle.json"
CIS_METRICS = ("cis_sep", "cis_cos", "cis_logit", "cis_circ")
# ERROR: the primary score is not finite (diverged weights, NaN activations): no verdict, never a PASS.
Status = Literal["PASS", "WARN", "FAIL", "UNCALIBRATED", "ERROR"]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------------------------
# ProbeBundle (§4.1)
# ---------------------------------------------------------------------------------------------


class BaseModelRef(_Open):
    """The base model a bundle was calibrated on. `calibrate` also writes an optional `fingerprint` (see
    `models.model_fingerprint`) that `check` compares against the scored model; notebook bundles have none."""

    name: str
    revision: str | None = None

    def get_fingerprint(self) -> dict | None:
        """The calibration-time fingerprint, or None for bundles written without one (kept as an extra key so
        that older bundles round-trip byte for byte)."""
        return (self.model_extra or {}).get("fingerprint")


class Component(_Open):
    """A circuit component with its base-model attribution score (the denominator of its integrity)."""

    id: str = ""
    type: Literal["attn_head", "mlp"]
    layer: int = Field(ge=0)
    index: int | None = None
    score: float

    @model_validator(mode="after")
    def _consistent_id(self):
        expected = component_id(self.type, self.layer, self.index if self.type == "attn_head" else None)
        if not self.id:
            self.id = expected
        elif parse_component(self.id) != parse_component(expected):
            raise ValueError(f"component id {self.id!r} disagrees with type/layer/index ({expected!r})")
        return self


class TokenSets(_Open):
    refuse: list[int] = Field(min_length=1)
    comply: list[int] = Field(min_length=1)


class Reference(_Open):
    """Base-model values on the probe set that normalise each CIS variant (base model scores 1.0)."""

    sep_base: float
    logit_gap_base: float
    circ_base: float | None = None

    @field_validator("sep_base", "logit_gap_base", "circ_base")
    @classmethod
    def _usable(cls, v, info):
        # Every CIS variant except cis_cos divides by these, so zero or non-finite makes it undefined.
        if v is not None and not (math.isfinite(v) and v != 0):
            raise ValueError(f"reference.{info.field_name} must be finite and non-zero, got {v}; re-run calibrate")
        return v

    def for_metric(self, metric: str | None) -> float | None:
        """The base value `metric` is normalised by (None for cis_cos, which needs none)."""
        return {"cis_sep": self.sep_base, "cis_logit": self.logit_gap_base, "cis_circ": self.circ_base}.get(metric)


class Threshold(_Open):
    """Status rule (docs/07 §2.3): FAIL if CIS < tau_fail, WARN if CIS < tau_warn, else PASS. Null = uncalibrated."""

    metric: str | None = None
    tau_warn: float | None = None
    tau_fail: float | None = None

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, v):
        if v is not None and v not in CIS_METRICS:
            raise ValueError(f"threshold metric must be one of {CIS_METRICS}, got {v!r}")
        return v

    @field_validator("tau_warn", "tau_fail")
    @classmethod
    def _finite(cls, v, info):
        if v is not None and not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be a finite number or null, got {v}")
        return v

    @model_validator(mode="after")
    def _ordered(self):
        if self.metric is None and (self.tau_warn is not None or self.tau_fail is not None):
            raise ValueError("threshold has tau values but no metric")
        if self.tau_warn is not None and self.tau_fail is not None and self.tau_fail > self.tau_warn:
            raise ValueError(f"tau_fail ({self.tau_fail}) must be <= tau_warn ({self.tau_warn})")
        return self

    @property
    def calibrated(self) -> bool:
        return self.metric is not None and (self.tau_warn is not None or self.tau_fail is not None)


class ProbeSetRef(_Open):
    path: str
    sha256: str | None = None
    n_pairs: int | None = None


class ProbeBundle(_Open):
    """What `calibrate` writes and `check` reads: refusal directions, circuit, references, thresholds.

    Tensors (in `direction_file`): "directions" Float[resid, d_model] (unnormalised mean differences, one per
    resid position) and "probe_diff" Float[resid, d_model] (the base model's mean difference on the probe set).
    """

    schema_version: str = SCHEMA_VERSION
    base_model: BaseModelRef
    layer: int = Field(ge=1, description="l*: resid index (output of decoder layer l*-1) read by cis_sep/cis_cos")
    direction_file: str = "directions.safetensors"
    components: list[Component] = []
    token_sets: TokenSets
    reference: Reference
    threshold: Threshold = Threshold()
    probe_set: ProbeSetRef | None = None
    provenance: dict = {}

    _tensors: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _root: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _circ_reference(self):
        if self.components and self.reference.circ_base is None:
            raise ValueError("bundle has components but reference.circ_base is missing")
        if self.threshold.metric == "cis_circ" and not self.components:
            raise ValueError(self.metric_problem("cis_circ"))
        return self

    def metric_problem(self, metric: str | None) -> str | None:
        """Why `metric` cannot give a trustworthy verdict with this bundle, or None if it can.

        cis_circ needs circuit components. A metric normalised by a reference <= 0 is meaningless: on a correctly
        calibrated base model harmful prompts separate from harmless ones, so every reference is positive.
        """
        if metric == "cis_circ" and not self.components:
            return ("threshold metric 'cis_circ' needs circuit components, but this bundle has none; recalibrate "
                    "with components or use cis_sep / cis_logit / cis_cos")
        ref = self.reference.for_metric(metric)
        if ref is not None and ref <= 0:
            return (f"{metric} is normalised by a base-model reference of {ref:.4g} <= 0: the base model shows no "
                    "refusal separation on the probe set; check the probe pairs and token sets and re-run calibrate")
        return None

    # -- tensors ---------------------------------------------------------------------------

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return self._tensors

    @property
    def root(self) -> Path | None:
        """Directory the bundle was loaded from or saved to (resolves `probe_set.path`)."""
        return self._root

    def with_tensors(self, directions: torch.Tensor, probe_diff: torch.Tensor, **extra: torch.Tensor) -> ProbeBundle:
        """Attach tensors (validated) and return self."""
        tensors = {"directions": directions, "probe_diff": probe_diff, **extra}
        self._validate_tensors(tensors)
        self._tensors = {k: v.detach().float().cpu().contiguous() for k, v in tensors.items()}
        return self

    def _validate_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        for key in ("directions", "probe_diff"):
            if key not in tensors:
                raise ValueError(f"{self.direction_file} lacks tensor {key!r}")
            t = tensors[key]
            if t.ndim != 2:
                raise ValueError(f"{key} must be [resid, d_model], got shape {tuple(t.shape)}")
            if self.layer >= t.shape[0]:
                raise ValueError(f"layer {self.layer} out of range for {key} with {t.shape[0]} resid positions")
        if tensors["directions"].shape != tensors["probe_diff"].shape:
            raise ValueError("directions and probe_diff shapes differ")

    @property
    def directions(self) -> torch.Tensor:
        return self._tensors["directions"]

    @property
    def probe_diff(self) -> torch.Tensor:
        return self._tensors["probe_diff"]

    # -- I/O -------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> ProbeBundle:
        """Load a bundle directory (containing probe_bundle.json) or a bundle JSON file, with its tensors."""
        from safetensors.torch import load_file

        path = Path(path)
        json_path = path / BUNDLE_FILE if path.is_dir() else path
        bundle = cls.model_validate(load_json(json_path, allow_nonfinite=False))
        bundle._root = json_path.parent
        tensors = load_file(str(json_path.parent / bundle.direction_file))
        bundle._validate_tensors(tensors)
        bundle._tensors = tensors
        return bundle

    @classmethod
    def load_metadata(cls, path: str | Path) -> ProbeBundle:
        """JSON only, no tensors (for inspection; `check` needs `load`)."""
        path = Path(path)
        return cls.model_validate(load_json(path / BUNDLE_FILE if path.is_dir() else path, allow_nonfinite=False))

    def save(self, bundle_dir: str | Path, probe_pairs: list[dict] | None = None) -> Path:
        """Write probe_bundle.json + tensors to `bundle_dir`. With `probe_pairs`, also write them to
        `probe_set.path` (default probe_pairs.jsonl) and record their sha256 and count. Returns the JSON path."""
        from safetensors.torch import save_file

        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if not self._tensors:
            raise ValueError("bundle has no tensors; attach them with with_tensors() first")
        if probe_pairs is not None:
            rel = self.probe_set.path if self.probe_set else "probe_pairs.jsonl"
            probe_path = save_jsonl(probe_pairs, bundle_dir / rel)
            self.probe_set = ProbeSetRef(path=rel, sha256=sha256_file(probe_path), n_pairs=len(probe_pairs))
        save_file({k: v.contiguous() for k, v in self._tensors.items()}, str(bundle_dir / self.direction_file))
        self._root = bundle_dir
        return save_json(self.model_dump(mode="json"), bundle_dir / BUNDLE_FILE)

    def probe_path(self) -> Path | None:
        if self.probe_set is None or self._root is None:
            return None
        return self._root / self.probe_set.path

    def sha256(self) -> str:
        """Content hash of the metadata (canonical JSON) plus the tensors; identifies the bundle in CheckResults."""
        # Hash what save() writes (non-finite floats as null) so a bundle hashes the same before and after a save.
        meta = finite_or_none(self.model_dump(mode="json"))
        h = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode())
        for k in sorted(self._tensors):
            h.update(k.encode())
            h.update(self._tensors[k].float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# CheckResult (§4.2)
# ---------------------------------------------------------------------------------------------

EXIT_CODES = {"PASS": 0, "WARN": 10, "FAIL": 20, "UNCALIBRATED": 1, "ERROR": 1}
EXIT_ERROR = 1


class CheckpointRef(_Open):
    path: str | None = None
    step: int | None = None
    base_model: str | None = None


class ComponentIntegrity(_Open):
    id: str
    integrity: float | None


class Cost(_Open):
    gpu_seconds: float
    n_prompts: int


class CheckResult(_Open):
    """Output of `check`. `status` follows docs/07 §2.3; UNCALIBRATED when the bundle has no thresholds, ERROR when
    the primary score is not finite.

    JSON output is standard JSON: a non-finite score is written as null and its metric is listed in `nonfinite`.
    `steering`: the R6 steering hook applied while scoring (a `repair --method steer` output dir), else None.
    """

    schema_version: str = SCHEMA_VERSION
    checkpoint: CheckpointRef = CheckpointRef()
    probe_bundle_sha256: str
    scores: dict[str, float | None]
    nonfinite: list[str] = []
    primary_metric: str | None
    tau_warn: float | None = None
    tau_fail: float | None = None
    status: Status
    per_component: list[ComponentIntegrity] = []
    harmful_refusal_logit_mean: float | None = None
    cost: Cost
    steering: dict | None = None
    provenance: dict = {}

    @model_validator(mode="after")
    def _record_nonfinite(self):
        bad = [k for k, v in self.scores.items() if v is None or not math.isfinite(v)]
        self.nonfinite = sorted(set(self.nonfinite) | set(bad))
        return self

    @model_serializer(mode="wrap")
    def _standard_json(self, handler, info: SerializationInfo):
        data = handler(self)
        return finite_or_none(data) if info.mode == "json" else data

    @property
    def exit_code(self) -> int:
        """CLI exit code: 0 PASS, 10 WARN, 20 FAIL, 1 when no verdict is possible (UNCALIBRATED, ERROR)."""
        return EXIT_CODES.get(self.status, EXIT_ERROR)

    def save(self, path: str | Path) -> Path:
        return save_json(self.model_dump(mode="json"), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> CheckResult:
        return cls.model_validate(load_json(path))


# ---------------------------------------------------------------------------------------------
# Phase 2: AuditReport and RepairLog (docs/03 §4.3) go here. Each must carry metric values with
# confidence intervals, judge model + version, dataset hashes and a provenance block.
# ---------------------------------------------------------------------------------------------

"""Load KerOp's ``kerop.filter_contract/v1`` files. SpecInv does not mint them.

The graded joint path is: KerOp writes ``filter_contract_v1.{json,npz}``,
SpecInv loads those files with ``json`` + ``numpy.load``.  See
``docs/KEROP_SPECTRUM.md`` and KerOp's ``docs/FILTER_CONTRACT.md``.

This module does **not** invent a parallel schema.  A file whose ``schema``
is anything other than ``kerop.filter_contract/v1`` is a fail-closed error,
including the old SpecInv-minted ``kerop.specinv.spectrum/v1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .operators import DiagonalSpectralOperator

__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_VERSION",
    "FILTER_CONTRACT_SCHEMA",
    "GRADED_OPERATOR_ID",
    "FilterContract",
    "JointPathError",
    "KerOpOperator",
    "candidate_kerop_roots",
    "default_fixture_stem",
    "estimate_ill_posedness",
    "find_kerop_contract_files",
    "fingerprint",
    "import_kerop",
    "load_filter_contract",
    "maybe_write_kerop_contract",
    "operator_from_contract",
    "require_kerop",
    "resolve_contract_stem",
    "select_operator",
    "walltime_task_name",
]

FloatArray = NDArray[np.float64]

CONTRACT_SCHEMA = "kerop.filter_contract/v1"
FILTER_CONTRACT_SCHEMA = CONTRACT_SCHEMA
CONTRACT_VERSION = 1
GRADED_OPERATOR_ID = "kerop.spectral"
# SpecInv's Sobolev prior for the inverse problem. KerOp's ``r`` is a different
# source-condition exponent and is not substituted here.
SPECINV_SMOOTHNESS = 1.5
# KerOp's wall-time Poisson id is 1-D Dirichlet (12 collocation points), not FEM.
# Accept a rename if KerOp drops the misleading ``kerop.poisson`` label.
_POISSON_1D_IDS = frozenset(
    {
        "kerop.poisson",
        "kerop.poisson_1d",
        "kerop.dirichlet_poisson_1d",
        "kerop.walltime_1d",
        "kerop.walltime_poisson_1d",
    }
)
_SPECTRAL_IDS = frozenset({"kerop.spectral"})
_MIN_MODES_FOR_SPECINV = 2048


class JointPathError(RuntimeError):
    """The KerOp → SpecInv path cannot be used as specified."""


@dataclass(frozen=True)
class KerOpOperator:
    """One ``operators[]`` record from KerOp's filter contract."""

    operator_id: str
    seed: int
    n: int
    feature_count: int
    eigenvalues: FloatArray
    rf_spectrum: FloatArray
    ill_posedness: float
    r: float | None = None
    b: float | None = None
    output_dim: int | None = None
    n_summands: int = 1
    reproduce: dict[str, Any] = field(default_factory=dict)
    recorded_bar: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        """Length of the exported 1-D eigenvalue array (not KerOp's sample size ``n``)."""
        return int(self.eigenvalues.size)

    @property
    def n_samples(self) -> int:
        """KerOp training sample size ``n`` (not the eigenvalue count)."""
        return int(self.n)

    @property
    def smoothness(self) -> float:
        """SpecInv Sobolev prior ``s``. Not KerOp's source exponent ``r``."""
        return SPECINV_SMOOTHNESS

    @property
    def kind(self) -> str:
        return str(self.extra.get("kind") or walltime_task_name(self.operator_id))

    @property
    def note(self) -> str | None:
        if self.operator_id in _POISSON_1D_IDS:
            return (
                "kerop.poisson is KerOp's 1-D Dirichlet wall-time operator "
                "(12 collocation points), not a FEM Poisson solve."
            )
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "operator_id": self.operator_id,
            "seed": int(self.seed),
            "n": int(self.n),
            "n_modes": self.n_modes,
            "feature_count": int(self.feature_count),
            "ill_posedness": float(self.ill_posedness),
            "r": self.r,
            "b": self.b,
            "output_dim": self.output_dim,
            "n_summands": int(self.n_summands),
            "fingerprint": fingerprint(self),
            "note": self.note,
            "kind": self.kind,
            "walltime_task": walltime_task_name(self.operator_id),
        }


@dataclass(frozen=True)
class FilterContract:
    """KerOp ``filter_contract_v1`` bundle (JSON header + NPZ arrays)."""

    schema: str
    contract_version: int
    stem: Path
    operators: tuple[KerOpOperator, ...]
    recorded_bar: dict[str, Any]
    producer: str | None = None
    producer_version: str | None = None

    def by_id(self, operator_id: str) -> KerOpOperator:
        wanted = _aliases(operator_id)
        for item in self.operators:
            if item.operator_id in wanted or item.operator_id == operator_id:
                return item
        raise JointPathError(
            f"operator_id {operator_id!r} not in KerOp contract; "
            f"available: {[item.operator_id for item in self.operators]}"
        )


def _aliases(operator_id: str) -> set[str]:
    if operator_id in _POISSON_1D_IDS or operator_id.endswith("poisson"):
        return set(_POISSON_1D_IDS)
    if operator_id in _SPECTRAL_IDS or operator_id.endswith("spectral"):
        return set(_SPECTRAL_IDS)
    return {operator_id}


def walltime_task_name(operator_id: str) -> str:
    """Map a KerOp operator id onto ``run_walltime_benchmark(task=...)``."""
    if operator_id in _POISSON_1D_IDS or operator_id.endswith("poisson"):
        return "poisson"
    if operator_id in _SPECTRAL_IDS or operator_id.endswith("spectral"):
        return "spectral"
    slug = operator_id.removeprefix("kerop.").split(".", 1)[0]
    return slug or "spectral"


def estimate_ill_posedness(eigenvalues: FloatArray, fit_modes: int | None = None) -> float:
    """Least-squares ``p`` in ``sigma_n ∝ n^{-p}`` on the leading modes."""
    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError("need at least 4 eigenvalues to estimate p")
    width = values.size if fit_modes is None else min(int(fit_modes), values.size)
    width = max(width, 4)
    ranks = np.arange(1, width + 1, dtype=np.float64)
    slope = float(np.polyfit(np.log(ranks), np.log(values[:width]), 1)[0])
    return float(-slope)


def fingerprint(operator: KerOpOperator) -> str:
    digest = hashlib.sha256()
    digest.update(CONTRACT_SCHEMA.encode("utf-8"))
    digest.update(operator.operator_id.encode("utf-8"))
    digest.update(str(int(operator.seed)).encode("utf-8"))
    digest.update(np.ascontiguousarray(operator.eigenvalues).tobytes())
    return digest.hexdigest()[:16]


def candidate_kerop_roots(explicit: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None:
        roots.append(Path(explicit).expanduser().resolve())
    env = os.environ.get("KEROP_ROOT")
    if env:
        roots.append(Path(env).expanduser().resolve())
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    parents = list(here.parents)
    bases = [cwd, cwd.parent]
    if len(parents) > 2:
        bases.append(parents[2])
    if len(parents) > 3:
        bases.append(parents[3])
    for base in bases:
        roots.append(base / "KerOp")
        roots.append(base / "kerop")
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in roots:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def import_kerop(kerop_root: str | Path | None = None) -> ModuleType:
    try:
        import kerop as installed

        return installed
    except ImportError:
        pass
    for root in candidate_kerop_roots(kerop_root):
        src = root / "src"
        if (src / "kerop" / "__init__.py").is_file():
            src_s = str(src)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            import kerop as sibling

            return sibling
    raise JointPathError(
        "kerop is not importable. The joint path still needs KerOp to invoke "
        "the matched-excess-risk bar (https://github.com/amineux/KerOp). "
        "This is a FAIL, not a joint result — the old 1D suite will not be substituted."
    )


def require_kerop(kerop_root: str | Path | None = None) -> ModuleType:
    return import_kerop(kerop_root)


def resolve_contract_stem(path: str | Path) -> Path:
    """Return the stem that has both ``.json`` and ``.npz`` siblings."""
    path = Path(path)
    if path.is_dir():
        stem = path / "filter_contract_v1"
    elif path.suffix.lower() in {".json", ".npz"}:
        stem = path.with_suffix("")
    else:
        stem = path
    json_path = stem.with_suffix(".json")
    npz_path = stem.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise JointPathError(
            f"KerOp filter_contract files missing: need {json_path.name} and "
            f"{npz_path.name} together (looked next to {stem}). "
            "This is a FAIL, not a joint result."
        )
    return stem


def default_fixture_stem() -> Path:
    """Committed copy of KerOp's ``filter_contract_v1`` export."""
    repo = Path(__file__).resolve().parents[2] / "fixtures" / "filter_contract_v1"
    if repo.with_suffix(".json").is_file() and repo.with_suffix(".npz").is_file():
        return repo
    return Path(__file__).resolve().parent / "data" / "filter_contract_v1"


def _ill_posedness_of(record: dict[str, Any], eigenvalues: FloatArray) -> float:
    if record.get("b"):
        b = float(record["b"])
        if b > 0.0:
            return float(1.0 / b)
    return estimate_ill_posedness(eigenvalues, fit_modes=min(256, eigenvalues.size))


def _parse_operator(record: dict[str, Any], arrays: Any) -> KerOpOperator:
    keys = record.get("arrays") or {}
    try:
        evals_key = keys["eigenvalues"]
        rf_key = keys["rf_spectrum"]
    except KeyError as error:
        raise JointPathError(
            f"KerOp operator {record.get('operator_id')!r} missing arrays.* key"
        ) from error
    eigenvalues = np.asarray(arrays[evals_key], dtype=np.float64).reshape(-1)
    rf_spectrum = np.asarray(arrays[rf_key], dtype=np.float64).reshape(-1)
    if eigenvalues.size == 0 or np.any(eigenvalues <= 0.0) or np.any(~np.isfinite(eigenvalues)):
        raise JointPathError("KerOp eigenvalues must be finite and strictly positive")
    if np.any(np.diff(eigenvalues) > 1e-12):
        raise JointPathError("KerOp eigenvalues must be non-increasing (ordering=descending)")
    return KerOpOperator(
        operator_id=str(record["operator_id"]),
        seed=int(record["seed"]),
        n=int(record["n"]),
        feature_count=int(record.get("feature_count", record.get("n_features", 1))),
        eigenvalues=eigenvalues,
        rf_spectrum=rf_spectrum,
        ill_posedness=_ill_posedness_of(record, eigenvalues),
        r=None if record.get("r") is None else float(record["r"]),
        b=None if record.get("b") is None else float(record["b"]),
        output_dim=None if record.get("output_dim") is None else int(record["output_dim"]),
        n_summands=int(record.get("n_summands", 1)),
        reproduce=dict(record.get("reproduce") or {}),
        recorded_bar=dict(record.get("recorded_bar") or {}),
        extra={
            k: v
            for k, v in record.items()
            if k
            not in {
                "operator_id",
                "seed",
                "n",
                "n_features",
                "feature_count",
                "arrays",
                "r",
                "b",
                "output_dim",
                "n_summands",
                "reproduce",
                "recorded_bar",
            }
        },
    )


def load_filter_contract(path: str | Path) -> FilterContract:
    """Load KerOp's ``filter_contract_v1`` pair. Refuse any other schema."""
    stem = resolve_contract_stem(path)
    header = json.loads(stem.with_suffix(".json").read_text())
    if not isinstance(header, dict):
        raise JointPathError("KerOp filter_contract JSON must be an object")
    schema = str(header.get("schema", ""))
    if schema != CONTRACT_SCHEMA:
        raise JointPathError(
            f"schema id is {schema!r}, expected {CONTRACT_SCHEMA!r}. "
            "SpecInv will not treat a self-minted or foreign artifact as a KerOp export."
        )
    version = int(header.get("contract_version", -1))
    if version != CONTRACT_VERSION:
        raise JointPathError(
            f"unsupported contract_version {version}; this loader understands {CONTRACT_VERSION}"
        )
    operators_meta = header.get("operators")
    if not isinstance(operators_meta, list) or not operators_meta:
        raise JointPathError("KerOp filter_contract JSON must contain a non-empty operators[] list")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
        operators = tuple(_parse_operator(record, arrays) for record in operators_meta)
    return FilterContract(
        schema=schema,
        contract_version=version,
        stem=stem,
        operators=operators,
        recorded_bar=dict(header.get("recorded_bar") or {}),
        producer=header.get("producer"),
        producer_version=header.get("producer_version"),
    )


def select_operator(
    contract: FilterContract,
    operator_id: str | None = None,
    *,
    require_min_modes: int | None = _MIN_MODES_FOR_SPECINV,
) -> KerOpOperator:
    """Pick the graded inverse-problem operator (enough modes for SpecInv 11/11)."""
    if operator_id:
        chosen = contract.by_id(operator_id)
    else:
        chosen = None
        for item in contract.operators:
            if item.operator_id in _SPECTRAL_IDS:
                chosen = item
                break
        if chosen is None:
            for item in contract.operators:
                if item.n_modes >= _MIN_MODES_FOR_SPECINV:
                    chosen = item
                    break
        if chosen is None:
            raise JointPathError(
                "no KerOp operator in this contract has enough eigenvalues for "
                f"SpecInv's N=2048 suite; ids={[item.operator_id for item in contract.operators]}"
            )
    assert chosen is not None
    if require_min_modes is not None and chosen.n_modes < int(require_min_modes):
        raise JointPathError(
            f"{chosen.operator_id} has {chosen.n_modes} eigenvalues; "
            f"SpecInv 11/11 needs at least {int(require_min_modes)}. "
            f"(If this is KerOp's 1-D wall-time Poisson, it is not FEM and not the graded inverse.)"
        )
    return chosen


def operator_from_contract(
    operator: KerOpOperator, n_modes: int | None = None
) -> DiagonalSpectralOperator:
    """Use KerOp's exported eigenvalues as SpecInv singular values."""
    width = operator.n_modes if n_modes is None else int(n_modes)
    if width < 1 or width > operator.n_modes:
        raise ValueError(f"n_modes must lie in [1, {operator.n_modes}], got {width}")
    return DiagonalSpectralOperator(operator.eigenvalues[:width].copy(), operator.ill_posedness)


def find_kerop_contract_files(kerop_root: str | Path | None = None) -> Path | None:
    """Look for ``filter_contract_v1`` already written by KerOp."""
    for root in candidate_kerop_roots(kerop_root):
        for rel in (
            Path("results") / "filter_contract" / "filter_contract_v1",
            Path("fixtures") / "filter_contract_v1",
        ):
            stem = root / rel
            if stem.with_suffix(".json").is_file() and stem.with_suffix(".npz").is_file():
                return stem
    fixture = default_fixture_stem()
    if fixture.with_suffix(".json").is_file() and fixture.with_suffix(".npz").is_file():
        return fixture
    return None


def maybe_write_kerop_contract(dest_dir: Path, kerop_root: str | Path | None = None) -> Path | None:
    """Ask KerOp to write *its* files. Not a SpecInv-minted schema."""
    try:
        kerop = import_kerop(kerop_root)
    except JointPathError:
        return None
    writer = getattr(getattr(kerop, "filter_contract", None), "write_filter_contract", None)
    if writer is None:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    writer(dest_dir)
    stem = dest_dir / "filter_contract_v1"
    if stem.with_suffix(".json").is_file() and stem.with_suffix(".npz").is_file():
        return stem
    return None

"""Configuration objects for Matryoshka embedding experiments.

This module is intentionally free of any heavy (torch / transformers) imports so
that configs can be loaded and validated without a GPU stack installed.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

METHODS = ("mrl", "ese", "mipic", "gsr")
POOLING_MODES = ("cls", "mean", "last")


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Backbone encoder used as the (self-distilled) student."""

    name_or_path: str = "google-bert/bert-base-uncased"
    hidden_dim: int = 768
    #: How a sequence is reduced to a single vector. The original notebooks used
    #: ``cls`` for MRL/MIPIC and ``mean`` for ESE - see ``configs/``.
    pooling: str = "cls"
    trust_remote_code: bool = False
    #: Optional dtype for ``from_pretrained`` (e.g. "bfloat16"). ``None`` = default.
    torch_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.pooling not in POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {POOLING_MODES}, got {self.pooling!r}"
            )


@dataclass
class DataConfig:
    """Location and shape of the local CSV corpus (``data/`` in this repo)."""

    root: str = "data"
    train_file: str = "train/final_data.csv"
    test_dir: str = "test"
    #: Column of ``train_file`` holding the raw sentence. SimCSE uses it twice.
    text_column: str = "text"
    max_length: int = 256
    num_workers: int = 2
    #: Optional cap on training rows, handy for smoke tests (``None`` = all).
    max_train_samples: int | None = None

    def resolve(self, base_dir: Path) -> DataConfig:
        """Return a copy with ``root`` made absolute relative to ``base_dir``."""
        out = copy.deepcopy(self)
        root = Path(self.root)
        if not root.is_absolute():
            root = (base_dir / root).resolve()
        out.root = str(root)
        return out

    @property
    def train_path(self) -> Path:
        return Path(self.root) / self.train_file

    @property
    def test_path(self) -> Path:
        return Path(self.root) / self.test_dir


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    min_lr: float = 2e-6
    scheduler: str = "cosine_with_min_lr"
    #: ``None`` disables gradient clipping (matches the MRL/ESE notebooks).
    max_grad_norm: float | None = None
    seed: int = 42
    fp16: bool = True
    output_dir: str = "outputs/run"
    #: Save the encoder + tokenizer after the final epoch. Off by default:
    #: runs are scored by the report, and the weights are large artifacts.
    save_model: bool = False
    #: Free the CUDA cache every N steps. 0 disables (the notebooks did it every
    #: step, which is very slow; 0 or a large value is usually better).
    empty_cache_every: int = 0

    def __post_init__(self) -> None:
        # A quoted "2e-5" in a YAML file, or any other string that slips into a
        # numeric field, otherwise travels all the way into the optimiser and
        # fails there with a TypeError that says nothing about the config.
        for name in ("lr", "weight_decay", "warmup_ratio", "min_lr"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"train.{name} must be a number, got {value!r} "
                    f"({type(value).__name__})"
                )
        for name in ("epochs", "batch_size", "seed", "empty_cache_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"train.{name} must be an integer, got {value!r} "
                    f"({type(value).__name__})"
                )
        if self.max_grad_norm is not None and not isinstance(
            self.max_grad_norm, (int, float)
        ):
            raise TypeError(
                f"train.max_grad_norm must be a number or null, "
                f"got {self.max_grad_norm!r}"
            )


@dataclass
class MatryoshkaConfig:
    """Nested dimensions shared by every loss and by evaluation."""

    dims: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 768])
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if not self.dims or self.dims != sorted(set(self.dims)):
            raise ValueError(
                f"matryoshka.dims must be non-empty and strictly increasing, got {self.dims}"
            )
        if any(dim <= 0 for dim in self.dims):
            raise ValueError(f"matryoshka.dims must be positive, got {self.dims}")

    @property
    def descending(self) -> list[int]:
        return sorted(self.dims, reverse=True)

    @property
    def ascending(self) -> list[int]:
        return sorted(self.dims)


@dataclass
class EvalConfig:
    enabled: bool = True
    #: ``test`` or ``validation`` - selects which CSV each task resolves to.
    split: str = "test"
    batch_size: int = 64
    sts_max_length: int = 128
    cls_max_length: int = 512
    logreg_max_iter: int = 200
    logreg_seed: int = 42
    #: Run the full evaluation suite at the end of every epoch.
    every_epoch: bool = True
    cls_tasks: list[str] = field(
        default_factory=lambda: ["banking77", "emotion", "tweet"]
    )
    sts_tasks: list[str] = field(
        default_factory=lambda: [
            "sick",
            "sts12",
            "stsb",
            "sick_r",
            "sts13",
            "sts14",
            "sts15",
            "sts16",
        ]
    )
    pair_tasks: list[str] = field(default_factory=lambda: ["mrpc", "scitail", "wic"])

    def __post_init__(self) -> None:
        if self.split not in ("test", "validation"):
            raise ValueError(
                f"eval.split must be 'test' or 'validation', got {self.split!r}"
            )


@dataclass
class MRLConfig:
    """Plain Matryoshka InfoNCE baseline - no extra knobs beyond the shared ones."""

    #: Weight of the Matryoshka InfoNCE term (kept explicit for ablations).
    w_matryoshka: float = 1.0
    #: Weight of the plain (full-dimension) SimCSE InfoNCE term. The notebook
    #: computed it for logging only, hence the 0.0 default.
    w_task: float = 0.0


@dataclass
class ESEConfig:
    """EPRESSO / ESE baseline: Matryoshka InfoNCE over dims *and* layers."""

    temperature: float = 0.07
    #: Number of intermediate layers sampled per optimisation step.
    n_layers_per_step: int = 1
    use_intermediate_layers: bool = True
    #: Log-based down-weighting of the smaller nested dimensions.
    use_layer_weight: bool = True


@dataclass
class MIPICConfig:
    """MIPIC: SIA (cross-dimensional) + PIC (depth-wise) alignment on top of MRL.

    Field names follow the paper (``docs/MIPIC.pdf``); see Table 7 and
    Appendix A.5 for the published per-backbone values.
    """

    #: Eq 18: ``L = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)``.
    #: Table 7: 0.4 for TinyBERT-6L and BERT-base, 0.5 for Qwen3-0.6B and BGE-M3.
    alpha: float = 0.4
    #: ``L`` - hidden-state indices SIA is applied at (index 0 = embeddings).
    layers: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 9, 10, 12])
    #: ``C`` - ordered ``(dim, layer)`` checkpoints chained by PIC.
    checkpoints: list[list[int]] = field(
        default_factory=lambda: [
            [16, 2], [32, 4], [64, 6], [128, 8], [256, 9], [512, 10], [768, 12]
        ]
    )
    #: Appendix A.5: ``k_i = max(k_min, ceil(gamma_i * m))``, one gamma per
    #: truncated prefix in ascending order of dimension.
    gamma_schedule: list[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    k_min: int = 8
    #: SIA softmax temperature. ``None`` reuses the shared ``tau``
    #: (``matryoshka.temperature``), which is what the paper's notation implies.
    attention_temperature: float | None = None
    #: Eq 13 sums the two SIA terms with equal weight; expose them for ablations.
    w_att: float = 1.0
    w_cka: float = 1.0
    w_pic: float = 1.0
    #: ``"sum"`` follows Eq 13/14/17. ``"mean"`` averages over dims/layers/steps.
    aggregate: str = "sum"
    #: Hidden width of the PIC projector ``phi_i``. ``None`` = ``max(d_i, d_j) // 2``.
    pic_hidden_dim: int | None = None
    #: Stop gradients on the deeper checkpoint so information flows downward.
    pic_detach_target: bool = True
    #: Learning-rate multiplier for the alignment module's own parameters.
    module_lr_scale: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"mipic.alpha must be in [0, 1], got {self.alpha}")
        if self.aggregate not in ("sum", "mean"):
            raise ValueError(
                f"mipic.aggregate must be 'sum' or 'mean', got {self.aggregate!r}"
            )
        bad = [c for c in self.checkpoints if len(c) != 2]
        if bad:
            raise ValueError(f"each mipic.checkpoint must be [dim, layer], got {bad}")
        if len(self.checkpoints) < 2:
            raise ValueError("mipic.checkpoints needs at least two entries to form a chain")

    def checkpoint_pairs(self) -> list[tuple[int, int]]:
        """``C`` as ``(dim, layer)`` tuples."""
        return [(int(dim), int(layer)) for dim, layer in self.checkpoints]

    @property
    def w_align(self) -> float:
        """The ``(1 - alpha)`` weight applied to ``L_SIA + L_PIC`` (Eq 18)."""
        return 1.0 - self.alpha

    @property
    def w_matryoshka(self) -> float:
        """The ``alpha`` weight applied to ``L_MRL`` (Eq 18)."""
        return self.alpha


@dataclass
class GSRConfig:
    """Geometric Successive Refinement on corpus-level spectral shells."""

    weight: float = 1.0
    warmup_epochs: int = 1
    refresh_every_epochs: int = 1
    teacher_batch_size: int = 64
    #: ``None`` reuses every Matryoshka endpoint.
    geometry_dims: list[int] | None = None
    #: Only merges numerical ties; near-gap sensitivity remains a diagnostic.
    eigengap_tolerance: float = 1e-6
    merge_tied_shells: bool = True
    eps: float = 1e-8
    cache_dtype: str = "float32"
    #: Dump the per-epoch teacher embedding cache to ``teacher_epoch{N}.pt``.
    #: Off by default: the JSON diagnostics already cover the analysis.
    save_teacher_tensors: bool = False
    diagnostics_every_steps: int = 100
    diagnostic_samples: int = 512
    diagnostic_pairs: int = 8192
    fail_on_nonfinite: bool = True

    def __post_init__(self) -> None:
        numeric_positive = {
            "weight": self.weight,
            "eigengap_tolerance": self.eigengap_tolerance,
            "eps": self.eps,
        }
        for name, value in numeric_positive.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"gsr.{name} must be a number, got {value!r}")
        if self.weight < 0:
            raise ValueError(f"gsr.weight must be non-negative, got {self.weight}")
        if self.eigengap_tolerance < 0:
            raise ValueError(
                "gsr.eigengap_tolerance must be non-negative, "
                f"got {self.eigengap_tolerance}"
            )
        if self.eps <= 0:
            raise ValueError(f"gsr.eps must be positive, got {self.eps}")
        integer_fields = {
            "warmup_epochs": self.warmup_epochs,
            "refresh_every_epochs": self.refresh_every_epochs,
            "teacher_batch_size": self.teacher_batch_size,
            "diagnostics_every_steps": self.diagnostics_every_steps,
            "diagnostic_samples": self.diagnostic_samples,
            "diagnostic_pairs": self.diagnostic_pairs,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"gsr.{name} must be an integer, got {value!r}")
        if self.warmup_epochs < 0:
            raise ValueError("gsr.warmup_epochs must be non-negative")
        for name in (
            "refresh_every_epochs",
            "teacher_batch_size",
            "diagnostics_every_steps",
            "diagnostic_pairs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"gsr.{name} must be positive")
        if self.diagnostic_samples < 2:
            raise ValueError("gsr.diagnostic_samples must be at least two")
        if self.cache_dtype != "float32":
            raise ValueError("gsr.cache_dtype currently supports only 'float32'")
        if self.geometry_dims is not None:
            dims = self.geometry_dims
            if not dims or dims != sorted(set(dims)) or any(dim <= 0 for dim in dims):
                raise ValueError(
                    "gsr.geometry_dims must be null or strictly increasing positive "
                    f"dimensions, got {dims}"
                )


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str = "run"
    method: str = "mrl"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    matryoshka: MatryoshkaConfig = field(default_factory=MatryoshkaConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    mrl: MRLConfig = field(default_factory=MRLConfig)
    ese: ESEConfig = field(default_factory=ESEConfig)
    mipic: MIPICConfig = field(default_factory=MIPICConfig)
    gsr: GSRConfig = field(default_factory=GSRConfig)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        bad = [d for d in self.matryoshka.dims if d > self.model.hidden_dim]
        if bad:
            raise ValueError(
                f"matryoshka.dims {bad} exceed model.hidden_dim={self.model.hidden_dim}"
            )
        if self.method == "mipic":
            bad_dims = [
                dim for dim, _ in self.mipic.checkpoint_pairs()
                if dim > self.model.hidden_dim
            ]
            if bad_dims:
                raise ValueError(
                    f"mipic.checkpoints reference dims {bad_dims} > hidden_dim="
                    f"{self.model.hidden_dim}"
                )
            truncated = [d for d in self.matryoshka.dims if d < self.model.hidden_dim]
            if len(self.mipic.gamma_schedule) != len(truncated):
                raise ValueError(
                    f"mipic.gamma_schedule has {len(self.mipic.gamma_schedule)} entries "
                    f"but there are {len(truncated)} truncated prefixes {sorted(truncated)}"
                )
        if self.method == "gsr":
            if self.gsr.weight > 0 and self.gsr.warmup_epochs >= self.train.epochs:
                raise ValueError(
                    "gsr.warmup_epochs must be smaller than train.epochs when "
                    "gsr.weight is positive"
                )
            geometry_dims = self.gsr.geometry_dims or self.matryoshka.dims
            missing = [dim for dim in geometry_dims if dim not in self.matryoshka.dims]
            if missing:
                raise ValueError(
                    f"gsr.geometry_dims {missing} are not Matryoshka endpoints "
                    f"{self.matryoshka.dims}"
                )
            if geometry_dims[-1] != self.model.hidden_dim:
                raise ValueError(
                    "the largest GSR geometry dimension must equal "
                    f"model.hidden_dim={self.model.hidden_dim}"
                )

    # -- (de)serialisation -------------------------------------------------- #
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        return _build(cls, raw)

    @classmethod
    def load(
        cls, path: str | Path, overrides: Sequence[str] | None = None
    ) -> ExperimentConfig:
        """Load a YAML config, resolving ``_base_`` inheritance and CLI overrides."""
        path = Path(path).resolve()
        raw = _load_yaml_with_base(path)
        for override in overrides or []:
            _apply_override(raw, override)
        cfg = cls.from_dict(raw)
        # Make data paths relative to the repo root, not the caller's cwd.
        cfg.data = cfg.data.resolve(_repo_root(path))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _repo_root(config_path: Path) -> Path:
    """``configs/<method>/<name>.yaml`` -> repo root."""
    for parent in config_path.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return config_path.parent


def _load_yaml_with_base(path: Path, _seen: set | None = None) -> dict[str, Any]:
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ reference at {path}")
    _seen.add(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("_base_", None)
    if base is None:
        return raw
    base_path = (path.parent / base).resolve()
    merged = _load_yaml_with_base(base_path, _seen)
    return _deep_merge(merged, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


#: Scientific notation that PyYAML refuses to read as a float (see below).
_SCIENTIFIC = re.compile(r"^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$")


def _parse_override_value(text: str) -> Any:
    """Parse one ``--set`` value.

    ``yaml.safe_load`` implements YAML 1.1, whose float resolver demands *both*
    a decimal point and a signed exponent. So ``2e-5`` - the obvious thing to
    type for a learning rate - comes back as the **string** ``"2e-5"`` and
    silently poisons a numeric field, surfacing much later as a confusing
    ``TypeError`` inside the optimiser. Close exactly that gap and leave every
    other YAML behaviour alone.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str) and _SCIENTIFIC.match(value.strip()):
        return float(value)
    return value


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a ``dotted.key=value`` override in place (value parsed as YAML)."""
    if "=" not in override:
        raise ValueError(f"override must look like key.path=value, got {override!r}")
    key, _, value = override.partition("=")
    node = raw
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-mapping while setting {key!r}")
    node[parts[-1]] = _parse_override_value(value)


def _build(cls: type, raw: Any) -> Any:
    if not is_dataclass(cls):
        return raw
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(
            f"expected a mapping for {cls.__name__}, got {type(raw).__name__}"
        )

    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs = {}
    for name, value in raw.items():
        field_type = known[name].type
        nested = _nested_dataclass(field_type)
        kwargs[name] = _build(nested, value) if nested else value
    return cls(**kwargs)


def _nested_dataclass(field_type: Any) -> type | None:
    """Resolve a (possibly string) annotation to a dataclass, else ``None``."""
    if isinstance(field_type, str):
        return (
            globals().get(field_type)
            if is_dataclass(globals().get(field_type))
            else None
        )
    return field_type if is_dataclass(field_type) else None


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj

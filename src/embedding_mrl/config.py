"""Configuration objects for Matryoshka embedding experiments.

This module is intentionally free of any heavy (torch / transformers) imports so
that configs can be loaded and validated without a GPU stack installed.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

METHODS = ("mrl", "ese", "mipic")
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
    #: Save the encoder + tokenizer after the final epoch.
    save_model: bool = True
    #: Free the CUDA cache every N steps. 0 disables (the notebooks did it every
    #: step, which is very slow; 0 or a large value is usually better).
    empty_cache_every: int = 0


@dataclass
class MatryoshkaConfig:
    """Nested dimensions shared by every loss and by evaluation."""

    dims: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 768])
    temperature: float = 0.07

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
    """MIPIC: horizontal (SIA) + vertical (PIC) alignment on top of MRL."""

    #: Hidden-state indices to align. Index 0 is the embedding layer.
    align_layers: list[int] = field(default_factory=lambda: [8, 10])
    #: Each entry is a flat ``(layer, dim, layer, dim, ...)`` chain.
    pipeline_pairs: list[list[int]] = field(
        default_factory=lambda: [[3, 16, 7, 128, 11, 768]]
    )
    alpha: float = 0.4  # horizontal attention (SIA) weight
    beta: float = 0.4  # submatrix CKA weight
    gamma: float = 0.2  # pipeline InfoNCE (PIC) weight
    base_k: int = 64
    d_att: int = 64
    attention_temperature: float = 1.0
    pipeline_temperature: float = 0.07
    #: Explicit ``dim -> k`` map for top-k token selection. ``None`` uses the
    #: default proportional rule ``max(8, int(dim / d_full * base_k))``.
    k_map: dict[int, int] | None = None
    #: The published notebooks zeroed the attention-KL term before returning it,
    #: so SIA contributed nothing to the gradient. ``False`` reproduces that;
    #: ``True`` enables the KL term as described in the paper.
    use_attention_kl: bool = False
    #: Learning-rate multiplier for the alignment module's own parameters.
    module_lr_scale: float = 2.0
    #: Final mixture: ``w_align * L_align + w_matryoshka * L_MRL``.
    w_align: float = 0.4
    w_matryoshka: float = 0.6

    def parsed_pipeline_pairs(self) -> list[tuple[int, int, int, int]]:
        """Expand each chain into consecutive ``(layer_i, dim_i, layer_j, dim_j)``."""
        out: list[tuple[int, int, int, int]] = []
        for chain in self.pipeline_pairs:
            if len(chain) % 2 != 0:
                raise ValueError(f"pipeline chain must have even length, got {chain}")
            stages = [(chain[i], chain[i + 1]) for i in range(0, len(chain), 2)]
            for (layer_i, dim_i), (layer_j, dim_j) in zip(stages, stages[1:]):
                out.append((layer_i, dim_i, layer_j, dim_j))
        if not out:
            raise ValueError("pipeline_pairs must define at least one transition")
        return out


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
                d
                for chain in self.mipic.pipeline_pairs
                for d in chain[1::2]
                if d > self.model.hidden_dim
            ]
            if bad_dims:
                raise ValueError(
                    f"mipic.pipeline_pairs reference dims {bad_dims} > hidden_dim="
                    f"{self.model.hidden_dim}"
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
    node[parts[-1]] = yaml.safe_load(value)


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

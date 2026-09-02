"""Corpus-level spectral teacher construction for GSR."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .losses.gsr import (
    Shell,
    build_shell_slices,
    full_normalize,
    merge_tied_shells,
    shell_key,
)
from .pooling import pool
from .utils import autocast


@dataclass
class SpectralTeacherCache:
    """Frozen teacher quantities indexed by stable training-corpus row id."""

    mean: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    scores: torch.Tensor
    shells: list[Shell]
    merged_boundaries: list[int]
    c_teacher: float
    refresh_index: int
    source_epoch: int
    diagnostics: dict[str, Any]

    def lookup(self, sample_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        ids = sample_ids.detach().cpu().long()
        if ids.ndim != 1:
            raise ValueError(f"sample_ids must be one-dimensional, got {ids.shape}")
        if ids.numel() == 0:
            raise ValueError("cannot look up an empty teacher batch")
        if int(ids.min()) < 0 or int(ids.max()) >= self.scores.size(0):
            raise IndexError(
                f"sample ids [{int(ids.min())}, {int(ids.max())}] exceed teacher "
                f"cache with {self.scores.size(0)} rows"
            )
        return self.scores.index_select(0, ids).to(device, non_blocking=True)

    def tensor_payload(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "scores": self.scores,
            "shells": self.shells,
            "merged_boundaries": self.merged_boundaries,
            "c_teacher": self.c_teacher,
            "refresh_index": self.refresh_index,
            "source_epoch": self.source_epoch,
        }


def exact_mean_fourth_distance(points: torch.Tensor) -> float:
    """Mean squared squared-distance over ordered distinct corpus pairs.

    This computes ``mean_{i != j} ||x_i - x_j||^4`` from first and second
    feature moments, without constructing an ``N x N`` Gram matrix.
    """
    if points.ndim != 2 or points.size(0) < 2:
        raise ValueError("points must be [N, D] with N >= 2")
    x = points.detach().cpu().double()
    n = x.size(0)
    denominator = float(n * (n - 1))

    norm_sq = x.square().sum(dim=1)
    sum_norm_sq = norm_sq.sum()
    sum_norm_fourth = norm_sq.square().sum()
    sum_vector = x.sum(dim=0)
    feature_gram = x.T @ x

    # Ordered off-diagonal sums for the expansion
    # (a_i + a_j - 2 <x_i,x_j>)^2.
    two_norm_fourth = 2.0 * (n - 1) * sum_norm_fourth
    cross_norms = 2.0 * (sum_norm_sq.square() - sum_norm_fourth)
    dot_squares = 4.0 * (feature_gram.square().sum() - sum_norm_fourth)
    weighted_dot = (
        norm_sq * (x @ sum_vector - norm_sq)
    ).sum()
    norm_dot_terms = -8.0 * weighted_dot

    value = (
        two_norm_fourth + cross_norms + dot_squares + norm_dot_terms
    ) / denominator
    return float(value.clamp_min(0.0))


def _effective_rank(eigenvalues: torch.Tensor, eps: float) -> float:
    values = eigenvalues.double().clamp_min(0)
    total = values.sum()
    if total <= eps:
        return 0.0
    probabilities = values[values > eps] / total
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def build_spectral_teacher_cache(
    embeddings: torch.Tensor,
    dims: Sequence[int],
    eigengap_tolerance: float = 1e-6,
    eps: float = 1e-8,
    refresh_index: int = 0,
    source_epoch: int = 0,
    merge_ties: bool = True,
) -> SpectralTeacherCache:
    """Build a numerically checked global spectral teacher from raw embeddings."""
    started = time.time()
    if embeddings.ndim != 2 or embeddings.size(0) < 2:
        raise ValueError("teacher embeddings must be [N, D] with N >= 2")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("teacher embeddings contain non-finite values")

    normalization_started = time.time()
    raw = embeddings.detach().cpu().float()
    raw_norms = raw.norm(dim=1)
    zero_rows = int((raw_norms <= eps).sum())
    if zero_rows:
        raise FloatingPointError(
            f"teacher produced {zero_rows} near-zero embeddings; geometry is undefined"
        )

    q = full_normalize(raw, eps=eps).cpu()
    row_norms = q.norm(dim=1)
    max_norm_error = float((row_norms - 1.0).abs().max())
    if max_norm_error > 1e-4:
        raise FloatingPointError(
            f"teacher normalization error {max_norm_error:.3e} exceeds tolerance"
        )
    normalization_seconds = time.time() - normalization_started

    n, hidden_dim = q.shape
    if list(dims) != sorted(set(int(dim) for dim in dims)):
        raise ValueError(f"dims must be strictly increasing, got {list(dims)}")
    if int(dims[-1]) != hidden_dim:
        raise ValueError(
            f"largest dim must equal teacher width {hidden_dim}, got {dims[-1]}"
        )

    covariance_started = time.time()
    mean = q.mean(dim=0)
    centered = q - mean
    covariance_raw = centered.T @ centered / n
    symmetry_residual = float(
        (covariance_raw - covariance_raw.T).norm()
        / covariance_raw.norm().clamp_min(eps)
    )
    covariance = ((covariance_raw + covariance_raw.T) * 0.5).double()
    covariance_seconds = time.time() - covariance_started
    if not torch.isfinite(covariance).all():
        raise FloatingPointError("teacher covariance contains non-finite values")
    if symmetry_residual > 1e-6:
        raise FloatingPointError(
            f"teacher covariance symmetry residual {symmetry_residual:.3e} is too large"
        )

    eigendecomposition_started = time.time()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    eigendecomposition_seconds = time.time() - eigendecomposition_started

    spectral_scale = float(eigenvalues.abs().max().clamp_min(eps))
    negative = eigenvalues[eigenvalues < 0]
    negative_mass = float((-negative).sum()) if negative.numel() else 0.0
    if float(eigenvalues.min()) < -1e-6 * spectral_scale:
        raise FloatingPointError(
            f"teacher covariance has a materially negative eigenvalue "
            f"{float(eigenvalues.min()):.3e}"
        )

    residual = covariance @ eigenvectors - eigenvectors * eigenvalues.unsqueeze(0)
    eigensolver_residual = float(
        residual.norm() / covariance.norm().clamp_min(eps)
    )
    identity = torch.eye(hidden_dim, dtype=eigenvectors.dtype)
    orthogonality_residual = float(
        (eigenvectors.T @ eigenvectors - identity).norm() / math.sqrt(hidden_dim)
    )
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(
        eigenvectors
    ).all():
        raise FloatingPointError(
            "teacher eigendecomposition contains non-finite values"
        )
    if eigensolver_residual > 1e-5:
        raise FloatingPointError(
            f"teacher eigensolver residual {eigensolver_residual:.3e} is too large"
        )
    if orthogonality_residual > 1e-5:
        raise FloatingPointError(
            "teacher eigenvector orthogonality residual "
            f"{orthogonality_residual:.3e} is too large"
        )

    projection_started = time.time()
    eigenvectors_f32 = eigenvectors.float()
    eigenvalues_f32 = eigenvalues.float()
    scores = centered @ eigenvectors_f32
    projection_seconds = time.time() - projection_started
    if not torch.isfinite(scores).all():
        raise FloatingPointError(
            "teacher spectral score cache contains non-finite values"
        )

    scale_started = time.time()
    c_teacher = exact_mean_fourth_distance(q)
    if not math.isfinite(c_teacher) or c_teacher <= eps:
        raise FloatingPointError(
            f"teacher geometry is degenerate: c_teacher={c_teacher}"
        )
    scale_seconds = time.time() - scale_started

    if merge_ties:
        shells, merged = merge_tied_shells(
            dims, eigenvalues_f32, tolerance=eigengap_tolerance, eps=eps
        )
    else:
        shells = build_shell_slices(dims, full_dim=hidden_dim)
        merged = []
    nonnegative = eigenvalues_f32.clamp_min(0)
    variance_total = float(nonnegative.sum().clamp_min(eps))
    energy_total = float(nonnegative.square().sum().clamp_min(eps))

    explained_variance: dict[str, float] = {}
    explained_energy: dict[str, float] = {}
    boundary_gaps: dict[str, dict[str, float]] = {}
    for dim in dims:
        dim = int(dim)
        explained_variance[f"dim_{dim}"] = float(
            nonnegative[:dim].sum() / variance_total
        )
        explained_energy[f"dim_{dim}"] = float(
            nonnegative[:dim].square().sum() / energy_total
        )
        if dim < hidden_dim:
            left = float(eigenvalues_f32[dim - 1])
            right = float(eigenvalues_f32[dim])
            boundary_gaps[f"dim_{dim}"] = {
                "absolute": left - right,
                "relative": (left - right) / max(abs(left), eps),
            }

    shell_energy = {
        shell_key(shell): float(
            nonnegative[shell[0] : shell[1]].square().sum() / energy_total
        )
        for shell in shells
    }
    stable_rank = float(nonnegative.sum() / nonnegative.max().clamp_min(eps))

    diagnostics: dict[str, Any] = {
        "geometry_type": "linear_pca",
        "refresh_index": refresh_index,
        "source_epoch": source_epoch,
        "corpus_size": n,
        "hidden_dim": hidden_dim,
        "cache_mib": scores.numel() * scores.element_size() / (1024**2),
        "raw_norm": {
            "min": float(raw_norms.min()),
            "mean": float(raw_norms.mean()),
            "max": float(raw_norms.max()),
        },
        "normalized_norm": {
            "min": float(row_norms.min()),
            "mean": float(row_norms.mean()),
            "max": float(row_norms.max()),
            "max_error": max_norm_error,
        },
        "mean_norm": float(mean.norm()),
        "covariance_trace": float(covariance.trace()),
        "covariance_frobenius": float(covariance.norm()),
        "covariance_symmetry_residual": symmetry_residual,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "negative_eigenvalue_count": int((eigenvalues < 0).sum()),
        "negative_eigenvalue_mass": negative_mass,
        "eigensolver_residual": eigensolver_residual,
        "orthogonality_residual": orthogonality_residual,
        "effective_rank": _effective_rank(eigenvalues, eps),
        "stable_rank": stable_rank,
        "explained_variance": explained_variance,
        "explained_energy": explained_energy,
        "boundary_gaps": boundary_gaps,
        "requested_boundaries": [int(dim) for dim in dims],
        "shells": [list(shell) for shell in shells],
        "merged_boundaries": merged,
        "shell_energy": shell_energy,
        "merge_reasons": {
            f"dim_{boundary}": "relative eigengap is within numerical tie tolerance"
            for boundary in merged
        },
        "c_teacher": c_teacher,
        "timing": {
            "normalization_seconds": normalization_seconds,
            "covariance_seconds": covariance_seconds,
            "eigendecomposition_seconds": eigendecomposition_seconds,
            "projection_seconds": projection_seconds,
            "scale_seconds": scale_seconds,
        },
        "build_seconds": time.time() - started,
        "eigenvalues": [float(value) for value in eigenvalues_f32],
    }

    return SpectralTeacherCache(
        mean=mean,
        eigenvalues=eigenvalues_f32,
        eigenvectors=eigenvectors_f32,
        scores=scores.float(),
        shells=shells,
        merged_boundaries=merged,
        c_teacher=c_teacher,
        refresh_index=refresh_index,
        source_epoch=source_epoch,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def build_semantic_kernel_teacher_cache(
    embeddings: torch.Tensor,
    dims: Sequence[int],
    *,
    temperature: float = 0.05,
    ridge: float = 1e-6,
    chunk_size: int = 2048,
    landmark_seed: int = 42,
    compute_device: torch.device | None = None,
    eigengap_tolerance: float = 1e-6,
    eps: float = 1e-8,
    refresh_index: int = 0,
    source_epoch: int = 0,
    merge_ties: bool = True,
) -> SpectralTeacherCache:
    """Build the nonlinear GSR teacher through a global Nyström feature map.

    The exponential cosine kernel

    ``k(x, y) = exp((<x, y> - 1) / temperature)``

    is positive semidefinite and has unit diagonal.  We use at most ``D``
    corpus landmarks to obtain a ``D``-wide Nyström map, pad only when the
    corpus itself has fewer than ``D`` rows, and normalize every mapped row.
    The ordinary spectral builder then merely rotates this unit-sphere feature
    representation.  Consequently, its full-dimensional pair geometry is
    exactly attainable by a unit-normalized ``D``-dimensional student.
    """
    started = time.time()
    if embeddings.ndim != 2 or embeddings.size(0) < 2:
        raise ValueError("teacher embeddings must be [N, D] with N >= 2")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("teacher embeddings contain non-finite values")
    if temperature <= 0:
        raise ValueError("kernel temperature must be positive")
    if ridge <= 0:
        raise ValueError("kernel ridge must be positive")
    if chunk_size <= 0:
        raise ValueError("kernel chunk size must be positive")

    raw = embeddings.detach().cpu().float()
    n, hidden_dim = raw.shape
    if int(dims[-1]) != hidden_dim:
        raise ValueError(
            f"largest dim must equal teacher width {hidden_dim}, got {dims[-1]}"
        )
    raw_norms = raw.norm(dim=1)
    zero_rows = int((raw_norms <= eps).sum())
    if zero_rows:
        raise FloatingPointError(
            f"teacher produced {zero_rows} near-zero embeddings; geometry is undefined"
        )

    device = compute_device or torch.device("cpu")
    q = full_normalize(raw, eps=eps).to(device)
    landmark_count = min(n, hidden_dim)
    generator = torch.Generator().manual_seed(landmark_seed)
    landmark_ids = torch.randperm(n, generator=generator)[:landmark_count]
    landmark_ids_device = landmark_ids.to(device)
    landmarks = q.index_select(0, landmark_ids_device)

    kernel_started = time.time()
    cross_kernel = torch.empty(
        n, landmark_count, dtype=torch.float32, device=device
    )
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        similarities = q[start:end] @ landmarks.T
        cross_kernel[start:end] = torch.exp(
            (similarities.clamp(-1.0, 1.0) - 1.0) / temperature
        )
    # Recompute W in float64. Extracting it from the float32 cross-kernel can
    # introduce negative eigenvalues larger than the small ridge when the
    # landmark set is ill-conditioned.
    landmarks_cpu = landmarks.detach().cpu().double()
    landmark_kernel_cpu = torch.exp(
        (
            (landmarks_cpu @ landmarks_cpu.T).clamp(-1.0, 1.0)
            - 1.0
        )
        / temperature
    )
    landmark_kernel_cpu = 0.5 * (
        landmark_kernel_cpu + landmark_kernel_cpu.T
    )
    landmark_kernel = landmark_kernel_cpu.float().to(device)
    kernel_seconds = time.time() - kernel_started

    factor_started = time.time()
    regularized = landmark_kernel_cpu + ridge * torch.eye(
        landmark_count, dtype=torch.float64
    )
    kernel_eigenvalues, kernel_eigenvectors = torch.linalg.eigh(regularized)
    minimum_regularized_eigenvalue = float(kernel_eigenvalues.min())
    if minimum_regularized_eigenvalue <= 0:
        raise FloatingPointError(
            "regularized landmark kernel is not positive definite: "
            f"minimum eigenvalue={minimum_regularized_eigenvalue:.3e}"
        )
    inverse_sqrt = (
        kernel_eigenvectors
        @ torch.diag(kernel_eigenvalues.rsqrt())
        @ kernel_eigenvectors.T
    ).float().to(device)
    nystrom = cross_kernel @ inverse_sqrt
    factor_seconds = time.time() - factor_started

    feature_started = time.time()
    feature_norms_before = nystrom.norm(dim=1)
    near_zero_features = int((feature_norms_before <= eps).sum())
    if near_zero_features:
        raise FloatingPointError(
            "semantic kernel produced "
            f"{near_zero_features} near-zero Nyström rows; increase "
            "gsr.kernel_temperature"
        )
    panel_count = min(n, 512)
    panel_generator = torch.Generator().manual_seed(landmark_seed + 1)
    panel_ids = torch.randperm(n, generator=panel_generator)[:panel_count].to(device)
    panel_q = q.index_select(0, panel_ids)
    exact_panel_kernel = torch.exp(
        ((panel_q @ panel_q.T).clamp(-1.0, 1.0) - 1.0) / temperature
    )
    raw_panel_features = nystrom.index_select(0, panel_ids)
    raw_panel_gram = raw_panel_features @ raw_panel_features.T
    raw_panel_relative_error = float(
        (raw_panel_gram - exact_panel_kernel).norm()
        / exact_panel_kernel.norm().clamp_min(eps)
    )
    nystrom = full_normalize(nystrom, eps=eps)
    spherical_panel_features = nystrom.index_select(0, panel_ids)
    spherical_panel_gram = spherical_panel_features @ spherical_panel_features.T
    spherical_panel_relative_error = float(
        (spherical_panel_gram - exact_panel_kernel).norm()
        / exact_panel_kernel.norm().clamp_min(eps)
    )
    if landmark_count < hidden_dim:
        nystrom = torch.nn.functional.pad(
            nystrom, (0, hidden_dim - landmark_count)
        )
    kernel_features = nystrom.detach().cpu().float()
    feature_seconds = time.time() - feature_started

    landmark_features = nystrom.index_select(0, landmark_ids_device)[
        :, :landmark_count
    ]
    reconstructed_landmark_kernel = landmark_features @ landmark_features.T
    landmark_reconstruction_error = float(
        (reconstructed_landmark_kernel - landmark_kernel).norm()
        / landmark_kernel.norm().clamp_min(eps)
    )

    cache = build_spectral_teacher_cache(
        kernel_features,
        dims,
        eigengap_tolerance=eigengap_tolerance,
        eps=eps,
        refresh_index=refresh_index,
        source_epoch=source_epoch,
        merge_ties=merge_ties,
    )
    cache.diagnostics["geometry_type"] = "semantic_exponential_kernel"
    cache.diagnostics["kernel"] = {
        "name": "exponential_cosine",
        "temperature": temperature,
        "ridge": ridge,
        "landmark_seed": landmark_seed,
        "landmark_count": landmark_count,
        "landmark_ids": landmark_ids.tolist(),
        "chunk_size": chunk_size,
        "minimum_regularized_eigenvalue": minimum_regularized_eigenvalue,
        "maximum_regularized_eigenvalue": float(kernel_eigenvalues.max()),
        "regularized_condition_number": float(
            kernel_eigenvalues.max() / kernel_eigenvalues.min()
        ),
        "cross_kernel": {
            "min": float(cross_kernel.min()),
            "mean": float(cross_kernel.mean()),
            "max": float(cross_kernel.max()),
            "near_zero_fraction": float((cross_kernel <= eps).float().mean()),
        },
        "feature_norm_before_normalization": {
            "min": float(feature_norms_before.min()),
            "mean": float(feature_norms_before.mean()),
            "max": float(feature_norms_before.max()),
        },
        "landmark_reconstruction_relative_error": landmark_reconstruction_error,
        "approximation_panel": {
            "sample_count": panel_count,
            "raw_nystrom_relative_frobenius_error": raw_panel_relative_error,
            "spherical_nystrom_relative_frobenius_error": (
                spherical_panel_relative_error
            ),
        },
        "timing": {
            "kernel_seconds": kernel_seconds,
            "factor_seconds": factor_seconds,
            "feature_seconds": feature_seconds,
        },
    }
    cache.diagnostics["input_raw_norm"] = {
        "min": float(raw_norms.min()),
        "mean": float(raw_norms.mean()),
        "max": float(raw_norms.max()),
    }
    cache.diagnostics["build_seconds"] = time.time() - started
    return cache


@torch.no_grad()
def encode_teacher_corpus(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    pooling: str,
    hidden_dim: int,
    device: torch.device,
    fp16: bool,
    expected_sample_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode every corpus row exactly once into a CPU tensor indexed by row id.

    The stable-id checks are deliberately strict: a silently shuffled or partial
    cache would train against the wrong teacher target and can still produce a
    perfectly finite loss, making it unusually difficult to diagnose later.
    """
    started = time.time()
    corpus_size = len(loader.dataset)
    if corpus_size < 2:
        raise ValueError("GSR teacher construction needs at least two corpus rows")

    if expected_sample_ids is None:
        expected_ids = torch.arange(corpus_size, dtype=torch.long)
        global_to_local = None
    else:
        expected_ids = expected_sample_ids.detach().cpu().long()
        if expected_ids.ndim != 1 or expected_ids.numel() != corpus_size:
            raise ValueError(
                "expected_sample_ids must contain one id per loader row, got "
                f"{tuple(expected_ids.shape)} for {corpus_size} rows"
            )
        if expected_ids.unique().numel() != corpus_size:
            raise ValueError("expected_sample_ids must be unique")
        if int(expected_ids.min()) < 0:
            raise ValueError("expected_sample_ids must be non-negative")
        global_to_local = {
            int(global_id): local_id
            for local_id, global_id in enumerate(expected_ids.tolist())
        }

    embeddings = torch.empty(corpus_size, hidden_dim, dtype=torch.float32)
    seen = torch.zeros(corpus_size, dtype=torch.bool)
    duplicate_ids: list[int] = []
    was_training = model.training
    model.eval()

    try:
        for raw_batch in loader:
            sample_ids = raw_batch["sample_ids"].long()
            if sample_ids.ndim != 1:
                raise ValueError(
                    "teacher sample_ids must be one-dimensional, got "
                    f"{sample_ids.shape}"
                )
            if sample_ids.numel() == 0:
                continue
            if global_to_local is None:
                if int(sample_ids.min()) < 0 or int(sample_ids.max()) >= corpus_size:
                    raise IndexError(
                        f"teacher sample ids [{int(sample_ids.min())}, "
                        f"{int(sample_ids.max())}] exceed corpus size {corpus_size}"
                    )
                local_ids = sample_ids
            else:
                unexpected = [
                    int(global_id)
                    for global_id in sample_ids.tolist()
                    if int(global_id) not in global_to_local
                ]
                if unexpected:
                    raise IndexError(
                        "teacher loader emitted unexpected sample ids "
                        f"{unexpected[:20]}"
                    )
                local_ids = torch.tensor(
                    [global_to_local[int(global_id)] for global_id in sample_ids],
                    dtype=torch.long,
                )
            duplicate_mask = seen.index_select(0, local_ids)
            if duplicate_mask.any():
                duplicate_ids.extend(sample_ids[duplicate_mask].tolist())

            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in raw_batch.items()
                if key != "sample_ids" and torch.is_tensor(value)
            }
            with autocast(fp16, device):
                outputs = model(
                    **inputs,
                    output_hidden_states=False,
                    return_dict=True,
                )
                pooled = pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                    pooling,
                )
            if pooled.shape != (sample_ids.numel(), hidden_dim):
                raise ValueError(
                    "teacher encoder returned shape "
                    f"{tuple(pooled.shape)}, expected "
                    f"{(sample_ids.numel(), hidden_dim)}"
                )
            embeddings.index_copy_(0, local_ids, pooled.detach().cpu().float())
            seen.index_fill_(0, local_ids, True)
    finally:
        model.train(was_training)

    missing_local_ids = (~seen).nonzero(as_tuple=False).flatten()
    missing_ids = expected_ids.index_select(0, missing_local_ids).tolist()
    if duplicate_ids or missing_ids:
        raise RuntimeError(
            "teacher corpus id coverage failed: "
            f"duplicates={duplicate_ids[:20]}, missing={missing_ids[:20]}"
        )
    if not torch.isfinite(embeddings).all():
        bad_rows = (~torch.isfinite(embeddings).all(dim=1)).nonzero().flatten()
        raise FloatingPointError(
            "teacher corpus contains non-finite embeddings at rows "
            f"{bad_rows[:20].tolist()}"
        )

    diagnostics = {
        "corpus_size": corpus_size,
        "seen_count": int(seen.sum()),
        "duplicate_count": len(duplicate_ids),
        "missing_count": len(missing_ids),
        "encode_seconds": time.time() - started,
    }
    return embeddings, diagnostics

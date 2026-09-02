"""Corpus spectral teacher mathematics and cache integrity."""

import pytest
import torch

from embedding_mrl.gsr_teacher import (
    build_semantic_kernel_teacher_cache,
    build_spectral_teacher_cache,
    exact_mean_fourth_distance,
)
from embedding_mrl.losses import full_normalize, gsr_shell_loss


def test_exact_fourth_distance_matches_explicit_pairs():
    points = full_normalize(torch.randn(7, 5))
    explicit = torch.pdist(points.double()).square().square().mean()
    assert exact_mean_fourth_distance(points) == pytest.approx(
        explicit.item(), rel=1e-10, abs=1e-10
    )


def test_spectral_cache_reconstructs_centered_covariance():
    raw = torch.randn(12, 6)
    cache = build_spectral_teacher_cache(raw, dims=[2, 4, 6])
    q = full_normalize(raw)
    centered = q - q.mean(0)
    expected = centered.T @ centered / len(raw)
    reconstructed = (
        cache.eigenvectors
        @ torch.diag(cache.eigenvalues)
        @ cache.eigenvectors.T
    )
    assert torch.allclose(reconstructed, expected, atol=1e-5)
    assert torch.allclose(
        cache.scores, centered @ cache.eigenvectors, atol=1e-6
    )


def test_pca_rotated_teacher_is_an_exact_gsr_solution():
    raw = torch.randn(20, 8)
    cache = build_spectral_teacher_cache(raw, dims=[2, 4, 8])
    q = full_normalize(raw)
    student = q @ cache.eigenvectors
    out = gsr_shell_loss(
        student, cache.scores, cache.shells, cache.c_teacher
    )
    assert out.total_loss.item() == pytest.approx(0.0, abs=1e-9)


def test_within_shell_rotation_preserves_zero_loss():
    raw = torch.randn(20, 8)
    cache = build_spectral_teacher_cache(raw, dims=[2, 4, 8])
    q = full_normalize(raw)
    student_bands = []
    for start, end in cache.shells:
        width = end - start
        orthogonal, _ = torch.linalg.qr(torch.randn(width, width))
        student_bands.append(
            (q @ cache.eigenvectors[:, start:end]) @ orthogonal
        )
    student = torch.cat(student_bands, dim=1)
    out = gsr_shell_loss(
        student, cache.scores, cache.shells, cache.c_teacher
    )
    assert out.total_loss.item() == pytest.approx(0.0, abs=1e-9)


def test_cache_lookup_preserves_requested_id_order():
    cache = build_spectral_teacher_cache(torch.randn(10, 4), dims=[2, 4])
    ids = torch.tensor([7, 1, 9, 0])
    selected = cache.lookup(ids, torch.device("cpu"))
    assert torch.equal(selected, cache.scores[ids])


def test_teacher_rejects_nonfinite_and_collapsed_embeddings():
    bad = torch.randn(5, 4)
    bad[2, 1] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        build_spectral_teacher_cache(bad, dims=[2, 4])
    with pytest.raises(FloatingPointError, match="near-zero"):
        build_spectral_teacher_cache(torch.zeros(5, 4), dims=[2, 4])


def test_kernel_teacher_has_an_attainable_full_dimensional_geometry():
    raw = torch.randn(20, 8)
    cache = build_semantic_kernel_teacher_cache(
        raw,
        dims=[2, 4, 8],
        temperature=0.1,
        landmark_seed=7,
        chunk_size=3,
    )

    # PCA centers the teacher only to estimate/order axes. Adding the cached
    # translation back gives the unit Nyström feature map after an orthogonal
    # rotation, and translations cancel in all pair distances.
    student = cache.scores + cache.mean @ cache.eigenvectors
    assert torch.allclose(student.norm(dim=1), torch.ones(20), atol=1e-5)
    out = gsr_shell_loss(
        student, cache.scores, cache.shells, cache.c_teacher
    )
    assert out.total_loss.item() == pytest.approx(0.0, abs=1e-9)
    assert cache.diagnostics["geometry_type"] == "semantic_exponential_kernel"
    assert cache.diagnostics["kernel"]["landmark_count"] == 8


def test_kernel_teacher_is_chunking_invariant_and_seed_deterministic():
    raw = torch.randn(13, 6)
    first = build_semantic_kernel_teacher_cache(
        raw, dims=[2, 6], temperature=0.1, landmark_seed=11, chunk_size=2
    )
    second = build_semantic_kernel_teacher_cache(
        raw, dims=[2, 6], temperature=0.1, landmark_seed=11, chunk_size=7
    )
    assert first.diagnostics["kernel"]["landmark_ids"] == second.diagnostics[
        "kernel"
    ]["landmark_ids"]
    assert torch.allclose(
        torch.pdist(first.scores).square(),
        torch.pdist(second.scores).square(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_kernel_teacher_pads_when_corpus_is_narrower_than_embedding():
    cache = build_semantic_kernel_teacher_cache(
        torch.randn(5, 8), dims=[2, 4, 8], temperature=0.1
    )
    assert cache.scores.shape == (5, 8)
    assert cache.eigenvectors.shape == (8, 8)
    assert cache.diagnostics["kernel"]["landmark_count"] == 5


def test_shell_to_prefix_bound_holds_numerically():
    student = torch.randn(9, 6)
    teacher = torch.randn(9, 6)
    shells = [(0, 2), (2, 4), (4, 6)]
    out = gsr_shell_loss(student, teacher, shells, c_teacher=1.0)
    errors = [
        out.student_distances[key] - out.teacher_distances[key]
        for key in out.shell_losses
    ]
    for k in range(1, len(shells) + 1):
        prefix_error = torch.stack(errors[:k]).sum(0).square().mean()
        bound = k * sum(error.square().mean() for error in errors[:k])
        assert prefix_error <= bound + 1e-6

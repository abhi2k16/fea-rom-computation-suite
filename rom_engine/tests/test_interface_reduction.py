# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the interface-reduction family of component mode synthesis."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine.cms import craig_bampton
from rom_engine.interface_reduction import (dual_craig_bampton, couple_dual, joint_interface_modes, jim_craig_bampton,
                                            gram_schmidt_interface_modes, interface_reduced_craig_bampton,
                                            fourier_interface_basis, multistage_fourier_basis,
                                            trial_vector_derivative_basis, natural_frequency_errors, mac_matrix,
                                            mac_diagonal)


def chain(n, k=100.0, m=1.0):
    """Fixed-free spring-mass chain: n masses, ground on the left."""
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k
        if i:
            K[i, i - 1] = K[i - 1, i] = -k
            K[i - 1, i - 1] += k
    return K, m * np.eye(n)


def plate(nx, ny):
    """Scalar membrane on an nx-by-ny grid, left edge fixed; node (i, j) -> i*ny + j."""
    n = nx * ny
    K = np.zeros((n, n))
    idx = lambda i, j: i * ny + j
    for i in range(nx):
        for j in range(ny):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < nx and j + dj < ny:
                    a, b = idx(i, j), idx(i + di, j + dj)
                    K[a, a] += 1; K[b, b] += 1; K[a, b] -= 1; K[b, a] -= 1
    for j in range(ny):
        K[idx(0, j), idx(0, j)] += 1.0         # grounding springs on the left edge (K positive definite)
    return K, np.eye(n), [idx(nx - 1, j) for j in range(ny)]


def test_dual_craig_bampton_assembly_converges_to_the_full_chain():
    nA, nB = 12, 12
    KA, MA = chain(nA)                             # grounded left, interface = last DOF
    KB, MB = chain(nB)
    KB, MB = KB[::-1, ::-1].copy(), MB                 # interface = first DOF, grounded at the far end
    N = nA + nB - 1                                    # the two interface masses are one node of mass 2
    Kfull = np.zeros((N, N))
    for i in range(N):
        Kfull[i, i] += 100.0                            # spring i to the left neighbour (or ground)
        if i:
            Kfull[i, i - 1] = Kfull[i - 1, i] = -100.0
            Kfull[i - 1, i - 1] += 100.0
    Kfull[-1, -1] += 100.0                              # ground at the far end of B
    Mfull = np.eye(N)
    Mfull[nA - 1, nA - 1] = 2.0
    wfull = np.sqrt(eigh(Kfull, Mfull, eigvals_only=True))
    errs = []
    for nm in (2, 4, 8):
        A = dual_craig_bampton(KA, MA, [nA - 1], nm)
        B = dual_craig_bampton(KB, MB, [0], nm)
        c = couple_dual([A, B], [(0, nA - 1, 1, 0)])
        errs.append(abs(c["omega"][:3] - wfull[:3]).max() / wfull[2])
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-4 and errs[0] < 0.05


def test_jim_reduces_the_joint_and_is_exact_with_all_joint_modes():
    K, M, iface = plate(6, 10)
    joint = np.arange(2, 8)                         # six joint DOFs inside the 10-node interface
    full = jim_craig_bampton(K, M, iface, joint, 6, len(joint))
    cb = craig_bampton(K, M, iface, 6)
    w_cb = np.sqrt(eigh(cb.K, cb.M, eigvals_only=True))
    w_full = np.sqrt(eigh(full.K, full.M, eigvals_only=True))
    assert np.allclose(w_cb, w_full, rtol=1e-9)
    red = jim_craig_bampton(K, M, iface, joint, 6, 2)
    assert red.T.shape[1] == full.T.shape[1] - 4
    e = natural_frequency_errors(K, M, red.T, 3)
    assert np.all(e >= -1e-9) and np.all(np.abs(e) < 0.2)               # Rayleigh-Ritz: reduced frequencies never fall below


def test_gram_schmidt_interface_modes_orthonormal_and_drop_dependent_vectors():
    K, M, iface = plate(6, 10)
    w, V = eigh(K, M)
    Q = gram_schmidt_interface_modes([V[:, :8], V[:, :8], V[:, 8:16]], iface)
    rank = np.linalg.matrix_rank(V[iface][:, :16], tol=1e-8)
    assert Q.shape[1] == rank                       # repeated and dependent interface shapes add nothing
    assert np.allclose(Q.T @ Q, np.eye(rank), atol=1e-10)
    assert np.allclose(Q @ (Q.T @ V[iface][:, :16]), V[iface][:, :16], atol=1e-8)
    Q2 = gram_schmidt_interface_modes([V[:, :20]], iface, n_keep=5)
    assert Q2.shape[1] == 5


def test_interface_reduction_accuracy_improves_with_more_interface_modes_and_is_exact_when_full():
    K, M, iface = plate(6, 10)
    w, V = eigh(K, M)
    psi_all = gram_schmidt_interface_modes([V], iface)
    assert psi_all.shape[1] == len(iface)
    errs = []
    for k in (2, 4, len(iface)):
        red = interface_reduced_craig_bampton(K, M, iface, 30, psi_all[:, :k])
        errs.append(np.abs(natural_frequency_errors(K, M, red.T, 6)).max())
    plain = craig_bampton(K, M, iface, 30)
    assert errs[0] > 100 * errs[1] and errs[1] >= errs[2] * (1 - 1e-6)
    assert errs[2] == pytest.approx(np.abs(natural_frequency_errors(K, M, plain.T, 6)).max(), rel=1e-6)
    red = interface_reduced_craig_bampton(K, M, iface, 30, psi_all[:, :4])
    assert np.all(mac_diagonal(K, M, red.T, 3) > 0.99)


def test_fourier_basis_represents_low_harmonics_exactly():
    th = np.arange(16) * 2 * np.pi / 16
    B = fourier_interface_basis(th, 3)
    assert B.shape == (16, 7) and np.allclose(B.T @ B, np.eye(7), atol=1e-12)
    f = 0.3 + np.cos(2 * th) - 0.5 * np.sin(3 * th)
    assert np.allclose(B @ (B.T @ f), f, atol=1e-12)


def test_multistage_fourier_compresses_across_rings():
    nth, rings = 16, 4
    th1 = np.arange(nth) * 2 * np.pi / nth
    ring_index = np.repeat(np.arange(rings), nth)
    ang = np.tile(th1, rings)
    rng = np.random.default_rng(0)
    snaps = []
    for _ in range(12):                              # separable fields: ring profile times a fixed angular shape
        a, b = rng.standard_normal(2)
        prof = 1.0 + 0.5 * np.arange(rings)
        snaps.append((np.outer(prof, a * np.cos(2 * th1) + b * np.sin(th1))).reshape(-1))
    S = np.array(snaps).T
    B, info = multistage_fourier_basis(ring_index, ang, 3, snapshots=S)
    assert B.shape[1] == 2                          # harmonics cos2t and sin t, each with ONE ring profile
    assert np.linalg.norm(S - B @ (B.T @ S)) < 1e-10 * np.linalg.norm(S)
    full, _ = multistage_fourier_basis(ring_index, ang, 3)
    assert full.shape[1] == 7 * rings


def test_trial_vector_derivatives_are_filtered_and_mass_orthonormal():
    n = 10
    K0, M = chain(n)
    k3 = 5.0
    tangent = lambda u: K0 + np.diag(3 * k3 * u ** 2)
    w, V = eigh(K0, M)
    Phi = V[:, :2]
    basis, info = trial_vector_derivative_basis(K0, Phi, tangent_fn=tangent, M=M, u0=0.3 * V[:, 0])
    assert 0 < info["n_added"] <= 3
    assert np.allclose(basis.T @ M @ basis, np.eye(basis.shape[1]), atol=1e-8)


def test_metrics_basic_properties():
    A = np.eye(3)
    assert np.allclose(np.diag(mac_matrix(A, A)), 1.0)
    K, M, iface = plate(5, 6)
    cb = craig_bampton(K, M, iface, None)
    assert np.allclose(natural_frequency_errors(K, M, cb.T), 0.0, atol=1e-9)

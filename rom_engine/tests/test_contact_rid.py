# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the RID-based hybrid hyper-reduction of contact (Fauque, Ramiere, Ryckelynck 2018).
Model: a 1D rope (tridiagonal K) pushed up against an obstacle, paper sign convention K U + B^T Lam = F, B U <= D."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp
import pytest

from rom_engine.contact_rid import (pod_by_information_rate, rid_elements, rid_dofs, contact_rows_in_rid,
                                    HybridHRContact, restricted_fom_solve, snapshot_interpolator, h1_norm,
                                    dual_h1_norm, interpolation_indicator)
from rom_engine.complementarity import solve_reduced_contact

N = 60
H = 1.0 / (N + 1)
X = np.linspace(H, 1 - H, N)
ELEMS = [[d for d in (e - 1, e) if 0 <= d < N] for e in range(N + 1)]        # element e joins free DOFs e-1, e


def stiffness(nu):
    return sp.diags([-nu / H * np.ones(N - 1), 2 * nu / H * np.ones(N), -nu / H * np.ones(N - 1)], [-1, 0, 1]).tocsc()


def fom(nu, d):
    """Paper convention reference solve via the package's reduced-contact LCP (B=-I, c=-d)."""
    K = stiffness(nu).toarray()
    u, lam, _, ok = solve_reduced_contact(K, H * np.ones(N), -np.eye(N), -d)
    assert ok
    return u, np.maximum(lam, 0)


@pytest.fixture(scope="module")
def snaps():
    nus = np.linspace(0.08, 0.2, 7)
    hs = np.linspace(0.3, 0.5, 4)
    cases = [(nu, np.full(N, h)) for nu in nus for h in hs]
    S = [fom(nu, d) for nu, d in cases]
    return cases, np.array([s[0] for s in S]).T, np.array([s[1] for s in S]).T


def test_pod_information_rate(snaps):
    _, SU, _ = snaps
    V = pod_by_information_rate(SU, 1e-8)
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
    assert np.linalg.norm(SU - V @ (V.T @ SU)) <= np.sqrt(1e-8) * np.linalg.norm(SU) * 1.01   # Eckart-Young bound


def test_rid_elements_layers_and_previous():
    e0 = rid_elements(ELEMS, N, [10], layers=0)
    assert set(e0) == {10, 11}                                                    # the support of the hat function of DOF 10
    e1 = rid_elements(ELEMS, N, [10], layers=1)
    assert set(e0) < set(e1)
    e2 = rid_elements(ELEMS, N, [30], layers=0, previous=e0)
    assert set(e0) <= set(e2) and {30, 31} <= set(e2)                             # the RID can only grow


def test_rid_dofs_balance_set():
    el = rid_elements(ELEMS, N, [10], layers=1)                                   # elements 9..12
    P, ext = rid_dofs(ELEMS, N, el)
    assert set(P) == {9, 10, 11}                                                  # DOFs whose both elements are inside
    assert set(ext) == set(range(8, 12 + 0)) | {8, 9, 10, 11, 12}
    assert set(P) <= set(ext)
    rows = contact_rows_in_rid(sp.identity(N), P)
    assert set(rows) == set(P)


def test_full_rid_reproduces_projection_and_active_set(snaps):
    """With the RID = whole mesh, the hybrid HR reduces to a Petrov-Galerkin ROM with every contact kept as an FE
    unknown: it must reproduce the FOM of an unseen parameter to POD accuracy and give the same active set."""
    cases, SU, SL = snaps
    V = pod_by_information_rate(SU, 1e-12)
    P, ext = rid_dofs(ELEMS, N, np.arange(N + 1))
    B = sp.identity(N, format="csr")
    rows = contact_rows_in_rid(B, P)
    hy = HybridHRContact(V, B, P, ext, rows)
    assert hy.inf_sup_deficiency() == 0
    nu, d = 0.123, np.full(N, 0.41)
    u0, l0 = fom(nu, d)
    res = hy.solve(stiffness(nu)[P][:, ext], H * np.ones(N)[P], d[rows])
    assert res.ok
    assert np.linalg.norm(res.u - u0) / np.linalg.norm(u0) < 1e-3
    assert np.array_equal(res.lam > 1e-8, l0 > 1e-8)
    assert np.all(res.lam >= -1e-12)                                              # Signorini: multipliers are non-negative
    assert np.all(B @ res.u - d <= 1e-8)                                          # and the obstacle is not penetrated


def test_enrichment_restores_full_row_rank(snaps):
    """A basis smaller than the number of contacts makes B_r rank deficient (violates inf-sup); the hybrid basis
    must repair it by appending FE shape functions."""
    _, SU, _ = snaps
    V = pod_by_information_rate(SU, 1e-3)[:, :3]
    el = rid_elements(ELEMS, N, [10, 30], layers=2)
    P, ext = rid_dofs(ELEMS, N, el)
    B = sp.identity(N, format="csr"); rows = contact_rows_in_rid(B, P)
    assert len(rows) > V.shape[1]
    raw = HybridHRContact(V, B, P, ext, rows, enrich=False)
    assert raw.inf_sup_deficiency() > 0
    hy = HybridHRContact(V, B, P, ext, rows, enrich=True)
    assert hy.inf_sup_deficiency() == 0
    assert hy.V.shape[1] > V.shape[1] and len(hy.added_dofs) > 0
    assert hy.n_unknowns == hy.V.shape[1] + len(rows)


def test_restricted_fom_with_exact_interface_data_matches_fom(snaps):
    """If the interface displacements are the exact FOM ones, the problem written on the RID alone must return the
    FOM displacements and multipliers inside it (this is why eta -> 0 as the interpolation error -> 0)."""
    nu, d = 0.12, np.full(N, 0.42)
    u0, l0 = fom(nu, d)
    el = rid_elements(ELEMS, N, [20], layers=3)
    P, ext = rid_dofs(ELEMS, N, el)
    I = np.setdiff1d(ext, P)
    K = stiffness(nu).tocsr()
    B_P = np.eye(len(P))
    u_P, lam, ok = restricted_fom_solve(K[P][:, P].toarray(), K[P][:, I], H * np.ones(len(P)), u0[I], B_P, d[P])
    assert ok
    assert np.allclose(u_P, u0[P], atol=1e-9)
    assert np.allclose(lam, l0[P], atol=1e-8)


def test_indicator_zero_for_exact_and_positive_otherwise():
    M = sp.identity(5, format="csr") * 2.0
    u, lam = np.arange(1.0, 6), np.array([1.0, 0.0, 2.0])
    eta, eu, el = interpolation_indicator(u, lam, u.copy(), lam.copy(), M, np.eye(3))
    assert eta == 0.0
    eta2, eu2, el2 = interpolation_indicator(1.1 * u, lam + 0.1, u, lam, M, np.eye(3))
    assert eu2 == pytest.approx(0.1) and el2 > 0 and eta2 == pytest.approx(eu2 + el2)
    assert h1_norm(u, M) == pytest.approx(np.sqrt(2 * np.sum(u ** 2)))
    assert dual_h1_norm(u, M) == pytest.approx(np.sqrt(np.sum(u ** 2) / 2))


def test_snapshot_interpolator_exact_on_linear_data_and_nearest_fallback():
    P = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    vals = np.vstack([P[:, 0] + 2 * P[:, 1], 3 * P[:, 0]])                          # 2 fields, linear in the parameters
    f = snapshot_interpolator(P, vals)
    assert np.allclose(f([0.3, 0.2]), [0.7, 0.9])                                    # linear data reproduced inside the hull
    g = snapshot_interpolator(P[:1], vals[:, :1])                                     # a single snapshot: nearest neighbour
    assert np.allclose(g([5.0, 5.0]), vals[:, 0])

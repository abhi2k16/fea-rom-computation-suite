# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the non-negative dual basis, the reduced contact solve and the sparse-dictionary alternative."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from contact_fixtures import ObstacleMembrane
from rom_engine.complementarity import (lcp_lemke, lcp_active_set, solve_lcp, solve_reduced_contact,
                                        complementarity_violation)
from rom_engine.contact_dual_basis import (nonneg_dual_basis, svd_dual_basis, active_snapshot_columns, ContactROM,
                                           sparse_dictionary_fit)


@pytest.fixture(scope="module")
def membrane():
    m = ObstacleMembrane(n=14)
    train = [(a, b) for a in np.linspace(0.3, 0.6, 5) for b in np.linspace(0.0, 0.1, 4)]
    Xu, Xl = m.training_set(train)
    return m, Xu, Xl


def test_dense_lcp_solvers_agree_with_each_other_and_satisfy_complementarity():
    rng = np.random.default_rng(3)
    for _ in range(5):
        X = rng.standard_normal((7, 7))
        A, q = X @ X.T + 0.3 * np.eye(7), rng.standard_normal(7) * 3
        z1, ok1 = solve_lcp(A, q)
        z2, ok2 = solve_lcp(A, q, "lemke")
        assert ok1 and ok2 and np.allclose(z1, z2, atol=1e-8)
        assert complementarity_violation(z1, A @ z1 + q) < 1e-8
    with pytest.raises(ValueError):
        solve_lcp(np.eye(2), np.ones(2), "nope")


def test_reduced_contact_solve_matches_a_scalar_obstacle():
    u, lam, gap, ok = solve_reduced_contact(np.array([[4.0]]), np.array([2.0]), np.array([[1.0]]), np.array([0.8]))
    assert ok and u[0] == pytest.approx(0.8) and lam[0] == pytest.approx(4 * 0.8 - 2.0)


def test_nmf_recovers_an_exactly_low_rank_nonnegative_matrix():
    rng = np.random.default_rng(0)
    W0, H0 = rng.random((60, 4)), rng.random((4, 40))
    X = W0 @ H0
    nmf = nonneg_dual_basis(X, 4, n_iter=2000)
    assert nmf.rel_error < 1e-4
    assert nmf.U.min() >= 0 and nmf.V.min() >= 0
    assert np.allclose(np.linalg.norm(nmf.U, axis=0), 1.0)
    with pytest.raises(ValueError):
        nonneg_dual_basis(-X, 2)


def test_nmf_error_is_no_better_than_the_svd_optimum(membrane):
    _, _, Xl = membrane
    for r in (3, 6, 10):
        U, S, Vt = np.linalg.svd(Xl, full_matrices=False)
        opt = np.linalg.norm(Xl - (U[:, :r] * S[:r]) @ Vt[:r]) / np.linalg.norm(Xl)
        nmf = nonneg_dual_basis(Xl, r)
        assert nmf.rel_error >= opt - 1e-9


def test_active_snapshot_columns_drops_contact_free_steps():
    X = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 3.0]])
    assert np.array_equal(active_snapshot_columns(X), [1, 3])


def test_nmf_dual_basis_keeps_multipliers_nonnegative_where_svd_does_not(membrane):
    m, Xu, Xl = membrane
    U = np.linalg.svd(Xu, full_matrices=False)[0][:, :12]
    nmf = nonneg_dual_basis(Xl, 12)
    rom_nmf = ContactROM(m.K, m.B, U, nmf.U)
    rom_svd = ContactROM(m.K, m.B, U, svd_dual_basis(Xl, 12))
    for gamma in [(0.37, 0.03), (0.52, 0.08)]:                     # unseen parameters
        u, lam, c = m.solve(gamma)
        r1, r2 = rom_nmf.solve(m.f, c), rom_svd.solve(m.f, c)
        assert r1.ok and r2.ok
        assert r1.lam.min() >= 0.0                                   # by construction
        assert np.linalg.norm(r1.u - u) / np.linalg.norm(u) < 5e-3
        assert r2.lam.min() < -1e-3 * lam.max()                      # the SVD basis breaks positivity
        assert np.linalg.norm(r2.u - u) > 10 * np.linalg.norm(r1.u - u)


def test_inf_sup_report_flags_rank_deficiency(membrane):
    m, Xu, Xl = membrane
    U = np.linalg.svd(Xu, full_matrices=False)[0][:5]
    U = np.linalg.svd(Xu, full_matrices=False)[0][:, :5]
    ok = ContactROM(m.K, m.B, U, nonneg_dual_basis(Xl, 4).U).inf_sup_report()
    assert ok["full_row_rank"] and ok["n_dual"] == 4 and ok["n_primal"] == 5
    bad = ContactROM(m.K, m.B, U, nonneg_dual_basis(Xl, 9).U).inf_sup_report()
    assert not bad["full_row_rank"] and bad["rank"] <= 5


def test_sparse_dictionary_fit_recovers_a_two_atom_target():
    rng = np.random.default_rng(1)
    D = rng.random((30, 12))
    y = 2.0 * D[:, 3] + 0.5 * D[:, 8]
    coef, sel, res = sparse_dictionary_fit(D, y, n_atoms=2)
    assert res < 1e-10 and set(sel) == {3, 8}
    assert coef[3] == pytest.approx(2.0) and coef[8] == pytest.approx(0.5)
    coef1, sel1, res1 = sparse_dictionary_fit(D, y, n_atoms=1)
    assert len(sel1) == 1 and res1 > 1e-3 and np.all(coef1 >= 0)

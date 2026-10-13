# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for fea_engine.contact_solvers: LCP solvers, static/dynamic/nonlinear unilateral contact."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize

from fea_engine.contact_solvers import (lcp_lemke, lcp_active_set, lcp_projected_gauss_seidel, solve_lcp,
                                        solve_contact_static, solve_contact_dynamics, solve_contact_newton,
                                        contact_complementarity_residual)


def _spd(n, rng):
    X = rng.standard_normal((n, n))
    return X @ X.T + 0.5 * np.eye(n)


def _lcp_kkt(A, q, z):
    w = A @ z + q
    return contact_complementarity_residual(z, w)


def _chain(n, k=1.0):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2 * k
        if i:
            K[i, i - 1] = K[i - 1, i] = -k
    return K


def _brute_force(K, f, B, c):
    """Enumerate active sets of an SPD contact problem; return (u, lam) of the (unique) KKT point."""
    m = B.shape[0]
    for r in range(m + 1):
        for S in itertools.combinations(range(m), r):
            S = list(S)
            if S:
                KKT = np.block([[K, -B[S].T], [B[S], np.zeros((len(S), len(S)))]])
                sol = np.linalg.solve(KKT, np.concatenate([f, c[S]]))
                u, lam_S = sol[:K.shape[0]], sol[K.shape[0]:]
            else:
                u, lam_S = np.linalg.solve(K, f), np.zeros(0)
            lam = np.zeros(m)
            lam[S] = lam_S
            if np.all(lam >= -1e-10) and np.all(B @ u - c >= -1e-10):
                return u, lam
    raise AssertionError("no KKT point found")


@pytest.mark.parametrize("seed", range(5))
def test_lcp_solvers_agree_and_satisfy_kkt(seed):
    rng = np.random.default_rng(seed)
    A = _spd(8, rng)
    q = rng.standard_normal(8) * 3
    zs = {}
    for name in ("active_set", "lemke", "pgs"):
        z, ok = solve_lcp(A, q, method=name)
        assert ok, name
        assert _lcp_kkt(A, q, z) < 1e-6, name
        zs[name] = z
    assert np.allclose(zs["active_set"], zs["lemke"], atol=1e-8)
    assert np.allclose(zs["pgs"], zs["lemke"], atol=1e-5)


def test_lcp_trivial_when_q_nonnegative():
    A = _spd(4, np.random.default_rng(0))
    for f in (lcp_lemke, lambda A, q: lcp_active_set(A, q)[:2], lcp_projected_gauss_seidel):
        z, ok = f(A, np.ones(4))
        assert ok and np.allclose(z, 0)


def test_lemke_handles_singular_psd_matrix():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])         # rank one, PSD
    q = np.array([-1.0, -1.0])
    z, ok = lcp_lemke(a, q)
    assert ok and _lcp_kkt(a, q, z) < 1e-9


def test_single_spring_obstacle_analytic():
    k, f = 4.0, 2.0                                 # free displacement 0.5
    for c, expected_u, expected_lam in ((0.3, 0.5, 0.0), (0.8, 0.8, k * 0.8 - f)):
        res = solve_contact_static(np.array([[k]]), np.array([f]), np.array([[1.0]]), np.array([c]))
        assert res.u[0] == pytest.approx(expected_u)
        assert res.lam[0] == pytest.approx(expected_lam)


@pytest.mark.parametrize("seed", range(4))
def test_static_contact_matches_active_set_enumeration(seed):
    rng = np.random.default_rng(seed)
    n, m = 6, 4
    K = _chain(n) + 0.1 * np.eye(n)
    f = rng.uniform(0.5, 2.0, n)
    B = np.zeros((m, n))
    nodes = rng.choice(n, m, replace=False)
    B[np.arange(m), nodes] = 1.0
    c = rng.uniform(0.2, 1.5, m)
    res = solve_contact_static(K, f, B, c)
    u_ref, lam_ref = _brute_force(K, f, B, c)
    assert np.allclose(res.u, u_ref, atol=1e-9)
    assert np.allclose(res.lam, lam_ref, atol=1e-9)
    assert contact_complementarity_residual(res.lam, res.gap) < 1e-8


def test_static_contact_sparse_k_and_b():
    n = 40
    K = sp.diags([-1, 2.1, -1], [-1, 0, 1], shape=(n, n), format="csc")
    f = np.ones(n)
    rows = np.arange(5, 35, 6)
    B = sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), rows)), shape=(len(rows), n))
    c = np.full(len(rows), 12.0)
    res = solve_contact_static(K, f, B, c)
    assert contact_complementarity_residual(res.lam, res.gap) < 1e-8
    assert np.all(res.lam >= 0) and len(res.active) > 0
    # equilibrium: K u - f = B^T lam
    assert np.allclose(K @ res.u - f, B.T @ res.lam, atol=1e-8)


def test_dynamics_oscillator_never_penetrates_and_multiplier_is_nonnegative():
    # unit mass on a spring dropped toward a floor at u = -0.2 (constraint u >= c)
    M, K = np.array([[1.0]]), np.array([[40.0]])
    B, c = np.array([[1.0]]), np.array([-0.2])
    U, Lam = solve_contact_dynamics(M, K, lambda t: np.zeros(1), B, c, h=1e-3, n_steps=600,
                                    u0=np.array([0.5]), v0=np.zeros(1))
    assert U[:, 0].min() >= -0.2 - 1e-9
    assert Lam.min() >= 0 and Lam.max() > 0
    # multiplier acts only when in contact
    inside = U[:, 0] > -0.2 + 1e-7
    assert np.all(Lam[inside, 0] < 1e-8)
    # the free part of the motion follows the linear oscillator (before first contact)
    t = np.arange(len(U)) * 1e-3
    free = t < 0.05
    assert np.allclose(U[free, 0], 0.5 * np.cos(np.sqrt(40.0) * t[free]), atol=2e-3)


def test_newton_nonlinear_contact_matches_energy_minimisation():
    n = 5
    K = _chain(n) + 0.2 * np.eye(n)
    k3 = 3.0
    f = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    B = np.zeros((2, n))
    B[0, 1] = 1.0
    B[1, 3] = 1.0
    c = np.array([0.9, 1.0])

    def residual(u):
        return K @ u + k3 * u ** 3 - f

    def tangent(u):
        return K + np.diag(3 * k3 * u ** 2)

    res = solve_contact_newton(residual, tangent, B, c, np.zeros(n))
    assert res.converged
    energy = lambda u: 0.5 * u @ K @ u + 0.25 * k3 * np.sum(u ** 4) - f @ u
    ref = minimize(energy, np.zeros(n), constraints=[{"type": "ineq", "fun": lambda u: B @ u - c}],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    assert ref.success
    assert np.allclose(res.u, ref.x, atol=1e-5)
    assert contact_complementarity_residual(res.lam, res.gap) < 1e-8
    assert len(res.active) >= 1


def test_complementarity_residual_detects_violations():
    assert contact_complementarity_residual([0.0, 1.0], [2.0, 0.0]) == 0.0
    assert contact_complementarity_residual([1.0], [1.0]) == pytest.approx(1.0)
    assert contact_complementarity_residual([0.0], [-0.5]) == pytest.approx(0.5)
    assert contact_complementarity_residual([-0.3], [0.0]) == pytest.approx(0.3)
    assert contact_complementarity_residual([], []) == 0.0

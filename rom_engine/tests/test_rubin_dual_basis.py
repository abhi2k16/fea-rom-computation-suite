# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Rubin + dual-mode basis for a contact patch, and polynomial-force re-identification."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rom_engine.rubin_dual_basis import (rubin_dual_basis, select_by_representation_error, physical_contact_basis,
                                         reidentify_polynomial_force)

N = 20
K3 = 40.0
CONTACT = np.array([N - 2, N - 1])


def system():
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += 100.0
        if i:
            K[i, i - 1] = K[i - 1, i] = -100.0
            K[i - 1, i - 1] += 100.0
    return K, np.eye(N)


def static_solver(K):
    """Nonlinear static response of the open-contact model: K u + K3 * sum_neighbours (du)^3-type hardening on
    grounded springs (a smooth, representable nonlinearity)."""
    def solve(load):
        u = np.linalg.solve(K, load)
        for _ in range(60):
            R = K @ u + K3 * u ** 3 - load
            if np.linalg.norm(R) < 1e-10 * max(1.0, np.linalg.norm(load)):
                break
            u = u - np.linalg.solve(K + np.diag(3 * K3 * u ** 2), R)
        return u
    return solve


def test_dual_modes_are_orthogonal_to_rubin_and_reduce_the_representation_error():
    K, M = system()
    solve = static_solver(K)
    rng = np.random.default_rng(0)
    snaps = np.column_stack([solve(a * K @ np.sin(np.pi * (k + 1) * (np.arange(N) + 0.5) / N * 0.5))
                             for a in (0.6, 1.2) for k in range(3)])
    T, info = rubin_dual_basis(K, M, CONTACT, 4, solve, amplitudes=[0.5, 1.0, 1.5], p=np.ones(N), n_dual=3,
                               snapshots=snaps)
    assert info["n_dual"] >= 1 and T.shape[1] == 4 + info["n_attachment"] + info["n_dual"]
    assert info["orthogonality"] < 1e-8
    assert info["representation_error_with_dual"] < info["representation_error_rubin"]
    errs = info["representation_error_by_count"]
    assert all(errs[i + 1] <= errs[i] + 1e-12 for i in range(len(errs) - 1))


def test_representation_error_selection_orders_candidates_by_usefulness():
    n = 12
    M = np.eye(n)
    base = np.eye(n)[:, :2]
    cand = np.eye(n)[:, 2:6]
    S = np.column_stack([3.0 * np.eye(n)[:, 4] + 0.1 * np.eye(n)[:, 3], np.eye(n)[:, 0]])
    order, errs = select_by_representation_error(cand, S, base, M, 2)
    assert order[0] == 2                                  # candidate index 2 is e_4, the dominant missing direction
    assert errs[0] > errs[1] > errs[2]


def test_physical_contact_basis_keeps_the_span_and_gives_identity_on_contact_dofs():
    rng = np.random.default_rng(1)
    T = rng.standard_normal((N, 7))
    Tp = physical_contact_basis(T, CONTACT)
    assert np.allclose(Tp[CONTACT][:, :2], np.eye(2), atol=1e-12)
    assert np.allclose(Tp[CONTACT][:, 2:], 0.0, atol=1e-12)
    # same subspace: each column of Tp is in span(T) and ranks match
    assert np.linalg.matrix_rank(np.hstack([T, Tp]), tol=1e-9) == np.linalg.matrix_rank(T)
    assert np.linalg.matrix_rank(Tp, tol=1e-9) == 7
    with pytest.raises(ValueError):
        physical_contact_basis(T[:, :1], CONTACT)


def test_reidentification_recovers_cubic_coefficients_from_a_reduced_time_history():
    r = 2
    Mr = np.diag([1.0, 1.5])
    Kr = np.array([[30.0, -5.0], [-5.0, 20.0]])
    Cr = 0.05 * Kr
    true = lambda q: np.array([8.0 * q[0] ** 3 + 2.0 * q[0] * q[1] ** 2, 5.0 * q[1] ** 3 - 1.0 * q[0] ** 2 * q[1] + 3.0 * q[0] ** 2])
    force = lambda t: np.array([4.0 * np.sin(3.0 * t), 3.0 * np.sin(4.1 * t + 0.4)])
    Minv = np.linalg.inv(Mr)
    rhs = lambda t, y: np.concatenate([y[2:], Minv @ (force(t) - Cr @ y[2:] - Kr @ y[:2] - true(y[:2]))])
    t = np.linspace(0, 30, 3000)
    sol = solve_ivp(rhs, [0, 30], [0.1, 0.0, 0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    q, qd = sol.y[:2].T, sol.y[2:].T
    f = np.array([force(tk) for tk in t])
    qdd = np.array([Minv @ (f[i] - Cr @ qd[i] - Kr @ q[i] - true(q[i])) for i in range(len(t))])
    fit, info = reidentify_polynomial_force(Mr, Cr, Kr, q, qd, qdd, f)
    assert info["relative_residual"] < 1e-8
    probe = np.array([0.07, -0.05])
    assert np.allclose(fit.force(probe), true(probe), atol=1e-8)
    # Jacobian against central differences
    J = fit.jacobian(probe)
    h = 1e-6
    Jfd = np.column_stack([(fit.force(probe + h * e) - fit.force(probe - h * e)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(J, Jfd, atol=1e-5)

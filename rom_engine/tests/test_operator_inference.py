# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Craig-Bampton-structured operator inference for unilateral-contact dynamics."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from fea_engine.contact_solvers import solve_contact_dynamics
from rom_engine.operator_inference import (static_coupling, lstsq_coupling, craig_bampton_transformation,
                                           fit_spd_operators, infer_contact_operators)

N = 30
BND = np.array([N - 2, N - 1])
R = 10


def system():
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += 100.0
        if i:
            K[i, i - 1] = K[i - 1, i] = -100.0
            K[i - 1, i - 1] += 100.0
    M = np.diag(1.0 + 0.2 * np.sin(np.arange(N)))
    return K, M


def training_data(K, M, Phi, seed=0, nt=500, noise=0.0):
    """Broadband-in-time but low-frequency forcing: random loads on the boundary DOFs plus loads M_II Phi c(t) on
    the interior, so the response stays (to the retained-mode accuracy) inside the Craig-Bampton subspace."""
    rng = np.random.default_rng(seed)
    interior = np.setdiff1d(np.arange(N), BND)
    Minv = np.linalg.inv(M)
    nb, r = len(BND), Phi.shape[1]
    freqs = rng.uniform(0.3, 2.5, (nb + r, 3))
    phases = rng.uniform(0, 2 * np.pi, (nb + r, 3))
    amps = rng.standard_normal((nb + r, 3))
    MPhi = M[np.ix_(interior, interior)] @ Phi

    def force(t):
        c = (amps * np.sin(freqs * t + phases)).sum(1)
        f = np.zeros(N)
        f[BND] = 5.0 * c[:nb]
        f[interior] = MPhi @ c[nb:]
        return f
    rhs = lambda t, y: np.concatenate([y[N:], Minv @ (force(t) - K @ y[:N])])
    t = np.linspace(0, 40, nt)
    sol = solve_ivp(rhs, [0, 40], np.zeros(2 * N), t_eval=t, rtol=1e-10, atol=1e-12)
    U = sol.y[:N]
    F = np.column_stack([force(tk) for tk in t])
    A = Minv @ (F - K @ U)
    return U, A, F


@pytest.fixture(scope="module")
def ctx():
    K, M = system()
    interior = np.setdiff1d(np.arange(N), BND)
    Kii, Mii, Kib = K[np.ix_(interior, interior)], M[np.ix_(interior, interior)], K[np.ix_(interior, BND)]
    w2, V = eigh(Kii, Mii)
    Phi, _ = np.linalg.qr(V[:, :R])                            # orthonormal fixed-boundary interior modes
    Psi = static_coupling(lambda e: -np.linalg.solve(Kii, Kib @ e), len(BND))
    return dict(K=K, M=M, interior=interior, Phi=Phi, Psi=Psi, Kii=Kii, Kib=Kib)


def test_static_coupling_equals_the_analytic_matrix(ctx):
    assert np.allclose(ctx["Psi"], -np.linalg.solve(ctx["Kii"], ctx["Kib"]), atol=1e-12)


def test_learned_operators_match_the_galerkin_projection_and_are_spd(ctx):
    U, A, F = training_data(ctx["K"], ctx["M"], ctx["Phi"])
    model = infer_contact_operators(U, A, F, BND, ctx["Phi"], ctx["Psi"])
    T = craig_bampton_transformation(BND, N, ctx["Phi"], ctx["Psi"])
    Mg, Kg = T.T @ ctx["M"] @ T, T.T @ ctx["K"] @ T
    assert np.linalg.norm(model.M - Mg) / np.linalg.norm(Mg) < 5e-2
    assert np.linalg.norm(model.K - Kg) / np.linalg.norm(Kg) < 5e-2
    assert np.linalg.eigvalsh(model.M).min() > 0 and np.linalg.eigvalsh(model.K).min() > 0
    assert model.info["projection_residual"] < 0.05


def test_spd_constraint_is_enforced_when_data_are_noisy(ctx):
    U, A, F = training_data(ctx["K"], ctx["M"], ctx["Phi"])
    rng = np.random.default_rng(5)
    A = A + 0.5 * np.abs(A).max() * rng.standard_normal(A.shape)         # heavy acceleration noise
    model = infer_contact_operators(U, A, F, BND, ctx["Phi"], ctx["Psi"], max_nfev=30)
    assert np.linalg.eigvalsh(model.M).min() > 0 and np.linalg.eigvalsh(model.K).min() > 0
    X = np.random.default_rng(1).standard_normal((4, 60))
    Mc, Kc, info = fit_spd_operators(X, X[::-1] * 0.3, -X, max_nfev=30)                # nonsense data: still SPD output
    assert np.linalg.eigvalsh(Mc).min() > 0 and np.linalg.eigvalsh(Kc).min() > 0


def test_static_coupling_is_more_accurate_than_the_data_fitted_coupling(ctx):
    U, A, F = training_data(ctx["K"], ctx["M"], ctx["Phi"])
    Psi_ls = lstsq_coupling(U[ctx["interior"]], U[BND], ctx["Phi"])
    err_static = np.linalg.norm(ctx["Psi"] - (-np.linalg.solve(ctx["Kii"], ctx["Kib"])))
    assert err_static < 1e-10                                             # static solves reproduce the analytic matrix
    # what matters is the accuracy of the resulting reduced state projection of the full-order data
    T_s = craig_bampton_transformation(BND, N, ctx["Phi"], ctx["Psi"])
    T_l = craig_bampton_transformation(BND, N, ctx["Phi"], Psi_ls)
    ps = lambda T: np.linalg.norm(U - T @ np.linalg.lstsq(T, U, rcond=None)[0]) / np.linalg.norm(U)
    assert ps(T_s) < 0.05 and ps(T_l) < 0.2
    assert ps(T_s) <= ps(T_l) + 1e-12                                       # the data-fitted coupling is never better


def test_contact_dynamics_is_as_accurate_as_the_intrusive_projection_with_the_same_basis(ctx):
    from rom_engine.operator_inference import OperatorInferenceModel
    K, M = ctx["K"], ctx["M"]
    U, A, F = training_data(K, M, ctx["Phi"])
    model = infer_contact_operators(U, A, F, BND, ctx["Phi"], ctx["Psi"])
    T = model.T
    galerkin = OperatorInferenceModel(T.T @ M @ T, T.T @ K @ T, T, BND)
    B = np.zeros((1, N))
    B[0, N - 1] = 1.0                                                      # last mass may not go below -0.02
    c = np.array([-0.02])
    h, steps = 2e-3, 1500
    f = np.zeros(N)
    f[N - 1] = -0.3                                                        # constant push toward the obstacle
    force = lambda t: f
    Ufom, Lfom = solve_contact_dynamics(M, K, force, B, c, h, steps)
    scale = np.abs(Ufom).max()
    errs = {}
    for name, mod in (("inferred", model), ("galerkin", galerkin)):
        X, Ured, Lred = mod.simulate_contact(force, B, c, h, steps)
        errs[name] = np.abs(Ured - Ufom).max() / scale
        assert Lred.max() > 0 and Ured[:, N - 1].min() >= -0.02 - 1e-8     # contact engages, no penetration
    assert Lfom.max() > 0
    assert errs["inferred"] < 0.08                                         # measured 0.044 (Galerkin 0.056)
    assert errs["inferred"] < 1.5 * errs["galerkin"] + 0.02

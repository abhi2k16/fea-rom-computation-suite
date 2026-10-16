# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for `solve_contact_dynamics(..., scheme="bdf2")` (fea_engine.contact_solvers): the default scheme is unchanged, observed order of
accuracy 1 against 2 on the free membrane (the check C4 of the contact-ROM validation), the contact conditions and the BDF2 equation
satisfied at every step of an obstacle problem, agreement of the dense and the sparse contact solves, failure reporting, and an impact
test whose measured energy history is recorded in the plan (no energy conservation is claimed)."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp

from fea_engine import solve_contact_dynamics, ContactDynamicsInfo
from test_contact_solvers_sparse import membrane, obstacles


def mem_system(ne):
    K, f, X, Y = membrane(ne)
    n = K.shape[0]
    M = sp.diags(np.full(n, (1.0 / ne) ** 2)).tocsr()
    return K, M, f, X, Y


def two_step_reference(M, K, force, B, c, h, n_steps, u0, v0):
    """The documented two-step scheme, written out independently (dense, one LCP solve by active-set enumeration is not needed:
    only used on problems without active contact)."""
    n = M.shape[0]
    S = M + h * h * K
    U = np.zeros((n_steps + 1, n))
    U[0], U[1] = u0, u0 + h * v0
    for k in range(1, n_steps):
        U[k + 1] = np.linalg.solve(S, h * h * force((k + 1) * h) + M @ (2 * U[k] - U[k - 1]))
    return U


# ------------------------------------------------------------------------------------------- unchanged default
def test_default_scheme_is_the_documented_two_step_scheme_and_unchanged():
    K, M, f, X, Y = mem_system(10)
    Kd, Md = K.toarray(), M.toarray()
    n = Kd.shape[0]
    B = np.eye(n)
    c = np.full(n, -100.0)                                # no contact
    h, N = 0.01, 30
    U, L = solve_contact_dynamics(Md, Kd, lambda t: f, B, c, h, N)
    Ue, Le = solve_contact_dynamics(Md, Kd, lambda t: f, B, c, h, N, scheme="two_step")
    assert np.array_equal(U, Ue) and np.array_equal(L, Le)
    ref = two_step_reference(Md, Kd, lambda t: f, B, c, h, N, np.zeros(n), np.zeros(n))
    assert np.abs(U - ref).max() < 1e-13 and np.abs(L).max() == 0


def test_two_step_with_return_info_has_no_velocities():
    K, M, f, X, Y = mem_system(6)
    n = K.shape[0]
    out = solve_contact_dynamics(M.toarray(), K.toarray(), lambda t: f, np.eye(n), np.full(n, -9.0), 0.01, 5, return_info=True)
    assert len(out) == 3 and isinstance(out[2], ContactDynamicsInfo) and out[2].V is None and out[2].ok


# --------------------------------------------------------------------------------------------- order of accuracy
def test_free_membrane_order_one_against_order_two():
    """Check C4 of the validation report: error at t = 2 against the exact modal solution, dt = 0.01 / 0.005 / 0.0025."""
    K, M, f, X, Y = mem_system(16)
    Kd, Md = K.toarray(), M.toarray()
    n = Kd.shape[0]
    w2, V = sl.eigh(Kd, Md)
    a = V.T @ f
    exact = V @ ((a / w2) * (1 - np.cos(np.sqrt(w2) * 2.0)))
    B, c = np.eye(n), np.full(n, -100.0)
    e_b, e_t = [], []
    for h in (0.01, 0.005, 0.0025):
        N = int(round(2 / h))
        U2, _ = solve_contact_dynamics(Md, Kd, lambda t: f, B, c, h, N, scheme="bdf2")
        U1, _ = solve_contact_dynamics(Md, Kd, lambda t: f, B, c, h, N)
        e_b.append(np.abs(U2[-1] - exact).max())
        e_t.append(np.abs(U1[-1] - exact).max())
    rate_b = np.log2(np.array(e_b[:-1]) / np.array(e_b[1:]))
    rate_t = np.log2(np.array(e_t[:-1]) / np.array(e_t[1:]))
    assert np.all(rate_b > 1.8) and np.all((rate_t > 0.8) & (rate_t < 1.3)), (rate_b, rate_t)
    # the numbers of the report (1.2e-2 / 3.1e-3 / 7.8e-4 and 1.6e-1 / 8.7e-2 / 4.7e-2), within a factor 2
    for got, rep in zip(e_b, (1.2e-2, 3.1e-3, 7.8e-4)):
        assert 0.5 * rep < got < 2 * rep
    for got, rep in zip(e_t, (1.6e-1, 8.7e-2, 4.7e-2)):
        assert 0.5 * rep < got < 2 * rep
    assert e_t[1] / e_b[1] > 20 and e_t[2] / e_b[2] > 40       # measured 28x at dt = 0.005 and 60x at 0.0025


def test_single_oscillator_with_initial_velocity_second_order():
    om = 3.0
    errs = []
    for h in (0.02, 0.01, 0.005):
        N = int(round(2 / h))
        U, _ = solve_contact_dynamics(np.eye(1), np.eye(1) * om ** 2, lambda t: np.zeros(1), np.zeros((1, 1)), np.array([-9.0]), h, N,
                                      u0=np.array([1.0]), v0=np.array([0.5]), scheme="bdf2")
        t = N * h
        exact = np.cos(om * t) + 0.5 / om * np.sin(om * t)
        errs.append(abs(U[-1, 0] - exact))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.8), (errs, rates)


# ------------------------------------------------------------------------------- contact conditions, every step
def obstacle_problem(ne=20, N=120):
    K, M, f, X, Y = mem_system(ne)
    n = K.shape[0]
    c = obstacles(X, Y)["bump"]
    return K, M, f, sp.identity(n, format="csr"), c, 0.005, N


def bdf2_residual(M, K, f, U, L, V, h):
    r = 0.0
    for k in range(2, len(U)):
        acc = (9 * U[k] - 12 * U[k - 1] + 3 * U[k - 2]) / (4 * h * h) - (4 * V[k - 1] - V[k - 2]) / (2 * h)
        r = max(r, np.abs(M @ acc + K @ U[k] - f - L[k]).max())
    return r


@pytest.mark.parametrize("method", ["active_set", "lemke", "pdas"])
def test_every_step_satisfies_the_contact_conditions_and_the_bdf2_equation(method):
    K, M, f, B, c, h, N = obstacle_problem()
    U, L, info = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N, scheme="bdf2", method=method, return_info=True)
    assert info.ok and info.failed_step is None and np.all(np.isfinite(U))
    gap = U[1:] - c
    assert gap.min() > -1e-9 and L[1:].min() > -1e-9 and np.abs(L[1:] * gap).max() < 1e-8
    assert (L > 1e-9).sum(1).max() >= 5                                    # contact really happens (many active nodes)
    assert bdf2_residual(M, K, f, U, L, info.V, h) < 1e-9
    assert np.abs(info.V[1] - (U[1] - U[0]) / h).max() < 1e-14


def test_dense_and_sparse_contact_solves_agree_and_the_active_set_is_warm_started():
    K, M, f, B, c, h, N = obstacle_problem()
    Ud, Ld = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N, scheme="bdf2")
    Us, Ls, info = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N, scheme="bdf2", method="pdas", return_info=True)
    Ul, Ll = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N, scheme="bdf2", method="lemke")
    assert np.abs(Ud - Us).max() < 1e-8 and np.abs(Ld - Ls).max() < 1e-6
    assert np.abs(Ud - Ul).max() < 1e-8
    it = np.array(info.iterations)
    assert len(it) == N and it.mean() < 3.0 and it.max() <= 6, it         # warm start: a few iterations per step


def test_sparse_and_dense_inputs_give_the_same_trajectory():
    K, M, f, B, c, h, N = obstacle_problem(ne=20, N=120)
    a, _ = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N, scheme="bdf2")
    b, _ = solve_contact_dynamics(M.toarray(), K.toarray(), lambda t: f, B.toarray(), c, h, N, scheme="bdf2")
    assert np.abs(a - b).max() < 1e-8                  # rounding differences of two factorisations, amplified over 120 steps


def test_consistent_mass_matrix_and_a_time_dependent_load():
    """A non-diagonal M and a load f(t): the BDF2 equation and the contact conditions hold at every step."""
    n = 12
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((n, n))
    K = Q @ Q.T + n * np.eye(n)
    R = rng.standard_normal((n, n))
    M = R @ R.T + n * np.eye(n)
    B = np.eye(n)[:4]
    c = np.full(4, -0.05)
    f0 = rng.standard_normal(n) * 5
    force = lambda t: f0 * np.sin(4 * t)
    h, N = 0.01, 80
    U, L, info = solve_contact_dynamics(M, K, force, B, c, h, N, scheme="bdf2", return_info=True)
    gap = U[1:, :4] - c
    assert gap.min() > -1e-9 and L[1:].min() > -1e-9 and np.abs(L[1:] * gap).max() < 1e-8
    assert (L > 1e-9).any()
    r = 0.0
    for k in range(2, N + 1):
        acc = (9 * U[k] - 12 * U[k - 1] + 3 * U[k - 2]) / (4 * h * h) - (4 * info.V[k - 1] - info.V[k - 2]) / (2 * h)
        r = max(r, np.abs(M @ acc + K @ U[k] - force(k * h) - B.T @ L[k]).max())
    assert r < 1e-9


# ----------------------------------------------------------------------------------------- failures and input
def test_failing_contact_solve_is_reported_with_the_step_and_a_suggestion():
    K, M, f, X, Y = mem_system(6)
    n = K.shape[0]
    B = sp.csr_matrix(np.ones((1, n)) / n)                  # not a node selection
    c = np.array([-1.0])
    U, L, info = solve_contact_dynamics(M, K, lambda t: f, B, c, 0.01, 5, scheme="bdf2", method="pdas", route="reduced",
                                         return_info=True)
    assert not info.ok and info.failed_step == 1 and info.reason == "not_applicable" and info.suggestion == "saddle"
    assert np.all(np.isnan(U[1:])) and np.all(U[0] == 0)
    U2, L2, info2 = solve_contact_dynamics(M, K, lambda t: f, B, c, 0.01, 5, scheme="bdf2", method="pdas", route="reduced",
                                            fallback=True, return_info=True)
    assert info2.ok and np.all(np.isfinite(U2))


def test_input_checks():
    K, M, f, X, Y = mem_system(5)
    n = K.shape[0]
    B, c = np.eye(n), np.full(n, -9.0)
    with pytest.raises(ValueError, match="scheme must be"):
        solve_contact_dynamics(M, K, lambda t: f, B, c, 0.01, 3, scheme="newmark")
    with pytest.raises(ValueError, match="bdf2"):
        solve_contact_dynamics(M, K, lambda t: f, B, c, 0.01, 3, method="pdas")


# ------------------------------------------------------------------------------------------------- impact test
def bar_impact(h, T=1.5):
    """Free-free elastic bar (40 nodes, weak ground springs make K positive definite) moving at unit speed into a rigid wall at
    node 0 (u_0 >= -0.02)."""
    n, dx = 40, 1.0 / 39
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i:i + 2, i:i + 2] += (1 / dx) * np.array([[1, -1], [-1, 1]])
    K += np.eye(n) * 1e-3 / n
    M = np.diag(np.full(n, dx))
    M[0, 0] = M[-1, -1] = dx / 2
    B = np.zeros((1, n))
    B[0, 0] = 1.0
    c = np.array([-0.02])
    v0 = -np.ones(n)
    N = int(round(T / h))
    Ub, Lb, ib = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0, scheme="bdf2", return_info=True)
    U2, L2 = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0)
    V2 = np.zeros_like(U2)
    V2[0] = v0
    V2[1:-1] = (U2[2:] - U2[:-2]) / (2 * h)
    V2[-1] = (U2[-1] - U2[-2]) / h
    energy = lambda U, V: np.array([0.5 * V[k] @ M @ V[k] + 0.5 * U[k] @ K @ U[k] for k in range(len(U))])
    return dict(Ub=Ub, Lb=Lb, Eb=energy(Ub, ib.V), U2=U2, L2=L2, E2=energy(U2, V2), c=c)


@pytest.mark.parametrize("h", [0.004, 0.002])
def test_impact_no_penetration_nonnegative_force_and_recorded_energy_history(h):
    r = bar_impact(h)
    for U, L in ((r["Ub"], r["Lb"]), (r["U2"], r["L2"])):
        assert (U[:, 0] - r["c"][0]).min() > -1e-9 and L.min() > -1e-12 and L.max() > 1.0      # the wall pushes
    E0 = r["Eb"][0]
    # Measured (energy = 1/2 v'Mv + 1/2 u'Ku with the schemes' own velocities): neither scheme conserves energy through the
    # impact. BDF2 gains up to 0.7% above the initial energy during the impact and ends below it; two_step never exceeds it.
    assert r["Eb"].max() < 1.01 * E0 and r["Eb"][-1] < E0 and r["Eb"][-1] > 0.9 * E0
    assert r["E2"].max() < 1.0001 * E0 and r["E2"][-1] < E0 and r["E2"][-1] > 0.85 * E0

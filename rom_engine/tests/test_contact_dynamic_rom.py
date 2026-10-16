# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for `ContactROM.solve_dynamic` (rom_engine.contact_dual_basis): the reduced BDF2 contact loop. Complete bases reproduce
fea_engine's full-order BDF2 contact dynamics; the reduced contact conditions and the reduced BDF2 equation hold at every step; the
cached factorisations equal a plain per-step `solve_reduced_contact` loop; the active-set warm start saves iterations without changing
the answer; POD + NMF reduction accuracy; failures (missing mass matrix, rank-deficient reduced constraints, a failing LCP) are reported."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp

from contact_fixtures import ObstacleMembrane
import rom_engine.contact_dual_basis as cdb
from rom_engine import ContactROM, ContactROMDynamicResult, nonneg_dual_basis, active_snapshot_columns, solve_reduced_contact
from fea_engine.contact_solvers import solve_contact_dynamics

H, N_STEPS = 0.005, 120


@pytest.fixture(scope="module")
def problem():
    mem = ObstacleMembrane(n=18)
    n = mem.n_dof
    M = sp.diags(np.full(n, mem.h ** 2)).tocsr()
    c = mem.obstacle((0.4, 0.15))
    force = lambda t: mem.f
    Uf, Lf = solve_contact_dynamics(M, mem.K, force, mem.B, c, H, N_STEPS, scheme="bdf2")
    assert (Lf > 1e-9).sum(1).max() >= 5          # the full-order trajectory really has contact
    return mem, M, c, force, Uf, Lf


def pod_nmf(Uf, Lf, p, pl):
    Ux = np.linalg.svd(Uf[1:].T, full_matrices=False)[0][:, :p]
    act = active_snapshot_columns(Lf[1:].T)
    return Ux, nonneg_dual_basis(Lf[1:].T[:, act], pl).U


def rel_error(Ua, Ub):
    return 100 * np.sum((Ua - Ub) ** 2) / np.sum(Ub ** 2)


def test_complete_bases_reproduce_the_full_order_bdf2_contact_dynamics(problem):
    mem, M, c, force, Uf, Lf = problem
    n = mem.n_dof
    rom = ContactROM(mem.K, mem.B, np.eye(n), np.eye(n), M=M)
    r = rom.solve_dynamic(force, c, H, N_STEPS)
    assert isinstance(r, ContactROMDynamicResult) and r.ok and r.failed_step is None and r.message is None
    assert np.abs(r.u - Uf).max() < 1e-8 and np.abs(r.lam - Lf).max() < 1e-6
    assert r.u.shape == (N_STEPS + 1, n) and r.lam.shape == (N_STEPS + 1, n) and r.gap.shape == (N_STEPS + 1, n)
    assert np.abs(r.u_r - r.u).max() == 0                                    # U = I: reduced equals full


def test_initial_state_is_projected_and_used(problem):
    mem, M, c, force, Uf, Lf = problem
    n = mem.n_dof
    rng = np.random.default_rng(0)
    u0 = np.maximum(0.02 * rng.standard_normal(n), 0) + c + 0.05                    # above the obstacle
    v0 = 0.1 * rng.standard_normal(n)
    ref, _ = solve_contact_dynamics(M, mem.K, force, mem.B, c, H, 40, u0=u0, v0=v0, scheme="bdf2")
    r = ContactROM(mem.K, mem.B, np.eye(n), np.eye(n), M=M).solve_dynamic(force, c, H, 40, u0=u0, v0=v0)
    assert r.ok and np.abs(r.u - ref).max() < 1e-8 and np.abs(r.u[0] - u0).max() < 1e-14


def test_reduced_conditions_and_the_reduced_bdf2_equation_hold_at_every_step(problem):
    mem, M, c, force, Uf, Lf = problem
    U, Ul = pod_nmf(Uf, Lf, 12, 12)
    rom = ContactROM(mem.K, mem.B, U, Ul, M=M)
    r = rom.solve_dynamic(force, c, H, N_STEPS)
    assert r.ok
    cr = Ul.T @ c
    gap = (rom.B_r @ r.u_r.T).T - cr
    assert gap[1:].min() > -1e-8 and r.lam_r.min() >= 0 and np.abs(r.lam_r[1:] * gap[1:]).max() < 1e-6
    assert r.kkt_residual.shape == (N_STEPS,) and r.kkt_residual.max() < 1e-6
    Mr, Ar, Br = rom.M_r, rom.A_r, rom.B_r
    res = 0.0
    for k in range(2, N_STEPS + 1):
        acc = (9 * r.u_r[k] - 12 * r.u_r[k - 1] + 3 * r.u_r[k - 2]) / (4 * H * H) - (4 * r.v_r[k - 1] - r.v_r[k - 2]) / (2 * H)
        res = max(res, np.abs(Mr @ acc + Ar @ r.u_r[k] - U.T @ force(k * H) - Br.T @ r.lam_r[k]).max())
    assert res < 1e-8


def test_cached_factorisations_equal_a_plain_per_step_reduced_solve(problem):
    """The prototype loop of the validation project: solve_reduced_contact on the reduced matrices at every step."""
    mem, M, c, force, Uf, Lf = problem
    U, Ul = pod_nmf(Uf, Lf, 12, 12)
    rom = ContactROM(mem.K, mem.B, U, Ul, M=M)
    r = rom.solve_dynamic(force, c, H, N_STEPS)
    Mr, Ar, Br, cr, fr = rom.M_r, rom.A_r, rom.B_r, Ul.T @ c, lambda t: U.T @ force(t)
    A1, A2 = Mr / H ** 2 + Ar, 9 * Mr / (4 * H ** 2) + Ar
    ur, vr = np.zeros((N_STEPS + 1, 12)), np.zeros((N_STEPS + 1, 12))
    for n in range(1, N_STEPS + 1):
        if n == 1:
            A, b = A1, fr(n * H) + Mr @ (ur[0] / H ** 2 + vr[0] / H)
        else:
            A, b = A2, fr(n * H) + Mr @ ((3 / (4 * H ** 2)) * (4 * ur[n - 1] - ur[n - 2]) + (4 * vr[n - 1] - vr[n - 2]) / (2 * H))
        u, lam, _, ok = solve_reduced_contact(A, b, Br, cr)
        ur[n] = u
        vr[n] = (u - ur[n - 1]) / H if n == 1 else (3 * u - 4 * ur[n - 1] + ur[n - 2]) / (2 * H)
    assert np.abs(r.u_r - ur).max() < 1e-9


def test_warm_start_saves_iterations_and_does_not_change_the_answer(problem):
    mem, M, c, force, Uf, Lf = problem
    n = mem.n_dof
    rom = ContactROM(mem.K, mem.B, np.eye(n), np.eye(n), M=M)
    warm = rom.solve_dynamic(force, c, H, N_STEPS)
    cold = rom.solve_dynamic(force, c, H, N_STEPS, warm_start=False)
    assert sum(warm.iterations) < sum(cold.iterations) and len(warm.iterations) == N_STEPS
    assert np.abs(warm.u_r - cold.u_r).max() < 1e-9
    rl = ContactROM(mem.K, mem.B, np.eye(n), np.eye(n), method="lemke", M=M).solve_dynamic(force, c, H, N_STEPS)
    assert rl.ok and rl.iterations == [] and np.abs(rl.u_r - warm.u_r).max() < 1e-7


@pytest.mark.parametrize("p,pl,bound", [(10, 10, 1.5), (30, 30, 0.5), (40, 40, 0.05)])
def test_pod_nmf_reduction_accuracy_in_reproduction_mode(problem, p, pl, bound):
    """Relative error (paper Eq. 17, in %) measured on this model (18 x 18 membrane, 120 steps, training = the test run):
    0.81% / 0.20% / 0.023% for (10, 10) / (30, 30) / (40, 40). Positivity of the lifted multipliers holds by construction."""
    mem, M, c, force, Uf, Lf = problem
    U, Ul = pod_nmf(Uf, Lf, p, pl)
    r = ContactROM(mem.K, mem.B, U, Ul, M=M).solve_dynamic(force, c, H, N_STEPS)
    assert r.ok and rel_error(r.u, Uf) < bound and r.lam.min() >= 0
    assert (r.lam > 1e-9).any()


def kron_membrane(ne, load=-10.0):
    """Q4 Laplace membrane on a uniform grid (Kronecker form), lumped mass h^2 I; obstacle: two Gaussian bumps on a floor at -1."""
    n, h = ne - 1, 1.0 / ne
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    Mm = sp.diags([1, 4, 1], [-1, 0, 1], shape=(n, n)) / 6.0
    K = (sp.kron(T / h ** 2, Mm) + sp.kron(Mm, T / h ** 2)).tocsr() * h ** 2
    x = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = (-1 + 0.4 * np.exp(-200 * ((X - 0.45) ** 2 + (Y - 0.5) ** 2)) + 0.4 * np.exp(-355.56 * ((X - 0.7) ** 2 + (Y - 0.5) ** 2))).ravel()
    return K, sp.diags(np.full(n * n, h * h)).tocsr(), load * h * h * np.ones(n * n), c


def test_rank_deficient_reduced_constraints_give_an_accurate_run_or_a_reported_failure_never_a_wrong_answer():
    """A second model (20 x 20 mesh, 44 steps with contact). With p = p_lam = 40 the reduced constraint matrix has rank 24 and condition
    number 9e9; in the development sandbox the reduced LCP then returned a wrong solution (error above 1000%) which the loop reported as
    a failure, while a run whose right-hand side differed by 3e-18 gave an accurate answer (1e-8%). Both outcomes are acceptable, a wrong
    answer reported as ok is not. (30, 30) (rank 23) and (40, 20) are accurate."""
    K, M, f, c = kron_membrane(20)
    n = K.shape[0]
    B = sp.identity(n, format="csr")
    Uf, Lf = solve_contact_dynamics(M, K, lambda t: f, B, c, H, N_STEPS, scheme="bdf2")
    for p, pl in ((30, 30), (40, 20), (40, 40)):
        U, Ul = pod_nmf(Uf, Lf, p, pl)
        rom = ContactROM(K, B, U, Ul, M=M)
        r = rom.solve_dynamic(lambda t: f, c, H, N_STEPS)
        if r.ok:
            assert rel_error(r.u, Uf) < 1e-2, (p, pl)
        else:
            assert (p, pl) == (40, 40) and "inf_sup_report" in r.message
            assert np.all(np.isnan(r.u_r[r.failed_step:])) and np.all(np.isfinite(r.u_r[:r.failed_step]))
    assert not rom.inf_sup_report()["full_row_rank"]


def test_wrong_lcp_solution_with_a_success_flag_is_caught_by_the_condition_check(problem, monkeypatch):
    """The guard of the loop: a solver that returns ok=True with a solution violating the reduced contact conditions (here an
    all-ones multiplier vector) is reported as a failure that points to the rank condition."""
    mem, M, c, force, Uf, Lf = problem
    U, Ul = pod_nmf(Uf, Lf, 12, 12)
    rom = ContactROM(mem.K, mem.B, U, Ul, M=M)
    real = cdb.lcp_active_set
    calls = {"k": 0}

    def wrong(A, q, **kw):
        calls["k"] += 1
        if calls["k"] == 30:
            return np.ones(len(q)), True, 1
        return real(A, q, **kw)

    monkeypatch.setattr(cdb, "lcp_active_set", wrong)
    r = rom.solve_dynamic(force, c, H, N_STEPS)
    assert not r.ok and r.failed_step == 30 and "inf_sup_report" in r.message and "rank" in r.message
    assert np.isnan(r.u_r[30:]).all() and np.isfinite(r.u_r[:30]).all()


def test_a_failing_lcp_is_reported_with_the_step(problem, monkeypatch):
    mem, M, c, force, Uf, Lf = problem
    n = mem.n_dof
    rom = ContactROM(mem.K, mem.B, np.eye(n), np.eye(n), M=M)
    calls = {"k": 0}
    real = cdb.lcp_active_set

    def failing(A, q, **kw):
        calls["k"] += 1
        z, ok, it = real(A, q, **kw)
        return z, (ok and calls["k"] < 7), it

    monkeypatch.setattr(cdb, "lcp_active_set", failing)
    r = rom.solve_dynamic(force, c, H, 20)
    assert not r.ok and r.failed_step == 7 and "method='lemke'" in r.message and np.isnan(r.u_r[7:]).all()


def test_lift_false_skips_the_full_arrays_and_inputs_are_checked(problem):
    mem, M, c, force, Uf, Lf = problem
    U, Ul = pod_nmf(Uf, Lf, 8, 8)
    r = ContactROM(mem.K, mem.B, U, Ul, M=M).solve_dynamic(force, c, H, 10, lift=False)
    assert r.u is None and r.lam is None and r.gap is None and r.u_r.shape == (11, 8)
    with pytest.raises(ValueError, match="needs the mass matrix"):
        ContactROM(mem.K, mem.B, U, Ul).solve_dynamic(force, c, H, 10)
    with pytest.raises(ValueError, match="scheme must be"):
        ContactROM(mem.K, mem.B, U, Ul, M=M).solve_dynamic(force, c, H, 10, scheme="newmark")


def test_existing_static_interface_is_unchanged_and_a_diagonal_mass_may_be_a_vector(problem):
    mem, M, c, force, Uf, Lf = problem
    U, Ul = pod_nmf(Uf, Lf, 8, 8)
    a = ContactROM(mem.K, mem.B, U, Ul)
    assert a.M_r is None and a.solve(mem.f, c).ok
    b = ContactROM(mem.K, mem.B, U, Ul, M=np.full(mem.n_dof, mem.h ** 2))
    cc = ContactROM(mem.K, mem.B, U, Ul, M=M)
    assert np.abs(b.M_r - cc.M_r).max() < 1e-15

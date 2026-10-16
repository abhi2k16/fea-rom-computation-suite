# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the contact-condition error indicator and the greedy snapshot selection."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from contact_fixtures import ObstacleMembrane
from rom_engine.contact_dual_basis import ContactROM
from rom_engine.contact_greedy import contact_condition_indicator, greedy_contact_sampling

CANDS = [(a, b) for a in np.linspace(0.3, 0.6, 6) for b in np.linspace(0.0, 0.1, 6)]
TEST = [(0.33, 0.02), (0.41, 0.07), (0.47, 0.04), (0.58, 0.09), (0.36, 0.095), (0.55, 0.01)]


@pytest.fixture(scope="module")
def setup():
    m = ObstacleMembrane(n=14)
    ref = [m.solve(g) for g in TEST]

    def make_rom(U, L):
        Q, _ = np.linalg.qr(U)
        Ul = L / np.maximum(np.linalg.norm(L, axis=0), 1e-300)        # positive combination of multiplier snapshots
        rom = ContactROM(m.K, m.B, Q, Ul)
        return lambda p: rom.solve(m.f, m.obstacle(p)).u

    def test_error(rom):
        return np.mean([np.linalg.norm(rom(g) - r[0]) / np.linalg.norm(r[0]) for g, r in zip(TEST, ref)])

    return m, make_rom, test_error


def test_indicator_vanishes_on_the_exact_solution_and_detects_violations(setup):
    m, _, _ = setup
    u, lam, c = m.solve((0.45, 0.05))
    for w in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
        assert contact_condition_indicator(m.K, m.B, m.f, c, u, w) < 1e-18
    assert contact_condition_indicator(m.K, m.B, m.f, c, u - 0.05)  > 0          # penetrates the obstacle
    u_deep = 2.0 * spsolve(m.K.tocsc(), m.f)                          # deeper than the free solution: negative force
    assert contact_condition_indicator(m.K, m.B, m.f, c, u_deep, (0, 0, 1)) > 0


def test_greedy_selection_beats_random_sampling_at_equal_size(setup):
    m, make_rom, test_error = setup
    ind = lambda p, u: contact_condition_indicator(m.K, m.B, m.f, m.obstacle(p), u, (1, 0, 0))
    rng = np.random.default_rng(0)
    for n_snap in (4, 8):
        rom, hist = greedy_contact_sampling(CANDS, lambda p: m.solve(p)[:2], make_rom, ind, n_snap)
        assert hist.n_snapshots == n_snap and len(set(hist.selected)) == n_snap
        err_greedy = test_error(rom)
        errs = []
        for _ in range(8):
            idx = rng.choice(len(CANDS), n_snap, replace=False)
            sols = [m.solve(CANDS[i]) for i in idx]
            errs.append(test_error(make_rom(np.array([s[0] for s in sols]).T, np.array([s[1] for s in sols]).T)))
        assert err_greedy < np.median(errs)


def test_greedy_stops_at_the_tolerance_and_records_the_indicator(setup):
    m, make_rom, _ = setup
    ind = lambda p, u: contact_condition_indicator(m.K, m.B, m.f, m.obstacle(p), u, (1, 0, 0))
    rom, hist = greedy_contact_sampling(CANDS, lambda p: m.solve(p)[:2], make_rom, ind, max_snapshots=30, tol=1e-6)
    assert hist.converged and hist.n_snapshots < 30
    assert hist.indicator_max[-1] <= 1e-6
    assert hist.indicator_max[0] > hist.indicator_max[-1]


# ======================================================================= trajectories, schedule, relative stopping rule
from rom_engine import trajectory_indicator, linear_basis_schedule                      # noqa: E402
from rom_engine.contact_greedy import GreedyContactHistory                              # noqa: E402


def fake_problem(n_cand=10):
    """Deterministic stand-ins: the 'ROM' of round r returns [r, p]; the indicator of candidate p falls by a factor 10 per round."""
    cands = list(range(n_cand))
    calls = {"make": [], "fom": []}

    def fom(p):
        calls["fom"].append(p)
        return np.full(4, float(p)), np.full(3, float(p))

    def make(U, L, *rest):
        calls["make"].append((U.shape, L.shape, rest))
        r = len(calls["make"])
        return lambda p: np.array([float(r), float(p)])

    ind = lambda p, u: 10.0 ** (-u[0]) * (1.0 + p / 100.0)
    return cands, fom, make, ind, calls


def test_default_call_is_unchanged_make_rom_gets_two_arguments_and_vector_snapshots_stack_as_columns():
    cands, fom, make, ind, calls = fake_problem()
    rom, hist = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=4)
    assert [c[2] for c in calls["make"]] == [(), (), (), ()]
    assert [c[0] for c in calls["make"]] == [(4, 1), (4, 2), (4, 3), (4, 4)] and [c[1] for c in calls["make"]] == [(3, 1), (3, 2), (3, 3), (3, 4)]
    assert hist.n_snapshots == 4 and hist.n_columns == 4 and hist.stop_reason == "max_snapshots" and not hist.converged
    assert isinstance(hist, GreedyContactHistory)


def test_pass_round_gives_the_round_number_to_make_rom():
    cands, fom, make, ind, calls = fake_problem()
    greedy_contact_sampling(cands, fom, make, ind, max_snapshots=5, pass_round=True)
    assert [c[2] for c in calls["make"]] == [(1,), (2,), (3,), (4,), (5,)]


def test_trajectory_snapshots_are_concatenated_column_wise_in_selection_order():
    T = {0: 3, 1: 5, 2: 4}
    cands = [0, 1, 2]
    seen = []

    def fom(p):
        return np.full((6, T[p]), float(p)), np.full((2, T[p]), 10.0 + p)

    def make(U, L, it):
        seen.append((U.shape, L.shape, U[0].copy(), it))
        return lambda p: np.full((6, 4), float(p))

    ind = lambda p, u: abs(p - 1) + 0.5                  # candidate 2 has the larger indicator after round 1 (1.5 against 0.5)
    rom, hist = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=3, pass_round=True)
    assert hist.selected == [0, 2, 1]
    assert [s[0] for s in seen] == [(6, 3), (6, 7), (6, 12)] and [s[1] for s in seen] == [(2, 3), (2, 7), (2, 12)]
    assert list(seen[2][2]) == [0.0] * 3 + [2.0] * 4 + [1.0] * 5                         # columns in selection order
    assert hist.n_columns == 12 and hist.n_snapshots == 3 and hist.stop_reason == "exhausted" and hist.converged


def test_trajectory_with_inconsistent_column_counts_is_refused():
    with pytest.raises(ValueError, match="displacement columns"):
        greedy_contact_sampling([0, 1], lambda p: (np.zeros((3, 4)), np.zeros((2, 5))), lambda U, L: (lambda p: np.zeros(3)),
                                lambda p, u: 0.0, 2)
    with pytest.raises(ValueError, match="shape"):
        greedy_contact_sampling([0, 1], lambda p: (np.zeros((3, 4, 2)), np.zeros((2, 4))), lambda U, L: (lambda p: np.zeros(3)),
                                lambda p, u: 0.0, 2)


def test_trajectory_indicator_sums_the_single_step_indicator_over_the_time_steps():
    step = lambda p, u: float(np.sum(np.minimum(u, 0.0) ** 2)) + p
    ind = trajectory_indicator(step)
    u = np.array([[1.0, -2.0, -1.0], [0.5, 0.0, -3.0]])                                   # columns: [1, .5], [-2, 0], [-1, -3]
    assert ind(7.0, u) == pytest.approx(sum(step(7.0, u[:, k]) for k in range(3)))
    assert ind(7.0, u[:, 1]) == pytest.approx(step(7.0, u[:, 1]))                         # 1-D passes through
    K = np.eye(2)
    ci = trajectory_indicator(lambda p, v: contact_condition_indicator(K, np.eye(2), np.zeros(2), np.zeros(2), v))
    assert ci(None, u) == pytest.approx(4 + 1 + 9)                                        # sum of [x]_-^2 over all entries


def test_relative_stopping_rule_compares_with_the_indicator_after_the_first_round():
    cands, fom, make, ind, calls = fake_problem()
    rom, hist = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=10, rel_tol=0.05)
    I = hist.indicator_max
    assert hist.stop_reason == "rel_tol" and hist.converged
    assert hist.n_snapshots == 3 and I[-1] <= 0.05 * I[0] and I[-2] > 0.05 * I[0]         # stops at the first round that satisfies it
    # never at the first round (the reference is the first round's own value), even for a rel_tol above 1
    _, h2 = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=10, rel_tol=1.5)
    assert h2.n_snapshots == 2 and h2.stop_reason == "rel_tol"


def test_absolute_and_relative_rules_stop_at_whichever_comes_first_and_the_reason_is_recorded():
    cands, fom, make, ind, calls = fake_problem()
    _, h_abs = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=10, tol=0.05, rel_tol=1e-6)
    assert h_abs.stop_reason == "tol" and h_abs.n_snapshots == 2                         # 10^-2 * (1 + p/100) <= 0.05 at round 2
    _, h_rel = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=10, tol=1e-9, rel_tol=0.5)
    assert h_rel.stop_reason == "rel_tol" and h_rel.n_snapshots == 2
    _, h_none = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=3)
    assert h_none.stop_reason == "max_snapshots"
    with pytest.raises(ValueError, match="rel_tol"):
        greedy_contact_sampling(cands, fom, make, ind, max_snapshots=3, rel_tol=-0.1)


def test_linear_basis_schedule_reproduces_the_validation_report_values_and_validates_input():
    p = linear_basis_schedule(2, 100, 10)
    assert [p(n) for n in range(1, 11)] == [2, 12, 23, 34, 45, 56, 67, 78, 89, 100]
    assert p(15) == 100 and p(1) == 2
    q = linear_basis_schedule(5, 5, 4)
    assert [q(n) for n in (1, 2, 3, 4)] == [5, 5, 5, 5]
    with pytest.raises(ValueError):
        linear_basis_schedule(10, 5, 4)
    with pytest.raises(ValueError):
        linear_basis_schedule(2, 100, 1)


def test_schedule_inside_make_rom_with_pass_round():
    cands, fom, _, ind, _ = fake_problem()
    sizes = []
    sched = linear_basis_schedule(2, 8, 4)

    def make(U, L, it):
        sizes.append(min(sched(it), U.shape[1]))
        r = it
        return lambda p: np.array([float(r), float(p)])

    greedy_contact_sampling(cands, fom, make, ind, max_snapshots=6, pass_round=True)
    assert [sched(i) for i in range(1, 7)] == [2, 4, 6, 8, 8, 8] and sizes == [min(sched(i), i) for i in range(1, 7)] == [1, 2, 3, 4, 5, 6]


# ------------------------------------------------------------------ dynamic contact: trajectories, schedule, relative rule together
def test_greedy_on_a_dynamic_contact_problem_with_trajectories_a_basis_schedule_and_the_relative_rule():
    """Parametric obstacle membrane (14 x 14, BDF2, 80 steps), 12 candidate obstacles. Full-order solves return trajectories; the ROM is
    POD (primal) + NMF (dual) with sizes from a linear schedule in the round number, run by `ContactROM.solve_dynamic`; the indicator is
    the sum over the time steps of the non-penetration violation. Measured: the worst error over all candidates (paper Eq. 17, %) falls
    from 6.7% after one round to 4.0% after five."""
    import scipy.sparse as sp
    from fea_engine.contact_solvers import solve_contact_dynamics
    from rom_engine import nonneg_dual_basis, active_snapshot_columns
    m = ObstacleMembrane(n=14)
    n = m.n_dof
    M = sp.diags(np.full(n, m.h ** 2)).tocsr()
    h, NS = 0.005, 80
    cands = [(a, b) for a in (0.3, 0.4, 0.5, 0.6) for b in (0.05, 0.1, 0.15)]

    def fom(p):
        U, L = solve_contact_dynamics(M, m.K, lambda t: m.f, m.B, m.obstacle(p), h, NS, scheme="bdf2")
        return U[1:].T, L[1:].T

    sched = linear_basis_schedule(4, 16, 4)

    def make(U, L, it):
        ps = sched(it)
        Ux = np.linalg.svd(U, full_matrices=False)[0][:, :min(ps, U.shape[1])]
        act = active_snapshot_columns(L)
        Ul = nonneg_dual_basis(L[:, act], min(ps, len(act))).U
        rom = ContactROM(m.K, m.B, Ux, Ul, M=M)

        def run(p):
            res = rom.solve_dynamic(lambda t: m.f, m.obstacle(p), h, NS)
            return res.u[1:].T if res.ok else np.full((n, NS), np.nan)
        return run

    ind = trajectory_indicator(lambda p, u: contact_condition_indicator(m.K, m.B, m.f, m.obstacle(p), u, (1, 0, 0)))
    rom5, h5 = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=5, pass_round=True)
    rom1, h1 = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=1, pass_round=True)
    assert h5.n_snapshots == 5 and len(set(h5.selected)) == 5 and h5.n_columns == 5 * NS and h5.stop_reason == "max_snapshots"
    assert all(np.isfinite(v) and v > 0 for v in h5.indicator_max)
    ref = {p: fom(p)[0] for p in cands}
    worst = lambda rom: max(100 * np.sum((rom(p) - ref[p]) ** 2) / np.sum(ref[p] ** 2) for p in cands)
    assert worst(rom5) < worst(rom1) and worst(rom5) < 6.0                               # measured 3.97% against 6.67%
    _, hr = greedy_contact_sampling(cands, fom, make, ind, max_snapshots=10, pass_round=True, rel_tol=0.1)
    assert hr.stop_reason == "rel_tol" and hr.indicator_max[-1] <= 0.1 * hr.indicator_max[0] and hr.n_snapshots < 12

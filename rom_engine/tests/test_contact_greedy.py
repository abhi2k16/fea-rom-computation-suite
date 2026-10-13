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

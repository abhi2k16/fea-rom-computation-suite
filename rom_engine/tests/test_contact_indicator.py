# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for `ContactConditionIndicator` (rom_engine.contact_greedy): the same numbers as the reference function
`contact_condition_indicator` for every structure of B (identity, node selection, full-row-rank sparse, rank-deficient, tall), with
the pseudo-inverse work done once, lazily, and not at all where it is not needed; trajectories; use inside the greedy loop."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp

from contact_fixtures import ObstacleMembrane
from rom_engine import ContactConditionIndicator, contact_condition_indicator, greedy_contact_sampling, ContactROM

N = 60
WEIGHTS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0.5, 2.0, 0.1)]


def spd(n, rng):
    Q = rng.standard_normal((n, n))
    return Q @ Q.T + n * np.eye(n)


def structures(rng):
    n = N
    gen = (sp.random(30, n, density=0.2, random_state=1, format="csr") + sp.csr_matrix(np.eye(30, n))).tocsr()
    return {
        "identity": (sp.identity(n, format="csr"), "identity"),
        "identity dense": (np.eye(n), "identity"),
        "selection": (sp.csr_matrix(np.eye(n)[::2] * np.where(rng.random(30) > 0.5, 2.0, -0.5)[:, None]), "selection"),
        "general full row rank": (gen, "normal_equations"),
        "duplicate rows": (sp.csr_matrix(np.vstack([np.eye(n)[:3], np.eye(n)[:3]])), "dense_pinv"),
        "tall": (sp.csr_matrix(rng.standard_normal((n + 10, n))), "dense_pinv"),
    }


@pytest.mark.parametrize("name", list(structures(np.random.default_rng(0))))
@pytest.mark.parametrize("w", WEIGHTS)
def test_equals_the_reference_function_for_every_structure_of_B(name, w):
    rng = np.random.default_rng(0)
    K, f, u = spd(N, rng), rng.standard_normal(N), rng.standard_normal(N)
    B, method = structures(np.random.default_rng(0))[name]
    c = rng.standard_normal(B.shape[0]) * 0.1
    ind = ContactConditionIndicator(K, B, f, w)
    a, b = ind(u, c), contact_condition_indicator(K, B, f, c, u, w)
    assert a == pytest.approx(b, rel=1e-10, abs=1e-12) and b > 0
    if w[1] or w[2]:
        assert ind.method == method
    else:
        assert ind.method is None                                  # nothing prepared when a2 = a3 = 0


def test_no_least_squares_for_identity_and_selection_and_none_for_the_matrix_free_weights(monkeypatch):
    rng = np.random.default_rng(1)
    K, f, u = spd(N, rng), rng.standard_normal(N), rng.standard_normal(N)

    def boom(*a, **k):
        raise AssertionError("a dense least-squares / pseudo-inverse call was made")

    monkeypatch.setattr(np.linalg, "lstsq", boom)
    monkeypatch.setattr(np.linalg, "pinv", boom)
    for name in ("identity", "identity dense", "selection"):
        B, _ = structures(np.random.default_rng(0))[name]
        c = np.zeros(B.shape[0])
        assert ContactConditionIndicator(K, B, f, (1, 1, 1))(u, c) > 0
    B, _ = structures(np.random.default_rng(0))["general full row rank"]
    assert ContactConditionIndicator(K, B, f, (1, 1, 1))(u, np.zeros(B.shape[0])) > 0            # sparse LU, still no dense solve
    ind = ContactConditionIndicator(K, structures(np.random.default_rng(0))["tall"][0], f, (1, 0, 0))
    assert ind(u, np.zeros(N + 10)) >= 0 and ind.method is None


def test_preparation_happens_once():
    rng = np.random.default_rng(2)
    K, f = spd(N, rng), rng.standard_normal(N)
    B, _ = structures(np.random.default_rng(0))["duplicate rows"]
    ind = ContactConditionIndicator(K, B, f, (0, 1, 1))
    ind(rng.standard_normal(N), np.zeros(6))
    lam1 = ind._lam
    ind(rng.standard_normal(N), np.zeros(6))
    assert ind._lam is lam1 and ind.method == "dense_pinv"


def test_trajectories_equal_the_sum_over_the_columns_and_time_dependent_loads_are_supported():
    rng = np.random.default_rng(3)
    K, u = spd(N, rng), rng.standard_normal((N, 7))
    f = rng.standard_normal(N)
    ft = rng.standard_normal((N, 7))
    for name in ("identity", "selection", "general full row rank", "tall"):
        B, _ = structures(np.random.default_rng(0))[name]
        c = rng.standard_normal(B.shape[0]) * 0.1
        ct = rng.standard_normal((B.shape[0], 7)) * 0.1
        ind = ContactConditionIndicator(K, B, f, (1, 1, 1))
        ref = sum(contact_condition_indicator(K, B, f, c, u[:, k], (1, 1, 1)) for k in range(7))
        assert ind(u, c) == pytest.approx(ref, rel=1e-10)                                      # constant f and c
        ref_t = sum(contact_condition_indicator(K, B, ft[:, k], ct[:, k], u[:, k], (1, 1, 1)) for k in range(7))
        assert ind(u, ct, f=ft) == pytest.approx(ref_t, rel=1e-10)                              # per-step load and offset


def test_rank_deficient_B_gives_the_minimum_norm_answer_of_the_reference():
    rng = np.random.default_rng(4)
    K, f, u = spd(N, rng), rng.standard_normal(N), rng.standard_normal(N)
    B = sp.csr_matrix(np.vstack([np.eye(N)[:5], 2 * np.eye(N)[:5]]))            # rows 5..9 are multiples of rows 0..4
    c = np.zeros(10)
    for w in ((0, 1, 0), (0, 0, 1)):
        assert ContactConditionIndicator(K, B, f, w)(u, c) == pytest.approx(contact_condition_indicator(K, B, f, c, u, w), rel=1e-10)


def test_dense_pinv_is_refused_above_the_memory_limit_with_a_message():
    rng = np.random.default_rng(5)
    K, f = spd(N, rng), rng.standard_normal(N)
    B, _ = structures(np.random.default_rng(0))["tall"]
    ind = ContactConditionIndicator(K, B, f, (0, 1, 0), max_dense_bytes=1000.0)
    with pytest.raises(ValueError, match="max_dense_bytes"):
        ind(rng.standard_normal(N), np.zeros(N + 10))


def test_shape_checks():
    K = np.eye(5)
    with pytest.raises(ValueError, match="shapes"):
        ContactConditionIndicator(K, np.ones((2, 4)), np.zeros(5))
    with pytest.raises(ValueError, match="shapes"):
        ContactConditionIndicator(K, np.ones((2, 5)), np.zeros(4))


def test_inside_the_greedy_loop_the_selection_is_that_of_the_reference_indicator():
    m = ObstacleMembrane(n=14)
    cands = [(a, b) for a in np.linspace(0.3, 0.6, 6) for b in np.linspace(0.0, 0.1, 6)]

    def make(U, L):
        Q, _ = np.linalg.qr(U)
        rom = ContactROM(m.K, m.B, Q, L / np.maximum(np.linalg.norm(L, axis=0), 1e-300))
        return lambda p: rom.solve(m.f, m.obstacle(p)).u

    fom = lambda p: m.solve(p)[:2]
    ref = lambda p, u: contact_condition_indicator(m.K, m.B, m.f, m.obstacle(p), u, (1, 1, 1))
    fast = ContactConditionIndicator(m.K, m.B, m.f, (1, 1, 1)).for_parameter(m.obstacle)
    _, h_ref = greedy_contact_sampling(cands, fom, make, ref, max_snapshots=6)
    _, h_fast = greedy_contact_sampling(cands, fom, make, fast, max_snapshots=6)
    assert h_ref.selected == h_fast.selected
    assert np.allclose(h_ref.indicator_max, h_fast.indicator_max, rtol=1e-9)

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for linear / nonlinear domain partitioning with static condensation."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp

from contact_fixtures import NonlinearChainContact
from fea_engine.contact_solvers import solve_contact_newton
from rom_engine.contact_hyper_reduction import fit_contact_aware_ecsw
from rom_engine.domain_partition import nonlinearity_indicator, select_nonlinear_elements, PartitionedModel
from rom_engine.hyper_reduction import ecsw_training_matrix

E0 = 24                                   # elements below this index are linear


@pytest.fixture(scope="module")
def chain():
    m = NonlinearChainContact(n=60, linear_below=E0)
    states = np.array([m.solve(s, -0.15)[0] for s in (0.5, 1.0, 2.0, 3.0)]).T
    return m, states


def _linear_stiffness(m):
    K = np.zeros((m.n, m.n))
    for e in range(E0):
        g = m.elem_dofs[e]
        K[np.ix_(g, g)] += m.elem_tangent(e, np.zeros(len(g)))
    return sp.csr_matrix(K)


def test_indicator_separates_linear_and_hardening_elements(chain):
    m, states = chain
    kappa = nonlinearity_indicator(m.elem_dofs, m.elem_force, m.elem_tangent, states)
    assert np.all(kappa[:E0] < 1e-12)                       # linear springs: no nonlinear force at all
    # hardening springs carry a cubic force; the few spanning two nodes held on the obstacle have zero elongation
    # in every state, hence zero force and kappa = 0 (the indicator alone cannot see them)
    weak = np.flatnonzero(kappa[E0:] < 1e-6) + E0
    assert len(weak) <= 6 and np.median(kappa[E0:]) > 1e-4
    sel = select_nonlinear_elements(kappa, 1e-6, always=[3])
    assert 3 in sel and set(sel.tolist()).isdisjoint(set(range(E0)) - {3})
    assert set(sel.tolist()) == {3} | (set(range(E0, m.n)) - set(weak.tolist()))
    assert np.all(kappa[weak] == 0.0)


def test_condensed_stiffness_is_the_series_formula(chain):
    m, _ = chain
    pm = PartitionedModel(m.elem_dofs, range(E0, m.n), _linear_stiffness(m), m.n)
    assert pm.interface.tolist() == [E0 - 1]
    assert pm.K_star.shape == (1, 1)
    assert pm.K_star[0, 0] == pytest.approx(m.k / E0)       # E0 springs in series, grounded at node 0
    assert len(pm.interior) == E0 - 1 and len(pm.nl_dofs) == m.n - (E0 - 1)


@pytest.mark.parametrize("params", [(1.0, -0.15), (2.5, -0.2)])
def test_partitioned_contact_solution_equals_the_full_order_solution(chain, params):
    m, _ = chain
    pm = PartitionedModel(m.elem_dofs, range(E0, m.n), _linear_stiffness(m), m.n)
    u_full, lam_full, c, fe = m.solve(*params)
    f_N = pm.condensed_load(fe)
    nl = pm.nl_dofs
    pos = {d: i for i, d in enumerate(nl)}
    nl_elems = list(range(E0, m.n))

    def scatter_force(uN):
        u = np.zeros(m.n)
        u[nl] = uN
        f = np.zeros(m.n)
        for e in nl_elems:
            g = m.elem_dofs[e]
            f[g] += m.elem_force(e, u[g])
        return f[nl]

    def scatter_tangent(uN):
        u = np.zeros(m.n)
        u[nl] = uN
        K = np.zeros((len(nl), len(nl)))
        for e in nl_elems:
            g = m.elem_dofs[e]
            li = [pos[d] for d in g]
            K[np.ix_(li, li)] += m.elem_tangent(e, u[g])
        return K

    B_N = np.zeros((len(m.obs_nodes), len(nl)))
    B_N[np.arange(len(m.obs_nodes)), [pos[d] for d in m.obs_nodes]] = 1.0
    res = solve_contact_newton(lambda uN: pm.residual(uN, scatter_force(uN), f_N),
                               lambda uN: pm.tangent(scatter_tangent(uN)), B_N, c, np.zeros(len(nl)))
    assert res.converged
    u = pm.expand(res.u, fe)
    assert np.allclose(u, u_full, atol=1e-8)
    assert np.allclose(res.lam, lam_full, atol=1e-6)


def test_partitioning_shrinks_the_ecsw_training_problem(chain):
    m, states = chain
    G_all, _ = ecsw_training_matrix(np.linalg.svd(states, full_matrices=False)[0][:, :6], states, m.elem_dofs, m.elem_force)
    pm = PartitionedModel(m.elem_dofs, range(E0, m.n), _linear_stiffness(m), m.n)
    S_N = states[pm.nl_dofs]
    pos = {d: i for i, d in enumerate(pm.nl_dofs)}
    nl_dofs_per_elem = [np.array([pos[d] for d in m.elem_dofs[e]]) for e in range(E0, m.n)]
    V_N = np.linalg.svd(S_N, full_matrices=False)[0][:, :6]
    G_nl, _ = ecsw_training_matrix(V_N, S_N, nl_dofs_per_elem, lambda i, ul: m.elem_force(E0 + i, ul))
    assert G_nl.shape[1] == m.n - E0 < G_all.shape[1] == m.n          # fewer elements to train over
    assert G_nl.shape[0] == G_all.shape[0]


def test_loads_on_the_linear_interior_are_condensed_correctly(chain):
    m, _ = chain
    pm = PartitionedModel(m.elem_dofs, range(E0, m.n), _linear_stiffness(m), m.n)
    f = np.zeros(m.n)
    f[5] = 1.0                                              # a point load inside the linear zone
    fN = pm.condensed_load(f)
    # DOF 5 is node 6 of a chain of E0 = 24 equal springs, grounded at node 0 with the interface node 24 held fixed:
    # the interface carries the fraction 6/24 of the load, and node 6 deflects by f * 6 * (24 - 6) / (24 k)
    assert fN[pm.iface_local[0]] == pytest.approx(6 / E0)
    assert pm.expand(np.zeros(len(pm.nl_dofs)), f)[5] == pytest.approx(6 * (E0 - 6) / (E0 * m.k))

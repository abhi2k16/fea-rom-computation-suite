# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for contact-aware hyper-reduction: pinned elements, constraint completion, force reconstruction, the rank
(inf-sup) check and the reduced contact Newton solver."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from contact_fixtures import NonlinearChainContact
from rom_engine.contact_dual_basis import nonneg_dual_basis
from rom_engine.contact_hyper_reduction import (ecsw_weights_with_required, fit_contact_aware_ecsw,
                                                complete_constraint_elements, reconstruct_contact_forces,
                                                rank_deficiency, enrich_basis_for_constraints,
                                                solve_reduced_contact_newton)
from rom_engine.hyper_reduction import ECSW, ecsw_training_matrix


@pytest.fixture(scope="module")
def chain():
    m = NonlinearChainContact()
    train = [(s, g) for s in (0.5, 1, 1.5, 2, 2.5, 3) for g in (-0.08, -0.15, -0.25)]
    sols = [m.solve(*p) for p in train]
    Xu = np.array([s[0] for s in sols]).T
    Xl = np.array([s[1] for s in sols]).T
    U = np.linalg.svd(Xu, full_matrices=False)[0][:, :8]
    req = [e for e, g in enumerate(m.elem_dofs) if set(g.tolist()) & set(m.obs_nodes.tolist())]
    return m, Xu, Xl, U, req


def test_required_elements_are_pinned_at_weight_one(chain):
    m, Xu, _, U, req = chain
    G, b = ecsw_training_matrix(U, Xu, m.elem_dofs, m.elem_force)
    w, info = ecsw_weights_with_required(G, b, req, tol=1e-6)
    assert np.all(w[req] == 1.0)
    assert info["n_selected"] >= len(req) and info["residual"] < 1e-5
    w0, i0 = ecsw_weights_with_required(G, b, [], tol=1e-6)         # no required set = plain ECSW
    assert i0["n_selected"] < info["n_selected"]
    ec = fit_contact_aware_ecsw(U, Xu, m.elem_dofs, m.elem_force, req, m.elem_tangent, tol=1e-6)
    assert set(req) <= set(ec.elements.tolist())


def test_reduced_force_matches_the_full_reduced_force_at_held_out_states(chain):
    m, Xu, _, U, req = chain
    ec = fit_contact_aware_ecsw(U, Xu, m.elem_dofs, m.elem_force, req, m.elem_tangent, tol=1e-6)
    u_test = m.solve(1.25, -0.12)[0]
    q = U.T @ u_test
    ref = U.T @ m.internal_force(U @ q)
    assert np.linalg.norm(ec.reduced_force(q) - ref) / np.linalg.norm(ref) < 1e-5


def test_complete_constraint_elements():
    elements, computable = complete_constraint_elements([2, 7], [[1, 2, 3], [4, 5], [7, 8], [9]])
    assert elements.tolist() == [1, 2, 3, 7, 8]
    assert computable.tolist() == [0, 2]


@pytest.mark.parametrize("params", [(1.25, -0.12), (2.2, -0.2), (0.8, -0.1)])
def test_reduced_contact_newton_reproduces_the_full_order_solution(chain, params):
    m, Xu, Xl, U, req = chain
    ec = fit_contact_aware_ecsw(U, Xu, m.elem_dofs, m.elem_force, req, m.elem_tangent, tol=1e-6)
    ec_plain = ECSW.fit(U, Xu, m.elem_dofs, m.elem_force, m.elem_tangent, tol=1e-6)
    nmf = nonneg_dual_basis(Xl, 8)
    u, lam, c, fe = m.solve(*params)
    for e in (ec, ec_plain):
        for U_lam in (None, nmf.U):
            r = solve_reduced_contact_newton(e, fe, U, m.B, c, U_lam=U_lam)
            assert r.converged
            assert np.linalg.norm(r.u - u) / np.linalg.norm(u) < 1e-5
            assert (m.B @ r.u - c).min() >= -1e-6
            assert np.linalg.norm(r.lam.sum() - lam.sum()) / lam.sum() < 1e-3
            if U_lam is not None:
                assert r.lam.min() >= 0.0


def test_reconstruct_contact_forces_is_nonnegative_and_matches_inside_values(chain):
    m, _, Xl, _, _ = chain
    u, lam, c, fe = m.solve(1.25, -0.12)
    gap = m.B @ u - c
    active = np.flatnonzero(lam > 1e-9)
    inside = active[:8]                                              # the reduced model resolves only part of the zone
    rec = reconstruct_contact_forces(Xl, inside, lam[inside], gap)
    assert rec.min() >= 0.0 and rec.shape == lam.shape
    assert np.linalg.norm(rec[inside] - lam[inside]) / np.linalg.norm(lam[inside]) < 0.05
    assert abs(rec.sum() - lam.sum()) / lam.sum() < 0.1            # total contact force recovered outside the region too
    assert np.max(rec[gap > 1e-6]) < 0.05 * lam.max() if np.any(gap > 1e-6) else True


def test_rank_deficiency_and_enrichment(chain):
    m, Xu, _, _, _ = chain
    V = np.linalg.svd(Xu, full_matrices=False)[0][:, :3]
    d = rank_deficiency(m.B, V)
    assert m.B.shape[0] - 3 <= d < m.B.shape[0]       # B V has at most 3 independent rows
    V2, added = enrich_basis_for_constraints(V, m.B)
    assert rank_deficiency(m.B, V2) == 0
    assert len(added) == d and set(added.tolist()) <= set(m.obs_nodes.tolist())
    assert np.allclose(V2.T @ V2, np.eye(V2.shape[1]), atol=1e-10)
    assert np.allclose(V2[:, :3], V)
    V3, none = enrich_basis_for_constraints(V2, m.B)                  # already full rank: unchanged
    assert none.size == 0 and V3.shape == V2.shape

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the dual mortar coupling of non-matching 2-D meshes (tied and unilateral)."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fea_engine.contact_solvers import solve_contact_static, contact_complementarity_residual
from fea_engine.elements import Quad4PlaneStress
from fea_engine.material import D_plane_stress, Material
from fea_engine.mesh import rectangle_mesh
from fea_engine.mortar import MortarTie2D, dual_mortar_operator
from fea_engine.solver import FESystem


def test_matching_meshes_give_identity():
    y = np.linspace(0, 1, 5)
    X = np.column_stack([np.ones(5), y])
    P, D, M = dual_mortar_operator(X, X)
    assert np.allclose(P, np.eye(5))
    assert np.allclose(D, [0.125, 0.25, 0.25, 0.25, 0.125])


@pytest.mark.parametrize("ns,nm", [(4, 6), (7, 3), (5, 8)])
def test_partition_of_unity_and_affine_reproduction_on_nonmatching_nodes(ns, nm):
    rng = np.random.default_rng(ns * 10 + nm)
    ys = np.sort(np.concatenate([[0, 1], rng.uniform(0.05, 0.95, ns - 2)]))
    ym = np.sort(np.concatenate([[0, 1], rng.uniform(0.05, 0.95, nm - 2)]))
    Xs = np.column_stack([np.full(ns, 1.0), ys])
    Xm = np.column_stack([np.full(nm, 1.0), ym])
    P, D, M = dual_mortar_operator(Xs, Xm)
    assert np.allclose(P.sum(1), 1.0)
    assert np.allclose(P @ ym, ys, atol=1e-12)               # linear field reproduced exactly
    assert np.allclose(P @ np.ones(nm), 1.0)
    # node ordering must not matter
    perm_s, perm_m = rng.permutation(ns), rng.permutation(nm)
    P2, _, _ = dual_mortar_operator(Xs[perm_s], Xm[perm_m])
    assert np.allclose(P2, P[np.ix_(perm_s, perm_m)])


def test_errors_for_uncovered_or_curved_interfaces():
    Xs = np.column_stack([np.ones(3), [0, 0.5, 1.0]])
    with pytest.raises(ValueError):
        dual_mortar_operator(Xs, np.column_stack([np.ones(3), [0.1, 0.5, 1.0]]))
    with pytest.raises(ValueError):
        dual_mortar_operator(Xs, np.array([[1.0, 0], [1.2, 0.5], [1.0, 1.0]]))


def _two_blocks(ny_a, ny_b, nx=3):
    ma = rectangle_mesh(1.0, 1.0, nx, ny_a)                          # x in [0, 1]
    mb = rectangle_mesh(1.0, 1.0, nx, ny_b, x0=1.0)                  # x in [1, 2]
    return ma, mb


def _assemble(ma, mb, nu=0.3):
    D = D_plane_stress(Material(E=1.0e3, nu=nu))
    Ks, Xs = [], []
    for m in (ma, mb):
        fes = FESystem(m, Quad4PlaneStress(), sparse=True)
        fes.assemble_stiffness(D, thickness=1.0)
        Ks.append(sp.csr_matrix(fes.K))
        Xs.append(np.asarray(m.nodes, float))
    K = sp.block_diag(Ks, format="csr")
    X = np.vstack(Xs)
    return K, X, len(Xs[0])


def test_patch_test_uniform_strain_across_a_nonmatching_interface():
    ma, mb = _two_blocks(3, 5)
    K, X, na = _assemble(ma, mb)
    n_nodes = len(X)
    slave = [i for i in range(na, n_nodes) if abs(X[i, 0] - 1.0) < 1e-9]        # block B's left edge
    master = [i for i in range(na) if abs(X[i, 0] - 1.0) < 1e-9]               # block A's right edge
    tie = MortarTie2D(slave, master, X, 2 * n_nodes)
    nu, eps = 0.3, 1e-3
    ux_end = eps * 2.0
    left = [i for i in range(n_nodes) if abs(X[i, 0]) < 1e-9]
    right = [i for i in range(n_nodes) if abs(X[i, 0] - 2.0) < 1e-9]
    fixed = {2 * i: 0.0 for i in left}
    fixed.update({2 * i: ux_end for i in right})
    fixed[2 * left[0] + 1] = 0.0 if abs(X[left[0], 1]) < 1e-9 else None
    fixed = {k: v for k, v in fixed.items() if v is not None}
    Kc, _ = tie.condense(K, np.zeros(2 * n_nodes))
    ind = tie.independent
    fixed_c = {int(np.searchsorted(ind, d)): v for d, v in fixed.items() if d in set(ind.tolist())}
    free = np.setdiff1d(np.arange(len(ind)), list(fixed_c))
    uc = np.zeros(len(ind))
    for k, v in fixed_c.items():
        uc[k] = v
    Kc = sp.csr_matrix(Kc)
    rhs = -Kc[free][:, list(fixed_c)] @ np.array([fixed_c[k] for k in fixed_c])
    uc[free] = spla.spsolve(Kc[free][:, free].tocsc(), rhs)
    u = tie.expand(uc)
    ux, uy = u[0::2], u[1::2]
    assert np.max(np.abs(ux - eps * X[:, 0])) < 1e-9
    assert np.max(np.abs(uy + nu * eps * X[:, 1])) < 1e-9
    assert np.max(np.abs(tie.tie_residual(u))) < 1e-12


def test_condensed_system_is_symmetric_positive_definite():
    ma, mb = _two_blocks(2, 3, nx=2)
    K, X, na = _assemble(ma, mb)
    n_nodes = len(X)
    slave = [i for i in range(na, n_nodes) if abs(X[i, 0] - 1.0) < 1e-9]
    master = [i for i in range(na) if abs(X[i, 0] - 1.0) < 1e-9]
    tie = MortarTie2D(slave, master, X, 2 * n_nodes)
    Kc, _ = tie.condense(K, np.zeros(2 * n_nodes))
    Kc = Kc.toarray()
    assert np.allclose(Kc, Kc.T, atol=1e-10)
    ev = np.linalg.eigvalsh(Kc)
    assert np.sum(ev < 1e-8 * ev.max()) == 3                      # exactly the 3 rigid-body modes of the joined body


def test_unilateral_mortar_contact_closes_gap_and_transmits_the_load():
    ma, mb = _two_blocks(3, 4)
    mb.nodes[:, 0] += 0.01                                        # initial gap of 0.01 between the blocks
    K, X, na = _assemble(ma, mb)
    n_nodes = len(X)
    slave = [i for i in range(na, n_nodes) if abs(X[i, 0] - 1.01) < 1e-9]     # block B's left edge
    master = [i for i in range(na) if abs(X[i, 0] - 1.0) < 1e-9]             # block A's right edge
    tie = MortarTie2D(slave, master, X, 2 * n_nodes)
    # the normal from the slave (B, on the right) toward the master (A, on the left) is -x
    B, c = tie.normal_gap_constraints(normal=[-1.0, 0.0], initial_gap=np.full(len(slave), 0.01))
    left = [i for i in range(na) if abs(X[i, 0]) < 1e-9]
    right = [i for i in range(na, n_nodes) if abs(X[i, 0] - 2.01) < 1e-9]
    Kd = K.toarray()

    def run(push):
        pres = {}
        for i in left:
            pres[2 * i] = 0.0
            pres[2 * i + 1] = 0.0
        for i in right:
            pres[2 * i] = -push                                  # block B's far edge is pushed toward A
            pres[2 * i + 1] = 0.0
        fixed = np.array(sorted(pres))
        free = np.setdiff1d(np.arange(2 * n_nodes), fixed)
        up = np.array([pres[d] for d in fixed])
        rhs = -Kd[np.ix_(free, fixed)] @ up
        res = solve_contact_static(Kd[np.ix_(free, free)], rhs, B[:, free], c - B[:, fixed] @ up)
        u = np.zeros(2 * n_nodes)
        u[fixed], u[free] = up, res.u
        return res, u

    res, u = run(0.05)                                            # 5x the initial gap: contact must engage
    assert contact_complementarity_residual(res.lam, res.gap) < 1e-7
    assert np.all(res.lam >= -1e-12) and len(res.active) > 0
    assert np.min(B @ u - c) >= -1e-9                             # no penetration anywhere
    F_support = (Kd @ u)[[2 * i for i in right]].sum()
    assert F_support == pytest.approx(-res.lam.sum(), rel=1e-8)   # the contact force equilibrates the support
    res0, _ = run(0.005)                                          # less than the gap: the interface stays open
    assert len(res0.active) == 0 and res0.lam.sum() == 0.0

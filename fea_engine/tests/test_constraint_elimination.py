# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_constraint_elimination.py -- linear constraints in the nonlinear drivers through the elimination matrix u = T q.

Oracle: a plate built in one piece. The same plate is also built as two halves with a duplicated interface and glued with
``FESystem.tie``; the constrained nonlinear solution must equal the monolithic one (a constrained patch test, large displacement).
Evidence:
  mapping       T has identity rows for independent DOFs, coefficient rows for dependent DOFs and zero rows for fixed DOFs; expand / reduce
                round-trip; the reduced load is T^T F;
  consistency   f_q = T^T f(Tq), K_q = T^T K T (dense and CSR equal, symmetric) and K_q is the derivative of f_q (finite differences);
                batched assembly and a ContactSet pass through;
  nonlinear     static driver (automatic elimination, explicit adapter, sparse tangent), displacement control (explicit adapter) and
                arc length (automatic) on the glued plate reproduce the monolithic solution; a constraint that no mesh can express
                (two nodes forced to the same vertical displacement) is satisfied exactly;
  scope         non-zero right-hand sides, non-zero prescribed values and the transient driver are refused with a clear error.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_conformance as bc
from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine import nonlinear_solver as nls
from fea_engine.constraint_elimination import EliminatedSystem
from fea_engine.contact_batch import ContactSet2D
from fea_engine.elements import Quad4PlaneStressTL
from fea_engine.material import D_plane_stress

D = D_plane_stress(Material(E=1.0e3, nu=0.3))
TH = 0.5
NX, NY, LX, LY = 8, 2, 4.0, 1.0


def _grid(i0, i1):
    xs, ys = np.linspace(0, LX, NX + 1)[i0:i1 + 1], np.linspace(0, LY, NY + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    w = i1 - i0 + 1
    el = np.array([[j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i] for j in range(NY) for i in range(w - 1)])
    return nodes, el, w


def monolith(load=60.0):
    nodes, el, w = _grid(0, NX)
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": el}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([j * w for j in range(NY + 1)], [0, 1])
    fes.add_nodal_force([j * w + NX for j in range(NY + 1)], 1, -load)
    return fes, nodes, {"q": D}


def glued(load=60.0, vectorized_blocks=True):
    """Two halves of the same plate, interface nodes duplicated, glued by ties on both DOFs."""
    m = NX // 2
    n1, e1, w1 = _grid(0, m)
    n2, e2, w2 = _grid(m, NX)
    nodes = np.vstack([n1, n2])
    off = len(n1)
    mesh = MultiBlockMesh(nodes=nodes, blocks={"a": e1, "b": e2 + off}, dim=2)
    fes = FESystem(mesh, {"a": Quad4PlaneStressTL(), "b": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([j * w1 for j in range(NY + 1)], [0, 1])
    fes.add_nodal_force([off + j * w2 + (w2 - 1) for j in range(NY + 1)], 1, -load)
    left_if = [j * w1 + m for j in range(NY + 1)]
    right_if = [off + j * w2 for j in range(NY + 1)]
    fes.tie(right_if, left_if, [0, 1])
    return fes, nodes, {"a": D, "b": D}, off


def _match(nodes_a, ua, nodes_b, ub):
    """Largest |u| difference between two displacement fields at coincident nodes (interface duplicates compared too)."""
    ua, ub = ua.reshape(-1, 2), ub.reshape(-1, 2)
    worst = 0.0
    for p, u in zip(nodes_a, ua):
        k = np.flatnonzero(np.linalg.norm(nodes_b - p, axis=1) < 1e-9)
        assert len(k) >= 1
        worst = max(worst, max(np.abs(ub[i] - u).max() for i in k))
    return worst


# ----------------------------------------------------------------------------- mapping
def test_elimination_matrix_structure():
    fes, nodes, mat, off = glued()
    es = fes.eliminate_constraints()
    T = es.T.toarray()
    n_tie = 2 * (NY + 1)
    assert es.dependent_dofs.size == n_tie and es.n_dof == fes.n_dof - n_tie - 2 * (NY + 1)     # ties and the clamped edge
    assert T.shape == (fes.n_dof, es.n_dof)
    assert np.array_equal(T[es.independent_dofs], np.eye(es.n_dof))                              # independent rows: identity
    fixed = fes.fixed_dofs_array
    assert not T[fixed].any()                                                                    # fixed rows: zero
    for s in es.dependent_dofs:                                                                  # dependent rows: one -(-1) coefficient
        assert np.count_nonzero(T[s]) == 1 and T[s].max() == 1.0
    assert es.free_dofs.tolist() == list(range(es.n_dof)) and es.fixed_dofs == set()
    q = np.random.default_rng(0).normal(size=es.n_dof)
    u = es.expand(q)
    assert np.array_equal(es.reduce(u), q)
    for d, c, _ in fes._constraints:                                                             # constraints hold identically
        assert abs(np.dot(c, u[d])) <= 1e-14
    assert np.allclose(es.F, es.T.T @ fes.F)
    assert np.allclose(es.expand(np.stack([q, 2 * q])), np.stack([u, 2 * u]))


# ----------------------------------------------------------------------------- consistency
def test_force_and_tangent_are_the_projected_ones_and_the_tangent_is_the_derivative():
    fes, nodes, mat, off = glued()
    es = fes.eliminate_constraints()
    q = np.random.default_rng(1).normal(size=es.n_dof) * 0.05
    u = es.expand(q)
    fes._constraints_eliminated = True
    f_full = fes.assemble_internal_force(u, mat, thickness=TH)
    K_full = fes.assemble_tangent_stiffness(u, mat, thickness=TH)
    fes._constraints_eliminated = False
    f_q = es.assemble_internal_force(q, mat, thickness=TH)
    K_q = es.assemble_tangent_stiffness(q, mat, thickness=TH)
    bc.equal_to_loop(f_q, es.T.T @ f_full, label="f_q")
    bc.equal_to_loop(K_q, es.T.T @ K_full @ es.T.toarray(), label="K_q")
    Ks = es.assemble_tangent_stiffness(q, mat, sparse=True, thickness=TH)
    assert hasattr(Ks, "tocsr")
    bc.equal_to_loop(Ks, K_q, label="CSR")
    assert np.max(np.abs(K_q - K_q.T)) <= 1e-12 * np.abs(K_q).max()
    bc.fd_tangent(lambda x: es.assemble_internal_force(x, mat, thickness=TH),
                  lambda x: es.assemble_tangent_stiffness(x, mat, thickness=TH), q)
    bc.equal_to_loop(es.assemble_internal_force(q, mat, vectorized=True, thickness=TH), f_q, label="batched f_q")
    bc.equal_to_loop(es.assemble_tangent_stiffness(q, mat, vectorized=True, thickness=TH), K_q, label="batched K_q")
    # the guard is back after the calls
    with pytest.raises(NotImplementedError, match="nonlinear"):
        fes.assemble_internal_force(u, mat, thickness=TH)


def test_a_contact_set_passes_through_the_elimination():
    """A penetrating pair (found by search at this q) changes f_q and K_q by exactly the projected contact terms."""
    fg, ng, mg, off = glued()
    es = fg.eliminate_constraints()
    q = np.random.default_rng(2).normal(size=es.n_dof) * 0.1
    u = es.expand(q)
    rng = np.random.default_rng(5)
    pair = None
    for _ in range(2000):
        s_, a_, b_ = rng.choice(len(ng), 3, replace=False)
        cs = ContactSet2D([[s_, a_, b_]], 500.0)
        if cs.active(ng, u)[0]:
            pair = cs
            break
    assert pair is not None
    f_plain = es.assemble_internal_force(q, mg, thickness=TH)
    K_plain = es.assemble_tangent_stiffness(q, mg, thickness=TH)
    fg.add_contact_set(pair)
    f_with = es.assemble_internal_force(q, mg, thickness=TH)
    K_with = es.assemble_tangent_stiffness(q, mg, thickness=TH)
    bc.equal_to_loop(f_with - f_plain, es.T.T @ pair.internal_force(ng, u, fg.n_dof), label="contact force")
    Kc = np.zeros((fg.n_dof, fg.n_dof))
    r, c, v = pair.tangent_entries(ng, u)
    np.add.at(Kc, (r, c), v)
    bc.equal_to_loop(K_with - K_plain, es.T.T @ Kc @ es.T.toarray(), label="contact tangent")
    assert np.abs(f_with - f_plain).max() > 1.0


# ----------------------------------------------------------------------------- nonlinear solves
def test_static_solve_with_ties_equals_the_monolithic_plate():
    fm, nm, mm = monolith()
    _, Um = nls.solve_nonlinear_static(fm, mm, n_steps=5, tol=1e-9, max_iter=40, thickness=TH)
    fg, ng, mg, off = glued()
    assert fg._constraints                                                   # constraints present: automatic elimination
    lf, Ug = nls.solve_nonlinear_static(fg, mg, n_steps=5, tol=1e-9, max_iter=40, thickness=TH)
    assert Ug.shape == (6, fg.n_dof) and np.abs(Um[-1]).max() > 0.2          # large deflection
    assert _match(nm, Um[-1], ng, Ug[-1]) <= 1e-8 * np.abs(Um[-1]).max()
    for d, c, _ in fg._constraints:                                          # the ties hold in every step
        assert np.abs(Ug[:, d] @ c).max() <= 1e-12


def test_explicit_adapter_sparse_tangent_and_batched_assembly_give_the_same_solution():
    fm, nm, mm = monolith()
    _, Um = nls.solve_nonlinear_static(fm, mm, n_steps=5, tol=1e-9, max_iter=40, thickness=TH)
    res = {}
    for name, kw in (("explicit", {}), ("sparse batched", dict(sparse_tangent=True, vectorized=True))):
        fg, ng, mg, off = glued()
        es = fg.eliminate_constraints()
        assert isinstance(es, EliminatedSystem)
        _, Uq = nls.solve_nonlinear_static(es, mg, n_steps=5, tol=1e-9, max_iter=40, thickness=TH, **kw)
        assert Uq.shape[1] == es.n_dof
        res[name] = es.expand(Uq)[-1]
        assert _match(nm, Um[-1], ng, res[name]) <= 1e-8 * np.abs(Um[-1]).max()
    assert np.max(np.abs(res["explicit"] - res["sparse batched"])) <= 1e-9 * np.abs(res["explicit"]).max()


def test_displacement_control_with_the_explicit_adapter():
    fm, nm, mm = monolith()
    tip_m = (NY + 1) * (NX + 1) - 1                                          # top right node of the monolith
    target = np.linspace(-0.2, -1.2, 5)
    Um, Rm = nls.solve_nonlinear_displacement_control(fm, mm, 2 * tip_m + 1, target, tol=1e-9, thickness=TH)
    fg, ng, mg, off = glued()
    n_half = (NX // 2 + 1)
    tip_g = off + (NY + 1) * n_half - 1
    es = fg.eliminate_constraints()
    q_dof = int(np.flatnonzero(es.independent_dofs == 2 * tip_g + 1)[0])
    Uq, Rq = nls.solve_nonlinear_displacement_control(es, mg, q_dof, target, tol=1e-9, thickness=TH)
    Ug = es.expand(Uq)
    assert _match(nm, Um[-1], ng, Ug[-1]) <= 1e-7 * np.abs(Um[-1]).max()
    assert np.allclose(Ug[:, 2 * tip_g + 1], target)


def test_arc_length_with_ties_stays_on_the_monolithic_equilibrium_path():
    fg, ng, mg, off = glued()
    lf, Ug = nls.solve_nonlinear_arc_length(fg, mg, 0.05, n_steps=6, tol=1e-9, thickness=TH)
    assert Ug.shape[1] == fg.n_dof and np.abs(Ug[-1]).max() > 0.05
    for d, c, _ in fg._constraints:
        assert np.abs(Ug[:, d] @ c).max() <= 1e-12
    fm, nm, mm = monolith()
    _, Um = nls.solve_nonlinear_static(fm, mm, load_factors=[float(lf[-1])], tol=1e-10, max_iter=60, thickness=TH)
    assert _match(nm, Um[-1], ng, Ug[-1]) <= 1e-6 * np.abs(Um[-1]).max()


def test_a_constraint_no_mesh_can_express_is_satisfied_exactly():
    """Two top-edge nodes forced to the same vertical displacement; the constrained solution differs from the free one."""
    fm, nm, mm = monolith()
    _, Ufree = nls.solve_nonlinear_static(fm, mm, n_steps=5, tol=1e-9, max_iter=40, thickness=TH)
    a, b = (NY + 1) * (NX + 1) - 1, (NY + 1) * (NX + 1) - 1 - NX // 2                 # tip and a mid-span top node
    fm.add_constraint([(a, "uy", 1.0), (b, "uy", -1.0)])
    _, Uc = nls.solve_nonlinear_static(fm, mm, n_steps=5, tol=1e-9, max_iter=40, thickness=TH)
    assert np.abs(Uc[:, 2 * a + 1] - Uc[:, 2 * b + 1]).max() <= 1e-13
    assert np.abs(Uc[-1] - Ufree[-1]).max() > 0.05 * np.abs(Ufree[-1]).max()


# ----------------------------------------------------------------------------- scope
def test_unsupported_cases_are_refused():
    fes, nodes, mat = monolith()
    fes.add_constraint([(3, "ux", 1.0), (4, "ux", -1.0)], value=0.1)
    with pytest.raises(NotImplementedError, match="right-hand side"):
        fes.eliminate_constraints()
    with pytest.raises(NotImplementedError, match="right-hand side"):
        nls.solve_nonlinear_static(fes, mat, n_steps=2, thickness=TH)
    f2, n2, m2 = monolith()
    f2.add_constraint([(3, "ux", 1.0), (4, "ux", -1.0)])
    f2.fix_dofs([20], [1], value=0.5)
    with pytest.raises(NotImplementedError, match="prescribed"):
        f2.eliminate_constraints()
    with pytest.raises(ValueError, match="no constraints"):
        EliminatedSystem(monolith()[0])
    f3, n3, m3, off3 = glued()
    f3.assemble_mass(1.0 * np.eye(2), thickness=TH)
    f3.C = np.zeros((f3.n_dof, f3.n_dof))
    with pytest.raises(NotImplementedError, match="linear constraints"):
        nls.solve_nonlinear_transient(f3, m3, lambda t: f3.F, 0.1, 0.05, thickness=TH)

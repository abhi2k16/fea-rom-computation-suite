# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_contact_set_2d.py -- ContactSet2D / FESystem.add_contact_set against one NodeToSegmentContact2D element per pair.

Oracle: the package's own element (``elements/contact.py``) registered pair by pair with ``add_contact_element``.
Evidence:
  search      the KD-tree broad phase returns exactly the pairs of ``find_contact_pairs_2d`` (same order);
  kernels     pair forces, gaps and tangents equal the elements' (the same closed-form force and central difference), equal and
              opposite force on every pair, per-pair stiffness, pairs that share a slave node or a master node;
  system      dense force, dense tangent and CSR tangent of a two-body model equal the per-element registration, with and
              without vectorized=True for the plates; a Newton solve with the set equals the one with the elements;
  smoothing   off by default; the smoothed law is continuous with continuous slope, equals the linear law shifted by
              k*eps/2 beyond eps, is named in describe(), and its tangent matches a finite difference;
  guards      bad stiffness, shapes, node numbers and DOF-per-node mismatch raise.
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
from fea_engine.contact_batch import ContactSet2D, candidate_pairs_2d
from fea_engine.elements import Quad4PlaneStressTL, NodeToSegmentContact2D, Hex8SolidTL
from fea_engine.elements.contact import find_contact_pairs_2d
from fea_engine.material import D_plane_stress

KP = 4.0e3
TH = 0.5
D = D_plane_stress(Material(E=1.0e3, nu=0.3))


def two_bodies(nx=6, ny=2, gap=0.02):
    """Lower plate (master body) and an upper plate whose bottom nodes are slaves, a small gap between them."""
    def plate(y0):
        xs, ys = np.linspace(0.0, nx / 2.0, nx + 1), y0 + np.linspace(0.0, ny / 4.0, ny + 1)
        return np.array([[x, y] for y in ys for x in xs])

    lo, up = plate(0.0), plate(ny / 4.0 + gap)
    nodes = np.vstack([lo, up])
    off = len(lo)
    quads = lambda o: np.array([[o + j * (nx + 1) + i, o + j * (nx + 1) + i + 1, o + (j + 1) * (nx + 1) + i + 1,
                                 o + (j + 1) * (nx + 1) + i] for j in range(ny) for i in range(nx)])
    top_lo = [ny * (nx + 1) + i for i in range(nx + 1)]                # top row of the lower body, left to right
    segs = [(top_lo[i + 1], top_lo[i]) for i in range(nx)]             # right-to-left so n_hat = (dy,-dx)/L points up
    slaves = [off + i for i in range(nx + 1)]                          # bottom row of the upper body
    mesh = MultiBlockMesh(nodes=nodes, blocks={"lo": quads(0), "up": quads(off)}, dim=2)
    fes = FESystem(mesh, {"lo": Quad4PlaneStressTL(), "up": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([i for i in range(nx + 1)], [0, 1])                   # lower body clamped along its bottom row
    mat = {"lo": D, "up": D}
    return fes, mat, nodes, segs, slaves, off


def _press(fes, nodes, off, depth=0.05, seed=0):
    """Pushes the left of the upper body down by ``depth`` and its right side up (a tilt), plus small noise, so that some
    pairs penetrate and some stay open."""
    u = np.zeros(fes.n_dof)
    x = nodes[off:, 0]
    u[2 * off + 1::2][: len(x)] = -depth * (1.0 - 1.6 * x / x.max())
    return u + np.random.default_rng(seed).normal(size=fes.n_dof) * 1.0e-3


@pytest.fixture(scope="module")
def setup():
    fes, mat, nodes, segs, slaves, off = two_bodies()
    pairs = [(s, seg) for s, seg in find_contact_pairs_2d(slaves, segs, nodes, 0.2)]
    return fes, mat, nodes, segs, slaves, off, pairs


def _rows(pairs):
    return np.array([[s, a, b] for s, (a, b) in pairs])


# ----------------------------------------------------------------------------- search
@pytest.mark.parametrize("radius", [0.0, 0.05, 0.2, 0.6, 5.0])
def test_kd_tree_search_returns_the_loop_pairs_in_the_same_order(radius, setup):
    fes, mat, nodes, segs, slaves, off, _ = setup
    assert candidate_pairs_2d(nodes, slaves, segs, radius) == find_contact_pairs_2d(slaves, segs, nodes, radius)


def test_kd_tree_search_on_random_segments_and_empty_inputs():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 5, size=(80, 2))
    segs = [(int(a), int(b)) for a, b in rng.integers(0, 40, size=(30, 2)) if a != b]
    slaves = list(range(40, 80))
    for r in (0.1, 0.7, 2.0):
        assert candidate_pairs_2d(X, slaves, segs, r) == find_contact_pairs_2d(slaves, segs, X, r)
    assert candidate_pairs_2d(X, [], segs, 1.0) == [] and candidate_pairs_2d(X, slaves, [], 1.0) == []
    assert candidate_pairs_2d(X, slaves, segs, 1e-9) == find_contact_pairs_2d(slaves, segs, X, 1e-9)


def test_from_search_builds_the_set(setup):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    cs = ContactSet2D.from_search(nodes, slaves, segs, 0.2, KP)
    assert cs.n_pairs == len(pairs) > len(slaves)                      # a slave node meets more than one segment
    assert np.array_equal(cs.pairs, _rows(pairs))


# ----------------------------------------------------------------------------- kernels
def test_pair_forces_gaps_and_tangents_equal_the_elements(setup):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    cs = ContactSet2D(_rows(pairs), KP)
    u = _press(fes, nodes, off)
    el = NodeToSegmentContact2D()
    ids = _rows(pairs)
    ref_f = np.array([el.internal_force(nodes[r], u[cs.dofs[k]], (KP,)) for k, r in enumerate(ids)])
    ref_k = np.array([el.tangent_stiffness(nodes[r], u[cs.dofs[k]], (KP,)) for k, r in enumerate(ids)])
    bc.equal_to_loop(cs.pair_forces(nodes, u), ref_f, label="pair force")
    bc.equal_to_loop(cs.pair_tangents(nodes, u), ref_k, label="pair tangent")
    ref_gap = np.array([el._geometry(nodes[r], u[cs.dofs[k]])["delta"] for k, r in enumerate(ids)])
    bc.equal_to_loop(cs.gaps(nodes, u), ref_gap, label="gap")
    act = cs.active(nodes, u)
    assert 0 < act.sum() < len(act)                                    # both penetrating and open pairs are exercised
    assert np.array_equal(act, ref_gap > 0.0)
    assert np.all(cs.pair_forces(nodes, u)[~act] == 0.0)


def test_each_pair_is_self_equilibrated_and_stiffness_may_vary_per_pair(setup):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    ids = _rows(pairs)
    kp = np.linspace(1.0e3, 9.0e3, len(ids))
    cs = ContactSet2D(ids, kp)
    u = _press(fes, nodes, off, seed=4)
    F = cs.pair_forces(nodes, u)
    f_slave, f_ma, f_mb = F[:, 0:2], F[:, 2:4], F[:, 4:6]
    assert np.max(np.abs(f_slave + f_ma + f_mb)) <= 1e-12 * np.max(np.abs(F))
    el = NodeToSegmentContact2D()
    ref = np.array([el.internal_force(nodes[r], u[cs.dofs[k]], (kp[k],)) for k, r in enumerate(ids)])
    bc.equal_to_loop(F, ref)
    assert abs(cs.internal_force(nodes, u, fes.n_dof).sum()) <= 1e-10 * np.abs(F).max()   # no net force on the model


def test_pairs_sharing_nodes_scatter_correctly():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, -0.1], [1.5, -0.1]])
    segs = [(1, 0), (2, 1)]                                            # right-to-left, normals up
    pairs = np.array([[3, 1, 0], [3, 2, 1], [4, 2, 1]])                # slave 3 twice, master node 1 and 2 shared
    cs = ContactSet2D(pairs, 100.0)
    u = np.zeros(10)
    el = NodeToSegmentContact2D()
    ref = np.zeros(10)
    for k, r in enumerate(pairs):
        g = cs.dofs[k]
        ref[g] += el.internal_force(X[r], u[g], (100.0,))
    got = cs.internal_force(X, u, 10)
    assert np.max(np.abs(got)) > 1.0
    bc.equal_to_loop(got, ref)


def test_empty_set_is_a_no_op():
    cs = ContactSet2D(np.zeros((0, 3), dtype=int), 1.0)
    X, u = np.zeros((4, 2)), np.zeros(8)
    assert cs.n_pairs == 0 and cs.gaps(X, u).shape == (0,) and not cs.internal_force(X, u, 8).any()
    r, c, v = cs.tangent_entries(X, u)
    assert r.size == c.size == v.size == 0


# ----------------------------------------------------------------------------- system
def _both(setup, kp=KP):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    fe_el, fe_set = two_bodies()[0], two_bodies()[0]
    for s, (a, b) in pairs:
        fe_el.add_contact_element(NodeToSegmentContact2D(), [s, a, b], (kp,))
    cs = fe_set.add_contact_set(ContactSet2D(_rows(pairs), kp))
    return fe_el, fe_set, cs


@pytest.mark.parametrize("vectorized", [False, True])
def test_system_force_and_tangent_equal_the_per_element_registration(setup, vectorized):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    fe_el, fe_set, cs = _both(setup)
    u = _press(fes, nodes, off, seed=2)
    kw = dict(thickness=TH, vectorized=vectorized)
    assert fe_set.contact_sets == [cs] and not fe_set.contact_elements
    f_ref = fe_el.assemble_internal_force(u, mat, **kw)
    plain = two_bodies()[0].assemble_internal_force(u, mat, **kw)
    assert np.max(np.abs(f_ref - plain)) > 1.0                         # the contact genuinely contributes
    bc.equal_to_loop(fe_set.assemble_internal_force(u, mat, **kw), f_ref, label="force")
    K_ref = fe_el.assemble_tangent_stiffness(u, mat, **kw)
    bc.equal_to_loop(fe_set.assemble_tangent_stiffness(u, mat, **kw), K_ref, label="dense tangent")
    bc.equal_to_loop(fe_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw), K_ref, label="CSR tangent")


def _loaded_system(as_set, kp=KP):
    """Upper body initially overlapping the lower by 0.01 (so the penalty contact carries it), its top row held in x,
    pressed down by a total force of 60. Pairs come from the KD-tree search on this geometry."""
    fes, mat, nodes, segs, slaves, off = two_bodies(gap=-0.01)
    pairs = find_contact_pairs_2d(slaves, segs, nodes, 0.2)
    if as_set:
        fes.add_contact_set(ContactSet2D(_rows(pairs), kp))
    else:
        for s, (a, b) in pairs:
            fes.add_contact_element(NodeToSegmentContact2D(), [s, a, b], (kp,))
    top = list(range(len(nodes) - 7, len(nodes)))                      # top row of the upper body
    fes.fix_dofs(top, [0])
    fes.add_nodal_force(top, 1, -60.0)                                  # total force, split evenly
    return fes, mat, nodes, pairs


def test_newton_solve_with_the_set_equals_the_solve_with_the_elements(setup):
    res = {}
    for name, as_set, kw in (("elements", False, {}), ("set", True, {}),
                             ("set sparse batched", True, dict(sparse_tangent=True, vectorized=True))):
        f, mat, nodes, pairs = _loaded_system(as_set)
        lf, U = nls.solve_nonlinear_static(f, mat, n_steps=5, tol=1e-8, max_iter=60, thickness=TH, **kw)
        res[name] = np.asarray(U)[-1]
    cs = ContactSet2D(_rows(pairs), KP)
    assert cs.active(nodes, res["set"]).sum() > 0                      # contact is genuinely active at the end of the path
    scale = np.max(np.abs(res["elements"]))
    assert scale > 0.01
    assert np.max(np.abs(res["set"] - res["elements"])) <= 1e-9 * scale
    assert np.max(np.abs(res["set sparse batched"] - res["elements"])) <= 1e-9 * scale


# ----------------------------------------------------------------------------- smoothing
def _normal_pair(delta):
    """A slave node `delta` below a flat horizontal master segment (penetration = delta)."""
    X = np.array([[0.5, 0.0], [1.0, 0.0], [0.0, 0.0]])
    u = np.zeros(6)
    u[1] = -delta
    return X, u


def test_default_is_the_package_law_and_smoothing_is_named():
    cs0 = ContactSet2D([[0, 1, 2]], 10.0)
    assert cs0.smoothing_length == 0.0 and "SMOOTHED" not in cs0.describe() and "NodeToSegmentContact2D" in cs0.describe()
    cs = ContactSet2D([[0, 1, 2]], 10.0, smoothing_length=0.1)
    assert "SMOOTHED" in cs.describe() and "SMOOTHED" in repr(cs)


def test_smoothed_law_values_continuity_and_slope():
    k, eps = 10.0, 0.1
    cs = ContactSet2D([[0, 1, 2]], k, smoothing_length=eps)
    mag = lambda d: np.linalg.norm(cs.pair_forces(*_normal_pair(d))[0, 0:2])
    assert mag(-0.01) == 0.0 and mag(0.0) == 0.0
    assert mag(0.05) == pytest.approx(k * 0.05 ** 2 / (2 * eps), rel=1e-12)
    assert mag(0.3) == pytest.approx(k * (0.3 - eps / 2), rel=1e-12)
    h = 1e-6
    for d0, tol in ((0.0, 0.0), (eps, 1e-4)):                           # continuity of value and slope at the two joints
        lo, hi = mag(d0 - h), mag(d0 + h)
        assert abs(hi - lo) <= 2 * k * h * 1.01
    slope = lambda d: (mag(d + h) - mag(d - h)) / (2 * h)
    assert slope(eps - 1e-4) == pytest.approx(slope(eps + 1e-4), rel=2e-3)
    assert slope(1e-4) == pytest.approx(0.0, abs=2e-3 * k)
    lin = ContactSet2D([[0, 1, 2]], k)
    assert np.linalg.norm(lin.pair_forces(*_normal_pair(0.3))[0, 0:2]) - mag(0.3) == pytest.approx(k * eps / 2, rel=1e-9)


def test_smoothed_tangent_matches_a_finite_difference_of_the_force():
    cs = ContactSet2D([[0, 1, 2]], 25.0, smoothing_length=0.2)
    X, u = _normal_pair(0.05)
    u = u + np.array([0.01, 0.0, 0.0, 0.02, 0.0, -0.01])
    K = cs.pair_tangents(X, np.concatenate([u]))[0]
    h = 1e-6
    Kfd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6); d[j] = h
        Kfd[:, j] = (cs.pair_forces(X, u + d)[0] - cs.pair_forces(X, u - d)[0]) / (2 * h)
    assert np.max(np.abs(K - Kfd)) <= 1e-5 * np.max(np.abs(Kfd))


def test_smoothing_changes_the_assembled_system_force(setup):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    fe_lin, fe_sm = two_bodies()[0], two_bodies()[0]
    fe_lin.add_contact_set(ContactSet2D(_rows(pairs), KP))
    fe_sm.add_contact_set(ContactSet2D(_rows(pairs), KP, smoothing_length=0.02))
    u = _press(fes, nodes, off)
    d = fe_lin.assemble_internal_force(u, mat, thickness=TH) - fe_sm.assemble_internal_force(u, mat, thickness=TH)
    assert np.max(np.abs(d)) > 1.0


# ----------------------------------------------------------------------------- guards
def test_guards(setup):
    fes, mat, nodes, segs, slaves, off, pairs = setup
    with pytest.raises(ValueError, match="k_p"):
        ContactSet2D([[0, 1, 2]], -1.0)
    with pytest.raises(ValueError, match="one value per pair"):
        ContactSet2D([[0, 1, 2], [0, 2, 3]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="smoothing_length"):
        ContactSet2D([[0, 1, 2]], 1.0, smoothing_length=-0.1)
    with pytest.raises(ValueError, match="node numbers"):
        ContactSet2D([[0, -1, 2]], 1.0)
    with pytest.raises(ValueError, match="outside the mesh"):
        two_bodies()[0].add_contact_set(ContactSet2D([[0, 1, 10 ** 6]], 1.0))
    m3 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    f3 = FESystem(MultiBlockMesh(nodes=m3, blocks={"h": np.arange(8)[None]}, dim=3), {"h": Hex8SolidTL()}, sparse=False)
    with pytest.raises(ValueError, match="DOFs per node"):
        f3.add_contact_set(ContactSet2D([[0, 1, 2]], 1.0))

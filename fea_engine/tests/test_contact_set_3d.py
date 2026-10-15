# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_contact_set_3d.py -- ContactSet3D / candidate_pairs_3d / closest_point_on_triangles against NodeToSurfaceContact3D.

Oracle: the package's element (``elements/contact.py``) and its scalar ``closest_point_on_triangle_3d`` /
``find_contact_pairs_3d``, called pair by pair. Evidence:
  projection  the array closest-point routine equals the scalar one in every region (vertices, edges, interior) and on region
              boundaries and for a zero-area triangle;
  search      the KD-tree broad phase returns the pairs of ``find_contact_pairs_3d`` in the same order;
  kernels     force, gap, consistent and central-difference tangents equal the element's; self-equilibrium; per-pair stiffness;
              pairs sharing slave or master nodes; empty set;
  system      force, dense tangent and CSR tangent of a two-body Hex8SolidTL model equal the per-element registration; a Newton solve
              (set, elements, set with sparse batched assembly) gives the same displacements with contact active;
  options     smoothing (named, continuous), the nearest-candidate selection (a different model, named, equal to the single nearest
              element), guards.
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
from fea_engine.contact_batch import ContactSet3D, ContactSet2D, candidate_pairs_3d, closest_point_on_triangles
from fea_engine.elements import Hex8SolidTL, NodeToSurfaceContact3D
from fea_engine.elements.contact import closest_point_on_triangle_3d, find_contact_pairs_3d
from fea_engine.material import D_solid3d

KP = 2.0e3
D = D_solid3d(Material(E=1.0e3, nu=0.3))
pytestmark = pytest.mark.filterwarnings("ignore:.*spurious zero-energy modes.*:UserWarning")


def two_bodies(nx=3, ny=2, gap=0.02, h=0.5):
    """Lower block (master, top face triangulated, normals up) and an upper block whose bottom nodes are the slaves."""
    def block(z0):
        xs, ys, zs = np.linspace(0, nx / 2.0, nx + 1), np.linspace(0, ny / 2.0, ny + 1), z0 + np.linspace(0, h, 2)
        nodes = np.array([[x, y, z] for z in zs for y in ys for x in xs])
        idx = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i
        el = np.array([[idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0),
                        idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)]
                       for j in range(ny) for i in range(nx)])
        return nodes, el, idx
    lo, el_lo, idx = block(0.0)
    up, el_up, _ = block(h + gap)
    off = len(lo)
    nodes = np.vstack([lo, up])
    tris = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)
            tris += [(a, b, c), (a, c, d)]                                   # counter-clockwise seen from +z: normal up
    slaves = [off + idx(i, j, 0) for j in range(ny + 1) for i in range(nx + 1)]
    mesh = MultiBlockMesh(nodes=nodes, blocks={"lo": el_lo, "up": el_up + off}, dim=3)
    fes = FESystem(mesh, {"lo": Hex8SolidTL(), "up": Hex8SolidTL()}, sparse=False)
    fes.fix_dofs([idx(i, j, 0) for j in range(ny + 1) for i in range(nx + 1)], [0, 1, 2])      # lower block clamped below
    top = [off + idx(i, j, 1) for j in range(ny + 1) for i in range(nx + 1)]
    return fes, {"lo": D, "up": D}, nodes, tris, slaves, off, top


def _rows(pairs):
    return np.array([[s, *t] for s, t in pairs])


def _press(fes, nodes, off, depth=0.05, seed=0):
    u = np.zeros(fes.n_dof)
    x = nodes[off:, 0]
    u[3 * off + 2::3][: len(x)] = -depth * (1.0 - 1.6 * x / x.max())      # tilted: some pairs penetrate, some stay open
    return u + np.random.default_rng(seed).normal(size=fes.n_dof) * 1.0e-3


@pytest.fixture(scope="module")
def setup():
    fes, mat, nodes, tris, slaves, off, top = two_bodies()
    pairs = find_contact_pairs_3d(slaves, tris, nodes, 0.45)
    return fes, mat, nodes, tris, slaves, off, pairs


# ----------------------------------------------------------------------------- projection
def test_array_closest_point_equals_the_scalar_one_in_every_region():
    rng = np.random.default_rng(0)
    a, b, c = np.array([0.0, 0, 0]), np.array([2.0, 0, 0]), np.array([0.5, 1.5, 0.3])
    P = rng.uniform(-2, 4, size=(4000, 3))
    # boundary points: on the planes through edges, exactly at vertices, edge midpoints, centroid
    P = np.vstack([P, a, b, c, (a + b) / 2, (b + c) / 2, (a + c) / 2, (a + b + c) / 3, a - (b - a), b + (b - a), c + (c - a)])
    W, proj = closest_point_on_triangles(P, np.tile(a, (len(P), 1)), np.tile(b, (len(P), 1)), np.tile(c, (len(P), 1)))
    for k, p in enumerate(P):
        w0, q0 = closest_point_on_triangle_3d(p, a, b, c)
        assert np.max(np.abs(W[k] - w0)) <= 1e-12 and np.max(np.abs(proj[k] - q0)) <= 1e-12, f"point {k}"
    assert np.allclose(W.sum(axis=1), 1.0) and np.all(W >= -1e-12)
    # every region is represented: vertices (a weight of exactly one), edges (a zero weight) and the interior
    nz = (W > 1e-12).sum(axis=1)
    assert {1, 2, 3} <= set(nz.tolist())


def test_array_closest_point_zero_area_triangle():
    a = np.array([[0.0, 0, 0]] * 3)
    b = np.array([[1.0, 0, 0]] * 3)
    c = np.array([[2.0, 0, 0]] * 3)                                         # collinear: zero area
    P = np.array([[0.5, 1.0, 0.0], [3.0, 0.5, 0.0], [-1.0, 0.0, 0.0]])
    W, proj = closest_point_on_triangles(P, a, b, c)
    for k in range(3):
        w0, q0 = closest_point_on_triangle_3d(P[k], a[k], b[k], c[k])
        assert np.allclose(W[k], w0) and np.allclose(proj[k], q0)


# ----------------------------------------------------------------------------- search
@pytest.mark.parametrize("radius", [0.0, 0.05, 0.3, 0.45, 2.0])
def test_kd_tree_search_returns_the_loop_pairs_in_the_same_order(radius, setup):
    fes, mat, nodes, tris, slaves, off, _ = setup
    assert candidate_pairs_3d(nodes, slaves, tris, radius) == find_contact_pairs_3d(slaves, tris, nodes, radius)


def test_kd_tree_search_on_random_triangles_and_empty_inputs():
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 4, size=(120, 3))
    tris = [tuple(int(v) for v in t) for t in rng.integers(0, 60, size=(40, 3)) if len(set(t)) == 3]
    slaves = list(range(60, 120))
    for r in (0.2, 0.8, 2.5):
        assert candidate_pairs_3d(X, slaves, tris, r) == find_contact_pairs_3d(slaves, tris, X, r)
    assert candidate_pairs_3d(X, [], tris, 1.0) == [] and candidate_pairs_3d(X, slaves, [], 1.0) == []


def test_from_search_builds_the_set(setup):
    fes, mat, nodes, tris, slaves, off, pairs = setup
    cs = ContactSet3D.from_search(nodes, slaves, tris, 0.45, KP, smoothing_length=0.0)
    assert cs.n_pairs == len(pairs) > len(slaves)
    assert np.array_equal(cs.pairs, _rows(pairs))


# ----------------------------------------------------------------------------- kernels
def test_pair_force_gap_and_tangents_equal_the_element(setup):
    fes, mat, nodes, tris, slaves, off, pairs = setup
    ids = _rows(pairs)
    cs = ContactSet3D(ids, KP)
    csd = ContactSet3D(ids, KP, consistent_tangent=False)
    u = _press(fes, nodes, off)
    el = NodeToSurfaceContact3D()
    ref_f = np.array([el.internal_force(nodes[r], u[cs.dofs[k]], (KP,)) for k, r in enumerate(ids)])
    ref_kc = np.array([el.tangent_stiffness(nodes[r], u[cs.dofs[k]], (KP,)) for k, r in enumerate(ids)])
    ref_kd = np.array([el.tangent_stiffness(nodes[r], u[cs.dofs[k]], (KP,), consistent=False) for k, r in enumerate(ids)])
    ref_gap = np.array([el._geometry(nodes[r], u[cs.dofs[k]])["delta"] for k, r in enumerate(ids)])
    bc.equal_to_loop(cs.pair_forces(nodes, u), ref_f, label="force")
    bc.equal_to_loop(cs.pair_tangents(nodes, u), ref_kc, label="consistent tangent")
    # a central difference with step 1e-7 amplifies rounding differences (summation order) by about 1/h: 1e-9 is measured, not 1e-15
    bc.equal_to_loop(csd.pair_tangents(nodes, u), ref_kd, rtol=1e-7, label="central-difference tangent")
    bc.equal_to_loop(cs.gaps(nodes, u), ref_gap, label="gap")
    act = cs.active(nodes, u)
    assert 0 < act.sum() < len(act) and np.array_equal(act, ref_gap > 0.0)
    # the consistent tangent is symmetric positive semidefinite
    K = cs.pair_tangents(nodes, u)
    assert np.max(np.abs(K - K.transpose(0, 2, 1))) <= 1e-12 * np.abs(K).max()
    assert np.linalg.eigvalsh(K).min() >= -1e-9 * np.abs(K).max()


def test_each_pair_is_self_equilibrated_and_stiffness_may_vary_per_pair(setup):
    fes, mat, nodes, tris, slaves, off, pairs = setup
    ids = _rows(pairs)
    kp = np.linspace(1.0e3, 9.0e3, len(ids))
    cs = ContactSet3D(ids, kp)
    u = _press(fes, nodes, off, seed=4)
    F = cs.pair_forces(nodes, u)
    assert np.max(np.abs(F[:, 0:3] + F[:, 3:6] + F[:, 6:9] + F[:, 9:12])) <= 1e-12 * np.abs(F).max()
    el = NodeToSurfaceContact3D()
    bc.equal_to_loop(F, np.array([el.internal_force(nodes[r], u[cs.dofs[k]], (kp[k],)) for k, r in enumerate(ids)]))
    assert np.abs(cs.internal_force(nodes, u, fes.n_dof).reshape(-1, 3).sum(axis=0)).max() <= 1e-10 * np.abs(F).max()


def test_pairs_sharing_nodes_scatter_correctly_and_empty_set_is_a_no_op():
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.3, 0.3, -0.1], [0.7, 0.6, -0.05]], float)
    pairs = np.array([[4, 0, 1, 2], [4, 0, 2, 3], [5, 0, 1, 2], [5, 0, 2, 3]])
    cs = ContactSet3D(pairs, 100.0)
    u = np.zeros(18)
    el = NodeToSurfaceContact3D()
    ref = np.zeros(18)
    for k, r in enumerate(pairs):
        ref[cs.dofs[k]] += el.internal_force(X[r], u[cs.dofs[k]], (100.0,))
    got = cs.internal_force(X, u, 18)
    assert np.abs(got).max() > 1.0
    bc.equal_to_loop(got, ref)
    e = ContactSet3D(np.zeros((0, 4), dtype=int), 1.0)
    assert e.n_pairs == 0 and not e.internal_force(X, u, 18).any() and e.gaps(X, u).shape == (0,)
    r, c, v = e.tangent_entries(X, u)
    assert r.size == c.size == v.size == 0


# ----------------------------------------------------------------------------- system
def _both(setup, kp=KP):
    fes, mat, nodes, tris, slaves, off, pairs = setup
    fe_el, fe_set = two_bodies()[0], two_bodies()[0]
    for s, t in pairs:
        fe_el.add_contact_element(NodeToSurfaceContact3D(), [s, *t], (kp,))
    return fe_el, fe_set, fe_set.add_contact_set(ContactSet3D(_rows(pairs), kp))


@pytest.mark.parametrize("vectorized", [False, True])
def test_system_force_and_tangent_equal_the_per_element_registration(setup, vectorized):
    fes, mat, nodes, tris, slaves, off, pairs = setup
    fe_el, fe_set, cs = _both(setup)
    u = _press(fes, nodes, off, seed=2)
    kw = dict(vectorized=vectorized)
    f_ref = fe_el.assemble_internal_force(u, mat, **kw)
    assert np.max(np.abs(f_ref - two_bodies()[0].assemble_internal_force(u, mat, **kw))) > 1.0     # contact contributes
    bc.equal_to_loop(fe_set.assemble_internal_force(u, mat, **kw), f_ref, label="force")
    K_ref = fe_el.assemble_tangent_stiffness(u, mat, **kw)
    bc.equal_to_loop(fe_set.assemble_tangent_stiffness(u, mat, **kw), K_ref, label="dense tangent")
    bc.equal_to_loop(fe_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw), K_ref, label="CSR tangent")


class _CentralDifferenceElement(NodeToSurfaceContact3D):
    """The package element with its central-difference tangent as the default (FESystem cannot pass ``consistent=False``)."""
    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, consistent=False, **kwargs):
        return super().tangent_stiffness(elem_coords, u_elem, mat, thickness, consistent=consistent, **kwargs)


def _loaded(as_set, kp, consistent=True):
    """Upper block initially overlapping the lower by 0.01, its top face held in x and y, pressed down by a total force of 40."""
    fes, mat, nodes, tris, slaves, off, top = two_bodies(gap=-0.01)
    pairs = find_contact_pairs_3d(slaves, tris, nodes, 0.45)
    if as_set:
        fes.add_contact_set(ContactSet3D(_rows(pairs), kp, consistent_tangent=consistent))
    else:
        for s, t in pairs:
            fes.add_contact_element(NodeToSurfaceContact3D() if consistent else _CentralDifferenceElement(), [s, *t], (kp,))
    fes.fix_dofs(top, [0, 1])
    fes.add_nodal_force(top, 2, -40.0)
    return fes, mat, nodes, pairs


@pytest.mark.parametrize("kp,consistent", [(400.0, True), (400.0, False), (2.0e3, False)])
def test_newton_solve_with_the_set_equals_the_solve_with_the_elements(kp, consistent):
    """k_p = 400 converges with either tangent; k_p = 2000 is run with the central-difference tangent only: with the consistent
    (Gauss-Newton) tangent this model stalls at load step 1 for k_p >= 700, with the element as well as with the set (measured; see
    the M5.3 status note in docs/batched_nonlinear_kernels_plan.md). That is a property of the element's tangent, not of the batching."""
    res = {}
    for name, as_set, kw in (("elements", False, {}), ("set", True, {}),
                             ("set sparse batched", True, dict(sparse_tangent=True, vectorized=True))):
        f, mat, nodes, pairs = _loaded(as_set, kp, consistent)
        _, U = nls.solve_nonlinear_static(f, mat, n_steps=4, tol=1e-7, max_iter=60, **kw)
        res[name] = np.asarray(U)[-1]
    cs = ContactSet3D(_rows(pairs), kp)
    assert cs.active(nodes, res["set"]).sum() > 0                       # contact is active at the end of the path
    scale = np.max(np.abs(res["elements"]))
    assert scale > 0.01
    assert np.max(np.abs(res["set"] - res["elements"])) <= 1e-8 * scale
    assert np.max(np.abs(res["set sparse batched"] - res["elements"])) <= 1e-8 * scale


def test_smoothing_lets_the_consistent_tangent_converge_where_the_plain_law_stalls():
    """Measured on this model (see the M5.3 / M5.4 notes in the plan): with the plain ramp and the consistent tangent, Newton stalls
    at k_p = 1000 (the active set changes between iterations); with smoothing_length = 0.005 it converges. Only the converging
    run is asserted (a stall is not asserted: it depends on the path), and the result must agree with the central-difference
    tangent on the same smoothed law."""
    res = {}
    for name, cons in (("consistent", True), ("central difference", False)):
        fes, mat, nodes, tris, slaves, off, top = two_bodies(gap=-0.01)
        pairs = find_contact_pairs_3d(slaves, tris, nodes, 0.45)
        cs = fes.add_contact_set(ContactSet3D(_rows(pairs), 1000.0, smoothing_length=0.005, consistent_tangent=cons))
        fes.fix_dofs(top, [0, 1])
        fes.add_nodal_force(top, 2, -40.0)
        _, U = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-7, max_iter=60, vectorized=True)
        res[name] = np.asarray(U)[-1]
    assert cs.active(nodes, res["consistent"]).sum() > 0
    assert np.max(np.abs(res["consistent"] - res["central difference"])) <= 1e-8 * np.max(np.abs(res["consistent"]))


# ----------------------------------------------------------------------------- options
def _flat_pair(delta, x=0.5, y=0.25):
    X = np.array([[x, y, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])    # slave + one master triangle, normal +z
    u = np.zeros(12)
    u[2] = -delta
    return X, u


def test_default_is_the_package_law_and_options_are_named():
    d0 = ContactSet3D([[0, 1, 2, 3]], 10.0).describe()
    assert "SMOOTHED" not in d0 and "NEAREST" not in d0 and "NodeToSurfaceContact3D" in d0 and "Gauss-Newton" in d0
    assert "SMOOTHED" in ContactSet3D([[0, 1, 2, 3]], 10.0, smoothing_length=0.1).describe()
    assert "NEAREST" in repr(ContactSet3D([[0, 1, 2, 3]], 10.0, nearest_only=True))
    assert "central-difference" in ContactSet3D([[0, 1, 2, 3]], 10.0, consistent_tangent=False).describe()


def test_smoothed_law_values_continuity_and_tangent():
    k, eps = 10.0, 0.1
    cs = ContactSet3D([[0, 1, 2, 3]], k, smoothing_length=eps)
    mag = lambda d: np.linalg.norm(cs.pair_forces(*_flat_pair(d))[0, 0:3])
    assert mag(-0.01) == 0.0 and mag(0.0) == 0.0
    assert mag(0.05) == pytest.approx(k * 0.05 ** 2 / (2 * eps), rel=1e-12)
    assert mag(0.3) == pytest.approx(k * (0.3 - eps / 2), rel=1e-12)
    h = 1e-6
    for d0 in (0.0, eps):
        assert abs(mag(d0 + h) - mag(d0 - h)) <= 2 * k * h * 1.01
    # the consistent tangent is dF/d(delta) * C C^T: its slave-z entry is the derivative of the force magnitude with respect to delta
    X, u = _flat_pair(0.05)
    K = cs.pair_tangents(X, u)[0]
    assert K[2, 2] == pytest.approx((mag(0.05 + h) - mag(0.05 - h)) / (2 * h), rel=1e-6)
    assert K[2, 2] == pytest.approx(k * 0.05 / eps, rel=1e-12)
    # the central-difference tangent differs from it by the neglected geometry terms, of the order of the contact force itself
    Kfd = ContactSet3D([[0, 1, 2, 3]], k, smoothing_length=eps, consistent_tangent=False).pair_tangents(X, u)[0]
    assert np.max(np.abs(K - Kfd)) <= 1.1 * mag(0.05)
    assert Kfd[2, 2] == pytest.approx(K[2, 2], rel=1e-6)


def test_nearest_only_keeps_one_candidate_per_slave_and_matches_the_single_element():
    # one slave above the shared edge region of two coplanar triangles; it is closer to the first
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.8, 0.3, -0.05]], float)
    pairs = np.array([[4, 0, 1, 2], [4, 0, 2, 3]])
    u = np.zeros(15)
    both = ContactSet3D(pairs, KP)
    near = ContactSet3D(pairs, KP, nearest_only=True)
    el = NodeToSurfaceContact3D()
    f0 = el.internal_force(X[pairs[0]], np.zeros(12), (KP,))
    f1 = el.internal_force(X[pairs[1]], np.zeros(12), (KP,))
    assert np.abs(f0).max() > 0 and np.abs(f1).max() > 0                     # both triangles would act if all pairs are kept
    Fb, Fn = both.pair_forces(X, u), near.pair_forces(X, u)
    bc.equal_to_loop(Fb[0], f0); bc.equal_to_loop(Fb[1], f1)
    d2 = [np.linalg.norm(X[4] - closest_point_on_triangle_3d(X[4], *X[p[1:]])[1]) for p in pairs]
    win = int(np.argmin(d2))
    assert np.array_equal(near.active(X, u), [win == 0, win == 1])
    bc.equal_to_loop(Fn[win], [f0, f1][win])
    assert not Fn[1 - win].any()
    assert near.pair_tangents(X, u)[1 - win].max() == 0.0
    # equal distances: the lower row number wins
    tie = ContactSet3D(np.array([[4, 0, 1, 2], [4, 0, 1, 2]]), KP, nearest_only=True)
    assert np.array_equal(tie.active(X, u), [True, False])


# ----------------------------------------------------------------------------- guards
def test_guards():
    with pytest.raises(ValueError, match="k_p"):
        ContactSet3D([[0, 1, 2, 3]], -1.0)
    with pytest.raises(ValueError, match="one value per pair"):
        ContactSet3D([[0, 1, 2, 3]], [1.0, 2.0])
    with pytest.raises(ValueError, match="smoothing_length"):
        ContactSet3D([[0, 1, 2, 3]], 1.0, smoothing_length=-1.0)
    with pytest.raises(ValueError, match="node numbers"):
        ContactSet3D([[0, 1, -2, 3]], 1.0)
    fes = two_bodies()[0]
    with pytest.raises(ValueError, match="outside the mesh"):
        fes.add_contact_set(ContactSet3D([[0, 1, 2, 10 ** 6]], 1.0))
    with pytest.raises(ValueError, match="DOFs per node"):
        fes.add_contact_set(ContactSet2D([[0, 1, 2]], 1.0))

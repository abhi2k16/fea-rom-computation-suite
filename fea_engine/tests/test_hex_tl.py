# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_hex_tl.py -- Total-Lagrangian St. Venant-Kirchhoff hexahedra (Hex8SolidTL and Hex20SolidTL), per-Gauss-point
reference implementation.

Evidence, none of it by comparison with another implementation of the same formulas:
  (1) the tangent equals central differences of the internal force on a distorted element, at small and large
      displacement, and is symmetric;
  (2) zero displacement gives zero force and the tangent equals the linear element's stiffness (Hex8Solid3D) for the
      same quadrature; the force at a tiny displacement equals the linear element's K u;
  (3) a rigid rotation about a general axis plus a translation gives no force (objectivity);
  (4) a uniform stretch gives the exact St. Venant force on every face of a cube;
  (5) a nu = 0 bar stretched 40 % gives the end force of TrussTL2D (E, A = cross-section);
  (6) a cantilever driven through solve_nonlinear_static is in equilibrium, equals the linear solution at a tiny
      load and is stiffer than the linear solution at a large load (stress stiffening), for both element types;
  (7) Hex20 only: curved-edge elements (mid-nodes displaced), the 3 x 3 x 3 default, and a Hex20 mesh built from a
      Hex8 mesh run through the nonlinear driver;
  plus argument checks, Material-or-D input, the element registry, and acceptance by MeshTransformation.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_conformance as bc
import fea_engine.nonlinear_solver as nls
from fea_engine import FESystem, Material, Mesh
from fea_engine import mesh as meshlib
from fea_engine.material import D_solid3d
from fea_engine.elements import Hex8Solid3D, Hex20Solid3D, Hex8SolidTL, Hex20SolidTL, TrussTL2D, ELEMENT_REGISTRY
import hex_meshes

# 2x2x2 Hex20 is used on purpose in some tests; its construction warns (checked in test_hex_tl_validation.py)
pytestmark = pytest.mark.filterwarnings("ignore:.*spurious zero-energy modes.*:UserWarning")

HEX_CLASSES = [Hex8SolidTL, Hex20SolidTL]
LINEAR = {Hex8SolidTL: Hex8Solid3D, Hex20SolidTL: Hex20Solid3D}
MAT = Material(E=1.0e3, nu=0.3)
D = D_solid3d(MAT)

_C8 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)


def unit_cell(cls):
    """Reference coordinates of the unit cube for the element (corners; Hex20 adds mid-edge nodes)."""
    el = cls()
    if el.n_nodes == 8:
        return _C8.copy()
    mids = np.array([0.5 * (_C8[a] + _C8[b]) for a, b in
                     [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]])
    return np.vstack([_C8, mids])


def distorted(cls, seed=0, amp=0.08):
    """Non-rectangular element: unit cube with a smooth fixed distortion (straight edges stay straight for Hex8;
    for Hex20 the mid-nodes follow the same smooth map, giving curved edges)."""
    X = unit_cell(cls)
    d = np.stack([amp * np.sin(2 * X[:, 1] + X[:, 2]), amp * np.sin(1.5 * X[:, 0] - X[:, 2]),
                  amp * np.sin(X[:, 0] + 2 * X[:, 1])], axis=1)
    return X + d


def _state(n_dof, seed, scale):
    return np.random.default_rng(seed).normal(size=n_dof) * scale


# ----------------------------------------------------------------------------- (1) tangent
@pytest.mark.parametrize("cls", HEX_CLASSES)
@pytest.mark.parametrize("scale", [0.02, 0.2])
def test_tangent_is_derivative_of_internal_force(cls, scale):
    el = cls()
    X = distorted(cls)
    n = 3 * el.n_nodes
    u = _state(n, 1, scale)
    f = lambda w: el.internal_force(X, w, D)
    K = lambda w: el.tangent_stiffness(X, w, D)
    bc.fd_tangent(f, K, u, h=1e-6, rtol=1e-6, sym_tol=1e-10)


# ----------------------------------------------------------------------------- (2) linear limit
@pytest.mark.parametrize("cls", HEX_CLASSES)
@pytest.mark.parametrize("order", [2, 3])
def test_linear_limit(cls, order):
    el = cls(gauss_order=order)
    lin = LINEAR[cls]()
    X = distorted(cls)
    n = 3 * el.n_nodes
    f0 = el.internal_force(X, np.zeros(n), D)
    assert np.max(np.abs(f0)) < 1e-12
    K0 = el.tangent_stiffness(X, np.zeros(n), D)
    assert bc.rel_err(K0, lin.stiffness(X, D, 1.0, gauss_order=order)) < 1e-10
    u = _state(n, 2, 1e-9)
    assert bc.rel_err(el.internal_force(X, u, D), lin.stiffness(X, D, 1.0, gauss_order=order) @ u) < 1e-6


# ----------------------------------------------------------------------------- (3) objectivity
@pytest.mark.parametrize("cls", HEX_CLASSES)
@pytest.mark.parametrize("theta", [0.3, 0.7, 2.0])
def test_rigid_rotation_gives_no_force(cls, theta):
    el = cls()
    X = distorted(cls)
    stretch = el.internal_force(X, 0.01 * (X.ravel() - np.tile(X.mean(axis=0), el.n_nodes)), D)
    scale = float(np.max(np.abs(stretch)))
    bc.objective(lambda w: el.internal_force(X, w, D), X, scale, theta=theta, tol=1e-9)


# ----------------------------------------------------------------------------- (4) uniform stretch
@pytest.mark.parametrize("cls", HEX_CLASSES)
def test_uniform_stretch_exact_force(cls):
    el = cls()
    X = unit_cell(cls)
    lam = 1.3
    u = np.zeros_like(X)
    u[:, 0] = (lam - 1.0) * X[:, 0]
    f = el.internal_force(X, u.ravel(), D).reshape(-1, 3)
    E11 = 0.5 * (lam ** 2 - 1.0)
    S = D @ np.array([E11, 0, 0, 0, 0, 0])
    right, left = np.isclose(X[:, 0], 1.0), np.isclose(X[:, 0], 0.0)
    assert f[right, 0].sum() == pytest.approx(lam * S[0], rel=1e-12)
    assert f[left, 0].sum() == pytest.approx(-lam * S[0], rel=1e-12)
    for axis, comp in ((1, 1), (2, 2)):                    # lateral faces carry S22, S33 (F = 1 in those directions)
        assert f[np.isclose(X[:, axis], 1.0), comp].sum() == pytest.approx(S[comp], rel=1e-12)
        assert f[np.isclose(X[:, axis], 0.0), comp].sum() == pytest.approx(-S[comp], rel=1e-12)
    assert np.max(np.abs(f.sum(axis=0))) < 1e-10          # self-equilibrated


# ----------------------------------------------------------------------------- (5) truss
def test_nu0_bar_matches_truss_at_large_stretch():
    E, w, t, L = 200.0, 0.4, 0.5, 3.0
    el = Hex8SolidTL()
    Dn0 = D_solid3d(Material(E=E, nu=0.0))
    X = _C8 * np.array([L, w, t])
    d = 0.4 * L
    u = np.zeros_like(X)
    u[:, 0] = d * X[:, 0] / L
    f = el.internal_force(X, u.ravel(), Dn0).reshape(-1, 3)
    f_t = TrussTL2D().internal_force(np.array([[0.0, 0.0], [L, 0.0]]), np.array([0, 0, d, 0.0]), (E, w * t))
    assert f[np.isclose(X[:, 0], L), 0].sum() == pytest.approx(f_t[2], rel=1e-12)
    assert f[np.isclose(X[:, 0], 0.0), 0].sum() == pytest.approx(f_t[0], rel=1e-12)


# ----------------------------------------------------------------------------- (6) cantilever
def _box(cls, nx, ny, nz, L=4.0, H=1.0, B=1.0):
    m = meshlib.box_mesh(L, H, B, nx, ny, nz)
    return hex_meshes.hex20_from_hex8(m) if cls is Hex20SolidTL else m


def _cantilever(cls, load, n_steps, nx=8, ny=2, nz=2, L=4.0, H=1.0, B=1.0):
    m = _box(cls, nx, ny, nz, L, H, B)
    fes = FESystem(m, cls(), sparse=False)
    fes.fix_dofs(np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    tip = np.where(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    fes.add_nodal_force(tip, 1, -load)                       # total force, split evenly over the tip nodes
    lf, U = nls.solve_nonlinear_static(fes, D, n_steps=n_steps, tol=1e-9, max_iter=40)
    u = U[-1]
    R = fes.assemble_internal_force(u, D) - lf[-1] * fes.F
    return u, tip, float(np.max(np.abs(R[fes.free_dofs]))), fes


def _linear_tip(load, cls=Hex8SolidTL, nx=8, ny=2, nz=2):
    m = _box(cls, nx, ny, nz)
    fes = FESystem(m, LINEAR[cls](), sparse=False)
    fes.assemble_stiffness(D)
    fes.fix_dofs(np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    tip = np.where(np.abs(m.nodes[:, 0] - 4.0) < 1e-9)[0]
    fes.add_nodal_force(tip, 1, -load)
    u = fes.solve_static()
    return u[3 * tip + 1].mean()


@pytest.mark.parametrize("cls,nx,ny,nz,big", [(Hex8SolidTL, 8, 2, 2, 8.0), (Hex20SolidTL, 4, 1, 1, 8.0)])
def test_cantilever_matches_linear_when_small_and_stiffens_when_large(cls, nx, ny, nz, big):
    u_t, tip, res, _ = _cantilever(cls, 1e-3, 1, nx, ny, nz)
    assert res < 1e-9
    assert u_t[3 * tip + 1].mean() == pytest.approx(_linear_tip(1e-3, cls, nx, ny, nz), rel=1e-4)
    u_l = _linear_tip(big, cls, nx, ny, nz)
    u_t, tip, res, _ = _cantilever(cls, big, 20, nx, ny, nz)
    assert res < 1e-6
    assert abs(u_l) > 0.5                                    # genuinely large for this mesh
    assert abs(u_t[3 * tip + 1].mean()) < 0.95 * abs(u_l)    # stiffer than linear: geometric stiffening


# ----------------------------------------------------------------------------- arguments, input forms, registry
@pytest.mark.parametrize("cls", HEX_CLASSES)
def test_material_and_matrix_inputs_agree(cls):
    el = cls()
    X = distorted(cls)
    u = _state(3 * el.n_nodes, 3, 0.05)
    assert bc.rel_err(el.internal_force(X, u, MAT), el.internal_force(X, u, D)) == 0.0
    assert bc.rel_err(el.tangent_stiffness(X, u, MAT), el.tangent_stiffness(X, u, D)) == 0.0


def test_hex20_defaults_and_curved_edges():
    assert Hex20SolidTL().gauss_order == 3 and Hex20SolidTL(gauss_order=2).gauss_order == 2
    el = Hex20SolidTL()
    X = distorted(Hex20SolidTL)
    mid = X[8:]
    straight = 0.5 * (X[[a for a, _ in hex_meshes.HEX20_EDGES]] + X[[b for _, b in hex_meshes.HEX20_EDGES]])
    assert np.max(np.abs(mid - straight)) > 1e-3             # the test element really has curved edges
    u = _state(60, 4, 0.1)
    K = el.tangent_stiffness(X, u, D)
    f = lambda w: el.internal_force(X, w, D)
    bc.fd_tangent(f, lambda w: el.tangent_stiffness(X, w, D), u, h=1e-6, rtol=1e-6)
    assert np.linalg.matrix_rank(el.tangent_stiffness(X, np.zeros(60), D), tol=1e-8 * np.abs(K).max()) == 60 - 6


def test_hex20_mesh_builder_shares_midnodes():
    m8 = meshlib.box_mesh(2.0, 1.0, 1.0, 2, 1, 1)
    m20 = hex_meshes.hex20_from_hex8(m8)
    assert m20.elements.shape == (2, 20)
    assert len(m20.nodes) == 12 + 20                          # 12 corners + 8 x-edges + 6 y-edges + 6 z-edges
    assert len(set(m20.elements[0, 8:]) & set(m20.elements[1, 8:])) == 4      # the four edges of the common face
    for conn in m20.elements:                                  # every mid-node sits at its edge midpoint
        X = m20.nodes[conn]
        for k, (a, b) in enumerate(hex_meshes.HEX20_EDGES):
            assert np.allclose(X[8 + k], 0.5 * (X[a] + X[b]))


def test_argument_validation():
    with pytest.raises(ValueError, match="gauss_order"):
        Hex8SolidTL(gauss_order=0)
    with pytest.raises(ValueError, match="gauss_order"):
        Hex8SolidTL(gauss_order=2.5)
    with pytest.raises(ValueError, match="6 x 6"):
        Hex8SolidTL().internal_force(_C8, np.zeros(24), np.eye(3))
    assert Hex8SolidTL().gauss_order == 2 and Hex8SolidTL(gauss_order=3).gauss_order == 3


def test_registry_and_mesh_transformation():
    assert ELEMENT_REGISTRY["hex8_solid_tl"] is Hex8SolidTL
    from fea_engine.mesh_transform import MeshTransformation
    m = meshlib.box_mesh(2.0, 1.0, 1.0, 2, 1, 1)
    mt = MeshTransformation(m, Hex8SolidTL())
    assert mt.shape_grad.shape == (2, 8, 8, 3) and mt.JxW.sum() == pytest.approx(2.0, rel=1e-12)

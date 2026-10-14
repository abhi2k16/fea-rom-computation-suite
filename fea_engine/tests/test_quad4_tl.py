# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_quad4_tl.py -- Quad4PlaneStressTL (Total-Lagrangian, St. Venant-Kirchhoff, plane stress), the element of
Nikiema et al. (2026) Sec. 2.3/6.1.

Evidence: (1) the tangent is the derivative of internal_force (central FD) on a distorted element at large strain;
(2) small displacements reproduce Quad4PlaneStress; (3) a rigid rotation produces no force; (4) a uniform stretch
gives the exact St. Venant force; (5) a nu=0 strip equals TrussTL2D (E, A=t*w) at 40% stretch; (6) the tangent is
symmetric; (7) a cantilever driven through solve_nonlinear_static is in equilibrium.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Material
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStress, Quad4PlaneStressTL, TrussTL2D

D = D_plane_stress(Material(E=1.0e3, nu=0.3))
X = np.array([[0.0, 0.0], [2.0, 0.1], [2.2, 1.1], [-0.1, 0.9]])      # distorted, non-rectangular
el = Quad4PlaneStressTL()


def _state(seed, scale):
    return np.random.default_rng(seed).normal(size=8) * scale


@pytest.mark.parametrize("scale", [0.02, 0.2])
def test_tangent_is_derivative_of_internal_force(scale):
    u = _state(1, scale)
    K = el.tangent_stiffness(X, u, D, 0.5)
    h = 1e-6
    Kfd = np.column_stack([(el.internal_force(X, u + h * e, D, 0.5) - el.internal_force(X, u - h * e, D, 0.5)) / (2 * h)
                           for e in np.eye(8)])
    assert np.max(np.abs(K - Kfd)) < 1e-6 * max(1.0, np.max(np.abs(K)))
    assert np.allclose(K, K.T, atol=1e-10 * np.max(np.abs(K)))


def test_small_displacement_limit_is_the_linear_element():
    lin = Quad4PlaneStress()
    u = _state(2, 1e-9)
    assert np.allclose(el.internal_force(X, u, D, 0.5), lin.internal_force(X, u, D, 0.5), rtol=1e-6, atol=1e-12)
    assert np.allclose(el.tangent_stiffness(X, np.zeros(8), D, 0.5), lin.stiffness(X, D, 0.5), rtol=1e-10)


def test_rigid_rotation_gives_no_force():
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    u = ((R - np.eye(2)) @ X.T).T.ravel() + np.tile([0.3, -0.2], 4)
    assert np.max(np.abs(el.internal_force(X, u, D, 1.0))) < 1e-9


def test_uniform_stretch_exact_force():
    Xr = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lam = 1.3
    u = np.zeros(8)
    u[2::2] = (lam - 1.0) * Xr[1:, 0]; u[0] = 0.0                    # x -> lam x
    E11 = 0.5 * (lam ** 2 - 1.0)
    S = D @ np.array([E11, 0.0, 0.0])
    f = el.internal_force(Xr, u, D, 1.0)
    # node 1 (x=1,y=0): f = F S dN, with dN/dX = (0.5, -0.5) integrated over unit square -> (lam*S11*0.5 ...)
    assert f[2] == pytest.approx(lam * S[0] * 0.5)
    assert f[0] == pytest.approx(-lam * S[0] * 0.5)
    assert abs(np.sum(f[0::2])) < 1e-12 and abs(np.sum(f[1::2])) < 1e-12


def test_strip_matches_truss_at_large_stretch():
    E, t, w, L = 200.0, 0.5, 0.4, 3.0
    Dnu0 = D_plane_stress(Material(E=E, nu=0.0))
    Xs = np.array([[0.0, 0.0], [L, 0.0], [L, w], [0.0, w]])
    d = 0.4 * L
    u = np.array([0, 0, d, 0, d, 0, 0, 0.0])
    f_q = el.internal_force(Xs, u, Dnu0, t)
    f_t = TrussTL2D().internal_force(np.array([[0.0, 0.0], [L, 0.0]]), np.array([0, 0, d, 0.0]), (E, t * w))
    assert f_q[2] + f_q[4] == pytest.approx(f_t[2], rel=1e-12)
    assert f_q[0] + f_q[6] == pytest.approx(f_t[0], rel=1e-12)


def _cantilever(el_cls, load, n_steps, nx=10, ny=2, L=4.0, H=1.0):
    from fea_engine import FESystem, MultiBlockMesh
    import fea_engine.nonlinear_solver as nls
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    elems = np.array([[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
                      for j in range(ny) for i in range(nx)])
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": el_cls()}, sparse=False)
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    fes.add_nodal_force(right, 1, -load / (ny + 1))
    mat = {"q": D}
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=n_steps, tol=1e-9, max_iter=40)
    u = U[-1]
    R = fes.assemble_internal_force(u, mat) - lf[-1] * fes.F
    return u, right[0], float(np.max(np.abs(R[fes.free_dofs])))


def test_cantilever_matches_linear_when_small_and_departs_when_large():
    # Locking of the 4-node element in bending is a property of Quad4PlaneStress itself (it shows up identically in
    # the linear element), so the check here is TL-versus-linear on the SAME mesh, not versus beam theory.
    u_l, tip, _ = _cantilever(Quad4PlaneStress, 1e-3, 1)
    u_t, _, res = _cantilever(Quad4PlaneStressTL, 1e-3, 1)
    assert res < 1e-9 and u_t[2 * tip + 1] == pytest.approx(u_l[2 * tip + 1], rel=1e-4)
    u_l, tip, _ = _cantilever(Quad4PlaneStress, 20.0, 1)
    u_t, _, res = _cantilever(Quad4PlaneStressTL, 20.0, 20)
    assert res < 1e-7                                        # equilibrium at the converged large-deflection state
    assert abs(u_l[2 * tip + 1]) > 1.0                       # the load is genuinely large for this mesh ...
    assert abs(u_t[2 * tip + 1]) < 0.9 * abs(u_l[2 * tip + 1])   # ... and the TL answer is stiffer (stress stiffening)

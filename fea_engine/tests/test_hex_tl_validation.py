# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_hex_tl_validation.py -- package-level validation of Hex8SolidTL / Hex20SolidTL: locking, large rotation, and the
2 x 2 x 2 Hex20 limitation, all measured through the package's own batched elements and drivers.

  locking     A 60 x 12 x 2 mm strip (30:1), E = 70 GPa, nu = 0.3, clamped, total tip force 1 N in the thickness direction,
              against Timoshenko beam theory (0.12868 mm). The tangent at zero displacement of the TL element is the linear
              stiffness, so the table is the element's own. Reproduces the sibling project's table (Hex8 0.28 to 0.80,
              Hex20 0.90 to 0.98 of Timoshenko) to three decimals.
  elastica    A 60 x 4 x 2 mm cantilever (nu = 0 so a 3-D solid matches beam theory) with a dead tip load P L^2 / (E I) = 1
              against the elastica (Bisshopp and Drucker 1945): tip deflection / L = 0.3017, shortening / L = 0.0561.
              Hex20 reproduces it to the discretisation level (about 3e-4 and 4e-4); Hex8 locks far off.
  2x2x2 Hex20 An isolated 2 x 2 x 2 element has 6 spurious zero-energy modes besides the 6 rigid-body ones; a mesh that is a
              single row of elements keeps them after clamping (the tangent at zero is singular, Newton raises at load step
              1); meshes with two elements across the width or the thickness have none. The 3 x 3 x 3 rule has none.
              This sharpens the sibling project's statement, which put the failure at "one layer" and reported a wrong
              value (0.1641) for it; that value came from a Newton loop that does not check convergence (the same script
              now prints 0.7518 for the same case, with 144 instead of 49 iterations).
Newton tolerance: the stiff 3-D cantilever's residual has a floating-point floor of about 2e-9 (absolute), so these tests
use tol = 1e-7.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import warnings
import numpy as np
import pytest
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hex_meshes
import fea_engine.nonlinear_solver as nls
from fea_engine import FESystem, Material, mesh as meshlib
from fea_engine.material import D_solid3d
from fea_engine.elements import Hex8SolidTL, Hex20SolidTL, Hex20Solid3D

pytestmark = pytest.mark.filterwarnings("ignore:.*spurious zero-energy modes.*:UserWarning")

E_MOD = 70000.0


def _mesh(cls, L, b, t, dims):
    m = meshlib.box_mesh(L, b, t, *dims)
    return hex_meshes.hex20_from_hex8(m) if cls is Hex20SolidTL else m


def _clamped(cls, order, L, b, t, dims, nu, total_tip_force):
    m = _mesh(cls, L, b, t, dims)
    fes = FESystem(m, cls(gauss_order=order), sparse=False)
    fes.fix_dofs(np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    tip = np.where(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    fes.add_nodal_force(tip, 2, -total_tip_force)                 # total force, split evenly over the tip nodes
    return fes, tip, D_solid3d(Material(E=E_MOD, nu=nu))


# ----------------------------------------------------------------------------- locking table
L_S, B_S, T_S, NU_S = 60.0, 12.0, 2.0, 0.3
I_S = B_S * T_S ** 3 / 12
D_TIM = L_S ** 3 / (3 * E_MOD * I_S) + L_S / ((5 / 6) * (E_MOD / (2 * (1 + NU_S))) * B_S * T_S)


def _locking_ratio(cls, order, dims):
    fes, tip, D = _clamped(cls, order, L_S, B_S, T_S, dims, NU_S, 1.0)
    K = fes.assemble_tangent_stiffness(np.zeros(fes.n_dof), D, vectorized=True, sparse=True)
    fr = fes.free_dofs
    u = np.zeros(fes.n_dof)
    u[fr] = spla.spsolve(K.tocsr()[fr][:, fr].tocsc(), fes.F[fr])
    return -u[3 * tip + 2].mean() / D_TIM, fes


LOCKING = [
    (Hex8SolidTL, 2, (12, 2, 2), 0.282), (Hex8SolidTL, 2, (24, 2, 2), 0.575), (Hex8SolidTL, 2, (48, 2, 4), 0.804),
    (Hex20SolidTL, 2, (3, 1, 1), 0.928), (Hex20SolidTL, 2, (6, 2, 1), 0.959), (Hex20SolidTL, 2, (12, 2, 1), 0.972),
    (Hex20SolidTL, 2, (12, 2, 2), 0.979),
    (Hex20SolidTL, 3, (3, 1, 1), 0.897), (Hex20SolidTL, 3, (6, 2, 1), 0.951), (Hex20SolidTL, 3, (12, 2, 1), 0.969),
    (Hex20SolidTL, 3, (12, 2, 2), 0.974),
]


@pytest.mark.parametrize("cls,order,dims,expected", LOCKING)
def test_locking_table_matches_the_measured_values(cls, order, dims, expected):
    ratio, _ = _locking_ratio(cls, order, dims)
    assert ratio == pytest.approx(expected, abs=1.5e-3)


def test_hex8_locks_and_hex20_does_not():
    r8 = [_locking_ratio(Hex8SolidTL, 2, d)[0] for d in ((12, 2, 2), (24, 2, 2), (48, 2, 4))]
    r20 = [_locking_ratio(Hex20SolidTL, 3, d)[0] for d in ((3, 1, 1), (12, 2, 2))]
    assert r8[0] < r8[1] < r8[2] < 0.81                       # improves with refinement but still locked
    assert min(r20) > 0.89 and max(r20) < 1.0                 # Hex20 within 11 % even with 20 mm elements on a 2 mm strip


def test_tl_tangent_at_zero_is_the_linear_parents_stiffness_for_the_locking_case():
    ratio_tl, fes = _locking_ratio(Hex20SolidTL, 3, (6, 2, 1))
    m = fes.mesh
    lin = FESystem(m, Hex20Solid3D(), sparse=False)
    D = D_solid3d(Material(E=E_MOD, nu=NU_S))
    lin.assemble_stiffness(D)
    lin.fix_dofs(np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    tip = np.where(np.abs(m.nodes[:, 0] - L_S) < 1e-9)[0]
    lin.add_nodal_force(tip, 2, -1.0)
    u = lin.solve_static()
    assert -u[3 * tip + 2].mean() / D_TIM == pytest.approx(ratio_tl, rel=1e-8)


# ----------------------------------------------------------------------------- elastica
L_E, B_E, T_E = 60.0, 4.0, 2.0
I_E = B_E * T_E ** 3 / 12
ELASTICA_V, ELASTICA_U = 0.3017, 0.0561


def _elastica(cls, order, dims, n_steps=10):
    fes, tip, D = _clamped(cls, order, L_E, B_E, T_E, dims, 0.0, E_MOD * I_E / L_E ** 2)
    lf, U = nls.solve_nonlinear_static(fes, D, n_steps=n_steps, tol=1e-7, max_iter=30, vectorized=True,
                                       sparse_tangent=True)
    u = U[-1]
    return -u[3 * tip + 2].mean() / L_E, -u[3 * tip].mean() / L_E


@pytest.mark.parametrize("order,dims,v_meas,u_meas", [
    (3, (12, 1, 1), 0.30156, 0.05639), (3, (12, 1, 2), 0.30159, 0.05640),
    (2, (12, 1, 2), 0.30198, 0.05653), (2, (12, 2, 1), 0.30194, 0.05651)])
def test_hex20_reproduces_the_elastica(order, dims, v_meas, u_meas):
    v, u = _elastica(Hex20SolidTL, order, dims)
    assert v == pytest.approx(v_meas, abs=2e-5) and u == pytest.approx(u_meas, abs=2e-5)    # regression on measured values
    assert abs(v - ELASTICA_V) < 1e-3 and abs(u - ELASTICA_U) < 7e-4                       # and close to the closed form


def test_hex8_locks_on_the_elastica_beam():
    v, u = _elastica(Hex8SolidTL, 2, (12, 1, 2))
    assert v == pytest.approx(0.0803, abs=2e-3) and u < 0.01          # far from 0.3017 and 0.0561


# ----------------------------------------------------------------------------- 2x2x2 Hex20 limitation
def _zero_modes(fes, clamped):
    K = fes.assemble_tangent_stiffness(np.zeros(fes.n_dof), D_solid3d(Material(E=E_MOD, nu=0.0)), vectorized=True)
    if clamped:
        fr = fes.free_dofs
        K = K[np.ix_(fr, fr)]
    ev = np.linalg.eigvalsh(K)
    return int((ev < 1e-12 * ev.max()).sum())


def _hex20_system(order, dims):
    return _clamped(Hex20SolidTL, order, L_E, B_E, T_E, dims, 0.0, 1.0)[0]


def test_isolated_hex20_spurious_modes_depend_on_the_rule():
    assert _zero_modes(_hex20_system(2, (1, 1, 1)), clamped=False) == 12          # 6 rigid + 6 spurious
    assert _zero_modes(_hex20_system(3, (1, 1, 1)), clamped=False) == 6           # rigid only


@pytest.mark.parametrize("order,dims,n_zero", [
    (2, (3, 1, 1), 3), (2, (12, 1, 1), 12),                                       # single row: spurious modes survive clamping
    (2, (12, 2, 1), 0), (2, (12, 1, 2), 0),                                       # two elements across either direction: none
    (3, (12, 1, 1), 0)])
def test_single_row_meshes_keep_spurious_modes_after_clamping(order, dims, n_zero):
    assert _zero_modes(_hex20_system(order, dims), clamped=True) == n_zero


def test_newton_fails_on_a_single_row_2x2x2_mesh_and_converges_otherwise():
    fes, tip, D = _clamped(Hex20SolidTL, 2, L_E, B_E, T_E, (12, 1, 1), 0.0, E_MOD * I_E / L_E ** 2)
    with pytest.raises(RuntimeError, match="failed to converge at load step 1"):
        nls.solve_nonlinear_static(fes, D, n_steps=10, tol=1e-7, max_iter=30, vectorized=True, sparse_tangent=True)
    v, u = _elastica(Hex20SolidTL, 3, (12, 1, 1))                                  # same mesh, 3x3x3: fine
    assert abs(v - ELASTICA_V) < 1e-3


def test_hex20_reduced_rule_warns_and_other_constructions_do_not():
    with pytest.warns(UserWarning, match="spurious zero-energy modes"):
        Hex20SolidTL(gauss_order=2)
    with pytest.warns(UserWarning, match="spurious zero-energy modes"):
        Hex20SolidTL(gauss_order=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Hex20SolidTL()
        Hex20SolidTL(gauss_order=3)
        Hex20SolidTL(gauss_order=4)
        Hex8SolidTL()
        Hex8SolidTL(gauss_order=2)

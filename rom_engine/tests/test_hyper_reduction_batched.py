# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Batched hyper-reduction routes (ecsw_training_matrix_batched, ECSW.fit_batched / batched reduced force and tangent,
nonlinearity_indicator_batched) against the per-element routes, on a real fea_engine nonlinear model: a Quad4PlaneStressTL
plate (distorted mesh, large deflection). The per-element oracle calls the element's own ``internal_force`` /
``tangent_stiffness`` one element at a time; the batched callbacks are ``FESystem.element_forces_local`` /
``element_tangents_local`` with ``vectorized=True``. Equality is asserted to 1e-12 relative (rounding level is about
1e-15); speed is never asserted."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine import nonlinear_solver as nls
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL
from rom_engine.hyper_reduction import ECSW, ecsw_training_matrix, ecsw_training_matrix_batched
from rom_engine.domain_partition import nonlinearity_indicator, nonlinearity_indicator_batched

TH = 0.5


def _plate(nx=12, ny=3):
    xs, ys = np.linspace(0, nx / 2.0, nx + 1), np.linspace(0, ny / 4.0, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    nodes[:, 0] += 0.04 * np.sin(2.0 * nodes[:, 1])
    elems = np.array([[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
                      for j in range(ny) for i in range(nx)])
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    fes.add_nodal_force([(j + 1) * (nx + 1) - 1 for j in range(ny + 1)], 1, -1.5)
    mat = {"q": D_plane_stress(Material(E=1.0e3, nu=0.3))}
    return fes, mat, nodes, elems


@pytest.fixture(scope="module")
def model():
    fes, mat, nodes, elems = _plate()
    _, U = nls.solve_nonlinear_static(fes, mat, n_steps=8, tol=1e-9, max_iter=40, thickness=TH)
    S = np.asarray(U).T                                    # (n_dof, n_snap) full-order states
    S = S[:, 1:]
    free = fes.free_dofs
    Ub, sv, _ = np.linalg.svd(S[free, :], full_matrices=False)
    V = np.zeros((fes.n_dof, 4))
    V[free, :] = Ub[:, :4]
    dofs = [np.asarray(g) for g in fes.element_dofs()]
    form, D = fes._blocks[0][1], mat["q"]
    X = fes.mesh.nodes

    def f_loop(e, ul):
        return form.internal_force(X[elems[e]], ul, D, thickness=TH)

    def k_loop(e, ul):
        return form.tangent_stiffness(X[elems[e]], ul, D, thickness=TH)

    def f_batch(e, UL):
        return fes.element_forces_local(e, UL, mat, vectorized=True, thickness=TH)

    def k_batch(e, UL):
        return fes.element_tangents_local(e, UL, mat, vectorized=True, thickness=TH)

    return dict(fes=fes, S=S, V=V, dofs=dofs, f_loop=f_loop, k_loop=k_loop, f_batch=f_batch, k_batch=k_batch,
                n_el=len(elems))


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_the_fixture_is_genuinely_nonlinear(model):
    S = model["S"]
    assert np.max(np.abs(S)) > 0.2                         # large deflection relative to the 3 x 0.75 plate


def test_training_matrix_batched_equals_per_element(model):
    G0, b0 = ecsw_training_matrix(model["V"], model["S"], model["dofs"], model["f_loop"])
    G1, b1 = ecsw_training_matrix_batched(model["V"], model["S"], model["dofs"], model["f_batch"])
    assert G1.shape == G0.shape == (model["S"].shape[1] * 4, model["n_el"])
    assert _rel(G1, G0) < 1e-12 and _rel(b1, b0) < 1e-12


def test_ecsw_fit_batched_gives_the_same_weights_and_reduced_force_and_tangent(model):
    V, S, dofs = model["V"], model["S"], model["dofs"]
    ref = ECSW.fit(V, S, dofs, model["f_loop"], model["k_loop"], tol=1e-4)
    bat = ECSW.fit_batched(V, S, dofs, model["f_batch"], model["k_batch"], tol=1e-4)
    assert 0 < ref.n_selected < model["n_el"]
    assert np.array_equal(bat.elements, ref.elements)
    assert _rel(bat.weights, ref.weights) < 1e-8           # NNLS on matrices equal to 1e-15: weights agree far inside this
    rng = np.random.default_rng(0)
    for q in rng.normal(size=(4, 4)) * 0.3:
        assert _rel(bat.reduced_force(q), ref.reduced_force(q)) < 1e-9
        assert _rel(bat.reduced_tangent(q), ref.reduced_tangent(q)) < 1e-9
    # same weights, batched evaluation of the selected elements only
    mixed = ECSW(V, dofs, None, ref.weights, None, elem_forces_fn=model["f_batch"], elem_tangents_fn=model["k_batch"])
    q = rng.normal(size=4) * 0.3
    assert _rel(mixed.reduced_force(q), ref.reduced_force(q)) < 1e-12
    assert _rel(mixed.reduced_tangent(q), ref.reduced_tangent(q)) < 1e-12


def test_batched_callback_is_called_once_per_snapshot_and_once_per_reduced_force(model):
    calls = []

    def counting(e, UL):
        calls.append(len(e))
        return model["f_batch"](e, UL)

    ecsw_training_matrix_batched(model["V"], model["S"], model["dofs"], counting)
    assert calls == [model["n_el"]] * model["S"].shape[1]
    bat = ECSW.fit_batched(model["V"], model["S"], model["dofs"], model["f_batch"], tol=1e-4)
    calls.clear()
    bat.elem_forces_fn = counting
    bat.reduced_force(np.zeros(4))
    assert calls == [bat.n_selected]                        # only the selected elements are evaluated, in one call


def test_nonlinearity_indicator_batched_equals_per_element(model):
    S = model["S"][:, [1, 4, 7]]
    k0 = nonlinearity_indicator(model["dofs"], model["f_loop"], model["k_loop"], S)
    k1 = nonlinearity_indicator_batched(model["dofs"], model["f_batch"], model["k_batch"], S)
    assert k1.shape == k0.shape and k0.max() > 0.01          # the large-deflection states are visibly nonlinear
    assert _rel(k1, k0) < 1e-9


def test_batched_routes_reject_unequal_dof_counts():
    dofs = [np.arange(4), np.arange(6)]
    with pytest.raises(ValueError, match="same number of DOFs"):
        ecsw_training_matrix_batched(np.eye(6), np.ones((6, 2)), dofs, lambda e, U: U)
    with pytest.raises(ValueError, match="same number of DOFs"):
        nonlinearity_indicator_batched(dofs, lambda e, U: U, lambda e, U: U, np.ones((6, 2)))


def test_a_callback_returning_the_wrong_shape_is_reported(model):
    with pytest.raises(ValueError, match="elem_forces_fn returned shape"):
        ecsw_training_matrix_batched(model["V"], model["S"], model["dofs"], lambda e, UL: UL[:, :-1])


def test_ecsw_with_no_selected_elements_returns_zeros():
    V = np.eye(6)[:, :2]
    ecsw = ECSW(V, [np.arange(6)], None, np.zeros(1), None, elem_forces_fn=lambda e, U: U,
                elem_tangents_fn=lambda e, U: np.zeros((len(e), 6, 6)))
    assert np.array_equal(ecsw.reduced_force(np.ones(2)), np.zeros(2))
    assert np.array_equal(ecsw.reduced_tangent(np.ones(2)), np.zeros((2, 2)))

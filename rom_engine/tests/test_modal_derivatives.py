# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_modal_derivatives.py -- modal derivatives, selection and basis augmentation.

Strict checks on a synthetic system whose derivatives are known in closed form, then a realistic check on the
nonlinear clamped-clamped beam from fea_engine.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine.linear_dynamics import modal_superposition
from rom_engine.modal_derivatives import (
    ModalDerivatives, augment_with_derivatives, directional_stiffness_derivative, dynamic_modal_derivatives,
    modal_coordinates, modal_interaction_weights, select_modal_derivatives, static_modal_derivatives)
from nonlinear_beam_fixtures import nonlinear_beam_model, reduced_static_solve, synthetic_cubic_model


@pytest.fixture(scope="module")
def synth():
    s = synthetic_cubic_model(n=10)
    w2, Phi = eigh(s["K0"], s["M"])
    s["omega2"], s["Phi"] = w2[:3], Phi[:, :3]
    return s


def _exact_dK_phi(s, Phi, i, j):
    """dK/dq_j phi_i = T[:, :, k] contracted: sum_{b,c} T[a, b, c] phi_i[b] phi_j[c]."""
    return np.einsum("abc,b,c->a", s["T"], Phi[:, i], Phi[:, j])


def test_directional_derivative_matches_exact_tensor(synth):
    Phi = synth["Phi"]
    dK = directional_stiffness_derivative(synth["K_t"], Phi[:, 1], length_scale=1.0)
    exact = np.einsum("abc,c->ab", synth["T"], Phi[:, 1])
    assert np.linalg.norm(dK - exact) / np.linalg.norm(exact) < 1e-7


def test_static_modal_derivative_solves_K_theta_equals_minus_dK_phi(synth):
    Phi = synth["Phi"]
    md = static_modal_derivatives(synth["K0"], Phi, tangent_fn=synth["K_t"])
    assert isinstance(md, ModalDerivatives) and md.theta.shape == (10, 3, 3)
    for i in range(3):
        for j in range(3):
            res = synth["K0"] @ md.theta[:, i, j] + _exact_dK_phi(synth, Phi, i, j)
            assert np.linalg.norm(res) < 1e-6 * np.linalg.norm(_exact_dK_phi(synth, Phi, i, j))


def test_static_modal_derivatives_are_symmetric_and_asymmetry_is_reported(synth):
    md = static_modal_derivatives(synth["K0"], synth["Phi"], tangent_fn=synth["K_t"], symmetrize=False)
    assert md.asymmetry < 1e-6
    assert np.allclose(md.theta, np.transpose(md.theta, (0, 2, 1)), atol=1e-6 * np.abs(md.theta).max())
    assert md.n_tangent_evals == 2 * 3


def test_precomputed_dK_gives_the_same_derivatives(synth):
    Phi = synth["Phi"]
    dK = [np.einsum("abc,c->ab", synth["T"], Phi[:, j]) for j in range(3)]
    a = static_modal_derivatives(synth["K0"], Phi, dK=dK)
    b = static_modal_derivatives(synth["K0"], Phi, tangent_fn=synth["K_t"])
    assert np.allclose(a.theta, b.theta, atol=1e-6 * np.abs(a.theta).max())
    with pytest.raises(ValueError):
        static_modal_derivatives(synth["K0"], Phi)
    with pytest.raises(ValueError):
        static_modal_derivatives(synth["K0"], Phi, tangent_fn=synth["K_t"], dK=dK)


def test_dynamic_modal_derivative_matches_finite_difference_of_the_eigenvector(synth):
    Phi, w2 = synth["Phi"], synth["omega2"]
    md = dynamic_modal_derivatives(synth["K0"], synth["M"], w2, Phi, tangent_fn=synth["K_t"])
    eps = 1e-5
    i, j = 0, 1
    vecs = []
    for sgn in (+1, -1):
        Kp = synth["K_t"](sgn * eps * Phi[:, j])
        w, V = eigh(Kp, synth["M"])
        v = V[:, i]
        v = v * np.sign(v @ synth["M"] @ Phi[:, i])
        vecs.append(v)
    fd = (vecs[0] - vecs[1]) / (2 * eps)
    assert np.linalg.norm(md.theta[:, i, j] - fd) / np.linalg.norm(fd) < 1e-4
    # normalisation constraint phi_i^T M theta_ij = 0
    assert abs(Phi[:, i] @ synth["M"] @ md.theta[:, i, j]) < 1e-9


def test_dynamic_modal_derivatives_reject_repeated_eigenvalues(synth):
    with pytest.raises(ValueError):
        dynamic_modal_derivatives(synth["K0"], synth["M"], np.array([1.0, 1.0, 2.0]), synth["Phi"],
                                  tangent_fn=synth["K_t"])


def test_static_and_dynamic_derivatives_differ_but_are_comparable(synth):
    Phi, w2 = synth["Phi"], synth["omega2"]
    st = static_modal_derivatives(synth["K0"], Phi, tangent_fn=synth["K_t"])
    dy = dynamic_modal_derivatives(synth["K0"], synth["M"], w2, Phi, tangent_fn=synth["K_t"])
    assert dy.kind == "dynamic" and st.kind == "static"
    assert not np.allclose(st.theta, dy.theta)


# ---------------------------------------------------------------- selection
def test_interaction_weights_and_selection_rank_the_coupled_pair_first():
    t = np.linspace(0, 1, 401)
    Q = np.column_stack([np.sin(2 * np.pi * 3 * t), 0.05 * np.sin(2 * np.pi * 7 * t), 0.4 * np.cos(2 * np.pi * 3 * t)])
    W = modal_interaction_weights(Q, dt=t[1] - t[0])
    assert np.allclose(W, W.T)
    pairs = select_modal_derivatives(W, n_select=3)
    assert pairs[0] == (0, 0)           # the largest amplitude squared
    assert (1, 1) not in pairs          # the tiny mode never matters
    few = select_modal_derivatives(W, fraction=0.5)
    assert 1 <= len(few) <= 6
    allp = select_modal_derivatives(W, fraction=1.0)
    assert len(allp) == 6 and len(set(allp)) == 6
    with pytest.raises(ValueError):
        select_modal_derivatives(W)
    with pytest.raises(ValueError):
        select_modal_derivatives(W, n_select=2, fraction=0.5)


# ---------------------------------------------------------------- augmentation
def test_augmented_basis_is_M_orthonormal_and_orthogonal_to_the_modes(synth):
    Phi = synth["Phi"]
    M = synth["M"]
    md = static_modal_derivatives(synth["K0"], Phi, tangent_fn=synth["K_t"])
    B, info = augment_with_derivatives(Phi, md, M=M)
    assert B.shape[1] == 3 + info["n_added"]
    assert info["n_added"] <= 6 and info["n_candidates"] == 6
    assert np.allclose(B.T @ M @ B, np.eye(B.shape[1]), atol=1e-9)
    assert np.allclose(B[:, :3], Phi)
    capped, ci = augment_with_derivatives(Phi, md, M=M, max_vectors=2)
    assert ci["n_added"] == 2 and capped.shape[1] == 5
    none, ni = augment_with_derivatives(Phi, md, pairs=[], M=M)
    assert ni["n_added"] == 0 and none.shape[1] == 3


def test_augmentation_drops_vectors_already_in_the_span(synth):
    Phi = synth["Phi"]
    fake = np.zeros((Phi.shape[0], 3, 3))
    fake[:, 0, 0] = Phi[:, 1]            # lies inside span(Phi): must be dropped
    fake[:, 0, 1] = fake[:, 1, 0] = np.ones(Phi.shape[0])
    B, info = augment_with_derivatives(Phi, fake, pairs=[(0, 0), (0, 1)], M=synth["M"])
    assert info["n_added"] == 1


# ---------------------------------------------------------------- realistic beam
@pytest.fixture(scope="module")
def beam():
    return nonlinear_beam_model(n_elem=20, n_modes=3)


def test_beam_static_derivatives_are_symmetric_and_membrane_dominated(beam):
    Phi = beam["Phi"]
    L = beam["L"]
    md = static_modal_derivatives(beam["K0"], Phi, tangent_fn=beam["K_t"], length_scale=L, symmetrize=False)
    assert md.asymmetry < 1e-4
    # beam dofs per node: (u axial, v transverse, theta). Bending modes have almost no axial motion, while the
    # derivatives are mostly axial: the membrane freedom the modes lack.
    ax = np.arange(0, beam["n_free"], 3)
    frac_modes = np.linalg.norm(Phi[ax, 0]) / np.linalg.norm(Phi[:, 0])
    frac_md = np.linalg.norm(md.theta[ax, 0, 0]) / np.linalg.norm(md.theta[:, 0, 0])
    assert frac_modes < 1e-6 and frac_md > 0.9


def test_augmented_basis_beats_modes_only_on_a_large_deflection_static_solve(beam):
    Phi, M = beam["Phi"], beam["M"]
    L = beam["L"]
    md = static_modal_derivatives(beam["K0"], Phi, tangent_fn=beam["K_t"], length_scale=L)
    B, info = augment_with_derivatives(Phi, md, M=M)
    h = np.sqrt(12 * 8.333e-9 / 1e-4)          # beam thickness, the scale where stretching starts to matter
    mid = int(np.argmax(np.abs(Phi[:, 0])))
    # a load whose LINEAR response is exactly mode 1 with a mid deflection of 3 thicknesses, so the whole error of
    # a modes-only basis is the missing membrane freedom (not missing bending modes)
    a = 3 * h / Phi[mid, 0]
    F = beam["K0"] @ (a * Phi[:, 0])
    u_fom = beam["static"](F)
    u_modes = Phi @ reduced_static_solve(Phi, beam["f_int"], beam["K_t"], F)
    u_aug = B @ reduced_static_solve(B, beam["f_int"], beam["K_t"], F)
    err = lambda u: np.linalg.norm(u - u_fom) / np.linalg.norm(u_fom)
    assert err(u_aug) < 0.5 * err(u_modes)
    assert err(u_aug) < 0.05


def test_mmi_pipeline_on_the_beam_selects_a_smaller_basis(beam):
    Phi, M, K0 = beam["Phi"], beam["M"], beam["K0"]
    n = beam["n_free"]
    f0 = np.sqrt(beam["omega2"][0]) / (2 * np.pi)
    dt, steps = 1.0 / (f0 * 200), 800
    load = np.zeros((steps + 1, n))
    mid = int(np.argmax(np.abs(Phi[:, 0])))
    load[:, mid] = np.sin(2 * np.pi * f0 * np.arange(steps + 1) * dt)
    _, U, info = modal_superposition(K0, M, load, dt, steps, n_modes=3, zeta=0.01)
    Q = modal_coordinates(Phi, M, U)
    W = modal_interaction_weights(Q, dt)
    pairs = select_modal_derivatives(W, n_select=2)
    assert pairs[0] == (0, 0)
    md = static_modal_derivatives(K0, Phi, tangent_fn=beam["K_t"], length_scale=beam["L"])
    B_all, i_all = augment_with_derivatives(Phi, md, M=M)
    B_sel, i_sel = augment_with_derivatives(Phi, md, pairs=pairs, M=M)
    assert B_sel.shape[1] < B_all.shape[1]
    assert i_sel["n_candidates"] == 2

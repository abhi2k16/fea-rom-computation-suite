# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_dual_modes_eed.py -- dual modes and enhanced enforced-displacement identification.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import root

from rom_engine.dual_modes import amplitude_to_load_factor, dual_modes
from rom_engine.modal_derivatives import augment_with_derivatives, static_modal_derivatives
from rom_engine.nonlinear_rom import (EnhancedEnforcedDisplacementROM, PolynomialModalROM,
                                      enhanced_enforced_displacement_rom)
from nonlinear_beam_fixtures import nonlinear_beam_model, reduced_static_solve, synthetic_cubic_model


@pytest.fixture(scope="module")
def synth():
    s = synthetic_cubic_model(n=10)
    w2, P = eigh(s["K0"], s["M"])
    s["Phi"], s["omega2"] = P[:, :4], w2[:4]
    s["Kr"] = s["Phi"].T @ s["K0"] @ s["Phi"]
    return s


def _true_force(s, q):
    return s["Phi"].T @ s["f_int"](s["Phi"] @ q) - s["Kr"] @ q


# ---- EED
def test_eed_recovers_a_cubic_polynomial_exactly(synth):
    model, info = enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02, 0.04])
    assert isinstance(model, PolynomialModalROM)
    rng = np.random.default_rng(1)
    for _ in range(15):
        q = rng.uniform(-0.05, 0.05, 4)
        t = _true_force(synth, q)
        assert np.linalg.norm(model.force(q) - t) < 1e-9 * np.linalg.norm(t)
    assert info["residual"] < 1e-9


def test_pair_probes_identify_every_term_including_three_distinct_mode_cubics(synth):
    m = 4
    model2, i2 = enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02, 0.04], max_distinct=2)
    model3, i3 = enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02, 0.04], max_distinct=3)
    assert i2["rank"] == i2["n_terms"] == i3["rank"]
    assert np.allclose(model2.coeffs, model3.coeffs, atol=1e-8 * np.abs(model3.coeffs).max())
    # a state that excites three modes: the three-distinct cubic term must be reproduced
    q = np.array([0.03, -0.02, 0.025, 0.0])
    t = _true_force(synth, q)
    assert np.linalg.norm(model2.force(q) - t) < 1e-9 * np.linalg.norm(t)


def test_eed_cost_grows_quadratically_with_the_number_of_modes(synth):
    counts = {}
    for m in (2, 3, 4):
        _, info = enhanced_enforced_displacement_rom(synth["Phi"][:, :m], synth["K_t"], [0.02, 0.04])
        counts[m] = info["n_tangent_evals"]
    # singles (2 signs) + pairs (2 relative signs), per amplitude (2 amplitudes), plus the reference tangent
    for m, c in counts.items():
        assert c == 2 * (2 * m + m * (m - 1)) + 1
    _, i3 = enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02, 0.04], max_distinct=3)
    assert i3["n_tangent_evals"] > counts[4]


def test_eed_mode_scales_shape_the_probes_and_are_validated(synth):
    sc = np.array([1.0, 0.5, 0.25, 0.1])
    model, info = enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02, 0.04], mode_scales=sc)
    rng = np.random.default_rng(3)
    for _ in range(8):
        q = rng.uniform(-0.04, 0.04, 4)
        t = _true_force(synth, q)
        assert np.linalg.norm(model.force(q) - t) < 1e-9 * np.linalg.norm(t)
    # a callable gives the same probes as the equivalent array
    m2, _ = enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02, 0.04], mode_scales=lambda a: sc)
    assert np.allclose(m2.coeffs, model.coeffs)
    with pytest.raises(ValueError):
        enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02], mode_scales=np.ones(3))


def test_eed_validates_arguments_and_named_wrapper(synth):
    with pytest.raises(ValueError):
        enhanced_enforced_displacement_rom(synth["Phi"], synth["K_t"], [0.02], max_distinct=4)
    model = EnhancedEnforcedDisplacementROM(synth["Phi"], synth["K_t"], [0.02, 0.04])
    q = np.array([0.01, 0.02, -0.01, 0.0])
    assert np.allclose(model.force(q), _true_force(synth, q), atol=1e-8 * np.linalg.norm(_true_force(synth, q)))


# ---- dual modes
def _synthetic_fom_solver(s):
    def solve(F):
        u = np.zeros(len(F))
        for lam in np.linspace(0.0, 1.0, 21)[1:]:          # load ramp keeps Newton on the physical branch
            sol = root(lambda x: s["f_int"](x) - lam * F, u, jac=s["K_t"], tol=1e-12)
            assert sol.success
            u = sol.x
        return u
    return solve


def test_load_factor_gives_the_requested_linear_peak(synth):
    a = amplitude_to_load_factor(synth["Phi"][:, 0], 0.5)
    assert np.isclose(np.max(np.abs(a * synth["Phi"][:, 0])), 0.5)
    with pytest.raises(ValueError):
        amplitude_to_load_factor(np.zeros(3), 1.0)


def test_dual_modes_basis_is_M_orthonormal_with_a_solve_count(synth):
    B, info = dual_modes(synth["K0"], synth["M"], synth["Phi"][:, :3], _synthetic_fom_solver(synth),
                         amplitudes=[0.005, 0.02], dominant=0)
    assert np.allclose(B.T @ synth["M"] @ B, np.eye(B.shape[1]), atol=1e-8)
    assert np.allclose(B[:, :3], synth["Phi"][:, :3])
    assert info["n_solves"] == 3 * 2 and info["dominant"] == 0
    assert info["n_dual"] == B.shape[1] - 3 and info["n_dual"] >= 1
    assert np.all(np.diff(info["scores"]) <= 1e-12)


def test_dual_modes_need_a_dominant_mode_or_a_load_pattern(synth):
    with pytest.raises(ValueError):
        dual_modes(synth["K0"], synth["M"], synth["Phi"][:, :2], _synthetic_fom_solver(synth), [0.1])
    p = synth["M"] @ synth["Phi"][:, 1]
    _, info = dual_modes(synth["K0"], synth["M"], synth["Phi"][:, :2], _synthetic_fom_solver(synth), [0.005, 0.01],
                         p=p)
    assert info["dominant"] == 1


# ------------------------------------------------------------ realistic beam
@pytest.fixture(scope="module")
def beam():
    return nonlinear_beam_model(n_elem=20, n_modes=3)


def test_dual_modes_close_the_membrane_gap_on_the_beam(beam):
    Phi, M, K0 = beam["Phi"], beam["M"], beam["K0"]
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    B, info = dual_modes(K0, M, Phi, beam["static"], amplitudes=[0.5 * h, 3 * h, 6 * h], dominant=0)
    assert info["n_solves"] == 3 * 3
    mid = int(np.argmax(np.abs(Phi[:, 0])))
    errs = {"modes": [], "dual": []}
    for k in (1, 3, 6):
        F = K0 @ (k * h / Phi[mid, 0] * Phi[:, 0])
        uf = beam["static"](F)
        e = lambda u: np.linalg.norm(u - uf) / np.linalg.norm(uf)
        errs["modes"].append(e(Phi @ reduced_static_solve(Phi, beam["f_int"], beam["K_t"], F)))
        errs["dual"].append(e(B @ reduced_static_solve(B, beam["f_int"], beam["K_t"], F)))
    assert all(d < 0.35 * m for d, m in zip(errs["dual"], errs["modes"]))
    assert errs["dual"][-1] < 0.03


def test_eed_on_the_beam_with_a_derivative_augmented_basis(beam):
    Phi, M, K0 = beam["Phi"], beam["M"], beam["K0"]
    md = static_modal_derivatives(K0, Phi, tangent_fn=beam["K_t"], length_scale=beam["L"])
    B, _ = augment_with_derivatives(Phi, md, M=M, max_vectors=3)
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    Kr = B.T @ K0 @ B
    scale = h / np.max(np.abs(B[:, 0]))
    model, info = enhanced_enforced_displacement_rom(B, beam["K_t"], [0.5 * scale, 1.0 * scale], K_r=Kr)
    assert info["rank"] == info["n_terms"]
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, B.shape[1]) * scale * np.r_[np.ones(3), 0.1 * np.ones(B.shape[1] - 3)]
        t = B.T @ beam["f_int"](B @ q) - Kr @ q
        errs.append(np.linalg.norm(model.force(q) - t) / np.linalg.norm(t))
    assert max(errs) < 0.05

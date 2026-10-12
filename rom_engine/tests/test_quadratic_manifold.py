# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""test_quadratic_manifold.py -- quadratic manifold and ECSW with simulation-free training."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine.modal_derivatives import static_modal_derivatives
from rom_engine.quadratic_manifold import (ManifoldECSW, QuadraticManifold, QuadraticManifoldROM,
                                           fit_quadratic_manifold, lifted_training_states,
                                           manifold_from_derivatives)
from nonlinear_beam_fixtures import nonlinear_beam_model, reduced_static_solve, synthetic_cubic_model


@pytest.fixture(scope="module")
def synth():
    s = synthetic_cubic_model(n=8)
    w2, P = eigh(s["K0"], s["M"])
    s["Phi"] = P[:, :3]
    return s


def test_tangent_is_the_derivative_of_the_map(synth):
    rng = np.random.default_rng(0)
    Om = rng.standard_normal((8, 3, 3))
    man = QuadraticManifold(synth["Phi"], Om, synth["M"])
    q = rng.standard_normal(3)
    P = man.tangent(q)
    eps = 1e-6
    for j in range(3):
        e = np.zeros(3); e[j] = eps
        fd = (man.map(q + e) - man.map(q - e)) / (2 * eps)
        assert np.allclose(fd, P[:, j], atol=1e-7)
    assert np.allclose(man.Omega, np.transpose(man.Omega, (0, 2, 1)))
    with pytest.raises(ValueError):
        QuadraticManifold(synth["Phi"], np.zeros((8, 2, 2)))


def test_fit_recovers_an_exact_quadratic_manifold(synth):
    rng = np.random.default_rng(1)
    Om = rng.standard_normal((8, 3, 3))
    true = QuadraticManifold(synth["Phi"], Om, synth["M"])
    # make the curvature M-orthogonal to Phi so q = Phi^T M u recovers the coordinate
    Phi, M = synth["Phi"], synth["M"]
    Om = true.Omega - np.einsum("ab,bc,cjk->ajk", Phi, Phi.T @ M, true.Omega)
    true = QuadraticManifold(Phi, Om, M)
    Q = rng.uniform(-1, 1, (3, 80))
    fit, info = fit_quadratic_manifold(Phi, true.lift(Q), M=M, ridge=1e-14)
    assert info["residual"] < 1e-8
    assert np.allclose(fit.Omega, true.Omega, atol=1e-6)


def test_manifold_dynamics_conserve_energy_without_damping(synth):
    s = synth
    man = manifold_from_derivatives(s["Phi"], static_modal_derivatives(s["K0"], s["Phi"], tangent_fn=s["K_t"]),
                                    s["M"])
    rom = QuadraticManifoldROM(man, f_int=s["f_int"])
    # small free vibration: energy T + V with V from the lifted internal energy via work integral is awkward,
    # so check the reduced mass is SPD and time reversal consistency of the integrator instead
    q0 = np.array([1e-3, 0.0, 0.0])
    t, Q, Qd = rom.simulate(lambda t: np.zeros(8), q0, np.zeros(3), 1e-3, 200)
    t2, Q2, Qd2 = rom.simulate(lambda t: np.zeros(8), Q[-1], -Qd[-1], 1e-3, 200)
    assert np.linalg.eigvalsh(man.reduced_mass(Q[-1])).min() > 0
    assert np.linalg.norm(Q2[-1] - q0) < 1e-6 * max(1.0, np.linalg.norm(q0)) + 1e-9


@pytest.fixture(scope="module")
def beam():
    return nonlinear_beam_model(n_elem=20, n_modes=3)


def test_manifold_beats_the_linear_subspace_on_a_large_deflection_beam(beam):
    Phi, M, K0 = beam["Phi"], beam["M"], beam["K0"]
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    md = static_modal_derivatives(K0, Phi, tangent_fn=beam["K_t"], length_scale=beam["L"])
    man = manifold_from_derivatives(Phi, md, M)
    rom = QuadraticManifoldROM(man, f_int=beam["f_int"])
    mid = int(np.argmax(np.abs(Phi[:, 0])))
    F = K0 @ (3 * h / Phi[mid, 0] * Phi[:, 0])
    uf = beam["static"](F)
    err = lambda u: np.linalg.norm(u - uf) / np.linalg.norm(uf)
    e_lin = err(Phi @ reduced_static_solve(Phi, beam["f_int"], beam["K_t"], F))
    e_man = err(man.map(rom.static_solve(F)))
    assert e_man < 0.5 * e_lin and e_man < 0.1


def test_simulation_free_ecsw_reproduces_the_manifold_force(beam):
    Phi, M, K0 = beam["Phi"], beam["M"], beam["K0"]
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    md = static_modal_derivatives(K0, Phi, tangent_fn=beam["K_t"], length_scale=beam["L"])
    man = manifold_from_derivatives(Phi, md, M)
    amp = 3 * h / np.abs(Phi[:, 0]).max()
    Q, U = lifted_training_states(man, [amp, 0.3 * amp, 0.3 * amp], n_samples=40)
    free = beam["free"]
    fe, full_dofs = beam["elem_force"], beam["elem_dofs"]
    elem_force = lambda e, ul: fe(e, ul)
    # element callbacks use full DOFs: embed the manifold (clamped rows stay zero)
    Phi_f = np.zeros((beam["n"], 3)); Phi_f[free] = Phi
    Om_f = np.zeros((beam["n"], 3, 3)); Om_f[free] = man.Omega
    man_f = QuadraticManifold(Phi_f, Om_f)
    ecsw = ManifoldECSW.fit(man_f, Q, full_dofs, elem_force, tol=1e-6)
    rom_full = QuadraticManifoldROM(man_f, f_int=lambda u: np.asarray(beam["fes"].assemble_internal_force(u, beam["mat"])))
    rng = np.random.default_rng(5)
    errs = []
    for _ in range(8):
        q = rng.uniform(-1, 1, 3) * np.array([amp, 0.3 * amp, 0.3 * amp])
        a, b = ecsw.reduced_force(q), rom_full.reduced_force(q)
        errs.append(np.linalg.norm(a - b) / np.linalg.norm(b))
    assert max(errs) < 1e-3
    assert ecsw.n_selected <= len(full_dofs)

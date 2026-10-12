# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""test_slow_fast.py -- slow-fast condensation of membrane against bending coordinates."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.modal_derivatives import augment_with_derivatives, static_modal_derivatives
from rom_engine.slow_fast import SlowFastROM
from nonlinear_beam_fixtures import nonlinear_beam_model, reduced_static_solve


@pytest.fixture(scope="module")
def setup():
    b = nonlinear_beam_model(n_elem=20, n_modes=3)
    Phi, M, K0 = b["Phi"], b["M"], b["K0"]
    md = static_modal_derivatives(K0, Phi, tangent_fn=b["K_t"], length_scale=b["L"])
    B, _ = augment_with_derivatives(Phi, md, M=M)
    Vf = B[:, 3:]
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    mid = int(np.argmax(np.abs(Phi[:, 0])))
    F = K0 @ (3 * h / Phi[mid, 0] * Phi[:, 0])
    return dict(b=b, Phi=Phi, Vf=Vf, B=B, F=F)


def test_condensation_equals_the_coupled_galerkin_solve(setup):
    b, B, F = setup["b"], setup["B"], setup["F"]
    sf = SlowFastROM(setup["Phi"], setup["Vf"], b["f_int"], b["K_t"])
    q = sf.static_solve(F)
    u_sf = sf.state(q, F)
    u_g = B @ reduced_static_solve(B, b["f_int"], b["K_t"], F)
    assert np.linalg.norm(u_sf - u_g) / np.linalg.norm(u_g) < 1e-8


def test_condensed_tangent_is_the_derivative_of_the_condensed_force(setup):
    b, F = setup["b"], setup["F"]
    sf = SlowFastROM(setup["Phi"], setup["Vf"], b["f_int"], b["K_t"])
    q = sf.static_solve(F) * 0.7
    K = sf.reduced_tangent(q, F)
    eps = 1e-6 * np.linalg.norm(q)
    for j in range(3):
        e = np.zeros(3); e[j] = eps
        fd = (sf.reduced_force(q + e, F) - sf.reduced_force(q - e, F)) / (2 * eps)
        assert np.linalg.norm(fd - K[:, j]) < 1e-5 * np.linalg.norm(K[:, j]) + 1e-6
    assert np.allclose(K, K.T, atol=1e-6 * np.abs(K).max())


def test_condensation_beats_slow_modes_alone(setup):
    b, F = setup["b"], setup["F"]
    uf = b["static"](F)
    err = lambda u: np.linalg.norm(u - uf) / np.linalg.norm(uf)
    sf = SlowFastROM(setup["Phi"], setup["Vf"], b["f_int"], b["K_t"])
    e_sf = err(sf.state(sf.static_solve(F), F))
    e_lin = err(setup["Phi"] @ reduced_static_solve(setup["Phi"], b["f_int"], b["K_t"], F))
    assert e_sf < 0.5 * e_lin and sf.n_fast_solves > 0

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""test_parametric_compare.py -- mode sensitivities and the basis comparison harness."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine.dual_modes import dual_modes
from rom_engine.modal_derivatives import augment_with_derivatives, static_modal_derivatives
from rom_engine.parametric_modes import mode_sensitivities, updated_modes
from rom_engine.validation import break_even_queries, compare_bases, effective_speedup
from nonlinear_beam_fixtures import nonlinear_beam_model, reduced_static_solve


def _pencil(n=9, seed=4):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)); B = rng.standard_normal((n, n))
    dA = rng.standard_normal((n, n)); dB = rng.standard_normal((n, n))
    K0 = A @ A.T + n * np.eye(n); M0 = B @ B.T + n * np.eye(n)
    return K0, M0, 0.5 * (dA + dA.T), 0.1 * (dB @ dB.T)


def test_eigenvalue_and_mode_sensitivities_match_finite_differences():
    K0, M0, dK, dM = _pencil()
    d_w2, dPhi, info = mode_sensitivities(K0, M0, dK, dM, n_modes=3)
    eps = 1e-6
    ref = []
    for sgn in (+1, -1):
        w2, P = eigh(K0 + sgn * eps * dK, M0 + sgn * eps * dM)
        P = P[:, :3] * np.sign(np.sum(P[:, :3] * (eigh(K0, M0)[1][:, :3]), axis=0))
        ref.append((w2[:3], P))
    fd_w2 = (ref[0][0] - ref[1][0]) / (2 * eps)
    fd_P = (ref[0][1] - ref[1][1]) / (2 * eps)
    assert np.allclose(d_w2, fd_w2, rtol=1e-5)
    assert np.allclose(dPhi, fd_P, rtol=1e-4, atol=1e-6)
    # normalisation is preserved to first order: phi^T M dphi + 1/2 phi^T dM phi = 0
    P0 = eigh(K0, M0)[1][:, :3]
    for i in range(3):
        assert abs(P0[:, i] @ M0 @ dPhi[:, i] + 0.5 * P0[:, i] @ dM @ P0[:, i]) < 1e-9


def test_truncated_set_form_and_repeated_eigenvalue_rejection():
    K0, M0, dK, dM = _pencil()
    w2, P = eigh(K0, M0)
    _, dP_full, _ = mode_sensitivities(K0, M0, dK, dM, n_modes=2)
    _, dP_tr, _ = mode_sensitivities(K0, M0, dK, dM, n_modes=2, exact=False, modes=P[:, :5], omega2=w2[:5])
    assert np.linalg.norm(dP_tr - dP_full) / np.linalg.norm(dP_full) < 1.0
    _, dP_all, _ = mode_sensitivities(K0, M0, dK, dM, n_modes=2, exact=False, modes=P, omega2=w2)
    assert np.allclose(dP_all, dP_full)
    with pytest.raises(ValueError):
        mode_sensitivities(K0, M0, dK, exact=False)
    with pytest.raises(ValueError):
        mode_sensitivities(np.eye(4), np.eye(4), np.eye(4), n_modes=2)


def test_updated_modes_track_the_perturbed_modes_to_first_order():
    K0, M0, dK, dM = _pencil()
    _, dPhi, _ = mode_sensitivities(K0, M0, dK, dM, n_modes=2)
    P0 = eigh(K0, M0)[1][:, :2]
    errs = []
    for h in (1e-2, 1e-3):
        w2, P = eigh(K0 + h * dK, M0 + h * dM)
        P = P[:, :2] * np.sign(np.sum(P[:, :2] * P0, axis=0))
        U = updated_modes(P0, dPhi, h, M0 + h * dM)
        U = U * np.sign(np.sum(U * P, axis=0))
        errs.append(np.linalg.norm(U - P))
    assert errs[1] < 0.2 * errs[0]            # error shrinks faster than linearly (second order)


def test_effective_speedup_and_break_even():
    assert effective_speedup(100, 0.01, 0.0, 1.0) == pytest.approx(100.0)
    assert effective_speedup(100, 0.01, 1.0, 1.0) == pytest.approx(50.0)
    assert break_even_queries(0.01, 99.0, 1.0) == 100
    assert break_even_queries(2.0, 1.0, 1.0) is None


def test_compare_bases_on_the_beam_reports_every_basis():
    b = nonlinear_beam_model(n_elem=10, n_modes=3)
    Phi, M, K0 = b["Phi"], b["M"], b["K0"]
    md = static_modal_derivatives(K0, Phi, tangent_fn=b["K_t"], length_scale=b["L"])
    Bmd, _ = augment_with_derivatives(Phi, md, M=M)
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    Bdu, _ = dual_modes(K0, M, Phi, b["static"], [0.5 * h, 3 * h], dominant=0)
    mid = int(np.argmax(np.abs(Phi[:, 0])))
    loads = [K0 @ (k * h / Phi[mid, 0] * Phi[:, 0]) for k in (2, 3)]
    pred = lambda V: (lambda F: V @ reduced_static_solve(V, b["f_int"], b["K_t"], F))
    out = compare_bases({"modes": Phi, "modal derivatives": Bmd, "dual modes": Bdu}, b["static"], pred, loads,
                        offline_times={"modal derivatives": 0.5, "dual modes": 1.0})
    assert set(out) == {"modes", "modal derivatives", "dual modes"}
    assert out["modes"]["size"] == 3 and out["modal derivatives"]["size"] > 3
    assert out["modal derivatives"]["max"] < out["modes"]["max"]
    assert out["dual modes"]["max"] < out["modes"]["max"]
    assert all(np.isfinite(v["effective_speedup"]) for v in out.values())

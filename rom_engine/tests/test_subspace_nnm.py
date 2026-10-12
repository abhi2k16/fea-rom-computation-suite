# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""test_subspace_nnm.py -- NNM backbone in a modal-derivative subspace, checked against an
independent full-order free-vibration period measurement."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.modal_derivatives import augment_with_derivatives, static_modal_derivatives
from rom_engine.subspace_nnm import diagonalize_basis, nnm_backbone_in_subspace
from nonlinear_beam_fixtures import nonlinear_beam_model


@pytest.fixture(scope="module")
def case():
    b = nonlinear_beam_model(n_elem=10, n_modes=2)
    Phi, M, K0 = b["Phi"], b["M"], b["K0"]
    md = static_modal_derivatives(K0, Phi, tangent_fn=b["K_t"], length_scale=b["L"])
    B, _ = augment_with_derivatives(Phi, md, M=M, max_vectors=3)
    return b, B


def test_diagonalised_basis_is_modal(case):
    b, B = case
    W, w2 = diagonalize_basis(B, b["K0"], b["M"])
    assert np.allclose(W.T @ b["M"] @ W, np.eye(W.shape[1]), atol=1e-9)
    assert np.allclose(W.T @ b["K0"] @ W, np.diag(w2), atol=1e-6 * w2.max())


def _fom_frequency(b, W, a, steps_per_period=200, periods=4):
    """Free vibration of the FULL model released from the first mode shape at modal amplitude a (average-
    acceleration Newmark with Newton iterations); frequency from the upward zero crossings of the first modal
    coordinate. An independent time-domain check, not the harmonic-balance solver checked against itself."""
    M, n = b["M"], b["n_free"]
    w0 = np.sqrt(W[:, 0] @ b["K0"] @ W[:, 0])
    T = 2 * np.pi / w0
    dt = T / steps_per_period
    u = a * W[:, 0]; v = np.zeros(n)
    acc = np.linalg.solve(M, -b["f_int"](u))
    hist = [(W[:, 0] @ M) @ u]
    for _ in range(periods * steps_per_period):
        un = u + dt * v + 0.25 * dt ** 2 * acc
        for _it in range(30):
            an = (un - u - dt * v) * 4 / dt ** 2 - acc
            r = M @ an + b["f_int"](un)
            if np.linalg.norm(r) < 1e-10 * max(np.linalg.norm(M @ an), 1e-30) + 1e-12:
                break
            un = un - np.linalg.solve(4 / dt ** 2 * M + b["K_t"](un), r)
        an = (un - u - dt * v) * 4 / dt ** 2 - acc
        v = v + 0.5 * dt * (acc + an)
        u, acc = un, an
        hist.append((W[:, 0] @ M) @ u)
    q = np.array(hist)
    t = np.arange(len(q)) * dt
    up = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    tc = t[up] - q[up] * dt / (q[up + 1] - q[up])
    return 2 * np.pi / np.mean(np.diff(tc))


def test_backbone_hardens_and_tracks_the_full_model(case):
    b, B = case
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    W, _ = diagonalize_basis(B, b["K0"], b["M"])
    scale = h / np.abs(W[:, 0]).max()                      # modal amplitude giving one thickness
    out = nnm_backbone_in_subspace(B, b["K0"], b["M"], b["K_t"], [0.5 * scale, 1.0 * scale, 2.0 * scale],
                                   (0.1 * scale, 2.0 * scale), n_points=4)
    assert out["converged"].all()
    om = out["omegas"]
    assert np.all(np.diff(om) > 0) and om[0] < 1.05 * out["omega_linear"]
    assert om[-1] > 1.02 * out["omega_linear"]
    for a, w in zip(out["amplitudes"][[1, -1]], om[[1, -1]]):
        w_fom = _fom_frequency(b, out["basis"], a)
        assert abs(w - w_fom) / w_fom < 0.03

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
subspace_nnm.py -- nonlinear normal mode (NNM) backbone of a reduced model built in an
augmented subspace (vibration modes plus modal derivatives or dual modes). After Sombroek, Tiso, Renson and
Kerschen (2018) and Tiso, Karamooz Mahdiabadi and Marconi (2021), Section 4.5.6.

This is deliberately thin glue between existing pieces: the harmonic-balance continuation of `nnm.py` expects a
DIAGONAL modal stiffness Lambda with M_r = I and a `ReducedForceModel`, while an augmented basis has a full
reduced stiffness. `diagonalize_basis` rotates the basis to the eigenvectors of its own reduced pencil (K_r, M_r),
which gives exactly that form, and `nnm_backbone_in_subspace` identifies the cubic reduced force by enhanced
enforced displacement (`nonlinear_rom.enhanced_enforced_displacement_rom`, tangent probes only) and hands both to
`solve_nnm_backbone`.

WHAT IS NOT CLAIMED. The backbone is for the identified polynomial force, so it is valid only inside the
amplitude range the identification used. Agreement with a full-order backbone is a property to be checked on each
model (the test suite does so for one beam), not guaranteed.

References (methods in this module are adopted from):
  - G. Kerschen, M. Peeters, J. C. Golinval, A. F. Vakakis (2009), "Nonlinear normal modes, Part I: A useful framework for the structural dynamicist", Mech. Syst. Signal Process. 23(1)
  - P. Tiso, M. Karamooz Mahdiabadi, J. Marconi (2021), "Modal methods for reduced order modeling", Chapter 4, ETH Research Collection, doi:10.3929/ethz-b-000526692
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh

from .modal_derivatives import _as_array, _matmul
from .nnm import HarmonicBalanceSystem, solve_nnm_backbone
from .nonlinear_rom import enhanced_enforced_displacement_rom


def diagonalize_basis(V, K, M):
    """Rotate basis V so that V^T M V = I and V^T K V = diag(omega^2). Returns (W, omega2)."""
    V = _as_array(V)
    Kr = V.T @ _matmul(K, V)
    Mr = V.T @ _matmul(M, V)
    w2, X = eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    return V @ X, w2


def nnm_backbone_in_subspace(V, K, M, tangent_fn, eed_amplitudes, q_amplitude_range, n_points=6, n_harmonics=3,
                             master_mode=0, max_distinct=2):
    """Backbone of the reduced model on span(V). `eed_amplitudes` are the modal-coordinate amplitudes used to
    identify the reduced force (they should cover `q_amplitude_range`). Returns a dict with `amplitudes`,
    `omegas` (rad/s), `converged`, `omega_linear`, `basis` (the diagonalised basis) and `eed_info`."""
    W, w2 = diagonalize_basis(V, K, M)
    model, info = enhanced_enforced_displacement_rom(W, tangent_fn, eed_amplitudes, max_distinct=max_distinct,
                                                     K_r=np.diag(w2))
    hb = HarmonicBalanceSystem(w2, n_harmonics=n_harmonics)
    amps, om, Z, conv = solve_nnm_backbone(hb, model, q_amplitude_range, n_points=n_points, domain="q_nl",
                                           master_mode=master_mode)
    return dict(amplitudes=amps, omegas=om, converged=conv, omega_linear=float(np.sqrt(w2[master_mode])),
                basis=W, eed_info=info)

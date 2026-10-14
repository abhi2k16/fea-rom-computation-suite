# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
parametric_modes.py -- first-order sensitivity of vibration modes to a parameter, for example a
shape imperfection amplitude, a thickness or a material constant. After Nelson (1976) and the imperfection
treatment in Tiso, Karamooz Mahdiabadi and Marconi (2021), Section 4.7.

For the pencil (K(p), M(p)) with M-orthonormal modes (K - lambda_i M) phi_i = 0, differentiating gives
        d lambda_i = phi_i^T (dK - lambda_i dM) phi_i,
        d phi_i    = sum_{j != i} c_ij phi_j + c_ii phi_i,   c_ij = phi_j^T (dK - lambda_i dM) phi_i / (lambda_i - lambda_j),
        c_ii = -1/2 phi_i^T dM phi_i   (keeps phi_i^T M phi_i = 1).
Here the sum runs over a supplied set of modes (the retained modes plus, optionally, more), so the result is the
projection of d phi_i onto that set (modal-truncation form of Nelson's method), which is exact only if the set is
complete. `mode_sensitivities` can also take `exact=True` with the full eigen-decomposition.

`updated_modes` applies the first-order change (phi + p d phi, re-orthonormalised) as a cheap parametric basis
update; it is not claimed accurate outside the linear range of p.

WHAT IS NOT CLAIMED. Repeated or close eigenvalues make c_ij blow up; they are rejected (relative gap `gap_tol`).

References (methods in this module are adopted from):
  - R. B. Nelson (1976), "Simplified calculation of eigenvector derivatives", AIAA J. 14(9)
  - P. Tiso, M. Karamooz Mahdiabadi, J. Marconi (2021), "Modal methods for reduced order modeling", Chapter 4, ETH Research Collection, doi:10.3929/ethz-b-000526692
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh

from .modal_derivatives import _as_array, _matmul


def mode_sensitivities(K, M, dK, dM=None, n_modes=None, exact=True, modes=None, omega2=None, gap_tol=1e-8):
    """Return (d_omega2, dPhi, info) for the lowest `n_modes` modes. With `exact=True` (default) the full
    eigen-decomposition of (K, M) is used (dense; use for moderate sizes). With `exact=False` supply `modes`
    (n, k) and `omega2` (k,) and the truncated-set form is returned."""
    K, M, dK = _as_array(K), _as_array(M), _as_array(dK)
    dM = np.zeros_like(M) if dM is None else _as_array(dM)
    if exact:
        w2, P = eigh(K, M)
        omega2_all, modes_all = w2, P
    else:
        if modes is None or omega2 is None:
            raise ValueError("exact=False needs `modes` and `omega2`")
        modes_all, omega2_all = _as_array(modes), _as_array(omega2)
    k = modes_all.shape[1] if n_modes is None else int(n_modes)
    d_w2 = np.zeros(k)
    dPhi = np.zeros((modes_all.shape[0], k))
    worst_gap = np.inf
    for i in range(k):
        phi = modes_all[:, i]
        A = dK - omega2_all[i] * dM
        d_w2[i] = phi @ A @ phi
        coeff = np.zeros(modes_all.shape[1])
        for j in range(modes_all.shape[1]):
            if j == i:
                coeff[j] = -0.5 * (phi @ dM @ phi)
                continue
            gap = omega2_all[i] - omega2_all[j]
            rel = abs(gap) / max(abs(omega2_all[i]), 1e-300)
            worst_gap = min(worst_gap, rel)
            if rel < gap_tol:
                raise ValueError(f"modes {i} and {j} have (nearly) repeated eigenvalues; sensitivities are not "
                                 "defined by Nelson's formula")
            coeff[j] = (modes_all[:, j] @ A @ phi) / gap
        dPhi[:, i] = modes_all @ coeff
    return d_w2, dPhi, dict(min_relative_gap=float(worst_gap), n_modes_in_sum=modes_all.shape[1], exact=exact)


def updated_modes(modes, dPhi, step, M=None):
    """First-order updated modes phi + step * dPhi, re-orthonormalised against M (Euclidean if None)."""
    V = _as_array(modes) + step * _as_array(dPhi)
    G = V.T @ (_matmul(M, V) if M is not None else V)
    lam, X = np.linalg.eigh(0.5 * (G + G.T))
    return V @ (X / np.sqrt(lam)) @ X.T

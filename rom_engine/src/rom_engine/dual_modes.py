# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
dual_modes.py -- dual modes, the non-intrusive
way to give a modal basis the membrane (in-plane) freedom that vibration modes of a thin structure lack. After
Kim, Radu, Wang and Mignolet (2013) and Mignolet, Przekop, Rizzi and Spottswood (2013), in the form summarised in
Tiso, Karamooz Mahdiabadi and Marconi (2021), Section 4.5.3.

THE IDEA. Apply a series of static loads to the FULL nonlinear model, from loads small enough to give a linear
response to loads large enough to stretch the structure, and keep what the nonlinear responses contain beyond the
vibration modes. Unlike modal derivatives (modal_derivatives.py) this needs only the full-order model as a plain
callable `fom_solver(load) -> displacement` -- the same contract the existing training strategies in
`nonlinear_rom.py` use -- never its stiffness derivatives.

THE PROCEDURE (for M-orthonormal vibration modes phi_1 ... phi_m)
  1. Dominant mode d = argmax_i |p^T phi_i| for an external load pattern p (or given).
  2. Scaling factors alpha_s that run from a linear to a strongly nonlinear response. They are chosen here from
     target response amplitudes, alpha_s = amplitude_s / max|phi_d|, since the linear response to alpha K phi_d is
     alpha phi_d.
  3. For each mode j the load set  g_s = alpha_s K (phi_d + phi_j) / 2  is applied to the nonlinear model and the
     responses x_s are collected.
  4. Each x_s is made M-orthogonal to the modes, which removes the linear contribution.
  5. An SVD of the collected responses for each j gives candidate shapes psi (largest singular values kept).
  6. Each candidate is scored by a linearised strain-energy measure E = (psi^T K psi) sum_s beta_s^2 with
     beta_s = psi^T x_s / (psi^T psi), the energy the nonlinear responses put into that shape.
  7. The highest scores over all j are kept and M-orthonormalised.

HONEST NOTES. Steps 1 to 4 and 7 follow the chapter. Step 5 keeps the singular vectors with the LARGEST singular
values (the OCR of the chapter's equation is ambiguous on this point), and the exact scoring in step 6 is
reconstructed from the chapter's partly garbled equation 4.66; the original papers (Kim et al. 2013, Perez et al.
2014) should be consulted before relying on the ranking for anything but a comparison of bases. The cost is
m * len(alphas) nonlinear static solves of the full model.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp

from .modal_derivatives import _as_array, _matmul


def amplitude_to_load_factor(mode, amplitude):
    """alpha such that the linear response alpha * mode has peak absolute value `amplitude`."""
    peak = float(np.max(np.abs(_as_array(mode))))
    if peak == 0.0:
        raise ValueError("mode is zero")
    return amplitude / peak


def dual_modes(K, M, modes, fom_solver, amplitudes, p=None, dominant=None, n_dual=None, n_keep_per_set=None,
               tol=1e-10):
    """Compute dual modes and return the augmented basis [modes, duals].

    Parameters
    ----------
    K, M : stiffness and mass (dense or sparse), the same DOF set as `modes` and `fom_solver`.
    modes : (n, m) M-orthonormal vibration modes.
    fom_solver : callable(load (n,)) -> displacement (n,), the nonlinear static response of the full model.
    amplitudes : peak response amplitudes of the dominant mode that the load factors are built from, from small
        (linear response) to large (strongly nonlinear), in the model's length unit. At least two are advisable.
    p : external load pattern used to pick the dominant mode (step 1); ignored if `dominant` is given.
    dominant : index of the dominant mode, overriding `p`.
    n_dual : how many dual modes to keep. Default: m (one per retained mode).
    n_keep_per_set : singular vectors kept from the response set of each j. Default: min(len(amplitudes), 2).
    tol : relative singular-value cut-off.

    Returns (basis, info). `info` has `dominant`, `alphas`, `n_solves`, `scores` (descending), `n_candidates`,
    `n_dual`, and `duals` (the M-orthonormal dual-mode matrix).
    """
    Phi = _as_array(modes)
    n, m = Phi.shape
    if dominant is None:
        if p is None:
            raise ValueError("give the load pattern p or the dominant mode index")
        dominant = int(np.argmax(np.abs(Phi.T @ _as_array(p))))
    amplitudes = [float(a) for a in amplitudes]
    alphas = np.array([amplitude_to_load_factor(Phi[:, dominant], a) for a in amplitudes])
    keep_k = int(min(len(alphas), 2)) if n_keep_per_set is None else int(n_keep_per_set)
    Mm = lambda B: _matmul(M, B)
    Kphi = lambda v: _matmul(K, v.reshape(-1, 1))[:, 0]
    cands, scores = [], []
    n_solves = 0
    for j in range(m):
        shape = 0.5 * (Phi[:, dominant] + Phi[:, j])
        Kshape = Kphi(shape)
        X = np.column_stack([fom_solver(a * Kshape) for a in alphas])
        n_solves += len(alphas)
        X = X - Phi @ (Phi.T @ Mm(X))                    # step 4: remove the linear (modal) part
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        ok = s > tol * max(float(s[0]), 1e-300)
        for k in range(min(keep_k, int(ok.sum()))):
            psi = U[:, k]
            beta = (psi @ X) / (psi @ psi)
            scores.append(float(psi @ Kphi(psi)) * float(np.sum(beta ** 2)))
            cands.append(psi)
    if not cands:
        return Phi.copy(), dict(dominant=dominant, alphas=alphas, n_solves=n_solves, scores=np.zeros(0),
                                n_candidates=0, n_dual=0, duals=np.zeros((n, 0)))
    order = np.argsort(-np.array(scores), kind="stable")
    n_dual = m if n_dual is None else int(n_dual)
    chosen = np.column_stack([cands[k] for k in order[:n_dual]])
    # M-orthogonalise against the modes and among themselves (Gram-matrix route, like the derivative basis)
    chosen = chosen - Phi @ (Phi.T @ Mm(chosen))
    G = chosen.T @ Mm(chosen)
    lam, W = np.linalg.eigh(0.5 * (G + G.T))
    lam, W = lam[::-1], W[:, ::-1]
    keep = lam > 1e-10 * max(float(lam[0]), 1e-300)
    duals = chosen @ (W[:, keep] / np.sqrt(lam[keep]))
    info = dict(dominant=dominant, alphas=alphas, n_solves=n_solves, scores=np.array(scores)[order],
                n_candidates=len(cands), n_dual=int(duals.shape[1]), duals=duals)
    return np.hstack([Phi, duals]), info

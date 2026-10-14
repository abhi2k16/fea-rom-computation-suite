# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
operator_inference.py -- second-order operator inference with Craig-Bampton structure, for contact dynamics.

Learn reduced mass and stiffness operators from data of the full model WITHOUT access to its matrices (only
states, accelerations and applied forces), then use them for contact dynamics through an LCP (after
Manvelyan-Stroot et al. 2025).

Reduced coordinates are Craig-Bampton-like:  x = [u_b ; q],  u_b the boundary DOFs (where contact can occur, kept
FULL and physical) and q the amplitudes of interior fixed-boundary modes Phi:

    u_interior = Psi u_b + Phi q,        T = [[I, 0], [Psi, Phi]]  (rows ordered boundary, interior).

The coupling matrix Psi can come from static solves (`static_coupling`: one interior solve per boundary DOF, exact
for a linear model) or from a least-squares fit of the data
(`lstsq_coupling`, which only sees the motion present in the data).

Operators: minimise  sum_t || M a_t + K x_t - f_t ||^2  over symmetric positive definite M and K, parametrised
through Cholesky factors M = L_M L_M^T, K = L_K L_K^T (no semidefinite-programming package needed). The
unconstrained least-squares solution is the starting point; if it is already SPD the constrained fit stays on it.

Contact: with the gap map B (m x n_full, acting on the full vector, or directly on the boundary DOFs), the
two-step scheme  M (x_{n+1} - 2 x_n + x_{n-1}) + h^2 K x_{n+1} = h^2 (f_{n+1} + B_r^T lam)  with
B_r x_{n+1} >= c is reduced to an LCP in lam (same scheme as `fea_engine.contact_solvers.solve_contact_dynamics`).

Scope, stated honestly: linear structure; the contact node set must be known when the operators are learned (the
boundary DOFs are chosen up front); no damping term; the data must excite both the boundary and the interior.

References (methods in this module are adopted from):
  - B. Peherstorfer, K. Willcox (2016), "Data-driven operator inference for nonintrusive projection-based model reduction", Comput. Methods Appl. Mech. Engrg. 306
  - D. Manvelyan-Stroot, Y. Filanova, I. Pontes Duff, P. Benner, U. Wever (2025), "Inference of Substructured Reduced-Order Models for Dynamic Contact from Contact-free Simulations", arXiv:2505.18050
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.optimize import least_squares

from .complementarity import solve_lcp

__all__ = ["static_coupling", "lstsq_coupling", "craig_bampton_transformation", "fit_spd_operators",
           "OperatorInferenceModel", "infer_contact_operators"]


def static_coupling(solve_interior, n_boundary):
    """Psi from static solves: `solve_interior(e_j)` must return the interior response to a unit displacement of
    boundary DOF j (with the other boundary DOFs fixed); this is Psi[:, j]."""
    cols = []
    for j in range(n_boundary):
        e = np.zeros(n_boundary)
        e[j] = 1.0
        cols.append(np.asarray(solve_interior(e), float))
    return np.column_stack(cols)


def lstsq_coupling(U_interior, U_boundary, Phi):
    """Psi minimising the part of the interior data not spanned by Phi:  (I - Phi Phi^+)(U_I - Psi U_b)."""
    Q = np.eye(Phi.shape[0]) - Phi @ np.linalg.pinv(Phi)
    return (Q @ np.asarray(U_interior, float)) @ np.linalg.pinv(np.asarray(U_boundary, float))


def craig_bampton_transformation(boundary, n_full, Phi, Psi):
    """T (n_full, n_b + r) with u = T [u_b; q]; `Phi` and `Psi` are given on the interior DOFs (ascending order of
    the DOFs not in `boundary`)."""
    boundary = np.asarray(boundary, int)
    interior = np.setdiff1d(np.arange(n_full), boundary)
    nb, r = len(boundary), Phi.shape[1]
    T = np.zeros((n_full, nb + r))
    T[boundary, :nb] = np.eye(nb)
    T[interior, :nb] = Psi
    T[interior, nb:] = Phi
    return T


def _tril_params(L):
    return L[np.tril_indices(L.shape[0])]


def _tril_matrix(p, n):
    L = np.zeros((n, n))
    L[np.tril_indices(n)] = p
    return L


def fit_spd_operators(X, A, F, tol=1e-12, max_nfev=200, floor=1e-8):
    """Fit symmetric positive definite (M, K) to  M A + K X = F  (columns are time samples) in the least-squares
    sense. Returns (M, K, info) (both strictly positive definite: a relative `floor` times the mean diagonal is added) with info['unconstrained'] the (M0, K0) of the plain least-squares fit,
    info['spd_unconstrained'] whether it was already SPD, and info['residual'] the relative data residual."""
    X, A, F = (np.asarray(v, float) for v in (X, A, F))
    n = X.shape[0]
    D = np.vstack([A, X])                                   # (2n, nt)
    W = np.linalg.lstsq(D.T, F.T, rcond=None)[0].T          # F ~ [M0 K0] D
    M0, K0 = W[:, :n], W[:, n:]
    M0, K0 = 0.5 * (M0 + M0.T), 0.5 * (K0 + K0.T)
    spd0 = bool(np.linalg.eigvalsh(M0).min() > 0 and np.linalg.eigvalsh(K0).min() > 0)
    fnorm = max(np.linalg.norm(F), 1e-300)

    def chol(S):
        w, V = np.linalg.eigh(S)
        S = (V * np.clip(w, 1e-8 * max(abs(w).max(), 1e-300), None)) @ V.T
        return np.linalg.cholesky(0.5 * (S + S.T))

    p0 = np.concatenate([_tril_params(chol(M0)), _tril_params(chol(K0))])
    m = n * (n + 1) // 2

    def resid(p):
        LM, LK = _tril_matrix(p[:m], n), _tril_matrix(p[m:], n)
        return ((LM @ LM.T) @ A + (LK @ LK.T) @ X - F).reshape(-1) / fnorm

    if spd0:
        p = p0
    else:
        p = least_squares(resid, p0, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev, method="trf").x
    LM, LK = _tril_matrix(p[:m], n), _tril_matrix(p[m:], n)
    M, K = LM @ LM.T, LK @ LK.T
    M = M + floor * np.trace(M) / n * np.eye(n)             # strictly positive definite, not just to round-off
    K = K + floor * np.trace(K) / n * np.eye(n)
    res = float(np.linalg.norm(M @ A + K @ X - F) / fnorm)
    return M, K, dict(unconstrained=(M0, K0), spd_unconstrained=spd0, residual=res)


class OperatorInferenceModel:
    """Reduced operators M, K in the coordinates x = [u_b; q] and the transformation T to the full state."""

    def __init__(self, M, K, T, boundary, info=None):
        self.M, self.K, self.T = M, K, T
        self.boundary = np.asarray(boundary, int)
        self.n_boundary = len(self.boundary)
        self.n_reduced = M.shape[0]
        self.info = {} if info is None else info

    def expand(self, x):
        return self.T @ np.asarray(x, float)

    def reduced_state(self, U):
        """Project full states (n_full, nt) onto the reduced coordinates by least squares."""
        return np.linalg.lstsq(self.T, np.asarray(U, float), rcond=None)[0]

    def simulate_contact(self, force, B, c, h, n_steps, u0=None, v0=None, method="active_set"):
        """Contact dynamics in reduced coordinates. force(t) -> full load vector; B (m, n_full) gap map (or
        (m, n_boundary) acting on the boundary coordinates when its width equals n_boundary); c offsets
        (B u >= c). Returns (X (n_steps+1, n_reduced), U (n_steps+1, n_full), Lam (n_steps+1, m))."""
        B = np.asarray(B, float)
        Br = B @ self.T if B.shape[1] == self.T.shape[0] else np.hstack([B, np.zeros((B.shape[0], self.n_reduced - B.shape[1]))])
        c = np.asarray(c, float)
        n = self.n_reduced
        x0 = np.zeros(n) if u0 is None else self.reduced_state(np.asarray(u0, float)[:, None])[:, 0]
        v0r = np.zeros(n) if v0 is None else self.reduced_state(np.asarray(v0, float)[:, None])[:, 0]
        S = self.M + h * h * self.K
        Sinv = np.linalg.inv(S)
        SBt = Sinv @ Br.T
        Ahat = h * h * (Br @ SBt)
        Ahat = 0.5 * (Ahat + Ahat.T)
        X = np.zeros((n_steps + 1, n))
        Lam = np.zeros((n_steps + 1, Br.shape[0]))
        X[0], X[1] = x0, x0 + h * v0r
        for k in range(1, n_steps):
            fr = self.T.T @ np.asarray(force((k + 1) * h), float)
            rhs = h * h * fr + self.M @ (2 * X[k] - X[k - 1])
            base = Sinv @ rhs
            q = Br @ base - c
            lam, _ = solve_lcp(Ahat, q, method=method)
            Lam[k + 1] = lam
            X[k + 1] = base + h * h * (SBt @ lam)
        return X, X @ self.T.T, Lam


def infer_contact_operators(U, A, F, boundary, Phi, Psi, **kw):
    """Fit the Craig-Bampton-structured model from full-order data.

    U, A, F : (n_full, nt) states, accelerations and applied forces. boundary : boundary DOF indices.
    Phi : (n_interior, r) fixed-boundary interior modes (orthonormal columns), Psi : (n_interior, n_b) coupling.
    The data are projected onto x = [u_b; q] with q = Phi^+ (u_I - Psi u_b); forces are mapped by T^T."""
    U, A, F = (np.asarray(v, float) for v in (U, A, F))
    n_full = U.shape[0]
    boundary = np.asarray(boundary, int)
    interior = np.setdiff1d(np.arange(n_full), boundary)
    T = craig_bampton_transformation(boundary, n_full, Phi, Psi)

    def to_x(Y):
        yb = Y[boundary]
        q = np.linalg.pinv(Phi) @ (Y[interior] - Psi @ yb)
        return np.vstack([yb, q])
    X, Ax, Fr = to_x(U), to_x(A), T.T @ F
    M, K, info = fit_spd_operators(X, Ax, Fr, **kw)
    info["projection_residual"] = float(np.linalg.norm(U - T @ X) / max(np.linalg.norm(U), 1e-300))
    return OperatorInferenceModel(M, K, T, boundary, info)

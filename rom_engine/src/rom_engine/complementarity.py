# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
complementarity.py -- small dense linear complementarity (LCP) solvers and the reduced unilateral-contact solve.

rom_engine never imports fea_engine, so the LCP routines of `fea_engine.contact_solvers` are carried here in a
dense, dependency-free form (same algorithms, same tests). The problem is

    w = A z + q >= 0,   z >= 0,   z . w = 0.

Reduced unilateral contact (Balajewicz et al. 2015, Eq. 7) is  A_r u = b_r + B_r^T lam,  B_r u >= c_r,
lam >= 0, lam . (B_r u - c_r) = 0, with A_r = U^T K U SPD, and is reduced to an LCP in the multipliers by
`solve_reduced_contact`. Functions: lcp_lemke, lcp_active_set, solve_lcp, solve_reduced_contact.

References (methods in this module are adopted from):
  - M. Balajewicz, D. Amsallem, C. Farhat (2015), "Projection-based model reduction for contact problems", arXiv:1503.01000 (Int. J. Numer. Meth. Engng, 2016)
  - R. W. Cottle, J.-S. Pang, R. E. Stone (1992), "The Linear Complementarity Problem", Academic Press
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["lcp_lemke", "lcp_active_set", "solve_lcp", "solve_reduced_contact", "complementarity_violation"]


def complementarity_violation(lam, gap):
    """Max violation of gap >= 0, lam >= 0, lam * gap = 0 (0 means all three hold)."""
    lam, gap = np.asarray(lam, float), np.asarray(gap, float)
    if lam.size == 0:
        return 0.0
    return float(max(np.max(np.maximum(-gap, 0.0)), np.max(np.maximum(-lam, 0.0)), np.max(np.abs(lam * gap))))


def lcp_lemke(A, q, max_iter=None, tol=1e-12):
    """Lemke's algorithm for w = A z + q, z >= 0, w >= 0, z.w = 0. Returns (z, ok).
    Terminates with a solution for positive semidefinite A; ok is False on ray termination."""
    A = np.asarray(A, float)
    q = np.asarray(q, float)
    n = len(q)
    if n == 0 or np.all(q >= -tol):
        return np.zeros(n), True
    max_iter = max_iter or 50 * n + 100
    T = np.hstack([np.eye(n), -A, -np.ones((n, 1)), q.reshape(-1, 1)])   # w - A z - z0 e = q
    z0 = 2 * n
    basis = list(range(n))
    r = int(np.argmin(q))
    T[r] /= T[r, z0]
    for i in range(n):
        if i != r:
            T[i] -= T[i, z0] * T[r]
    leaving = basis[r]
    basis[r] = z0
    for _ in range(max_iter):
        entering = leaving + n if leaving < n else leaving - n
        col = T[:, entering]
        pos = np.flatnonzero(col > 1e-14)
        if pos.size == 0:
            return np.zeros(n), False
        ratios = T[pos, -1] / col[pos]
        rmin = ratios.min()
        cand = pos[ratios <= rmin + 1e-12 * max(1.0, abs(rmin))]
        # prefer to drive z0 out if tied, else smallest basis index (anti-cycling)
        r = next((i for i in cand if basis[i] == z0), cand[np.argmin([basis[i] for i in cand])])
        T[r] /= T[r, entering]
        for i in range(n):
            if i != r:
                T[i] -= T[i, entering] * T[r]
        leaving, basis[r] = basis[r], entering
        if leaving == z0:
            z = np.zeros(n)
            for i, b in enumerate(basis):
                if n <= b < 2 * n:
                    z[b - n] = T[i, -1]
            return np.maximum(z, 0.0), True
    return np.zeros(n), False


def lcp_active_set(A, q, max_iter=100, tol=1e-12, active0=None):
    """Primal-dual active-set method for the LCP of a symmetric positive definite A (Hintermueller-Ito-Kunisch).
    The set of positive multipliers is updated by the sign of (z - w); if the set revisits a previous one
    (cycling) the call falls back to `lcp_lemke`. `active0` (indices of multipliers assumed positive) warm-starts the
    iteration, e.g. with the previous time step's set. Returns (z, ok, iterations)."""
    A = np.asarray(A, float)
    q = np.asarray(q, float)
    n = len(q)
    if n == 0 or np.all(q >= 0):
        return np.zeros(n), True, 0
    active = set() if active0 is None else set(int(i) for i in np.asarray(active0, dtype=int))
    seen = []
    z = np.zeros(n)
    for it in range(1, max_iter + 1):
        idx = np.array(sorted(active), dtype=int)
        z = np.zeros(n)
        if idx.size:
            try:
                z[idx] = np.linalg.solve(A[np.ix_(idx, idx)], -q[idx])
            except np.linalg.LinAlgError:
                break
        w = A @ z + q
        new = set(np.flatnonzero(z - w > tol).tolist())
        # keep strictly positive multipliers only on the proposed set
        if new == active:
            if np.all(z >= -1e-10) and np.all(w >= -1e-10):
                return np.maximum(z, 0.0), True, it
        if frozenset(new) in seen:
            break
        seen.append(frozenset(active))
        active = new
    z, ok = lcp_lemke(A, q)
    return z, ok, max_iter


def solve_lcp(A, q, method="active_set", active0=None):
    """Dispatch: 'active_set' (default, optional warm start `active0`) or 'lemke'. Returns (z, ok)."""
    if method == "active_set":
        z, ok, _ = lcp_active_set(A, q, active0=active0)
        return z, ok
    if method == "lemke":
        return lcp_lemke(A, q)
    raise ValueError("method must be 'active_set' or 'lemke'")




def solve_reduced_contact(A_r, b_r, B_r, c_r, method="active_set"):
    """Solve the reduced unilateral contact problem  A_r u = b_r + B_r^T lam,  B_r u >= c_r,  lam >= 0,
    lam . (B_r u - c_r) = 0  with A_r symmetric positive definite (p x p), B_r (m x p), c_r (m,).
    Returns (u, lam, gap, ok). B_r may have more rows than columns (a rank-deficient LCP matrix); Lemke's
    algorithm then handles the semidefinite LCP."""
    from scipy.linalg import cho_factor, cho_solve
    A_r = np.asarray(A_r, float)
    B_r = np.asarray(B_r, float)
    cf = cho_factor(0.5 * (A_r + A_r.T))
    Ainv_b = cho_solve(cf, np.asarray(b_r, float))
    Ainv_Bt = cho_solve(cf, B_r.T)
    Alcp = B_r @ Ainv_Bt
    Alcp = 0.5 * (Alcp + Alcp.T)
    q = B_r @ Ainv_b - np.asarray(c_r, float)
    lam, ok = solve_lcp(Alcp, q, method=method)
    u = Ainv_b + Ainv_Bt @ lam
    return u, lam, B_r @ u - c_r, ok

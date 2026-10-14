# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_solvers.py -- multi-node frictionless (Signorini) contact on plain arrays.

The existing contact elements enforce contact by a penalty, and
`nonlinear_solver.solve_contact_lagrange_static` enforces ONE node exactly. This module adds exact
enforcement of many simultaneous unilateral constraints, for static problems, implicit dynamics and
nonlinear residuals, without any element class: everything takes matrices and callables.

Problem and sign convention (the one of Balajewicz et al. 2015 and Manvelyan-Stroot et al. 2025):

    K u = f + B^T lam,     g = B u - c >= 0,     lam >= 0,     lam . g = 0.

`B` (m x n) extracts the m potential contact gaps from the displacements, `c` is the offset
(negative initial clearance), and the multipliers `lam` are the contact forces.

Eliminating `u` gives a linear complementarity problem (LCP)  w = A lam + q >= 0, lam >= 0, lam.w = 0
with A = B K^-1 B^T (symmetric positive semidefinite) and q = B K^-1 f - c. Three LCP solvers are provided:

- `lcp_active_set`       primal-dual active set (a semismooth Newton method), fast, with a block-pivoting
                         fallback so it cannot cycle silently;
- `lcp_lemke`            Lemke's complementary pivoting, which always terminates for positive (semi)definite A;
- `lcp_projected_gauss_seidel`  slow but trivially robust, useful as an independent check.

Scope, stated honestly: frictionless, adhesive-free, small-displacement kinematics for the linear
solvers (the gap map B is constant). `solve_contact_newton` handles a nonlinear internal force by
re-solving the linearised contact problem at every Newton iterate (a sequential-linearisation method);
B stays constant there too, so large sliding is out of scope. Friction is not handled here.

References (methods in this module are adopted from):
  - M. Balajewicz, D. Amsallem, C. Farhat (2015), "Projection-based model reduction for contact problems", arXiv:1503.01000 (Int. J. Numer. Meth. Engng, 2016)
  - D. Manvelyan-Stroot, Y. Filanova, I. Pontes Duff, P. Benner, U. Wever (2025), "Inference of Substructured Reduced-Order Models for Dynamic Contact from Contact-free Simulations", arXiv:2505.18050
  - R. W. Cottle, J.-S. Pang, R. E. Stone (1992), "The Linear Complementarity Problem", Academic Press
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["lcp_lemke", "lcp_active_set", "lcp_projected_gauss_seidel", "solve_lcp",
           "contact_complementarity_residual", "solve_contact_static", "solve_contact_dynamics",
           "solve_contact_newton", "ContactResult"]


class ContactResult:
    """u: displacements, lam: multipliers (contact forces), gap: B u - c, active: indices with lam > 0,
    iterations: solver iterations, converged: bool."""

    def __init__(self, u, lam, gap, iterations, converged):
        self.u, self.lam, self.gap = u, lam, gap
        self.active = np.flatnonzero(lam > 0)
        self.iterations, self.converged = iterations, converged

    def __repr__(self):
        return (f"ContactResult(n_active={len(self.active)}, iterations={self.iterations}, "
                f"converged={self.converged})")


def contact_complementarity_residual(lam, gap):
    """Max violation of the three conditions: gap >= 0, lam >= 0, lam*gap = 0 (a scalar, 0 = satisfied)."""
    lam, gap = np.asarray(lam, float), np.asarray(gap, float)
    if lam.size == 0:
        return 0.0
    return float(max(np.max(np.maximum(-gap, 0.0)), np.max(np.maximum(-lam, 0.0)), np.max(np.abs(lam * gap))))


# ----------------------------------------------------------------------------------------------- LCP
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


def lcp_projected_gauss_seidel(A, q, max_iter=20000, tol=1e-10):
    """Projected Gauss-Seidel for symmetric positive (semi)definite A with positive diagonal. Returns (z, ok)."""
    A = np.asarray(A, float)
    q = np.asarray(q, float)
    n = len(q)
    z = np.zeros(n)
    d = np.diag(A)
    if np.any(d <= 0):
        return z, False
    for _ in range(max_iter):
        z_old = z.copy()
        for i in range(n):
            z[i] = max(0.0, z[i] - (A[i] @ z + q[i]) / d[i])
        if np.max(np.abs(z - z_old)) < tol * (1.0 + np.max(np.abs(z))):
            return z, True
    return z, False


def lcp_active_set(A, q, max_iter=100, tol=1e-12):
    """Primal-dual active-set method for the LCP of a symmetric positive definite A (Hintermueller-Ito-Kunisch).
    The set of positive multipliers is updated by the sign of (z - w); if the set revisits a previous one
    (cycling) the call falls back to `lcp_lemke`. Returns (z, ok, iterations)."""
    A = np.asarray(A, float)
    q = np.asarray(q, float)
    n = len(q)
    if n == 0 or np.all(q >= 0):
        return np.zeros(n), True, 0
    active = set()
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


def solve_lcp(A, q, method="active_set"):
    """Dispatch: 'active_set' (default), 'lemke' or 'pgs'. Returns (z, ok)."""
    if method == "active_set":
        z, ok, _ = lcp_active_set(A, q)
        return z, ok
    if method == "lemke":
        return lcp_lemke(A, q)
    if method == "pgs":
        return lcp_projected_gauss_seidel(A, q)
    raise ValueError("method must be 'active_set', 'lemke' or 'pgs'")


# ------------------------------------------------------------------------------------- contact solves
def _factor(K):
    if sp.issparse(K):
        lu = spla.splu(sp.csc_matrix(K))
        return lambda X: lu.solve(np.asarray(X))
    K = np.asarray(K, float)
    from scipy.linalg import cho_factor, cho_solve
    try:
        cf = cho_factor(K)
        return lambda X: cho_solve(cf, np.asarray(X))
    except np.linalg.LinAlgError:
        return lambda X: np.linalg.solve(K, np.asarray(X))


def solve_contact_static(K, f, B, c, method="active_set"):
    """Static unilateral contact K u = f + B^T lam, B u >= c, lam >= 0, complementarity.
    K must be symmetric positive definite (constrain rigid-body motion first). B may be sparse or dense."""
    B = B.toarray() if sp.issparse(B) else np.asarray(B, float)
    c = np.asarray(c, float)
    f = np.asarray(f, float)
    solve = _factor(K)
    Kinv_f = solve(f)
    Kinv_Bt = solve(B.T)
    A = B @ Kinv_Bt
    A = 0.5 * (A + A.T)
    q = B @ Kinv_f - c
    lam, ok = solve_lcp(A, q, method=method)
    u = Kinv_f + Kinv_Bt @ lam
    return ContactResult(u, lam, B @ u - c, 0, ok)


def solve_contact_dynamics(M, K, force, B, c, h, n_steps, u0=None, v0=None, method="active_set"):
    """Implicit unilateral-contact dynamics with the two-step scheme of Manvelyan-Stroot et al. 2025:
    M (u_{n+1} - 2 u_n + u_{n-1}) + h^2 K u_{n+1} = h^2 (f_{n+1} + B^T lam_{n+1}),  B u_{n+1} >= c.
    The LCP matrix A = h^2 B (M + h^2 K)^-1 B^T is constant, so it is formed once.
    `force(t)` returns the load vector. Returns (U, Lam) with shapes (n_steps + 1, n) and (n_steps + 1, m)."""
    B = B.toarray() if sp.issparse(B) else np.asarray(B, float)
    c = np.asarray(c, float)
    n = M.shape[0]
    u0 = np.zeros(n) if u0 is None else np.asarray(u0, float)
    v0 = np.zeros(n) if v0 is None else np.asarray(v0, float)
    S = (M + h * h * K)
    solve = _factor(S)
    SBt = solve(B.T)
    A = h * h * (B @ SBt)
    A = 0.5 * (A + A.T)
    U = np.zeros((n_steps + 1, n))
    Lam = np.zeros((n_steps + 1, B.shape[0]))
    U[0] = u0
    U[1] = u0 + h * v0
    for k in range(1, n_steps):
        rhs = h * h * force((k + 1) * h) + M @ (2.0 * U[k] - U[k - 1])
        base = solve(rhs)
        q = B @ base - c
        lam, _ = solve_lcp(A, q, method=method)
        Lam[k + 1] = lam
        U[k + 1] = base + h * h * (SBt @ lam)
    return U, Lam


def solve_contact_newton(residual, tangent, B, c, u0, tol=1e-10, max_iter=50, method="active_set"):
    """Nonlinear internal force with unilateral constraints, by sequential linearisation: at iterate u_k
    solve  K_t du = -R(u_k) + B^T lam,  B (u_k + du) >= c,  lam >= 0  (a linear contact problem, via the
    LCP route), then u_{k+1} = u_k + du. `residual(u)` returns R(u) = f_int(u) - f_ext, `tangent(u)` its
    Jacobian. Converges when ||R - B^T lam|| is below tol relative to the load scale."""
    B = B.toarray() if sp.issparse(B) else np.asarray(B, float)
    c = np.asarray(c, float)
    u = np.asarray(u0, float).copy()
    lam = np.zeros(B.shape[0])
    scale = max(1.0, np.linalg.norm(residual(u)))
    for it in range(1, max_iter + 1):
        R = residual(u)
        Kt = tangent(u)
        solve = _factor(Kt)
        Kinv_R = solve(-R)
        Kinv_Bt = solve(B.T)
        A = B @ Kinv_Bt
        A = 0.5 * (A + A.T)
        q = B @ (u + Kinv_R) - c
        lam, ok = solve_lcp(A, q, method=method)
        du = Kinv_R + Kinv_Bt @ lam
        u = u + du
        r = np.linalg.norm(residual(u) - B.T @ lam)
        if r < tol * scale and np.linalg.norm(du) < 1e-8 * (1.0 + np.linalg.norm(u)):
            return ContactResult(u, lam, B @ u - c, it, True)
    return ContactResult(u, lam, B @ u - c, max_iter, False)

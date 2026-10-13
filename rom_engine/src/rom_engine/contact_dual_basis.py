# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_dual_basis.py -- reduced unilateral contact with a POSITIVE basis for the contact forces.

With the contact conditions enforced by Lagrange multipliers (the contact forces), the displacements can be
reduced by POD as usual but the multipliers cannot: an SVD basis of multiplier snapshots does not keep the
reduced multipliers non-negative, so the reduced solution can pull on a surface (Balajewicz, Amsallem and Farhat
2015). Their remedy, implemented here, is a non-negative matrix factorization (NMF) of the multiplier snapshots,
X_lam ~ U_lam V with U_lam >= 0 and V >= 0, together with reduced multipliers lam_r >= 0, so that
lam = U_lam lam_r >= 0 holds by construction. The reduced problem (their Eq. 7) is

    A_r u_r = b_r + B_r^T lam_r,   B_r u_r >= c_r,   lam_r >= 0,   complementarity,
    A_r = U^T K U,  B_r = U_lam^T B U,  c_r = U_lam^T c,

solved as a small LCP (`complementarity.solve_reduced_contact`).

Contents
--------
nonneg_dual_basis       NMF of multiplier snapshots (hierarchical ALS with an SVD-based NNDSVD initialisation)
svd_dual_basis          the SVD alternative, kept for comparison (it violates positivity)
active_snapshot_columns drop contact-free snapshots, which are exactly zero and carry no information
ContactROM              the reduced contact model: assemble A_r, B_r once, solve for any load and offset
sparse_dictionary_fit   non-negative orthogonal matching pursuit over a dictionary of multiplier snapshots (the
                        sparse-dictionary alternative of Kollepara et al. 2024, for comparison only)

Honest scope. NMF is non-convex and initialisation matters; the dual basis is far less accurate than the primal
one at equal size (Fauque et al. 2018 report the same). The dual basis size should not exceed the number of
independent reduced constraints the primal basis can carry (the rank condition `ContactROM.inf_sup_report`
measures). Frictionless, small-displacement contact only.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.optimize import nnls

from .complementarity import solve_reduced_contact

__all__ = ["NMFBasis", "nonneg_dual_basis", "svd_dual_basis", "active_snapshot_columns", "ContactROM",
           "ContactROMResult", "sparse_dictionary_fit"]


class NMFBasis:
    """U (n x r, columns of unit 2-norm, >= 0), V (r x m, >= 0), rel_error (Frobenius, relative), n_iter."""

    def __init__(self, U, V, rel_error, n_iter):
        self.U, self.V, self.rel_error, self.n_iter = U, V, rel_error, n_iter

    @property
    def rank(self):
        return self.U.shape[1]


def _nndsvd(X, r):
    """NNDSVDa initialisation (Boutsidis and Gaussier 2008): non-negative parts of the leading singular vectors,
    zeros filled with a small mean so multiplicative or HALS updates can leave them."""
    Uu, S, Vt = np.linalg.svd(X, full_matrices=False)
    n, m = X.shape
    W = np.zeros((n, r))
    H = np.zeros((r, m))
    W[:, 0] = np.sqrt(S[0]) * np.abs(Uu[:, 0])
    H[0] = np.sqrt(S[0]) * np.abs(Vt[0])
    for j in range(1, min(r, len(S))):
        x, y = Uu[:, j], Vt[j]
        xp, xn = np.maximum(x, 0), np.maximum(-x, 0)
        yp, yn = np.maximum(y, 0), np.maximum(-y, 0)
        mp = np.linalg.norm(xp) * np.linalg.norm(yp)
        mn = np.linalg.norm(xn) * np.linalg.norm(yn)
        if mp >= mn:
            u, v, sig = xp / (np.linalg.norm(xp) + 1e-300), yp / (np.linalg.norm(yp) + 1e-300), mp
        else:
            u, v, sig = xn / (np.linalg.norm(xn) + 1e-300), yn / (np.linalg.norm(yn) + 1e-300), mn
        W[:, j] = np.sqrt(S[j] * sig) * u
        H[j] = np.sqrt(S[j] * sig) * v
    fill = X.mean() / 100.0
    W[W < 1e-12 * max(1.0, W.max())] = fill
    H[H < 1e-12 * max(1.0, H.max())] = fill
    return W, H


def nonneg_dual_basis(snapshots, rank, n_iter=500, tol=1e-9, seed=None):
    """NMF X ~ U V of non-negative multiplier snapshots (columns), by hierarchical alternating least squares.
    `seed=None` uses the deterministic NNDSVDa start (recommended); an integer uses a random start instead.
    Returns an NMFBasis whose U has unit-norm columns."""
    X = np.asarray(snapshots, float)
    if np.any(X < -1e-12):
        raise ValueError("multiplier snapshots must be non-negative")
    X = np.maximum(X, 0.0)
    n, m = X.shape
    r = int(min(rank, n, m))
    if seed is None:
        W, H = _nndsvd(X, r)
    else:
        rng = np.random.default_rng(seed)
        W, H = rng.random((n, r)) * X.mean() ** 0.5, rng.random((r, m)) * X.mean() ** 0.5
    nx = np.linalg.norm(X)
    prev = np.inf
    eps = 1e-14
    for it in range(1, n_iter + 1):
        HHt, XHt = H @ H.T, X @ H.T
        for k in range(r):
            W[:, k] = np.maximum(eps, (XHt[:, k] - W @ HHt[:, k] + W[:, k] * HHt[k, k]) / max(HHt[k, k], 1e-300))
        WtW, WtX = W.T @ W, W.T @ X
        for k in range(r):
            H[k] = np.maximum(eps, (WtX[k] - WtW[k] @ H + WtW[k, k] * H[k]) / max(WtW[k, k], 1e-300))
        err = np.linalg.norm(X - W @ H) / (nx + 1e-300)
        if abs(prev - err) < tol * max(err, 1e-300) or err < 1e-13:
            break
        prev = err
    norms = np.linalg.norm(W, axis=0)
    norms[norms == 0] = 1.0
    return NMFBasis(W / norms, H * norms[:, None], float(err), it)


def svd_dual_basis(snapshots, rank):
    """Leading left singular vectors of the multiplier snapshots (no positivity). For comparison."""
    U, _, _ = np.linalg.svd(np.asarray(snapshots, float), full_matrices=False)
    return U[:, :rank]


def active_snapshot_columns(multiplier_snapshots, tol=0.0):
    """Indices of snapshot columns with any multiplier above `tol`: contact-free steps give an all-zero column that
    adds nothing to the dual basis (Balajewicz et al. 2015, Section 4.4)."""
    X = np.asarray(multiplier_snapshots, float)
    return np.flatnonzero(np.max(X, axis=0) > tol)


class ContactROMResult:
    """u (full displacements), lam (full multipliers, >= 0 when the dual basis is non-negative), lam_r, u_r, gap
    (B u - c on the full constraints), ok (LCP solved)."""

    def __init__(self, u, lam, u_r, lam_r, gap, ok):
        self.u, self.lam, self.u_r, self.lam_r, self.gap, self.ok = u, lam, u_r, lam_r, gap, ok


class ContactROM:
    """Reduced unilateral contact model with a primal basis U (n x p) and a dual basis U_lam (m x p_lam).

    K (n x n, SPD, dense or sparse) and B (m x n) are the full-order stiffness and gap map. A_r and B_r are built
    once; `solve(f, c)` projects the load and offset and solves the small LCP."""

    def __init__(self, K, B, U, U_lam, method="active_set"):
        self.U = np.asarray(U, float)
        self.U_lam = np.asarray(U_lam, float)
        self.B = B
        self.method = method
        KU = K @ self.U
        self.A_r = self.U.T @ KU
        BU = B @ self.U
        self.B_r = self.U_lam.T @ BU

    @property
    def sizes(self):
        return self.U.shape[1], self.U_lam.shape[1]

    def solve(self, f, c):
        f = np.asarray(f, float)
        c = np.asarray(c, float)
        u_r, lam_r, _, ok = solve_reduced_contact(self.A_r, self.U.T @ f, self.B_r, self.U_lam.T @ c, self.method)
        u = self.U @ u_r
        return ContactROMResult(u, self.U_lam @ lam_r, u_r, lam_r, self.B @ u - c, ok)

    def inf_sup_report(self):
        """Rank diagnostic of the reduced constraint matrix B_r = U_lam^T B U (Fauque et al. 2018): the saddle-point
        problem is well posed only if B_r has full row rank, which needs p >= p_lam. Returns a dict with the numerical
        rank, the number of dual vectors, the smallest nonzero singular value and the condition number."""
        s = np.linalg.svd(self.B_r, compute_uv=False)
        rank = int(np.sum(s > 1e-10 * s[0])) if s.size else 0
        p_lam = self.B_r.shape[0]
        return {"rank": rank, "n_dual": p_lam, "n_primal": self.B_r.shape[1], "full_row_rank": rank == p_lam,
                "sigma_min": float(s[rank - 1]) if rank else 0.0,
                "condition": float(s[0] / s[rank - 1]) if rank else np.inf}


def sparse_dictionary_fit(dictionary, target, n_atoms, tol=1e-12):
    """Approximate `target` (m,) by a NON-NEGATIVE combination of at most `n_atoms` columns of `dictionary`
    (m x N): greedy atom selection by correlation with the residual, then non-negative least squares on the
    selected set. Returns (coefficients (N,), selected indices, relative residual)."""
    D = np.asarray(dictionary, float)
    y = np.asarray(target, float)
    sel = []
    coef = np.zeros(D.shape[1])
    resid = y.copy()
    ny = np.linalg.norm(y) + 1e-300
    for _ in range(int(n_atoms)):
        corr = D.T @ resid
        corr[sel] = -np.inf
        j = int(np.argmax(corr))
        if corr[j] <= tol:
            break
        sel.append(j)
        x, _ = nnls(D[:, sel], y)
        coef[:] = 0.0
        coef[sel] = x
        resid = y - D @ coef
        if np.linalg.norm(resid) / ny < tol:
            break
    return coef, sel, float(np.linalg.norm(resid) / ny)

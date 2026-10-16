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
ContactROM              the reduced contact model: assemble A_r, B_r once, solve for any load and offset;
                        `solve_dynamic` runs a reduced BDF2 time loop (needs the mass matrix `M`)
sparse_dictionary_fit   non-negative orthogonal matching pursuit over a dictionary of multiplier snapshots (the
                        sparse-dictionary alternative of Kollepara et al. 2024, for comparison only)

Honest scope. NMF is non-convex and initialisation matters; the dual basis is far less accurate than the primal
one at equal size (Fauque et al. 2018 report the same). The dual basis size should not exceed the number of
independent reduced constraints the primal basis can carry (the rank condition `ContactROM.inf_sup_report`
measures). Frictionless, small-displacement contact only.

References (methods in this module are adopted from):
  - M. Balajewicz, D. Amsallem, C. Farhat (2015), "Projection-based model reduction for contact problems", arXiv:1503.01000 (Int. J. Numer. Meth. Engng, 2016)
  - A. Fauque, I. Ramiere, D. Ryckelynck (2018), "Hybrid hyper-reduced modeling for contact mechanics problems", Int. J. Numer. Meth. Engng 115(1):117-139
  - C. Boutsidis, E. Gaussier (2008), "SVD based initialization: a head start for nonnegative matrix factorization", Pattern Recognition 41(4)
  - C. L. Lawson, R. J. Hanson (1974), "Solving Least Squares Problems", Prentice-Hall (non-negative least squares)
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.optimize import nnls

from .complementarity import solve_reduced_contact, lcp_active_set, lcp_lemke

__all__ = ["NMFBasis", "nonneg_dual_basis", "svd_dual_basis", "active_snapshot_columns", "ContactROM",
           "ContactROMResult", "ContactROMDynamicResult", "sparse_dictionary_fit"]


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


class ContactROMDynamicResult:
    """Output of `ContactROM.solve_dynamic`. u_r, lam_r, v_r: reduced displacements, multipliers and velocities,
    (n_steps + 1, p), (n_steps + 1, p_lam), (n_steps + 1, p). u, lam, gap: the lifted full-order displacements (n_steps + 1, n),
    multipliers U_lam lam_r (n_steps + 1, m) and gaps B u - c; None when called with lift=False. ok: every reduced contact
    solve succeeded; failed_step (1-based) and message otherwise, and the rows from that step on are NaN.
    iterations: the active-set iterations per step (empty for method="lemke"). kkt_residual: per solved step, the largest relative
    violation of the reduced contact conditions (gap, sign, complementarity); a step above 1e-4 is reported as a failure."""

    def __init__(self, u_r, lam_r, v_r, u, lam, gap, ok, failed_step, message, iterations):
        self.u_r, self.lam_r, self.v_r, self.u, self.lam, self.gap = u_r, lam_r, v_r, u, lam, gap
        self.ok, self.failed_step, self.message, self.iterations = ok, failed_step, message, iterations


class ContactROM:
    """Reduced unilateral contact model with a primal basis U (n x p) and a dual basis U_lam (m x p_lam).

    K (n x n, SPD, dense or sparse) and B (m x n) are the full-order stiffness and gap map. A_r and B_r are built
    once; `solve(f, c)` projects the load and offset and solves the small LCP."""

    def __init__(self, K, B, U, U_lam, method="active_set", M=None):
        self.U = np.asarray(U, float)
        self.U_lam = np.asarray(U_lam, float)
        self.B = B
        self.method = method
        KU = K @ self.U
        self.A_r = self.U.T @ KU
        BU = B @ self.U
        self.B_r = self.U_lam.T @ BU
        # reduced mass U^T M U, only needed by solve_dynamic (M: dense or sparse, n x n, or a 1-D diagonal)
        if M is None:
            self.M_r = None
        else:
            MU = (np.asarray(M, float)[:, None] * self.U) if np.ndim(M) == 1 else M @ self.U
            self.M_r = self.U.T @ MU

    @property
    def sizes(self):
        return self.U.shape[1], self.U_lam.shape[1]

    def solve(self, f, c):
        f = np.asarray(f, float)
        c = np.asarray(c, float)
        u_r, lam_r, _, ok = solve_reduced_contact(self.A_r, self.U.T @ f, self.B_r, self.U_lam.T @ c, self.method)
        u = self.U @ u_r
        return ContactROMResult(u, self.U_lam @ lam_r, u_r, lam_r, self.B @ u - c, ok)

    def solve_dynamic(self, force, c, dt, n_steps, u0=None, v0=None, scheme="bdf2", warm_start=True, lift=True):
        """Reduced contact dynamics  M_r u_r'' + A_r u_r = U^T f(t) + B_r^T lam_r,  B_r u_r >= c_r,  lam_r >= 0, complementarity,
        stepped with BDF2 (a BDF1 first step; the same scheme as `fea_engine.solve_contact_dynamics(scheme="bdf2")`, written out
        on the reduced matrices): (9 M_r / (4 dt^2) + A_r) u_{n+1} = f_r + B_r^T lam_r + M_r [3 (4 u_n - u_{n-1}) / (4 dt^2)
        + (4 v_n - v_{n-1}) / (2 dt)], v_{n+1} = (3 u_{n+1} - 4 u_n + u_{n-1}) / (2 dt). Needs the mass matrix (`M=` of the
        constructor). The two effective matrices are factored once, together with the small LCP matrices they induce, and the
        active set of the previous step warm-starts the next LCP (`warm_start=False` switches it off).

        force(t): full-order load vector; c: full-order offset (constant in time); u0, v0: full-order initial state, projected
        on the basis by least squares (zero by default). lift=False skips the (n_steps + 1, n) and (n_steps + 1, m) arrays.
        Returns `ContactROMDynamicResult`; a failing LCP stops the loop and is reported (the rows from the failed step are NaN).
        Only scheme="bdf2" exists; frictionless contact, constant gap map, reduced offset c_r = U_lam^T c."""
        if scheme != "bdf2":
            raise ValueError("scheme must be 'bdf2'")
        if self.M_r is None:
            raise ValueError("solve_dynamic needs the mass matrix: construct ContactROM(K, B, U, U_lam, M=M)")
        from scipy.linalg import cho_factor, cho_solve
        h = float(dt)
        p, pl = self.U.shape[1], self.U_lam.shape[1]
        c = np.asarray(c, float)
        c_r = self.U_lam.T @ c
        Mr, Ar, Br = self.M_r, self.A_r, self.B_r
        prep = {}
        for key, a in (("first", 1.0 / (h * h)), ("later", 9.0 / (4.0 * h * h))):
            Aeff = a * Mr + Ar
            cf = cho_factor(0.5 * (Aeff + Aeff.T))
            AinvBt = cho_solve(cf, Br.T)
            Alcp = Br @ AinvBt
            prep[key] = (cf, AinvBt, 0.5 * (Alcp + Alcp.T))
        proj = lambda x: np.linalg.lstsq(self.U, np.asarray(x, float), rcond=None)[0]
        U_r = np.zeros((n_steps + 1, p))
        V_r = np.zeros((n_steps + 1, p))
        L_r = np.zeros((n_steps + 1, pl))
        if u0 is not None:
            U_r[0] = proj(u0)
        if v0 is not None:
            V_r[0] = proj(v0)
        iterations, ok, failed, message, prev, residuals = [], True, None, None, None, []
        for k in range(1, n_steps + 1):
            f_r = self.U.T @ np.asarray(force(k * h), float)
            if k == 1:
                key, rhs = "first", f_r + Mr @ (U_r[0] / (h * h) + V_r[0] / h)
            else:
                key = "later"
                rhs = f_r + Mr @ (3.0 * (4.0 * U_r[k - 1] - U_r[k - 2]) / (4.0 * h * h) + (4.0 * V_r[k - 1] - V_r[k - 2]) / (2.0 * h))
            cf, AinvBt, Alcp = prep[key]
            base = cho_solve(cf, rhs)
            q = Br @ base - c_r
            if self.method == "active_set":
                z, good, it = lcp_active_set(Alcp, q, active0=prev if warm_start else None)
                iterations.append(it)
            else:
                z, good = lcp_lemke(Alcp, q)
            kkt = False
            if good:                                 # a solver flag is not enough: verify the reduced contact conditions
                w = Alcp @ z + q
                qs = 1.0 + np.max(np.abs(q))
                zs = 1.0 + np.max(np.abs(z))
                rel = max(max(-float(w.min()), 0.0) / qs, float(np.max(np.abs(z * w))) / (qs * zs), max(-float(z.min()), 0.0) / zs)
                residuals.append(rel)
                good = rel <= 1e-4                   # a sanity guard, not an accuracy claim: a singular LCP matrix limits Lemke to about 1e-5
                kkt = not good
            if not good:
                ok, failed = False, k
                if kkt:
                    message = (f"the reduced contact conditions are not satisfied in step {k}: B_r = U_lam^T B U is probably rank "
                               f"deficient or ill conditioned (see inf_sup_report()); use a dual basis not larger than its rank")
                else:
                    message = f"the reduced contact LCP failed in step {k}" + ("; try method='lemke' in ContactROM" if self.method != "lemke" else "")
                U_r[k:], V_r[k:], L_r[k:] = np.nan, np.nan, np.nan
                break
            prev = np.flatnonzero(z > 0)
            u = base + AinvBt @ z
            U_r[k], L_r[k] = u, z
            V_r[k] = (u - U_r[k - 1]) / h if k == 1 else (3.0 * u - 4.0 * U_r[k - 1] + U_r[k - 2]) / (2.0 * h)
        if lift:
            U_f, L_f = U_r @ self.U.T, L_r @ self.U_lam.T
            gap = (self.B @ U_f.T).T - c
        else:
            U_f = L_f = gap = None
        out = ContactROMDynamicResult(U_r, L_r, V_r, U_f, L_f, gap, ok, failed, message, iterations)
        out.kkt_residual = np.array(residuals)
        return out


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

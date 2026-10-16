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

`solve_contact_static_sparse` solves the same static problem for a sparse K without forming A: primal-dual active set on
(u, lam), with the optional routes "reduced" (B selects nodes), "saddle" (general sparse B) and "condensed" (the dense LCP routes
above); a failing route returns a flag, a reason and a suggestion of the next route (see its docstring).

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
           "solve_contact_newton", "ContactResult", "solve_contact_static_sparse"]


class ContactResult:
    """u: displacements, lam: multipliers (contact forces), gap: B u - c, active: indices with lam > 0,
    iterations: solver iterations, converged: bool.

    The sparse solver `solve_contact_static_sparse` fills four more fields (None for the other solvers):
    `route` (the route that produced the result), `reason` (a short code when `converged` is False:
    "cycled", "max_iter", "singular", "kkt", "not_applicable", "memory"), `message` (one sentence, including a
    suggestion), `suggestion` (the name of the next route to try, or None), `active_mask` (boolean mask over the
    constraints, usable as `active0` of the next call) and `fallbacks` (routes that failed before this one,
    only with `fallback=True`). `ok` is an alias of `converged`."""

    def __init__(self, u, lam, gap, iterations, converged, route=None, reason=None, message=None, suggestion=None,
                 active_mask=None, fallbacks=None):
        self.u, self.lam, self.gap = u, lam, gap
        self.active = np.flatnonzero(lam > 0)
        self.iterations, self.converged = iterations, converged
        self.route, self.reason, self.message, self.suggestion = route, reason, message, suggestion
        self.active_mask = active_mask
        self.fallbacks = [] if fallbacks is None else list(fallbacks)

    @property
    def ok(self):
        return self.converged

    def __repr__(self):
        extra = "" if self.route is None else f", route={self.route!r}"
        extra += "" if self.reason is None else f", reason={self.reason!r}"
        return (f"ContactResult(n_active={len(self.active)}, iterations={self.iterations}, "
                f"converged={self.converged}{extra})")


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
    K must be symmetric positive definite (constrain rigid-body motion first). B may be sparse or dense.
    method: "active_set" (default), "lemke" or "pgs" (all three form A = B K^-1 B^T densely), or "pdas": the sparse
    route of `solve_contact_static_sparse` with route="auto" and no fallback (see there)."""
    if method == "pdas":
        return solve_contact_static_sparse(K, f, B, c)
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


# ------------------------------------------------------------------ sparse (non-condensed) unilateral solver
_ROUTES = ("reduced", "saddle", "condensed")


def _selection_structure(B):
    """(columns, scales) when every row of B has exactly one nonzero, positive, in a distinct column
    (a node selection: u[col] >= c / scale); otherwise None."""
    B = sp.csr_matrix(B)
    B.eliminate_zeros()
    if B.shape[0] == 0 or np.any(np.diff(B.indptr) != 1):
        return None
    cols, vals = B.indices.copy(), B.data.copy()
    if np.any(vals <= 0) or len(np.unique(cols)) != len(cols):
        return None
    return cols, vals


def _mask(active0, m):
    if active0 is None:
        return np.zeros(m, bool)
    a = np.asarray(active0)
    if a.dtype == bool:
        if a.shape != (m,):
            raise ValueError(f"active0 mask must have shape ({m},)")
        return a.copy()
    out = np.zeros(m, bool)
    out[a.astype(int)] = True
    return out


def _kkt_ok(u, lam, gap, tol):
    su = max(1.0, float(np.max(np.abs(u)))) if u.size else 1.0
    sl = max(1.0, float(np.max(np.abs(lam)))) if lam.size else 1.0
    if lam.size == 0:
        return True
    return bool(gap.min() >= -tol * su and lam.min() >= -tol * sl and np.max(np.abs(lam * gap)) <= tol * su * sl)


def _fail(reason, route, iterations, n, m, B, c, mask, suggestion, msg):
    u = np.zeros(n)
    lam = np.zeros(m)
    return ContactResult(u, lam, np.asarray(B @ u - c), iterations, False, route=route, reason=reason,
                         message=msg, suggestion=suggestion, active_mask=mask)


def _dense_bytes(n, m):
    return 8.0 * (n * m + m * m)


def _next_route(route, n, m):
    nxt = {"reduced": "saddle", "saddle": "condensed", "condensed": None}[route]
    if nxt == "condensed":
        return nxt, (f" The dense route 'condensed' needs about {_dense_bytes(n, m) / 1e9:.2g} GB "
                     f"(n = {n}, m = {m}: K^-1 B^T is n x m and B K^-1 B^T is m x m).")
    return nxt, ""


def _route_reduced(K, f, B, c, mask0, max_iter, tol):
    n, m = K.shape[0], B.shape[0]
    sel = _selection_structure(B)
    if sel is None:
        return _fail("not_applicable", "reduced", 0, n, m, B, c, mask0, "saddle",
                     "route 'reduced' needs every row of B to select one node (one positive nonzero, distinct "
                     "columns); use route='saddle' for a general B.")
    cols, sc = sel
    g = c / sc                                   # bounds on u[cols]
    cp = float(K.diagonal().mean())
    mask = mask0.copy()
    seen = {mask.tobytes()}
    for it in range(1, max_iter + 1):
        fixed = cols[mask]
        u = np.zeros(n)
        u[fixed] = g[mask]
        free = np.ones(n, bool)
        free[fixed] = False
        if free.any():
            Kc = K.tocsr()
            rhs = f[free] - (Kc[free][:, fixed] @ g[mask] if fixed.size else 0.0)
            try:
                u[free] = spla.splu(sp.csc_matrix(Kc[free][:, free])).solve(rhs)
            except RuntimeError:
                return _fail("singular", "reduced", it, n, m, B, c, mask, "saddle",
                             "the stiffness of the free unknowns is singular (K must be positive definite).")
        mu = (K @ u - f)[cols]                    # bound multipliers; lam = mu / scale
        lam = np.where(mask, mu / sc, 0.0)
        gap = sc * u[cols] - c
        new = (np.where(mask, mu, 0.0) + cp * (g - u[cols])) > 0     # the bound form of 'lam - cp * gap > 0'
        if np.array_equal(new, mask) and _kkt_ok(u, lam, gap, tol):
            lam = np.maximum(lam, 0.0)
            return ContactResult(u, lam, gap, it, True, route="reduced", active_mask=mask)
        key = new.tobytes()
        if key in seen:
            return _fail("cycled", "reduced", it, n, m, B, c, new, "saddle",
                         f"the active set repeated after {it} iterations (cycling).")
        seen.add(key)
        mask = new
    return _fail("max_iter", "reduced", max_iter, n, m, B, c, mask, "saddle",
                 f"no convergence in {max_iter} iterations.")


def _route_saddle(K, f, B, c, mask0, max_iter, tol):
    n, m = K.shape[0], B.shape[0]
    B = sp.csr_matrix(B)
    rown = np.asarray(B.multiply(B).sum(axis=1)).ravel()
    cp = float(K.diagonal().mean()) / max(float(rown.mean()), 1e-300)
    mask = mask0.copy()
    seen = {mask.tobytes()}
    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(mask)
        Ba = B[idx]
        if idx.size:
            M = sp.bmat([[K, -Ba.T], [Ba, None]], format="csc")
            rhs = np.concatenate([f, c[idx]])
        else:
            M = sp.csc_matrix(K)
            rhs = f
        try:
            sol = spla.splu(M).solve(rhs)
        except RuntimeError:
            sol = np.full(rhs.shape, np.nan)
        res = np.linalg.norm(M @ sol - rhs) / max(np.linalg.norm(rhs), 1e-300) if np.all(np.isfinite(sol)) else np.inf
        if not np.isfinite(res) or res > 1e-8:
            return _fail("singular", "saddle", it, n, m, B, c, mask, "condensed",
                         "the saddle system is singular (dependent active constraints or a singular K).")
        u = sol[:n]
        lam = np.zeros(m)
        lam[idx] = sol[n:]
        gap = np.asarray(B @ u - c)
        new = (lam - cp * gap) > 0
        if np.array_equal(new, mask) and _kkt_ok(u, lam, gap, tol):
            return ContactResult(u, np.maximum(lam, 0.0), gap, it, True, route="saddle", active_mask=mask)
        key = new.tobytes()
        if key in seen:
            return _fail("cycled", "saddle", it, n, m, B, c, new, "condensed",
                         f"the active set repeated after {it} iterations (cycling).")
        seen.add(key)
        mask = new
    return _fail("max_iter", "saddle", max_iter, n, m, B, c, mask, "condensed", f"no convergence in {max_iter} iterations.")


def _route_condensed(K, f, B, c):
    r = solve_contact_static(K, f, B, c, method="active_set")
    ok = r.converged and _kkt_ok(r.u, r.lam, r.gap, 1e-8)
    return ContactResult(r.u, r.lam, r.gap, r.iterations, bool(ok), route="condensed",
                         reason=None if ok else "kkt",
                         message=None if ok else "the dense active-set route did not satisfy the contact conditions; "
                                                  "try method='lemke' of solve_contact_static.",
                         suggestion=None if ok else "lemke", active_mask=r.lam > 0)


def solve_contact_static_sparse(K, f, B, c, route="auto", active0=None, fallback=False, max_iter=100, tol=1e-10,
                                max_dense_bytes=1.0e9):
    """Static unilateral contact  K u = f + B^T lam,  B u - c >= 0,  lam >= 0,  lam . (B u - c) = 0  for a sparse
    symmetric positive definite K, without forming B K^-1 B^T (primal-dual active set, i.e. a semismooth Newton method).

    Three routes, each usable on its own:

    - "reduced"   B selects nodes (every row has one positive nonzero, distinct columns, e.g. B = I): the active
                  unknowns are fixed to their bounds and the free block of K is solved. The fast path.
    - "saddle"    general sparse B: each iteration solves [[K, -B_A^T], [B_A, 0]] with the active rows B_A.
    - "condensed" the existing dense routes (`solve_contact_static`, active set), for small problems; the reference.
    - "auto"      "reduced" when B is a node selection, otherwise "saddle". Never "condensed" by itself.

    active0 : boolean mask (m,) or index array of constraints assumed active at the start (warm start); the result's
              `active_mask` can be passed to the next call.
    A route that fails (cycling of the active set, iteration limit, a singular saddle system, a route that does not
    apply) returns `converged=False` (alias `ok`), a `reason` code, a one-sentence `message` and a `suggestion`, the
    name of the next route to try, and never returns the last iterate as if it were a solution. With
    `fallback=True` the next suggested route is tried by itself (the failed routes are listed in `fallbacks`); the
    dense route is only tried automatically when its estimated memory is at most `max_dense_bytes`.
    Guarantees: none about convergence for a stiffness matrix that is not an M-matrix (elasticity); that is why the
    flag exists. Returns `ContactResult`."""
    K = sp.csr_matrix(K) if not sp.issparse(K) else K.tocsr()
    f = np.asarray(f, float)
    c = np.asarray(c, float)
    B = sp.csr_matrix(B) if not sp.issparse(B) else B.tocsr()
    n, m = K.shape[0], B.shape[0]
    if route not in ("auto",) + _ROUTES:
        raise ValueError("route must be 'auto', 'reduced', 'saddle' or 'condensed'")
    if B.shape[1] != n or f.shape != (n,) or c.shape != (m,):
        raise ValueError("shapes: K (n, n), f (n,), B (m, n), c (m,)")
    if m == 0:
        u = spla.splu(sp.csc_matrix(K)).solve(f)
        return ContactResult(u, np.zeros(0), np.zeros(0), 0, True, route="reduced", active_mask=np.zeros(0, bool))
    mask0 = _mask(active0, m)
    if route == "auto":
        route = "reduced" if _selection_structure(B) is not None else "saddle"
    failed = []
    while True:
        if route == "reduced":
            r = _route_reduced(K, f, B, c, mask0, max_iter, tol)
        elif route == "saddle":
            r = _route_saddle(K, f, B, c, mask0, max_iter, tol)
        else:
            r = _route_condensed(K, f, B, c)
        if r.converged:
            r.fallbacks = failed
            return r
        nxt, memo = _next_route(route, n, m)
        if r.suggestion not in (None, "lemke"):
            r.suggestion = nxt
        r.message = f"route {route!r} failed ({r.reason}): {r.message}" + (f" Try route={r.suggestion!r}." if r.suggestion else "") + memo
        r.fallbacks = failed
        if not fallback or r.suggestion is None or r.suggestion == "lemke":
            return r
        if r.suggestion == "condensed" and _dense_bytes(n, m) > max_dense_bytes:
            r.reason = "memory"
            r.message += f" Not tried automatically: more than max_dense_bytes = {max_dense_bytes:.3g}."
            r.suggestion = None
            return r
        failed = failed + [route]
        route = r.suggestion


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

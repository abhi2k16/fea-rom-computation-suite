# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_greedy.py -- weak-greedy snapshot selection for parametric contact ROMs, driven by the violation of the
contact conditions (Balajewicz, Amsallem and Farhat 2015, Section 4.5).

A greedy loop needs an error indicator that does not require the full-order solution. For contact the reduced
solution u_r can be checked against the three Hertz-Signorini-Moreau conditions directly:

    r1 = B u - c                                     non-penetration (must be >= 0),
    r2 = lam_hat * (B u - c)                         complementary slackness (must be 0),
    r3 = lam_hat = (B^T)^+ (K u - f)                 contact-force positivity (must be >= 0),

and the indicator is  I = a1 |[r1]_-|^2 + a2 |r2|^2 + a3 |[r3]_-|^2  with [x]_- = min(x, 0). Evaluating it needs
matrix-vector products only, no solve; a1 = 1, a2 = a3 = 0 is the cheapest and, in the source's static test, the best.

`greedy_contact_sampling` runs the loop for any parametrisation: you supply the full-order solver, a function that
builds a ROM from the snapshots collected so far, and the indicator. The loop adds the candidate with the largest
indicator and stops at `tol` (absolute), `rel_tol` (relative, see below) or `max_snapshots`.

`contact_condition_indicator` is the reference implementation: with a2 or a3 nonzero it densifies B and runs a dense
least-squares solve on EVERY call. `ContactConditionIndicator` computes the same numbers with the pseudo-inverse work done once (or not
at all for B = I and node selections) and accepts trajectories; use it inside greedy loops.

Dynamic problems (Algorithm 2 of the source): `fom_solve` may return whole trajectories, matrices (n, T) and (m, T) with one column
per time step, instead of single vectors; the columns of all rounds are concatenated into the snapshot matrices passed to
`make_rom`. The ROM then returns a trajectory too, and `trajectory_indicator` turns a single-step indicator into the sum over the
time steps (the source's I is a sum over n). The basis sizes of Algorithm 2 grow with the round number; `make_rom` receives the round
number when `pass_round=True` (`linear_basis_schedule` builds the usual linear growth).

Relative stopping rule. Algorithm 2 line 15 is printed in the source as  I(gamma_{N+1}) > eps * I(gamma_2)  with eps < 1; read
literally it would stop at the first round whose indicator is large, so this module reads it as the evident intent: stop when the
largest indicator over the remaining candidates is at most `rel_tol` times the largest indicator after the FIRST round (the indicator
of the second sampled parameter, gamma_2). The reading is the author's; the printed inequality is most likely a typo.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["contact_condition_indicator", "ContactConditionIndicator", "GreedyContactHistory", "greedy_contact_sampling", "trajectory_indicator",
           "linear_basis_schedule"]


def contact_condition_indicator(K, B, f, c, u, weights=(1.0, 0.0, 0.0)):
    """Violation of the contact conditions by a (reduced) displacement u, per the formula above. `weights` =
    (a1, a2, a3). B may be dense or sparse. Returns a non-negative float (0 = the conditions all hold).
    Reference implementation: with a2 or a3 nonzero it builds a dense B and a least-squares solve on every call; use
    `ContactConditionIndicator` in loops."""
    u = np.asarray(u, float)
    gap = B @ u - np.asarray(c, float)
    a1, a2, a3 = weights
    val = 0.0
    if a1:
        val += a1 * np.sum(np.minimum(gap, 0.0) ** 2)
    if a2 or a3:
        Bd = B.toarray() if hasattr(B, "toarray") else np.asarray(B, float)
        lam_hat = np.linalg.lstsq(Bd.T, K @ u - f, rcond=None)[0]
        if a2:
            val += a2 * np.sum((lam_hat * gap) ** 2)
        if a3:
            val += a3 * np.sum(np.minimum(lam_hat, 0.0) ** 2)
    return float(val)


class ContactConditionIndicator:
    """The contact-condition indicator of `contact_condition_indicator` with the force estimate lam_hat = (B^T)^+ (K u - f) prepared once.

    ind = ContactConditionIndicator(K, B, f, weights=(1, 0, 0));  ind(u, c)  ->  float, for u of shape (n,) or a trajectory (n, T)
    (the sum over the T columns; f may then be (n,) or (n, T), c (m,) or (m, T)). `ind.for_parameter(c_of_p)` gives the
    `indicator(p, u)` callable that `greedy_contact_sampling` expects. The result equals `contact_condition_indicator` (the reference)
    to rounding.

    The least-squares force estimate is built lazily, only when a2 or a3 is nonzero (a = (1, 0, 0) needs K, B and f not at all), by
    the cheapest exact route (`method`):
      "identity"         B = I: lam_hat = K u - f, no solve;
      "selection"        every row of B has one nonzero in a distinct column: lam_hat_i = (K u - f)[col_i] / B_i,col_i, no solve;
      "normal_equations" a sparse B with full row rank (m <= n): one sparse LU of B B^T, checked on random vectors;
      "dense_pinv"       otherwise (rank-deficient B, or m > n): the dense pseudo-inverse of B^T, cached (m x n; refused above
                         `max_dense_bytes`, default 1e9, with a message), the same minimum-norm answer as the reference's lstsq.
    """

    def __init__(self, K, B, f, weights=(1.0, 0.0, 0.0), max_dense_bytes=1.0e9):
        self.K, self.f = K, np.asarray(f, float)
        self.B = B if sp.issparse(B) else np.asarray(B, float)
        self.weights = tuple(float(w) for w in weights)
        self.max_dense_bytes = float(max_dense_bytes)
        n = K.shape[0]
        if self.B.ndim != 2 or self.B.shape[1] != n or self.f.shape[0] != n:
            raise ValueError("shapes: K (n, n), B (m, n), f (n,) or (n, T)")
        self._m = self.B.shape[0]
        self._lam = None
        self.method = None            # set when the force estimate is first needed

    # ---- preparation of lam_hat = (B^T)^+ r ---------------------------------------------------------------------
    def _prepare(self):
        B = sp.csr_matrix(self.B)
        B.eliminate_zeros()
        m, n = B.shape
        if m == n and B.nnz == n and np.all(np.diff(B.indptr) == 1) and np.array_equal(B.indices, np.arange(n)) and np.all(B.data == 1.0):
            self.method, self._lam = "identity", lambda r: r
            return
        if m > 0 and np.all(np.diff(B.indptr) == 1) and len(np.unique(B.indices)) == m:
            cols, vals = B.indices.copy(), B.data.copy()
            self.method = "selection"
            self._lam = lambda r: r[cols] / (vals if r.ndim == 1 else vals[:, None])
            return
        if m <= n and m > 0:
            G = (B @ B.T).tocsc()
            try:
                lu = spla.splu(G)
                rng = np.random.default_rng(0)
                ok = True
                for _ in range(3):
                    g = rng.standard_normal(m)
                    x = lu.solve(g)
                    ok = ok and np.all(np.isfinite(x)) and np.linalg.norm(G @ x - g) <= 1e-10 * np.linalg.norm(g) \
                        and np.linalg.norm(x) <= 1e8 * np.linalg.norm(g)
                if ok:
                    self.method = "normal_equations"
                    self._lam = lambda r: lu.solve(np.asarray(B @ r))
                    return
            except RuntimeError:
                pass
        need = 8.0 * m * n
        if need > self.max_dense_bytes:
            raise ValueError(f"the dense pseudo-inverse of B^T needs about {need / 1e9:.2g} GB (m = {m}, n = {n}) which exceeds "
                             f"max_dense_bytes; use weights (1, 0, 0), or a B that is a node selection or has full row rank")
        Pt = np.linalg.pinv(B.toarray().T)          # (m, n): lam_hat = Pt @ r, the minimum-norm least-squares solution of B^T lam = r
        self.method, self._lam = "dense_pinv", lambda r: Pt @ r

    # ---- evaluation ----------------------------------------------------------------------------------------------
    def __call__(self, u, c, f=None):
        u = np.asarray(u, float)
        c = np.asarray(c, float)
        traj = u.ndim == 2
        gap = self.B @ u - (c[:, None] if (traj and c.ndim == 1) else c)
        a1, a2, a3 = self.weights
        val = 0.0
        if a1:
            val += a1 * np.sum(np.minimum(gap, 0.0) ** 2)
        if a2 or a3:
            if self._lam is None:
                self._prepare()
            ff = self.f if f is None else np.asarray(f, float)
            r = self.K @ u - (ff[:, None] if (traj and ff.ndim == 1) else ff)
            lam = self._lam(np.asarray(r))
            if a2:
                val += a2 * np.sum((lam * gap) ** 2)
            if a3:
                val += a3 * np.sum(np.minimum(lam, 0.0) ** 2)
        return float(val)

    def for_parameter(self, c_of_p):
        """`indicator(p, u)` for `greedy_contact_sampling`: the offset of parameter p is `c_of_p(p)`."""
        return lambda p, u: self(u, c_of_p(p))


class GreedyContactHistory:
    """selected (indices into the candidate list, in order), indicator_max (maximum indicator over the remaining
    candidates after each round), n_snapshots (parameters sampled), n_columns (snapshot columns collected: n_snapshots for vector
    snapshots, the total number of time steps for trajectories), converged, stop_reason ("tol", "rel_tol", "exhausted" when no
    candidate is left, "max_snapshots")."""

    def __init__(self):
        self.selected, self.indicator_max, self.converged = [], [], False
        self.n_columns, self.stop_reason = 0, None

    @property
    def n_snapshots(self):
        return len(self.selected)


def trajectory_indicator(step_indicator):
    """Turn an indicator for one state, `step_indicator(p, u)` with u of shape (n,), into one for a trajectory: for a ROM output
    of shape (n, T) it returns the sum of the indicator over the T columns (the source's indicator sums over the time steps); a
    1-D output is passed through unchanged. Example: `trajectory_indicator(lambda p, u: contact_condition_indicator(K, B, f, c(p), u))`
    (constant load f; for a time-dependent load write the step indicator with the time index)."""
    def indicator(p, u):
        u = np.asarray(u, float)
        if u.ndim == 1:
            return float(step_indicator(p, u))
        return float(sum(step_indicator(p, u[:, k]) for k in range(u.shape[1])))
    return indicator


def linear_basis_schedule(p_first, p_last, n_last):
    """Basis size as a function of the round number n = 1, 2, ...: p(n) = floor(p_first + (n - 1) (p_last - p_first) / (n_last - 1)),
    growing linearly from p_first at round 1 to p_last at round n_last (and beyond, clipped to p_last). The report of the contact-ROM
    validation used p = p_lambda = floor(2 + (n - 1) 98 / 9) (2 to 100 over 10 rounds) for the dynamic obstacle problem:
    `linear_basis_schedule(2, 100, 10)`. Use it inside `make_rom` with `pass_round=True`."""
    if n_last < 2 or p_last < p_first:
        raise ValueError("need n_last >= 2 and p_last >= p_first")

    def p_of(n):
        n = min(int(n), n_last)
        return int(np.floor(p_first + (n - 1) * (p_last - p_first) / (n_last - 1) + 1e-9))
    return p_of


def _columns(x, name):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"fom_solve must return {name} of shape (n,) or (n, T)")


def greedy_contact_sampling(candidates, fom_solve, make_rom, indicator, max_snapshots, tol=0.0, first=0, rel_tol=None,
                            pass_round=False):
    """Weak greedy loop.

    candidates       list of parameter values;
    fom_solve(p)     returns (u, lam), the full-order displacement and multiplier snapshots at p: vectors (n,) and (m,), or
                     trajectories (n, T) and (m, T) with one column per time step (the same T for both);
    make_rom(U, L)   builds a reduced model from the displacement snapshots U (n x k) and multiplier snapshots L (m x k) collected so
                     far (k = rounds so far for vectors, the total number of time steps for trajectories) and returns a callable
                     rom(p) -> u_reduced_lifted, (n,) or (n, T). With pass_round=True it is called make_rom(U, L, n_iter) with the
                     round number n_iter = 1, 2, ..., so the basis sizes can follow a schedule (see `linear_basis_schedule`);
    indicator(p, u) the error indicator for a ROM output u at parameter p (see contact_condition_indicator; for trajectories wrap a
                     single-state indicator with `trajectory_indicator`);
    tol              stop when the largest indicator over the remaining candidates is at most tol (absolute; 0 = never);
    rel_tol          stop when that largest indicator is at most rel_tol times the largest indicator after the first round
                     (Algorithm 2 line 15, read as described in the module docstring); checked from the second round on;
    first            index of the initial candidate.

    Returns (rom, GreedyContactHistory). Every round costs one full-order solve plus one cheap ROM solve and one
    indicator evaluation per remaining candidate."""
    if rel_tol is not None and rel_tol < 0:
        raise ValueError("rel_tol must be >= 0")
    hist = GreedyContactHistory()
    U, L = [], []
    chosen = int(first)
    rom = None
    for it in range(1, int(max_snapshots) + 1):
        u, lam = fom_solve(candidates[chosen])
        uc, lc = _columns(u, "u"), _columns(lam, "lam")
        if uc.shape[1] != lc.shape[1]:
            raise ValueError(f"fom_solve returned {uc.shape[1]} displacement columns and {lc.shape[1]} multiplier columns")
        U.append(uc)
        L.append(lc)
        hist.selected.append(chosen)
        hist.n_columns += uc.shape[1]
        Us, Ls = np.hstack(U), np.hstack(L)
        rom = make_rom(Us, Ls, it) if pass_round else make_rom(Us, Ls)
        rest = [i for i in range(len(candidates)) if i not in hist.selected]
        if not rest:
            hist.converged, hist.stop_reason = True, "exhausted"
            break
        vals = np.array([indicator(candidates[i], rom(candidates[i])) for i in rest])
        hist.indicator_max.append(float(vals.max()))
        if vals.max() <= tol:
            hist.converged, hist.stop_reason = True, "tol"
            break
        if rel_tol is not None and it >= 2 and vals.max() <= rel_tol * hist.indicator_max[0]:
            hist.converged, hist.stop_reason = True, "rel_tol"
            break
        chosen = rest[int(np.argmax(vals))]
    else:
        hist.stop_reason = "max_snapshots"
    return rom, hist

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_rid.py -- hybrid hyper-reduction of unilateral contact on a Reduced Integration Domain (RID).

Implements the method of Fauque, Ramiere and Ryckelynck (2018), "Hybrid hyper-reduced modeling for contact
mechanics problems", Int. J. Numer. Meth. Engng 115(1):117-139.  It is a different method from the ECSW route in
`contact_hyper_reduction.py`: there the element weights are fitted; here the equations are simply written on a
small piece of the mesh (the RID) and the contact forces on that piece are kept as FINITE-ELEMENT unknowns (the
"hybrid" part), so no positive dual basis is needed.

Sign convention (the PAPER's, which differs from `fea_engine.contact_solvers`):

        K U + B^T Lam = F,      B U <= D,      Lam >= 0,      Lam . (D - B U) = 0              (Eq. 16 of the source)

Pipeline (offline, then online)
-------------------------------
1. POD basis V of displacement snapshots; the basis size is set by the information rate (Eq. 18 of the source).
2. DEIM indices of V (and of a coarse POD of the multiplier snapshots) -> `rid_elements` -> RID (paper Sec. 4.2).
3. `rid_dofs` splits the DOFs into P (balance equations are written: every element touching them is in the RID)
   and `ext` (all DOFs of RID elements: what the stiffness rows of P couple to).
4. `contact_rows_in_rid`: the potential contacts whose gap can be evaluated inside the RID.
5. `HybridHRContact` enriches V with unit vectors so that B_r = B[rows] V has full row rank (the discrete inf-sup /
   LBB condition, Eq. 39-40 of the source), then solves the reduced saddle-point problem (Eq. 38 of the source) online.
6. `reconstruct_contact_forces` (in contact_hyper_reduction.py) extends the multipliers outside the RID (Eq. 41).
7. `interpolation_indicator` is the cheap a-posteriori error estimate eta of Eq. 42-43, used to drive a greedy
   choice of snapshots.

Honest scope. Frictionless, small displacement (constant B), linear elastic K in the examples (K may depend on the
parameter: it is passed in per query). Nothing here knows about a particular mesh: the caller supplies, per element,
the list of DOFs it touches.

References (methods in this module are adopted from):
  - A. Fauque, I. Ramiere, D. Ryckelynck (2018), "Hybrid hyper-reduced modeling for contact mechanics problems", Int. J. Numer. Meth. Engng 115(1):117-139
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp

from .complementarity import lcp_lemke, solve_reduced_contact
from .contact_hyper_reduction import enrich_basis_for_constraints, rank_deficiency
from .hyper_reduction import deim_indices

__all__ = ["pod_by_information_rate", "rid_elements", "rid_dofs", "contact_rows_in_rid", "HybridHRContact",
           "HybridHRResult", "restricted_fom_solve", "snapshot_interpolator", "h1_norm", "dual_h1_norm",
           "interpolation_indicator", "deim_rid_indices"]


# ----------------------------------------------------------------------------------------------------------------
# Basis and RID construction (offline)
# ----------------------------------------------------------------------------------------------------------------
def pod_by_information_rate(S, eps_tol):
    """POD basis of the snapshot matrix S (N x Ns) truncated by the information rate (Eq. 18 of Fauque et al.):
    the smallest l with  sum_{j<=l} sigma_j^2 / sum_j sigma_j^2  >=  1 - eps_tol.  Returns V (N x l)."""
    Us, s, _ = np.linalg.svd(np.asarray(S, float), full_matrices=False)
    rate = np.cumsum(s ** 2) / np.sum(s ** 2)
    l = int(np.searchsorted(rate, 1.0 - eps_tol) + 1)       # first index where the rate reaches 1 - eps_tol
    return Us[:, :min(l, len(s))]


def deim_rid_indices(*bases):
    """Union of the DEIM interpolation indices of several bases (paper Sec. 4.2: apply DEIM to the primal basis AND
    to the multiplier basis so the RID also contains the important contact zones)."""
    return np.unique(np.concatenate([deim_indices(np.asarray(b, float)) for b in bases]))


def _incident(elem_dofs, n_dof):
    """dof -> list of elements that contain it."""
    inc = [[] for _ in range(n_dof)]
    for e, dofs in enumerate(elem_dofs):
        for d in dofs:
            inc[int(d)].append(e)
    return inc


def rid_elements(elem_dofs, n_dof, dof_indices, layers=0, previous=None):
    """RID = every element in the support of the shape functions of `dof_indices` (paper Eq. 20), plus `layers`
    rings of neighbouring elements (an element is a neighbour if it shares a DOF with the current set), plus the
    elements of `previous` (the RID can only grow during a greedy loop). Returns a sorted int array."""
    inc = _incident(elem_dofs, n_dof)
    sel = set()
    for i in np.atleast_1d(dof_indices):
        sel.update(inc[int(i)])
    for _ in range(layers):                                   # add one ring of elements around the current set
        dofs = {int(d) for e in sel for d in elem_dofs[e]}
        for d in dofs:
            sel.update(inc[d])
    if previous is not None:
        sel.update(int(e) for e in previous)
    return np.array(sorted(sel), dtype=int)


def rid_dofs(elem_dofs, n_dof, rid_elems):
    """Split the DOFs for the RID.
    P    : DOFs ALL of whose incident elements lie in the RID. The balance equation of such a DOF can be written
           exactly on the RID (the set Gamma-hat-sub-A of the source).
    ext  : every DOF of every RID element. The rows of K for P only couple to ext (the source's union set), so only
           K[P, ext] is needed online.
    Returns (P, ext), both sorted int arrays. ext \\ P is the interface Gamma_I where pseudo-Dirichlet data lives."""
    inc = _incident(elem_dofs, n_dof)
    inrid = np.zeros(len(elem_dofs), bool)
    inrid[np.asarray(rid_elems, int)] = True
    P = np.array([d for d in range(n_dof) if inc[d] and all(inrid[e] for e in inc[d])], dtype=int)
    ext = np.unique(np.concatenate([np.asarray(elem_dofs[e], int) for e in rid_elems])) if len(rid_elems) else P
    return P, ext


def contact_rows_in_rid(B, P):
    """Rows of the potential-contact matrix B whose gap can be evaluated with DOFs of P only (every nonzero column
    of the row lies in P). For node-to-node / obstacle contact this is simply the contact nodes inside P; for
    node-to-surface contact rows touching the interface are dropped (the Gamma_I^C extension of the source,
    (Sec. 5.1): there the gap is not computable, so no multiplier is kept."""
    Bc = sp.csr_matrix(B)
    inP = np.zeros(Bc.shape[1], bool)
    inP[np.asarray(P, int)] = True
    rows = [i for i in range(Bc.shape[0])
            if Bc.indptr[i + 1] > Bc.indptr[i] and inP[Bc.indices[Bc.indptr[i]:Bc.indptr[i + 1]]].all()]
    return np.array(rows, dtype=int)


# ----------------------------------------------------------------------------------------------------------------
# Online solve
# ----------------------------------------------------------------------------------------------------------------
class HybridHRResult:
    """u (N,) reconstructed displacement V gamma; lam (len(rows),) multipliers on the RID contacts; gamma; ok."""

    def __init__(self, u, lam, gamma, ok):
        self.u, self.lam, self.gamma, self.ok = u, lam, gamma, ok


class HybridHRContact:
    """Hybrid hyper-reduced contact model (Eq. 38 of the source).

    V        (N x l) POD basis;      B (m x N) potential-contact matrix;
    P, ext   from `rid_dofs`;        rows from `contact_rows_in_rid`.
    enrich   append unit vectors to V until B[rows] V has full row rank (Eq. 39-40 of the source). Without it the reduced
             saddle-point problem is singular whenever len(rows) > l.

    Online `solve(K_Pext, F_P, D_rows)` needs ONLY the RID blocks K[P, ext], F[P] and D[rows]: the cost does not
    depend on the full mesh size (apart from forming V gamma at the end, which is the reconstruction on Omega).
    """

    def __init__(self, V, B, P, ext, rows, enrich=True):
        self.P, self.ext, self.rows = np.asarray(P, int), np.asarray(ext, int), np.asarray(rows, int)
        self.B = sp.csr_matrix(B)
        V = np.asarray(V, float)
        self.n_modes = V.shape[1]
        if enrich and len(self.rows):
            V, added = enrich_basis_for_constraints(V, self.B, rows=self.rows)
        else:
            added = np.zeros(0, int)
        self.V, self.added_dofs = V, added
        self.V_P, self.V_ext = V[self.P], V[self.ext]                 # test space rows / trial space rows
        self.BV = (self.B[self.rows] @ V) if len(self.rows) else np.zeros((0, V.shape[1]))   # reduced constraints

    @property
    def n_unknowns(self):
        """Primal reduced unknowns + RID multipliers (the 'total number of unknowns' of the source's Table 1)."""
        return self.V.shape[1] + len(self.rows)

    def inf_sup_deficiency(self):
        """0 when B_r = B[rows] V has full row rank (extended LBB condition of Property 1)."""
        return rank_deficiency(self.B, self.V, rows=self.rows)

    def solve(self, K_Pext, F_P, D_rows):
        """Solve   (V_P^T K[P,ext] V_ext) gamma + BV^T Lam = V_P^T F_P ,   BV gamma <= D_rows,  Lam >= 0,
        complementarity.  The matrix is NOT symmetric (Petrov-Galerkin) and its condition number can reach 1e14, so
        a primal-dual active-set loop on the saddle-point system may pick a wrong contact set. Instead gamma is
        eliminated, giving an LCP in Lam alone:
              gamma = A^-1 (b - BV^T Lam),     w = (D - BV A^-1 b) + (BV A^-1 BV^T) Lam >= 0,  Lam >= 0,  Lam.w = 0
        which Lemke's algorithm solves."""
        A = self.V_P.T @ (K_Pext @ self.V_ext)
        b = self.V_P.T @ np.asarray(F_P, float)
        try:
            Ai_b = np.linalg.solve(A, b)
            Ai_Bt = np.linalg.solve(A, self.BV.T)
        except np.linalg.LinAlgError:                                  # singular A: least-squares fallback
            Ai_b = np.linalg.lstsq(A, b, rcond=None)[0]
            Ai_Bt = np.linalg.lstsq(A, self.BV.T, rcond=None)[0]
        if not len(self.rows):
            return HybridHRResult(self.V @ Ai_b, np.zeros(0), Ai_b, True)
        lam, ok = lcp_lemke(self.BV @ Ai_Bt, np.asarray(D_rows, float) - self.BV @ Ai_b)
        gamma = Ai_b - Ai_Bt @ lam
        return HybridHRResult(self.V @ gamma, lam, gamma, bool(ok))


# ----------------------------------------------------------------------------------------------------------------
# Error indicator eta (paper Sec. 5.4)
# ----------------------------------------------------------------------------------------------------------------
def restricted_fom_solve(K_PP, K_PI, F_P, u_I, B_P, D, B_I=None):
    """Full-order contact problem written ONLY on the RID, with the interface DOFs imposed (paper Sec. 5.4):
         K_PP u_P + B_P^T Lam = F_P - K_PI u_I ,   B_P u_P <= D - B_I u_I ,   Lam >= 0,  complementarity.
    K_PP (|P|x|P|) must be SPD. Returns (u_P, Lam, ok). It reuses `solve_reduced_contact`, whose convention is
    K u = b + B^T lam with B u >= c, hence the sign flips below."""
    b = np.asarray(F_P, float) - K_PI @ np.asarray(u_I, float)
    c = np.asarray(D, float) - (0.0 if B_I is None else B_I @ np.asarray(u_I, float))
    if B_P.shape[0] == 0:
        return np.linalg.solve(K_PP, b), np.zeros(0), True
    u, lam, _, ok = solve_reduced_contact(K_PP, b, -np.asarray(B_P, float), -c)
    return u, lam, ok


def snapshot_interpolator(params, values):
    """Interpolate snapshot columns in the parameter space: values (n x Ns) at params (Ns x d) -> callable mu -> (n,).
    Linear (Delaunay) interpolation inside the convex hull of the sampled parameters -- the scattered-data version
    of the bilinear interpolation used on a regular grid -- and nearest neighbour outside it or when there are too
    few / degenerate points (early greedy iterations)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    X = np.atleast_2d(np.asarray(params, float))
    Y = np.asarray(values, float).T
    near = NearestNDInterpolator(X, Y)
    lin = None
    if len(X) > X.shape[1]:
        try:
            lin = LinearNDInterpolator(X, Y)
        except Exception:                                             # Qhull fails for collinear points
            lin = None

    def f(mu):
        mu = np.atleast_2d(np.asarray(mu, float))
        out = lin(mu)[0] if lin is not None else None
        if out is None or np.any(np.isnan(out)):
            out = near(mu)[0]
        return out
    return f


def h1_norm(x, M_H1):
    """sqrt(x^T M x) with M the H1 Gram (mass + stiffness) matrix of the primal space."""
    x = np.asarray(x, float)
    return float(np.sqrt(max(x @ (M_H1 @ x), 0.0)))


def dual_h1_norm(x, M_H1):
    """(H1)' norm of a dual vector: sqrt(x^T M^-1 x) (paper Sec. 2, notation: <v,v'>_F = xi^T M_E^-1 xi')."""
    x = np.asarray(x, float)
    M = M_H1.toarray() if sp.issparse(M_H1) else np.asarray(M_H1, float)
    return float(np.sqrt(max(x @ np.linalg.solve(M, x), 0.0)))


def interpolation_indicator(u_hr, lam_hr, u_ref, lam_ref, M_H1_P, M_H1_rows):
    """eta = eta_u + eta_lam (paper Eq. 43), the relative discrepancy between the hyper-reduced solution and the
    FOM solved on the RID with interpolated interface data (u_ref, lam_ref from `restricted_fom_solve`):
         eta_u   = ||u_hr - u_ref||_H1 / ||u_ref||_H1         on P,
         eta_lam = ||lam_hr - lam_ref||_(H1)' / ||lam_ref||_(H1)'   on the RID contacts.
    No full-order solve at the candidate parameter is needed: that is what makes it usable in a greedy loop.
    Returns (eta, eta_u, eta_lam)."""
    nu = h1_norm(u_ref, M_H1_P)
    eu = h1_norm(np.asarray(u_hr) - u_ref, M_H1_P) / nu if nu > 0 else 0.0
    nl = dual_h1_norm(lam_ref, M_H1_rows) if len(lam_ref) else 0.0
    el = dual_h1_norm(np.asarray(lam_hr) - lam_ref, M_H1_rows) / nl if nl > 0 else 0.0
    return eu + el, eu, el

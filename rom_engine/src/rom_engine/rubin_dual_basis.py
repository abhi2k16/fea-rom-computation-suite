# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
rubin_dual_basis.py -- Rubin free-interface basis enriched with dual modes, for structures with a (frictional)
contact patch, and re-identification of the reduced polynomial force from a reduced time history (after
Mashayekhi and Zucca, 2026):

- `rubin_dual_basis`  the free-interface (Rubin) basis from `cms.free_interface_cms` with the contact DOFs as
  the interface, plus dual modes from `dual_modes.dual_modes` computed with the OPEN-contact system (the model
  without contact forces, so `fom_solver` is the open-contact nonlinear static response), made M-orthogonal to
  the modes AND to the residual-flexibility attachment modes, then compressed. Candidate dual modes can be ranked
  by their REPRESENTATION ERROR of training states (`select_by_representation_error`) in addition to the
  linearised strain-energy score `dual_modes` already uses.
- `physical_contact_basis`  change of coordinates inside the same subspace so that the first columns have the
  identity on the contact DOFs, i.e. the contact DOFs are carried as physical coordinates (what a contact law
  needs). The span of the basis is unchanged.
- `reidentify_polynomial_force`  least-squares fit of the quadratic and cubic reduced force from a reduced time
  history of an intrusive run: M q'' + C q' + K q + g(q) = f. Coefficients identified from open-contact static
  solves can be inaccurate in slip-lift-off regimes; re-fitting from one dynamic run is the remedy this function
  provides (the accuracy gain itself is a finding of the source and is not reproduced here).

Scope: linear-algebra level; K must be non-singular (as in `cms.free_interface_cms`); dual modes need the
caller's nonlinear static solver; no friction law is built in (see `friction_contact_models`).

References (methods in this module are adopted from):
  - F. Mashayekhi, S. Zucca (2026), "An efficient reduced order modeling for vibration prediction of geometrically nonlinear structures with friction contact", Nonlinear Dyn. 114:327
  - S. Rubin (1975), "Improved component-mode representation for structural dynamic analysis", AIAA J. 13(8)
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools
import numpy as np

from .cms import free_interface_cms
from .dual_modes import dual_modes

__all__ = ["rubin_dual_basis", "select_by_representation_error", "physical_contact_basis", "PolynomialForceFit",
           "reidentify_polynomial_force"]


def _mnorm_orthogonalise(X, against, M):
    """Remove from the columns of X their M-projection on span(against)."""
    Q, _ = np.linalg.qr(against)
    # QR is Euclidean-orthonormal; project in the M inner product with a Gram solve
    G = against.T @ M @ against
    coef = np.linalg.solve(G + 1e-14 * np.trace(G) / len(G) * np.eye(len(G)), against.T @ M @ X)
    return X - against @ coef


def _compress(X, M, tol):
    G = X.T @ M @ X
    lam, W = np.linalg.eigh(0.5 * (G + G.T))
    lam, W = lam[::-1], W[:, ::-1]
    keep = lam > tol * max(float(lam[0]), 1e-300)
    return X @ (W[:, keep] / np.sqrt(lam[keep]))


def select_by_representation_error(candidates, snapshots, base, M, n_select):
    """Greedy forward selection of columns of `candidates` that most reduce the M-norm representation error of
    `snapshots` (n, s) in span([base, selected]). Returns (indices, errors) with errors[k] the relative error after
    k selections (errors[0] is for `base` alone)."""
    S = np.asarray(snapshots, float)
    chosen, current = [], np.asarray(base, float)

    def rel_err(B):
        G = B.T @ M @ B
        coef = np.linalg.solve(G + 1e-14 * np.trace(G) / len(G) * np.eye(len(G)), B.T @ M @ S)
        R = S - B @ coef
        return float(np.sqrt(np.sum(R * (M @ R))) / np.sqrt(np.sum(S * (M @ S))))
    errors = [rel_err(current)]
    for _ in range(int(n_select)):
        best, best_e = None, np.inf
        for j in range(candidates.shape[1]):
            if j in chosen:
                continue
            e = rel_err(np.hstack([current, candidates[:, [j]]]))
            if e < best_e:
                best, best_e = j, e
        if best is None:
            break
        chosen.append(best)
        current = np.hstack([current, candidates[:, [best]]])
        errors.append(best_e)
    return chosen, errors


def rubin_dual_basis(K, M, contact_dofs, n_modes, fom_solver, amplitudes, p=None, dominant=None, n_dual=None,
                     snapshots=None, n_keep_per_set=None, tol=1e-10):
    """Rubin basis plus dual modes. Returns (T, info): T = [free-interface modes | residual attachment modes | dual
    modes], info with 'n_modes', 'n_attachment', 'n_dual', 'dual_info' (the `dual_modes` info), 'orthogonality'
    (max |T_dual^T M T_rubin| after orthogonalisation) and, if `snapshots` are given, the representation errors
    without and with the dual modes. `fom_solver(load) -> displacement` is the open-contact nonlinear static response."""
    K, M = np.asarray(K, float), np.asarray(M, float)
    fi = free_interface_cms(K, M, contact_dofs, n_modes, method="rubin")
    T0 = fi.T
    Phi = T0[:, :fi.n_modes]
    # dual_modes expects M-orthonormal modes: the Rubin free-interface modes are
    _, dinfo = dual_modes(K, M, Phi, fom_solver, amplitudes, p=p, dominant=dominant, n_dual=n_dual,
                          n_keep_per_set=n_keep_per_set, tol=tol)
    duals = dinfo["duals"]
    n_dual_raw = duals.shape[1]
    if n_dual_raw:
        duals = _mnorm_orthogonalise(duals, T0, M)          # against modes AND attachment modes
        duals = _compress(duals, M, 1e-10)
    out = np.hstack([T0, duals])
    orth = float(np.max(np.abs(duals.T @ M @ T0))) if duals.shape[1] else 0.0
    info = dict(n_modes=fi.n_modes, n_attachment=T0.shape[1] - fi.n_modes, n_dual=duals.shape[1],
                dual_info=dinfo, orthogonality=orth)
    if snapshots is not None and duals.shape[1]:
        order, errs = select_by_representation_error(duals, snapshots, T0, M, duals.shape[1])
        info["representation_error_rubin"] = errs[0]
        info["representation_error_with_dual"] = errs[-1]
        info["dual_selection_order"] = order
        info["representation_error_by_count"] = errs
    return out, info


def physical_contact_basis(T, contact_dofs):
    """Basis spanning the same subspace as T whose first len(contact_dofs) columns equal the identity on the contact
    DOFs (contact DOFs are physical coordinates), followed by columns that vanish on the contact DOFs. Requires the
    contact rows of T to have full row rank."""
    T = np.asarray(T, float)
    c = np.asarray(contact_dofs, int)
    Tc = T[c]
    if np.linalg.matrix_rank(Tc) < len(c):
        raise ValueError("the basis cannot represent independent motion of every contact DOF")
    U, s, Vt = np.linalg.svd(Tc)
    nc = len(c)
    Tc_pinv = Vt[:nc].T @ np.diag(1.0 / s[:nc]) @ U.T          # (m, nc), Tc @ Tc_pinv = I
    N = Vt[nc:].T                                             # null space of Tc
    return np.hstack([T @ Tc_pinv, T @ N]) if N.shape[1] else T @ Tc_pinv


class PolynomialForceFit:
    """g(q) = sum over monomials (index tuples) coeff[m, :] * prod q[idx]; force(q) for a single state (r,) or a
    history (nt, r); jacobian(q) for a single state."""

    def __init__(self, monomials, coeff, r):
        self.monomials, self.coeff, self.r = monomials, coeff, r

    def _features(self, Q):
        Q = np.atleast_2d(Q)
        return np.column_stack([np.prod(Q[:, list(m)], axis=1) for m in self.monomials])

    def force(self, q):
        q = np.asarray(q, float)
        out = self._features(q) @ self.coeff
        return out[0] if q.ndim == 1 else out

    def jacobian(self, q):
        q = np.asarray(q, float)
        J = np.zeros((self.r, self.r))
        for m, c in zip(self.monomials, self.coeff):
            for pos in range(len(m)):
                rest = np.prod([q[k] for i, k in enumerate(m) if i != pos]) if len(m) > 1 else 1.0
                J[:, m[pos]] += c * rest
        return J


def reidentify_polynomial_force(M_r, C_r, K_r, q, qd, qdd, f_ext, orders=(2, 3), ridge=0.0):
    """Fit quadratic/cubic reduced force coefficients from a reduced time history.

    q, qd, qdd, f_ext : (nt, r) reduced displacement, velocity, acceleration and applied-force histories (from an
    intrusive run). The nonlinear force is the part of the balance the linear operators leave over,
    g = f - M qdd - C qd - K q, and each coordinate's g is regressed on all monomials of the requested orders.
    Returns (PolynomialForceFit, info) with info['relative_residual']."""
    q, qd, qdd, f_ext = (np.asarray(v, float) for v in (q, qd, qdd, f_ext))
    nt, r = q.shape
    g = f_ext - qdd @ np.asarray(M_r).T - qd @ np.asarray(C_r).T - q @ np.asarray(K_r).T
    monomials = []
    for o in orders:
        monomials += list(itertools.combinations_with_replacement(range(r), o))
    fit = PolynomialForceFit(monomials, np.zeros((len(monomials), r)), r)
    X = fit._features(q)
    if ridge > 0:
        A = X.T @ X + ridge * np.eye(X.shape[1])
        coeff = np.linalg.solve(A, X.T @ g)
    else:
        coeff = np.linalg.lstsq(X, g, rcond=None)[0]
    fit.coeff = coeff
    res = float(np.linalg.norm(X @ coeff - g) / max(np.linalg.norm(g), 1e-300))
    return fit, dict(relative_residual=res, n_terms=len(monomials))

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_hyper_reduction.py -- hyper-reduction for unilateral contact problems.

Hyper-reduction (ECSW) replaces the sum over all elements of the internal force by a weighted sum over a few.
Contact adds three things it does not handle (Fauque, Ramiere and Ryckelynck 2018; Nikiema et al. 2026):

1. the elements around the potential contact zone should be evaluated EXACTLY, with weight one, so the region
   where the constraints act keeps full fidelity: `fit_contact_aware_ecsw` fits the weights of all other
   elements to the remaining force, with the required elements held at weight one;
2. a contact constraint can only be evaluated if every element it needs is present, so the selected set has to
   be completed: `complete_constraint_elements`;
3. the contact forces outside the selected region are not computed by the reduced model, so they are rebuilt by
   a non-negative least-squares fit to snapshots: `reconstruct_contact_forces` (Fauque et al., their Eq. 41);
   and the reduced saddle-point problem is only well posed when B_r = B V has full row rank on the active
   constraints, which `rank_deficiency` measures and `enrich_basis_for_constraints` repairs by appending unit
   vectors at constraint DOFs (the hybrid reduced / full-order basis of the source).

`solve_reduced_contact_newton` is the reduced counterpart of `fea_engine.contact_solvers.solve_contact_newton`:
sequential linearisation with the hyper-reduced force and tangent and a reduced LCP at every iterate.

Honest scope. Frictionless, small-displacement constraints (constant B). The ECSW weights are trained on smooth
states; whether they transfer across changes of the active contact set is not guaranteed and is measured in the
tests, not assumed.

References (methods in this module are adopted from):
  - A. Fauque, I. Ramiere, D. Ryckelynck (2018), "Hybrid hyper-reduced modeling for contact mechanics problems", Int. J. Numer. Meth. Engng 115(1):117-139
  - P. Nikiema, V. Denis, F. Matzke, S. Gonzalez, S. Van Der Veen, J.-M. Mencik (2026), "Model order reduction for structural assembly simulations with contact and geometric nonlinearities", Finite Elem. Anal. Des. 260:104605
  - C. Farhat, P. Avery, T. Chapman, J. Cortial (2014), "Dimensional reduction of nonlinear finite element dynamic models with finite rotations and energy-based mesh sampling and weighting for computational efficiency", Int. J. Numer. Meth. Engng 98(9)
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import qr
from scipy.optimize import nnls

from .complementarity import solve_reduced_contact
from .hyper_reduction import ECSW, ecsw_training_matrix, ecsw_weights

__all__ = ["ecsw_weights_with_required", "fit_contact_aware_ecsw", "complete_constraint_elements",
           "reconstruct_contact_forces", "rank_deficiency", "enrich_basis_for_constraints",
           "solve_reduced_contact_newton", "ReducedContactNewtonResult"]


def ecsw_weights_with_required(G, b, required, tol=1e-6, max_elements=None):
    """ECSW weights with the elements in `required` pinned at weight one: the remaining columns are fitted to
    b - G[:, required].sum(1) by the same sparse non-negative least squares as `ecsw_weights`. Returns (w, info)."""
    G = np.asarray(G, float)
    b = np.asarray(b, float)
    n = G.shape[1]
    req = np.unique(np.asarray(required, int)) if len(required) else np.zeros(0, int)
    free = np.setdiff1d(np.arange(n), req)
    target = b - G[:, req].sum(axis=1)
    w = np.zeros(n)
    w[req] = 1.0
    info = dict(residual=0.0, n_selected=len(req), iterations=0)
    if free.size and np.linalg.norm(target) > 0:
        cap = None if max_elements is None else max(0, int(max_elements) - len(req))
        wf, inf2 = ecsw_weights(G[:, free], target, tol=tol * np.linalg.norm(b) / max(np.linalg.norm(target), 1e-300),
                                max_elements=cap)
        w[free] = wf
        info = dict(residual=float(np.linalg.norm(b - G @ w) / max(np.linalg.norm(b), 1e-300)),
                    n_selected=int(np.count_nonzero(w)), iterations=inf2["iterations"])
    else:
        info["residual"] = float(np.linalg.norm(b - G @ w) / max(np.linalg.norm(b), 1e-300))
    return w, info


def fit_contact_aware_ecsw(V, snapshots, elem_dofs, elem_force_fn, required_elements, elem_tangent_fn=None,
                           tol=1e-6, max_elements=None):
    """`ECSW.fit` with `required_elements` (for instance every element touching a potential contact DOF) kept at
    weight one. Returns an ordinary `ECSW`, usable wherever an ECSW is."""
    G, b = ecsw_training_matrix(V, snapshots, elem_dofs, elem_force_fn)
    w, info = ecsw_weights_with_required(G, b, required_elements, tol=tol, max_elements=max_elements)
    return ECSW(V, elem_dofs, elem_force_fn, w, elem_tangent_fn, info)


def complete_constraint_elements(selected, constraint_elements):
    """Complete a selected element set so that every constraint it touches can be evaluated.

    selected: iterable of element indices; constraint_elements: list (one entry per constraint) of the element
    indices that constraint's gap needs (for a node-to-segment constraint, the elements holding the slave node and
    the master segment). A constraint with at least one selected element is added in full. Returns
    (completed_elements sorted array, computable_constraints sorted array)."""
    sel = set(int(e) for e in selected)
    computable = []
    for j, els in enumerate(constraint_elements):
        els = [int(e) for e in els]
        if any(e in sel for e in els):
            computable.append(j)
            sel.update(els)
    return np.array(sorted(sel), int), np.array(computable, int)


def reconstruct_contact_forces(S_lam, inside, lam_inside, gap):
    """Contact forces on ALL constraints from the values on the `inside` ones (Fauque et al. 2018, Eq. 41).

    S_lam (m x k): multiplier snapshots; inside: indices of the constraints the reduced model resolves;
    lam_inside: the reduced model's multipliers there; gap (m,): the gap B u - c from the reduced displacement.
    Solves  min_{g >= 0} || [S[inside] g - lam_inside ; max(gap, 0) * (S g)] ||  (match inside, and keep forces off
    the constraints that are open) and returns S g >= 0."""
    S = np.asarray(S_lam, float)
    inside = np.asarray(inside, int)
    A = np.vstack([S[inside], np.maximum(np.asarray(gap, float), 0.0)[:, None] * S])
    rhs = np.concatenate([np.asarray(lam_inside, float), np.zeros(S.shape[0])])
    g, _ = nnls(A, rhs)
    return S @ g


def rank_deficiency(B, V, rows=None, tol=1e-10):
    """Number of reduced constraints that are linearly dependent: rows - rank(B[rows] V). Zero means B_r has full
    row rank, the discrete inf-sup (LBB) condition of Fauque et al. 2018. `rows` selects the constraints."""
    Bd = B.toarray() if hasattr(B, "toarray") else np.asarray(B, float)
    Br = (Bd if rows is None else Bd[np.asarray(rows, int)]) @ np.asarray(V, float)
    if Br.size == 0:
        return 0
    s = np.linalg.svd(Br, compute_uv=False)
    rank = int(np.sum(s > tol * max(s[0], 1e-300)))
    return Br.shape[0] - rank


def enrich_basis_for_constraints(V, B, rows=None, tol=1e-10):
    """Append unit vectors at constraint DOFs until B[rows] V' has full row rank (hybrid reduced / full-order
    basis, Fauque et al. 2018). The new columns are orthonormalised against V. Returns (V_new, added_dofs)."""
    Bd = B.toarray() if hasattr(B, "toarray") else np.asarray(B, float)
    Bs = Bd if rows is None else Bd[np.asarray(rows, int)]
    V = np.asarray(V, float)
    d = rank_deficiency(Bs, V, tol=tol)
    if d == 0:
        return V, np.zeros(0, int)
    Br = Bs @ V
    U, s, _ = np.linalg.svd(Br, full_matrices=True)
    rank = int(np.sum(s > tol * max(s[0], 1e-300)))
    N = np.eye(Br.shape[0]) - U[:, :rank] @ U[:, :rank].T                 # directions of constraint space not yet reached
    support = np.flatnonzero(np.any(Bs != 0, axis=0))
    cand = N @ Bs[:, support]
    _, _, piv = qr(cand, pivoting=True, mode="economic")
    added, cols = [], [V]
    Q = V.copy()
    for j in piv:
        dof = int(support[j])
        e = np.zeros(V.shape[0])
        e[dof] = 1.0
        e -= Q @ (Q.T @ e)
        if np.linalg.norm(e) < 1e-10:
            continue
        e /= np.linalg.norm(e)
        trial = np.column_stack([Q, e])
        added.append(dof)
        Q = trial
        if rank_deficiency(Bs, Q, tol=tol) == 0:
            break
    return Q, np.array(added, int)


class ReducedContactNewtonResult:
    """q (reduced coordinates), u (lifted displacements), lam_r, lam (lifted multipliers when U_lam is given),
    iterations, converged."""

    def __init__(self, q, u, lam_r, lam, iterations, converged):
        self.q, self.u, self.lam_r, self.lam, self.iterations, self.converged = q, u, lam_r, lam, iterations, converged


def solve_reduced_contact_newton(ecsw, f_ext, V, B, c, U_lam=None, q0=None, tol=1e-9, max_iter=60, method="active_set"):
    """Unilateral contact with a hyper-reduced nonlinear internal force.

    Residual R_r(q) = ecsw.reduced_force(q) - V^T f_ext, tangent K_r(q) = ecsw.reduced_tangent(q). The constraints
    B V q >= c are reduced as B_r = U_lam^T B V, c_r = U_lam^T c with a non-negative dual basis `U_lam`; with
    U_lam=None every constraint is kept (B_r = B V, c_r = c). Sequential linearisation: at each iterate the linear
    contact problem K_r dq = -R_r + B_r^T lam, B_r (q + dq) >= c_r is solved by the reduced LCP route."""
    V = np.asarray(V, float)
    Bd = B.toarray() if hasattr(B, "toarray") else np.asarray(B, float)
    c = np.asarray(c, float)
    Ul = np.eye(Bd.shape[0]) if U_lam is None else np.asarray(U_lam, float)
    Br = Ul.T @ (Bd @ V)
    cr = Ul.T @ c
    b_r = V.T @ np.asarray(f_ext, float)
    q = np.zeros(V.shape[1]) if q0 is None else np.asarray(q0, float).copy()
    lam_r = np.zeros(Br.shape[0])
    scale = max(1.0, np.linalg.norm(b_r))
    for it in range(1, max_iter + 1):
        R = ecsw.reduced_force(q) - b_r
        Kt = ecsw.reduced_tangent(q)
        # K_t dq = -R + B_r^T lam,  B_r (q + dq) >= c_r   ->  reduced contact problem in dq
        dq, lam_r, _, ok = solve_reduced_contact(Kt, -R, Br, cr - Br @ q, method)
        q = q + dq
        res = np.linalg.norm(ecsw.reduced_force(q) - b_r - Br.T @ lam_r)
        if res < tol * scale and np.linalg.norm(dq) < 1e-9 * (1.0 + np.linalg.norm(q)):
            return ReducedContactNewtonResult(q, V @ q, lam_r, Ul @ lam_r, it, True)
    return ReducedContactNewtonResult(q, V @ q, lam_r, Ul @ lam_r, max_iter, False)

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
modal_derivatives.py -- 
modal derivatives and their selection and use as a basis augmentation, after Idelsohn and Cardona (1985),
Weeger, Wever and Simeon (2016), Wu and Tiso (2016), Jain, Tiso, Rutzmoser and Rixen (2017), and the overview in
Tiso, Karamooz Mahdiabadi and Marconi (2021), Section 4.5.2.

THE IDEA. A geometrically nonlinear structure stretches when it bends, but vibration modes of a flat or slender
structure carry almost no membrane motion. The derivative of mode i with respect to the amplitude q_j of mode j is
the displacement shape that has to be ADDED to mode i when the structure deflects finitely along mode j. Adding it
to the basis gives the reduced model the in-plane (membrane) freedom it lacks.

THE MATHEMATICS. With u = sum_j phi_j q_j, differentiating the free-vibration problem (K - w_i^2 M) phi_i = 0 with
respect to q_j gives the dynamic modal derivative (a singular system, solved here with a bordered matrix)

    (K - w_i^2 M) theta_ij = -(dK/dq_j - dw_i^2/dq_j M) phi_i ,   phi_i^T M theta_ij = 0 ,

and dropping the inertial terms gives the static modal derivative

    K theta_ij = -(dK/dq_j) phi_i ,                               theta_ij = theta_ji   (chapter Eq. 4.55).

K is already factorised for the static problem, so every derivative needs one new right-hand side only.

dK/dq_j is the directional derivative of the TANGENT stiffness along phi_j at the reference state. The package
boundary is kept as everywhere else: this module takes a plain callable `tangent_fn(u) -> K_T(u)` (dense or scipy
sparse) or precomputed derivative matrices, never an fea_engine object. The derivative is a central difference of
`tangent_fn`; a symmetric third derivative makes the static derivatives symmetric, and the size of the asymmetry
that is measured here is therefore also a check on the finite-difference step.

WHAT IS IMPLEMENTED
  directional_stiffness_derivative -- central difference of the tangent along one direction
  static_modal_derivatives         -- theta_ij from K, directions and a tangent callable (or given dK matrices)
  dynamic_modal_derivatives        -- the same for the dynamic form (distinct eigenvalues only)
  modal_coordinates                -- reduced coordinates q(t) = Phi^T M u(t) of a time history
  modal_interaction_weights        -- w_ij = integral |q_i q_j| dt from a cheap linear run
  select_modal_derivatives         -- rank and choose a subset (the full set grows as m(m+1)/2)
  augment_with_derivatives         -- orthogonalise against the modes, compress, return the augmented basis

WHAT IS NOT CLAIMED. The ranking heuristic is the maximum-modal-interaction idea of Jain et al. (2017); other
selection heuristics in the literature are not implemented. The `directions` need not be vibration modes (Krylov or
other vectors work for the static form, as in Rutzmoser et al. 2017) but symmetry theta_ij = theta_ji then holds only
if the directions enter a symmetric third derivative, which is the case for any smooth potential.

References (methods in this module are adopted from):
  - I. Idelsohn, A. Cardona (1985), "A reduction method for nonlinear structural dynamic analysis", Comput. Methods Appl. Mech. Engrg. 49(3)
  - L. Wu, P. Tiso (2016), "Nonlinear model order reduction for flexible multibody dynamics: a modal derivatives approach", Multibody Syst. Dyn. 36
  - S. Jain, P. Tiso, J. B. Rutzmoser, D. J. Rixen (2017), "A quadratic manifold for model order reduction of nonlinear structural dynamics", Comput. Struct. 188
  - J. B. Rutzmoser, D. J. Rixen, P. Tiso, S. Jain (2017), "Generalization of quadratic manifolds for reduced order modeling of nonlinear structural dynamics", Comput. Struct. 192
  - P. Tiso, M. Karamooz Mahdiabadi, J. Marconi (2021), "Modal methods for reduced order modeling", Chapter 4, ETH Research Collection, doi:10.3929/ethz-b-000526692
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh, lu_factor, lu_solve


def _as_array(x):
    return np.asarray(x, dtype=float)


def _factorize(A):
    """Return a callable solving A x = b for a dense or scipy-sparse square A."""
    if sp.issparse(A):
        lu = spla.splu(sp.csc_matrix(A))
        return lu.solve
    lu = lu_factor(_as_array(A))
    return lambda b: lu_solve(lu, b)


def _matmul(A, B):
    """A @ B for dense or sparse A and a dense B, always returning a dense ndarray."""
    out = A @ B
    return out.toarray() if sp.issparse(out) else np.asarray(out)


def directional_stiffness_derivative(tangent_fn, direction, u0=None, step=None, length_scale=1.0, rel_step=1e-6):
    """dK/dq along `direction`: [K_T(u0 + eps d) - K_T(u0 - eps d)] / (2 eps), a central difference.

    Parameters
    ----------
    tangent_fn : callable(u) -> (n, n) tangent stiffness (dense or scipy sparse).
    direction : (n,) displacement shape (typically a mass-normalised vibration mode).
    u0 : (n,) reference state, default zero.
    step : eps, used as given if not None.
    length_scale, rel_step : if `step` is None, eps = rel_step * length_scale / max|direction|, i.e. the largest
        displacement perturbation is rel_step * length_scale. Set `length_scale` to a typical displacement
        magnitude of the model (its length, for a beam) so the step is neither lost in round-off nor large
        enough to feel the cubic terms.

    Returns the derivative matrix, dense or sparse like `tangent_fn`'s output.
    """
    d = _as_array(direction)
    u0 = np.zeros_like(d) if u0 is None else _as_array(u0)
    if step is None:
        m = float(np.max(np.abs(d)))
        if m == 0.0:
            raise ValueError("direction is zero")
        step = rel_step * length_scale / m
    Kp = tangent_fn(u0 + step * d)
    Km = tangent_fn(u0 - step * d)
    return (Kp - Km) / (2.0 * step)


@dataclass
class ModalDerivatives:
    """Result of `static_modal_derivatives` / `dynamic_modal_derivatives`.

    theta[:, i, j] is d phi_i / d q_j, shape (n, m, m). `asymmetry` is max |theta_ij - theta_ji| relative to the
    largest derivative, measured BEFORE any symmetrisation (it is a finite-difference quality indicator for the
    static form; the dynamic form is not symmetric in general). `n_tangent_evals` counts calls to `tangent_fn`.
    """
    theta: np.ndarray
    directions: np.ndarray
    asymmetry: float
    n_tangent_evals: int
    kind: str

    @property
    def n_modes(self):
        return self.directions.shape[1]


def _asymmetry(theta):
    scale = float(np.max(np.abs(theta)))
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs(theta - np.transpose(theta, (0, 2, 1)))) / scale)


def static_modal_derivatives(K, directions, tangent_fn=None, dK=None, u0=None, step=None, length_scale=1.0,
                             rel_step=1e-6, symmetrize=True):
    """Static modal derivatives theta_ij = -K^-1 (dK/dq_j) phi_i for every pair of `directions`.

    Parameters
    ----------
    K : (n, n) stiffness at the reference state (dense or sparse, positive definite on the retained DOFs).
    directions : (n, m) columns phi_1 ... phi_m.
    tangent_fn : callable(u) -> K_T(u). Needed unless `dK` is given. Called 2 m times.
    dK : optional list of m derivative matrices dK/dq_j (dense or sparse), instead of `tangent_fn`.
    symmetrize : replace theta_ij and theta_ji by their mean (the exact static derivatives are symmetric; the
        measured asymmetry is reported in the result either way).

    K is factorised once; each derivative then costs one back-substitution.
    """
    Phi = _as_array(directions)
    n, m = Phi.shape
    if (tangent_fn is None) == (dK is None):
        raise ValueError("give exactly one of tangent_fn or dK")
    solve = _factorize(K)
    theta = np.zeros((n, m, m))
    evals = 0
    for j in range(m):
        if dK is not None:
            dKj = dK[j]
        else:
            dKj = directional_stiffness_derivative(tangent_fn, Phi[:, j], u0, step, length_scale, rel_step)
            evals += 2
        rhs = -_matmul(dKj, Phi)                 # column i = -(dK_j phi_i)
        theta[:, :, j] = solve(rhs)
    asym = _asymmetry(theta)
    if symmetrize:
        theta = 0.5 * (theta + np.transpose(theta, (0, 2, 1)))
    return ModalDerivatives(theta, Phi, asym, evals, "static")


def dynamic_modal_derivatives(K, M, omega2, modes, tangent_fn=None, dK=None, u0=None, step=None,
                              length_scale=1.0, rel_step=1e-6):
    """Dynamic modal derivatives from the bordered system

        [ K - w_i^2 M   M phi_i ] [theta_ij]   [ -(dK_j phi_i - dw_i^2 M phi_i) ]
        [ phi_i^T M        0    ] [   c    ] = [               0               ]

    (c comes out zero; the border only fixes the free multiple of phi_i). Requires `modes` to be
    M-orthonormal eigenvectors of (K, M) with DISTINCT eigenvalues `omega2`, and a constant mass matrix
    (dM/dq = 0). One bordered factorisation per mode i, then one solve per j.
    """
    Phi = _as_array(modes)
    omega2 = _as_array(omega2)
    n, m = Phi.shape
    if (tangent_fn is None) == (dK is None):
        raise ValueError("give exactly one of tangent_fn or dK")
    if np.min(np.abs(np.diff(np.sort(omega2)))) <= 1e-9 * max(1.0, float(np.max(np.abs(omega2)))):
        raise ValueError("repeated eigenvalues: dynamic modal derivatives are not defined for a repeated frequency")
    evals = 0
    dKs = []
    for j in range(m):
        if dK is not None:
            dKs.append(dK[j])
        else:
            dKs.append(directional_stiffness_derivative(tangent_fn, Phi[:, j], u0, step, length_scale, rel_step))
            evals += 2
    Mphi = _matmul(M, Phi)
    theta = np.zeros((n, m, m))
    for i in range(m):
        A = sp.bmat([[sp.csc_matrix(K) - omega2[i] * sp.csc_matrix(M), sp.csc_matrix(Mphi[:, [i]])],
                     [sp.csc_matrix(Mphi[:, [i]].T), None]], format="csc")
        solve = spla.splu(A).solve
        for j in range(m):
            dKphi = _matmul(dKs[j], Phi[:, [i]])[:, 0]
            domega2 = float(Phi[:, i] @ dKphi)
            rhs = np.concatenate([-(dKphi - domega2 * Mphi[:, i]), [0.0]])
            theta[:, i, j] = solve(rhs)[:n]
    return ModalDerivatives(theta, Phi, _asymmetry(theta), evals, "dynamic")


# -----------------------------------------------------------------------
# Selection and augmentation
# -----------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or np.trapz


def modal_coordinates(Phi, M, u_history):
    """q(t) = Phi^T M u(t) for a displacement history of shape (n_steps, n); returns (n_steps, m)."""
    Phi = _as_array(Phi)
    U = _as_array(u_history)
    return (U @ _matmul(M, Phi))


def modal_interaction_weights(q_history, dt=1.0):
    """w_ij = integral of |q_i(t) q_j(t)| dt (trapezoid rule) from a cheap LINEAR modal run under the load of
    interest. A large w_ij says that the nonlinearity coupling modes i and j will be excited, so the derivative
    theta_ij matters. Returns a symmetric (m, m) matrix."""
    Q = _as_array(q_history)
    m = Q.shape[1]
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            W[i, j] = W[j, i] = _trapz(np.abs(Q[:, i] * Q[:, j]), dx=dt)
    return W


def select_modal_derivatives(weights, n_select=None, fraction=None):
    """Rank the m(m+1)/2 derivatives (i <= j) by `weights` and return the chosen pairs, highest weight first.

    n_select : keep this many pairs. fraction : or keep the fewest pairs whose weights add up to at least this
    fraction of the total. Exactly one of the two is required.
    """
    W = _as_array(weights)
    m = W.shape[0]
    pairs = [(i, j) for i in range(m) for j in range(i, m)]
    w = np.array([W[i, j] for i, j in pairs])
    order = np.argsort(-w, kind="stable")
    if (n_select is None) == (fraction is None):
        raise ValueError("give exactly one of n_select or fraction")
    if n_select is None:
        cum = np.cumsum(w[order]) / max(float(w.sum()), 1e-300)
        n_select = int(np.searchsorted(cum, fraction - 1e-12) + 1)
    n_select = int(min(max(n_select, 0), len(pairs)))
    return [pairs[k] for k in order[:n_select]]


def augment_with_derivatives(modes, theta, pairs=None, M=None, tol=1e-8, max_vectors=None):
    """Augmented basis [Phi, Psi]: the chosen derivative vectors, made orthogonal to the modes and compressed.

    The vectors theta_ij (i, j in `pairs`, default all i <= j) are projected orthogonal to `modes` in the M inner
    product (Euclidean if M is None) and compressed by an eigen-decomposition of their Gram matrix: directions with
    Gram eigenvalue below `tol` times the largest are dropped, and the rest are scaled so that Psi is
    M-orthonormal. `max_vectors` caps the number kept (largest first).

    Returns (basis, info): basis = [Phi, Psi] with shape (n, m + r); info has `n_candidates`, `n_added`,
    `gram_eigenvalues` (descending) and the retained fraction `retained`.
    """
    Phi = _as_array(modes)
    th = theta.theta if isinstance(theta, ModalDerivatives) else _as_array(theta)
    n, m = Phi.shape
    if pairs is None:
        pairs = [(i, j) for i in range(m) for j in range(i, m)]
    X = np.column_stack([th[:, i, j] for i, j in pairs]) if pairs else np.zeros((n, 0))
    if X.shape[1] == 0:
        return Phi.copy(), dict(n_candidates=0, n_added=0, gram_eigenvalues=np.zeros(0), retained=1.0)
    Mm = (lambda B: B) if M is None else (lambda B: _matmul(M, B))
    X = X - Phi @ (Phi.T @ Mm(X))            # remove the component along the modes
    G = X.T @ Mm(X)
    G = 0.5 * (G + G.T)
    lam, U = eigh(G)
    lam, U = lam[::-1], U[:, ::-1]
    keep = lam > tol * max(float(lam[0]), 1e-300)
    if max_vectors is not None:
        keep &= np.arange(len(lam)) < int(max_vectors)
    Psi = X @ (U[:, keep] / np.sqrt(lam[keep]))
    retained = float(lam[keep].sum() / max(float(lam.sum()), 1e-300))
    return np.hstack([Phi, Psi]), dict(n_candidates=X.shape[1], n_added=int(keep.sum()), gram_eigenvalues=lam,
                                       retained=retained)

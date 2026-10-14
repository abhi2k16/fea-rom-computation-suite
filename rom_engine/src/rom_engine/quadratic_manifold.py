# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
quadratic_manifold.py -- a quadratic
manifold for geometrically nonlinear reduced-order models, its hyper-reduction with ECSW, and simulation-free
training. After Jain, Tiso, Rutzmoser, Rixen (2017), Rutzmoser, Rixen, Tiso, Jain (2017), Geelen, Wright and
Willcox (2023) and the summary in Tiso, Karamooz Mahdiabadi and Marconi (2021), Sections 4.5.4 and 4.6.

THE MAP. A linear basis restricts the state to a subspace, u = Phi q. A quadratic manifold keeps the same
coordinates q but lets the state curve away from the subspace,
        u = Gamma(q) = Phi q + 1/2 Omega(q, q),      Omega(q, q)_a = sum_jk Omega[a, j, k] q_j q_k,
with Omega symmetric in (j, k). The tangent space at q is spanned by the columns of
        P(q) = dGamma/dq = Phi + Omega q,      (Omega q)[a, j] = sum_k Omega[a, j, k] q_k.
Because the tangent changes with q, the reduced equations of motion (Lagrange's equations on the manifold) are
        M~(q) q'' + P^T M Omega(q', q') + P^T f_int(Gamma(q)) = P^T f_ext,        M~(q) = P^T M P,
that is, a configuration-dependent reduced mass and a convective (Coriolis-like) inertial force. For a STATIC
problem the reduced equations are  P(q)^T (f_int(Gamma(q)) - f_ext) = 0.

WAYS TO GET Omega (all return a `QuadraticManifold`)
  fit_quadratic_manifold      least squares (optionally ridge) fit of the part of snapshots that Phi misses, as a
                              quadratic form in the modal coordinates (Geelen et al. 2023).
  manifold_from_derivatives   Omega = theta, the modal derivatives of modal_derivatives.py (Jain et al. 2017):
                              no snapshots, no full-order solve, only stiffness derivatives.
  lifted_training_states      the SIMULATION-FREE training set: draw modal coordinates inside a box and lift them
                              through the manifold, so ECSW can be trained without running the full model.

HYPER-REDUCTION. ManifoldECSW selects and weights elements so that the TANGENT-SPACE-PROJECTED internal
force  sum_e w_e P(q)^T L_e^T f_e(L_e Gamma(q))  matches the full one over the training coordinates. Training
states are the lifted coordinates, never full-order trajectories.

WHAT IS NOT CLAIMED. The manifold is exact only to second order about the origin; accuracy beyond the training
range of q is not guaranteed and the quadratic form is only as good as the data (or derivatives) it comes from.
The reduced dynamics use an explicit RK4 integrator suited to testing and small problems, not a production
time integrator; the sparsity/cost benefit of ECSW is measured by the user, not asserted here.

References (methods in this module are adopted from):
  - S. Jain, P. Tiso, J. B. Rutzmoser, D. J. Rixen (2017), "A quadratic manifold for model order reduction of nonlinear structural dynamics", Comput. Struct. 188
  - J. B. Rutzmoser, D. J. Rixen, P. Tiso, S. Jain (2017), "Generalization of quadratic manifolds for reduced order modeling of nonlinear structural dynamics", Comput. Struct. 192
  - S. Geelen, S. Wright, K. Willcox (2023), "Operator inference for non-intrusive model reduction with quadratic manifolds", Comput. Methods Appl. Mech. Engrg. 403
  - P. Tiso, M. Karamooz Mahdiabadi, J. Marconi (2021), "Modal methods for reduced order modeling", Chapter 4, ETH Research Collection, doi:10.3929/ethz-b-000526692
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp
from scipy.optimize import root

from .hyper_reduction import ecsw_weights
from .modal_derivatives import ModalDerivatives, _as_array, _matmul


class QuadraticManifold:
    """u = Phi q + 1/2 Omega(q, q); Omega is (n, m, m) and symmetrised on construction."""

    def __init__(self, Phi, Omega, M=None):
        self.Phi = _as_array(Phi)
        Om = _as_array(Omega)
        n, m = self.Phi.shape
        if Om.shape != (n, m, m):
            raise ValueError(f"Omega must have shape {(n, m, m)}, got {Om.shape}")
        self.Omega = 0.5 * (Om + np.transpose(Om, (0, 2, 1)))
        self.M = M
        self.n, self.m = n, m

    def map(self, q):
        q = _as_array(q)
        return self.Phi @ q + 0.5 * np.einsum("ajk,j,k->a", self.Omega, q, q)

    def tangent(self, q):
        """P(q) = Phi + Omega q, shape (n, m)."""
        return self.Phi + np.einsum("ajk,k->aj", self.Omega, _as_array(q))

    def second_derivative(self, qd):
        """Omega(q', q') (n,), the convective acceleration of the lifted state."""
        qd = _as_array(qd)
        return np.einsum("ajk,j,k->a", self.Omega, qd, qd)

    def reduced_mass(self, q):
        P = self.tangent(q)
        return P.T @ _matmul(self.M, P) if self.M is not None else P.T @ P

    def convective_force(self, q, qd):
        P = self.tangent(q)
        w = self.second_derivative(qd)
        return P.T @ (_matmul(self.M, w.reshape(-1, 1))[:, 0] if self.M is not None else w)

    def project(self, u):
        """Linear coordinates of a state, q = Phi^T M u (Euclidean if M is None) -- the coordinate the manifold
        was built around, not a nearest-point projection onto the manifold."""
        u = _as_array(u)
        Mu = _matmul(self.M, u.reshape(-1, 1))[:, 0] if self.M is not None else u
        return self.Phi.T @ Mu

    def lift(self, Q):
        """Map columns of Q (m, k) to states (n, k)."""
        Q = _as_array(Q)
        return np.column_stack([self.map(Q[:, k]) for k in range(Q.shape[1])])


def fit_quadratic_manifold(Phi, snapshots, M=None, ridge=1e-8):
    """Least-squares Omega from snapshots (n, k): minimise sum_k || (u_k - Phi q_k) - 1/2 Omega(q_k, q_k) ||^2,
    with q_k = Phi^T M u_k (M-orthonormal Phi) and a ridge penalty `ridge` * scale on the coefficients.
    Returns (QuadraticManifold, info) with the relative `residual` of the fit and `captured` fraction."""
    Phi = _as_array(Phi)
    X = _as_array(snapshots)
    n, m = Phi.shape
    MX = _matmul(M, X) if M is not None else X
    Q = Phi.T @ MX                                        # (m, k)
    R = X - Phi @ Q                                       # what the linear subspace misses
    pairs = [(i, j) for i in range(m) for j in range(i, m)]
    F = np.column_stack([Q[i] * Q[j] for i, j in pairs])  # (k, n_pairs)
    scale = np.linalg.norm(F, 2) ** 2
    A = F.T @ F + ridge * scale * np.eye(len(pairs))
    C = np.linalg.solve(A, F.T @ R.T).T                   # (n, n_pairs): R ~ C F^T
    Om = np.zeros((n, m, m))
    for c, (i, j) in enumerate(pairs):
        if i == j:
            Om[:, i, i] = 2.0 * C[:, c]
        else:
            Om[:, i, j] = Om[:, j, i] = C[:, c]
    rn = np.linalg.norm(R)
    res = np.linalg.norm(R - C @ F.T) / rn if rn > 0 else 0.0
    return QuadraticManifold(Phi, Om, M), dict(residual=float(res), n_pairs=len(pairs), n_snapshots=X.shape[1])


def manifold_from_derivatives(Phi, theta, M=None):
    """Omega = modal derivatives (Jain et al. 2017). `theta` is a ModalDerivatives or an (n, m, m) array.
    Under this identification Gamma(q) = Phi q + 1/2 sum_ij theta_ij q_i q_j."""
    th = theta.theta if isinstance(theta, ModalDerivatives) else _as_array(theta)
    return QuadraticManifold(Phi, th, M)


def lifted_training_states(manifold, amplitudes, n_samples=60, seed=0):
    """SIMULATION-FREE training set: modal coordinates drawn uniformly in the box |q_i| <= amplitudes_i (plus the
    box corners along each axis, so the extremes are covered), lifted through the manifold. Returns (Q, U)."""
    rng = np.random.default_rng(seed)
    a = _as_array(amplitudes) * np.ones(manifold.m)
    cols = [np.diag(a), -np.diag(a)]
    cols.append(rng.uniform(-1.0, 1.0, (manifold.m, n_samples)) * a[:, None])
    Q = np.hstack(cols)
    return Q, manifold.lift(Q)


class ManifoldECSW:
    """ECSW weights on a quadratic manifold, trained on lifted coordinates (no full-order simulation).

    Fit with `ManifoldECSW.fit(manifold, Q, elem_dofs, elem_force_fn)`; `reduced_force(q)` then evaluates
    sum_e w_e P(q)^T L_e^T f_e(L_e Gamma(q)) over the selected elements only."""

    def __init__(self, manifold, elem_dofs, elem_force_fn, weights, info=None):
        self.manifold = manifold
        self.weights = _as_array(weights)
        self.elements = np.flatnonzero(self.weights > 0)
        self._dofs = [np.asarray(elem_dofs[e]) for e in self.elements]
        self._w = self.weights[self.elements]
        self.elem_force_fn = elem_force_fn
        self.n_total = len(elem_dofs)
        self.info = info or {}

    @classmethod
    def fit(cls, manifold, Q, elem_dofs, elem_force_fn, tol=1e-4, max_elements=None):
        Q = _as_array(Q)
        m, k = manifold.m, Q.shape[1]
        n_elem = len(elem_dofs)
        G = np.zeros((k * m, n_elem))
        for s in range(k):
            q = Q[:, s]
            u = manifold.map(q)
            P = manifold.tangent(q)
            for e in range(n_elem):
                g = np.asarray(elem_dofs[e])
                G[s * m:(s + 1) * m, e] = P[g, :].T @ _as_array(elem_force_fn(e, u[g]))
        w, info = ecsw_weights(G, G.sum(axis=1), tol=tol, max_elements=max_elements)
        info = dict(info, n_training_states=k, n_elements=n_elem)
        return cls(manifold, elem_dofs, elem_force_fn, w, info)

    @property
    def n_selected(self):
        return len(self.elements)

    def reduced_force(self, q):
        q = _as_array(q)
        u = self.manifold.map(q)
        P = self.manifold.tangent(q)
        f = np.zeros(self.manifold.m)
        for e, wv, g in zip(self.elements, self._w, self._dofs):
            f += wv * (P[g, :].T @ _as_array(self.elem_force_fn(int(e), u[g])))
        return f


class QuadraticManifoldROM:
    """Reduced model on a quadratic manifold. `f_int` is the full internal-force callable u -> f(u), or pass an
    object with `reduced_force(q)` (ManifoldECSW) as `hyper`."""

    def __init__(self, manifold, f_int=None, hyper=None):
        if (f_int is None) == (hyper is None):
            raise ValueError("give exactly one of f_int or hyper")
        self.man, self.f_int, self.hyper = manifold, f_int, hyper

    def reduced_force(self, q):
        if self.hyper is not None:
            return self.hyper.reduced_force(q)
        return self.man.tangent(q).T @ _as_array(self.f_int(self.man.map(q)))

    def static_solve(self, f_ext, q0=None, tol=1e-12):
        """Solve P(q)^T (f_int(Gamma(q)) - f_ext) = 0 for q. `f_ext` is the full load vector."""
        f_ext = _as_array(f_ext)
        q0 = np.zeros(self.man.m) if q0 is None else q0
        res = lambda q: self.reduced_force(q) - self.man.tangent(q).T @ f_ext
        sol = root(res, q0, method="hybr", tol=tol)
        if not sol.success:
            raise RuntimeError(f"manifold static solve did not converge: {sol.message}")
        return sol.x

    def simulate(self, f_ext_fn, q0, qd0, dt, n_steps, damping=None):
        """Explicit RK4 for  M~(q) q'' = P^T f_ext(t) - P^T M Omega(q',q') - reduced_force(q) - C q'.
        Returns (t, Q, Qd). For tests and small problems; mind the stability limit of RK4."""
        m = self.man.m
        C = np.zeros((m, m)) if damping is None else _as_array(damping)

        def acc(t, q, qd):
            P = self.man.tangent(q)
            rhs = P.T @ _as_array(f_ext_fn(t)) - self.man.convective_force(q, qd) - self.reduced_force(q) - C @ qd
            return np.linalg.solve(self.man.reduced_mass(q), rhs)

        def rhs_fn(t, y):
            return np.concatenate([y[m:], acc(t, y[:m], y[m:])])

        y = np.concatenate([_as_array(q0), _as_array(qd0)])
        Y = [y]
        for k in range(n_steps):
            t = k * dt
            k1 = rhs_fn(t, y)
            k2 = rhs_fn(t + dt / 2, y + dt / 2 * k1)
            k3 = rhs_fn(t + dt / 2, y + dt / 2 * k2)
            k4 = rhs_fn(t + dt, y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Y.append(y)
        Y = np.array(Y)
        return np.arange(n_steps + 1) * dt, Y[:, :m], Y[:, m:]

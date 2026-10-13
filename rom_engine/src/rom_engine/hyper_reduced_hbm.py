# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
hyper_reduced_hbm.py -- ECSW hyper-reduction of the interface elements of a jointed structure in harmonic balance,
and the residual-based error indicator that needs no full-order solution. After Morsy, Kast and Tiso (2022,
arXiv:2204.12160), Sections 3.3-3.5, whose training and indicator are used as published:

- TRAINING WITHOUT TIME INTEGRATION (their Eqs. 45-48): the training snapshots are the harmonic components of the
  nonlinear force that the alternating frequency-time step already produces for a reduced orbit Q, so no
  high-fidelity simulation is needed. For orbit Q and joint element e, the columns of the training matrix are
  g_{k,e} = (W_e)^T [F_e]_k for the 2H+1 harmonic components k (mean, cos_h, sin_h), the target is their sum over e,
  and weights xi_e >= 0 are found by sparse NNLS (`hyper_reduction.ecsw_weights`, Lawson-Hanson with early stop at
  ||G xi - b|| <= tau ||b||). The paper reports tau = 0.01 giving sparse, accurate hyper meshes; that is the default.
- ERROR INDICATOR (their Eqs. 49-50): evaluate the residual of the FULL equations of motion at the displacement
  reconstructed from the reduced solution, r = M u'' + C u' + K u + f(u) - p, and report the RMS of its component
  at the excitation DOF divided by the RMS of the excitation there. The paper finds errors below 15 % indicate a
  hyper-reduced solution that captures the amplitude-dependent stiffness and damping in their two case studies;
  that threshold is theirs, not validated here beyond the tests in this repository.

Joint elements are given as (dofs, model) pairs: `dofs` the global DOFs of the element and `model` an object with
`force(q_local_time)` / `jacobian(q_local_time)` over one period (e.g. a one-DOF `JenkinsContact`).

Scope, stated honestly: the weights are trained at one excitation frequency/level (or a few orbits) and reused
along the sweep, as in the paper; the basis is supplied by the caller (the paper's fictitious-interface vibration
modes are not built here); only the joint force is hyper-reduced, the linear part is exact.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .hyper_reduction import ecsw_weights

__all__ = ["JointElementSet", "harmonic_training_matrix", "fit_harmonic_ecsw", "HyperReducedJointForce",
           "residual_error_indicator"]


class JointElementSet:
    """A list of joint elements with global DOFs. `dofs[e]` index arrays, `models[e]` force models."""

    def __init__(self, n_dof, elements):
        self.n = int(n_dof)
        self.dofs = [np.atleast_1d(np.asarray(d, int)) for d, _ in elements]
        self.models = [m for _, m in elements]

    def __len__(self):
        return len(self.models)


def _local(q_time, dofs, W=None):
    """Element-local displacement histories: from full-space samples (NT, n), or from reduced samples (NT, r) with
    the basis W (n, r)."""
    return q_time[:, dofs] if W is None else q_time @ W[dofs].T


def harmonic_training_matrix(hbm, joints, W, orbits):
    """(G, b) of the harmonic ECSW training. hbm is the REDUCED `ForcedHBM` (coefficient layout of the orbits);
    `orbits` an iterable of reduced coefficient arrays Q of shape (NC, r). Rows: for each orbit, each of the 2H+1
    harmonic components, r entries."""
    W = np.asarray(W, float)
    r = W.shape[1]
    blocks = []
    for Q in orbits:
        Q = np.asarray(Q, float).reshape(hbm.NC, r)
        q_time = hbm.T @ Q                                  # (NT, r) reduced displacement samples
        cols = []
        for dofs, model in zip(joints.dofs, joints.models):
            f_loc = model.force(_local(q_time, dofs, W))    # (NT, nloc)
            F_loc = hbm.Ti @ f_loc                          # (NC, nloc) harmonic components of the element force
            cols.append((F_loc @ W[dofs]).reshape(-1))      # (NC * r): rows = harmonic components x reduced index
        blocks.append(np.array(cols).T)
    G = np.vstack(blocks)
    return G, G.sum(axis=1)


def fit_harmonic_ecsw(hbm, joints, W, orbits, tau=0.01, max_elements=None):
    """Sparse non-negative weights for the joint elements. Returns (weights (n_elements,), info) where info has
    'residual' (relative training error), 'n_selected' and 'selected' (indices)."""
    G, b = harmonic_training_matrix(hbm, joints, W, orbits)
    xi, info = ecsw_weights(G, b, tol=tau, max_elements=max_elements)
    info = dict(info)
    info["selected"] = np.flatnonzero(xi)
    return xi, info


class HyperReducedJointForce:
    """Reduced joint force for `ForcedHBM` on the reduced coordinates q (n = r): sum_e xi_e W_e^T f_e(W_e q).
    With `weights=None` every element has weight one (the unreduced reduced model); with ECSW weights only the
    elements with nonzero weight are evaluated. Provides `force(q_time)` and the exact `jacobian(q_time)`."""

    def __init__(self, joints, W, weights=None):
        self.joints = joints
        self.W = np.asarray(W, float)
        self.xi = np.ones(len(joints)) if weights is None else np.asarray(weights, float)
        self.active = np.flatnonzero(self.xi)

    def force(self, q_time):
        q_time = np.asarray(q_time, float)
        out = np.zeros_like(q_time)
        for e in self.active:
            dofs, model = self.joints.dofs[e], self.joints.models[e]
            f = model.force(_local(q_time, dofs, self.W))          # (NT, nloc)
            out += self.xi[e] * (f @ self.W[dofs])
        return out

    def jacobian(self, q_time):
        q_time = np.asarray(q_time, float)
        NT, r = q_time.shape
        J = np.zeros((NT * r, NT * r))
        for e in self.active:
            dofs, model = self.joints.dofs[e], self.joints.models[e]
            nloc = len(dofs)
            Jl = model.jacobian(_local(q_time, dofs, self.W))      # (NT*nloc, NT*nloc)
            L = np.kron(np.eye(NT), self.W[dofs])                  # (NT*nloc, NT*r): reduced -> local samples
            J += self.xi[e] * (L.T @ Jl @ L)
        return J


class _FullJointForce:
    """Full-space joint force with the same force()/jacobian() protocol, for the unreduced model and the indicator."""

    def __init__(self, joints):
        self.joints = joints

    def force(self, q_time):
        q_time = np.asarray(q_time, float)
        out = np.zeros_like(q_time)
        for dofs, model in zip(self.joints.dofs, self.joints.models):
            out[:, dofs] += model.force(q_time[:, dofs])
        return out

    def jacobian(self, q_time):
        q_time = np.asarray(q_time, float)
        NT, n = q_time.shape
        J = np.zeros((NT * n, NT * n))
        for dofs, model in zip(self.joints.dofs, self.joints.models):
            nloc = len(dofs)
            Jl = model.jacobian(q_time[:, dofs])
            rows = (np.arange(NT)[:, None] * n + dofs[None, :]).reshape(-1)
            J[np.ix_(rows, rows)] += Jl
        return J


def residual_error_indicator(hbm_full, joints, W, Q, omega, F_full, excitation_dof):
    """Error indicator E of Eq. 50 for a reduced orbit Q (NC, r) at excitation frequency `omega`: RMS over the period
    of the full-model residual at `excitation_dof` divided by the RMS of the excitation there. `hbm_full` is the
    FULL-order `ForcedHBM` (same harmonics and time grid as the reduced one), `F_full` its excitation coefficients
    (NC, n). No full-order solve is made: the residual is evaluated at u = W q only."""
    Qa = np.asarray(Q, float).reshape(hbm_full.NC, -1)
    Z = Qa @ np.asarray(W, float).T                                 # (NC, n) full-space coefficients
    z = Z.reshape(-1)
    R = hbm_full.residual(z, omega, np.asarray(F_full, float), _FullJointForce(joints))   # (NC*n,) coefficients
    R = R.reshape(hbm_full.NC, hbm_full.n)
    r_t = hbm_full.T @ R[:, excitation_dof]
    p_t = hbm_full.T @ np.asarray(F_full, float)[:, excitation_dof]
    return float(np.sqrt(np.mean(r_t ** 2)) / np.sqrt(np.mean(p_t ** 2)))

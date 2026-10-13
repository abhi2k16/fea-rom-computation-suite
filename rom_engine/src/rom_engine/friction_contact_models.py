# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
friction_contact_models.py -- dry-friction contact elements for harmonic balance (time-domain evaluation).

`JenkinsContact` is a set of one-dimensional macro-slip contact elements in the form used in the alternating
frequency-time (AFT) method. Element e couples a tangential DOF v_t and, optionally, a normal DOF v_n:

    normal load      N   = max(N0 + k_n v_n, 0)               (lift-off when the spring unloads to zero)
    tangential force f_t = k_t (v_t - w),   |f_t| <= mu N     (stick: w fixed; slip: w moves so |f_t| = mu N)

The slider position w carries the history, so one period of motion is run twice from w = 0 and the SECOND cycle
is returned (the periodic state, standard for Jenkins elements in AFT). The element force acts on the structure as
f_t on the tangential DOF and (N - N0) on the normal DOF (so zero normal motion gives zero normal force; the
preload N0 is not part of the dynamic force). With no normal DOF the normal load is the constant N0.

`force(q_time)` takes samples q_time of shape (NT, n) on a uniform grid over ONE period and returns (NT, n).
`jacobian(q_time)` returns the exact derivative d f_time / d q_time as a dense (NT*n, NT*n) matrix (index
t*n + dof), obtained by differentiating the stick/slip/lift-off recursion, and is checked against finite
differences in the tests. At the switching points (stick-slip, contact-separation) the force is only
piecewise smooth; the Jacobian there is one of the one-sided derivatives.

Scope: 1-D elements with constant stiffnesses, no Coulomb-friction regularisation, normal-load variation through a
linear spring, and no 2-D tangential motion (the 2-D macro-slip element is not implemented).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["JenkinsContact", "jenkins_hysteresis_energy"]


def _run_element(vt, vn, kt, mu, N0, kn, with_derivatives):
    """Two-cycle recursion for one element. vt, vn: (NT,) samples (vn may be None). Returns (f_t, N, dft_dvt,
    dft_dvn, dN_dvn) with the derivative matrices (NT, NT) (None when not requested)."""
    NT = len(vt)
    have_n = vn is not None
    w = 0.0
    dw_t = np.zeros(NT)
    dw_n = np.zeros(NT)
    ft = np.zeros(NT)
    Nl = np.zeros(NT)
    J_t = np.zeros((NT, NT)) if with_derivatives else None
    J_n = np.zeros((NT, NT)) if with_derivatives else None
    JN = np.zeros((NT, NT)) if with_derivatives else None
    for cycle in range(2):
        for i in range(NT):
            Nraw = N0 + (kn * vn[i] if have_n else 0.0)
            active = Nraw > 0.0
            N = Nraw if active else 0.0
            lim = mu * N
            trial = kt * (vt[i] - w)
            if with_derivatives:
                dtr_t = -kt * dw_t
                dtr_t[i] += kt
                dtr_n = -kt * dw_n
            if abs(trial) <= lim:
                f = trial
                df_t = dtr_t if with_derivatives else None
                df_n = dtr_n if with_derivatives else None
            else:
                s = 1.0 if trial > 0 else -1.0
                f = s * lim
                w = vt[i] - f / kt
                if with_derivatives:
                    df_t = np.zeros(NT)
                    df_n = np.zeros(NT)
                    if have_n and active:
                        df_n[i] = s * mu * kn
                    dw_t = -df_t / kt
                    dw_t[i] += 1.0
                    dw_n = -df_n / kt
            if cycle == 1:
                ft[i], Nl[i] = f, N
                if with_derivatives:
                    J_t[i], J_n[i] = df_t, df_n
                    if have_n and active:
                        JN[i, i] = kn
    return ft, Nl, J_t, J_n, JN


class JenkinsContact:
    """A set of Jenkins macro-slip contact elements acting on an n-DOF reduced (or physical) model.

    Parameters
    ----------
    n_dof : number of DOFs of the model the force is added to.
    tangential_dofs : sequence of DOF indices, one per element.
    k_t, mu, N0 : scalars or per-element arrays (tangential stiffness, friction coefficient, normal preload).
    normal_dofs : optional sequence (same length) of normal DOF indices, `None` entries allowed; k_n : normal
        stiffness (scalar or array), used only for elements that have a normal DOF.
    """

    def __init__(self, n_dof, tangential_dofs, k_t, mu, N0, normal_dofs=None, k_n=0.0):
        self.n = int(n_dof)
        self.t_dofs = np.asarray(tangential_dofs, int)
        ne = len(self.t_dofs)
        bc = lambda x: np.broadcast_to(np.asarray(x, float), (ne,)).copy()
        self.kt, self.mu, self.N0, self.kn = bc(k_t), bc(mu), bc(N0), bc(k_n)
        self.n_dofs = [None] * ne if normal_dofs is None else [None if d is None else int(d) for d in normal_dofs]
        if len(self.n_dofs) != ne:
            raise ValueError("normal_dofs must have one entry per element")
        if np.any(self.kt <= 0) or np.any(self.mu < 0) or np.any(self.N0 < 0):
            raise ValueError("k_t must be positive and mu, N0 non-negative")

    def _elements(self, q_time, derivatives):
        q_time = np.asarray(q_time, float)
        for e in range(len(self.t_dofs)):
            vt = q_time[:, self.t_dofs[e]]
            vn = None if self.n_dofs[e] is None else q_time[:, self.n_dofs[e]]
            yield e, _run_element(vt, vn, self.kt[e], self.mu[e], self.N0[e], self.kn[e], derivatives)

    def force(self, q_time):
        q_time = np.asarray(q_time, float)
        out = np.zeros_like(q_time)
        for e, (ft, Nl, *_rest) in self._elements(q_time, False):
            out[:, self.t_dofs[e]] += ft
            if self.n_dofs[e] is not None:
                out[:, self.n_dofs[e]] += Nl - self.N0[e]
        return out

    def jacobian(self, q_time):
        q_time = np.asarray(q_time, float)
        NT, n = q_time.shape
        J = np.zeros((NT * n, NT * n))
        for e, (ft, Nl, Jt, Jn, JN) in self._elements(q_time, True):
            rows_t = np.arange(NT) * n + self.t_dofs[e]
            cols_t = rows_t
            J[np.ix_(rows_t, cols_t)] += Jt
            if self.n_dofs[e] is not None:
                rows_n = np.arange(NT) * n + self.n_dofs[e]
                J[np.ix_(rows_t, rows_n)] += Jn
                J[np.ix_(rows_n, rows_n)] += JN
        return J

    def tangential_force_and_load(self, q_time, element=0):
        """(f_t, N) time histories of one element, for hysteresis plots and energy checks."""
        for e, (ft, Nl, *_rest) in self._elements(q_time, False):
            if e == element:
                return ft, Nl
        raise IndexError(element)


def jenkins_hysteresis_energy(vt, ft):
    """Energy dissipated per period, the area of the closed (v_t, f_t) loop, by the trapezoid rule over the periodic
    sample sequence."""
    vt, ft = np.asarray(vt, float), np.asarray(ft, float)
    return float(np.sum(0.5 * (ft + np.roll(ft, -1)) * (np.roll(vt, -1) - vt)))

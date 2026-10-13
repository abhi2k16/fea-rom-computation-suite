# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
domain_partition.py -- linear / nonlinear domain partitioning with static condensation of the linear zone.

In an assembly the nonlinearity (contact, large rotation) is usually confined to a small region. Everything
else can be treated as linear and condensed onto the interface B it shares with the nonlinear zone (Nikiema et al.
2026, Section 5):

    K* = K_BB - K_BI K_II^-1 K_IB,        f*_B = f_B - K_BI K_II^-1 f_I,

so the nonlinear problem lives on the nonlinear zone's DOFs only,

    R_N(u_N) = f_int^NL(u_N) + E_B^T K* E_B u_N - f*_ext,N = 0     (see `PartitionedModel.residual`),

and the linear interior is recovered afterwards by u_I = K_II^-1 (f_I - K_IB u_B). Any hyper-reduction trained on this
model sums over the nonlinear elements only, which shortens the offline stage (fewer columns in the training matrix).

`nonlinearity_indicator` chooses the nonlinear zone. The indicator of the source paper is not published (withheld for
confidentiality), so this one is our own design: for element e and state u,
    kappa_e = max over states of || f_e(u_e) - K0_e u_e || / || K0_e u_e ||,
the size of the element's nonlinear force relative to its linear force (K0_e is the element tangent at zero). Elements
above `threshold`, and any element touching a potential contact DOF, go to the nonlinear zone.

Limit, as in the source: if the nonlinear zone is large the interface B is large and K* is a dense B x B matrix; the
method pays off only when the nonlinear zone is small.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["nonlinearity_indicator", "select_nonlinear_elements", "PartitionedModel"]


def nonlinearity_indicator(elem_dofs, elem_force_fn, elem_tangent_fn, states):
    """kappa_e (see module docstring) for every element. `states` is (n_dof, n_states) of representative full states.
    Elements whose linear force vanishes at every state get kappa = 0."""
    S = np.asarray(states, float)
    n_elem = len(elem_dofs)
    kappa = np.zeros(n_elem)
    for e in range(n_elem):
        g = np.asarray(elem_dofs[e])
        K0 = np.atleast_2d(np.asarray(elem_tangent_fn(e, np.zeros(len(g))), float))
        for k in range(S.shape[1]):
            ul = S[g, k]
            lin = K0 @ ul
            nl = np.asarray(elem_force_fn(e, ul), float) - lin
            den = np.linalg.norm(lin)
            if den > 1e-300:
                kappa[e] = max(kappa[e], np.linalg.norm(nl) / den)
    return kappa


def select_nonlinear_elements(kappa, threshold, always=()):
    """Indices of elements with kappa > threshold, plus the `always` set (for example elements at contact DOFs)."""
    sel = set(np.flatnonzero(np.asarray(kappa) > threshold).tolist())
    sel.update(int(e) for e in always)
    return np.array(sorted(sel), int)


class PartitionedModel:
    """Nonlinear zone + condensed linear zone.

    elem_dofs: list of integer arrays (global DOFs per element); nonlinear_elements: indices of the nonlinear zone;
    K_linear: (n_dof x n_dof) stiffness of the LINEAR elements only (sparse or dense; the nonlinear elements must not
    contribute to it); fixed_dofs: constrained DOFs, removed from everything. Attributes: nl_dofs (the nonlinear
    zone's DOFs including the interface), interface (B), interior (I), K_star ((|B| x |B|) dense)."""

    def __init__(self, elem_dofs, nonlinear_elements, K_linear, n_dof, fixed_dofs=()):
        nl_el = set(int(e) for e in nonlinear_elements)
        all_nl = set()
        all_lin = set()
        for e, g in enumerate(elem_dofs):
            (all_nl if e in nl_el else all_lin).update(int(d) for d in g)
        fixed = set(int(d) for d in fixed_dofs)
        self.n_dof = int(n_dof)
        self.nl_dofs = np.array(sorted(all_nl - fixed), int)
        self.interface = np.array(sorted((all_nl & all_lin) - fixed), int)
        self.interior = np.array(sorted(all_lin - all_nl - fixed), int)
        self._pos = {d: i for i, d in enumerate(self.nl_dofs)}
        self.iface_local = np.array([self._pos[d] for d in self.interface], int)
        K = sp.csc_matrix(K_linear)
        B, I = self.interface, self.interior
        self.K_BB = K[B][:, B].toarray() if B.size else np.zeros((0, 0))
        if I.size:
            self._lu = spla.splu(sp.csc_matrix(K[I][:, I]))
            self.K_BI = K[B][:, I].toarray()
            self.K_IB = K[I][:, B].toarray()
            self.K_star = self.K_BB - self.K_BI @ self._lu.solve(self.K_IB)
        else:
            self._lu, self.K_BI, self.K_IB, self.K_star = None, np.zeros((B.size, 0)), np.zeros((0, B.size)), self.K_BB
        self.n_nl = len(self.nl_dofs)

    def condensed_load(self, f_full):
        """The external load on the nonlinear zone, (n_nl,), with the loads applied to the linear interior condensed onto
        the interface: f*_B = f_B - K_BI K_II^-1 f_I."""
        f = np.asarray(f_full, float)
        fN = f[self.nl_dofs].copy()
        if self.interior.size:
            corr = -self.K_BI @ self._lu.solve(f[self.interior])
            fN[self.iface_local] += corr
        return fN

    def residual(self, u_N, f_int_NL, f_ext_N):
        """R_N(u_N) = f_int^NL(u_N) + E_B^T K* E_B u_N - f_ext_N, with `f_int_NL` the nonlinear elements' internal force
        scattered onto the nonlinear zone's DOFs (an (n_nl,) vector) and `f_ext_N = condensed_load(...)`."""
        R = np.asarray(f_int_NL, float) - np.asarray(f_ext_N, float)
        if self.interface.size:
            R[self.iface_local] += self.K_star @ np.asarray(u_N, float)[self.iface_local]
        return R

    def tangent(self, Kt_NL):
        """Tangent of the partitioned residual: Kt_NL (the nonlinear elements' tangent on the zone's DOFs) plus K* at
        the interface."""
        K = np.array(Kt_NL, float, copy=True)
        if self.interface.size:
            K[np.ix_(self.iface_local, self.iface_local)] += self.K_star
        return K

    def expand(self, u_N, f_full=None):
        """Full displacement vector from the nonlinear-zone solution: interior by u_I = K_II^-1 (f_I - K_IB u_B)."""
        u = np.zeros(self.n_dof)
        u[self.nl_dofs] = u_N
        if self.interior.size:
            fI = np.zeros(self.interior.size) if f_full is None else np.asarray(f_full, float)[self.interior]
            u[self.interior] = self._lu.solve(fI - self.K_IB @ np.asarray(u_N, float)[self.iface_local])
        return u

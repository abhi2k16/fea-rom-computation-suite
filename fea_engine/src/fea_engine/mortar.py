# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
mortar.py -- dual mortar coupling of non-matching 2-D meshes along a straight interface.

Two bodies are meshed independently, so the nodes on their common interface do not coincide. The mortar
method couples them weakly. With the DUAL Lagrange-multiplier shape functions of Wohlmuth (biorthogonal to the
slave displacement shape functions) the mortar matrix D is diagonal, so the multipliers and the slave
interface DOFs can be eliminated exactly (Ritzert et al. 2025, Section 2.1.3):

    u_slave_interface = P u_master_interface,        P = D^-1 M,

with D_jj = integral of N_j^slave and M_jk = integral of the dual function (phi_j) times N_k^master over the
slave side. For linear 2-node segments the dual functions are phi_1 = 2 N_1 - N_2 and phi_2 = -N_1 + 2 N_2.

Provided here:

- `dual_mortar_operator`  the nodal coupling operator P (and the D, M it comes from) for two ordered node lists;
- `MortarTie2D`           the tied-contact transformation u = T u_independent for a whole model, with `condense`
                          and `expand` (the condensed system is symmetric positive definite when K is);
- `MortarTie2D.normal_gap_constraints`  the weighted normal-gap map B and offset c of unilateral contact on the
                          same interface, in the form `contact_solvers.solve_contact_static` takes.

Scope, stated honestly: straight interfaces (the master and slave node lists are projected onto the slave's
tangent), linear 2-node interface segments, small displacements (fixed normal and a constant P), full overlap
(every slave segment is covered by master segments, otherwise a ValueError). Curved interfaces, quadratic
segments, 3-D surfaces and large sliding are out of scope.

References (methods in this module are adopted from):
  - S. Ritzert, J. Kehls, S. Reese, T. Brepols (2025), "Component-Based Model-Order Reduction With Mortar Tied Contact for Nonlinear Quasi-Static Mechanical Problems", Int. J. Numer. Meth. Engng
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp

__all__ = ["dual_mortar_operator", "MortarTie2D"]

_GP = np.polynomial.legendre.leggauss(3)


def _arclength(coords, t, x0):
    return (np.asarray(coords, float) - x0) @ t


def dual_mortar_operator(slave_coords, master_coords, tol=1e-9):
    """Dual mortar operator for two node lists on a common straight line.

    slave_coords (n_s, 2), master_coords (n_m, 2): interface node coordinates, in any order. Returns
    (P, D, M): P (n_s, n_m) with u_s = P u_m, D (n_s,) the diagonal of the slave mortar matrix, M (n_s, n_m).
    Rows of P sum to one and P reproduces affine fields exactly (tested). The two sides may be parallel lines at a distance (an initial gap). Raises ValueError if the master does
    not cover the slave interface or either side is not straight within `tol`."""
    Xs = np.asarray(slave_coords, float)
    Xm = np.asarray(master_coords, float)
    if len(Xs) < 2 or len(Xm) < 2:
        raise ValueError("each interface needs at least two nodes")
    order_s = np.argsort(Xs @ (Xs[-1] - Xs[0]))
    t = Xs[order_s[-1]] - Xs[order_s[0]]
    length = np.linalg.norm(t)
    if length < tol:
        raise ValueError("degenerate slave interface")
    t = t / length
    x0 = Xs[order_s[0]]
    nrm = np.array([-t[1], t[0]])
    off = np.concatenate([(Xs - x0) @ nrm, (Xm - Xm[0]) @ nrm])    # each side straight; the two may be offset (a gap)
    if np.max(np.abs(off)) > max(tol, 1e-6 * length):
        raise ValueError("interface nodes are not collinear (curved interfaces are out of scope)")
    ss, sm = _arclength(Xs, t, x0), _arclength(Xm, t, x0)      # parallel lines: project along the normal
    os_, om = np.argsort(ss), np.argsort(sm)
    s_sorted, m_sorted = ss[os_], sm[om]
    if m_sorted[0] > s_sorted[0] + tol or m_sorted[-1] < s_sorted[-1] - tol:
        raise ValueError("the master interface does not cover the slave interface")
    ns, nm = len(Xs), len(Xm)
    D = np.zeros(ns)
    M = np.zeros((ns, nm))
    for e in range(ns - 1):
        a, b = s_sorted[e], s_sorted[e + 1]
        ell = b - a
        if ell < tol:
            continue
        D[os_[e]] += ell / 2
        D[os_[e + 1]] += ell / 2
        for k in range(nm - 1):
            lo, hi = max(a, m_sorted[k]), min(b, m_sorted[k + 1])
            if hi - lo < tol:
                continue
            ellm = m_sorted[k + 1] - m_sorted[k]
            s = 0.5 * (hi - lo) * _GP[0] + 0.5 * (hi + lo)
            w = 0.5 * (hi - lo) * _GP[1]
            xi = (s - a) / ell
            xim = (s - m_sorted[k]) / ellm
            N1, N2 = 1 - xi, xi
            phi = (2 * N1 - N2, -N1 + 2 * N2)
            Nm = (1 - xim, xim)
            for p in range(2):
                for q in range(2):
                    M[os_[e + p], om[k + q]] += np.sum(w * phi[p] * Nm[q])
    return M / D[:, None], D, M


class MortarTie2D:
    """Tied mortar coupling of a slave interface to a master interface inside a model with `n_dof_total` DOFs.

    slave_nodes, master_nodes: global node indices of the interface nodes (any order); node_coords (n_nodes, 2)
    reference coordinates; dofs_per_node (default 2). The model DOF of node i, component c is
    i * dofs_per_node + c. Every component of the slave interface is tied to the same component of the master
    (global axes), the standard displacement-continuity condition of tied contact."""

    def __init__(self, slave_nodes, master_nodes, node_coords, n_dof_total, dofs_per_node=2, tol=1e-9):
        self.slave_nodes = np.asarray(slave_nodes, int)
        self.master_nodes = np.asarray(master_nodes, int)
        self.dpn = int(dofs_per_node)
        self.n_dof = int(n_dof_total)
        X = np.asarray(node_coords, float)
        self.P, self.D, self.M = dual_mortar_operator(X[self.slave_nodes], X[self.master_nodes], tol)
        self._X = X
        self.slave_dofs = np.array([i * self.dpn + c for i in self.slave_nodes for c in range(self.dpn)])
        self.independent = np.setdiff1d(np.arange(self.n_dof), self.slave_dofs)
        pos = {d: k for k, d in enumerate(self.independent)}
        rows, cols, vals = [], [], []
        for d in self.independent:
            rows.append(d)
            cols.append(pos[d])
            vals.append(1.0)
        for js, i in enumerate(self.slave_nodes):
            for c in range(self.dpn):
                r = i * self.dpn + c
                for km, m in enumerate(self.master_nodes):
                    if self.P[js, km] != 0.0:
                        rows.append(r)
                        cols.append(pos[m * self.dpn + c])
                        vals.append(self.P[js, km])
        self._T = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_dof, len(self.independent)))

    def transformation(self):
        """Sparse T with u = T u_independent (independent = every DOF except the slave interface DOFs)."""
        return self._T

    def condense(self, K, f):
        """(T^T K T, T^T f): the condensed stiffness (sparse) and load. Symmetric positive (semi)definite when K is."""
        T = self._T
        Kc = T.T @ (sp.csr_matrix(K) if not sp.issparse(K) else K) @ T
        return Kc, T.T @ np.asarray(f, float)

    def expand(self, u_independent):
        return self._T @ np.asarray(u_independent, float)

    def tie_residual(self, u):
        """Weak tie violation u_slave - P u_master on the interface (zero for any u in the range of T)."""
        u = np.asarray(u, float)
        out = []
        for c in range(self.dpn):
            us = u[self.slave_nodes * self.dpn + c]
            um = u[self.master_nodes * self.dpn + c]
            out.append(us - self.P @ um)
        return np.stack(out, axis=1)

    def normal_gap_constraints(self, normal, initial_gap):
        """Weighted normal-gap constraints of UNILATERAL contact on this interface, in the form
        `contact_solvers.solve_contact_static(K, f, B, c)` takes (B u >= c, K u = f + B^T lam, lam >= 0).

        `normal` is the unit vector pointing from the slave body toward the master body. The weighted normal gap
        at slave node j is g_j = g0_j + n . (sum_k P_jk u_m,k - u_s,j) and must stay >= 0, with `initial_gap`
        (n_s,) the initial gaps g0 >= 0. That is B u >= c with B = n . (P e_master - e_slave) and c = -g0.
        Returns (B, c) with B of shape (n_s, n_dof). The multipliers lam are the contact forces; they push the
        slave node along -n and the master nodes along +n P^T lam."""
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        g0 = np.asarray(initial_gap, float)
        if g0.shape != (len(self.slave_nodes),):
            raise ValueError("initial_gap must have one entry per slave node")
        B = np.zeros((len(self.slave_nodes), self.n_dof))
        for j, i in enumerate(self.slave_nodes):
            for c in range(self.dpn):
                B[j, i * self.dpn + c] -= n[c]
                for k, m in enumerate(self.master_nodes):
                    B[j, m * self.dpn + c] += n[c] * self.P[j, k]
        return B, -g0

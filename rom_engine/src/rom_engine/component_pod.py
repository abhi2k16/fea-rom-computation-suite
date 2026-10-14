# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
component_pod.py -- component-wise POD reduction of non-matching assemblies coupled through a mortar operator.

Each component is reduced on its own from snapshots computed with PRESCRIBED interface displacements: the
interface data is parametrised by a few smooth functions (translation, rotation, polynomial modes) whose
coefficients are sampled with an optimal Latin hypercube, so a component basis does not depend on the neighbour it
will meet, and the same basis can be reused in several assemblies (after Ritzert et al. 2025).

In an assembly of a MASTER component a and a SLAVE component b, tied through the mortar operator P (slave
interface = P * master interface, as `fea_engine.mortar` produces; this module only receives the array):

    u_a = V_a q_a
    u_b,interior = Phi_b q_b          u_b,interface = P (V_a q_a)[interface_a]     (slave interface UNREDUCED
                                                                                  as an independent field: it is
                                                                                  eliminated through P)

so the reduced unknowns are x = [q_a; q_b] and u = T x. The reduced Newton system is T^T R(T x) = 0 with
tangent T^T K_t(T x) T. As in the source, the component force is still evaluated on the FULL component meshes (no
hyper-reduction in this first version), so the speed-up is in the linear algebra only.

Scope: the interface is whatever `P` couples (the caller builds it); small displacements through a constant P; two
components per call (chain more by repeating the pattern).

References (methods in this module are adopted from):
  - S. Ritzert, J. Kehls, S. Reese, T. Brepols (2025), "Component-Based Model-Order Reduction With Mortar Tied Contact for Nonlinear Quasi-Static Mechanical Problems", Int. J. Numer. Meth. Engng
  - M. Balajewicz, D. Amsallem, C. Farhat (2015), "Projection-based model reduction for contact problems", arXiv:1503.01000 (Int. J. Numer. Meth. Engng, 2016)
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .sampling import optimal_lhs

__all__ = ["legendre_interface_parametrisation", "component_snapshots", "ComponentBasis", "ReducedAssembly",
           "ReducedAssemblyResult"]


def legendre_interface_parametrisation(s, degree, dofs_per_node=1):
    """Smooth interface displacement fields: Legendre polynomials up to `degree` along the coordinate `s` of the
    interface nodes, one set per component of the field. Returns G (n_nodes*dofs_per_node, (degree+1)*dofs_per_node)
    with DOF ordering node*dofs_per_node + component. For vector fields the constant columns are the translations;
    add rotation columns yourself if the interface can rotate rigidly."""
    s = np.asarray(s, float)
    x = 2 * (s - s.min()) / max(s.max() - s.min(), 1e-300) - 1
    P = np.polynomial.legendre.legvander(x, degree)
    n = len(s)
    G = np.zeros((n * dofs_per_node, (degree + 1) * dofs_per_node))
    for c in range(dofs_per_node):
        G[c::dofs_per_node, c * (degree + 1):(c + 1) * (degree + 1)] = P
    return G


def component_snapshots(solve_fn, n_params, n_samples, scale=1.0, rng=None, include_extremes=True):
    """Snapshots of one component from `solve_fn(p) -> u` over an optimal Latin hypercube of parameters
    p in [-scale, scale]^n_params (interface-data coefficients, plus load factors if the caller appends them).
    With `include_extremes` the zero parameter and the +-scale axis points are added. Returns (S (n, n_snap),
    P (n_snap, n_params))."""
    rng = np.random.default_rng(rng)
    D = (2 * optimal_lhs(n_samples, n_params, rng=rng) - 1) * scale
    if include_extremes:
        ext = [np.zeros(n_params)]
        for j in range(n_params):
            for sgn in (1.0, -1.0):
                e = np.zeros(n_params)
                e[j] = sgn * scale
                ext.append(e)
        D = np.vstack([np.array(ext), D])
    S = np.column_stack([np.asarray(solve_fn(p), float) for p in D])
    return S, D


class ComponentBasis:
    """POD basis of one component.

    snapshots : (n, n_snap). rank or tol: number of modes, or the relative singular-value cut-off.
    interface : DOFs of the component on the coupling interface.
    unreduced_interface : build the basis as the SLAVE form [E_interface | Phi_interior] (identity on the interface
        DOFs plus a POD basis of the interior rows of the snapshots), otherwise the plain POD of all rows.
    """

    def __init__(self, snapshots, interface, rank=None, tol=1e-8, unreduced_interface=False):
        S = np.asarray(snapshots, float)
        self.n = S.shape[0]
        self.interface = np.asarray(interface, int)
        self.unreduced_interface = bool(unreduced_interface)
        rows = np.setdiff1d(np.arange(self.n), self.interface) if unreduced_interface else np.arange(self.n)
        U, s, _ = np.linalg.svd(S[rows], full_matrices=False)
        k = int((s > tol * s[0]).sum()) if rank is None else int(min(rank, len(s)))
        self.singular_values = s
        V = np.zeros((self.n, k))
        V[rows] = U[:, :k]
        self.interior = V                                   # POD part (zero on the interface rows when unreduced)
        self.rank = k

    @property
    def size(self):
        return self.rank + (len(self.interface) if self.unreduced_interface else 0)


class ReducedAssemblyResult:
    def __init__(self, x, u, residual_norm, iterations, converged, n_reduced, n_full):
        self.x, self.u = x, u
        self.residual_norm, self.iterations, self.converged = residual_norm, iterations, converged
        self.n_reduced, self.n_full = n_reduced, n_full

    def __repr__(self):
        return (f"ReducedAssemblyResult(n_reduced={self.n_reduced}, n_full={self.n_full}, "
                f"iterations={self.iterations}, converged={self.converged})")


class ReducedAssembly:
    """Master component `a` (reduced POD basis), slave component `b` (interior POD basis, interface eliminated
    through the mortar operator P of shape (n_slave_interface, n_master_interface)).

    Stacked full-order vector: [u_a; u_b] (n_a + n_b entries); `interface_a`, `interface_b` are the interface DOFs
    of each component in the same order that P assumes. The master basis is a `ComponentBasis` with
    unreduced_interface=False, the slave basis one with unreduced_interface=True.
    """

    def __init__(self, basis_a, basis_b, P, interface_a=None, interface_b=None):
        self.a, self.b = basis_a, basis_b
        self.P = np.asarray(P, float)
        ia = basis_a.interface if interface_a is None else np.asarray(interface_a, int)
        ib = basis_b.interface if interface_b is None else np.asarray(interface_b, int)
        if self.P.shape != (len(ib), len(ia)):
            raise ValueError("P must have shape (n_interface_b, n_interface_a)")
        na, nb = basis_a.n, basis_b.n
        ra, rb = basis_a.rank, basis_b.rank
        T = np.zeros((na + nb, ra + rb))
        T[:na, :ra] = basis_a.interior
        T[na:, ra:] = basis_b.interior
        T[na + ib, :ra] = self.P @ basis_a.interior[ia]              # slave interface follows the master through P
        self.T = T
        self.n_full = na + nb
        self.n_reduced = ra + rb
        self.interface_a, self.interface_b = ia, ib

    def expand(self, x):
        return self.T @ np.asarray(x, float)

    def solve(self, residual, tangent, x0=None, tol=1e-9, max_iter=30):
        """Newton on T^T R(T x) = 0. residual(u) -> R (n_full,) = internal force - external force; tangent(u) ->
        K_t (n_full, n_full), dense or sparse."""
        x = np.zeros(self.n_reduced) if x0 is None else np.asarray(x0, float).copy()
        R0 = None
        for it in range(1, max_iter + 1):
            u = self.T @ x
            r = self.T.T @ residual(u)
            if R0 is None:
                R0 = max(np.linalg.norm(r), 1e-300)
            if np.linalg.norm(r) <= tol * R0 or np.linalg.norm(r) < 1e-14:
                return ReducedAssemblyResult(x, u, float(np.linalg.norm(r)), it - 1, True, self.n_reduced, self.n_full)
            Kt = tangent(u)
            Kr = self.T.T @ (Kt @ self.T)
            x = x - np.linalg.solve(Kr, r)
        u = self.T @ x
        r = self.T.T @ residual(u)
        return ReducedAssemblyResult(x, u, float(np.linalg.norm(r)), max_iter, False, self.n_reduced, self.n_full)

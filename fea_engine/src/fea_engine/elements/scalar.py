# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
scalar.py -- scalar-field elements: one DOF per node, for the Laplace and Poisson equations.

`Quad4Laplace` is the 4-node bilinear quadrilateral for  -div(k grad u) = q  (membrane under transverse load, steady heat
conduction, potential flow, torsion function) and, with its mass matrix, for the wave equation  rho u_tt - div(k grad u) = q
(a vibrating membrane) and the heat equation  rho c u_t - div(k grad u) = q.

    stiffness  Ke = sum_gp  B^T D B |J| w * thickness,   B = [dN/dx; dN/dy] (2 x 4),   D = k I  (or a 2 x 2 matrix)
    mass       Me = sum_gp  N^T rho N |J| w * thickness   (consistent), or the HRZ-lumped diagonal
    source     qe = sum_gp  N^T q |J| w * thickness       (consistent load of a source, `source_vector`)

The generic Gauss loop of `Element` is used unchanged; only `shape_and_derivs` and `B_matrix` are specific. Inputs are more
forgiving than for plane elements: the conductivity may be a scalar `k` or a 2 x 2 matrix, the density a scalar or a 1 x 1 matrix.
The single DOF is named "u" (`fix_dofs(nodes, ["u"])`).

Sign convention of the report this element comes from (Balajewicz et al. 2015, obstacle membrane): the membrane equation is
-laplace(u) = f with f = -10 (downward load); assemble the stiffness with k = 1 and the load with `source_vector` or `M @ q`.

Checked: the element matrix of a unit square equals the closed form (1/6)[[4,-1,-2,-1],...]; a linear field is reproduced
exactly on distorted meshes (patch test); the nodal error of a manufactured Poisson solution falls at second order; the
lowest Dirichlet-membrane eigenvalue converges to 2 pi^2 (see tests/test_quad4_laplace.py). Not provided: triangles and
hexahedra, variable coefficients, flux recovery.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .base import Element, gauss_product, jacobian, jacobian_measure


def _conductivity(D, dim):
    """A scalar k becomes k * I; a (dim, dim) matrix is used as given."""
    D = np.asarray(D, float)
    if D.ndim == 0:
        return float(D) * np.eye(dim)
    if D.shape != (dim, dim):
        raise ValueError(f"conductivity must be a scalar or a ({dim}, {dim}) matrix, got shape {D.shape}")
    return D


def _density(rho):
    """A scalar rho (or rho*c) or a 1 x 1 matrix -> 1 x 1 matrix."""
    r = np.asarray(rho, float)
    if r.ndim == 0:
        return r.reshape(1, 1)
    if r.shape != (1, 1):
        raise ValueError(f"density must be a scalar or a 1 x 1 matrix (one DOF per node), got shape {r.shape}")
    return r


class Quad4Laplace(Element):
    """4-node bilinear scalar-field quadrilateral (Laplace / Poisson / membrane / heat), one DOF per node ("u").

    stiffness(coords, D, thickness=1.0): D = k (scalar) or a 2 x 2 conductivity matrix.
    mass(coords, rho, thickness=1.0) and lumped_mass(...) (HRZ): rho scalar or 1 x 1.
    source_vector(coords, q, thickness=1.0): consistent nodal load of a source q (scalar, or 4 nodal values).
    Node order is counter-clockwise, as for `Quad4PlaneStress`."""
    n_nodes, dofs_per_node, dim, gauss_order = 4, 1, 2, 2
    dof_names = ("u",)

    def shape_and_derivs(self, natural_coords):
        xi, eta = natural_coords
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return N, np.vstack([dN_dxi, dN_deta])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        return np.linalg.solve(J, dN_nat), detJ          # (2, 4): [dN/dx; dN/dy]

    def stiffness(self, elem_coords, D, thickness=1.0, gauss_order=None):
        return super().stiffness(elem_coords, _conductivity(D, self.dim), thickness, gauss_order)

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        return super().mass(elem_coords, _density(rho_matrix), thickness)

    def lumped_mass(self, elem_coords, rho_matrix, thickness=1.0):
        return super().lumped_mass(elem_coords, _density(rho_matrix), thickness)

    def source_vector(self, elem_coords, q, thickness=1.0):
        """Consistent load vector (4,) of the source q in  -div(k grad u) = q: a scalar (uniform) or the 4 nodal values
        (interpolated bilinearly)."""
        q = np.asarray(q, float)
        if q.ndim and q.shape != (self.n_nodes,):
            raise ValueError(f"q must be a scalar or {self.n_nodes} nodal values, got shape {q.shape}")
        pts, wts = gauss_product(self.gauss_order, self.dim)
        out = np.zeros(self.n_nodes)
        for p, w in zip(pts, wts):
            N, dN = self._cached_shape_and_derivs(p)
            qp = float(q) if q.ndim == 0 else float(N @ q)
            out += N * qp * jacobian_measure(dN, elem_coords) * w * thickness
        return out

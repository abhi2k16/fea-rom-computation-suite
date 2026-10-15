# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
hex_tl.py -- Total-Lagrangian, St. Venant-Kirchhoff hexahedra: Hex8SolidTL and Hex20SolidTL.

Geometric nonlinearity only (large displacement and rotation, small strain), linear elastic material applied to the
Green-Lagrange strain: S = D : E with E = (F^T F - I) / 2, F = I + du/dX. Same constitutive model as Tet10SolidTL and
Quad4PlaneStressTL, on the 8-node hexahedron and on the 20-node serendipity hexahedron.

Per Gauss point, with dN = dN/dX (3 x n) from the reference geometry and u the nodal displacements (n x 3):

    F        = I + u^T dN^T
    E_voigt  = [E_xx, E_yy, E_zz, 2E_xy, 2E_yz, 2E_xz]           (the row order of Hex8Solid3D.B_matrix)
    B_nl     : (6 x 3n), column (a, i) holds  dE_voigt / du_{a,i}
    f_int    = sum_gp  B_nl^T S                 detJ w
    K_T      = sum_gp (B_nl^T D B_nl + (dN^T S dN) (x) I_3)  detJ w          S = D E_voigt as a 3 x 3 tensor

This file holds the per-Gauss-point REFERENCE implementation, written in the style of Tet10SolidTL so it can serve as an
independent check on the batched kernels (added separately). It is a subclass of the linear element, so shape
functions, node order and mass are those of Hex8Solid3D (Hex20Solid3D for Hex20SolidTL, whose mid-edge nodes follow
the order 8-11 bottom edges (0-1, 1-2, 2-3, 3-0), 12-15 top edges, 16-19 vertical edges (0-4, 1-5, 2-6, 3-7); a mesh
from Gmsh needs ``Hex20Solid3D.GMSH_NODE_ORDER`` first).

``mat`` is either the 6 x 6 elasticity matrix (material.D_solid3d(Material)), as Hex8Solid3D.stiffness takes, or the
Material itself, as Tet10SolidTL takes. A Material is converted with D_solid3d on every call.

Quadrature: ``gauss_order`` points per direction. Defaults are those of the linear parents: 2 (2 x 2 x 2) for
Hex8SolidTL and 3 (3 x 3 x 3) for Hex20SolidTL; pass ``Hex20SolidTL(gauss_order=2)`` or set the attribute to change it.
Reduced 2 x 2 x 2 integration of the 20-node element is cheaper (27 against 8 Gauss points per element), but the
sibling project reported that a single layer of such elements failed at large rotation; that claim has not been
reproduced in this package yet, so no warning is issued and the 3 x 3 x 3 default stays. Like Hex8Solid3D, Hex8SolidTL
shear-locks in bending-dominated problems on coarse meshes (measured in this package, see the tests).

References (methods in this module are adopted from):
  - K.-J. Bathe (1996), "Finite Element Procedures", Prentice Hall (total-Lagrangian formulation, Sec. 6.3)
  - T. Belytschko, W. K. Liu, B. Moran, K. Elkhodary (2014), "Nonlinear Finite Elements for Continua and Structures",
    2nd ed., Wiley (St. Venant-Kirchhoff material, B_nl and geometric stiffness)
  - O. C. Zienkiewicz, R. L. Taylor (2000), "The Finite Element Method", 5th ed., Butterworth-Heinemann
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .base import gauss_product, jacobian
from .solids import Hex8Solid3D, Hex20Solid3D
from ..material import D_solid3d


class _HexTLReference:
    """Per-Gauss-point total-Lagrangian internal force and tangent for a 3-D hexahedron. Mixin: the concrete class
    also derives from the linear hexahedron that supplies ``n_nodes``, ``gauss_order`` and
    ``_cached_shape_and_derivs``."""

    def __init__(self, gauss_order=None):
        if gauss_order is not None:
            if int(gauss_order) != gauss_order or gauss_order < 1:
                raise ValueError(f"gauss_order must be a positive integer, got {gauss_order!r}")
            self.gauss_order = int(gauss_order)

    @staticmethod
    def _elasticity(mat):
        """6 x 6 elasticity matrix from a D matrix or a Material."""
        if hasattr(mat, "E") and hasattr(mat, "nu"):
            return D_solid3d(mat)
        D = np.asarray(mat, dtype=float)
        if D.shape != (6, 6):
            raise ValueError(f"mat must be a Material or a 6 x 6 elasticity matrix, got shape {D.shape}")
        return D

    def _gp_kinematics(self, nat, elem_coords, u_elem):
        """dN/dX (3, n), detJ and F (3, 3) at one Gauss point."""
        _, dN_nat = self._cached_shape_and_derivs(tuple(nat))
        J, detJ = jacobian(dN_nat, elem_coords)
        dN = np.linalg.solve(J, dN_nat)
        F = np.eye(3) + np.asarray(u_elem, float).reshape(self.n_nodes, 3).T @ dN.T
        return dN, detJ, F

    @staticmethod
    def _green_lagrange_voigt(F):
        E = 0.5 * (F.T @ F - np.eye(3))
        return np.array([E[0, 0], E[1, 1], E[2, 2], 2.0 * E[0, 1], 2.0 * E[1, 2], 2.0 * E[0, 2]])

    @staticmethod
    def _B_nl(dN, F):
        """(6, 3n) matrix of dE_voigt/du; column 3a+i is node a, displacement component i."""
        n = dN.shape[1]
        B = np.zeros((6, n, 3))
        for i in range(3):
            B[0, :, i] = F[i, 0] * dN[0]
            B[1, :, i] = F[i, 1] * dN[1]
            B[2, :, i] = F[i, 2] * dN[2]
            B[3, :, i] = F[i, 0] * dN[1] + F[i, 1] * dN[0]
            B[4, :, i] = F[i, 1] * dN[2] + F[i, 2] * dN[1]
            B[5, :, i] = F[i, 0] * dN[2] + F[i, 2] * dN[0]
        return B.reshape(6, 3 * n)

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        D = self._elasticity(mat)
        pts, wts = gauss_product(self.gauss_order, 3)
        f = np.zeros(3 * self.n_nodes)
        for nat, w in zip(pts, wts):
            dN, detJ, F = self._gp_kinematics(nat, elem_coords, u_elem)
            S = D @ self._green_lagrange_voigt(F)
            f += self._B_nl(dN, F).T @ S * (detJ * w)
        return f

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        D = self._elasticity(mat)
        pts, wts = gauss_product(self.gauss_order, 3)
        n = self.n_nodes
        K = np.zeros((3 * n, 3 * n))
        for nat, w in zip(pts, wts):
            dN, detJ, F = self._gp_kinematics(nat, elem_coords, u_elem)
            S = D @ self._green_lagrange_voigt(F)
            S3 = np.array([[S[0], S[3], S[5]], [S[3], S[1], S[4]], [S[5], S[4], S[2]]])
            B = self._B_nl(dN, F)
            Kg = dN.T @ S3 @ dN                                  # (n, n): dN_a . S dN_b
            c = detJ * w
            K += (B.T @ D @ B) * c
            for i in range(3):
                K[i::3, i::3] += Kg * c
        return K


class Hex8SolidTL(_HexTLReference, Hex8Solid3D):
    """Hex8Solid3D with Total-Lagrangian geometric nonlinearity and a St. Venant-Kirchhoff law (see the module
    docstring). Default 2 x 2 x 2 integration; shear-locks in bending like its linear parent."""


class Hex20SolidTL(_HexTLReference, Hex20Solid3D):
    """Hex20Solid3D (20-node serendipity hexahedron) with Total-Lagrangian geometric nonlinearity and a
    St. Venant-Kirchhoff law (see the module docstring). Default 3 x 3 x 3 integration."""

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

This file holds two separately written implementations of the same formulas: the per-Gauss-point REFERENCE
(``internal_force`` / ``tangent_stiffness``, written in the style of Tet10SolidTL) and the opt-in BATCHED kernels
(``internal_force_batch`` / ``tangent_stiffness_batch``, all elements of a block at once, with the Gauss-point sum folded
into one matrix product). ``FESystem.assemble_*(vectorized=True)`` uses the batched kernels; the reference stays the
default and the oracle that the batched kernels are tested against. A 3-D hexahedron is compute-bound, so batching removes
much less interpreter overhead than for the 2-D quadrilateral; measured ratios are in tests/batch_timing_hex.py output and
in docs/batched_nonlinear_kernels_plan.md. It is a subclass of the linear element, so shape
functions, node order and mass are those of Hex8Solid3D (Hex20Solid3D for Hex20SolidTL, whose mid-edge nodes follow
the order 8-11 bottom edges (0-1, 1-2, 2-3, 3-0), 12-15 top edges, 16-19 vertical edges (0-4, 1-5, 2-6, 3-7); a mesh
from Gmsh needs ``Hex20Solid3D.GMSH_NODE_ORDER`` first).

``mat`` is either the 6 x 6 elasticity matrix (material.D_solid3d(Material)), as Hex8Solid3D.stiffness takes, or the
Material itself, as Tet10SolidTL takes. A Material is converted with D_solid3d on every call.

Quadrature: ``gauss_order`` points per direction. Defaults are those of the linear parents: 2 (2 x 2 x 2) for
Hex8SolidTL and 3 (3 x 3 x 3) for Hex20SolidTL; pass ``Hex20SolidTL(gauss_order=2)`` or set the attribute to change it.
Reduced 2 x 2 x 2 integration of the 20-node element is cheaper (8 against 27 Gauss points per element) but has
spurious zero-energy modes: an isolated element has 6 beyond the 6 rigid-body modes, and a mesh that is a single row of
elements (one element across both width and thickness) keeps them after clamping (one per element in the meshes tried),
so its tangent at zero displacement is singular and Newton fails at the first load step. Meshes with two elements across
either direction had none. ``Hex20SolidTL(gauss_order=2)`` therefore warns; the 3 x 3 x 3 default has no spurious modes.
Measured in this package (tests/test_hex_tl_validation.py). Like Hex8Solid3D, Hex8SolidTL shear-locks in bending-dominated
problems on coarse meshes (tip deflection 0.28 to 0.80 of Timoshenko for a 30:1 strip against 0.90 to 0.98 for Hex20).

References (methods in this module are adopted from):
  - K.-J. Bathe (1996), "Finite Element Procedures", Prentice Hall (total-Lagrangian formulation, Sec. 6.3)
  - T. Belytschko, W. K. Liu, B. Moran, K. Elkhodary (2014), "Nonlinear Finite Elements for Continua and Structures",
    2nd ed., Wiley (St. Venant-Kirchhoff material, B_nl and geometric stiffness)
  - O. C. Zienkiewicz, R. L. Taylor (2000), "The Finite Element Method", 5th ed., Butterworth-Heinemann
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import warnings

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
        if self.n_nodes == 20 and self.gauss_order < 3:
            warnings.warn(
                f"{type(self).__name__}(gauss_order={self.gauss_order}): {self.gauss_order}x{self.gauss_order}x"
                f"{self.gauss_order} integration of the 20-node hexahedron has 6 spurious zero-energy modes per isolated "
                f"element; a mesh that is a single row of elements (one element across both width and thickness) keeps "
                f"them, its tangent is singular and Newton fails. Use gauss_order=3 (default) or at least two elements "
                f"across the width or the thickness.", UserWarning, stacklevel=2)

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


class _HexTLBatch:
    """Batched total-Lagrangian kernels for a 3-D hexahedron: E elements at once, ``coords`` (E, n, 3), ``u`` (E, 3n),
    ``mat`` a Material or the 6 x 6 elasticity matrix (constant over the block), results (E, 3n) and (E, 3n, 3n).

    ``geometry`` is an optional precomputed pair (dN/dX with shape (E, Q, n, 3), det(J0) * w with shape (E, Q)) for these
    elements and this element's Gauss rule; ``FESystem`` passes its cached one. Without it the pair is computed from
    ``coords``. Written independently of ``_HexTLReference`` (array expressions instead of per-point loops)."""

    supports_batch = True

    def batch_bytes_per_element(self, kind="tangent"):
        """Working bytes per element: the generic estimate (strain-displacement array and the element matrix) plus the
        deformation-gradient / strain / stress arrays; see batched_kernels.default_bytes_per_element."""
        from ..batched_kernels import default_bytes_per_element
        return default_bytes_per_element(self.gauss_order ** 3, 3 * self.n_nodes, 3, kind)

    def _reference_geometry_batch(self, coords):
        pts, wts = gauss_product(self.gauss_order, 3)
        dN_nat = np.stack([self._cached_shape_and_derivs(tuple(p))[1] for p in pts])           # (Q, 3, n)
        coords = np.asarray(coords, float)
        J = np.einsum('qin,enj->eqij', dN_nat, coords)                                         # (E, Q, 3, 3)
        detJ = np.linalg.det(J)
        rhs = np.ascontiguousarray(np.broadcast_to(dN_nat, (coords.shape[0],) + dN_nat.shape))
        dN = np.transpose(np.linalg.solve(J, rhs), (0, 1, 3, 2))                               # (E, Q, n, 3)
        return dN, detJ * np.asarray(wts)[None, :]

    def _batch_kinematics(self, coords, u, mat, geometry):
        D = self._elasticity(mat)
        dN, JxW = self._reference_geometry_batch(coords) if geometry is None else geometry
        n = self.n_nodes
        u = np.asarray(u, float).reshape(-1, n, 3)
        F = np.eye(3) + np.einsum('eai,eqaj->eqij', u, dN)                                      # F[i,j] = d_ij + u_a,i dN_a,j
        C = np.einsum('eqki,eqkj->eqij', F, F)
        Ev = np.empty(F.shape[:2] + (6,))
        Ev[..., 0] = 0.5 * (C[..., 0, 0] - 1.0)
        Ev[..., 1] = 0.5 * (C[..., 1, 1] - 1.0)
        Ev[..., 2] = 0.5 * (C[..., 2, 2] - 1.0)
        Ev[..., 3], Ev[..., 4], Ev[..., 5] = C[..., 0, 1], C[..., 1, 2], C[..., 0, 2]          # 2 E_xy, 2 E_yz, 2 E_xz
        S = Ev @ D.T                                                                            # (E, Q, 6)
        nx, ny, nz = dN[..., 0], dN[..., 1], dN[..., 2]                                         # (E, Q, n)
        B = np.empty(F.shape[:2] + (6, n, 3))
        for i in range(3):
            f0, f1, f2 = F[:, :, i, 0, None], F[:, :, i, 1, None], F[:, :, i, 2, None]         # (E, Q, 1)
            B[:, :, 0, :, i] = f0 * nx
            B[:, :, 1, :, i] = f1 * ny
            B[:, :, 2, :, i] = f2 * nz
            B[:, :, 3, :, i] = f0 * ny + f1 * nx
            B[:, :, 4, :, i] = f1 * nz + f2 * ny
            B[:, :, 5, :, i] = f0 * nz + f2 * nx
        return D, dN, B.reshape(B.shape[:3] + (3 * n,)), S, JxW

    def internal_force_batch(self, coords, u, mat, thickness=1.0, geometry=None, **kwargs):
        """Internal forces (E, 3n): sum over Gauss points of B_nl^T S detJ w."""
        _, _, B, S, c = self._batch_kinematics(coords, u, mat, geometry)
        return np.einsum('eqsd,eqs,eq->ed', B, S, c, optimize=True)

    def tangent_stiffness_batch(self, coords, u, mat, thickness=1.0, geometry=None, **kwargs):
        """Tangent matrices (E, 3n, 3n): sum over Gauss points of detJ w (B_nl^T D B_nl + (dN^T S dN) (x) I3), each
        sum folded into one batched matrix product (contraction lengths Q*6 and Q*3)."""
        D, dN, B, S, c = self._batch_kinematics(coords, u, mat, geometry)
        E, Q = c.shape
        n = self.n_nodes
        nd = 3 * n
        Bc = (B * c[:, :, None, None]).reshape(E, Q * 6, nd)
        DB = (D @ B).reshape(E, Q * 6, nd)
        K = Bc.transpose(0, 2, 1) @ DB                                                          # (E, 3n, 3n)
        S3 = np.empty((E, Q, 3, 3))
        S3[..., 0, 0], S3[..., 1, 1], S3[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
        S3[..., 0, 1] = S3[..., 1, 0] = S[..., 3]
        S3[..., 1, 2] = S3[..., 2, 1] = S[..., 4]
        S3[..., 0, 2] = S3[..., 2, 0] = S[..., 5]
        SdN = S3 @ dN.transpose(0, 1, 3, 2)                                                     # (E, Q, 3, n): S dN_b
        A = (dN * c[:, :, None, None]).transpose(0, 1, 3, 2).reshape(E, Q * 3, n)               # c dN_a,k
        Kg = A.transpose(0, 2, 1) @ SdN.reshape(E, Q * 3, n)                                    # (E, n, n)
        for i in range(3):
            K[:, i::3, i::3] += Kg
        return K


class Hex8SolidTL(_HexTLReference, _HexTLBatch, Hex8Solid3D):
    """Hex8Solid3D with Total-Lagrangian geometric nonlinearity and a St. Venant-Kirchhoff law (see the module
    docstring). Default 2 x 2 x 2 integration; shear-locks in bending like its linear parent."""


class Hex20SolidTL(_HexTLReference, _HexTLBatch, Hex20Solid3D):
    """Hex20Solid3D (20-node serendipity hexahedron) with Total-Lagrangian geometric nonlinearity and a
    St. Venant-Kirchhoff law (see the module docstring). Default 3 x 3 x 3 integration."""

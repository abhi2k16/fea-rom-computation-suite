# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
golden_loop_baselines.py -- frozen reference results of the CURRENT per-element-loop nonlinear assembly.

Purpose: before any batched/sparse kernel exists, record what the loop code returns for fixed inputs, so that later
changes (batched kernels, sparse tangent, sparse Newton) can be compared with the loop results of this date rather
than with whatever the loop happens to return after the edit. The golden file is NOT a validation of the physics
(tests/test_quad4_tl.py and tests/test_tet10_geometric_nonlinear.py do that); it only pins current behaviour.

Contents of tests/data/golden_loop_baselines.npz (all float64):
  quad_*  : 12x4 Quad4PlaneStressTL plate (60 nodes, 48 elements, distorted by a fixed smooth map), thickness 0.5,
            D_plane_stress(E=1e3, nu=0.3), three random displacement states (scales 0.02, 0.1, 0.3):
            internal force and dense tangent per state; cantilever load-controlled Newton path (U at every step).
  tet_*   : 2x1x1-cell Tet10SolidTL box (6 tets per cell, mid-edge nodes added), D_solid3d(E=1e3, nu=0.3), two random
            states: internal force and dense tangent; clamped-end load-controlled Newton path (U at every step).

Size of the geometric nonlinearity in the two Newton paths (measured on 2026-10-15 with this generator): the
converged TL displacement norm is 0.956 of the linear-static norm for the Quad4 plate (max |u| = 1.77 on a 6 x 2
plate, load 40) and 0.994 for the Tet10 box (max |u| = 0.236 on a 2 x 1 x 1 box, load 60). The Tet10 case is only
mildly nonlinear; it pins current behaviour but is a weak test of the nonlinear terms.

Regenerate (only when the loop code is changed ON PURPOSE, and say so in the commit/CHANGELOG):
    python tests/golden_loop_baselines.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import numpy as np

from fea_engine import FESystem, Material, MultiBlockMesh, Mesh
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL, Tet10SolidTL
import fea_engine.nonlinear_solver as nls

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden_loop_baselines.npz")
E_MOD, NU = 1.0e3, 0.3
QUAD_SCALES = (0.02, 0.1, 0.3)
TET_SCALES = (0.01, 0.05)


# ----------------------------------------------------------------------------- meshes
def quad_plate(nx=12, ny=4, L=6.0, H=2.0):
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    nodes[:, 0] += 0.08 * np.sin(2.0 * nodes[:, 1])                    # fixed distortion: non-rectangular elements
    nodes[:, 1] += 0.05 * np.sin(1.5 * nodes[:, 0])
    elems = np.array([[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
                      for j in range(ny) for i in range(nx)])
    return nodes, elems, nx, ny


def tet10_box(nx=2, ny=1, nz=1, L=2.0, W=1.0, T=1.0):
    """Structured Tet10 box: each hexahedral cell split into 6 tetrahedra (Kuhn), mid-edge nodes shared.
    Node order inside an element is Tet10Solid3D's: 4 corners, then edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3)."""
    def cid(i, j, k):
        return (k * (ny + 1) + j) * (nx + 1) + i
    corners = np.array([[L * i / nx, W * j / ny, T * k / nz] for k in range(nz + 1) for j in range(ny + 1)
                        for i in range(nx + 1)])
    kuhn = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    offs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    nodes = [c for c in corners]
    mid = {}
    elems = []
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = [cid(i + o[0], j + o[1], k + o[2]) for o in offs]
                for t in kuhn:
                    c = [cell[v] for v in t]
                    # orient positively
                    P = corners[c]
                    if np.linalg.det(np.array([P[1] - P[0], P[2] - P[0], P[3] - P[0]])) < 0:
                        c[1], c[2] = c[2], c[1]
                    row = list(c)
                    for a, b in edges:
                        key = tuple(sorted((c[a], c[b])))
                        if key not in mid:
                            mid[key] = len(nodes)
                            nodes.append(0.5 * (corners[key[0]] + corners[key[1]]))
                        row.append(mid[key])
                    elems.append(row)
    return np.array(nodes), np.array(elems, dtype=int)


# ----------------------------------------------------------------------------- generators
def _quad_system():
    nodes, elems, nx, ny = quad_plate()
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    return fes, nodes, elems, nx, ny


def _tet_system():
    nodes, elems = tet10_box()
    fes = FESystem(Mesh(nodes, elems, 3), Tet10SolidTL(), sparse=False)
    return fes, nodes, elems


def quad_case():
    """Return (fes, mat, thickness, dict of kwargs for assemble_*) for the Quad4 plate."""
    fes, nodes, elems, nx, ny = _quad_system()
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    fes.add_nodal_force(right, 1, -40.0 / (ny + 1))
    return fes, {"q": D_plane_stress(Material(E=E_MOD, nu=NU))}, 0.5


def tet_case():
    fes, nodes, elems = _tet_system()
    mat = Material(E=E_MOD, nu=NU)
    left = np.where(np.abs(nodes[:, 0]) < 1e-9)[0]
    right = np.where(np.abs(nodes[:, 0] - 2.0) < 1e-9)[0]
    fes.fix_dofs(left, [0, 1, 2])
    fes.add_nodal_force(right, 1, -60.0 / len(right))
    return fes, mat                                   # Tet10SolidTL takes the Material itself (it builds D internally)


def random_state(n_dof, scale, seed):
    return np.random.default_rng(seed).normal(size=n_dof) * scale


def generate():
    out = {}
    fes, mat, t = quad_case()
    for k, s in enumerate(QUAD_SCALES):
        u = random_state(fes.n_dof, s, 100 + k)
        out[f"quad_u{k}"] = u
        out[f"quad_f{k}"] = fes.assemble_internal_force(u, mat, thickness=t)
        out[f"quad_K{k}"] = fes.assemble_tangent_stiffness(u, mat, thickness=t)
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=t)
    out["quad_newton_lf"], out["quad_newton_U"] = np.asarray(lf), np.asarray(U)

    fes, mat = tet_case()
    for k, s in enumerate(TET_SCALES):
        u = random_state(fes.n_dof, s, 200 + k)
        out[f"tet_u{k}"] = u
        out[f"tet_f{k}"] = fes.assemble_internal_force(u, mat)
        out[f"tet_K{k}"] = fes.assemble_tangent_stiffness(u, mat)
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-5, max_iter=40)
    out["tet_newton_lf"], out["tet_newton_U"] = np.asarray(lf), np.asarray(U)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    data = generate()
    np.savez_compressed(GOLDEN_PATH, **data)
    print("wrote", GOLDEN_PATH)
    for k, v in data.items():
        print(f"  {k:18s} {v.shape}")

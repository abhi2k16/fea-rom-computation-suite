# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
import_abaqus_mesh_demo.py -- an end-to-end example: a mesh written by another program -> loads and constraints -> solve -> ParaView file.

The "other program" is simulated by writing an Abaqus .inp file by hand, as a pre-processor would (nodes, C3D8 elements, node and
element sets, a material card, boundary conditions and a load step). fea_engine reads only the NODES and the ELEMENT CONNECTIVITY of
that file; sets, material, boundary conditions and loads are defined again in Python (see Chapter 2 of the book and the user guide).

To use your own file, run:   python import_abaqus_mesh_demo.py path/to/model.inp
(then edit the selections below: they assume a cantilever along x, clamped at x = 0 and loaded at x = L).

Needs the optional `meshio` package:  pip install meshio
Writes beam_c3d8.inp (unless a file is given) and beam_c3d8.vtu in the current directory.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import sys
import numpy as np

from fea_engine import mesh_io
from fea_engine.material import Material, D_solid3d
from fea_engine.elements import Hex8Solid3D
from fea_engine.solver import FESystem
from fea_engine.export import write_vtu


def write_foreign_inp(path, L=10.0, h=1.0, nx=20, ny=2, nz=2):
    """A small Abaqus input file as a pre-processor would write it (not produced by fea_engine or meshio)."""
    xs, ys, zs = np.linspace(0, L, nx + 1), np.linspace(0, h, ny + 1), np.linspace(0, h, nz + 1)
    nid = lambda i, j, k: 1 + i + (nx + 1) * (j + (ny + 1) * k)          # Abaqus ids start at 1
    lines = ["*HEADING", "cantilever beam, units: N, m, Pa", "*NODE"]
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                lines.append(f"{nid(i, j, k)}, {xs[i]:.10g}, {ys[j]:.10g}, {zs[k]:.10g}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=BEAM")
    e = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                lines.append(f"{e}, " + ", ".join(map(str, n))); e += 1
    root = [nid(0, j, k) for k in range(nz + 1) for j in range(ny + 1)]
    lines += ["*NSET, NSET=ROOT"] + [", ".join(map(str, root[m:m + 8])) for m in range(0, len(root), 8)]
    lines += ["*SOLID SECTION, ELSET=BEAM, MATERIAL=STEEL", "*MATERIAL, NAME=STEEL", "*ELASTIC", "2.1e11, 0.3",
              "*STEP", "*STATIC", "*BOUNDARY", "ROOT, 1, 3", "*CLOAD", f"{nid(nx, 0, nz)}, 3, -1000.", "*END STEP"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def solve_imported(mesh, E=2.1e11, nu=0.3, P=-1000.0):
    """Clamp x = 0, spread the vertical load P over the nodes of the plane x = L, solve."""
    L = mesh.nodes[:, 0].max()
    s = FESystem(mesh, Hex8Solid3D())
    s.assemble_stiffness(D_solid3d(Material(E=E, nu=nu, rho=7850.0)))
    root = mesh.select_nodes(x=0.0, name="root")
    tip = mesh.select_nodes(x=L, name="tip")
    s.fix_dofs(root, [0, 1, 2])
    s.add_nodal_force(tip, dof_index=2, total_force=P)
    return s, s.solve_static(), tip


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "beam_c3d8.inp"
    if len(sys.argv) == 1:
        write_foreign_inp(path)
    mesh = mesh_io.read_mesh(path, dim=3)                       # dim: keep the 3-D cells, drop boundary faces
    print(f"read {path}: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")
    mesh.check_quality(Hex8Solid3D())                           # all Jacobians must be positive (node order, orientation)
    s, U, tip = solve_imported(mesh)
    w = U[3 * tip + 2].mean()
    print(f"mean tip deflection (z): {w:.6e} m")
    L, h = mesh.nodes[:, 0].max(), mesh.nodes[:, 1].max()           # beam theory needs the section: h x h here
    print(f"ratio to Euler-Bernoulli P L^3/(3 E I): {w / (-1000.0 * L**3 / (3 * 2.1e11 * h**4 / 12)):.4f}")
    write_vtu("beam_c3d8.vtu", fields={"displacement": s.field(U, "displacement")},
              point_data={"von_mises": s.von_mises(U)})
    print("wrote beam_c3d8.vtu (open in ParaView)")

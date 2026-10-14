# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_07_pyvista_visualization.py -- Example Gallery: 3D visualization of FE results with PyVista
(fea_engine.visualization.pyvista_plot; needs ``pip install fea_engine[viz]``).

Two models, each rendered off-screen to a PNG:

  1. A 2-D plane-stress cantilever (Quad4PlaneStress): von Mises stress on the deformed mesh, with the undeformed mesh
     as a wire outline.
  2. A 3-D cantilever (Hex8Solid3D): displacement magnitude on the deformed shape.

Validation printed to stdout: the FE tip deflection against the Euler-Bernoulli value P L^3 / (3 E I) for each model
and the number of cells and points in the converted VTK grid. The .vtu files written alongside open in ParaView.

On a headless Linux machine run it under a virtual display:  xvfb-run -a python fea_07_pyvista_visualization.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, Hex8Solid3D, mesh
from fea_engine.visualization import plot_deformed, plot_field, save_vtk, to_pyvista

E, NU = 210e9, 0.3


def cantilever_2d(L=4.0, H=1.0, t=0.1, P=-2.0e5, nx=32, ny=8):
    m = mesh.rectangle_mesh(L, H, nx, ny)
    s = FESystem(m, Quad4PlaneStress(), thickness=t)
    s.assemble_stiffness(D_plane_stress(Material(E=E, nu=NU, rho=7800.0)), thickness=t)
    for n in m.nodes_on_line(0, 0.0):
        s.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    s.F[2 * tip + 1] = P / len(tip)
    U = s.solve_static()
    I = t * H ** 3 / 12.0
    return m, s, U, U[2 * tip + 1].mean(), P * L ** 3 / (3 * E * I)


def cantilever_3d(L=4.0, H=1.0, B=1.0, P=-2.0e6, nx=16, ny=4, nz=4):
    from fea_engine import D_solid3d
    m = mesh.box_mesh(L, H, B, nx, ny, nz)
    s = FESystem(m, Hex8Solid3D())
    s.assemble_stiffness(D_solid3d(Material(E=E, nu=NU, rho=7800.0)))
    for n in np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0]:
        s.fix_dofs([n], [0, 1, 2])
    tip = np.where(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    s.F[3 * tip + 1] = P / len(tip)
    U = s.solve_static()
    I = B * H ** 3 / 12.0
    return m, s, U, U[3 * tip + 1].mean(), P * L ** 3 / (3 * E * I)


def main():
    m2, s2, U2, tip2, ref2 = cantilever_2d()
    vm = s2.von_mises(U2)
    plot_field(m2, vm, component="von_mises", U=U2, scale="auto", clim=None,
               title="2-D cantilever: von Mises stress on the deformed mesh",
               screenshot="fea_07_pyvista_2d_von_mises.png", show=False, window_size=(1100, 450))
    plot_deformed(m2, U2, scale="auto", screenshot="fea_07_pyvista_2d_deformed.png", show=False,
                  window_size=(1100, 450))
    g = to_pyvista(m2)
    print(f"2-D: grid {g.n_points} points / {g.n_cells} cells; tip deflection {tip2:.6e} m, "
          f"Euler-Bernoulli {ref2:.6e} m (ratio {tip2 / ref2:.3f})")
    save_vtk(m2, "fea_07_cantilever_2d.vtu", U=U2, point_data={"von_mises": vm.component("von_mises")})

    m3, s3, U3, tip3, ref3 = cantilever_3d()
    plot_deformed(m3, U3, scale="auto", title="3-D cantilever (Hex8): displacement magnitude",
                  screenshot="fea_07_pyvista_3d_deformed.png", show=False, window_size=(1000, 650))
    g3 = to_pyvista(m3)
    print(f"3-D: grid {g3.n_points} points / {g3.n_cells} cells; tip deflection {tip3:.6e} m, "
          f"Euler-Bernoulli {ref3:.6e} m (ratio {tip3 / ref3:.3f})")
    save_vtk(m3, "fea_07_cantilever_3d.vtu", U=U3)
    print("Saved fea_07_pyvista_2d_von_mises.png, fea_07_pyvista_2d_deformed.png, fea_07_pyvista_3d_deformed.png "
          "and two .vtu files")


if __name__ == "__main__":
    main()

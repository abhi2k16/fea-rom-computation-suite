# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_08_pyvista_animation_sections_rom.py -- Example Gallery: animations, slices and full-versus-reduced comparison
plots with PyVista (fea_engine.visualization; needs ``pip install fea_engine[viz]``, which includes imageio
for GIF output).

  1. Transient: a 2-D steel cantilever under a suddenly applied tip load (Newmark implicit,
     ``solve_transient_implicit``), animated as a GIF coloured by von Mises stress (evaluated per frame).
  2. Mode shape: the first bending mode oscillating, as a GIF.
  3. Sections: a 3-D Hex8 cantilever's von Mises stress, shown with three slices across the span and with a
     threshold that keeps only the highly stressed cells.
  4. Full-order vs reduced-order: a POD basis built from static load cases, used to reconstruct a held-out
     solution at two ranks; each is shown next to the full-order field with the pointwise error. The POD here is a
     plain numpy SVD so the example needs only fea_engine; ``rom_engine.pod.PodBasis`` computes the same basis.

Validation printed to stdout (all measured here): the first natural frequency against the Euler-Bernoulli value, the
peak tip deflection of the step response against the static deflection (a linear step response overshoots the static
value by up to a factor of 2; lightly damped here, 1 %), and the ROM errors by rank, computed two ways (from the plotted metrics
and directly from the vectors).

On a headless Linux machine run it under a virtual display:  xvfb-run -a python fea_08_pyvista_animation_sections_rom.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import (RayleighDamping, FESystem, Material, D_plane_stress, D_solid3d, Quad4PlaneStress, Hex8Solid3D, mesh)
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine.visualization import (animate_history, animate_mode, compare_fields, plot_slice, plot_threshold)

E, NU, RHO = 210e9, 0.3, 7800.0
MAT = Material(E=E, nu=NU, rho=RHO)


def cantilever_2d(L=4.0, H=0.4, t=0.1, nx=40, ny=6):
    m = mesh.rectangle_mesh(L, H, nx, ny)
    s = FESystem(m, Quad4PlaneStress(), thickness=t)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=t)
    s.assemble_mass(RHO * np.eye(2), thickness=t)
    for n in m.nodes_on_line(0, 0.0):
        s.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    return m, s, tip, (L, H, t)


def transient_and_mode():
    m, s, tip, (L, H, t) = cantilever_2d()
    freq, shapes = s.solve_modal(n_modes=3)
    f_eb = (1.8751 ** 2 / (2 * np.pi * L ** 2)) * np.sqrt(E * (H ** 3 / 12.0) / (RHO * H))
    print(f"mode 1: {freq[0]:.3f} Hz (Euler-Bernoulli {f_eb:.3f} Hz, ratio {freq[0] / f_eb:.3f})")

    animate_mode(m, shapes[:, 0], "fea_08_mode1.gif", n_frames=24, freq=freq[0], fps=15)

    P = -2.0e4
    s.F[:] = 0.0
    s.F[2 * tip + 1] = P / len(tip)
    u_static = s.solve_static()[2 * tip + 1].mean()
    s.F[:] = 0.0
    T1 = 1.0 / freq[0]
    s.assemble_damping(RayleighDamping.calibrate(2 * np.pi * freq[0], 2 * np.pi * freq[1], 0.01))
    pattern = LoadPattern(node_ids=tip, dof_index=1)
    load = TimeHistoryLoad(pattern, time_fn=lambda tt: P)
    t_hist, U = s.solve_transient_implicit(load, 2.0 * T1, T1 / 60.0)
    tip_u = U[:, 2 * tip + 1].mean(axis=1)
    print(f"step response: peak tip deflection / static = {tip_u.min() / u_static:.3f} "
          f"(a 1 % damped linear step response stays below 2)")
    out = animate_history(m, U, "fea_08_transient_von_mises.gif", times=t_hist, n_frames=60, fps=15,
                          field=lambda W: s.von_mises(W), component="von_mises",
                          title="step load: von Mises stress")
    print(f"transient animation: {out['n_frames']} frames, colour limits {out['clim'][0]:.3g}..{out['clim'][1]:.3g} Pa, "
          f"deformation x{out['scale']:.3g}")


def sections_3d(L=4.0, H=1.0, B=1.0, P=-2.0e6, nx=16, ny=4, nz=4):
    m = mesh.box_mesh(L, H, B, nx, ny, nz)
    s = FESystem(m, Hex8Solid3D())
    s.assemble_stiffness(D_solid3d(MAT))
    for n in np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0]:
        s.fix_dofs([n], [0, 1, 2])
    tip = np.where(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    s.F[3 * tip + 1] = P / len(tip)
    U = s.solve_static()
    vm = s.von_mises(U)
    peak = vm.component("von_mises").max()
    plot_slice(m, vm, component="von_mises", normal="x", n_slices=3, U=U, scale="auto",
               title="3-D cantilever: von Mises on three cross-sections", screenshot="fea_08_slices.png", show=False)
    plot_threshold(m, vm, (0.5 * peak, None), component="von_mises", U=U, scale="auto",
                   title="cells with von Mises >= 50 % of the peak", screenshot="fea_08_threshold.png", show=False)
    print(f"3-D: peak nodal von Mises {peak:.4g} Pa; slices and threshold saved")


def rom_comparison(n_snapshots=8, ranks=(1, 4)):
    m, s, tip, _ = cantilever_2d(nx=40, ny=6)
    rng = np.random.default_rng(0)
    n = len(m.nodes)

    def solve(F):
        s.F[:] = 0.0
        s.F[:] = F
        return np.asarray(s.solve_static())

    snaps = np.array([solve(np.where(np.arange(2 * n) % 2 == 1, rng.standard_normal(2 * n), 0.0) * 1e3)
                      for _ in range(n_snapshots)]).T
    Ub, sv, _ = np.linalg.svd(snaps, full_matrices=False)                   # POD basis = left singular vectors
    F_test = np.zeros(2 * n)
    F_test[2 * tip + 1] = -2.0e4 / len(tip)
    U = solve(F_test)
    for r in ranks:
        V = Ub[:, :r]
        Ur = V @ (V.T @ U)
        _, met = compare_fields(m, U, Ur, screenshot=f"fea_08_rom_rank{r}.png", show=False,
                                title=f"POD rank {r} of {n_snapshots} snapshots")
        direct = np.linalg.norm(U - Ur) / np.linalg.norm(U)
        print(f"ROM rank {r}: plotted rel L2 {met['rel_l2_displacement']:.4e}, direct {direct:.4e}, "
              f"max |error| {met['max_abs_error']:.3e} m")


def main():
    transient_and_mode()
    sections_3d()
    rom_comparison()


if __name__ == "__main__":
    main()

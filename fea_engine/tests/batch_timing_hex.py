# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batch_timing_hex.py -- timing report for the total-Lagrangian hexahedra (a script, not a test; speed is never asserted).

For each element class and mesh it reports, with element count, DOFs, Gauss rule, repeats (best of) and what is included:
  kernel only     per-element reference loop vs the batched kernel on the same coordinates and displacements (no scatter);
  system assembly FESystem loop route vs vectorized=True (dense scatter) vs vectorized=True, sparse=True (CSR);
  solve           dense np.linalg.solve vs SuperLU on the free block (only when the dense block fits, <= 6000 free DOFs).
Agreement between the routes is printed next to every ratio. A 3-D hexahedron is compute-bound, so expect much smaller
kernel ratios than for the 2-D quadrilateral (tests/batch_timing.py); the sparse path matters more at larger sizes.

    python tests/batch_timing_hex.py [--repeats 3] [--hex8 12 4 4] [--hex20 8 3 3]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse
import os
import platform
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hex_meshes
from fea_engine import FESystem, Material, mesh as meshlib
from fea_engine.material import D_solid3d
from fea_engine.elements import Hex8SolidTL, Hex20SolidTL

D = D_solid3d(Material(E=1.0e3, nu=0.3))


def best_of(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def run(cls, dims, repeats):
    m = meshlib.box_mesh(4.0, 1.0, 1.0, *dims)
    if cls is Hex20SolidTL:
        m = hex_meshes.hex20_from_hex8(m)
    el = cls()
    fes = FESystem(m, el, sparse=False)
    ne, n_dof = len(m.elements), fes.n_dof
    fes.fix_dofs(np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    u = np.random.default_rng(0).normal(size=n_dof) * 0.02
    print(f"\n{cls.__name__} box {dims}: {ne} elements, {n_dof} DOFs, {el.gauss_order}^3 = {el.gauss_order ** 3} Gauss "
          f"points per element, best of {repeats}")
    X = m.nodes[m.elements]
    dofs = bk_dofs = (3 * m.elements[:, :, None] + np.arange(3)).reshape(ne, -1)
    ue = u[dofs]
    geo = el._reference_geometry_batch(X)
    t_lf = best_of(lambda: [el.internal_force(x, w, D) for x, w in zip(X, ue)], repeats)
    t_lk = best_of(lambda: [el.tangent_stiffness(x, w, D) for x, w in zip(X, ue)], repeats)
    t_bf = best_of(lambda: el.internal_force_batch(X, ue, D, geometry=geo), repeats)
    t_bk = best_of(lambda: el.tangent_stiffness_batch(X, ue, D, geometry=geo), repeats)
    t_b1f = best_of(lambda: [el.internal_force_batch(x[None], w[None], D) for x, w in zip(X, ue)], repeats)
    t_b1k = best_of(lambda: [el.tangent_stiffness_batch(x[None], w[None], D) for x, w in zip(X, ue)], repeats)
    f_ref = np.array([el.internal_force(x, w, D) for x, w in zip(X, ue)])
    k_ref = np.array([el.tangent_stiffness(x, w, D) for x, w in zip(X, ue)])
    ef = np.abs(el.internal_force_batch(X, ue, D) - f_ref).max() / np.abs(f_ref).max()
    ek = np.abs(el.tangent_stiffness_batch(X, ue, D) - k_ref).max() / np.abs(k_ref).max()
    print("  kernel only (no scatter; geometry precomputed for the batched call)")
    print(f"    internal force  reference loop {1e3 * t_lf:9.2f} ms   batched {1e3 * t_bf:8.2f} ms   x{t_lf / t_bf:6.1f}"
          f"   (batch of one per element: {1e3 * t_b1f:8.2f} ms)   rel diff {ef:.1e}")
    print(f"    tangent         reference loop {1e3 * t_lk:9.2f} ms   batched {1e3 * t_bk:8.2f} ms   x{t_lk / t_bk:6.1f}"
          f"   (batch of one per element: {1e3 * t_b1k:8.2f} ms)   rel diff {ek:.1e}")
    fes.assemble_internal_force(u, D, vectorized=True)
    fes.assemble_tangent_stiffness(u, D, vectorized=True, sparse=True)                  # caches built, not timed
    t_f = best_of(lambda: fes.assemble_internal_force(u, D), repeats)
    t_fv = best_of(lambda: fes.assemble_internal_force(u, D, vectorized=True), repeats)
    print("  FESystem assembly (geometry and sparsity pattern caches warm)")
    print(f"    internal force  loop {1e3 * t_f:9.2f} ms   vectorized {1e3 * t_fv:8.2f} ms   x{t_f / t_fv:6.1f}")
    dense_ok = n_dof <= 6000
    if dense_ok:
        t_k = best_of(lambda: fes.assemble_tangent_stiffness(u, D), repeats)
        t_kv = best_of(lambda: fes.assemble_tangent_stiffness(u, D, vectorized=True), repeats)
    t_ks = best_of(lambda: fes.assemble_tangent_stiffness(u, D, vectorized=True, sparse=True), repeats)
    if dense_ok:
        print(f"    tangent (dense) loop {1e3 * t_k:9.2f} ms   vectorized {1e3 * t_kv:8.2f} ms   x{t_k / t_kv:6.1f}")
    print(f"    tangent (CSR)   vectorized {1e3 * t_ks:8.2f} ms" + (f"   x{t_kv / t_ks:5.1f} vs vectorized dense" if dense_ok else
                                                                      "   (dense route skipped: dense block too large)"))
    free = fes.free_dofs
    Ks = fes.assemble_tangent_stiffness(u, D, vectorized=True, sparse=True).tocsr()[free][:, free]
    import scipy.sparse.linalg as spla
    rhs = np.ones(len(free))
    t_sl = best_of(lambda: spla.splu(Ks.tocsc()).solve(rhs), repeats)
    if dense_ok:
        Kd = fes.assemble_tangent_stiffness(u, D)[np.ix_(free, free)]
        t_ds = best_of(lambda: np.linalg.solve(Kd, rhs), repeats)
        diff = np.max(np.abs(spla.splu(Ks.tocsc()).solve(rhs) - np.linalg.solve(Kd, rhs))) / np.max(np.abs(np.linalg.solve(Kd, rhs)))
        print(f"  solve ({len(free)} free DOFs): dense {1e3 * t_ds:9.2f} ms   sparse LU {1e3 * t_sl:8.2f} ms   "
              f"x{t_ds / t_sl:6.1f}   solution rel diff {diff:.1e}")
    else:
        print(f"  solve ({len(free)} free DOFs): sparse LU {1e3 * t_sl:8.2f} ms (dense solve skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--hex8", type=int, nargs=3, default=[12, 4, 4])
    ap.add_argument("--hex20", type=int, nargs=3, default=[8, 3, 3])
    a = ap.parse_args()
    print(f"machine: {platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, "
          f"cores {os.cpu_count()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    run(Hex8SolidTL, a.hex8, a.repeats)
    run(Hex20SolidTL, a.hex20, a.repeats)


if __name__ == "__main__":
    main()

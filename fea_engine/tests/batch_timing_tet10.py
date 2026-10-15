# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batch_timing_tet10.py -- timing report for the batched Tet10SolidTL (a script, not a test; speed is never asserted).

Structured Tet10 box (nx x ny x nz hexahedral cells, 6 tetrahedra each), clamped at x = 0. Reports, with element count, DOFs, repeats
(best of) and the maximum relative difference: per-point reference loop vs batched kernel on the same coordinates (no scatter),
FESystem loop vs vectorized=True force, dense and CSR tangent, and dense vs sparse LU solve (dense only when the free block <= 6000 DOFs).

    python tests/batch_timing_tet10.py [--repeats 3] [--box 6 2 2]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse, os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_loop_baselines as gb
from fea_engine import FESystem, Material, Mesh
from fea_engine.elements import Tet10SolidTL


def best_of(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--box", type=int, nargs=3, default=[6, 2, 2])
    a = ap.parse_args()
    nodes, elems = gb.tet10_box(*a.box, L=float(a.box[0]), W=float(a.box[1]), T=float(a.box[2]))
    mat = Material(E=gb.E_MOD, nu=gb.NU)
    el = Tet10SolidTL()
    fes = FESystem(Mesh(nodes, elems, 3), el, sparse=False)
    fes.fix_dofs(np.where(np.abs(nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    u = np.random.default_rng(0).normal(size=fes.n_dof) * 0.02
    print(f"machine: {platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, cores {os.cpu_count()}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"Tet10SolidTL box {a.box}: {len(elems)} elements, {fes.n_dof} DOFs, 4 Gauss points, best of {a.repeats}")
    X, dofs = nodes[elems], (3 * elems[:, :, None] + np.arange(3)).reshape(len(elems), -1)
    ue = u[dofs]
    geo = el._reference_geometry_batch(X)
    tlf = best_of(lambda: [el.internal_force(x, w, mat) for x, w in zip(X, ue)], 1)
    tlk = best_of(lambda: [el.tangent_stiffness(x, w, mat) for x, w in zip(X, ue)], 1)
    tbf = best_of(lambda: el.internal_force_batch(X, ue, mat, geometry=geo), a.repeats)
    tbk = best_of(lambda: el.tangent_stiffness_batch(X, ue, mat, geometry=geo), a.repeats)
    fr = np.array([el.internal_force(x, w, mat) for x, w in zip(X, ue)]); kr = np.array([el.tangent_stiffness(x, w, mat) for x, w in zip(X, ue)])
    ef = np.abs(el.internal_force_batch(X, ue, mat) - fr).max() / np.abs(fr).max()
    ek = np.abs(el.tangent_stiffness_batch(X, ue, mat) - kr).max() / np.abs(kr).max()
    print("  kernel only (no scatter; geometry precomputed for the batched call; reference loop = 1 run)")
    print(f"    internal force  reference loop {1e3*tlf:9.2f} ms   batched {1e3*tbf:8.2f} ms   x{tlf/tbf:6.1f}   rel diff {ef:.1e}")
    print(f"    tangent         reference loop {1e3*tlk:9.2f} ms   batched {1e3*tbk:8.2f} ms   x{tlk/tbk:6.1f}   rel diff {ek:.1e}")
    fes.assemble_internal_force(u, mat, vectorized=True); fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True)
    tf = best_of(lambda: fes.assemble_internal_force(u, mat), 1); tfv = best_of(lambda: fes.assemble_internal_force(u, mat, vectorized=True), a.repeats)
    print(f"  FESystem force   loop {1e3*tf:9.2f} ms (1 run)   vectorized {1e3*tfv:8.2f} ms   x{tf/tfv:6.1f}")
    free = fes.free_dofs
    dense_ok = len(free) <= 6000
    tks = best_of(lambda: fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True), a.repeats)
    if dense_ok:
        tk = best_of(lambda: fes.assemble_tangent_stiffness(u, mat), 1); tkv = best_of(lambda: fes.assemble_tangent_stiffness(u, mat, vectorized=True), a.repeats)
        print(f"  FESystem dense tangent  loop {1e3*tk:9.2f} ms (1 run)   vectorized {1e3*tkv:8.2f} ms   x{tk/tkv:6.1f}")
    print(f"  FESystem CSR tangent    vectorized {1e3*tks:8.2f} ms")
    import scipy.sparse.linalg as spla
    Ks = fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True).tocsr()[free][:, free]
    rhs = np.ones(len(free))
    tsl = best_of(lambda: spla.splu(Ks.tocsc()).solve(rhs), a.repeats)
    if dense_ok:
        Kd = fes.assemble_tangent_stiffness(u, mat)[np.ix_(free, free)]
        tds = best_of(lambda: np.linalg.solve(Kd, rhs), a.repeats)
        print(f"  solve ({len(free)} free DOFs): dense {1e3*tds:9.2f} ms   sparse LU {1e3*tsl:8.2f} ms   x{tds/tsl:6.1f}")
    else:
        print(f"  solve ({len(free)} free DOFs): sparse LU {1e3*tsl:8.2f} ms (dense skipped)")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batch_timing.py -- timing report for nonlinear assembly and solve (a script, not a test; speed is never asserted).

Every line states what was timed so ratios cannot be misread: model, element count, DOFs, operation, repeats (best
of), and the machine's BLAS thread setting. Today it times the per-element LOOP assembly and the dense solve, i.e. the
baseline that batched/sparse paths will be compared against. When a batched path exists, add its callables to
``CASES`` and the same script prints loop and batched side by side.

    python tests/batch_timing.py [--repeats 5] [--plate 18 18]

Notes on reading the numbers: single machine, wall-clock via time.perf_counter, best of N repeats (reduces scheduler
noise, hides cold-cache cost). Operation "tangent" includes building the dense matrix; "solve" is
np.linalg.solve on the free-DOF block.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse
import os
import platform
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL
import golden_loop_baselines as gb


def best_of(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def quad_plate_case(nx, ny):
    nodes, elems, _, _ = gb.quad_plate(nx, ny, L=nx / 2.0, H=ny / 4.0)
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    mat = {"q": D_plane_stress(Material(E=gb.E_MOD, nu=gb.NU))}
    u = np.random.default_rng(0).normal(size=fes.n_dof) * 0.02
    return fes, mat, u, len(elems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--plate", type=int, nargs=2, default=[18, 18])
    a = ap.parse_args()
    print(f"machine: {platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, "
          f"cores {os.cpu_count()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    fes, mat, u, ne = quad_plate_case(*a.plate)
    kw = dict(thickness=0.5)
    print(f"Quad4PlaneStressTL plate {a.plate[0]}x{a.plate[1]}: {ne} elements, {fes.n_dof} DOFs, best of {a.repeats}")
    t_f = best_of(lambda: fes.assemble_internal_force(u, mat, **kw), a.repeats)
    t_k = best_of(lambda: fes.assemble_tangent_stiffness(u, mat, **kw), a.repeats)
    K = fes.assemble_tangent_stiffness(u, mat, **kw)
    free = fes.free_dofs
    Kff, rhs = K[np.ix_(free, free)], np.ones(len(free))
    t_s = best_of(lambda: np.linalg.solve(Kff, rhs), a.repeats)
    print(f"  loop internal force        {1e3 * t_f:9.2f} ms")
    print(f"  loop tangent (dense)       {1e3 * t_k:9.2f} ms")
    print(f"  dense solve (free block)   {1e3 * t_s:9.2f} ms   ({len(free)} free DOFs)")
    fes.assemble_internal_force(u, mat, vectorized=True, **kw)            # builds the geometry cache (not timed below)
    t_fb = best_of(lambda: fes.assemble_internal_force(u, mat, vectorized=True, **kw), a.repeats)
    t_kb = best_of(lambda: fes.assemble_tangent_stiffness(u, mat, vectorized=True, **kw), a.repeats)
    err_f = np.max(np.abs(fes.assemble_internal_force(u, mat, vectorized=True, **kw) - fes.assemble_internal_force(u, mat, **kw)))
    err_k = np.max(np.abs(fes.assemble_tangent_stiffness(u, mat, vectorized=True, **kw) - K))
    print(f"  batched internal force     {1e3 * t_fb:9.2f} ms   x{t_f / t_fb:6.1f} faster than loop "
          f"(max abs diff {err_f:.1e}, geometry cache warm)")
    print(f"  batched tangent (dense)    {1e3 * t_kb:9.2f} ms   x{t_k / t_kb:6.1f} faster than loop "
          f"(max abs diff {err_k:.1e}, includes dense scatter)")
    fes.clear_block_geometry()
    t_cold = best_of(lambda: (fes.clear_block_geometry(), fes.assemble_internal_force(u, mat, vectorized=True, **kw)),
                     a.repeats)
    print(f"  batched force, cold cache  {1e3 * t_cold:9.2f} ms   (geometry rebuilt every call)")
    # sparse path: tangent as CSR, free block, SuperLU factor + solve
    import scipy.sparse.linalg as spla
    fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True, **kw)          # builds the pattern (not timed)
    t_ks = best_of(lambda: fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True, **kw), a.repeats)
    Ks = fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True, **kw)
    err_s = abs(Ks - K).max() if Ks.shape[0] < 6000 else float("nan")
    print(f"  batched tangent (CSR)      {1e3 * t_ks:9.2f} ms   x{t_kb / t_ks:6.1f} vs batched dense "
          f"(max abs diff {err_s:.1e}, pattern cache warm, nnz {Ks.nnz})")
    Kfs = Ks.tocsr()[free][:, free]
    t_sl = best_of(lambda: spla.splu(Kfs.tocsc()).solve(rhs), a.repeats)
    xs = spla.splu(Kfs.tocsc()).solve(rhs)
    xd = np.linalg.solve(Kff, rhs)
    print(f"  sparse LU factor+solve     {1e3 * t_sl:9.2f} ms   x{t_s / t_sl:6.1f} vs dense solve "
          f"(solution relative difference {np.max(np.abs(xs - xd)) / np.max(np.abs(xd)):.1e})")


if __name__ == "__main__":
    main()

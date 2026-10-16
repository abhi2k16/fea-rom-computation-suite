# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
sparse_contact_timing.py -- timing and memory report for solve_contact_static_sparse (a script, not a test; speed is never asserted).

Membrane obstacle problem (Q4 Laplace stiffness, uniform grid, unknowns N = (ne-1)^2, obstacle = the plateau at -0.3 with a few
hundred active constraints). For each size the routes "reduced" and "saddle" are timed; the dense route (`solve_contact_static`,
which forms K^-1 B^T and B K^-1 B^T densely) only up to --dense-max unknowns, because it needs about 8 (N^2 + N^2) bytes and
cubic time. Prints the difference between the routes where both ran, the iteration count and the active count.

    python tests/sparse_contact_timing.py [--ne 30 50 70 90 200] [--dense-max 5000] [--repeats 2]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse, os, platform, sys, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fea_engine import solve_contact_static, solve_contact_static_sparse
from test_contact_solvers_sparse import membrane


def best_of(fn, n):
    t, out = [], None
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); t.append(time.perf_counter() - t0)
    return min(t), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, nargs="+", default=[30, 50, 70, 90, 200])
    ap.add_argument("--dense-max", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    print(f"{platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, cpus {os.cpu_count()}")
    print(f"{'N':>7} {'active':>6} | {'reduced s':>10} {'it':>3} | {'saddle s':>9} {'it':>3} | {'dense s':>9} {'dense GB':>8} | max diff vs dense / reduced vs saddle")
    for ne in a.ne:
        K, f, X, Y = membrane(ne)
        N = K.shape[0]
        g = np.full(N, -0.3)
        B = sp.identity(N, format="csr")
        tr, r = best_of(lambda: solve_contact_static_sparse(K, f, B, g, route="reduced"), a.repeats)
        ts, s = best_of(lambda: solve_contact_static_sparse(K, f, B, g, route="saddle"), a.repeats)
        gb = 8.0 * 2 * N * N / 1e9
        if N <= a.dense_max:
            td, d = best_of(lambda: solve_contact_static(K.toarray(), f, B, g), 1)
            dd = max(np.abs(r.u - d.u).max(), np.abs(r.lam - d.lam).max())
            dtxt, dd_txt = f"{td:9.3f}", f"{dd:.1e}"
        else:
            dtxt, dd_txt = f"{'not run':>9}", "-"
        rs = max(np.abs(r.u - s.u).max(), np.abs(r.lam - s.lam).max())
        print(f"{N:7d} {int((r.lam > 1e-12).sum()):6d} | {tr:10.3f} {r.iterations:3d} | {ts:9.3f} {s.iterations:3d} | {dtxt} {gb:8.2f} | {dd_txt} / {rs:.1e}", flush=True)


if __name__ == "__main__":
    main()

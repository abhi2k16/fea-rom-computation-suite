# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_indicator_timing.py -- timing report for ContactConditionIndicator against the reference contact_condition_indicator (a script,
not a test; speed is never asserted).

Obstacle membrane (n x n interior nodes, B = I, weights (1, 1, 1): the a2 and a3 terms need the force estimate). The reference densifies B
and runs a dense least-squares solve on every call (cubic in the number of constraints), so it is only run up to --reference-max
constraints; the class is run at every size, plus a sparse general B (3 nonzeros per row, half as many rows as unknowns) that takes the
normal-equations route. Prints the time per call (best of --repeats) and the relative difference where both ran.

    python tests/contact_indicator_timing.py [--n 20 30 40 50 100 200] [--reference-max 2500] [--repeats 3]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse, os, platform, sys, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_fixtures import ObstacleMembrane
from rom_engine import ContactConditionIndicator, contact_condition_indicator


def best_of(fn, n):
    t, out = [], None
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); t.append(time.perf_counter() - t0)
    return min(t), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20, 30, 40, 50, 100, 200])
    ap.add_argument("--reference-max", type=int, default=2500)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    print(f"{platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, cpus {os.cpu_count()}")
    print(f"{'constraints':>11} {'structure':>18} | {'reference s':>11} | {'class 1st s':>11} {'class next s':>12} {'method':>16} | rel diff")
    rng = np.random.default_rng(0)
    for n in a.n:
        m = ObstacleMembrane(n=n)
        N = m.n_dof
        u = 0.9 * np.linalg.solve(m.K.toarray(), m.f) if N <= 2500 else 0.9 * rng.standard_normal(N) * 0.01
        mm = N // 2
        rows = np.repeat(np.arange(mm), 3)
        cols = np.stack([2 * np.arange(mm), 2 * np.arange(mm) + 1, (2 * np.arange(mm) + 7) % N], axis=1).ravel()
        Bs = sp.csr_matrix((np.ones(3 * mm), (rows, cols)), shape=(mm, N))
        for label, B in (("identity", m.B), ("sparse, m = n/2", Bs)):
            c = np.full(B.shape[0], -0.1)
            ind = ContactConditionIndicator(m.K, B, m.f, (1, 1, 1))
            t0 = time.perf_counter(); v1 = ind(u, c); first = time.perf_counter() - t0
            tn, v2 = best_of(lambda: ind(u, c), a.repeats)
            if B.shape[0] <= a.reference_max:
                tr, vr = best_of(lambda: contact_condition_indicator(m.K, B, m.f, c, u, (1, 1, 1)), 1)
                rtxt, dtxt = f"{tr:11.3f}", f"{abs(v1 - vr) / max(abs(vr), 1e-300):.1e}"
            else:
                rtxt, dtxt = f"{'not run':>11}", "-"
            print(f"{B.shape[0]:11d} {label:>18} | {rtxt} | {first:11.4f} {tn:12.5f} {str(ind.method):>16} | {dtxt}", flush=True)


if __name__ == "__main__":
    main()

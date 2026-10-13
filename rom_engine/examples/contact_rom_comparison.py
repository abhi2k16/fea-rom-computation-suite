# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_rom_comparison.py -- accuracy, positivity and cost of reduced unilateral-contact models on the parametrised
obstacle membrane (Balajewicz et al. 2015 model problem, much smaller mesh).

Both reduced models share the same primal POD basis and the same dual size; they differ only in the dual basis for the
contact forces: leading SVD vectors (no sign constraint) versus an NMF basis (non-negative by construction).
Reported: displacement, gap and contact-force errors (max-normalised), the complementarity violation, agreement of
the active set, online time per query, offline time and the effective speed-up with offline cost charged against
`--queries` online queries. Printed as JSON and written to results/contact_rom_<n>.json.

    python examples/contact_rom_comparison.py 24 12 --queries=200

Arguments: n (interior nodes per side), basis size p (primal = dual). Wall-clock numbers depend on the machine and the
BLAS build: run it where you will use it. The membrane is a model problem, not a structural benchmark.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tests"), os.path.join(HERE, "..", "src")]      # fixtures use fea_engine
from contact_fixtures import ObstacleMembrane
from rom_engine import (nonneg_dual_basis, svd_dual_basis, ContactROM, compare_contact_models,
                        format_contact_comparison)


def main(n=24, p=12, queries=200):
    m = ObstacleMembrane(n=n)
    train = [(a, b) for a in np.linspace(0.3, 0.6, 6) for b in np.linspace(0.0, 0.1, 5)]
    tests = [(0.33, 0.02), (0.37, 0.03), (0.44, 0.05), (0.52, 0.08), (0.57, 0.09)]
    t0 = time.perf_counter()
    Xu, Xl = m.training_set(train)
    t_snap = time.perf_counter() - t0

    def fom(g):
        u, lam, c = m.solve(g)
        return u, lam, m.B @ u - c

    t0 = time.perf_counter()
    U = np.linalg.svd(Xu, full_matrices=False)[0][:, :p]
    t_pod = time.perf_counter() - t0
    t0 = time.perf_counter()
    U_svd = svd_dual_basis(Xl, p)
    t_svd = time.perf_counter() - t0
    t0 = time.perf_counter()
    nmf = nonneg_dual_basis(Xl, p)
    t_nmf = time.perf_counter() - t0
    variants = {}
    for name, dual, t_dual in (("svd dual", U_svd, t_svd), ("nmf dual", nmf.U, t_nmf)):
        t0 = time.perf_counter()
        rom = ContactROM(m.K, m.B, U, dual)
        t_build = time.perf_counter() - t0

        def predict(g, rom=rom):
            r = rom.solve(m.f, m.obstacle(g))
            return r.u, r.lam, r.gap
        variants[name] = dict(predict=predict, size=rom.sizes, offline_time=t_snap + t_pod + t_dual + t_build)
    table = compare_contact_models(fom, variants, tests, n_queries=queries, repeats=3)
    print(format_contact_comparison(table))
    out = dict(n=n, p=p, n_dof=m.n_dof, n_train=len(train), n_test=len(tests), queries=queries,
               t_snapshots=t_snap, t_pod=t_pod, nmf_relative_fit_error=nmf.rel_error, table=table)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"contact_rom_{n}.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kw = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
    main(int(args[0]) if args else 24, int(args[1]) if len(args) > 1 else 12, int(kw.get("queries", 200)))

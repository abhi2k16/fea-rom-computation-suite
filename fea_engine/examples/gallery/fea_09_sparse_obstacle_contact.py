# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_09_sparse_obstacle_contact.py -- Example Gallery: a membrane pushed onto a rigid obstacle (the obstacle problem), assembled with the
scalar element `Quad4Laplace` and solved with `solve_contact_static_sparse`.

Contact convention: K u = f + B^T lam,  B u - c >= 0,  lam >= 0,  lam . (B u - c) = 0.  Here B = I (every node may touch the obstacle).

Panel 1: membrane displacement with the contact set (nodes where lam > 0) outlined.
Panel 2: contact force lam on the membrane.
Panel 3: wall time of the routes "reduced" and "saddle" against the number of unknowns, and of the dense route `solve_contact_static`
         where it fits (the dense route forms K^-1 B^T and B K^-1 B^T; speed is measured here, never asserted).
The script prints the difference between the routes and the iteration counts. Times depend on the machine; the figure is the one
produced by the run named in the figure footer.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os, platform, time
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fea_engine import FESystem, Quad4Laplace, solve_contact_static, solve_contact_static_sparse
from fea_engine import mesh as meshlib

HERE = os.path.dirname(os.path.abspath(__file__))


def membrane_system(ne, load=-10.0):
    """Unit-square membrane, clamped edge, uniform downward load; returns (K_free, f_free, X_free) with Quad4Laplace stiffness."""
    m = meshlib.rectangle_mesh(1.0, 1.0, ne, ne)
    fes = FESystem(m, Quad4Laplace(), sparse=True)
    fes.assemble_stiffness(1.0)
    fes.assemble_mass(1.0)
    X = m.nodes
    edge = np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > 1 - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > 1 - 1e-9))
    fes.fix_dofs(edge, ["u"])
    free = np.asarray(fes.free_dofs)
    K = sp.csr_matrix(fes.K)[free][:, free]
    M = sp.csr_matrix(fes.M)[free][:, free]
    f = load * (M @ np.ones(len(free)))                 # consistent load for a uniform pressure
    return K, f, X[free], m, free


def obstacle(X):
    """Flat floor at -0.30 with two bumps (the shape used in the validation project)."""
    x, y = X[:, 0], X[:, 1]
    return (-0.30 + 0.12 * np.exp(-60 * ((x - 0.4) ** 2 + (y - 0.5) ** 2)) + 0.25 * np.exp(-60 * ((x - 0.7) ** 2 + (y - 0.5) ** 2)))


def best_of(fn, n):
    t, out = [], None
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); t.append(time.perf_counter() - t0)
    return min(t), out


def main():
    # ---- the picture: one mesh
    K, f, X, m, free = membrane_system(40)
    N = K.shape[0]
    c = obstacle(X)
    B = sp.identity(N, format="csr")
    r = solve_contact_static_sparse(K, f, B, c, route="reduced")
    s = solve_contact_static_sparse(K, f, B, c, route="saddle")
    d = solve_contact_static(K.toarray(), f, B, c)
    n_act = int((r.lam > 1e-10).sum())
    print(f"mesh 40x40: N = {N} unknowns, {n_act} active constraints, reduced {r.iterations} iterations, saddle {s.iterations}")
    print(f"max |u|, |lam| difference reduced vs saddle {max(np.abs(r.u - s.u).max(), np.abs(r.lam - s.lam).max()):.1e}; "
          f"reduced vs dense {max(np.abs(r.u - d.u).max(), np.abs(r.lam - d.lam).max()):.1e}")
    print(f"min gap {r.gap.min():.1e}, min lam {r.lam.min():.1e}, max |lam*gap| {np.abs(r.lam * r.gap).max():.1e}")

    # ---- timing sweep
    rows = []
    for ne in (20, 30, 40, 50, 60, 80, 100):
        Kn, fn, Xn, _, _ = membrane_system(ne)
        Nn = Kn.shape[0]
        cn, Bn = obstacle(Xn), sp.identity(Nn, format="csr")
        tr, rr = best_of(lambda: solve_contact_static_sparse(Kn, fn, Bn, cn, route="reduced"), 2)
        ts, ss = best_of(lambda: solve_contact_static_sparse(Kn, fn, Bn, cn, route="saddle"), 2)
        if Nn <= 4000:
            td, dd = best_of(lambda: solve_contact_static(Kn.toarray(), fn, Bn, cn), 1)
            diff = max(np.abs(rr.u - dd.u).max(), np.abs(rr.lam - dd.lam).max())
        else:
            td, diff = np.nan, np.nan
        rows.append((Nn, int((rr.lam > 1e-10).sum()), tr, ts, td, rr.iterations, ss.iterations, diff))
        print(f"N {Nn:6d} active {rows[-1][1]:5d} reduced {tr:7.3f}s ({rr.iterations} it) saddle {ts:7.3f}s ({ss.iterations} it) "
              f"dense {td:8.3f}s  diff to dense {diff:.1e}")
    R = np.array(rows)

    # ---- figure
    ne = 40
    ngrid = ne - 1
    U = r.u.reshape(ngrid, ngrid)
    L = r.lam.reshape(ngrid, ngrid)
    xs = X[:, 0].reshape(ngrid, ngrid)
    ys = X[:, 1].reshape(ngrid, ngrid)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    im = ax[0].pcolormesh(xs, ys, U, shading="auto", cmap="viridis")
    ax[0].contour(xs, ys, (L > 1e-10).astype(float), levels=[0.5], colors="red", linewidths=1.6)
    ax[0].set_title(f"displacement u, contact set outlined (red)\n{N} unknowns, {n_act} active")
    plt.colorbar(im, ax=ax[0])
    im = ax[1].pcolormesh(xs, ys, L, shading="auto", cmap="magma")
    ax[1].set_title(r"contact force $\lambda$ (zero off the contact set)")
    plt.colorbar(im, ax=ax[1])
    for a in ax[:2]:
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    ax[2].loglog(R[:, 0], R[:, 2], "o-", label="reduced route")
    ax[2].loglog(R[:, 0], R[:, 3], "s-", label="saddle route")
    ok = ~np.isnan(R[:, 4])
    ax[2].loglog(R[ok, 0], R[ok, 4], "^--", label="dense: solve_contact_static")
    ax[2].set_xlabel("unknowns N"); ax[2].set_ylabel("wall time (s)")
    ax[2].set_title("route cost")
    ax[2].grid(True, which="both", alpha=0.3); ax[2].legend()
    fig.suptitle("Obstacle problem with Quad4Laplace and solve_contact_static_sparse", y=1.02)
    fig.text(0.5, -0.03, f"timing from {platform.platform()}, {os.cpu_count()} CPUs; difference between routes <= "
             f"{np.nanmax(R[:, 7]):.0e} where the dense route ran", ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "fea_09_sparse_obstacle_contact.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

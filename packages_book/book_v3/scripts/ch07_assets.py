# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch07_assets.py -- figures, tables and number macros of Chapter 7 (exact unilateral contact).

Runs the solvers of fea_engine (and nothing else) and writes, next to this script's parent folder (book/):

    figures/ch07_penalty_vs_exact.pdf   penalty penetration and conditioning against the penalty stiffness; the exact solve
    figures/ch07_obstacle.pdf           displacement with the contact set, and the contact force
    figures/ch07_active_trace.pdf       the active set along the primal-dual iterations; iterations against problem size
    figures/ch07_routes_time.pdf        wall time of the routes against the number of unknowns
    figures/ch07_bdf2_order.pdf         error of the two contact time schemes against the time step
    figures/ch07_impact.pdf             bar impact: wall force and energy for both schemes
    tables/ch07_agreement.tex           routes against the dense solvers on six obstacles
    tables/ch07_scaling.tex             the timing sweep
    tables/ch07_order.tex               the order-of-accuracy table
    tables/ch07_impact.tex              the impact energy table
    tables/numbers_ch07.tex             every number quoted in the prose, as a macro

Times are single runs (minimum of two for the sparse routes) on whatever machine runs this script; no number here is asserted
by a test. Run from the repository root:

    python packages_book/book/scripts/ch07_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg as sl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fea_engine import FESystem, Quad4Laplace, solve_contact_static, solve_contact_static_sparse, solve_contact_dynamics
from fea_engine import mesh as meshlib

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
FIG = os.path.join(BOOK, "figures")
TAB = os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(name, value):
    NUM[name] = value


# ------------------------------------------------------------------------------------------------ problem definitions
def membrane_system(ne, load=-10.0):
    """Unit-square membrane with Quad4Laplace, clamped edge, uniform downward pressure: (K, f, X) on the free nodes."""
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
    return K, load * (M @ np.ones(len(free))), X[free]


def bump(X):
    x, y = X[:, 0], X[:, 1]
    return (-0.30 + 0.12 * np.exp(-60 * ((x - 0.4) ** 2 + (y - 0.5) ** 2)) + 0.25 * np.exp(-60 * ((x - 0.7) ** 2 + (y - 0.5) ** 2)))


def six_obstacles(X):
    x, y = X[:, 0], X[:, 1]
    rng = np.random.default_rng(0)
    return {"bump": bump(X), "plateau $-0.5$": np.full(len(x), -0.5), "plateau $-0.3$": np.full(len(x), -0.3),
            "ridge": -0.2 - 0.8 * ((x - 0.5) ** 2 + (y - 0.5) ** 2), "rough": -0.45 + 0.1 * rng.standard_normal(len(x)),
            "tilted": -0.2 - 0.6 * x}


def best_of(fn, n):
    t, out = [], None
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); t.append(time.perf_counter() - t0)
    return min(t), out


def sci(x):
    """LaTeX scientific notation of a float."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------------------------------------ E1 penalty against exact
def penalty_solve(K, f, c, kp):
    n = K.shape[0]
    act = np.zeros(n, bool)
    for it in range(1, 300):
        Kp = K + kp * sp.diags(act.astype(float))
        rhs = f + kp * np.where(act, c, 0.0)
        u = sp.linalg.spsolve(sp.csc_matrix(Kp), rhs)
        new = u < c
        if np.array_equal(new, act):
            return u, Kp, it
        act = new
    return u, Kp, -1


def exp_penalty():
    K, f, X = membrane_system(30)
    N = K.shape[0]
    c = bump(X)
    B = sp.identity(N, format="csr")
    ex = solve_contact_static_sparse(K, f, B, c)
    pen_exact = max(0.0, float((c - ex.u).max()))
    kd = float(K.diagonal().mean())
    ratios = 10.0 ** np.arange(0, 9)
    pen, cnd, rel_force = [], [], []
    for r in ratios:
        u, Kp, it = penalty_solve(K, f, c, r * kd)
        pen.append(max(0.0, float((c - u).max())))
        cnd.append(np.linalg.cond(Kp.toarray()))
        lam_p = r * kd * np.maximum(c - u, 0.0)
        rel_force.append(float(np.abs(lam_p - ex.lam).max() / ex.lam.max()))
    free = np.ones(N, bool)
    free[ex.active] = False
    cond_exact = np.linalg.cond(K.toarray()[np.ix_(free, free)])
    cond_K = np.linalg.cond(K.toarray())
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax[0].loglog(ratios, np.maximum(pen, 1e-18), "o-", label="penalty")
    ax[0].axhline(max(pen_exact, 1e-18), color="k", ls="--", label="exact (Lagrange)")
    ax[0].set_xlabel(r"$k_p/\overline{K_{ii}}$"); ax[0].set_ylabel("largest penetration"); ax[0].legend()
    ax[1].loglog(ratios, cnd, "s-", label=r"penalty: $\kappa(K+k_pP_A)$")
    ax[1].axhline(cond_exact, color="k", ls="--", label=r"exact: $\kappa(K_{FF})$")
    ax[1].set_xlabel(r"$k_p/\overline{K_{ii}}$"); ax[1].set_ylabel("condition number"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch07_penalty_vs_exact.pdf")); plt.close(fig)
    num("ctcPenN", N)
    num("ctcPenExact", sci(pen_exact) if pen_exact > 0 else "$0$ (to rounding)")
    num("ctcPenAtOne", f"{pen[0]:.3f}")
    num("ctcPenAtSix", sci(pen[6]))
    num("ctcPenAtEight", sci(pen[8]))
    num("ctcCondAtEight", sci(cnd[8]))
    num("ctcCondExact", f"{cond_exact:.0f}")
    num("ctcRelForceSix", sci(rel_force[6]))
    num("ctcRelForceOne", f"{rel_force[0]:.2f}")
    num("ctcActiveThirty", int(len(ex.active)))
    print("penalty:", [f"{p:.2e}" for p in pen], "cond", [f"{c_:.1e}" for c_ in cnd], "exact cond", cond_exact)


# ------------------------------------------------------------------------------------------------ large case of the repository timing script
def exp_large():
    """The large case of the text: the same function and problem as fea_engine/tests/sparse_contact_timing.py (plateau at -0.3)."""
    import sys
    tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "fea_engine", "tests")
    sys.path.insert(0, os.path.abspath(tests))
    from test_contact_solvers_sparse import membrane
    ne = 200
    K, f, X, Y = membrane(ne)
    N = K.shape[0]
    g = np.full(N, -0.3)
    B = sp.identity(N, format="csr")
    tr, r = best_of(lambda: solve_contact_static_sparse(K, f, B, g, route="reduced"), 2)
    ts, s = best_of(lambda: solve_contact_static_sparse(K, f, B, g, route="saddle"), 2)
    num("ctcLargeN", f"{N:,}".replace(",", "{,}")); num("ctcLargeActive", f"{int((r.lam > 1e-12).sum()):,}".replace(",", "{,}"))
    num("ctcLargeIt", r.iterations); num("ctcLargeItSaddle", s.iterations)
    num("ctcLargeRed", f"{tr:.1f}"); num("ctcLargeSad", f"{ts:.1f}")
    num("ctcLargeDiff", sci(max(np.abs(r.u - s.u).max(), np.abs(r.lam - s.lam).max())))
    num("ctcLargeDenseGB", f"{8.0 * 2 * N * N / 1e9:.0f}")
    print("large case:", N, int((r.lam > 1e-12).sum()), r.iterations, s.iterations, f"{tr:.2f}", f"{ts:.2f}", flush=True)


# ------------------------------------------------------------------------------------------------ E2 obstacle figure and trace
def active_trace(K, f, B, c, route):
    counts, changed, prev = [], [], np.zeros(K.shape[0] if route == "reduced" else B.shape[0], bool)
    k = 1
    while k < 60:
        r = solve_contact_static_sparse(K, f, B, c, route=route, max_iter=k)
        mask = r.active_mask
        counts.append(int(mask.sum())); changed.append(int((mask != prev).sum())); prev = mask.copy()
        if r.ok:
            return counts, changed, r.iterations
        k += 1
    return counts, changed, -1


def exp_obstacle():
    ne = 40
    K, f, X = membrane_system(ne)
    N = K.shape[0]
    c = bump(X)
    B = sp.identity(N, format="csr")
    r = solve_contact_static_sparse(K, f, B, c, route="reduced")
    s = solve_contact_static_sparse(K, f, B, c, route="saddle")
    g = ne - 1
    xs, ys = X[:, 0].reshape(g, g), X[:, 1].reshape(g, g)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    im = ax[0].pcolormesh(xs, ys, r.u.reshape(g, g), shading="auto", cmap="viridis")
    ax[0].contour(xs, ys, (r.lam > 1e-10).reshape(g, g).astype(float), levels=[0.5], colors="red", linewidths=1.3)
    ax[0].set_title("displacement $u$; contact set (red)"); plt.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].pcolormesh(xs, ys, r.lam.reshape(g, g), shading="auto", cmap="magma")
    ax[1].set_title(r"contact force $\lambda$"); plt.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y"); a.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch07_obstacle.pdf")); plt.close(fig)
    gap_min, lam_min = float(r.gap.min()), float(r.lam.min())
    comp = float(np.abs(r.lam * r.gap).max())
    num("ctcObsN", N); num("ctcObsActive", int((r.lam > 1e-10).sum())); num("ctcObsIt", r.iterations); num("ctcObsItSaddle", s.iterations)
    num("ctcObsGapMin", sci(gap_min) if gap_min != 0 else "$0$"); num("ctcObsLamMin", sci(lam_min) if lam_min != 0 else "$0$")
    num("ctcObsComp", sci(comp) if comp != 0 else "$0$")
    num("ctcObsRS", sci(max(np.abs(r.u - s.u).max(), np.abs(r.lam - s.lam).max())))
    num("ctcObsResid", sci(float(np.linalg.norm(K @ r.u - f - B.T @ r.lam) / np.linalg.norm(f))))   # equilibrium residual, relative to the load
    num("ctcObsOpenGap", sci(float(r.gap[r.lam <= 1e-10].min())))                                    # nearest open node
    # traces
    cr, chg_r, itr = active_trace(K, f, B, c, "reduced")
    cs, chg_s, its = active_trace(K, f, B, c, "saddle")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].plot(range(1, len(cr) + 1), cr, "o-", label="reduced")
    ax[0].plot(range(1, len(cs) + 1), cs, "s--", label="saddle")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("active constraints"); ax[0].set_title(f"active set, N = {N}"); ax[0].legend()
    ax[1].semilogy(range(1, len(chg_r) + 1), np.maximum(chg_r, 0.5), "o-", label="reduced")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("constraints that changed status"); ax[1].set_title("changes per iteration")
    ax[1].set_ylim(0.4, None)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch07_active_trace.pdf")); plt.close(fig)
    num("ctcTraceFirst", cr[0]); num("ctcTraceFinal", cr[-1])
    print("trace reduced", cr, chg_r, "saddle", cs)


# ------------------------------------------------------------------------------------------------ E3 scaling
def exp_scaling():
    rows = []
    for ne in (20, 30, 40, 50, 60, 80, 100):
        K, f, X = membrane_system(ne)
        N = K.shape[0]
        c = bump(X)
        B = sp.identity(N, format="csr")
        tr, r = best_of(lambda: solve_contact_static_sparse(K, f, B, c, route="reduced"), 2)
        ts, s = best_of(lambda: solve_contact_static_sparse(K, f, B, c, route="saddle"), 2)
        if N <= 3500:
            td, d = best_of(lambda: solve_contact_static(K.toarray(), f, B.toarray(), c), 1)
            diff = max(np.abs(r.u - d.u).max(), np.abs(r.lam - d.lam).max())
        else:
            td, diff = np.nan, np.nan
        rows.append((N, int((r.lam > 1e-10).sum()), r.iterations, tr, ts, td, diff))
        print(f"N {N:6d} active {rows[-1][1]:5d} it {r.iterations:3d} reduced {tr:7.3f} saddle {ts:7.3f} dense {td:8.3f} diff {diff:.1e}", flush=True)
    R = np.array(rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.loglog(R[:, 0], R[:, 3], "o-", label="reduced")
    ax.loglog(R[:, 0], R[:, 4], "s-", label="saddle")
    ok = ~np.isnan(R[:, 5])
    ax.loglog(R[ok, 0], R[ok, 5], "^--", label="dense (condensed)")
    ax.set_xlabel("unknowns $N$"); ax.set_ylabel("wall time (s)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch07_routes_time.pdf")); plt.close(fig)
    with open(os.path.join(TAB, "ch07_scaling.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n$N$ & active & iterations & reduced (s) & saddle (s) & dense (s) & difference to dense \\\\\n\\midrule\n")
        for N, a, it, tr, ts, td, df in rows:
            dense = f"{td:.2f}" if not np.isnan(td) else "not run"
            dif = sci(df) if not np.isnan(df) else "--"
            fh.write(f"{int(N)} & {a} & {it} & {tr:.3f} & {ts:.3f} & {dense} & {dif} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    big = R[ok][-1]
    num("ctcScaleDenseN", int(big[0])); num("ctcScaleDenseT", f"{big[5]:.2f}"); num("ctcScaleDenseRed", f"{big[3]:.2f}"); num("ctcScaleDenseSad", f"{big[4]:.2f}")
    num("ctcScaleMaxN", int(R[-1, 0])); num("ctcScaleMaxRed", f"{R[-1, 3]:.2f}"); num("ctcScaleMaxSad", f"{R[-1, 4]:.2f}"); num("ctcScaleMaxIt", int(R[-1, 2]))
    num("ctcScaleDiff", sci(np.nanmax(R[:, 6])))


# ------------------------------------------------------------------------------------------------ E4 agreement
def exp_agreement():
    K, f, X = membrane_system(23)
    N = K.shape[0]
    B = sp.identity(N, format="csr")
    Kd, Bd = K.toarray(), np.eye(N)
    off = Kd - np.diag(np.diag(Kd))
    num("ctcOffMax", f"{off.max():.1f}"); num("ctcMinEig", sci(float(np.linalg.eigvalsh(Kd).min())))
    rows = []
    for name, c in six_obstacles(X).items():
        r = solve_contact_static_sparse(K, f, B, c, route="reduced")
        s = solve_contact_static_sparse(K, f, B, c, route="saddle")
        da = solve_contact_static(Kd, f, Bd, c, method="active_set")
        dl = solve_contact_static(Kd, f, Bd, c, method="lemke")
        dp = solve_contact_static(Kd, f, Bd, c, method="pgs")
        err = lambda a, b: max(np.abs(a.u - b.u).max(), np.abs(a.lam - b.lam).max())
        rows.append((name, int((r.lam > 1e-10).sum()), r.iterations, s.iterations, err(r, da), err(s, da), err(dl, da), err(dp, da)))
        print(name, rows[-1][1:], flush=True)
    with open(os.path.join(TAB, "ch07_agreement.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nobstacle & active & it.\\ red. & it.\\ sad. & reduced & saddle & Lemke & PGS \\\\\n\\midrule\n")
        for nm, a, i1, i2, e1, e2, e3, e4 in rows:
            fh.write(f"{nm} & {a} & {i1} & {i2} & {sci(e1)} & {sci(e2)} & {sci(e3)} & {sci(e4)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("ctcAgreeN", N)
    num("ctcAgreeRedMax", sci(max(r[4] for r in rows))); num("ctcAgreeSadMax", sci(max(r[5] for r in rows)))
    num("ctcAgreeLemkeMax", sci(max(r[6] for r in rows))); num("ctcAgreePgsMax", sci(max(r[7] for r in rows)))
    num("ctcAgreeActMin", min(r[1] for r in rows)); num("ctcAgreeActMax", max(r[1] for r in rows))


# ------------------------------------------------------------------------------------------------ E5 order of accuracy
def exp_order():
    K, f, X = membrane_system(15)
    N = K.shape[0]
    fes_m = None
    # lumped mass of the same membrane
    m = meshlib.rectangle_mesh(1.0, 1.0, 15, 15)
    fes = FESystem(m, Quad4Laplace(), sparse=True)
    fes.assemble_stiffness(1.0); fes.assemble_lumped_mass(1.0)
    Xn = m.nodes
    edge = np.flatnonzero((Xn[:, 0] < 1e-9) | (Xn[:, 0] > 1 - 1e-9) | (Xn[:, 1] < 1e-9) | (Xn[:, 1] > 1 - 1e-9))
    fes.fix_dofs(edge, ["u"])
    free = np.asarray(fes.free_dofs)
    Kd = sp.csr_matrix(fes.K)[free][:, free].toarray()
    Md = np.diag(np.asarray(sp.csr_matrix(fes.M_lumped).diagonal())[free])
    fl = -10.0 * (Md @ np.ones(len(free)))
    w2, V = sl.eigh(Kd, Md)
    a = V.T @ fl
    T = 2.0
    exact = V @ ((a / w2) * (1 - np.cos(np.sqrt(w2) * T)))
    B, c = np.eye(len(free)), np.full(len(free), -100.0)         # obstacle far away: contact inactive
    dts = (0.01, 0.005, 0.0025)
    e_t, e_b = [], []
    for h in dts:
        n = int(round(T / h))
        U2, _ = solve_contact_dynamics(Md, Kd, lambda t: fl, B, c, h, n, scheme="bdf2")
        U1, _ = solve_contact_dynamics(Md, Kd, lambda t: fl, B, c, h, n)
        e_b.append(float(np.abs(U2[-1] - exact).max())); e_t.append(float(np.abs(U1[-1] - exact).max()))
    ord_t = np.log2(np.array(e_t[:-1]) / np.array(e_t[1:])); ord_b = np.log2(np.array(e_b[:-1]) / np.array(e_b[1:]))
    with open(os.path.join(TAB, "ch07_order.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$\\Delta t$ & error, two-step & observed order & error, BDF2 & observed order & ratio \\\\\n\\midrule\n")
        for i, h in enumerate(dts):
            o1 = f"{ord_t[i - 1]:.2f}" if i else "--"; o2 = f"{ord_b[i - 1]:.2f}" if i else "--"
            fh.write(f"{h:g} & {sci(e_t[i])} & {o1} & {sci(e_b[i])} & {o2} & {e_t[i] / e_b[i]:.0f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    d = np.array(dts)
    ax.loglog(d, e_t, "o-", label="two-step (default)"); ax.loglog(d, e_b, "s-", label="BDF2")
    ax.loglog(d, e_t[0] * d / d[0], "k:", lw=1, label="slope 1"); ax.loglog(d, e_b[0] * (d / d[0]) ** 2, "k--", lw=1, label="slope 2")
    ax.set_xlabel(r"time step $\Delta t$"); ax.set_ylabel("error at $t=2$"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch07_bdf2_order.pdf")); plt.close(fig)
    num("ctcOrdN", len(free)); num("ctcOrdTwoLast", f"{ord_t[-1]:.2f}"); num("ctcOrdBdfLast", f"{ord_b[-1]:.2f}"); num("ctcOrdBdfFirst", f"{ord_b[0]:.2f}")
    num("ctcOrdRatioMid", f"{e_t[1] / e_b[1]:.0f}"); num("ctcOrdRatioFine", f"{e_t[2] / e_b[2]:.0f}"); num("ctcOrdRatioCoarse", f"{e_t[0] / e_b[0]:.0f}")
    print("order", e_t, e_b, ord_t, ord_b)


# ------------------------------------------------------------------------------------------------ E6 impact
def bar_impact(h, T=1.5):
    n, dx = 40, 1.0 / 39
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i:i + 2, i:i + 2] += (1 / dx) * np.array([[1, -1], [-1, 1]])
    K += np.eye(n) * 1e-3 / n
    M = np.diag(np.full(n, dx)); M[0, 0] = M[-1, -1] = dx / 2
    B = np.zeros((1, n)); B[0, 0] = 1.0
    c = np.array([-0.02])
    v0 = -np.ones(n)
    N = int(round(T / h))
    Ub, Lb, ib = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0, scheme="bdf2", return_info=True)
    U2, L2 = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0)
    V2 = np.zeros_like(U2); V2[0] = v0
    V2[1:-1] = (U2[2:] - U2[:-2]) / (2 * h); V2[-1] = (U2[-1] - U2[-2]) / h
    en = lambda U, V: np.array([0.5 * V[k] @ M @ V[k] + 0.5 * U[k] @ K @ U[k] for k in range(len(U))])
    return dict(t=np.arange(N + 1) * h, Lb=Lb[:, 0], L2=L2[:, 0], Eb=en(Ub, ib.V), E2=en(U2, V2), pb=float(max(0, c[0] - Ub[:, 0].min())),
                p2=float(max(0, c[0] - U2[:, 0].min())))


def free_bar_energy(h, T=1.5):
    """Energy kept at t = T / initial, for the same bar without any obstacle and with a broadband initial velocity
    (seeded noise): the numerical dissipation of each scheme alone. Returns (bdf2, two_step)."""
    n, dx = 40, 1.0 / 39
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i:i + 2, i:i + 2] += (1 / dx) * np.array([[1, -1], [-1, 1]])
    K += np.eye(n) * 1e-3 / n
    M = np.diag(np.full(n, dx)); M[0, 0] = M[-1, -1] = dx / 2
    B = np.zeros((1, n)); B[0, 0] = 1.0
    c = np.array([-100.0])
    v0 = 0.1 * np.random.default_rng(0).standard_normal(n)
    N = int(round(T / h))
    Ub, _, ib = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0, scheme="bdf2", return_info=True)
    U2, _ = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0)
    V2 = np.zeros_like(U2); V2[0] = v0
    V2[1:-1] = (U2[2:] - U2[:-2]) / (2 * h); V2[-1] = (U2[-1] - U2[-2]) / h
    en = lambda U, V, k: 0.5 * V[k] @ M @ V[k] + 0.5 * U[k] @ K @ U[k]
    return en(Ub, ib.V, -1) / en(Ub, ib.V, 0), en(U2, V2, -1) / en(U2, V2, 0)


def exp_impact():
    res = {h: bar_impact(h) for h in (0.008, 0.004, 0.002, 0.001)}
    with open(os.path.join(TAB, "ch07_impact.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrr}\n\\toprule\n$\\Delta t$ & scheme & largest energy / initial & final energy / initial & largest penetration \\\\\n\\midrule\n")
        for h, r in res.items():
            E0 = r["Eb"][0]
            fh.write(f"{h:g} & two-step & {r['E2'].max() / E0:.4f} & {r['E2'][-1] / E0:.4f} & {sci(r['p2']) if r['p2'] > 0 else '0'} \\\\\n")
            fh.write(f" & BDF2 & {r['Eb'].max() / E0:.4f} & {r['Eb'][-1] / E0:.4f} & {sci(r['pb']) if r['pb'] > 0 else '0'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    r = res[0.002]; E0 = r["Eb"][0]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax[0].plot(r["t"], r["L2"], label="two-step"); ax[0].plot(r["t"], r["Lb"], label="BDF2")
    ax[0].set_xlim(0, 0.4); ax[0].set_xlabel("t"); ax[0].set_ylabel("wall force"); ax[0].legend(); ax[0].set_title(r"contact force, $\Delta t=0.002$")
    ax[1].plot(r["t"], r["E2"] / E0, label="two-step"); ax[1].plot(r["t"], r["Eb"] / E0, label="BDF2"); ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("energy / initial energy"); ax[1].legend(); ax[1].set_title("energy")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch07_impact.pdf")); plt.close(fig)
    for h, tag in ((0.008, "Vcoarse"), (0.004, "Coarse"), (0.002, "Fine"), (0.001, "Vfine")):
        q = res[h]; E0 = q["Eb"][0]
        num("ctcImpBdfMax" + tag, f"{q['Eb'].max() / E0:.4f}"); num("ctcImpBdfEnd" + tag, f"{q['Eb'][-1] / E0:.4f}")
        num("ctcImpTwoMax" + tag, f"{q['E2'].max() / E0:.4f}"); num("ctcImpTwoEnd" + tag, f"{q['E2'][-1] / E0:.4f}")
        num("ctcImpBdfForce" + tag, f"{q['Lb'].max():.1f}"); num("ctcImpTwoForce" + tag, f"{q['L2'].max():.1f}")
        fd = free_bar_energy(h)
        num("ctcFreeBdf" + tag, f"{fd[0]:.4f}"); num("ctcFreeTwo" + tag, f"{fd[1]:.4f}")
        num("ctcFreeBdfPct" + tag, f"{100 * fd[0]:.0f}"); num("ctcFreeTwoPct" + tag, f"{100 * fd[1]:.0f}")
        num("ctcImpBdfGain" + tag, f"{100 * (q['Eb'].max() / E0 - 1):.2f}")
    gains = [100 * (res[h]["Eb"].max() / res[h]["Eb"][0] - 1) for h in res]
    for i in range(3):
        num("ctcImpBdfInc" + "ABC"[i], f"{gains[i + 1] - gains[i]:.2f}")
    ends_b = [res[h]["Eb"][-1] / res[h]["Eb"][0] for h in res]; ends_2 = [res[h]["E2"][-1] / res[h]["Eb"][0] for h in res]
    num("ctcImpBdfEndMin", f"{min(ends_b):.2f}"); num("ctcImpBdfEndMax", f"{max(ends_b):.2f}")
    num("ctcImpTwoEndMin", f"{min(ends_2):.2f}"); num("ctcImpTwoEndMax", f"{max(ends_2):.2f}")
    with open(os.path.join(TAB, "ch07_impact_force.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrr}\n\\toprule\n & \\multicolumn{2}{c}{largest wall force} & \\multicolumn{2}{c}{free bar: energy kept} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n$\\Delta t$ & two-step & BDF2 & two-step & BDF2 \\\\\n\\midrule\n")
        for h in res:
            q = res[h]; fd = free_bar_energy(h)
            fh.write(f"{h:g} & {q['L2'].max():.2f} & {q['Lb'].max():.2f} & {fd[1]:.4f} & {fd[0]:.4f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print({h: (r_['Eb'].max() / r_['Eb'][0], r_['E2'].max() / r_['Eb'][0]) for h, r_ in res.items()})


def main():
    print(platform.platform(), "python", platform.python_version(), "numpy", np.__version__, "cpus", os.cpu_count(), flush=True)
    exp_penalty(); exp_obstacle(); exp_agreement(); exp_order(); exp_impact(); exp_scaling(); exp_large()
    num("ctcMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch07.tex"), "w") as fh:
        fh.write("% generated by scripts/ch07_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

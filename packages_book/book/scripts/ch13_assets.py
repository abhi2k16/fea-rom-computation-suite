# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch13_assets.py -- figures and number macros of Chapter 13 (hyper-reduction and contact reduced-order models).

Experiments
  A  dynamic contact ROM (POD primal basis, NMF or SVD dual basis) of an obstacle membrane: reproduction and held-out error,
     positivity of the multipliers, rank of the reduced constraints, online time
  B  greedy snapshot selection by the contact-condition indicator against random selection (static problem)
  C  ECSW on a hardening chain with contact: plain against contact-aware weights, tolerance sweep, held-out and
     out-of-range parameters, rank deficiency and basis enrichment, batched-route timing on a plate
  D  linear / nonlinear domain partitioning with static condensation
  E  hybrid hyper-reduction on a reduced integration domain (1D rope)
Writes into book/: figures/ch13_dyn.pdf, ch13_greedy.pdf, ch13_ecsw.pdf; tables/numbers_ch13.tex.

Run from the repository root:
    PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests:fea_engine/tests python packages_book/book/scripts/ch13_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import re
import subprocess
import sys
import time
import warnings
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def pct(x):
    return f"{x:.2g}\\,\\%"


def best_of(fn, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


from contact_fixtures import ObstacleMembrane, NonlinearChainContact                      # noqa: E402
from fea_engine.contact_solvers import solve_contact_dynamics, solve_contact_newton        # noqa: E402
from rom_engine import (ContactROM, ContactConditionIndicator, greedy_contact_sampling,    # noqa: E402
                        nonneg_dual_basis, active_snapshot_columns)
from rom_engine.contact_dual_basis import svd_dual_basis                                   # noqa: E402
from rom_engine.hyper_reduction import ECSW, ecsw_training_matrix, ecsw_weights            # noqa: E402
from rom_engine.contact_hyper_reduction import (fit_contact_aware_ecsw, solve_reduced_contact_newton,   # noqa: E402
                                                rank_deficiency, enrich_basis_for_constraints,
                                                reconstruct_contact_forces)
from rom_engine.domain_partition import nonlinearity_indicator, select_nonlinear_elements, PartitionedModel  # noqa: E402

H, N_STEPS = 0.005, 120


# ------------------------------------------------------------------------------------------ A
def exp_a(ax_store):
    mem = ObstacleMembrane(18)
    n = mem.n_dof
    M = sp.diags(np.full(n, mem.h ** 2)).tocsr()
    force = lambda t: mem.f
    train_g = [(0.35, 0.10), (0.45, 0.12), (0.40, 0.18), (0.40, 0.12)]
    test_g = {"inside": (0.40, 0.15), "outside": (0.55, 0.05)}
    fom = {}
    for g in train_g + list(test_g.values()):
        c = mem.obstacle(g)
        fom[g] = (c,) + solve_contact_dynamics(M, mem.K, force, mem.B, c, H, N_STEPS, scheme="bdf2")
    Utr = np.hstack([fom[g][1][1:].T for g in train_g])
    Ltr = np.hstack([fom[g][2][1:].T for g in train_g])
    act = active_snapshot_columns(Ltr)
    num("chdN", str(n)); num("chdSteps", str(N_STEPS)); num("chdNTrain", str(len(train_g)))
    num("chdNCols", str(Utr.shape[1])); num("chdNAct", str(len(act)))
    num("chdMaxActive", str(int(max((fom[g][2] > 1e-9).sum(1).max() for g in fom))))
    Ux = np.linalg.svd(Utr, full_matrices=False)[0]
    sizes = (10, 20, 30, 40)
    res = {}
    for p in sizes:
        Ul_nmf = nonneg_dual_basis(Ltr[:, act], p).U
        Ul_svd = svd_dual_basis(Ltr[:, act], p)
        for name, Ul in (("nmf", Ul_nmf), ("svd", Ul_svd)):
            rom = ContactROM(mem.K, mem.B, Ux[:, :p], Ul, M=M)
            row = {}
            for key, g in [("train", train_g[3]), ("inside", test_g["inside"]), ("outside", test_g["outside"])]:
                c, Uf, Lf = fom[g]
                r = rom.solve_dynamic(force, c, H, N_STEPS)
                if r.ok:
                    row[key] = np.sum((r.u - Uf) ** 2) / np.sum(Uf ** 2)
                    row[key + "_minlam"] = float(r.lam.min())
                    row[key + "_kkt"] = float(r.kkt_residual.max())
                    row[key + "_traj"] = r.u[:, n // 2]
                else:
                    row[key] = np.nan; row[key + "_minlam"] = np.nan; row[key + "_kkt"] = np.nan
                row[key + "_ok"] = bool(r.ok)
            row["rank"] = rom.inf_sup_report()
            res[(name, p)] = row
    for p in sizes:
        for name in ("nmf", "svd"):
            r = res[(name, p)]
            tag = ("Nmf" if name == "nmf" else "Svd") + {10: "Ten", 20: "Twenty", 30: "Thirty", 40: "Forty"}[p]
            for key, K in (("train", "Tr"), ("inside", "In"), ("outside", "Out")):
                num(f"chd{tag}{K}", sci(r[key]) if np.isfinite(r[key]) else "failed")
                num(f"chd{tag}{K}MinLam", sci(r[key + "_minlam"]) if np.isfinite(r[key + "_minlam"]) else "--")
            num(f"chd{tag}Rank", f"{r['rank']['rank']}/{r['rank']['n_dual']}")
            num(f"chd{tag}KktOut", sci(r["outside_kkt"]) if np.isfinite(r["outside_kkt"]) else "--")
            print(name, p, {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items() if "traj" not in k and k != "rank"}, r["rank"]["rank"], flush=True)
    # timing (p = 30)
    p = 30
    Ul = nonneg_dual_basis(Ltr[:, act], p).U
    rom = ContactROM(mem.K, mem.B, Ux[:, :p], Ul, M=M)
    c = fom[test_g["inside"]][0]
    t_fom = best_of(lambda: solve_contact_dynamics(M, mem.K, force, mem.B, c, H, N_STEPS, scheme="bdf2"))
    t_rom = best_of(lambda: rom.solve_dynamic(force, c, H, N_STEPS, lift=False))
    r_w = rom.solve_dynamic(force, c, H, N_STEPS, lift=False)
    r_c = rom.solve_dynamic(force, c, H, N_STEPS, lift=False, warm_start=False)
    num("chdTimeFom", f"{1e3 * t_fom:.0f}"); num("chdTimeRom", f"{1e3 * t_rom:.0f}"); num("chdSpeed", f"{t_fom / t_rom:.1f}")
    num("chdItersWarm", str(int(sum(r_w.iterations)))); num("chdItersCold", str(int(sum(r_c.iterations))))
    t_setup = best_of(lambda: ContactROM(mem.K, mem.B, Ux[:, :p], Ul, M=M), 3)
    num("chdTimeSetup", f"{1e3 * t_setup:.1f}")
    # reporting: offline cost, lift cost, local quantities (peak centre displacement, peak total contact force)
    t0 = time.perf_counter()
    for g in train_g:
        solve_contact_dynamics(M, mem.K, force, mem.B, mem.obstacle(g), H, N_STEPS, scheme="bdf2")
    t_trajs = time.perf_counter() - t0
    t0 = time.perf_counter(); Ux_ = np.linalg.svd(Utr, full_matrices=False)[0]; t_svd = time.perf_counter() - t0
    t0 = time.perf_counter(); nonneg_dual_basis(Ltr[:, act], p); t_nmf = time.perf_counter() - t0
    offline = t_trajs + t_svd + t_nmf + t_setup
    t_lift = best_of(lambda: rom.solve_dynamic(force, c, H, N_STEPS, lift=True)) - t_rom
    num("chdTimeTrajs", f"{1e3 * t_trajs:.0f}"); num("chdTimeSvd", f"{1e3 * t_svd:.0f}"); num("chdTimeNmf", f"{1e3 * t_nmf:.0f}")
    num("chdTimeOffline", f"{1e3 * offline:.0f}"); num("chdTimeLift", f"{1e3 * max(t_lift, 0):.1f}")
    saving = t_fom - t_rom - max(t_lift, 0)
    num("chdBreakEven", f"{offline / saving:.0f}" if saving > 0 else "never")
    num("chdSpeedLift", f"{t_fom / (t_rom + max(t_lift, 0)):.1f}")
    for key in ("inside", "outside"):
        c_, Uf_, Lf_ = fom[test_g[key]]
        r_ = rom.solve_dynamic(force, c_, H, N_STEPS)
        pk = lambda U_: np.abs(U_[:, n // 2]).max()
        fk = lambda L_: L_.sum(1).max()
        tag = "In" if key == "inside" else "Out"
        num(f"chdPeakDisp{tag}", sci(abs(pk(r_.u) - pk(Uf_)) / pk(Uf_)))
        num(f"chdPeakForce{tag}", sci(abs(fk(r_.lam) - fk(Lf_)) / fk(Lf_)))
        num(f"chdGlobal{tag}", sci(np.sqrt(np.sum((r_.u - Uf_) ** 2) / np.sum(Uf_ ** 2))))
    # complete basis reproduction
    rc = ContactROM(mem.K, mem.B, np.eye(n), np.eye(n), M=M).solve_dynamic(force, c, H, N_STEPS)
    num("chdComplete", sci(np.abs(rc.u - fom[test_g["inside"]][1]).max()))
    num("chdDualNmfErr", sci(nonneg_dual_basis(Ltr[:, act], 30).rel_error))
    # figure
    t = np.arange(N_STEPS + 1) * H
    ax = ax_store
    c_o, Uf_o, _ = fom[test_g["outside"]]
    ax[0].plot(t, Uf_o[:, n // 2], "k", lw=2.2, label="full order")
    for p_, st in zip((10, 30, 40), ("--", "-.", ":")):
        tj = res[("nmf", p_)]["outside_traj"] if "outside_traj" in res[("nmf", p_)] else None
        if tj is not None:
            ax[0].plot(t, tj, st, label=f"ROM, p = {p_}")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("centre displacement"); ax[0].set_title("held-out obstacle (outside the training range)"); ax[0].legend()
    for key, mk, lab in (("train", "o-", "training obstacle"), ("inside", "s--", "held out, inside"), ("outside", "^:", "held out, outside")):
        ax[1].semilogy(sizes, [res[("nmf", p_)][key] for p_ in sizes], mk, label=lab)
    ax[1].set_xlabel("p = p$_\\lambda$"); ax[1].set_ylabel("relative trajectory error"); ax[1].set_title("POD + NMF basis"); ax[1].legend()
    ax[2].plot(sizes, [res[("nmf", p_)]["outside_minlam"] for p_ in sizes], "o-", label="NMF dual basis")
    ax[2].plot(sizes, [res[("svd", p_)]["outside_minlam"] for p_ in sizes], "s--", label="SVD dual basis")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("p$_\\lambda$"); ax[2].set_ylabel("smallest lifted multiplier"); ax[2].set_title("positivity of the contact force"); ax[2].legend()
    return res


# ------------------------------------------------------------------------------------------ B
def exp_b(ax):
    mem = ObstacleMembrane(18)
    cands = [(a, b) for a in np.linspace(0.3, 0.6, 6) for b in np.linspace(0.0, 0.1, 6)]
    test = [(0.33, 0.02), (0.41, 0.07), (0.47, 0.04), (0.58, 0.09), (0.36, 0.095), (0.55, 0.01)]
    ref = [mem.solve(g) for g in test]

    def make_rom(U, L):
        Q, _ = np.linalg.qr(U)
        Ul = L / np.maximum(np.linalg.norm(L, axis=0), 1e-300)
        rom = ContactROM(mem.K, mem.B, Q, Ul)
        return lambda p: rom.solve(mem.f, mem.obstacle(p)).u

    test_err = lambda rom: np.mean([np.linalg.norm(rom(g) - r[0]) / np.linalg.norm(r[0]) for g, r in zip(test, ref)])
    indic = ContactConditionIndicator(mem.K, mem.B, mem.f, weights=(1, 0, 0))
    ind = lambda p, u: indic(u, mem.obstacle(p))
    rng = np.random.default_rng(0)
    ns = (2, 4, 6, 8)
    eg, em, ew, eb = [], [], [], []
    n_rand = 20
    for k in ns:
        rom, hist = greedy_contact_sampling(cands, lambda p: mem.solve(p)[:2], make_rom, ind, k)
        eg.append(test_err(rom))
        rr = []
        for _ in range(n_rand):
            idx = rng.choice(len(cands), k, replace=False)
            sols = [mem.solve(cands[i]) for i in idx]
            rr.append(test_err(make_rom(np.array([s[0] for s in sols]).T, np.array([s[1] for s in sols]).T)))
        em.append(np.median(rr)); ew.append(np.max(rr)); eb.append(np.min(rr))
        print("greedy", k, f"{eg[-1]:.3g}", "random median", f"{em[-1]:.3g}", flush=True)
    names = {2: "Two", 4: "Four", 6: "Six", 8: "Eight"}
    for k, a, b, c_, d in zip(ns, eg, em, ew, eb):
        num(f"cgr{names[k]}Greedy", sci(a)); num(f"cgr{names[k]}Med", sci(b)); num(f"cgr{names[k]}Worst", sci(c_)); num(f"cgr{names[k]}Best", sci(d))
    num("cgrNRand", str(n_rand)); num("cgrNCand", str(len(cands))); num("cgrNTest", str(len(test)))
    num("cgrWinCount", str(int(sum(g < m for g, m in zip(eg, em)))))
    num("cgrBeatBestCount", str(int(sum(g < b for g, b in zip(eg, eb)))))
    ax.semilogy(ns, eg, "o-", label="greedy (contact-condition indicator)")
    ax.semilogy(ns, em, "s--", label=f"random, median of {n_rand}")
    ax.fill_between(ns, eb, ew, alpha=0.2, label="random, best to worst")
    ax.set_xlabel("snapshots"); ax.set_ylabel("mean relative test error"); ax.legend()
    ax.set_title("static contact ROM: snapshot selection")


# ------------------------------------------------------------------------------------------ C
def exp_c(ax):
    m = NonlinearChainContact()
    train = [(s, g) for s in (0.5, 1, 1.5, 2, 2.5, 3) for g in (-0.08, -0.15, -0.25)]
    sols = [m.solve(*p) for p in train]
    Xu = np.array([s[0] for s in sols]).T
    Xl = np.array([s[1] for s in sols]).T
    r = 8
    U = np.linalg.svd(Xu, full_matrices=False)[0][:, :r]
    req = [e for e, g in enumerate(m.elem_dofs) if set(g.tolist()) & set(m.obs_nodes.tolist())]
    n_elem = len(m.elem_dofs)
    num("cecN", str(m.n)); num("cecNElem", str(n_elem)); num("cecNTrain", str(len(train))); num("cecRank", str(r))
    num("cecNReq", str(len(req))); num("cecNObs", str(len(m.obs_nodes)))
    G, b = ecsw_training_matrix(U, Xu, m.elem_dofs, m.elem_force)
    held = [(1.25, -0.12), (2.2, -0.2), (0.8, -0.1)]
    out = [(4.0, -0.30), (5.0, -0.40)]
    # force error at held-out states: full reduced force vs ECSW reduced force
    def force_err(ec, prm):
        u = m.solve(*prm)[0]
        q = U.T @ u
        ref = U.T @ m.internal_force(U @ q)
        return np.linalg.norm(ec.reduced_force(q) - ref) / np.linalg.norm(ref)
    # tolerance sweep, plain ECSW
    tols = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    nsel, ferr, ferr_out = [], [], []
    for tol in tols:
        w, info = ecsw_weights(G, b, tol=tol)
        ec = ECSW(U, m.elem_dofs, m.elem_force, w, m.elem_tangent, info)
        nsel.append(ec.n_selected)
        ferr.append(max(force_err(ec, p) for p in held))
        ferr_out.append(max(force_err(ec, p) for p in out))
    names = {1e-2: "TwoE", 1e-3: "ThreeE", 1e-4: "FourE", 1e-6: "SixE", 1e-8: "EightE"}
    for tol, ns_, fe, fo in zip(tols, nsel, ferr, ferr_out):
        num(f"cec{names[tol]}N", str(ns_)); num(f"cec{names[tol]}Err", sci(fe)); num(f"cec{names[tol]}ErrOut", sci(fo))
    print("tol sweep", nsel, [f"{x:.2g}" for x in ferr], [f"{x:.2g}" for x in ferr_out], flush=True)
    # plain vs contact-aware at 1e-6, reduced contact Newton
    ec_plain = ECSW.fit(U, Xu, m.elem_dofs, m.elem_force, m.elem_tangent, tol=1e-6)
    ec_ca = fit_contact_aware_ecsw(U, Xu, m.elem_dofs, m.elem_force, req, m.elem_tangent, tol=1e-6)
    num("cecPlainN", str(ec_plain.n_selected)); num("cecCaN", str(ec_ca.n_selected))
    num("cecPlainRes", sci(ec_plain.info["residual"])); num("cecCaRes", sci(ec_ca.info["residual"]))
    nmf = nonneg_dual_basis(Xl, r)
    num("cecNmfErr", sci(nmf.rel_error))
    rows = []
    for tag, prm in [("In", held[0]), ("InB", held[1]), ("InC", held[2]), ("Out", out[0]), ("OutB", out[1])]:
        u, lam, c, fe = m.solve(*prm)
        d = {}
        for name, ec in (("Plain", ec_plain), ("Ca", ec_ca)):
            for ul_name, Ul in (("Full", None), ("Nmf", nmf.U)):
                try:
                    rr = solve_reduced_contact_newton(ec, fe, U, m.B, c, U_lam=Ul)
                    err = np.linalg.norm(rr.u - u) / np.linalg.norm(u)
                    pen = float((m.B @ rr.u - c).min())
                    d[(name, ul_name)] = (rr.converged, err, pen, float(rr.lam.min()) if Ul is not None else None)
                except Exception as ex:                                       # pragma: no cover
                    d[(name, ul_name)] = (False, np.nan, np.nan, None)
        rows.append((tag, prm, d))
        for (name, ul_name), (cv, err, pen, ml) in d.items():
            num(f"cec{name}{ul_name}{tag}Err", sci(err) if np.isfinite(err) else "--")
            num(f"cec{name}{ul_name}{tag}Conv", "yes" if cv else "no")
            num(f"cec{name}{ul_name}{tag}Pen", sci(abs(pen)) if np.isfinite(pen) else "--")
        print(tag, prm, {k: (v[0], f"{v[1]:.2g}", f"{v[2]:.2g}") for k, v in d.items()}, flush=True)
    # reporting: offline (full solves, training matrix, NNLS, dual basis), online (reduced Newton), lift, local quantities
    t0 = time.perf_counter(); [m.solve(*p_) for p_ in train]; t_sol = time.perf_counter() - t0
    t0 = time.perf_counter(); Gt, bt = ecsw_training_matrix(U, Xu, m.elem_dofs, m.elem_force); t_G = time.perf_counter() - t0
    t0 = time.perf_counter(); ecsw_weights(Gt, bt, tol=1e-6); t_w = time.perf_counter() - t0
    t0 = time.perf_counter(); nonneg_dual_basis(Xl, r); t_n = time.perf_counter() - t0
    prm = held[0]
    u_f, lam_f, c_f, fe_f = m.solve(*prm)
    t_full = best_of(lambda: m.solve(*prm))
    t_red = best_of(lambda: solve_reduced_contact_newton(ec_plain, fe_f, U, m.B, c_f, U_lam=nmf.U))
    qq = U.T @ u_f
    t_lift = best_of(lambda: U @ qq, 20)
    rr = solve_reduced_contact_newton(ec_plain, fe_f, U, m.B, c_f, U_lam=nmf.U)
    off = t_sol + t_G + t_w + t_n
    num("cecTimeSolves", f"{1e3 * t_sol:.0f}"); num("cecTimeG", f"{1e3 * t_G:.0f}"); num("cecTimeW", f"{1e3 * t_w:.0f}"); num("cecTimeN", f"{1e3 * t_n:.0f}")
    num("cecTimeOffline", f"{1e3 * off:.0f}"); num("cecTimeFull", f"{1e3 * t_full:.1f}"); num("cecTimeRed", f"{1e3 * t_red:.1f}"); num("cecTimeLiftUs", f"{1e6 * t_lift:.0f}")
    num("cecSpeedOnline", f"{t_full / t_red:.1f}")
    sav = t_full - t_red - t_lift
    num("cecBreakEven", f"{off / sav:.0f}" if sav > 0 else "never")
    num("cecLocForce", sci(abs(rr.lam.sum() - lam_f.sum()) / lam_f.sum()))
    num("cecLocPeak", sci(abs(rr.u.min() - u_f.min()) / abs(u_f.min())))
    num("cecGlobU", sci(np.linalg.norm(rr.u - u_f) / np.linalg.norm(u_f)))
    num("cecOutScale", f"{out[0][0]:g}"); num("cecOutGap", f"{abs(out[0][1]):g}")
    num("cecTrainScaleMax", "3"); num("cecTrainGapMax", "0.25")
    # contact-force reconstruction from a partial zone
    u, lam, c, fe = m.solve(*held[0])
    gap = m.B @ u - c
    active = np.flatnonzero(lam > 1e-9)
    inside = active[:8]
    rec = reconstruct_contact_forces(Xl, inside, lam[inside], gap)
    num("cecRecIn", sci(np.linalg.norm(rec[inside] - lam[inside]) / np.linalg.norm(lam[inside])))
    num("cecRecTotal", sci(abs(rec.sum() - lam.sum()) / lam.sum())); num("cecNActive", str(len(active)))
    # rank deficiency and enrichment
    V3 = np.linalg.svd(Xu, full_matrices=False)[0][:, :3]
    d3 = rank_deficiency(m.B, V3)
    V3e, added = enrich_basis_for_constraints(V3, m.B)
    num("cecRankDef", str(d3)); num("cecRankAdded", str(len(added))); num("cecRankAfter", str(rank_deficiency(m.B, V3e)))
    num("cecRankCols", str(V3e.shape[1]))
    # figure: tolerance sweep
    ax[0].semilogy(nsel, ferr, "o-", label="held out, inside the training range")
    ax[0].semilogy(nsel, ferr_out, "s--", label="outside the training range")
    ax[0].set_xlabel("selected elements (of %d)" % n_elem); ax[0].set_ylabel("rel. error of reduced force"); ax[0].legend()
    ax[0].set_title("ECSW: tolerance sweep")
    sel_plain = set(ec_plain.elements.tolist()); sel_ca = set(ec_ca.elements.tolist())
    ax[1].bar(np.arange(n_elem), [1 if e in sel_plain else 0 for e in range(n_elem)], width=1.0, alpha=0.6, label="plain")
    ax[1].bar(np.arange(n_elem), [-1 if e in sel_ca else 0 for e in range(n_elem)], width=1.0, alpha=0.6, label="contact-aware")
    ax[1].axvspan(min(req) - 0.5, max(req) + 0.5, color="C3", alpha=0.12, label="contact elements")
    ax[1].set_xlabel("element"); ax[1].set_yticks([]); ax[1].legend(loc="lower right"); ax[1].set_title("selected elements (tol $10^{-6}$)")
    # batched route timing on a plate (external script)
    try:
        root = os.path.abspath(os.path.join(BOOK, "..", ".."))
        env = dict(os.environ)
        out_txt = subprocess.run([sys.executable, os.path.join(root, "rom_engine", "tests", "hyper_reduction_timing.py")], cwd=root, env=env,
                                 capture_output=True, text=True, timeout=600).stdout
        print(out_txt, flush=True)
        mm = re.search(r"training matrix\s+per-element\s+([\d.]+) ms.*?batched\s+([\d.]+) ms\s+x\s*([\d.]+)", out_txt)
        ms = re.search(r"selected (\d+) \(per-element\) / (\d+) \(batched\) of (\d+).*same set: (\w+)", out_txt)
        mp = re.search(r"plate (\d+)x(\d+): (\d+) elements, (\d+) DOFs", out_txt)
        if mm and ms and mp:
            num("cbtElems", mp.group(3)); num("cbtDofs", mp.group(4))
            num("cbtPer", mm.group(1)); num("cbtBat", mm.group(2)); num("cbtSpeed", mm.group(3))
            num("cbtSelected", ms.group(1)); num("cbtSame", ms.group(4))
            mf = re.search(r"reduced force \(\d+ elements\)\s+per-element\s+([\d.]+) ms\s+batched\s+([\d.]+) ms\s+x\s*([\d.]+)", out_txt)
            mt = re.search(r"reduced tangent\s+per-element\s+([\d.]+) ms\s+batched\s+([\d.]+) ms\s+x\s*([\d.]+)", out_txt)
            mi = re.search(r"nonlinearity indicator.*?per-element\s+([\d.]+) ms.*?batched\s+([\d.]+) ms\s+x\s*([\d.]+)", out_txt)
            if mf and mt and mi:
                num("cbtForceSpeed", mf.group(3)); num("cbtTangentSpeed", mt.group(3)); num("cbtIndSpeed", mi.group(3))
                num("cbtIndPer", mi.group(1)); num("cbtIndBat", mi.group(2))
    except Exception as ex:                                                  # pragma: no cover
        print("timing script failed", ex)


# ------------------------------------------------------------------------------------------ D
def exp_d():
    E0 = 24
    m = NonlinearChainContact(n=60, linear_below=E0)
    states = np.array([m.solve(s, -0.15)[0] for s in (0.5, 1.0, 2.0, 3.0)]).T
    K = np.zeros((m.n, m.n))
    for e in range(E0):
        g = m.elem_dofs[e]
        K[np.ix_(g, g)] += m.elem_tangent(e, np.zeros(len(g)))
    pm = PartitionedModel(m.elem_dofs, range(E0, m.n), sp.csr_matrix(K), m.n)
    nl = pm.nl_dofs
    pos = {d: i for i, d in enumerate(nl)}
    nl_elems = list(range(E0, m.n))

    def scatter_force(uN):
        u = np.zeros(m.n); u[nl] = uN
        f = np.zeros(m.n)
        for e in nl_elems:
            g = m.elem_dofs[e]; f[g] += m.elem_force(e, u[g])
        return f[nl]

    def scatter_tangent(uN):
        u = np.zeros(m.n); u[nl] = uN
        Kt = np.zeros((len(nl), len(nl)))
        for e in nl_elems:
            g = m.elem_dofs[e]; li = [pos[d] for d in g]
            Kt[np.ix_(li, li)] += m.elem_tangent(e, u[g])
        return Kt

    B_N = np.zeros((len(m.obs_nodes), len(nl)))
    B_N[np.arange(len(m.obs_nodes)), [pos[d] for d in m.obs_nodes]] = 1.0
    errs_u, errs_l = [], []
    for prm in [(1.0, -0.15), (2.5, -0.2), (0.7, -0.1)]:
        u_full, lam_full, c, fe = m.solve(*prm)
        f_N = pm.condensed_load(fe)
        res = solve_contact_newton(lambda uN: pm.residual(uN, scatter_force(uN), f_N), lambda uN: pm.tangent(scatter_tangent(uN)),
                                   B_N, c, np.zeros(len(nl)))
        u = pm.expand(res.u, fe)
        errs_u.append(np.abs(u - u_full).max() / np.abs(u_full).max()); errs_l.append(np.abs(res.lam - lam_full).max() / np.abs(lam_full).max())
        assert res.converged
    kappa = nonlinearity_indicator(m.elem_dofs, m.elem_force, m.elem_tangent, states)
    sel = select_nonlinear_elements(kappa, 1e-6, always=[3])
    num("cptN", str(m.n)); num("cptLinear", str(E0)); num("cptNl", str(pm.n_nl)); num("cptIface", str(len(pm.interface)))
    num("cptInterior", str(len(pm.interior))); num("cptKstar", f"{pm.K_star[0, 0]:.4g}"); num("cptKseries", f"{m.k / E0:.4g}")
    num("cptErrU", sci(max(errs_u))); num("cptErrLam", sci(max(errs_l)))
    num("cptKappaLin", sci(float(kappa[:E0].max()))); num("cptKappaNlMin", sci(float(np.sort(kappa[E0:])[len(np.flatnonzero(kappa[E0:] < 1e-6))]) if np.any(kappa[E0:] >= 1e-6) else 0.0))
    num("cptSelected", str(len(sel)))
    print("partition", pm.n_nl, errs_u, errs_l, len(sel), flush=True)
    # ECSW training columns: all elements vs nonlinear zone only
    num("cptColsAll", str(m.n)); num("cptColsNl", str(m.n - E0))
    # hold-out: a nonlinear load factor beyond training
    return pm


# ------------------------------------------------------------------------------------------ E
def exp_e():
    import test_contact_rid as t
    from rom_engine.contact_rid import (pod_by_information_rate, rid_elements, rid_dofs, contact_rows_in_rid,
                                        HybridHRContact, deim_rid_indices)
    nus = np.linspace(0.08, 0.2, 7); hs = np.linspace(0.3, 0.5, 4)
    cases = [(nu, np.full(t.N, h)) for nu in nus for h in hs]
    S = [t.fom(nu, d) for nu, d in cases]
    SU, SL = np.array([s[0] for s in S]).T, np.array([s[1] for s in S]).T
    V = pod_by_information_rate(SU, 1e-8)
    VL = pod_by_information_rate(SL, 1e-4)
    B = sp.identity(t.N, format="csr")
    held = [(0.123, 0.41), (0.15, 0.35), (0.09, 0.47)]
    num("crdN", str(t.N)); num("crdNSnap", str(len(cases))); num("crdBasis", str(V.shape[1]))
    P_all, ext_all = rid_dofs(t.ELEMS, t.N, np.arange(t.N + 1))
    rows_all = contact_rows_in_rid(B, P_all)
    out = {}
    for tag, layers in (("Full", None), ("One", 1), ("Three", 3)):
        if layers is None:
            P, ext, rows = P_all, ext_all, rows_all; nel = t.N + 1
        else:
            idx = deim_rid_indices(V, VL)
            el = rid_elements(t.ELEMS, t.N, idx, layers=layers)
            P, ext = rid_dofs(t.ELEMS, t.N, el); rows = contact_rows_in_rid(B, P); nel = len(el)
        hy = HybridHRContact(V, B, P, ext, rows)
        eu, el_, act_ok = [], [], []
        for nu, h in held:
            d = np.full(t.N, h)
            u0, l0 = t.fom(nu, d)
            Kp = t.stiffness(nu).tocsr()
            res = hy.solve(Kp[P][:, ext], t.H * np.ones(t.N)[P], d[rows])
            eu.append(np.linalg.norm(res.u[P] - u0[P]) / np.linalg.norm(u0[P]))
            act_ok.append(bool(np.array_equal(res.lam > 1e-8, l0[rows] > 1e-8)) and res.ok)
        out[tag] = (nel, len(P), len(rows), hy.n_unknowns, hy.inf_sup_deficiency(), max(eu), all(act_ok))
        num(f"crd{tag}Elems", str(nel)); num(f"crd{tag}P", str(len(P))); num(f"crd{tag}Rows", str(len(rows)))
        num(f"crd{tag}Unk", str(hy.n_unknowns)); num(f"crd{tag}Err", sci(max(eu))); num(f"crd{tag}Active", "yes" if all(act_ok) else "no")
        num(f"crd{tag}Added", str(len(hy.added_dofs)))
        print("rid", tag, out[tag], flush=True)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    fa, aa = plt.subplots(1, 3, figsize=(10.5, 3.2))
    exp_a(aa); fa.tight_layout(); fa.savefig(os.path.join(FIG, "ch13_dyn.pdf")); plt.close(fa)
    fb, ab = plt.subplots(figsize=(4.8, 3.2))
    exp_b(ab); fb.tight_layout(); fb.savefig(os.path.join(FIG, "ch13_greedy.pdf")); plt.close(fb)
    fc, ac = plt.subplots(1, 2, figsize=(9.0, 3.0))
    exp_c(ac); fc.tight_layout(); fc.savefig(os.path.join(FIG, "ch13_ecsw.pdf")); plt.close(fc)
    exp_d(); exp_e()
    num("chdMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch13.tex"), "w") as fh:
        fh.write("% generated by scripts/ch13_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

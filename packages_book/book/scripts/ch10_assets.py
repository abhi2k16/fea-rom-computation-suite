# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch10_assets.py -- figures, tables and number macros of Chapter 10 (frequency-domain ROMs, greedy training,
certified error bounds, random vibration).

Model: the damped cantilever of rom_engine/examples/greedy_frequency_training.py (20 Euler-Bernoulli elements, Rayleigh
damping alpha=2, beta=1e-5, unit tip force pattern). Reference: fea_engine's own solve_harmonic / solve_random_vibration.
Writes into book/:
    figures/ch10_greedy.pdf      selected frequencies; held-out error against basis size for greedy, uniform, modal bases
    figures/ch10_indicators.pdf  residual norm and hierarchical indicator against the true error
    figures/ch10_bound.pdf       certified bound, true error and singular-value lower bounds against frequency
    tables/ch10_psd.tex          random-vibration RMS: ROMs against the full-order formula
    tables/numbers_ch10.tex      every number quoted in the prose

Run from the repository root:
    PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/examples python packages_book/book/scripts/ch10_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import spearmanr

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


from greedy_frequency_training import build_damped_cantilever                              # noqa: E402
from rom_engine import FrequencyROM, build_pod_basis_from_frf_snapshots, greedy_train_frequency_basis  # noqa: E402
from rom_engine.scm import SingularValueLowerBound, certified_error_bound                  # noqa: E402
from rom_engine.scm_lp import LPSingularValueLowerBound, certified_error_bound as lp_certified  # noqa: E402
from rom_engine.random_vibration import psd_response                                       # noqa: E402


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    sysobj, alpha, beta = build_damped_cantilever(n=20)
    free = np.asarray(sysobj.free_dofs); n_dof = sysobj.n_dof
    ix = np.ix_(free, free)
    K, M, C = sysobj.K[ix], sysobj.M[ix], sysobj.C[ix]
    lam, Vm = eigh(K, M)
    wn = np.sqrt(lam)
    w1 = wn[0]
    F_full = np.zeros(n_dof); F_full[n_dof - 2] = 1000.0
    F = F_full[free]
    tip = len(free) - 2
    num("cfrNFree", str(len(free))); num("cfrFOne", f"{w1 / (2 * np.pi):.1f}")
    band = np.linspace(0.1, 20.0, 400) * w1
    rng = np.random.default_rng(0)
    test_near = np.concatenate([wn[i] * (1.0 + 0.05 * rng.standard_normal(20)) for i in range(3)])
    test_near = test_near[test_near > 0]
    test_band = np.sort(rng.uniform(0.1, 20.0, 100)) * w1

    def true_resp(om):
        return sysobj.solve_harmonic(om, F_full)[free]

    cache = {}

    def truth(om):
        k = float(om)
        if k not in cache:
            cache[k] = true_resp(om)
        return cache[k]

    def rel_err(rom, oms):
        e = []
        for om in oms:
            u = truth(om)
            x = rom.frequency_response([om], F)[0]
            e.append(np.linalg.norm(x - u) / np.linalg.norm(u))
        return np.array(e)

    # ------------------------------------------------------------ A: greedy vs uniform vs modal
    ranks = [3, 4, 5, 6, 7, 8, 10]
    res = {"greedy": [], "uniform": [], "modal": []}
    sel_store = {}
    for r in ranks:
        gb, hist = greedy_train_frequency_basis(band, M, K, F, rayleigh=(alpha, beta), n_seed=3, tol=1e-12, max_modes=r)
        sel_store[r] = np.array([h[0] for h in hist])
        rom_g = FrequencyROM.from_MCK(M, K, gb.V, rayleigh=(alpha, beta))
        ub = build_pod_basis_from_frf_snapshots(np.linspace(band[0], band[-1], gb.n_modes), M, K, F, C=C, n_modes=gb.n_modes)
        rom_u = FrequencyROM.from_MCK(M, K, ub.V, rayleigh=(alpha, beta))
        rom_m = FrequencyROM.from_MCK(M, K, Vm[:, :r], rayleigh=(alpha, beta))
        for k_, rom in (("greedy", rom_g), ("uniform", rom_u), ("modal", rom_m)):
            en, eb = rel_err(rom, test_near), rel_err(rom, test_band)
            res[k_].append((en.max(), np.median(en), eb.max()))
        print(r, {k_: f"{v[-1][0]:.1e}/{v[-1][2]:.1e}" for k_, v in res.items()}, "rank", gb.n_modes, flush=True)
    sel5 = sel_store[5] / w1
    num("cfrSelFive", ", ".join(f"{x:.2f}" for x in sel5[:6]))
    ir = {r: i for i, r in enumerate(ranks)}
    for tag, key in (("Greedy", "greedy"), ("Uniform", "uniform"), ("Modal", "modal")):
        for r, nm in ((5, "Five"), (8, "Eight")):
            num(f"cfr{tag}Near{nm}", sci(res[key][ir[r]][0])); num(f"cfr{tag}Med{nm}", sci(res[key][ir[r]][1])); num(f"cfr{tag}Band{nm}", sci(res[key][ir[r]][2]))
    num("cfrNTestNear", str(len(test_near))); num("cfrNTestBand", str(len(test_band)))
    wins = sum(1 for g, u in zip(res["greedy"], res["uniform"]) if g[0] < u[0])
    num("cfrGreedyWinsNear", str(wins)); num("cfrRankCount", str(len(ranks)))
    wins_m = sum(1 for g, u in zip(res["greedy"], res["modal"]) if g[0] < u[0])
    num("cfrGreedyWinsModal", str(wins_m))
    winsmed = sum(1 for g, u in zip(res["greedy"], res["uniform"]) if g[1] < u[1])
    num("cfrGreedyWinsMed", str(winsmed))
    worst_om = []
    num("cfrThirdRes", f"{wn[2] / w1:.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3))
    sw = np.linspace(0.1, 15.0, 250) * w1
    Hf = np.array([truth(om)[tip] for om in sw])
    ax[0].semilogy(sw / w1, np.abs(Hf), "k-", lw=1.5, label="full order")
    for v in sel_store[5] / w1:
        ax[0].axvline(v, color="C2", alpha=0.25, lw=3)
    ax[0].set_xlabel("$\\omega/\\omega_1$"); ax[0].set_ylabel("$|$tip displacement$|$"); ax[0].legend(fontsize=7, title="bands: greedy picks, rank 5", title_fontsize=7)
    for k_, st in (("greedy", "o-"), ("uniform", "s--"), ("modal", "^:")):
        ax[1].semilogy(ranks, [v[0] for v in res[k_]], st, ms=4, label=k_)
    ax[1].set_xlabel("basis columns"); ax[1].set_ylabel("max error, 60 test frequencies near resonances"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch10_greedy.pdf")); plt.close(fig)

    # ------------------------------------------------------------ B: indicators vs true error (rank 5 greedy ROM, rank 4 previous)
    b5, h5 = greedy_train_frequency_basis(band, M, K, F, rayleigh=(alpha, beta), n_seed=3, tol=1e-12, max_modes=5)
    b4, _ = greedy_train_frequency_basis(band, M, K, F, rayleigh=(alpha, beta), n_seed=3, tol=1e-12, max_modes=4)
    rom5 = FrequencyROM.from_MCK(M, K, b5.V, rayleigh=(alpha, beta))
    rom4 = FrequencyROM.from_MCK(M, K, b4.V, rayleigh=(alpha, beta))
    cand = np.linspace(0.1, 20.0, 300) * w1
    e_true = np.array([np.linalg.norm(rom5.frequency_response([om], F)[0] - truth(om)) for om in cand])
    resid = np.array([rom5.residual_norm(om, F) for om in cand])
    hier = np.array([rom5.hierarchical_error_indicator(om, F, rom4) for om in cand])
    sp_r = spearmanr(resid, e_true)[0]; sp_h = spearmanr(hier, e_true)[0]
    num("cfrSpearResid", f"{sp_r:.2f}"); num("cfrSpearHier", f"{sp_h:.2f}")
    top = lambda a, k: set(np.argsort(a)[-k:])
    num("cfrTopOverlapResid", str(len(top(resid, 30) & top(e_true, 30)))); num("cfrTopOverlapHier", str(len(top(hier, 30) & top(e_true, 30))))
    num("cfrTopK", "30"); num("cfrNCand", str(len(cand)))
    num("cfrHierUnderMax", f"{(hier / np.maximum(e_true, 1e-300)).min():.2f}")
    num("cfrHierOverMax", f"{(hier / np.maximum(e_true, 1e-300)).max():.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax[0].semilogy(cand / w1, e_true, "k-", label="true error $\\|x-x_r\\|$")
    ax[0].semilogy(cand / w1, resid, "C0--", label="residual norm")
    ax[0].semilogy(cand / w1, hier, "C1:", label="hierarchical indicator")
    ax[0].set_xlabel("$\\omega/\\omega_1$"); ax[0].legend(fontsize=7)
    ax[1].loglog(e_true, resid, ".", ms=3, label="residual norm"); ax[1].loglog(e_true, hier, ".", ms=3, label="hierarchical")
    ax[1].set_xlabel("true error"); ax[1].set_ylabel("indicator"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch10_indicators.pdf")); plt.close(fig)
    print("spearman", sp_r, sp_h, flush=True)

    # ------------------------------------------------------------ C: certified bounds
    scm = SingularValueLowerBound.from_affine(rom5.affine)
    t0 = time.perf_counter(); hist_s = scm.greedy_train(cand, tol=0.05, max_references=15); t_s = time.perf_counter() - t0
    lp = LPSingularValueLowerBound.from_affine(rom5.affine)
    t0 = time.perf_counter(); hist_l = lp.greedy_train(cand, tol=0.05, max_references=15); t_l = time.perf_counter() - t0
    num("cfrScmRefs", str(len(scm.reference_omegas))); num("cfrLpRefs", str(len(lp.reference_omegas)))
    num("cfrScmGap", f"{hist_s[-1][1]:.2f}" if hist_s else "0"); num("cfrLpGap", f"{hist_l[-1][1]:.2f}" if hist_l else "0")
    # true sigma_min on a fine grid
    grid = np.linspace(0.1, 20.0, 200) * w1
    def smin(om):
        A = -om ** 2 * M + 1j * om * C + K
        return np.linalg.svd(A, compute_uv=False).min()
    sig = np.array([smin(om) for om in grid])
    lb_s = np.array([scm.lower_bound(om) for om in grid]); lb_l = np.array([lp.lower_bound(om) for om in grid])
    num("cfrScmValid", str(int(np.all(lb_s <= sig * (1 + 1e-9))))); num("cfrLpValid", str(int(np.all(lb_l <= sig * (1 + 1e-9)))))
    num("cfrScmPositive", str(int(np.sum(lb_s > 0)))); num("cfrLpPositive", str(int(np.sum(lb_l > 0)))); num("cfrNGrid", str(len(grid)))
    e_grid = np.array([np.linalg.norm(rom5.frequency_response([om], F)[0] - truth(om)) for om in grid])
    cb_s = np.array([certified_error_bound(rom5, scm, om, F) for om in grid])
    cb_l = np.array([lp_certified(rom5, lp, om, F) for om in grid])
    for nm, cb in (("Scm", cb_s), ("Lp", cb_l)):
        fin = np.isfinite(cb)
        num(f"cfr{nm}BoundHolds", str(int(np.sum(cb[fin] >= e_grid[fin] * (1 - 1e-9))))); num(f"cfr{nm}BoundFinite", str(int(fin.sum())))
        ratio = cb[fin] / np.maximum(e_grid[fin], 1e-300)
        num(f"cfr{nm}RatioMed", f"{np.median(ratio):.0f}"); num(f"cfr{nm}RatioMin", f"{ratio.min():.1f}"); num(f"cfr{nm}RatioMax", sci(ratio.max()))
        # near resonance: within 5% of a natural frequency
        nearres = np.zeros(len(grid), bool)
        for w in wn[:3]:
            nearres |= np.abs(grid / w - 1) < 0.05
        num(f"cfr{nm}InfNear", str(int(np.sum(~fin & nearres)))); num(f"cfr{nm}InfFar", str(int(np.sum(~fin & ~nearres))))
        num(f"cfr{nm}NearCount", str(int(nearres.sum())))
    # query cost
    t0 = time.perf_counter()
    for om in grid: scm.lower_bound(om)
    q_s = (time.perf_counter() - t0) / len(grid)
    t0 = time.perf_counter()
    for om in grid[:40]: lp.lower_bound(om)
    q_l = (time.perf_counter() - t0) / 40
    t0 = time.perf_counter()
    for om in grid[:40]: smin(om)
    q_f = (time.perf_counter() - t0) / 40
    num("cfrQueryScmUs", f"{q_s * 1e6:.0f}"); num("cfrQueryLpUs", f"{q_l * 1e6:.0f}"); num("cfrQuerySvdUs", f"{q_f * 1e6:.0f}")
    print("scm refs", len(scm.reference_omegas), "lp refs", len(lp.reference_omegas), "finite", np.isfinite(cb_s).sum(), np.isfinite(cb_l).sum(), flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax[0].semilogy(grid / w1, sig, "k-", label="true $\\sigma_{\\min}$")
    ax[0].semilogy(grid / w1, np.maximum(lb_s, 1e-3 * sig.min()), "C0--", label="Lipschitz bound")
    ax[0].semilogy(grid / w1, np.maximum(lb_l, 1e-3 * sig.min()), "C1:", label="LP (SCM) bound")
    ax[0].set_xlabel("$\\omega/\\omega_1$"); ax[0].legend(fontsize=7)
    ax[1].semilogy(grid / w1, e_grid, "k-", label="true error")
    ax[1].semilogy(grid / w1, np.where(np.isfinite(cb_s), cb_s, np.nan), "C0--", label="certified, Lipschitz")
    ax[1].semilogy(grid / w1, np.where(np.isfinite(cb_l), cb_l, np.nan), "C1:", label="certified, LP")
    ax[1].set_xlabel("$\\omega/\\omega_1$"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch10_bound.pdf")); plt.close(fig)

    # reference-count sweep: fraction of the grid where a finite certified bound exists
    sweep = []
    for nref in (15, 40, 100, 200):
        sc = SingularValueLowerBound.from_affine(rom5.affine)
        sc.greedy_train(cand, tol=1e-12, max_references=nref)
        cbs = np.array([certified_error_bound(rom5, sc, om, F) for om in grid])
        fin = np.isfinite(cbs)
        sweep.append(("Lipschitz", nref, int(fin.sum()), float(np.median(cbs[fin] / e_grid[fin])) if fin.any() else np.nan))
    for nref in (15, 40, 80):
        lq = LPSingularValueLowerBound.from_affine(rom5.affine)
        lq.greedy_train(cand, tol=1e-12, max_references=nref)
        cbs = np.array([lp_certified(rom5, lq, om, F) for om in grid])
        fin = np.isfinite(cbs)
        sweep.append(("LP", nref, int(fin.sum()), float(np.median(cbs[fin] / e_grid[fin])) if fin.any() else np.nan))
    for r_ in sweep:
        print("sweep", r_, flush=True)
    with open(os.path.join(TAB, "ch10_bound_sweep.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrr}\n\\toprule\nbound & references & finite at (of %d) & median bound / true error \\\\\n\\midrule\n" % len(grid))
        for nm, nref, nf, med in sweep:
            fh.write(f"{nm} & {nref} & {nf} & {'--' if np.isnan(med) else f'{med:.0f}'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cfrSweepLipMax", str(sweep[3][1])); num("cfrSweepLipFinite", str(sweep[3][2])); num("cfrSweepLipRatio", f"{sweep[3][3]:.1f}")
    num("cfrSweepLpMax", str(sweep[6][1])); num("cfrSweepLpFinite", str(sweep[6][2])); num("cfrSweepLpRatio", f"{sweep[6][3]:.1f}")

    # ------------------------------------------------------------ D: random vibration
    f_max = 20.0 * w1 / (2 * np.pi)
    rows = []
    for npts in (1500, 3000, 6000):
        fq = np.linspace(0.1 * w1 / (2 * np.pi), f_max, npts)
        psd = np.ones_like(fq)
        _, sig_fom = sysobj.solve_random_vibration(fq, psd, F_full, n_dof - 2)
        rows.append((npts, sig_fom))
    num("cfrRvGridA", sci(abs(rows[0][1] / rows[2][1] - 1))); num("cfrRvGridB", sci(abs(rows[1][1] / rows[2][1] - 1)))
    fq = np.linspace(0.1 * w1 / (2 * np.pi), f_max, 6000); psd = np.ones_like(fq)
    S_fom, sig_fom = sysobj.solve_random_vibration(fq, psd, F_full, n_dof - 2)
    num("cfrRvNPts", "6000"); num("cfrRvSig", f"{sig_fom:.4e}")
    out = []
    for r in (3, 5, 8):
        gb, _ = greedy_train_frequency_basis(band, M, K, F, rayleigh=(alpha, beta), n_seed=3, tol=1e-12, max_modes=r)
        for nm, V in (("greedy", gb.V), ("modal", Vm[:, :r])):
            rom = FrequencyROM.from_MCK(M, K, V, rayleigh=(alpha, beta))
            S, sg = psd_response(rom, fq, psd, F_full[free], tip)
            out.append((r, nm, sg, abs(sg / sig_fom - 1), np.max(np.abs(S - S_fom)) / S_fom.max()))
            print("psd", r, nm, out[-1], flush=True)
    with open(os.path.join(TAB, "ch10_psd.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rlrr}\n\\toprule\ncolumns & basis & RMS error & max PSD error / peak \\\\\n\\midrule\n")
        for r, nm, sg, e1, e2 in out:
            fh.write(f"{r} & {nm} & {sci(e1)} & {sci(e2)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cfrMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch10.tex"), "w") as fh:
        fh.write("% generated by scripts/ch10_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch12_assets.py -- figures, tables and number macros of Chapter 12 (non-intrusive identification and nonlinear ROMs).

Experiments
  A  Loewner identification of the modes of a damped cantilever from sampled frequency-response data (noise sweep)
  B  geometrically nonlinear clamped-clamped beam: polynomial (ICE) and RBF (multi-fidelity surrogate) modal ROMs
     against a held-out full-order static load sweep
  C  nonlinear normal mode backbone by harmonic balance against an independent time-integration period (Duffing)
  D  operator inference of a Craig-Bampton-structured model and unilateral-contact dynamics
Writes into book/: figures/ch12_loewner.pdf, ch12_nlrom.pdf, ch12_nnm.pdf, ch12_opinf.pdf;
tables/ch12_nlrom.tex, numbers_ch12.tex.

Run from the repository root:
    PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests python packages_book/book/scripts/ch12_assets.py
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
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

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


import fea_fixtures as ff                                                                  # noqa: E402
import loewner_fixtures as lf                                                              # noqa: E402
import test_operator_inference as toi                                                      # noqa: E402
from fea_engine.contact_solvers import solve_contact_dynamics                              # noqa: E402
from rom_engine.loewner import LoewnerROM                                                  # noqa: E402
from rom_engine.nonlinear_rom import PolynomialModalROM, MultiFidelitySurrogate, AppliedLoadStrategy  # noqa: E402
from rom_engine.nnm import HarmonicBalanceSystem, solve_nnm_backbone, solve_nnm_backbone_arclength   # noqa: E402
from rom_engine.operator_inference import (craig_bampton_transformation, infer_contact_operators,   # noqa: E402
                                           OperatorInferenceModel)


# ------------------------------------------------------------------------------------------ A
def exp_a():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    ix = np.ix_(free, free)
    Kff, Mff, Cff = fx["K"][ix], fx["M"][ix], fx["C"][ix]
    sysobj, n_dof = fx["sys"], fx["n_dof"]
    wn = np.sqrt(np.abs(eigh(Kff, Mff, eigvals_only=True)))
    f_und = wn / (2 * np.pi)
    fmin, fmax = 0.7 * f_und[0], 1.3 * f_und[3]
    f_true, eta_true, _ = lf.ground_truth_modes(Mff, Cff, Kff, fx["alpha"], fx["beta"], fmin, fmax)
    n_modes = len(f_true)
    tip = n_dof - 2
    F_full = np.zeros(n_dof); F_full[tip] = 1000.0
    rng0 = np.random.default_rng(7)
    n_pool = 100
    f_pool = np.linspace(fmin, fmax, n_pool) * (1 + 1e-3 * rng0.standard_normal(n_pool))
    om_pool = 2 * np.pi * f_pool
    x_pool = np.array([sysobj.solve_harmonic(w, F_full)[tip] for w in om_pool])
    num("cldNModes", str(n_modes)); num("cldFOne", f"{f_true[0]:.1f}"); num("cldBandLo", f"{fmin:.0f}"); num("cldBandHi", f"{fmax:.0f}")
    num("cldEtaTrueMin", sci(eta_true.min())); num("cldEtaTrueMax", sci(eta_true.max())); num("cldFreqs", ", ".join(f"{x:.0f}" for x in f_true))
    noises = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
    names = ("Zero", "FourE", "ThreeE", "TwoE", "OneE")
    n_draw = 30
    stats = []
    for nz in noises:
        errs_f, errs_eta, found = [], [], 0
        for d in range(n_draw):
            rng = np.random.default_rng(100 + d)
            noise = nz * np.abs(x_pool) * (rng.standard_normal(n_pool) + 1j * rng.standard_normal(n_pool)) / np.sqrt(2)
            xn = x_pool + noise
            n_interp = n_modes + 3
            idx = rng.choice(n_pool, size=2 * n_interp, replace=False)
            ia, ib = idx[:n_interp], idx[n_interp:]
            rom = LoewnerROM.fit(om_pool[ia], om_pool[ib], xn[ia], xn[ib])
            inb = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1)
            f_id, eta_id = rom.f[inb], rom.eta[inb]
            if len(f_id) == 0:
                continue
            ef, ee = [], []
            for i in range(n_modes):
                j = np.argmin(np.abs(f_id - f_true[i]))
                ef.append(abs(f_id[j] - f_true[i]) / f_true[i]); ee.append(abs(eta_id[j] - eta_true[i]))
            if max(ef) < 0.02:                      # success: every true mode found within 2 percent
                found += 1; errs_f.append(max(ef)); errs_eta.append(max(ee))
        stats.append((nz, found, np.median(errs_f) if errs_f else np.nan, np.max(errs_f) if errs_f else np.nan,
                      np.median(errs_eta) if errs_eta else np.nan))
        print("loewner", stats[-1], flush=True)
    for (nz, found, mf, xf, me), nm in zip(stats, names):
        num(f"cld{nm}FreqMed", sci(mf) if np.isfinite(mf) else "--"); num(f"cld{nm}FreqMax", sci(xf) if np.isfinite(xf) else "--"); num(f"cld{nm}EtaMed", sci(me) if np.isfinite(me) else "--"); num(f"cld{nm}Found", str(found))
    num("cldNDraw", str(n_draw)); num("cldNInterp", str(n_modes + 3)); num("cldNPool", str(n_pool))
    # one clean fit for the figure: frequencies, and number of spurious in-band eigenvalues
    rng = np.random.default_rng(100)
    idx = rng.choice(n_pool, size=2 * (n_modes + 3), replace=False)
    ia, ib = idx[:n_modes + 3], idx[n_modes + 3:]
    rom = LoewnerROM.fit(om_pool[ia], om_pool[ib], x_pool[ia], x_pool[ib])
    inb = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1)
    num("cldRawEig", str(len(rom.f))); num("cldInBand", str(int(inb.sum())))
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax[0].semilogx([max(s[0], 1e-5) for s in stats], [s[1] / n_draw for s in stats], "o-")
    ax[0].set_xlabel("relative noise on the response samples (0 plotted at $10^{-5}$)"); ax[0].set_ylabel("fraction of draws with all modes within 2 %")
    ax[1].plot(rom.f[inb], rom.eta[inb], "o", label="identified (in band)")
    ax[1].plot(f_true, eta_true, "k+", ms=10, label="damped eigenproblem")
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("loss factor $\\eta$"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch12_loewner.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ B
def exp_b():
    N_ELEM = 16
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3))
    res_store = {}
    for n_modes in (2, 3, 4):
        fx = clamped(N_ELEM, n_modes)
        V, M_ff, freq_hz, free = fx["V"], fx["M_ff"], fx["freq_hz"], fx["free_dofs"]
        Lam = (2 * np.pi * freq_hz) ** 2
        mid_local = list(free).index((N_ELEM // 2) * 3 + 1)
        rng = np.random.default_rng(0)
        strat = AppliedLoadStrategy(target_fracs=(0.3, 3.0), reference_scale=0.01, n_samples=45, rng=rng)
        t0 = time.perf_counter()
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freq_hz, fx["mode_shape_peaks"], fx["fom_solver"])
        t_train = time.perf_counter() - t0
        ice = PolynomialModalROM(n_modes).fit(q_nl, F_nl)
        mfs = MultiFidelitySurrogate().fit(q_l, F_nl)
        shape = np.zeros(len(free)); shape[mid_local] = -1.0
        f_hat = V.T @ shape
        loads = np.linspace(0.0, 6000.0, 12)
        mid_f, mid_i, mid_m, mid_lin = [], [], [], []
        t_fom = t_ice = t_mfs = t_lift = 0.0
        g_ice = g_mfs = 0.0
        q_prev = np.zeros(n_modes)
        for P in loads:
            t0 = time.perf_counter(); u = fx["fom_solver"](P * shape, n_steps=30); t_fom += time.perf_counter() - t0
            mid_f.append(u[mid_local])
            t0 = time.perf_counter(); qs, conv = ice.predict(F_ext=P * f_hat, Lambda=Lam, q0=q_prev); t_ice += time.perf_counter() - t0
            t0 = time.perf_counter(); ui = V @ qs; t_lift += time.perf_counter() - t0
            q_prev = qs; mid_i.append(ui[mid_local])
            t0 = time.perf_counter(); qn = mfs.reconstruct_q_nl(P * f_hat / Lam, Lam); t_mfs += time.perf_counter() - t0
            mid_m.append((V @ qn)[mid_local])
            if P > 0:
                g_ice = max(g_ice, np.linalg.norm(ui - u) / np.linalg.norm(u)); g_mfs = max(g_mfs, np.linalg.norm(V @ qn - u) / np.linalg.norm(u))
            mid_lin.append((V @ (P * f_hat / Lam))[mid_local])
        mid_f, mid_i, mid_m, mid_lin = map(np.array, (mid_f, mid_i, mid_m, mid_lin))
        Kff = np.asarray(fx["K"])[np.ix_(free, free)]
        u_lin_full = np.linalg.solve(Kff, loads[-1] * shape)[mid_local]
        nl_effect = abs(mid_f[-1] - u_lin_full) / abs(mid_f[-1])
        rel = lambda a: np.abs(a[1:] - mid_f[1:]) / np.abs(mid_f[1:])
        rows.append(dict(r=n_modes, ice=rel(mid_i).max(), mfs=rel(mid_m).max(), lin=rel(mid_lin).max(), ice_end=rel(mid_i)[-1],
                         mfs_end=rel(mid_m)[-1], t_train=t_train, t_fom=t_fom / len(loads), t_ice=t_ice / len(loads), t_mfs=t_mfs / len(loads),
                         mid_end=mid_f[-1], nl=nl_effect, h=None, t_lift=t_lift / len(loads), g_ice=g_ice, g_mfs=g_mfs))
        res_store[n_modes] = (loads, mid_f, mid_i, mid_m, mid_lin)
        print("nlrom", n_modes, {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    loads, mf, mi, mm, ml = res_store[3]
    ax[0].plot(loads, -mf * 1000, "ko-", ms=4, label="full order")
    ax[0].plot(loads, -ml * 1000, "C3:", label="linear (3 modes)")
    ax[0].plot(loads, -mi * 1000, "C0--", label="polynomial (ICE), 3 modes")
    ax[0].plot(loads, -mm * 1000, "C1-.", label="RBF surrogate, 3 modes")
    ax[0].set_xlabel("central point load (N)"); ax[0].set_ylabel("mid-span deflection (mm)"); ax[0].legend(fontsize=7)
    for key, st, lab in (("ice", "o-", "polynomial (ICE)"), ("mfs", "s--", "RBF surrogate"), ("lin", "^:", "linear")):
        ax[1].semilogy([r["r"] for r in rows], [r[key] for r in rows], st, label=lab)
    ax[1].set_xlabel("retained modes"); ax[1].set_ylabel("max relative error of mid-span deflection"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch12_nlrom.pdf")); plt.close(fig)
    with open(os.path.join(TAB, "ch12_nlrom.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrr}\n\\toprule\nmodes & linear & polynomial (ICE) & RBF surrogate \\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r['r']} & {sci(r['lin'])} & {sci(r['ice'])} & {sci(r['mfs'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    d = {r["r"]: r for r in rows}
    for r, nm in ((2, "Two"), (3, "Three"), (4, "Four")):
        num(f"cnlLin{nm}", sci(d[r]["lin"])); num(f"cnlIce{nm}", sci(d[r]["ice"])); num(f"cnlMfs{nm}", sci(d[r]["mfs"]))
    num("cnlDefl", f"{abs(d[3]['mid_end']) * 1000:.1f}"); num("cnlNlEffect", sci(d[3]["nl"])); num("cnlNlEffectPct", f"{100 * d[3]['nl']:.0f}"); num("cnlNLoads", "12"); num("cnlMaxLoad", "6000"); num("cnlTrainSec", f"{d[3]['t_train']:.1f}")
    num("cnlFomMs", f"{d[3]['t_fom'] * 1000:.0f}"); num("cnlIceMs", f"{d[3]['t_ice'] * 1000:.2f}"); num("cnlMfsMs", f"{d[3]['t_mfs'] * 1000:.3f}")
    num("cnlSpeedIce", f"{d[3]['t_fom'] / d[3]['t_ice']:.0f}"); num("cnlSpeedMfs", f"{d[3]['t_fom'] / d[3]['t_mfs']:.0f}")
    num("cnlLiftUs", f"{d[3]['t_lift'] * 1e6:.0f}"); num("cnlGlobIce", sci(d[3]["g_ice"])); num("cnlGlobMfs", sci(d[3]["g_mfs"]))
    num("cnlLocIce", sci(d[3]["ice"])); num("cnlLocMfs", sci(d[3]["mfs"]))
    sav = d[3]["t_fom"] - d[3]["t_ice"] - d[3]["t_lift"]
    num("cnlBreakEvenIce", f"{d[3]['t_train'] / sav:.0f}")
    sav2 = d[3]["t_fom"] - d[3]["t_mfs"] - d[3]["t_lift"]
    num("cnlBreakEvenMfs", f"{d[3]['t_train'] / sav2:.0f}")
    num("cnlSpeedIceLift", f"{d[3]['t_fom'] / (d[3]['t_ice'] + d[3]['t_lift']):.0f}"); num("cnlSpeedMfsLift", f"{d[3]['t_fom'] / (d[3]['t_mfs'] + d[3]['t_lift']):.0f}")
    num("cnlNElem", str(N_ELEM)); num("cnlNDof", str(fx["n_dof"]))
    # sample-count sensitivity for ICE, 3 modes
    fx = clamped(N_ELEM, 3)
    V, M_ff, freq_hz, free = fx["V"], fx["M_ff"], fx["freq_hz"], fx["free_dofs"]
    Lam = (2 * np.pi * freq_hz) ** 2
    mid_local = list(free).index((N_ELEM // 2) * 3 + 1)
    shape = np.zeros(len(free)); shape[mid_local] = -1.0
    f_hat = V.T @ shape
    loads, mf = res_store[3][0], res_store[3][1]
    out = []
    for ns in (10, 20, 45, 90):
        strat = AppliedLoadStrategy(target_fracs=(0.3, 3.0), reference_scale=0.01, n_samples=ns, rng=np.random.default_rng(0))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freq_hz, fx["mode_shape_peaks"], fx["fom_solver"])
        ice = PolynomialModalROM(3).fit(q_nl, F_nl); mfs = MultiFidelitySurrogate().fit(q_l, F_nl)
        q_prev = np.zeros(3); ei, em = [], []
        for P, ref in zip(loads[1:], mf[1:]):
            qs, conv = ice.predict(F_ext=P * f_hat, Lambda=Lam, q0=q_prev); q_prev = qs
            ei.append(abs((V @ qs)[mid_local] - ref) / abs(ref))
            em.append(abs((V @ mfs.reconstruct_q_nl(P * f_hat / Lam, Lam))[mid_local] - ref) / abs(ref))
        out.append((ns, max(ei), max(em)))
        print("samples", out[-1], flush=True)
    d2 = {o[0]: o for o in out}
    num("cnlSamTenIce", sci(d2[10][1])); num("cnlSamTenMfs", sci(d2[10][2])); num("cnlSamNinetyIce", sci(d2[90][1])); num("cnlSamNinetyMfs", sci(d2[90][2]))
    num("cnlSamTwentyIce", sci(d2[20][1])); num("cnlSamTwentyMfs", sci(d2[20][2]))


def clamped(n_elem, n_modes):
    return ff.clamped_clamped_nonlinear_beam_system(n_elem=n_elem, n_modes=n_modes, L=1.0)


# ------------------------------------------------------------------------------------------ C
class CubicQnl:
    def __init__(self, k3):
        self.k3 = np.asarray(k3, float)

    def predict(self, q):
        return self.k3 * np.asarray(q) ** 3

    def jacobian(self, q):
        return np.diag(3 * self.k3 * np.asarray(q) ** 2)


def measure_duffing(lam, k3, a, n_periods=3):
    """Frequency of the Duffing oscillator x'' + lam x + k3 x^3 = 0 released at rest from x = a, by integration."""
    rhs = lambda t, y: [y[1], -lam * y[0] - k3 * y[0] ** 3]
    T0 = 2 * np.pi / np.sqrt(lam)
    ev = lambda t, y: y[1]
    ev.direction = 1
    sol = solve_ivp(rhs, [0, 5 * T0], [a, 0.0], events=ev, rtol=1e-12, atol=1e-14, dense_output=False)
    te = sol.t_events[0]
    if len(te) < 2:
        return np.nan
    return 2 * np.pi / np.mean(np.diff(te))


def exp_c():
    lam, k3 = 100.0, 50000.0
    amps = np.linspace(0.005, 0.03, 6)
    ref = np.array([measure_duffing(lam, k3, a) for a in amps])
    rows = []
    for nh in (1, 3, 7):
        hb = HarmonicBalanceSystem(np.array([lam]), n_harmonics=nh, n_time_samples=max(128, 8 * nh + 8))
        a_, om, Z, conv = solve_nnm_backbone(hb, CubicQnl([k3]), (amps[0], amps[-1]), n_points=len(amps), domain="q_nl")
        peak = Z[:, 0, 0] + sum(Z[:, 2 * k - 1, 0] for k in range(1, nh + 1))      # q(0): the release amplitude
        ref_h = np.array([measure_duffing(lam, k3, p) for p in peak])
        err = np.abs(om - ref_h) / ref_h
        rows.append((nh, err.max(), bool(conv.all())))
        print("nnm", nh, f"{err.max():.2e}", conv.all(), flush=True)
    d = {r[0]: r for r in rows}
    num("cnmErrOne", sci(d[1][1])); num("cnmErrThree", sci(d[3][1])); num("cnmErrSeven", sci(d[7][1]))
    num("cnmConvAll", "yes" if all(r[2] for r in rows) else "no")
    # closed form of single-harmonic balance: omega^2 = lam + (3/4) k3 a^2
    hb1 = HarmonicBalanceSystem(np.array([lam]), n_harmonics=1, n_time_samples=64)
    _, om1, _, _ = solve_nnm_backbone(hb1, CubicQnl([k3]), (amps[0], amps[-1]), n_points=len(amps), domain="q_nl")
    num("cnmClosedForm", sci(np.max(np.abs(om1 - np.sqrt(lam + 0.75 * k3 * amps ** 2)) / om1)))
    num("cnmHard", f"{ref[-1] / np.sqrt(lam):.2f}"); num("cnmAmpMax", f"{amps[-1]:.3f}")
    # natural-parameter against arclength driver on the same backbone
    hb7 = HarmonicBalanceSystem(np.array([lam]), n_harmonics=7, n_time_samples=128)
    a_n, om_n, _, conv_n = solve_nnm_backbone(hb7, CubicQnl([k3]), (amps[0], amps[-1]), n_points=len(amps), domain="q_nl")
    try:
        res = solve_nnm_backbone_arclength(hb7, CubicQnl([k3]), a0=amps[0], n_points=30, domain="q_nl")
        a_a, om_a = res[0], res[1]
        ok = np.isfinite(om_a)
        inr = amps <= a_a[ok].max() * (1 + 1e-12)
        om_i = np.interp(amps[inr], a_a[ok], om_a[ok])
        num("cnmArcVsNat", sci(np.max(np.abs(om_i - om_n[inr]) / om_n[inr]))); num("cnmArcPts", str(int(inr.sum())))
        num("cnmArcMaxAmp", f"{a_a[ok].max():.3f}")
    except Exception as ex:
        print("arclength failed", ex, flush=True)
        num("cnmArcVsNat", "failed"); num("cnmArcMaxAmp", "n/a")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(amps, ref / np.sqrt(lam), "ko", label="time integration")
    for nh, st in ((1, "C3--"), (7, "C0-")):
        hb = HarmonicBalanceSystem(np.array([lam]), n_harmonics=nh, n_time_samples=max(128, 8 * nh + 8))
        a_, om, _, _ = solve_nnm_backbone(hb, CubicQnl([k3]), (amps[0], amps[-1]), n_points=40, domain="q_nl")
        ax.plot(a_, om / np.sqrt(lam), st, label=f"harmonic balance, {nh} harmonic{'s' if nh > 1 else ''}")
    ax.set_xlabel("amplitude"); ax.set_ylabel("$\\omega/\\omega_0$"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch12_nnm.pdf")); plt.close(fig)
    # backbone of the 3-mode polynomial beam ROM (no full-order reference in this book)
    fx = clamped(16, 3)
    V, M_ff, freq_hz = fx["V"], fx["M_ff"], fx["freq_hz"]
    Lam = (2 * np.pi * freq_hz) ** 2
    strat = AppliedLoadStrategy(target_fracs=(0.3, 3.0), reference_scale=0.01, n_samples=45, rng=np.random.default_rng(0))
    q_l, q_nl, F_nl = strat.generate(V, M_ff, freq_hz, fx["mode_shape_peaks"], fx["fom_solver"])
    ice = PolynomialModalROM(3).fit(q_nl, F_nl)
    hb = HarmonicBalanceSystem(Lam, n_harmonics=5, n_time_samples=64)
    a_b, om_b, _, conv_b = solve_nnm_backbone(hb, ice, (1e-4, 0.01), n_points=8, domain="q_nl")
    num("cnmBeamConv", f"{int(conv_b.sum())}"); num("cnmBeamPts", str(len(conv_b)))
    num("cnmBeamHard", f"{om_b[conv_b][-1] / np.sqrt(Lam[0]):.3f}" if conv_b.any() else "n/a")
    num("cnmBeamMonotone", "yes" if np.all(np.diff(om_b[conv_b]) > 0) else "no")


# ------------------------------------------------------------------------------------------ D
def exp_d():
    ctx_K, ctx_M = toi.system()
    N, BND, R = toi.N, toi.BND, toi.R
    interior = np.setdiff1d(np.arange(N), BND)
    Kii, Mii, Kib = ctx_K[np.ix_(interior, interior)], ctx_M[np.ix_(interior, interior)], ctx_K[np.ix_(interior, BND)]
    w2, V = eigh(Kii, Mii)
    Phi, _ = np.linalg.qr(V[:, :R])
    Psi = -np.linalg.solve(Kii, Kib)
    U, A, F = toi.training_data(ctx_K, ctx_M, Phi)
    T = craig_bampton_transformation(BND, N, Phi, Psi)
    Mg, Kg = T.T @ ctx_M @ T, T.T @ ctx_K @ T
    rows = []
    for nz in (0.0, 1e-3, 1e-2, 1e-1, 5e-1):
        rng = np.random.default_rng(5)
        An = A + nz * np.abs(A).max() * rng.standard_normal(A.shape)
        model = infer_contact_operators(U, An, F, BND, Phi, Psi, max_nfev=60)
        eM = np.linalg.norm(model.M - Mg) / np.linalg.norm(Mg); eK = np.linalg.norm(model.K - Kg) / np.linalg.norm(Kg)
        spd = bool(np.linalg.eigvalsh(model.M).min() > 0 and np.linalg.eigvalsh(model.K).min() > 0)
        rows.append((nz, eM, eK, spd, model.info["spd_unconstrained"]))
        print("opinf", rows[-1], flush=True)
    names = ("Zero", "ThreeE", "TwoE", "OneE", "Half")
    for (nz, eM, eK, spd, spd0), nm in zip(rows, names):
        num(f"coiM{nm}", sci(eM)); num(f"coiK{nm}", sci(eK)); num(f"coiSpd{nm}", "yes" if spd else "no"); num(f"coiSpdRaw{nm}", "yes" if spd0 else "no")
    model = infer_contact_operators(U, A, F, BND, Phi, Psi)
    num("coiProjRes", sci(model.info["projection_residual"])); num("coiNFull", str(N)); num("coiNRed", str(model.n_reduced))
    # contact dynamics
    galerkin = OperatorInferenceModel(Mg, Kg, T, BND)
    B = np.zeros((1, N)); B[0, N - 1] = 1.0
    c = np.array([-0.02]); h, steps = 2e-3, 1500
    f = np.zeros(N); f[N - 1] = -0.3
    force = lambda t: f
    Ufom, Lfom = solve_contact_dynamics(ctx_M, ctx_K, force, B, c, h, steps)
    scale = np.abs(Ufom).max()
    errs = {}
    pen = {}
    tl = {}
    for nm, mod in (("inferred", model), ("galerkin", galerkin)):
        t0 = time.perf_counter()
        X, Ured, Lred = mod.simulate_contact(force, B, c, h, steps)
        tl[nm] = time.perf_counter() - t0
        errs[nm] = np.abs(Ured - Ufom).max() / scale
        pen[nm] = float(Ured[:, N - 1].min())
    t0 = time.perf_counter(); solve_contact_dynamics(ctx_M, ctx_K, force, B, c, h, steps); tfom = time.perf_counter() - t0
    num("coiContInf", sci(errs["inferred"])); num("coiContGal", sci(errs["galerkin"]))
    num("coiPenInf", f"{pen['inferred']:.4f}"); num("coiPenGal", f"{pen['galerkin']:.4f}")
    num("coiLamFom", f"{Lfom.max():.3f}")
    num("coiTimeFom", f"{tfom * 1000:.0f}"); num("coiTimeInf", f"{tl['inferred'] * 1000:.0f}")
    t = np.arange(steps + 1) * h
    X, Ured, Lred = model.simulate_contact(force, B, c, h, steps)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax[0].plot(t, Ufom[N - 1] if Ufom.shape[0] == N else Ufom[:, N - 1], "k-", label="full order")
    ax[0].plot(t, Ured[:, N - 1], "C1--", label="inferred, 12 unknowns"); ax[0].axhline(-0.02, color="gray", ls=":")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("displacement of the contact dof"); ax[0].legend(fontsize=7)
    ax[1].semilogy([max(r[0], 1e-4) for r in rows], [r[1] for r in rows], "o-", label="mass $M$")
    ax[1].semilogy([max(r[0], 1e-4) for r in rows], [r[2] for r in rows], "s--", label="stiffness $K$")
    ax[1].set_xlabel("relative noise on the accelerations (0 plotted at $10^{-4}$)"); ax[1].set_ylabel("relative error of the operator"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch12_opinf.pdf")); plt.close(fig)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_a(); exp_b(); exp_c(); exp_d()
    num("cldMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch12.tex"), "w") as fh:
        fh.write("% generated by scripts/ch12_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

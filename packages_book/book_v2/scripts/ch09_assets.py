# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch09_assets.py -- figures, tables and number macros of Chapter 9 (balanced truncation, Krylov, SOAR, passivity).

Model: the damped cantilever of rom_engine/examples/gallery/rom_04_balanced_truncation.py (24 Euler-Bernoulli elements,
Rayleigh damping alpha=2, beta=1e-5, collocated tip force -> tip displacement). The independent reference is the
full-order fea_engine frequency sweep.
Writes into book/:
    figures/ch09_bt.pdf        Hankel singular values; actual error and a-priori bound against the retained order
    figures/ch09_compare.pdf   error against reduced state count for modal truncation, balanced truncation, Krylov, SOAR
    figures/ch09_response.pdf  frequency responses of the order-8 models
    tables/ch09_stability.tex  stability and passivity of the reduced models
    tables/numbers_ch09.tex    every number quoted in the prose

Run from the repository root:   PYTHONPATH=fea_engine/src:rom_engine/src python packages_book/book/scripts/ch09_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


from fea_engine.material import EI_beam, Material, Section                                   # noqa: E402
from fea_engine.geometry import generate_mesh                                                 # noqa: E402
from fea_engine.solver import FESystem                                                        # noqa: E402
from fea_engine.damping import RayleighDamping                                                # noqa: E402
from fea_engine import elements                                                               # noqa: E402
from rom_engine.balanced_truncation import BalancedTruncationROM                              # noqa: E402
from rom_engine.krylov import KrylovROM                                                       # noqa: E402
from rom_engine.soar import SOARROM, soar_basis                                                           # noqa: E402
from rom_engine.passivity import velocity_transfer_function, passivity_margin                 # noqa: E402

N_EL, L = 24, 1.0
E, RHO, A_SEC, I_SEC = 210e9, 7800.0, 0.01, 8.33e-6
ALPHA, BETA = 2.0, 1e-5
N_PRE = 15


def build():
    mesh = generate_mesh(dim=1, L=L, n=N_EL)
    fes = FESystem(mesh, elements.Beam2DEulerBernoulli())
    mat, sec = Material(E=E, nu=0.3, rho=RHO), Section(A=A_SEC, I=I_SEC)
    fes.assemble_stiffness(EI_beam(mat, sec)); fes.assemble_mass(RHO * A_SEC)
    fes.assemble_damping(RayleighDamping(alpha=ALPHA, beta=BETA))
    fes.fix_dofs([0], [0, 1])
    free = np.asarray(fes.free_dofs)
    ix = np.ix_(free, free)
    K, M, C = fes.K[ix], fes.M[ix], fes.C[ix]
    tip = len(free) - 2
    B = np.zeros(len(free)); B[tip] = 1.0
    return fes, free, K, M, C, B, tip


def h_second_order(Mr, Cr, Kr, Br, Cor, omegas):
    return np.array([(Cor @ np.linalg.solve(-w * w * Mr + 1j * w * Cr + Kr, Br)).item() for w in omegas])


def soar_rescaled(M, C, K, B, k, w1):
    """SOAR with the load rescaled to unit-norm static response before the basis is built (see the text: the package's
    deflation tolerance is absolute). The basis depends on B only through its direction, so the model is the same."""
    scale = 1.0 / np.linalg.norm(np.linalg.solve(K, B))
    # time rescaling s = w1 * sigma: (sigma^2 w1^2 M + sigma w1 C + K); the Krylov subspace about s0 = 0 is unchanged
    V = soar_basis(w1 ** 2 * M, K, B * scale, k, C=w1 * C)
    return SOARROM(M, C, K, B.reshape(-1, 1), B.reshape(1, -1), V, 0.0)


def errs(H, Hf):
    e = np.abs(H - Hf)
    return e.max() / np.abs(Hf).max(), np.mean(e / np.abs(Hf))


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    fes, free, K, M, C, B, tip = build()
    n = len(free)
    lam, Vf = eigh(K, M)
    w1 = np.sqrt(lam[0])
    omegas = np.linspace(0.05, 8.0, 400) * w1
    F0 = np.zeros(fes.n_dof); F0[free[tip]] = 1.0
    Hf = fes.solve_frequency_sweep(omegas, F0)[:, free[tip]]
    # self-check of the reference against a direct solve of the same matrices
    Hd = h_second_order(M, C, K, B, B, omegas[::40])
    num("cbtNDof", str(n)); num("cbtNState", str(2 * n)); num("cbtBandMax", "8"); num("cbtFOne", f"{w1 / (2 * np.pi):.1f}")
    num("cbtNPre", str(N_PRE))

    # SOAR on the unscaled load: report what the package returns
    Vraw = soar_basis(M, K, B, 4, C=C)
    Vscl = soar_basis(w1 ** 2 * M, K, B / np.linalg.norm(np.linalg.solve(K, B)), 4, C=w1 * C)
    num("cbtSoarScaledCols", str(Vscl.shape[1]))
    num("cbtSoarRawCols", str(Vraw.shape[1])); num("cbtStaticNorm", sci(np.linalg.norm(np.linalg.solve(K, B))))

    # ---------------- modal pre-reduction (as in the example) and the intermediate model
    Vp = Vf[:, :N_PRE]
    Mp, Kp, Cp, Bp = Vp.T @ M @ Vp, Vp.T @ K @ Vp, Vp.T @ C @ Vp, Vp.T @ B
    Hpre = h_second_order(Mp, Cp, Kp, Bp, Bp, omegas)
    ep = errs(Hpre, Hf)
    num("cbtPreErrMax", sci(ep[0])); num("cbtPreErrMean", sci(ep[1]))

    # ---------------- balanced truncation: Hankel values, bound, error against order
    full = BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=2 * N_PRE)
    hsv = full.hsv
    Hint = full.frequency_response(omegas)                     # exact transfer function of the intermediate model
    num("cbtHsvRatioTen", sci(hsv[9] / hsv[0])); num("cbtHsvRatioTwenty", sci(hsv[19] / hsv[0]))
    orders = np.arange(2, 2 * N_PRE, 2)
    bt_bound, bt_err_int, bt_err_fom, bt_mean = [], [], [], []
    holds, stable_all = 0, 0
    miss = []
    for r in orders:
        rom = BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=int(r))
        H = rom.frequency_response(omegas)
        b = rom.h_infinity_error_bound()
        ei = np.max(np.abs(H - Hint))
        bt_bound.append(b); bt_err_int.append(ei)
        bt_err_fom.append(errs(H, Hf)[0]); bt_mean.append(errs(H, Hf)[1])
        holds += int(ei <= b * 1.001); stable_all += int(rom.is_stable())
        if ei > b * 1.001:
            miss.append((int(r), ei, b))
    num("cbtBoundHolds", str(holds)); num("cbtBoundCases", str(len(orders))); num("cbtBtStable", str(stable_all))
    for k_, (r_, e_, b_) in enumerate(miss[:1]):
        num("cbtMissOrder", str(r_)); num("cbtMissErr", sci(e_)); num("cbtMissBound", sci(b_)); num("cbtMissRel", sci(e_ / hsv[0]))
    if not miss:
        num("cbtMissOrder", "none"); num("cbtMissErr", "0"); num("cbtMissBound", "0"); num("cbtMissRel", "0")
    ratio = [e / b for e, b in zip(bt_err_int, bt_bound) if b > 1e-12 * hsv[0]]
    num("cbtBoundRatioMin", f"{min(ratio):.2f}"); num("cbtBoundRatioMax", f"{max(ratio):.2f}")
    i8 = list(orders).index(8)
    num("cbtBtErrEight", sci(bt_err_fom[i8])); num("cbtBtBoundEight", sci(bt_bound[i8]))
    num("cbtBtErrIntEight", sci(bt_err_int[i8]))
    print("BT bound/actual r=8", bt_bound[i8], bt_err_int[i8], "holds", holds, "of", len(orders), flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax[0].semilogy(np.arange(1, len(hsv) + 1), hsv / hsv[0], "o-", ms=3); ax[0].set_xlabel("index"); ax[0].set_ylabel("Hankel singular value / largest")
    ax[1].semilogy(orders, bt_bound, "s-", ms=3, label="a priori bound $2\\sum_{k>r}\\sigma_k$")
    ax[1].semilogy(orders, bt_err_int, "o-", ms=3, label="actual error vs intermediate model")
    ax[1].set_xlabel("retained states $r$"); ax[1].set_ylabel("$H_\\infty$ error (grid maximum)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch09_bt.pdf")); plt.close(fig)

    # balanced truncation applied to the raw full model
    try:
        raw = BalancedTruncationROM.from_MCK(M, K, B, B, C=C, r=8)
        Hraw = raw.frequency_response(omegas)
        er = errs(Hraw, Hf)
        num("cbtRawErrMax", sci(er[0])); num("cbtRawStable", "yes" if raw.is_stable() else "no")
        num("cbtRawHsvOne", sci(raw.hsv[0])); num("cbtPreHsvOne", sci(hsv[0]))
        print("raw BT r=8", er, raw.is_stable(), flush=True)
    except Exception as ex:                                     # report, do not hide
        num("cbtRawErrMax", "failed"); num("cbtRawStable", "n/a"); num("cbtRawHsvOne", "n/a"); num("cbtPreHsvOne", "n/a")
        print("raw BT failed:", ex, flush=True)

    # ---------------- comparison at equal reduced state count
    methods = {k: ([], []) for k in ("modal", "bt", "kry1", "kry2", "soar")}
    rom_store = {}
    ords = [4, 6, 8, 10, 12, 16, 20]
    for r in ords:
        # modal truncation of the damped beam: r/2 undamped modes (r states)
        V = Vf[:, :r // 2]
        H = h_second_order(V.T @ M @ V, V.T @ C @ V, V.T @ K @ V, V.T @ B, V.T @ B, omegas)
        methods["modal"][0].append(errs(H, Hf)); rom_store[("modal", r)] = H
        rom = BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=r)
        H = rom.frequency_response(omegas); methods["bt"][0].append(errs(H, Hf)); rom_store[("bt", r)] = H
        k1 = KrylovROM.from_MCK(M, K, B, B, C=C, s0=0.0, k=r)
        H = k1.frequency_response(omegas); methods["kry1"][0].append(errs(H, Hf)); rom_store[("kry1", r)] = H
        rom_store[("kry1rom", r)] = k1
        try:
            k2 = KrylovROM.from_MCK(M, K, B, B, C=C, s0=0.0, k=r // 2, two_sided=True)
            H = k2.frequency_response(omegas); methods["kry2"][0].append(errs(H, Hf)); rom_store[("kry2", r)] = H
            rom_store[("kry2rom", r)] = k2
        except Exception as ex:
            methods["kry2"][0].append((np.nan, np.nan)); print("two-sided failed at", r, ex, flush=True)
        so = soar_rescaled(M, C, K, B, r // 2, w1)
        H = so.frequency_response(omegas); methods["soar"][0].append(errs(H, Hf)); rom_store[("soar", r)] = H
        rom_store[("soarrom", r)] = so
    for r_i, r in enumerate(ords):
        print(r, {k: f"{v[0][r_i][0]:.1e}/{v[0][r_i][1]:.1e}" for k, v in methods.items()}, flush=True)
    labels = {"modal": "modal truncation", "bt": "balanced truncation", "kry1": "Krylov, one-sided", "kry2": "Krylov, two-sided", "soar": "SOAR"}
    sty = {"modal": "x--", "bt": "o-", "kry1": "s-", "kry2": "^-", "soar": "d-"}
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=False)
    for k in methods:
        ax[0].semilogy(ords, [e[0] for e in methods[k][0]], sty[k], ms=4, label=labels[k])
        ax[1].semilogy(ords, [e[1] for e in methods[k][0]], sty[k], ms=4, label=labels[k])
    ax[0].set_ylabel("max error / peak $|H|$"); ax[1].set_ylabel("mean relative error $|H_r-H|/|H|$")
    for a in ax:
        a.set_xlabel("reduced state count");
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch09_compare.pdf")); plt.close(fig)
    ix = {r: i for i, r in enumerate(ords)}
    for tag, key in (("Modal", "modal"), ("Bt", "bt"), ("KryOne", "kry1"), ("KryTwo", "kry2"), ("Soar", "soar")):
        for r, nm in ((8, "Eight"), (12, "Twelve")):
            e = methods[key][0][ix[r]]
            num(f"cbt{tag}Max{nm}", sci(e[0])); num(f"cbt{tag}Mean{nm}", sci(e[1]))
    # ---------------- frequency response figure, order 8
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.semilogy(omegas / w1, np.abs(Hf), "k-", lw=2, label="full order")
    for k, c in (("bt", "C1"), ("kry1", "C2"), ("soar", "C3"), ("kry2", "C4")):
        ax.semilogy(omegas / w1, np.abs(rom_store[(k, 8)]), "--", color=c, lw=1.2, label=labels[k])
    ax.set_xlabel("$\\omega/\\omega_1$"); ax.set_ylabel("$|$tip displacement / tip force$|$"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch09_response.pdf")); plt.close(fig)

    # locality: error against frequency on an extended band, order 8
    om2 = np.linspace(0.05, 25.0, 700) * w1
    Hf2 = fes.solve_frequency_sweep(om2, F0)[:, free[tip]]
    cand = {"bt": BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=8).frequency_response(om2),
            "kry1": rom_store[("kry1rom", 8)].frequency_response(om2),
            "kry2": rom_store[("kry2rom", 8)].frequency_response(om2),
            "soar": rom_store[("soarrom", 8)].frequency_response(om2)}
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for k_, c_ in (("bt", "C1"), ("kry1", "C2"), ("soar", "C3"), ("kry2", "C4")):
        ax.semilogy(om2 / w1, np.abs(cand[k_] - Hf2) / np.abs(Hf2), color=c_, lw=1.1, label=labels[k_])
    ax.axvline(8.0, color="gray", ls=":"); ax.set_xlabel("$\\omega/\\omega_1$"); ax.set_ylabel("relative error $|H_r-H|/|H|$"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch09_locality.pdf")); plt.close(fig)
    far = om2 > 12 * w1
    for k_, nm in (("bt", "Bt"), ("kry1", "KryOne"), ("kry2", "KryTwo"), ("soar", "Soar")):
        num(f"cbtFar{nm}", sci(np.mean(np.abs(cand[k_] - Hf2)[far] / np.abs(Hf2)[far])))
        num(f"cbtNear{nm}", sci(np.mean(np.abs(cand[k_] - Hf2)[om2 < 8 * w1] / np.abs(Hf2)[om2 < 8 * w1])))

    # ---------------- stability and passivity
    def pm(H):
        Hv = velocity_transfer_function(H, om2)
        m = passivity_margin(Hv)
        return (m / np.abs(Hv)).min()           # cosine of the phase of the mobility: >= 0 for a passive port
    rows = []
    num("cbtPassFull", sci(pm(Hf2)))
    for r in (8, 12, 16):
        for key, name in (("modal", "modal truncation"), ("bt", "balanced truncation"), ("kry1", "Krylov, one-sided"), ("kry2", "Krylov, two-sided"), ("soar", "SOAR")):
            if rom_store.get((key, r)) is None:
                continue
            if key == "modal":
                V = Vf[:, :r // 2]; H = h_second_order(V.T @ M @ V, V.T @ C @ V, V.T @ K @ V, V.T @ B, V.T @ B, om2)
            elif key == "bt":
                H = BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=r).frequency_response(om2)
            elif key == "kry1":
                H = rom_store[("kry1rom", r)].frequency_response(om2)
            elif key == "kry2":
                H = rom_store[("kry2rom", r)].frequency_response(om2)
            else:
                H = rom_store[("soarrom", r)].frequency_response(om2)
            if key == "bt":
                stab = BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=r).is_stable()
            elif key == "kry1":
                stab = rom_store[("kry1rom", r)].is_stable()
            elif key == "kry2":
                stab = rom_store[("kry2rom", r)].is_stable()
            else:
                stab = None
            rows.append((r, name, stab, pm(H)))
    for rr in rows:
        print(rr, flush=True)
    with open(os.path.join(TAB, "ch09_stability.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rlcr}\n\\toprule\nstates & method & poles in left half-plane & min $\\cos\\arg H_v$ \\\\\n\\midrule\n")
        for r, name, stab, p in rows:
            s = "--" if stab is None else ("yes" if stab else "\\textbf{no}")
            fh.write(f"{r} & {name} & {s} & {sci(p)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    neg = [(r, name) for r, name, s, p in rows if p < -1e-6]
    num("cbtNegCount", str(len(neg))); num("cbtRowCount", str(len(rows)))
    unst = [(r, name) for r, name, s, p in rows if s is False]
    num("cbtUnstCount", str(len(unst)))
    print("negative passivity margin:", neg, "unstable:", unst, flush=True)
    num("cbtMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch09.tex"), "w") as fh:
        fh.write("% generated by scripts/ch09_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

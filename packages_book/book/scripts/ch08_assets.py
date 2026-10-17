# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch08_assets.py -- figures, tables and number macros of Chapter 8 (POD, Galerkin and affine sweeps).

Model: the two-region cantilever beam of rom_engine/examples/two_region_beam_rom.py (K(EI1, EI2) = EI1 K1 + EI2 K2).
Writes into book/:
    figures/ch08_pod.pdf          singular values; projection and Galerkin error against the number of modes
    figures/ch08_snapshots.pdf    Galerkin error against the number of snapshots, optimal Latin hypercube against random design
    figures/ch08_sweep.pdf        online cost: full re-assembly, full solve with affine assembly, reduced solve
    tables/ch08_modal.tex         lowest frequencies of the reduced model against the full model
    tables/ch08_cost.tex          offline and online cost split and the break-even number of queries
    tables/numbers_ch08.tex       every number quoted in the prose

Run from the repository root:   python packages_book/book/scripts/ch08_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import time
import platform
import warnings
import numpy as np
import scipy.linalg as sl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(BOOK, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "rom_engine", "examples"))
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


from two_region_beam_rom import build_two_region_beam                                  # noqa: E402
from rom_engine import PodBasis, GalerkinROM, AffineDecomposition                       # noqa: E402
from rom_engine.sampling import optimal_lhs                                             # noqa: E402

LO, HI = 0.5, 5.0
TIP_LOAD = -1000.0
XI_LO, XI_HI = 0.1, 0.9


class Beam:
    def __init__(self, n):
        self.n = n
        self.model = build_two_region_beam(n=n)
        m = self.model
        self.free = m["free_dofs"]
        f = self.free
        d = lambda A: A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        self.K1, self.K2 = d(m["K1"])[np.ix_(f, f)], d(m["K2"])[np.ix_(f, f)]
        self.M = d(m["M"])[np.ix_(f, f)]
        self.nf = len(f)
        self.tip = self.nf - 2
        self.keep = np.arange(self.nf)
        self.F = np.zeros(self.nf); self.F[self.tip] = TIP_LOAD

    def load_at(self, xi):
        """Unit point load at the transverse dof of the node nearest to the fraction xi of the span."""
        k = int(min(max(round(xi * self.n), 1), self.n))
        F = np.zeros(self.nf); F[2 * (k - 1)] = -1.0
        return F

    def solve(self, mu):
        F = self.load_at(mu[2]) if len(mu) > 2 else self.F
        return np.linalg.solve(mu[0] * self.K1 + mu[1] * self.K2, F)

    def snapshots(self, mus):
        return np.column_stack([self.solve(m) for m in mus])


def design(n, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "olhs":
        u = optimal_lhs(n, 3, rng=rng)
    else:
        u = rng.uniform(size=(n, 3))
    mu = LO + (HI - LO) * u
    mu[:, 2] = XI_LO + (XI_HI - XI_LO) * u[:, 2]
    return mu


def tip_err(b, V, test_mus):
    """Worst relative state error ||x - V x_r|| / ||x|| of the Galerkin solve on the basis V over the test parameters."""
    worst = 0.0
    K1r, K2r = V.T @ b.K1 @ V, V.T @ b.K2 @ V
    for mu in test_mus:
        ref = b.solve(mu)
        Fr = V.T @ b.load_at(mu[2])
        x = V @ np.linalg.solve(mu[0] * K1r + mu[1] * K2r, Fr)
        worst = max(worst, np.linalg.norm(x - ref) / np.linalg.norm(ref))
    return worst


def proj_err(b, V, test_mus, Wn=None):
    """Worst relative projection error of the held-out solutions (Euclidean norm)."""
    worst = 0.0
    for mu in test_mus:
        x = b.solve(mu)
        worst = max(worst, np.linalg.norm(x - V @ (V.T @ x)) / np.linalg.norm(x))
    return worst


# ------------------------------------------------------------------------------------------ POD and Galerkin vs modes
def exp_pod(b):
    mus = design(60, "olhs", 0)
    S = b.snapshots(mus)
    test = design(40, "rand", 7)
    pod = PodBasis().fit(S, n_modes=24)
    sv = pod.singular_values
    ranks = np.arange(1, 25)
    pe = [proj_err(b, pod.V[:, :r], test) for r in ranks]
    ge = [tip_err(b, pod.V[:, :r], test) for r in ranks]
    # mass-weighted POD on the same snapshots (columns scaled by the mass inner product)
    podM = PodBasis().fit(S, n_modes=24, M=b.M)
    geM = [tip_err(b, podM.V[:, :r], test) for r in ranks]
    print("singular values", sv[:8], flush=True)
    print("projection err", pe[:8], "galerkin tip err", ge[:8], "mass-weighted", geM[:8], flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax[0].semilogy(np.arange(1, len(sv) + 1), sv / sv[0], "o-"); ax[0].set_xlabel("mode"); ax[0].set_ylabel("$\\sigma_k/\\sigma_1$")
    ax[1].semilogy(ranks, pe, "o-", label="projection error (state)")
    ax[1].semilogy(ranks, ge, "s-", label="Galerkin state error, Euclidean POD")
    ax[1].semilogy(ranks, geM, "^--", label="Galerkin state error, mass-weighted POD")
    ax[1].set_xlabel("modes retained"); ax[1].set_ylabel("worst relative error, 40 unseen parameters"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch08_pod.pdf")); plt.close(fig)
    num("cpdNSnap", "60"); num("cpdNTest", "40"); num("cpdNFree", str(b.nf))
    num("cpdSvDecade", f"{sv[0] / sv[3]:.0f}")
    for tag, thr in (("FivePct", 5e-2), ("OnePct", 1e-2)):
        r_hit = next((int(r) for r, e in zip(ranks, ge) if e < thr), None)
        num("cpdModesFor" + tag, str(r_hit) if r_hit else "more than 24")
    num("cpdGalerkinErrTwelve", sci(ge[11])); num("cpdGalerkinErrLast", sci(ge[-1])); num("cpdProjErrLast", sci(pe[-1]))
    num("cpdSvLast", sci(sv[-1] / sv[0])); num("cpdSvTwelve", sci(sv[11] / sv[0]))
    num("cpdGalerkinOverProj", f"{max(g / p for g, p in zip(ge, pe)):.2f}")
    num("cpdMassWins", str(sum(1 for g, m in zip(ge, geM) if m < g))); num("cpdMassCompared", str(len(ge)))
    num("cpdMassErrTwelve", sci(geM[11]))
    num("cpdGalerkinErrFour", sci(ge[3])); num("cpdGalerkinErrSix", sci(ge[5]))
    num("cpdProjErrSix", sci(pe[5])); num("cpdGalerkinErrEight", sci(ge[7]))
    num("cpdMassErrSix", sci(geM[5])); num("cpdMassOrth", sci(podM.orthonormality_error()))
    num("cpdEucOrth", sci(pod.orthonormality_error()))
    return pod


# ------------------------------------------------------------------------------------------ snapshots vs design
def exp_snapshots(b):
    test = design(40, "rand", 7)
    ns = [6, 10, 15, 20, 30, 45, 60]
    res = {"olhs": [], "rand": []}
    for kind in res:
        for n in ns:
            errs = []
            for seed in range(5):
                S = b.snapshots(design(n, kind, 100 + seed))
                V = PodBasis().fit(S, n_modes=min(n, 12)).V
                errs.append(tip_err(b, V, test))
            res[kind].append((np.median(errs), np.max(errs)))
        print(kind, [f"{m:.1e}" for m, _ in res[kind]], flush=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for kind, st in (("olhs", "o-"), ("rand", "s--")):
        ax.semilogy(ns, [m for m, _ in res[kind]], st, label=("optimal LHS" if kind == "olhs" else "random") + ", median of 5")
    ax.set_xlabel("snapshots (basis size $\\min(n,12)$)"); ax.set_ylabel("Galerkin state error, 40 unseen parameters"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch08_snapshots.pdf")); plt.close(fig)
    num("cpdSnapOlhsB", sci(res["olhs"][3][0])); num("cpdSnapRandB", sci(res["rand"][3][0]))
    num("cpdSnapOlhsA", sci(res["olhs"][1][0])); num("cpdSnapRandA", sci(res["rand"][1][0]))
    better = sum(1 for (a, _), (c, _) in zip(res["olhs"], res["rand"]) if a < c)
    num("cpdOlhsBetterCount", str(better)); num("cpdSnapCases", str(len(ns)))


# ------------------------------------------------------------------------------------------ modal
def exp_modal(b, pod):
    mu = (2.0, 1.0)
    K = mu[0] * b.K1 + mu[1] * b.K2
    w_full = np.sqrt(sl.eigh(K, b.M, eigvals_only=True)[:5]) / (2 * np.pi)
    rows = []
    for r in (4, 6, 8, 12):
        rom = GalerkinROM(pod.V[:, :r]).reduce_system(K, b.M)
        f, _, _ = rom.solve_modal(n_modes=5)
        e = np.abs(f / w_full[:len(f)] - 1)
        rows.append((r, e))
        print("modal r", r, e, flush=True)
    with open(os.path.join(TAB, "ch08_modal.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrr}\n\\toprule\nmodes $r$ & mode 1 & mode 2 & mode 3 & mode 5 \\\\\n\\midrule\n")
        for r, e in rows:
            fh.write(f"{r} & {sci(e[0]) if e[0] > 0 else '0'} & {sci(e[1])} & {sci(e[2])} & {sci(e[4]) if len(e) > 4 else '--'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cpdModalFOne", f"{w_full[0]:.2f}"); num("cpdModalErrOneAtSix", sci(rows[1][1][0])); num("cpdModalErrFiveAtTwelve", sci(rows[3][1][4]))
    num("cpdModalErrThreeAtSix", sci(rows[1][1][2]))


# ------------------------------------------------------------------------------------------ cost
def exp_cost():
    rows = []
    figdata = {}
    for n in (120, 600):
        b = Beam(n)
        mus = design(16, "olhs", 0)[:, :2]
        t0 = time.perf_counter(); S = b.snapshots(mus); t_snap = time.perf_counter() - t0
        t0 = time.perf_counter(); pod = PodBasis().fit(S, n_modes=8); t_pod = time.perf_counter() - t0
        t0 = time.perf_counter()
        aff = AffineDecomposition([b.K1, b.K2], lambda mu: [mu[0], mu[1]]).project(pod.V)
        Fr = pod.V.T @ b.F
        t_proj = time.perf_counter() - t0
        test = design(200, "rand", 3)[:, :2]
        t0 = time.perf_counter()
        for mu in test:
            np.linalg.solve(aff.assemble_reduced(mu), Fr)
        t_on = (time.perf_counter() - t0) / len(test)
        t0 = time.perf_counter()
        for mu in test:
            np.linalg.solve(mu[0] * b.K1 + mu[1] * b.K2, b.F)
        t_full = (time.perf_counter() - t0) / len(test)
        m = b.model
        t0 = time.perf_counter()
        for mu in test[:20]:
            np.linalg.solve(m["K_direct"](mu[0], mu[1])[np.ix_(b.free, b.free)][np.ix_(b.keep, b.keep)], b.F)
        t_reasm = (time.perf_counter() - t0) / 20
        err = np.nan
        offline = t_snap + t_pod + t_proj
        be = offline / (t_full - t_on) if t_full > t_on else np.inf
        be2 = offline / (t_reasm - t_on)
        rows.append((n, b.nf, offline, t_on, t_full, t_reasm, be, be2, err))
        figdata[n] = (t_on, t_full, t_reasm)
        print(f"cost n={n} ({b.nf} dofs): offline {offline*1e3:.1f} ms; online {t_on*1e6:.1f} us; full solve {t_full*1e6:.1f} us; full with re-assembly {t_reasm*1e3:.2f} ms; "
              f"break-even {be:.0f} (vs affine full) / {be2:.0f} (vs re-assembly)", flush=True)
    with open(os.path.join(TAB, "ch08_cost.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n & & offline & \\multicolumn{3}{c}{one query ($\\mu$s)} & \\\\\n\\cmidrule(lr){4-6}\nelements & dofs & (ms) & reduced & full, affine & full, re-assembled & break-even queries \\\\\n\\midrule\n")
        for n, nf, off, ton, tf, tr, be, be2, err in rows:
            fh.write(f"{n} & {nf} & {off * 1e3:.0f} & {ton * 1e6:.0f} & {tf * 1e6:.0f} & {tr * 1e6:.0f} & {be2:.0f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    n, nf, off, ton, tf, tr, be, be2, err = rows[0]
    num("cpdCostSpeedAffine", f"{tf / ton:.1f}"); num("cpdCostSpeedReasm", f"{tr / ton:.0f}")
    n2, nf2, off2, ton2, tf2, tr2, bb, bb2, err2 = rows[1]
    num("cpdCostSpeedAffineBig", f"{tf2 / ton2:.0f}"); num("cpdCostSpeedReasmBig", f"{tr2 / ton2:.0f}")
    num("cpdBreakEven", f"{be2:.0f}"); num("cpdBreakEvenBig", f"{bb2:.0f}"); num("cpdBreakEvenAffBig", f"{bb:.0f}")
    num("cpdBigNf", str(nf2)); pass
    num("cpdOnlineUs", f"{ton * 1e6:.0f}")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = np.arange(2); w = 0.25
    for k, (lab, idx) in enumerate((("reduced", 0), ("full, affine assembly", 1), ("full, re-assembled", 2))):
        ax.bar(xs + (k - 1) * w, [figdata[120][idx] * 1e6, figdata[600][idx] * 1e6], w, label=lab)
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{rows[0][1]} dofs", f"{rows[1][1]} dofs"]); ax.set_ylabel("time per query ($\\mu$s)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch08_sweep.pdf")); plt.close(fig)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    b = Beam(120)
    pod = exp_pod(b); exp_snapshots(b); exp_modal(b, pod); exp_cost()
    num("cpdMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch08.tex"), "w") as fh:
        fh.write("% generated by scripts/ch08_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

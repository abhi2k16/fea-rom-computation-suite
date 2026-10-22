# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch14_pipeline.py -- the integrated parametric pipeline of Chapter 14: mesh -> FESystem -> snapshots -> POD -> affine
Galerkin ROM -> held-out parameters and held-out loads -> report by the convention of Appendix B.

Model: a cantilever plate (0.4 m x 0.2 m, t = 0.02 m, plane stress, nu = 0.3) of two material regions (left and right half)
with Young's moduli E1, E2 (the two parameters, in units of 1e11 Pa, range [0.5, 2.5]) under a traction on the free edge.
Training: pairs (parameter, load) from a Latin hypercube and random combinations of eight polynomial traction shapes; Euclidean POD.
Basis size: chosen on a VALIDATION set of the training kind (never on the test sets).
Test set A: new parameters, loads from the span of the training shapes. Test set B: new parameters and a load shape outside
that span (a sin(pi y) traction, vertical).
Reported per Appendix B: error of the full state next to errors of four local quantities (tip deflection, peak von Mises stress,
total vertical reaction, root moment; the last two relative to the load scale, because a resultant can cancel); offline / online / lift / stress-recovery cost separately; break-even number of
queries including the offline cost, against a sparse-affine and a re-assembled full-order baseline.

Writes figures/ch14_pipeline.pdf, tables/ch14_pipeline.tex, tables/numbers_ch14_pl.tex.
Run from the repository root:
    PYTHONPATH=fea_engine/src:rom_engine/src python packages_book/book/scripts/ch14_pipeline.py [--large 64 32] [--repeats 5]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse
import os
import platform
import time
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
BLUE, VIOLET, GREEN, ORANGE, GREY = "#2b4c9b", "#6a3d9a", "#2e8b57", "#d2691e", "#555555"
NUM = {}
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem                  # noqa: E402
from fea_engine.mesh import rectangle_mesh, MultiBlockMesh                                   # noqa: E402
from rom_engine import PodBasis, AffineDecomposition                                         # noqa: E402

E0, NU, RHO, T, LX, LY = 1.0e11, 0.3, 7850.0, 0.02, 0.4, 0.2
PRANGE = (0.5, 2.5)


def num(k, v):
    NUM[k] = v


def sci(x):
    from decimal import Decimal, ROUND_HALF_UP
    if x == 0:
        return "0"
    d = Decimal(repr(float(x))).normalize(); e = d.adjusted(); m = d.scaleb(-e).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if m >= 10:
        m, e = Decimal("1.0"), e + 1
    return f"${m}\\times10^{{{e}}}$"


def g3(x):
    return f"{x:,.0f}".replace(",", "\\,") if abs(x) >= 1000 else f"{x:.3g}"


def best(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


class Model:
    """mesh, two-region FESystem, affine components K1, K2 (free dofs), mass, traction shapes"""
    def __init__(self, nx, ny):
        m0 = rectangle_mesh(Lx=LX, Ly=LY, nx=nx, ny=ny)
        cx = m0.nodes[m0.elements].mean(axis=1)[:, 0]
        self.mesh = MultiBlockMesh(nodes=m0.nodes, blocks={"left": m0.elements[cx < LX / 2], "right": m0.elements[cx >= LX / 2]}, dim=2)
        el = Quad4PlaneStress(); self.fe = FESystem(self.mesh, {"left": el, "right": el}, sparse=True)
        self.D = lambda e: D_plane_stress(Material(E=e * E0, nu=NU, rho=RHO))
        self.Dq = lambda mu: {"left": self.D(mu[0]), "right": self.D(mu[1])}
        self.mesh.select_nodes(x=0.0, name="root"); self.mesh.select_nodes(x=LX, name="tip")
        self.fe.fix_dofs("root", ["ux", "uy"])
        self.free = np.asarray(self.fe.free_dofs); self.n = self.fe.n_dof
        fixed = np.setdiff1d(np.arange(self.n), self.free); self.fixed = fixed
        self.fe.assemble_stiffness({"left": self.D(1.0), "right": self.D(0.0)}, thickness=T); K1 = sp.csr_matrix(self.fe.K)
        self.fe.assemble_stiffness({"left": self.D(0.0), "right": self.D(1.0)}, thickness=T); K2 = sp.csr_matrix(self.fe.K)
        self.K1, self.K2 = K1, K2
        self.K1f, self.K2f = K1[self.free][:, self.free].tocsc(), K2[self.free][:, self.free].tocsc()
        self.fe.assemble_mass(RHO * np.eye(2), thickness=T); self.Mf = sp.csr_matrix(self.fe.M)[self.free][:, self.free]
        tn = np.asarray(self.mesh.node_sets["tip"]); self.tn = tn
        self.y = self.mesh.nodes[tn, 1] / LY
        self.root = np.asarray(self.mesh.node_sets["root"]); self.yroot = self.mesh.nodes[self.root, 1]
        shapes = np.zeros((self.n, 8))
        for j, (d, k) in enumerate([(d, k) for d in (0, 1) for k in range(4)]):
            shapes[2 * tn + d, j] = self.y ** k
        self.shapes = shapes[self.free]
        self.tipdofs = np.searchsorted(self.free, 2 * tn + 1)
        self.fine_tip = np.asarray(2 * tn + 1)

    def Kff(self, mu):
        return (mu[0] * self.K1f + mu[1] * self.K2f).tocsc()

    def load_sine(self, a):
        F = np.zeros(self.n); F[2 * self.tn + 1] = a * np.sin(np.pi * self.y); return F[self.free]

    def solve(self, mu, F):
        return spla.spsolve(self.Kff(mu), F)

    def full(self, uf):
        u = np.zeros(self.n); u[self.free] = uf; return u

    def locals_(self, mu, uf, F):
        """tip deflection, peak von Mises, total vertical reaction, root moment (z) for a free-dof solution"""
        u = self.full(uf)
        tip = float(np.mean(u[self.fine_tip]))
        vm = float(np.max(self.fe.von_mises(u, D=self.Dq(mu))))
        Rall = (mu[0] * self.K1 + mu[1] * self.K2) @ u
        # fixed dofs are the root ux, uy: reactions are K u - F with F = 0 there
        ry = float(np.sum(Rall[2 * self.root + 1])); mz = float(np.sum(self.yroot * Rall[2 * self.root + 0]))
        return np.array([tip, vm, ry, mz])


def lhs(n, rng):
    from scipy.stats import qmc
    s = qmc.LatinHypercube(d=2, seed=int(rng.integers(1 << 30))).random(n)
    return PRANGE[0] + (PRANGE[1] - PRANGE[0]) * s


def queries(mod, kind, n, rng):
    mus = rng.uniform(PRANGE[0], PRANGE[1], size=(n, 2))
    if kind == "span":
        Fs = [mod.shapes @ rng.uniform(-1, 1, 8) * 1e4 for _ in range(n)]
    else:
        Fs = [mod.load_sine(rng.choice([-1, 1]) * rng.uniform(0.5, 2.0) * 1e4) for _ in range(n)]
    return mus, Fs


def run(nx, ny, repeats, seed=7, ntrain=24):
    t_all = time.perf_counter(); mod = Model(nx, ny); rng = np.random.default_rng(seed)
    out = {"nx": nx, "ny": ny, "n": mod.n, "nfree": len(mod.free), "ne": nx * ny}
    # ---------------- offline
    t0 = time.perf_counter()
    mus = lhs(ntrain, rng); loads = [mod.shapes @ rng.uniform(-1, 1, 8) * 1e4 for _ in range(ntrain)]
    t1 = time.perf_counter(); S = np.column_stack([mod.solve(m, f) for m, f in zip(mus, loads)]); t_snap = time.perf_counter() - t1
    t1 = time.perf_counter(); basis = PodBasis().fit(S, n_modes=ntrain); t_pod = time.perf_counter() - t1
    sv = np.asarray(basis.singular_values) if hasattr(basis, "singular_values") else None
    # affine projection for the largest basis; smaller bases use leading columns (nested)
    Vmax = basis.V; t1 = time.perf_counter()
    aff = AffineDecomposition([mod.K1f.toarray(), mod.K2f.toarray()], lambda mu: [mu[0], mu[1]]).project(Vmax); t_proj = time.perf_counter() - t1
    t_off_fixed = t_pod + t_proj
    # ---------------- query helpers (use the leading r columns)
    def rom_solve(mu, F, r):
        Kred = (mu[0] * aff.reduced_components[0] + mu[1] * aff.reduced_components[1])[:r, :r]
        q = np.linalg.solve(Kred, Vmax[:, :r].T @ F)
        return q

    cache = {}

    def scale_of(lf, F):                         # tip and peak stress: own value; reaction and moment: the load scale (a resultant can cancel)
        Fy = np.sum(np.abs(F[mod.free % 2 == 1])); Fa = np.sum(np.abs(F))
        return np.array([abs(lf[0]), abs(lf[1]), Fy, LX * Fa])

    def fom_of(key, mus_, Fs_):                                  # full-order solution and local quantities, computed once per set
        if key not in cache:
            ufs = [mod.solve(mu, F) for mu, F in zip(mus_, Fs_)]
            cache[key] = (ufs, [mod.locals_(mu, uf, F) for mu, uf, F in zip(mus_, ufs, Fs_)])
        return cache[key]

    def err_set(key, mus_, Fs_, r):
        ufs, lfs = fom_of(key, mus_, Fs_); eg, el, eta = [], [], []
        for mu, F, uf, lf in zip(mus_, Fs_, ufs, lfs):
            q = rom_solve(mu, F, r); ur = Vmax[:, :r] @ q; lr = mod.locals_(mu, ur, F)
            eg.append(np.linalg.norm(ur - uf) / np.linalg.norm(uf)); el.append(np.abs(lr - lf) / scale_of(lf, F))
            eta.append(np.linalg.norm(F - mod.Kff(mu) @ ur) / np.linalg.norm(F))
        return np.array(eg), np.array(el), np.array(eta)

    # ---------------- choose r on a validation set of the training kind
    mv, Fv = queries(mod, "span", 20, rng); ma, Fa = queries(mod, "span", 40, rng); mb, Fb = queries(mod, "sine", 40, rng)
    grid = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]; tol = 1e-3
    curve = {"val": [], "A": [], "B": []}
    for r in grid:
        for nm, (m_, F_) in (("val", (mv, Fv)), ("A", (ma, Fa)), ("B", (mb, Fb))):
            eg, el, eta = err_set(nm, m_, F_, r); curve[nm].append((eg.max(), el.max(axis=0)))
    sel = next((r for r, c in zip(grid, curve["val"]) if max(c[0], c[1].max()) <= tol), grid[-1])
    out["r"] = sel; out["tol"] = tol; out["curve"] = curve; out["grid"] = grid; out["sel_met"] = max(curve["val"][grid.index(sel)][0], curve["val"][grid.index(sel)][1].max()) <= tol
    # ---------------- held-out errors at the chosen r
    res = {}
    for nm, (m_, F_) in (("A", (ma, Fa)), ("B", (mb, Fb))):
        eg, el, eta = err_set(nm, m_, F_, sel); res[nm] = (eg, el, eta)
    out["res"] = res
    # ---------------- costs (minimum over repeats), per query, on a few representative queries
    mu0, F0 = ma[0], Fa[0]; qs = [(ma[i], Fa[i]) for i in range(5)]
    q0 = rom_solve(mu0, F0, sel); u0 = Vmax[:, :sel] @ q0
    def t_fom_sparse():
        for mu, F in qs: mod.solve(mu, F)
    def t_fom_reasm():
        for mu, F in qs[:2]:
            mod.fe.assemble_stiffness(mod.Dq(mu), thickness=T); K = sp.csr_matrix(mod.fe.K)[mod.free][:, mod.free].tocsc(); spla.spsolve(K, F)
    def t_rom():
        for mu, F in qs: rom_solve(mu, F, sel)
    def t_lift():
        for _ in qs: Vmax[:, :sel] @ q0
    def t_stress():
        for mu, F in qs: mod.locals_(mu, u0, F)
    def t_resid():
        for mu, F in qs: np.linalg.norm(F - mod.Kff(mu) @ u0)
    c = {k: best(f, repeats) / (2 if k == "fom_reasm" else len(qs)) for k, f in (("fom_sparse", t_fom_sparse), ("fom_reasm", t_fom_reasm), ("rom", t_rom), ("lift", t_lift), ("recover", t_stress), ("resid", t_resid))}
    out["cost"] = c; out["offline"] = {"snap": t_snap, "pod": t_pod, "proj": t_proj, "total": t_snap + t_pod + t_proj, "ntrain": ntrain}
    out["sv"] = sv; out["total_run_s"] = time.perf_counter() - t_all
    return out


def readme_example(n):
    """run rom_engine/examples/two_region_beam_rom.py n times (the example of the README) and parse its 500-point sweep"""
    import re
    import subprocess
    import sys
    ex = os.path.normpath(os.path.join(BOOK, "..", "..", "rom_engine", "examples", "two_region_beam_rom.py"))
    rows = []
    for _ in range(n):
        o = subprocess.run([sys.executable, ex], capture_output=True, text=True, env=dict(os.environ)).stdout
        f = re.search(r"full-order .*?:\s+([\d.]+) ms", o); r = re.search(r"reduced-order .*?:\s+([\d.]+) ms", o); x = re.search(r"speedup:\s+([\d.]+)x", o)
        rows.append((float(f.group(1)), float(r.group(1)), float(x.group(1))))
    return np.array(rows)


def be(off, t_fom, t_rom):
    return off / (t_fom - t_rom) if t_fom > t_rom else float("inf")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--small", nargs=2, type=int, default=[24, 12]); ap.add_argument("--large", nargs=2, type=int, default=[64, 32])
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--readme-runs", type=int, default=5); a = ap.parse_args()
    print(platform.platform(), "numpy", np.__version__, flush=True)
    S = run(*a.small, a.repeats); print("small done", S["n"], "r", S["r"], round(S["total_run_s"], 1), "s", flush=True)
    L = run(*a.large, max(2, a.repeats - 2)); print("large done", L["n"], "r", L["r"], round(L["total_run_s"], 1), "s", flush=True)
    # ---------------- numbers
    for tag, R in (("S", S), ("L", L)):
        num(f"cpl{tag}Dofs", f"{R['n']:,}".replace(",", "\\,")); num(f"cpl{tag}Free", f"{R['nfree']:,}".replace(",", "\\,")); num(f"cpl{tag}Elems", f"{R['ne']:,}".replace(",", "\\,")); num(f"cpl{tag}R", str(R["r"]))
        for nm in "AB":
            eg, el, eta = R["res"][nm]
            num(f"cpl{tag}{nm}Glob", sci(eg.max())); num(f"cpl{tag}{nm}GlobMean", sci(eg.mean()))
            for j, lab in enumerate(("Tip", "Vm", "Ry", "Mz")):
                num(f"cpl{tag}{nm}{lab}", sci(el[:, j].max()))
        c, o = R["cost"], R["offline"]
        for k, v in c.items():
            num(f"cpl{tag}T{k.capitalize().replace('_', '')}", g3(v * 1e3))
        num(f"cpl{tag}Off", g3(o["total"] * 1e3)); num(f"cpl{tag}OffSnap", g3(o["snap"] * 1e3)); num(f"cpl{tag}OffPod", g3(o["pod"] * 1e3)); num(f"cpl{tag}OffProj", g3(o["proj"] * 1e3))
        t_rom_total = c["rom"] + c["lift"]
        num(f"cpl{tag}BeSparse", f"{be(o['total'], c['fom_sparse'], t_rom_total):.3g}"); num(f"cpl{tag}BeReasm", f"{be(o['total'], c['fom_reasm'], t_rom_total):.3g}")
        num(f"cpl{tag}SpeedSparse", g3(c['fom_sparse'] / c['rom'])); num(f"cpl{tag}SpeedReasm", g3(c['fom_reasm'] / c['rom']))
        num(f"cpl{tag}SpeedFlowSparse", f"{(c['fom_sparse'] + c['recover']) / (c['rom'] + c['lift'] + c['recover']):.3g}")
        num(f"cpl{tag}SpeedFlowReasm", f"{(c['fom_reasm'] + c['recover']) / (c['rom'] + c['lift'] + c['recover']):.3g}")
        num(f"cpl{tag}Ntrain", str(o["ntrain"])); num(f"cpl{tag}SelMet", "met" if R["sel_met"] else "not met")
    eg, el, eta = S["res"]["A"]; eg2, el2, eta2 = S["res"]["B"]
    from scipy.stats import spearmanr
    rho = spearmanr(np.concatenate([eta, eta2]), np.concatenate([eg, eg2])).correlation; num("cplSpearman", f"{rho:.2f}")
    num("cplSpearmanA", f"{spearmanr(eta, eg).correlation:.2f}"); num("cplSpearmanB", f"{spearmanr(eta2, eg2).correlation:.2f}")
    num("cplBFloor", sci(min(c[0] for c in S["curve"]["B"][-4:]))); num("cplAFloor", sci(S["curve"]["A"][-1][0]))
    num("cplResidB", f"{eta2.min():.2g}--{eta2.max():.2g}"); num("cplResidA", f"{eta.min():.2g}--{eta.max():.2g}")
    if a.readme_runs:
        X = readme_example(a.readme_runs); sp_ = X[:, 2]
        num("cplRdRuns", str(len(X))); num("cplRdMin", f"{sp_.min():.0f}"); num("cplRdMed", f"{np.median(sp_):.0f}"); num("cplRdMax", f"{sp_.max():.0f}")
        num("cplRdFomMs", f"{np.median(X[:, 0]):.0f}"); num("cplRdRomMs", f"{np.median(X[:, 1]):.1f}")
        print("README example speed-ups:", sp_)
    num("cplMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    # ---------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.3))
    g = S["grid"]
    for nm, ls, mk in (("val", ":", "d"), ("A", "-", "o"), ("B", "--", "s")):
        ax[0].semilogy(g, [c[0] for c in S["curve"][nm]], ls + mk, color=BLUE, ms=3.5, label=f"{ {'val': 'validation', 'A': 'test A', 'B': 'test B'}[nm]}, full state")
        ax[0].semilogy(g, [c[1][1] for c in S["curve"][nm]], ls + mk, color=ORANGE, ms=3.5, label=f"{ {'val': 'validation', 'A': 'test A', 'B': 'test B'}[nm]}, peak stress" if nm == "B" else None)
    ax[0].axhline(S["tol"], color=GREY, lw=0.8); ax[0].axvline(S["r"], color=GREEN, lw=0.8)
    ax[0].set(xlabel="basis size $r$", ylabel="largest relative error", title="(a) error against basis size"); ax[0].legend(fontsize=6)
    ax[1].loglog(eta, eg, "o", color=BLUE, ms=3.5, label="test A"); ax[1].loglog(eta2, eg2, "s", color=VIOLET, ms=3.5, label="test B")
    ax[1].set(xlabel="online residual indicator", ylabel="true full-state error", title="(b) cheap indicator against error"); ax[1].legend(fontsize=7)
    cs = S["cost"]; labs = ["FOM\nre-\nassembled", "FOM\nsparse\naffine", "ROM\nsolve", "ROM\nlift", "stress\nrecov-\nery"]
    vals = [cs["fom_reasm"], cs["fom_sparse"], cs["rom"], cs["lift"], cs["recover"]]
    ax[2].bar(range(5), [v * 1e3 for v in vals], color=[BLUE, BLUE, VIOLET, VIOLET, GREEN]); ax[2].set_yscale("log"); ax[2].set_xticks(range(5)); ax[2].set_xticklabels(labs, fontsize=7)
    ax[2].set(ylabel="time per query (ms)", title="(c) cost per query"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch14_pipeline.pdf")); plt.close(fig)
    # ---------------- cost table
    with open(os.path.join(TAB, "ch14_pipeline.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}lrr@{}}\n\\toprule\n & small mesh & large mesh \\\\\n\\midrule\n")
        fh.write(f"free dofs & {S['nfree']:,} & {L['nfree']:,} \\\\\nelements & {S['ne']:,} & {L['ne']:,} \\\\\nbasis size $r$ & {S['r']} & {L['r']} \\\\\n\\midrule\n".replace(",", "\\,"))
        for lab, k in (("offline: {n} snapshot solves", "snap"), ("offline: POD", "pod"), ("offline: projection", "proj"), ("offline: total", "total")):
            fh.write(f"{lab.format(n=S['offline']['ntrain'])} (ms) & {g3(S['offline'][k]*1e3)} & {g3(L['offline'][k]*1e3)} \\\\\n")
        fh.write("\\midrule\n")
        for lab, k in (("FOM re-assembled + sparse solve (ms/query)", "fom_reasm"), ("FOM sparse affine + solve (ms/query)", "fom_sparse"), ("ROM online solve (ms/query)", "rom"), ("ROM lift to full space (ms/query)", "lift"), ("stress recovery, either model (ms/query)", "recover"), ("residual indicator (ms/query)", "resid")):
            fh.write(f"{lab} & {g3(S['cost'][k]*1e3)} & {g3(L['cost'][k]*1e3)} \\\\\n")
        fh.write("\\midrule\n")
        for lab, kk in (("break-even queries, against re-assembled FOM", "BeReasm"), ("break-even queries, against sparse affine FOM", "BeSparse")):
            fh.write(f"{lab} & {NUM['cplS'+kk]} & {NUM['cplL'+kk]} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB, "numbers_ch14_pl.tex"), "w") as fh:
        fh.write("% generated by scripts/ch14_pipeline.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    for tag, R in (("small", S), ("large", L)):
        print(tag, "r", R["r"], "selection met:", R["sel_met"])
        for nm in "AB":
            eg, el, eta = R["res"][nm]; print("  set", nm, "global max", f"{eg.max():.2e}", "local max (tip, vm, Ry, Mz)", np.array2string(el.max(axis=0), precision=2))
        print("  cost ms:", {k: round(v * 1e3, 3) for k, v in R["cost"].items()}, "offline ms", {k: round(v * 1e3, 1) for k, v in R["offline"].items() if k != "ntrain"})
    print("spearman residual vs error", rho, "A", spearmanr(eta, eg).correlation, "B", spearmanr(eta2, eg2).correlation)


if __name__ == "__main__":
    main()

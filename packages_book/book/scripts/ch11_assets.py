# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch11_assets.py -- figures, tables and number macros of Chapter 11 (substructuring and component reduction).

Experiments
  A  a clamped-clamped beam split in two halves: Craig-Bampton coupled model against the full model, and Guyan
     condensation at equal reduced size (same beam as paper/experiments/e3_cms.py)
  B  one membrane substructure with a 21-dof interface: fixed-interface modes and interface reduction
     (characteristic constraint modes, interface modes from full-model modes)
  C  free-interface reduction (Rubin, MacNeal) against Craig-Bampton: frequencies and static flexibility
  D  component POD of two non-matching nonlinear membranes tied by a mortar operator
Writes into book/: figures/ch11_cb.pdf, ch11_interface.pdf, ch11_component.pdf; tables/ch11_free.tex, ch11_component.tex,
numbers_ch11.tex.

Run from the repository root:   PYTHONPATH=fea_engine/src:rom_engine/src python packages_book/book/scripts/ch11_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import warnings
import numpy as np
import scipy.sparse as sp
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


from fea_engine import elements                                                                  # noqa: E402
from fea_engine.geometry import generate_mesh                                                    # noqa: E402
from fea_engine.material import EI_beam, Material, Section                                        # noqa: E402
from fea_engine.solver import FESystem                                                           # noqa: E402
from fea_engine.mortar import MortarTie2D                                                        # noqa: E402
from rom_engine.cms import couple, craig_bampton, guyan, free_interface_cms, reduce_interface    # noqa: E402
from rom_engine.interface_reduction import (gram_schmidt_interface_modes, interface_reduced_craig_bampton,  # noqa: E402
                                            natural_frequency_errors, mac_diagonal)
from rom_engine.component_pod import (legendre_interface_parametrisation, component_snapshots, ComponentBasis,  # noqa: E402
                                      ReducedAssembly)

E, RHO, A_SEC, I_SEC, NEL = 210e9, 7800.0, 0.01, 8.33e-6, 20
MAT = Material(E=E, nu=0.3, rho=RHO)


def beam(L, n, fixed):
    s = FESystem(generate_mesh(dim=1, L=L, n=n), elements.Beam2DEulerBernoulli())
    s.assemble_stiffness(EI_beam(MAT, Section(A=A_SEC, I=I_SEC)))
    s.assemble_mass(RHO * A_SEC)
    s.fix_dofs(fixed, [0, 1])
    fr = np.asarray(s.free_dofs)
    return fr, np.array(s.K)[np.ix_(fr, fr)], np.array(s.M)[np.ix_(fr, fr)]


def freqs_hz(K, M, k):
    return np.sqrt(np.clip(eigh(K, M, eigvals_only=True)[:k], 0, None)) / (2 * np.pi)


# ------------------------------------------------------------------------------------------ A
def exp_a():
    fr, K, M = beam(1.0, 2 * NEL, [0, 2 * NEL])
    f_full = freqs_hz(K, M, 4)
    frL, KL, ML = beam(0.5, NEL, [0]); frR, KR, MR = beam(0.5, NEL, [NEL])
    pos = lambda f, d: int(np.where(f == d)[0][0])
    ibL = [pos(frL, 2 * NEL), pos(frL, 2 * NEL + 1)]; ibR = [pos(frR, 0), pos(frR, 1)]
    cb_rows = []
    for nm in (1, 2, 3, 5, 8, 10, 15, 20, None):
        cl, cr = craig_bampton(KL, ML, ibL, nm), craig_bampton(KR, MR, ibR, nm)
        cp = couple([cl, cr], [[0, 1], [0, 1]])
        f, _ = cp.solve_modal(4)
        cb_rows.append((nm, cp.K.shape[0], np.abs(f - f_full) / f_full))
    # Guyan on the full beam: masters = transverse dof of m evenly spaced nodes
    n_free = K.shape[0]
    vdofs = np.arange(0, n_free, 2)          # free dofs alternate [v, theta] for the nodes between the clamps
    gu_rows = []
    for m in (4, 6, 8, 12, 16, 20):
        idx = vdofs[np.round(np.linspace(0, len(vdofs) - 1, m + 2)[1:-1]).astype(int)]
        g = guyan(K, M, idx)
        f = freqs_hz(g.K, g.M, 4)
        gu_rows.append((m, m, np.abs(f - f_full) / f_full))
    for r in cb_rows:
        print("CB", r[0], r[1], f"{r[2].max():.1e}", flush=True)
    for r in gu_rows:
        print("Guyan", r[0], f"{r[2].max():.1e}", flush=True)
    num("ccmBeamFull", str(K.shape[0])); num("ccmBeamFOne", f"{f_full[0]:.2f}")
    d = {r[0]: r for r in cb_rows}
    num("ccmCbOneSize", str(d[1][1])); num("ccmCbOneErr", sci(d[1][2].max()))
    num("ccmCbFiveSize", str(d[5][1])); num("ccmCbFiveErr", sci(d[5][2].max()))
    num("ccmCbTwentySize", str(d[20][1])); num("ccmCbTwentyErr", sci(d[20][2].max()))
    num("ccmCbAllSize", str(d[None][1])); num("ccmCbAllErr", sci(d[None][2].max()))
    num("ccmCbMonotone", "yes" if all(a[2].max() >= b[2].max() * (1 - 1e-9) for a, b in zip(cb_rows[:-2], cb_rows[1:-1])) else "no")
    # Guyan at the size closest to CB(5) (12 dofs) and CB(10)
    gd = {r[0]: r for r in gu_rows}
    num("ccmGuyTwelveErr", sci(gd[12][2].max())); num("ccmGuyTwentyErr", sci(gd[20][2].max()))
    num("ccmGuyEightErr", sci(gd[8][2].max()))
    num("ccmGuyOneErrFirst", sci(gd[12][2][0])); num("ccmGuyTwelveHigh", sci(gd[12][2][3]))
    # CB size 12 = 2 + 2*5 -> compare
    # static exactness: Guyan with 2 masters on a statically loaded beam
    from numpy.linalg import solve
    F = np.zeros(n_free); F[vdofs[len(vdofs) // 2]] = 1.0
    u_full = solve(K, F)
    idx = np.array([vdofs[len(vdofs) // 2]])
    g = guyan(K, M, idx)
    u_g = g.expand(solve(g.K, F[idx]))
    num("ccmGuyStatic", sci(np.linalg.norm(u_g - u_full) / np.linalg.norm(u_full)))
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.semilogy([r[1] for r in cb_rows[:-1]], [r[2].max() for r in cb_rows[:-1]], "o-", label="Craig-Bampton, two halves")
    ax.semilogy([r[1] for r in gu_rows], [r[2].max() for r in gu_rows], "s--", label="Guyan, transverse masters")
    ax.set_xlabel("reduced model size"); ax.set_ylabel("max relative error, first 4 frequencies"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch11_cb.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ B
def membrane(nx, ny):
    n = nx * ny
    K = np.zeros((n, n))
    idx = lambda i, j: i * ny + j
    for i in range(nx):
        for j in range(ny):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < nx and j + dj < ny:
                    a, b = idx(i, j), idx(i + di, j + dj)
                    K[a, a] += 1; K[b, b] += 1; K[a, b] -= 1; K[b, a] -= 1
    for j in range(ny):
        K[idx(0, j), idx(0, j)] += 1.0
    return K, np.eye(n), [idx(nx - 1, j) for j in range(ny)]


NMODES = 15


def exp_b():
    K, M, iface = membrane(16, 21)
    n = K.shape[0]
    ref = np.sqrt(eigh(K, M, eigvals_only=True)[:6])
    num("ccmMemN", str(n)); num("ccmMemIface", str(len(iface)))
    rows_nm = []
    for nm in (5, 10, 20, 40, 80):
        cb = craig_bampton(K, M, iface, nm)
        e = np.abs(natural_frequency_errors(K, M, cb.T, NMODES))
        rows_nm.append((nm, cb.K.shape[0], e.max()))
        print("mem CB", nm, cb.K.shape[0], f"{e.max():.1e}", flush=True)
    d = {r[0]: r for r in rows_nm}
    num("ccmMemNmTenErr", sci(d[10][2])); num("ccmMemNmFortyErr", sci(d[40][2])); num("ccmMemNmTenSize", str(d[10][1]))
    num("ccmMemNmFortySize", str(d[40][1]))
    # interface reduction at nm = 40
    cb = craig_bampton(K, M, iface, 40)
    w, V = eigh(K, M)
    psi_all = gram_schmidt_interface_modes([V], iface)
    num("ccmPsiCols", str(psi_all.shape[1]))
    keeps = (2, 3, 4, 6, 8, 10, 12, 16, 21)
    cc, gs, sizes, mac_cc, mac_gs = [], [], [], [], []
    for k in keeps:
        r = reduce_interface(cb, k)
        cc.append(np.abs(natural_frequency_errors(K, M, r.T, NMODES)).max())
        g = interface_reduced_craig_bampton(K, M, iface, 40, psi_all[:, :k])
        gs.append(np.abs(natural_frequency_errors(K, M, g.T, NMODES)).max())
        sizes.append(k + 40)
        mac_cc.append(mac_diagonal(K, M, r.T, NMODES).min()); mac_gs.append(mac_diagonal(K, M, g.T, NMODES).min())
        print("iface", k, f"cc {cc[-1]:.1e} gs {gs[-1]:.1e} mac {mac_cc[-1]:.4f} {mac_gs[-1]:.4f}", flush=True)
    ik = {k: i for i, k in enumerate(keeps)}
    for k, nm in ((2, "Two"), (4, "Four"), (8, "Eight"), (12, "Twelve"), (21, "All")):
        num(f"ccmCc{nm}Err", sci(cc[ik[k]])); num(f"ccmGs{nm}Err", sci(gs[ik[k]]))
    num("ccmCcFourMac", f"{mac_cc[ik[4]]:.4f}"); num("ccmGsFourMac", f"{mac_gs[ik[4]]:.4f}")
    num("ccmCcEightMac", f"{mac_cc[ik[8]]:.4f}"); num("ccmGsEightMac", f"{mac_gs[ik[8]]:.4f}")
    num("ccmCcSizeFour", str(4 + 40)); num("ccmCcSavedDofs", str(21 - 4))
    plain = np.abs(natural_frequency_errors(K, M, cb.T, NMODES)).max()
    num("ccmPlainErr", sci(plain))
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax[0].semilogy([r[1] for r in rows_nm], [r[2] for r in rows_nm], "o-")
    ax[0].set_xlabel("reduced size (21 interface + $n_m$)"); ax[0].set_ylabel("max relative error, first 15 frequencies")
    ax[1].semilogy(keeps, cc, "o-", label="characteristic constraint modes")
    ax[1].semilogy(keeps, gs, "s--", label="interface modes from full-model modes")
    ax[1].axhline(plain, color="gray", ls=":", label="no interface reduction")
    ax[1].set_xlabel("interface coordinates kept (of 21), $n_m=40$"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch11_interface.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ C
def chain(n=40, k=1.0e4, m=1.0):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k
        if i + 1 < n:
            K[i + 1, i + 1] += k
            K[i, i + 1] = K[i + 1, i] = -k
    return K, m * np.eye(n)


def exp_c():
    K, M = chain()
    n = K.shape[0]
    wref = np.sqrt(eigh(K, M, eigvals_only=True)[:20])
    iface = [39, 20]
    F = np.zeros(n); F[39] = 1.0
    u_ref = np.linalg.solve(K, F)
    rows = []
    for nm in (3, 6, 10):
        cb = craig_bampton(K, M, iface, nm)
        wcb = np.sqrt(eigh(cb.K, cb.M, eigvals_only=True))
        u_cb = cb.T @ np.linalg.solve(cb.K, cb.T.T @ F)
        row = {"nm": nm, "cb_size": cb.K.shape[0], "cb_s": abs(u_cb[39] - u_ref[39]) / abs(u_ref[39])}
        # frequencies of the first nm+2 modes (the model has nm+2 states): max relative error, and the lowest nm separately
        k_all = cb.K.shape[0]
        row["cb_low"] = np.max(np.abs(wcb[:nm] - wref[:nm]) / wref[:nm]); row["cb_top"] = np.max(np.abs(wcb[nm:k_all] - wref[nm:k_all]) / wref[nm:k_all])
        for meth in ("rubin", "macneal"):
            m = free_interface_cms(K, M, iface, nm, meth)
            w = np.sqrt(eigh(m.K, m.M, eigvals_only=True))
            u = m.T @ np.linalg.solve(m.K, m.T.T @ F)
            row[meth + "_size"] = m.K.shape[0]
            row[meth + "_low"] = np.max(np.abs(w[:nm] - wref[:nm]) / wref[:nm])
            ks = m.K.shape[0]
            row[meth + "_top"] = np.max(np.abs(w[nm:ks] - wref[nm:ks]) / wref[nm:ks]) if ks > nm else np.nan
            row[meth + "_s"] = abs(u[39] - u_ref[39]) / abs(u_ref[39])
        w2, V = eigh(K, M)
        Vm = V[:, :nm]
        um = Vm @ np.linalg.solve(Vm.T @ K @ Vm, Vm.T @ F)
        row["modal_s"] = abs(um[39] - u_ref[39]) / abs(u_ref[39])
        rows.append(row)
        print("free", {k: (f"{v:.1e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(os.path.join(TAB, "ch11_free.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rlrrrr}\n\\toprule\n$n_m$ & model & size & lowest $n_m$ freq. & modes $n_m{+}1$ to size & static tip error \\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r['nm']} & modal truncation & {r['nm']} & {sci(1e-16) if False else '0'} & -- & {sci(r['modal_s'])} \\\\\n")
            fh.write(f" & Craig--Bampton & {r['cb_size']} & {sci(r['cb_low'])} & {sci(r['cb_top'])} & {sci(r['cb_s'])} \\\\\n")
            fh.write(f" & Rubin & {r['rubin_size']} & {sci(r['rubin_low'])} & {sci(r['rubin_top'])} & {sci(r['rubin_s'])} \\\\\n")
            fh.write(f" & MacNeal & {r['macneal_size']} & {sci(r['macneal_low'])} & -- & {sci(r['macneal_s'])} \\\\\n")
            fh.write("\\midrule\n" if r is not rows[-1] else "")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    r0 = rows[0]
    num("ccmFreeModalStatic", sci(r0["modal_s"])); num("ccmFreeRubinStatic", sci(r0["rubin_s"]))
    num("ccmFreeMacStatic", sci(r0["macneal_s"])); num("ccmFreeCbStatic", sci(r0["cb_s"]))
    num("ccmFreeModalStaticTen", sci(rows[2]["modal_s"])); num("ccmFreeMacStaticTen", sci(rows[2]["macneal_s"]))
    num("ccmFreeRubinTopThree", sci(r0["rubin_top"])); num("ccmFreeCbTopThree", sci(r0["cb_top"]))
    num("ccmFreeRubinTopTen", sci(rows[2]["rubin_top"])); num("ccmFreeCbTopTen", sci(rows[2]["cb_top"]))
    num("ccmFreeRubinLow", sci(max(r["rubin_low"] for r in rows)))
    num("ccmFreeMacLow", sci(max(r["macneal_low"] for r in rows)))
    num("ccmFreeCbLowThree", sci(r0["cb_low"]))


# ------------------------------------------------------------------------------------------ D
K3 = 4.0


class Grid:
    def __init__(self, nx, ny, x0, x1, side):
        xs, ys = np.linspace(x0, x1, nx), np.linspace(0, 1, ny)
        self.coords = np.array([[x, y] for x in xs for y in ys])
        self.n = nx * ny
        idx = lambda i, j: i * ny + j
        K = np.zeros((self.n, self.n))
        for i in range(nx):
            for j in range(ny):
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < nx and j + dj < ny:
                        a, b = idx(i, j), idx(i + di, j + dj)
                        K[a, a] += 1; K[b, b] += 1; K[a, b] -= 1; K[b, a] -= 1
        K += 0.5 * np.eye(self.n)
        outer = 0 if side == "right" else nx - 1
        self.outer = np.array([idx(outer, j) for j in range(ny)])
        self.interface = np.array([idx(nx - 1 if side == "right" else 0, j) for j in range(ny)])
        for d in self.outer:
            K[d, d] += 1e4
        self.K, self.f, self.ys = K, np.ones(self.n), ys

    def resid(self, u, lam):
        return self.K @ u + K3 * u ** 3 - lam * self.f

    def tang(self, u):
        return self.K + np.diag(3 * K3 * u ** 2)

    def solve_with_interface(self, g, lam):
        u = np.zeros(self.n); u[self.interface] = g
        free = np.setdiff1d(np.arange(self.n), self.interface)
        for _ in range(40):
            r = self.resid(u, lam)[free]
            if np.linalg.norm(r) < 1e-11:
                break
            u[free] -= np.linalg.solve(self.tang(u)[np.ix_(free, free)], r)
        return u


def full_solution(A, B, tie):
    na = A.n
    T = tie.transformation()
    resid = lambda u: np.concatenate([A.resid(u[:na], 1.5), B.resid(u[na:], 1.5)])
    tang = lambda u: sp.block_diag([A.tang(u[:na]), B.tang(u[na:])]).tocsr()
    x = np.zeros(T.shape[1])
    for _ in range(30):
        u = T @ x
        r = T.T @ resid(u)
        if np.linalg.norm(r) < 1e-11:
            break
        x -= np.linalg.solve((T.T @ tang(u) @ T).toarray(), r)
    return T @ x, resid, tang


def make_basis(A, B, ra, rb):
    GA, GB = legendre_interface_parametrisation(A.ys, 3), legendre_interface_parametrisation(B.ys, 3)
    wrap = lambda f: (lambda p: f(np.concatenate([p[:4], [1.5 + 1.5 * p[4]]])))
    fa = lambda p: A.solve_with_interface(GA @ p[:4], p[4]); fb = lambda p: B.solve_with_interface(GB @ p[:4], p[4])
    Sa, _ = component_snapshots(wrap(fa), 5, 30, scale=0.5, rng=1)
    Sb, _ = component_snapshots(wrap(fb), 5, 30, scale=0.5, rng=2)
    return (ComponentBasis(Sa, A.interface, rank=ra), ComponentBasis(Sb, B.interface, rank=rb, unreduced_interface=True))


def exp_d():
    A, B = Grid(9, 9, 0.0, 1.0, "right"), Grid(9, 7, 1.0, 2.0, "left")
    na = A.n
    tie = MortarTie2D(na + B.interface, A.interface, np.vstack([A.coords, B.coords]), na + B.n, dofs_per_node=1)
    u_ref, resid, tang = full_solution(A, B, tie)
    rows = []
    for r in (3, 6, 12):
        ba, bb = make_basis(A, B, r, r)
        res = ReducedAssembly(ba, bb, tie.P).solve(resid, tang)
        err = np.linalg.norm(res.u - u_ref) / np.linalg.norm(u_ref)
        tieerr = np.max(np.abs(res.u[na + B.interface] - tie.P @ res.u[A.interface]))
        rows.append((r, res.n_reduced, res.n_full, err, res.iterations, res.converged, tieerr))
        print("comp", rows[-1], flush=True)
    # reuse the master basis against a different slave mesh
    ba, bb = make_basis(A, B, 10, 10)
    B2 = Grid(9, 5, 1.0, 2.0, "left")
    tie2 = MortarTie2D(na + B2.interface, A.interface, np.vstack([A.coords, B2.coords]), na + B2.n, dofs_per_node=1)
    u_ref2, resid2, tang2 = full_solution(A, B2, tie2)
    G2 = legendre_interface_parametrisation(B2.ys, 3)
    Sb2, _ = component_snapshots(lambda p: B2.solve_with_interface(G2 @ p[:4], 1.5 + 1.5 * p[4]), 5, 30, scale=0.5, rng=3)
    bb2 = ComponentBasis(Sb2, B2.interface, rank=10, unreduced_interface=True)
    res2 = ReducedAssembly(ba, bb2, tie2.P).solve(resid2, tang2)
    err2 = np.linalg.norm(res2.u - u_ref2) / np.linalg.norm(u_ref2)
    with open(os.path.join(TAB, "ch11_component.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrr}\n\\toprule\nrank per component & reduced unknowns & full unknowns & relative error & Newton iterations & tie violation \\\\\n\\midrule\n")
        for r, nr, nf, e, it, cv, te in rows:
            fh.write(f"{r} & {nr} & {nf} & {sci(e)} & {it} & {sci(te)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("ccmCompFull", str(rows[0][2])); num("ccmCompErrThree", sci(rows[0][3])); num("ccmCompErrSix", sci(rows[1][3]))
    num("ccmCompErrTwelve", sci(rows[2][3])); num("ccmCompRedTwelve", str(rows[2][1]))
    num("ccmCompTie", sci(max(r[6] for r in rows))); num("ccmCompConv", "all" if all(r[5] for r in rows) else "not all")
    num("ccmCompReuseErr", sci(err2)); num("ccmCompReuseConv", "yes" if res2.converged else "no")
    num("ccmCompMono", "yes" if rows[0][3] > rows[1][3] > rows[2][3] else "no")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.semilogy([r[0] for r in rows], [r[3] for r in rows], "o-")
    ax.set_xlabel("POD rank per component"); ax.set_ylabel("relative error of the tied solution")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch11_component.pdf")); plt.close(fig)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_a(); exp_b(); exp_c(); exp_d()
    num("ccmMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch11.tex"), "w") as fh:
        fh.write("% generated by scripts/ch11_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

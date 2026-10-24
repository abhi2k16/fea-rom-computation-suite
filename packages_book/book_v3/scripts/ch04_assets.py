# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch04_assets.py -- figures, tables and number macros of Chapter 4 (dynamics).

Writes into book/:
    figures/ch04_newmark.pdf      Newmark average acceleration: error against the time step, against the exact discrete solution
    figures/ch04_explicit.pdf     central differences on a lumped mass: response against step / critical step
    figures/ch04_modal.pdf        step response of a cantilever: modal superposition with 1, 2, 4, 8 modes against all modes
    figures/ch04_harmonic.pdf     frequency sweep: direct complex solve against an independent modal sum
    tables/ch04_stability.tex     explicit integration around the critical step
    tables/numbers_ch04.tex       every number quoted in the prose

Reference solutions are exact for the discrete model or independent of the solver (modal sums built from the eigenpairs).
Run from the repository root:   python packages_book/book/scripts/ch04_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
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


from fea_engine import (FESystem, Material, Beam2DEulerBernoulli, Quad4PlaneStress, D_plane_stress, convergence as cv)   # noqa: E402
from fea_engine.damping import RayleighDamping                                                                          # noqa: E402
from fea_engine.geometry import generate_mesh                                                                           # noqa: E402
from fea_engine.loads import LoadPattern, TimeHistoryLoad                                                               # noqa: E402
from fea_engine.material import EI_beam, Section                                                                        # noqa: E402
from fea_engine.mesh import rectangle_mesh                                                                              # noqa: E402

E, RHO, A, I, L = 210e9, 7800.0, 0.01, 8.33e-6, 2.0
NEL = 20


def beam():
    b = FESystem(generate_mesh(dim=1, L=L, n=NEL), Beam2DEulerBernoulli())
    b.assemble_stiffness(EI_beam(Material(E=E, nu=0.3, rho=RHO), Section(A=A, I=I)))
    b.assemble_mass(RHO * A)
    b.fix_dofs([0], ["uy", "rz"])
    b.assemble_damping(RayleighDamping(0.0, 0.0))      # undamped; later calls add to it
    return b


def tip_dof(b):
    return b.npn * NEL + 0


# ------------------------------------------------------------------------------------------ (A) Newmark order
def exp_newmark():
    b = beam()
    f, Phi = b.solve_modal(n_modes=1)
    Phi = np.asarray(Phi)[:, 0]
    w = 2 * np.pi * f[0]
    T1 = 1 / f[0]
    free = b.free_dofs
    errs, dts = [], []
    amp = []
    for k in (10, 20, 40, 80, 160):
        dt = T1 / k
        tt, U = b.solve_transient_implicit(TimeHistoryLoad(LoadPattern(np.array([NEL]), 0), lambda t: 0.0), 3 * T1, dt, v0=Phi)
        exact = np.outer(np.sin(w * tt) / w, Phi)
        err = np.abs(U - exact).max() / np.abs(exact).max()
        errs.append(err); dts.append(dt / T1)
        amp.append(np.abs(U[:, tip_dof(b)]).max() / np.abs(exact[:, tip_dof(b)]).max())
        print(f"newmark dt=T1/{k}: error {err:.3e}, tip amplitude ratio {amp[-1]:.6f}", flush=True)
    order = cv.observed_order(np.array(dts), np.array(errs))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(dts, errs, "o-", label=f"observed order {order:.2f}")
    d = np.array(dts); ax.loglog(d, errs[0] * (d / d[0]) ** 2, "k:", lw=1, label="slope 2")
    ax.set_xlabel("$\\Delta t/T_1$"); ax.set_ylabel("max error over 3 periods (relative)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch04_newmark.pdf")); plt.close(fig)
    num("cdyNewmarkOrder", f"{order:.2f}"); num("cdyNewmarkErrCoarse", sci(errs[0])); num("cdyNewmarkErrFine", sci(errs[-1]))
    num("cdyNewmarkAmpWorst", f"{max(abs(a - 1) for a in amp) * 100:.0f}\\%")
    num("cdyFirstFreq", f"{f[0]:.3f}")


# ------------------------------------------------------------------------------------------ (B) explicit stability
def exp_explicit():
    Eb, rho, Lb, n = 1.0e9, 1000.0, 1.0, 20
    mesh = rectangle_mesh(Lx=Lb, Ly=0.1, nx=n, ny=1)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
    s.assemble_stiffness(D_plane_stress(Material(E=Eb, nu=0.0, rho=rho)), thickness=0.1)
    s.assemble_mass(rho * np.eye(2), thickness=0.1)
    s.assemble_lumped_mass(rho * np.eye(2), thickness=0.1)
    mesh.select_nodes(x=0.0, name="root")
    s.fix_dofs("root", ["ux"])
    s.fix_dofs(np.arange(len(mesh.nodes)), ["uy"])
    free = s.free_dofs
    K = np.asarray(s.K.toarray() if hasattr(s.K, "toarray") else s.K)[np.ix_(free, free)]
    Ml = np.asarray(s.M_lumped.toarray() if hasattr(s.M_lumped, "toarray") else s.M_lumped)[np.ix_(free, free)]
    wmax_l = np.sqrt(sl.eigh(K, Ml, eigvals_only=True)[-1])
    dtc_l = 2 / wmax_l
    dtc_c = s.critical_timestep()
    dtc_loc = s.critical_timestep_local(np.sqrt(Eb / rho), alpha=1.0)
    print(f"critical dt: lumped {dtc_l:.4e}, consistent (critical_timestep) {dtc_c:.4e}, local estimate {dtc_loc:.4e}", flush=True)
    tip = np.flatnonzero(np.abs(mesh.nodes[:, 0] - Lb) < 1e-9)
    rows = []
    for r in (0.5, 0.9, 0.99, 1.01, 1.1):
        dt = r * dtc_l
        N = 600
        load = TimeHistoryLoad(LoadPattern(tip, 0), lambda t: 1.0e5 if t < 20 * dtc_l else 0.0)
        t, U = s.solve_transient_explicit(load, N * dt, dt)
        Uf = np.abs(U[:, :]).max(axis=1)
        grow = Uf[-50:].max() / max(Uf[100:200].max(), 1e-300)
        rows.append((r, grow, Uf.max()))
        print(f"explicit dt/dtc_lumped={r}: late/early amplitude {grow:.3g}, max |u| {Uf.max():.3e}", flush=True)
    with open(os.path.join(TAB, "ch04_stability.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrr}\n\\toprule\n$\\Delta t/\\Delta t_c$ & late/early amplitude & largest $|u|$ (m) \\\\\n\\midrule\n")
        for r, g, mx in rows:
            fh.write(f"{r} & {sci(g) if g > 100 else f'{g:.2f}'} & {sci(mx)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for r in (0.9, 1.1):
        dt = r * dtc_l
        load = TimeHistoryLoad(LoadPattern(tip, 0), lambda t: 1.0e5 if t < 20 * dtc_l else 0.0)
        t, U = s.solve_transient_explicit(load, 600 * dt, dt)
        ax.semilogy(t / dtc_l, np.maximum(np.abs(U).max(axis=1), 1e-30), label=f"$\\Delta t = {r}\\,\\Delta t_c$")
    ax.set_xlabel("$t/\\Delta t_c$"); ax.set_ylabel("max displacement (m)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch04_explicit.pdf")); plt.close(fig)
    num("cdyDtcLumped", sci(dtc_l)); num("cdyDtcConsistent", sci(dtc_c))
    num("cdyDtcRatio", f"{dtc_l / dtc_c:.2f}")
    num("cdyDtcLocal", f"{dtc_loc / dtc_l:.2f}")
    num("cdyGrowStable", f"{rows[1][1]:.2f}"); num("cdyGrowUnstable", sci(rows[4][1]) if rows[4][1] > 100 else f"{rows[4][1]:.2f}")
    num("cdyGrowJust", f"{rows[2][1]:.2f}"); num("cdyGrowJustOver", sci(rows[3][1]) if rows[3][1] > 100 else f"{rows[3][1]:.2f}")


# ------------------------------------------------------------------------------------------ (C) modal superposition
def exp_modal():
    b = beam()
    zeta = 0.01
    nfree = len(b.free_dofs)
    f, _ = b.solve_modal(n_modes=1)
    T1 = 1 / f[0]
    P = 1000.0
    dt = T1 / 200
    Tt = 4 * T1
    load = TimeHistoryLoad(LoadPattern(np.array([NEL]), 0), lambda t: P)
    # uy is dof index 0 for this element at the tip node
    td = tip_dof(b)
    static = P * L ** 3 / (3 * E * I)
    hist = {}
    for nm in (1, 2, 4, 8, nfree - 1):
        t, U, _ = b.solve_modal_superposition(load, Tt, dt, nm, zeta)
        hist[nm] = U[:, td]
        print(f"modal nm={nm}: peak/static {np.abs(U[:, td]).max() / static:.4f}", flush=True)
    ref = hist[nfree - 1]
    errs = {nm: np.abs(hist[nm] - ref).max() / np.abs(ref).max() for nm in hist}
    sdof = 1 + np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for nm in (1, 2, 8):
        ax[0].plot(t / T1, hist[nm] / static, label=f"{nm} mode{'s' if nm > 1 else ''}")
    ax[0].plot(t / T1, ref / static, "k:", lw=1, label=f"all {nfree - 1}")
    ax[0].set_xlabel("$t/T_1$"); ax[0].set_ylabel("tip deflection / static"); ax[0].legend(fontsize=8)
    ks = sorted(errs)
    ax[1].semilogy(ks[:-1], [errs[k] for k in ks[:-1]], "o-")
    ax[1].set_xlabel("modes retained"); ax[1].set_ylabel("max error against all modes (relative)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch04_modal.pdf")); plt.close(fig)
    num("cdyPeakAll", f"{np.abs(ref).max() / static:.3f}")
    num("cdyPeakOne", f"{np.abs(hist[1]).max() / static:.3f}")
    num("cdyPeakSDOF", f"{sdof:.3f}")
    num("cdyPeakRatioOne", f"{np.abs(hist[1]).max() / static / sdof:.3f}")
    num("cdyModalErrOne", sci(errs[1])); num("cdyModalErrFour", sci(errs[4])); num("cdyModalErrEight", sci(errs[8]))
    num("cdyModalAll", str(nfree - 1))
    # full-system Newmark with Rayleigh damping against the same step load
    f2, _ = b.solve_modal(n_modes=3)
    ray = RayleighDamping.calibrate(2 * np.pi * f2[0], 2 * np.pi * f2[2], zeta)
    b.assemble_damping(ray)
    t2, U2 = b.solve_transient_implicit(load, Tt, dt)
    num("cdyPeakNewmark", f"{np.abs(U2[:, td]).max() / static:.3f}")
    num("cdyRayleighMid", f"{ray.modal_ratio(2 * np.pi * f2[1]) * 100:.2f}")
    print(f"newmark rayleigh peak/static {np.abs(U2[:, td]).max()/static:.4f}; zeta at mode 2 {ray.modal_ratio(2*np.pi*f2[1]):.4f}", flush=True)


# ------------------------------------------------------------------------------------------ (D) harmonic and random
def exp_harmonic():
    b = beam()
    nfree = len(b.free_dofs)
    fq, Phi = b.solve_modal(n_modes=nfree - 1)
    Phi = np.asarray(Phi)
    w_i = 2 * np.pi * fq
    zeta = 0.01
    ray = RayleighDamping.calibrate(w_i[0], w_i[2], zeta)
    b.assemble_damping(ray)
    zi = np.array([ray.modal_ratio(w) for w in w_i])
    td = tip_dof(b)
    F0 = np.zeros(b.n_dof); F0[td] = 1.0
    # static check at Omega = 0
    U0 = np.asarray(b.solve_harmonic(0.0, F0).values if hasattr(b.solve_harmonic(0.0, F0), "values") else b.solve_harmonic(0.0, F0))
    Us = b.solve_static_with_force(F0) if hasattr(b, "solve_static_with_force") else None
    stat = 1.0 * L ** 3 / (3 * E * I)
    num("cdyHarmZeroErr", sci(abs(U0[td].real / stat - 1)))
    freqs = np.unique(np.concatenate([np.linspace(0.5, 160.0, 800), np.linspace(20.0, 22.0, 401)]))
    Om = 2 * np.pi * freqs
    H_dir = np.array([np.asarray(b.solve_harmonic(w, F0).values if hasattr(b.solve_harmonic(w, F0), "values") else b.solve_harmonic(w, F0))[td] for w in Om])
    ph = Phi[td, :]
    H_mod = np.array([np.sum(ph * ph / (w_i ** 2 - w ** 2 + 2j * zi * w_i * w)) for w in Om])
    diff = np.abs(H_dir - H_mod).max() / np.abs(H_mod).max()
    k1 = np.argmax(np.abs(H_dir)); peak = np.abs(H_dir[k1])
    print(f"harmonic: static check {abs(U0[td].real/stat-1):.2e}; direct vs modal sum {diff:.2e}; peak at {freqs[k1]:.2f} Hz, amplification {peak/stat:.1f}", flush=True)
    share = ph[0] ** 2 / w_i[0] ** 2 / stat
    num("cdyShareOne", f"{share:.3f}"); num("cdyHarmPred", f"{share / (2 * zeta):.1f}")
    num("cdyHarmDiff", sci(diff)); num("cdyHarmPeakHz", f"{freqs[k1]:.1f}"); num("cdyHarmAmp", f"{peak / stat:.0f}")
    num("cdyQuality", f"{1 / (2 * zeta):.0f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.semilogy(freqs, np.abs(H_dir) / stat, label="direct solve (\\texttt{solve\\_harmonic})")
    ax.semilogy(freqs, np.abs(H_mod) / stat, "k--", lw=0.9, label="modal sum (independent)")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|tip receptance| / static"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch04_harmonic.pdf")); plt.close(fig)
    # random vibration: flat input PSD, RMS against the same integral of the independent modal sum
    S0 = 1.0
    fr = np.linspace(1.0, 160.0, 1600)
    S_out, sig = b.solve_random_vibration(fr, np.full_like(fr, S0), F0, td)
    Hm = np.array([np.sum(ph * ph / (w_i ** 2 - (2 * np.pi * f_) ** 2 + 2j * zi * w_i * 2 * np.pi * f_)) for f_ in fr])
    sig_ref = np.sqrt(np.trapz(np.abs(Hm) ** 2 * S0, fr))
    num("cdyRandomDiff", sci(abs(sig / sig_ref - 1)))
    print(f"random: rms {sig:.4e} vs modal-sum {sig_ref:.4e}", flush=True)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_newmark(); exp_explicit(); exp_modal(); exp_harmonic()
    num("cdyMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch04.tex"), "w") as fh:
        fh.write("% generated by scripts/ch04_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

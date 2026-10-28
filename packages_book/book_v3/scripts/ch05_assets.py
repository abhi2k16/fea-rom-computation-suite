# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch05_assets.py -- figures, tables and number macros of Chapter 5 (material and geometric nonlinearity).

Runs fea_engine (and nothing else) and writes into book/:

    figures/ch05_objectivity.pdf   internal-force norm of a rigidly rotated element: total Lagrangian against linear kinematics
    figures/ch05_vonmises.pdf      von Mises truss: load control, displacement control and arc length against the closed form
    figures/ch05_plasticity.pdf    TrussPlastic2D cycle: stress-strain loop against the closed form
    tables/ch05_tangent.tex        analytic tangent against central finite differences, five elements
    tables/numbers_ch05.tex        every number quoted in the prose, as a macro

Run from the repository root:   python packages_book/book/scripts/ch05_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls
from fea_engine.material import PlasticMaterial1D

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    """x as LaTeX scientific notation (math mode included)."""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# ------------------------------------------------------------------------------------------ (A) objectivity
def exp_objectivity():
    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    D = 1.0e3 / (1 - 0.3 ** 2) * np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, (1 - 0.3) / 2]])
    el_tl, el_lin = elmod.Quad4PlaneStressTL(), elmod.Quad4PlaneStress()
    K_lin = el_lin.stiffness(X, D)
    th = np.radians(np.linspace(0, 180, 91))
    f_tl, f_lin = [], []
    for a in th:
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        u = ((X @ R.T) - X).ravel()                    # rigid rotation about the origin
        f_tl.append(np.linalg.norm(el_tl.internal_force(X, u, D)))
        f_lin.append(np.linalg.norm(K_lin @ u))
    f_tl, f_lin = np.array(f_tl), np.array(f_lin)
    deg = np.degrees(th)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.semilogy(deg, np.maximum(f_lin, 1e-18), label="linear kinematics, $\\|K u\\|$")
    ax.semilogy(deg, np.maximum(f_tl, 1e-18), "--", label="total Lagrangian, $\\|f_{int}(u)\\|$")
    ax.set_xlabel("rigid rotation angle (degrees)"); ax.set_ylabel("force norm for a stress-free motion")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch05_objectivity.pdf")); plt.close(fig)
    i10, i90 = int(np.argmin(abs(deg - 10))), int(np.argmin(abs(deg - 90)))
    num("cnlObjTLMax", sci(f_tl.max()))
    num("cnlObjLinTen", f"{f_lin[i10]:.2f}")
    num("cnlObjLinNinety", f"{f_lin[i90]:.2f}")
    num("cnlObjLinMax", f"{f_lin.max():.2f}")
    print(f"objectivity: TL max {f_tl.max():.2e}; linear at 10 deg {f_lin[i10]:.3f}, 90 deg {f_lin[i90]:.3f}", flush=True)


# ------------------------------------------------------------------------------------------ (B) tangent check
def fd_tangent(el, X, u, mat, h=1e-6):
    n = len(u)
    K = np.zeros((n, n))
    for j in range(n):
        up, um = u.copy(), u.copy()
        up[j] += h; um[j] -= h
        K[:, j] = (el.internal_force(X, up, mat) - el.internal_force(X, um, mat)) / (2 * h)
    return K


def exp_tangent():
    rng = np.random.default_rng(3)
    D2 = 2.1e5 / (1 - 0.09) * np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 0.35]])
    from fea_engine.material import Material
    mat3 = Material(E=2.1e5, nu=0.3)
    Xq = np.array([[0, 0], [1, 0], [1.1, 1], [0, 0.9]], float)
    Xh = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    Xt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5], [.5, 0, .5], [0, .5, .5]], float)
    Xtr = np.array([[0.0, 0.0], [1.0, 0.2]])
    Xb = np.array([[0.0, 0.0], [1.0, 0.1]])
    cases = [("Quad4PlaneStressTL", elmod.Quad4PlaneStressTL(), Xq, D2, 0.05, "plane-stress $D$"),
             ("Hex8SolidTL", elmod.Hex8SolidTL(), Xh, mat3, 0.05, "\\code{Material}"),
             ("Tet10SolidTL", elmod.Tet10SolidTL(), Xt, mat3, 0.05, "\\code{Material}"),
             ("TrussTL2D", elmod.TrussTL2D(), Xtr, (210e9, 2e-4), 0.1, "$(E,A)$"),
             ("Beam2DCorotational", elmod.Beam2DCorotational(), Xb, (2.1e5, 1.0, 0.05), 0.05, "$(E,A,I)$")]
    rows = []
    for name, el, X, mat, amp, mt in cases:
        n = el.n_nodes * el.dofs_per_node
        u = amp * rng.standard_normal(n)
        if name == "Beam2DCorotational":
            u[2::3] *= 0.5
        K = el.tangent_stiffness(X, u, mat)
        Kf = fd_tangent(el, X, u, mat)
        rel = np.linalg.norm(K - Kf) / np.linalg.norm(Kf)
        sym = np.linalg.norm(K - K.T) / np.linalg.norm(K)
        rows.append((name, mt, n, rel, sym))
        print(f"tangent {name}: {n} dofs, |K-K_fd|/|K_fd| = {rel:.1e}, asymmetry {sym:.1e}", flush=True)
    with open(os.path.join(TAB, "ch05_tangent.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llrrr}\n\\toprule\nelement & \\code{mat} & dofs & $\\|K_T-K_{fd}\\|/\\|K_{fd}\\|$ & $\\|K_T-K_T^{\\top}\\|/\\|K_T\\|$ \\\\\n\\midrule\n")
        for name, mt, n, rel, sym in rows:
            fh.write(f"\\code{{{name}}} & {mt} & {n} & {sci(rel)} & {sci(sym)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cnlTangentWorst", sci(max(r[3] for r in rows)))


# ------------------------------------------------------------------------------------------ (C) von Mises truss
A_HALF, H0, E_, A_ = 1.0, 0.10, 210e9, 2e-4
L0 = np.sqrt(A_HALF ** 2 + H0 ** 2)
P_cf = lambda d: (E_ * A_ / L0 ** 3) * d * (H0 - d) * (2 * H0 - d)


def truss():
    mesh = Mesh(nodes=np.array([[-A_HALF, 0.0], [0.0, H0], [A_HALF, 0.0]]), elements=np.array([[0, 1], [1, 2]], dtype=int), dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    return fes, 1 * fes.npn + 1


def exp_vonmises():
    dpk = H0 * (3 - np.sqrt(3)) / 3
    Ppk = P_cf(dpk)
    num("cnlTrussPeak", f"{Ppk:.0f}")
    num("cnlTrussPeakDelta", f"{dpk * 1e3:.2f}")
    fes, _ = truss()
    fes.add_nodal_force([1], 1, -1.1 * Ppk)
    try:
        nls.solve_nonlinear_static(fes, (E_, A_), n_steps=20, tol=1e-12)
        failed = False
    except RuntimeError:
        failed = True
    num("cnlLoadCtrlFailed", "fails" if failed else "converges")
    fes, cd = truss()
    dd = np.linspace(0.0, 2 * H0, 81)
    Udc, R = nls.solve_nonlinear_displacement_control(fes, (E_, A_), cd, -dd, tol=1e-12, max_iter=50)
    Pdc = -R
    err_dc = np.max(np.abs(Pdc - P_cf(dd))) / np.abs(Pdc).max()
    fes, cd = truss()
    fes.add_nodal_force([1], 1, -1.0)
    lam, Ual = nls.solve_nonlinear_arc_length(fes, (E_, A_), delta_L=0.01, n_steps=60, tol=1e-10, max_iter=60)
    dal = -Ual[:, cd]
    ref = P_cf(dal)
    sig = np.abs(ref) > 1e-4 * np.abs(ref).max()
    err_al = np.max(np.abs(lam - ref)[sig] / np.abs(ref)[sig])
    num("cnlTrussErrDC", sci(err_dc)); num("cnlTrussErrAL", sci(err_al))
    num("cnlTrussALpts", f"{int((dal <= 2.0 * H0).sum())}")
    # Koiter--Newton: reference load = the limit load, so lambda is of order one; cubic predictor, chord corrector
    def run_kn(**kw):
        f, c_ = truss(); f.add_nodal_force([1], 1, -Ppk)
        l_, U_ = nls.solve_nonlinear_koiter_newton(f, (E_, A_), delta_L=0.01, n_steps=60, tol=1e-9, max_iter=60, **kw)
        d_ = -U_[:, c_]; P_ = l_ * Ppk
        return d_, P_
    dkn, Pkn = run_kn(max_growth_iters=0)
    kn_keep = dkn <= 2.0 * H0
    kn_sig = np.abs(P_cf(dkn)) > 1e-4 * np.abs(P_cf(dkn)).max()
    err_kn = np.max((np.abs(Pkn - P_cf(dkn))[kn_sig & kn_keep]) / np.abs(P_cf(dkn))[kn_sig & kn_keep])
    num("cnlTrussErrKN", sci(err_kn)); num("cnlTrussKNpts", f"{int(kn_keep.sum())}")
    dkd, Pkd = run_kn()                       # default settings: the step grows by 2**6 at every step
    num("cnlTrussKNdefStep", f"{dkd[1] * 1e3:.0f}")
    num("cnlTrussKNdefErr", sci(np.max(np.abs(Pkd - P_cf(dkd))[1:] / np.abs(P_cf(dkd))[1:])))
    print(f"koiter-newton: fixed step {int(kn_keep.sum())} pts err {err_kn:.1e}; default growth first step {dkd[1]*1e3:.0f} mm", flush=True)
    print(f"truss: P_peak {Ppk:.1f} N at {dpk*1e3:.2f} mm; load control past peak {'fails' if failed else 'converges'}; "
          f"disp-control err {err_dc:.1e}; arc-length err {err_al:.1e}, delta up to {dal.max()*1e3:.1f} mm", flush=True)
    d = np.linspace(0, 2 * H0, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(d * 1e3, P_cf(d) / 1e3, "k", lw=1, label="closed form")
    ax.plot(dd * 1e3, Pdc / 1e3, "o", ms=3, mfc="none", label="displacement control")
    keep = dal <= 2.0 * H0
    ax.plot(dal[keep] * 1e3, lam[keep] / 1e3, "x", ms=3.5, label="arc length")
    ax.plot(dkn[kn_keep] * 1e3, Pkn[kn_keep] / 1e3, "+", ms=4, color="tab:green", label="Koiter\u2013Newton")
    ax.axhline(Ppk / 1e3, color="r", ls=":", lw=0.8)
    ax.annotate("limit load: load control cannot pass", (dpk * 1e3 + 4, Ppk / 1e3), fontsize=8, color="r", va="bottom")
    ax.set_xlabel("apex deflection $\\delta$ (mm)"); ax.set_ylabel("load $P$ (kN)")
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch05_vonmises.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ (D) plasticity
def exp_plasticity():
    E, sy, H, A, L = 200e9, 250e6, 5e9, 1.0e-4, 1.0
    el, mat = elmod.TrussPlastic2D(), (PlasticMaterial1D(E=E, sigma_y=sy, H=H), 1.0e-4)
    X = np.array([[0.0, 0.0], [L, 0.0]])
    st = el.init_state()
    peak = 1.8 * sy
    e_peak = peak / E + (peak - sy) / H
    n1 = 1000
    path = np.concatenate([np.linspace(0, e_peak, n1 + 1), np.linspace(e_peak, -0.6 * e_peak, 2 * n1 + 1)[1:]])
    sig, epsp = [], []
    for e in path:
        u = np.array([0, 0, e * L, 0.0])
        f = el.internal_force(X, u, mat, state=st)
        sig.append(f[2] / A)
        st = el.commit_state(X, u, mat, st)
        epsp.append(st["eps_p"])
    sig = np.array(sig)
    # closed form
    s_ref = np.empty_like(path); sp_ = 0.0; ep = 0.0; al = 0.0
    for k, e in enumerate(path):
        s_tr = E * (e - ep)
        fy = abs(s_tr) - (sy + H * al)
        if fy > 0:
            dg = fy / (E + H); s_tr -= E * dg * np.sign(s_tr); ep += dg * np.sign(s_tr if s_tr != 0 else 1); al += dg
            ep = e - s_tr / E
        s_ref[k] = s_tr
    err = np.abs(sig - s_ref).max() / peak
    slope = np.diff(sig) / np.diff(path)
    rev = n1 + 1 + int(np.flatnonzero(slope[n1:] < 0.9 * E)[0]) if (slope[n1:] < 0.9 * E).any() else None
    sig_rev = sig[rev - 1]            # stress at which the reversed branch leaves the elastic line
    num("cnlPlPeak", f"{sig[:n1 + 1].max() / sy:.2f}")
    num("cnlPlReverseYield", f"{-sig_rev / sy:.2f}")
    num("cnlPlHardenedYield", f"{peak / sy:.1f}")
    num("cnlPlErr", sci(err))
    num("cnlPlUnloadSlope", f"{slope[n1] / E:.3f}")
    num("cnlPlTangentSlope", f"{E * H / (E + H) / 1e9:.2f}")
    num("cnlPlMin", f"{sig.min() / sy:.2f}")
    ep_end = st["eps_p"]
    print(f"plasticity: peak {sig.max()/sy:.3f} sy, min {sig.min()/sy:.3f} sy, vs closed form {err:.1e}, unload slope/E {slope[n1]/E:.3f}, reverse yield at {-sig_rev/sy:.3f} sy, "
          f"eps_p end {ep_end:.4e}", flush=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(path * 1e3, sig / 1e6, label="TrussPlastic2D")
    ax.plot(path[::40] * 1e3, s_ref[::40] / 1e6, "k.", ms=3, label="closed form")
    ax.set_xlabel("strain ($10^{-3}$)"); ax.set_ylabel("stress (MPa)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ch05_plasticity.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ (E) drivers through FESystem
def exp_drivers():
    from scipy.optimize import brentq
    from fea_engine import NewtonOptions
    dpk = H0 * (3 - np.sqrt(3)) / 3
    Ppk = P_cf(dpk)
    # (1) load control below the limit point, options object
    fes, cd = truss()
    fes.add_nodal_force([1], 1, -0.8 * Ppk)
    opts = NewtonOptions(tol=1e-10, max_iter=40, line_search=True)
    lf, U = nls.solve_nonlinear_static(fes, (E_, A_), n_steps=8, options=opts)
    d_fe = -U[-1, cd]
    d_ref = brentq(lambda d: P_cf(d) - 0.8 * Ppk, 0.0, dpk)
    num("cnlLCdelta", f"{d_fe * 1e3:.3f}")
    num("cnlLCerr", sci(abs(d_fe - d_ref) / d_ref))
    # (2) plasticity through FESystem: init_state, load history, commit_state once per converged step
    E, sy, H, A = 200e9, 250e6, 5e9, 1.0e-4
    mesh = Mesh(nodes=np.array([[0.0, 0.0], [1.0, 0.0]]), elements=np.array([[0, 1]], dtype=int), dim=2)
    fes = FESystem(mesh, elmod.TrussPlastic2D())
    fes.fix_dofs([0], [0, 1]); fes.fix_dofs([1], [1])
    mat = (PlasticMaterial1D(E=E, sigma_y=sy, H=H), A)
    peak = 1.8 * sy
    fes.add_nodal_force([1], 0, peak * A)
    fes.init_state()
    load = np.concatenate([np.linspace(0, 1, 41), np.linspace(1, 0, 41)[1:]])
    _, Uh = nls.solve_nonlinear_static(fes, mat, load_factors=load, tol=1e-9, max_iter=50)
    eps = Uh[:, 2]
    e_peak = peak / E + (peak - sy) / H
    ep_ref = (peak - sy) / H
    num("cnlFesPeakErr", sci(abs(eps[40] - e_peak) / e_peak))
    num("cnlFesSetErr", sci(abs(eps[-1] - ep_ref) / ep_ref))
    num("cnlFesSet", f"{eps[-1] * 1e3:.3f}")
    print(f"drivers: load control delta {d_fe*1e3:.4f} mm vs {d_ref*1e3:.4f} mm; FESystem plastic peak err {abs(eps[40]-e_peak)/e_peak:.1e}, "
          f"permanent set {eps[-1]:.4e} vs {ep_ref:.4e}", flush=True)


# ------------------------------------------------------------------------------------------ (F) elastica
def exp_elastica():
    """Cantilever with a dead tip load, PL^2/EI = 1: corotational beam chain against the elastica ODE."""
    from scipy.integrate import solve_bvp, cumulative_trapezoid
    E, A, I, L, P = 1.0, 1.0e5, 1.0, 1.0, 1.0          # EA large: the elastica rod is inextensible
    s = np.linspace(0, L, 400)
    sol = solve_bvp(lambda t, y: np.vstack([y[1], (P / (E * I)) * np.cos(y[0])]),
                    lambda ya, yb: np.array([ya[0], yb[1]]), s, np.zeros((2, s.size)), tol=1e-12, max_nodes=100000)
    assert sol.success
    ss = np.linspace(0, L, 20001)
    phi = sol.sol(ss)[0]
    w_ref = abs(cumulative_trapezoid(np.sin(phi), ss, initial=0.0)[-1]) / L
    sh_ref = 1.0 - cumulative_trapezoid(np.cos(phi), ss, initial=0.0)[-1] / L
    num("cnlElasticaRef", f"{w_ref:.6f}")
    num("cnlElasticaShortRef", f"{sh_ref:.6f}")
    rows = []
    for ne in (10, 20, 40):
        x = np.linspace(0, L, ne + 1).reshape(-1, 1)
        mesh = Mesh(nodes=np.hstack([x, np.zeros_like(x)]),
                    elements=np.array([[i, i + 1] for i in range(ne)], dtype=int), dim=1)
        fes = FESystem(mesh, elmod.Beam2DCorotational())
        fes.fix_dofs([0], [0, 1, 2])
        fes.add_nodal_force([ne], 1, P)
        _, U = nls.solve_nonlinear_static(fes, (E, A, I), load_factors=np.linspace(0.05, 1.0, 20), tol=1e-8, max_iter=50)
        w, sh = U[-1, ne * 3 + 1] / L, -U[-1, ne * 3] / L
        ew, es = abs(w - w_ref) / w_ref, abs(sh - sh_ref) / sh_ref
        rows.append((ne, w, sh, ew, es))
        nm = {10: "Ten", 20: "Twenty", 40: "Forty"}[ne]
        num(f"cnlEl{nm}W", sci(ew)); num(f"cnlEl{nm}S", sci(es))
        print(f"elastica n={ne}: w/L {w:.6f} (ref {w_ref:.6f}) err {ew:.1e}; shortening/L {sh:.6f} (ref {sh_ref:.6f}) err {es:.1e}", flush=True)
    with open(os.path.join(TAB, "ch05_elastica.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrr}\n\\toprule\nelements & $w/L$ & shortening$/L$ & error in $w$ & error in shortening \\\\\n\\midrule\n")
        for ne, w, sh, ew, es in rows:
            fh.write(f"{ne} & {w:.6f} & {sh:.6f} & {sci(ew)} & {sci(es)} \\\\\n")
        fh.write(f"reference & {w_ref:.6f} & {sh_ref:.6f} & -- & -- \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_objectivity(); exp_tangent(); exp_vonmises(); exp_plasticity(); exp_drivers(); exp_elastica()
    num("cnlMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch05.tex"), "w") as fh:
        fh.write("% generated by scripts/ch05_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

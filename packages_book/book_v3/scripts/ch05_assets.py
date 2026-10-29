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
    # logarithmic force axis, linear angle axis (0 degrees is not plotted: both forces are exactly zero there)
    deg = np.degrees(th)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.semilogy(deg[1:], f_lin[1:], label="linear kinematics, $\\|K u\\|$")
    ax.semilogy(deg[1:], f_tl[1:], "--", label="total Lagrangian, $\\|f_{int}(u)\\|$")
    ax.set_xlim(0, 180)
    ax.set_xlabel("rigid rotation angle (degrees)"); ax.set_ylabel("force norm for a stress-free motion")
    ax.grid(True, which="major", alpha=0.3); ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch05_objectivity.pdf")); plt.close(fig)
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
        failed = False; msg = ""
    except RuntimeError as exc:
        failed = True; msg = str(exc)
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
    keep = dal <= 2.0 * H0                    # the path up to the inversion at 2 h0 (the points of the listing and the figure)
    err_al = np.max(np.abs(lam[keep] - ref[keep])) / lam[keep].max()      # one measure for all methods: error / peak load
    num("cnlTrussALPeak", f"{lam[keep].max():.0f}")
    num("cnlTrussErrDC", sci(err_dc)); num("cnlTrussErrAL", sci(err_al))
    # the output block printed under the listing of the chapter: the lines of that listing, run here
    with open(os.path.join(TAB, "ch05_vm_output.txt"), "w") as fh:
        fh.write(f"limit load {Ppk:.0f} N at apex deflection {dpk*1e3:.2f} mm\n")
        fh.write(f"load control at 1.1 P_max: {'failed - ' + msg[:60] if failed else 'converged'}\n")
        fh.write(f"displacement control: peak load {np.max(Pdc):.0f} N, worst error against the closed form {err_dc:.1e}\n")
        fh.write(f"arc length: {int(keep.sum())} points up to {2*H0*1e3:.0f} mm, peak load {lam[keep].max():.0f} N, worst error {err_al:.1e}\n")
    num("cnlTrussALpts", f"{int((dal <= 2.0 * H0).sum())}")
    # Koiter--Newton: reference load = the limit load, so lambda is of order one; cubic predictor, chord corrector
    def run_kn(**kw):
        f, c_ = truss(); f.add_nodal_force([1], 1, -Ppk)
        l_, U_ = nls.solve_nonlinear_koiter_newton(f, (E_, A_), delta_L=0.01, n_steps=60, tol=1e-10, max_iter=60, **kw)
        d_ = -U_[:, c_]; P_ = l_ * Ppk
        return d_, P_
    import io, contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):                # the corrector's own log: iterations and retries
        dkn, Pkn = run_kn(max_growth_iters=0, verbose=True)
    _log = _buf.getvalue().splitlines()
    num("cnlTrussKNits", str(sum(l.strip().startswith("it ") for l in _log)))
    num("cnlTrussKNretries", str(sum("shrinking and retrying" in l for l in _log)))
    kn_keep = dkn <= 2.0 * H0
    err_kn = np.max(np.abs(Pkn - P_cf(dkn))[kn_keep]) / Pkn[kn_keep].max()
    num("cnlTrussErrKN", sci(err_kn)); num("cnlTrussKNpts", f"{int(kn_keep.sum())}")
    dkd, Pkd = run_kn()                       # default settings: the step grows by 2**6 at every step
    num("cnlTrussKNdefStep", f"{dkd[1] * 1e3:.0f}")
    num("cnlTrussKNdefErr", sci(np.max(np.abs(Pkd - P_cf(dkd))[1:] / np.abs(P_cf(dkd))[1:])))
    print(f"koiter-newton: fixed step {int(kn_keep.sum())} pts err {err_kn:.1e}; default growth first step {dkd[1]*1e3:.0f} mm", flush=True)
    print(f"truss: P_peak {Ppk:.1f} N at {dpk*1e3:.2f} mm; load control past peak {'fails' if failed else 'converges'}; "
          f"disp-control err {err_dc:.1e}; arc-length err {err_al:.1e}, delta up to {dal.max()*1e3:.1f} mm", flush=True)
    d = np.linspace(0, 2 * H0, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(d * 1e3, P_cf(d) / 1e3, "k", lw=1, label="hand solution")
    ax.plot(dd * 1e3, Pdc / 1e3, "o", ms=3, mfc="none", label="displacement control")
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
    # hand solution: the loop is a polygon whose corners follow from E, sigma_y, H alone (no return map)
    ET, e_y = E * H / (E + H), sy / E
    e_rev = e_peak - 2 * peak / E                       # strain at which the reversed branch yields (at -peak)
    up, dn = path[:n1 + 1], path[n1 + 1:]
    s_ref = np.concatenate([np.where(up <= e_y, E * up, sy + ET * (up - e_y)),
                            np.where(dn >= e_rev, peak - E * (e_peak - dn), -peak + ET * (dn - e_rev))])
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
    ax.plot(path[::40] * 1e3, s_ref[::40] / 1e6, "k.", ms=3, label="hand solution")
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
    num("cnlElasticaAxial", "$10^{%d}$" % int(round(np.log10(E * A * L ** 2 / (E * I)))))   # EA L^2 / EI of the chain
    num("cnlElasticaTipAngle", f"{abs(np.degrees(sol.sol(L)[0])):.0f}")      # tip rotation of the reference, degrees
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



# ------------------------------------------------------------------------------------------ (G) geometry of the running example
def fig_truss_geometry():
    """Geometry of the shallow two-bar truss (a = 1 m, h0 = 0.10 m): reference shape and four apex positions."""
    a, h0 = A_HALF, H0
    dstar = h0 * (3 - np.sqrt(3)) / 3
    states = [(0.0, "k", "-", 1.8, "reference, $\\delta=0$"),
              (dstar, "tab:red", "--", 1.0, "limit load, $\\delta=%.2f\\,h_0$" % (dstar / h0)),
              (h0, "tab:blue", "--", 1.0, "bars flat, $P=0$, $\\delta=h_0$"),
              (2 * h0, "tab:green", "--", 1.0, "mirror position, $P=0$, $\\delta=2h_0$")]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.grid(False)
    for d, col, ls, lw, lab in states:
        ax.plot([-a, 0, a], [0, h0 - d, 0], color=col, ls=ls, lw=lw, label=lab)
        ax.plot([0], [h0 - d], "o", color=col, ms=3.5)
    for x in (-a, a):                                           # pinned supports
        ax.plot([x, x - 0.035, x + 0.035, x], [0, -0.028, -0.028, 0], "k", lw=1, solid_joinstyle="miter")
        ax.plot([x - 0.06, x + 0.06], [-0.028, -0.028], "k", lw=1)
    ext = dict(color="0.45", lw=0.6, ls=":")
    ax.annotate("", (0, h0 + 0.005), xytext=(0, h0 + 0.075), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax.text(0.025, h0 + 0.045, "load $P$", ha="left", va="center", fontsize=8.5)
    # half-span a: dimension line below everything, with extension lines to the support and to the apex axis
    ya = -0.14
    ax.plot([-a, -a], [-0.03, ya - 0.01], **ext); ax.plot([0, 0], [-h0 - 0.012, ya - 0.01], **ext)
    ax.annotate("", (-a, ya), xytext=(0, ya), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-a / 2, ya - 0.008, "half-span $a$", ha="center", va="top", fontsize=8)
    # apex height h0: dimension line at the right, with extension lines from the apex and from the support level
    xh = 0.66
    ax.plot([0, xh + 0.03], [h0, h0], **ext); ax.plot([a * 0.45, xh + 0.03], [0, 0], **ext)
    ax.annotate("", (xh, 0), xytext=(xh, h0), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(xh + 0.025, h0 / 2, "$h_0$", ha="left", va="center", fontsize=8.5)
    ax.text(-0.62, 0.062, "bar of length $L_0$", ha="center", va="bottom", fontsize=8)
    # deflection delta, drawn for the limit-load position
    xd = -0.16
    ax.plot([xd - 0.02, 0], [h0, h0], **ext); ax.plot([xd - 0.02, 0], [h0 - dstar, h0 - dstar], **ext)
    ax.annotate("", (xd, h0 - dstar), xytext=(xd, h0), arrowprops=dict(arrowstyle="<->", lw=0.9, color="tab:red"))
    ax.text(xd - 0.025, h0 - dstar / 2, "$\\delta$", ha="right", va="center", fontsize=9, color="tab:red")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("horizontal position (m)"); ax.set_ylabel("height (m)")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5, frameon=False, title="apex position", title_fontsize=7.5)
    fig.tight_layout(); fig.canvas.draw()
    bb = ax.get_window_extent()
    exag = (bb.height / 0.4) / (bb.width / 2.4)                 # vertical against horizontal scale (data units per pixel)
    num("cnlTrussGeomExag", f"{exag:.1f}")
    num("cnlTrussRise", f"{h0/a:.1f}")
    leg.set_title(f"apex position (vertical scale exaggerated {exag:.1f} times)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch05_trussgeom.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ (H) defaults of the Newton settings
def tab_options():
    """Defaults of the shared Newton settings in the four solution methods, read from the function signatures."""
    import inspect
    names = ["solve_nonlinear_static", "solve_nonlinear_displacement_control", "solve_nonlinear_arc_length", "solve_nonlinear_koiter_newton"]
    def default(n, k):
        g = getattr(nls, n)
        while hasattr(g, "__wrapped__"):
            g = g.__wrapped__
        return inspect.signature(g).parameters[k].default
    def pw(x):
        return "$10^{%d}$" % int(round(np.log10(x)))
    rows = [("residual tolerance (relative to the norm of the applied load; absolute, in force units, for displacement control)", [pw(default(n, "tol")) for n in names]),
            ("maximum iterations per step", [str(default(n, "max_iter")) for n in names])]
    with open(os.path.join(TAB, "ch05_options.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lcccc}\n\\toprule\nsetting & load control & displacement control & arc length & Koiter--Newton \\\\\n\\midrule\n")
        for lab, v in rows:
            fh.write(lab + " & " + " & ".join(v) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("options defaults:", rows, flush=True)


# ------------------------------------------------------------------------------------------ (I) neo-Hookean small-strain limit
def exp_hyper():
    """Tangent of Tet4NeoHookean at zero displacement against the stiffness of the linear tetrahedron (same E, nu)."""
    from fea_engine.material import NeoHookeanMaterial, Material, D_solid3d
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    E, nu = 210e9, 0.3
    K_nh = elmod.Tet4NeoHookean().tangent_stiffness(X, np.zeros(12), NeoHookeanMaterial(E=E, nu=nu))
    K_lin = elmod.Tet4Solid3D().stiffness(X, D_solid3d(Material(E=E, nu=nu)))
    err = np.linalg.norm(K_nh - K_lin) / np.linalg.norm(K_lin)
    num("cnlNHSmallStrain", sci(err))
    print(f"neo-Hookean small-strain limit: |K_nh - K_lin|/|K_lin| = {err:.1e}", flush=True)


# ------------------------------------------------------------------------------------------ (J) arc-length radius
def exp_radius():
    """Arc length on the truss for several radii: follows the curve, fails, or converges far beyond it."""
    rows = {"follows": [], "fails": [], "far": []}
    for dL in (0.01, 0.05, 0.1, 0.2, 0.4):
        f, cd = truss(); f.add_nodal_force([1], 1, -1.0)
        n = int(round(0.7 / dL)) + 5
        try:
            l_, U_ = nls.solve_nonlinear_arc_length(f, (E_, A_), delta_L=dL, n_steps=n, tol=1e-10, max_iter=60)
        except RuntimeError:
            rows["fails"].append(dL); continue
        d_ = -U_[:, cd]
        ok = (d_ <= 2.0 * H0).sum() >= 3 and np.max(np.abs(l_ - P_cf(d_))) / np.abs(l_).max() < 1e-8
        rows["follows" if ok else "far"].append(dL)
    fmt = lambda xs: " and ".join(f"{x:g}" for x in xs) + " m"
    num("cnlALradFollow", fmt(rows["follows"])); num("cnlALradFail", fmt(rows["fails"])); num("cnlALradFar", fmt(rows["far"]))
    print("arc-length radius:", rows, flush=True)

def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_objectivity(); exp_tangent(); exp_vonmises(); exp_plasticity(); exp_drivers(); exp_elastica(); fig_truss_geometry(); tab_options(); exp_hyper(); exp_radius()
    num("cnlMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch05.tex"), "w") as fh:
        fh.write("% generated by scripts/ch05_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

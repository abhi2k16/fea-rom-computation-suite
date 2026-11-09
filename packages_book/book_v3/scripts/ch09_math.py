"""Chapter 9, Section 9.3 figures and worked examples (balanced truncation part):
figures/ch09_math_bt.pdf (M1) and tables/numbers_ch09_math.tex.
E1: one mass-spring-damper (m=1, k=1, c=0.2), force in, displacement out: Gramians in closed form, Hankel singular values,
one-state balanced truncation against the bound 2*sigma_2. M1 uses the same example for the ellipses and the beam for the singular values."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov, eigh, cholesky, svd, eig
from rom_engine.krylov import KrylovROM, arnoldi_basis
from rom_engine.state_space import to_state_space
from rom_engine.balanced_truncation import BalancedTruncationROM
from rom_engine.soar import soar_basis
import ch09_assets as A
import ch09_intro as I

BLUE, VIOLET, GREY, ORANGE = I.BLUE, I.VIOLET, I.GREY, I.ORANGE
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7})
NUM = {}
KS, CS = 1.0, 0.2                                  # stiffness (N/m) and damping (N s/m) of the oscillator, mass 1 kg


def e1_case(c, tag):
    k = KS
    Am = np.array([[0, 1], [-k, -c]]); Bm = np.array([[0.0], [1.0]]); Cm = np.array([[1.0, 0.0]])
    P = solve_continuous_lyapunov(Am, -Bm @ Bm.T); Q = solve_continuous_lyapunov(Am.T, -Cm.T @ Cm)
    Pc = np.diag([1 / (2 * c * k), 1 / (2 * c)])                      # closed forms printed in the text
    Qc = np.array([[(c ** 2 + k) / (2 * c * k), 1 / (2 * k)], [1 / (2 * k), 1 / (2 * c * k)]])
    assert np.allclose(P, Pc) and np.allclose(Q, Qc)
    s2 = np.array([((c ** 2 + 2 * k) + s * c * np.sqrt(c ** 2 + 4 * k)) / (8 * c ** 2 * k ** 2) for s in (1, -1)])
    sig = np.sqrt(s2)
    assert np.allclose(np.sort(np.sqrt(np.linalg.eigvals(P @ Q).real))[::-1], sig)
    S = cholesky(P, lower=True); R = cholesky(Q, lower=True)           # square-root balancing
    U, sv, Vt = svd(R.T @ S); T = S @ Vt.T @ np.diag(sv ** -0.5); W = R @ U @ np.diag(sv ** -0.5)
    Ab, Bb, Cb = W.T @ Am @ T, W.T @ Bm, Cm @ T
    om = np.concatenate([np.linspace(0, 4, 4001), np.logspace(np.log10(4), 3, 600)])
    H = np.array([(Cm @ np.linalg.solve(1j * x * np.eye(2) - Am, Bm))[0, 0] for x in om])
    H1 = np.array([Cb[0, 0] * Bb[0, 0] / (1j * x - Ab[0, 0]) for x in om])
    err = np.max(np.abs(H - H1)); bound = 2 * sig[1]; peak = np.max(np.abs(H))
    NUM.update({f"cbtEx{tag}C": f"{c:g}", f"cbtEx{tag}Zeta": f"{c / (2 * np.sqrt(k)):.2g}",
                f"cbtEx{tag}SigOne": f"{sig[0]:.3f}", f"cbtEx{tag}SigTwo": f"{sig[1]:.3f}",
                f"cbtEx{tag}SigRatio": f"{sig[0] / sig[1]:.2f}", f"cbtEx{tag}Err": f"{err:.3f}", f"cbtEx{tag}Bound": f"{bound:.3f}",
                f"cbtEx{tag}Peak": f"{peak:.2f}", f"cbtEx{tag}ErrRel": f"{100 * err / peak:.0f}", f"cbtEx{tag}BoundRel": f"{100 * bound / peak:.0f}",
                f"cbtEx{tag}POne": f"{P[0, 0]:.3f}", f"cbtEx{tag}QOne": f"{Q[0, 0]:.3f}", f"cbtEx{tag}QTwo": f"{Q[1, 1]:.3f}",
                f"cbtEx{tag}QOff": f"{Q[0, 1]:.3f}", f"cbtEx{tag}Ab": f"{Ab[0, 0]:.3f}"})
    return dict(P=P, Q=Q, T=T, sig=sig, Ab=Ab, Bb=Bb, Cb=Cb)


def e1():
    NUM.update(cbtExK="1", cbtExMass="1")
    heavy = e1_case(3.0, "Heavy"); light = e1_case(0.2, "Light")
    return heavy


def ell(M, n=200):
    """points x with x^T M x = 1"""
    t = np.linspace(0, 2 * np.pi, n); L = np.linalg.cholesky(M)
    return np.linalg.solve(L.T, np.vstack([np.cos(t), np.sin(t)]))


def fig_bt(d, e):
    P, Q, T, sig = e["P"], e["Q"], e["T"], e["sig"]
    fig = plt.figure(figsize=(9.4, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.5)
    # (a) original coordinates
    a = fig.add_subplot(gs[0])
    xr = ell(np.linalg.inv(P)); xo = ell(Q)
    a.plot(*xr, color=BLUE, lw=1.6, label="reachable with unit input energy"); a.plot(*xo, color=ORANGE, lw=1.6, label="unit output energy")
    a.set_aspect("equal"); a.axhline(0, color="0.8", lw=0.5); a.axvline(0, color="0.8", lw=0.5)
    a.set_xlabel("displacement $q$ (m)"); a.set_ylabel("velocity $\\dot q$ (m/s)"); a.set_title("(a) physical states", loc="left")
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1.6, label="reachable with unit input energy"),
                        plt.Line2D([], [], color=ORANGE, lw=1.6, label="states giving unit output energy")],
               loc="lower center", bbox_to_anchor=(0.30, -0.13), ncol=2, frameon=False, fontsize=7)
    # (b) balanced coordinates (z = T^{-1} x)
    b = fig.add_subplot(gs[1])
    Ti = np.linalg.inv(T)
    # x = T z: ellipse x^T P^-1 x = 1  ->  z^T (T^T P^-1 T) z = 1
    Pz = T.T @ np.linalg.inv(P) @ T; Qz = T.T @ Q @ T
    assert np.allclose(Pz, np.diag(np.diag(Pz)), atol=1e-9) and np.allclose(Qz, np.diag(np.diag(Qz)), atol=1e-9)
    NUM["cbtExHeavyReach"] = f"{1 / np.sqrt(np.diag(Pz)).min():.2f}"
    zr = ell(Pz); zo = ell(Qz)
    b.plot(*zr, color=BLUE, lw=1.6); b.plot(*zo, color=ORANGE, lw=1.6)
    b.set_aspect("equal"); b.axhline(0, color="0.8", lw=0.5); b.axvline(0, color="0.8", lw=0.5)
    sg = np.sqrt(np.diag(np.linalg.inv(Pz)))
    b.annotate("$\\sqrt{\\sigma_1}$", (sg[0], 0), (sg[0] + 0.5, 1.2), fontsize=7, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    b.annotate("$1/\\sqrt{\\sigma_2}$", (0, 1 / np.sqrt(np.diag(Qz))[1]), (0.8, 3.9), fontsize=7, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
    b.set_xlim(-2.3, 2.3); b.set_xlabel("balanced state $z_1$"); b.set_ylabel("balanced state $z_2$"); b.set_title("(b) balanced states", loc="left")
    # (c) Hankel singular values of the beam with bound
    c = fig.add_subplot(gs[2])
    r = 8
    h = I.pre_reduced(d, r).hsv
    kk = np.arange(1, len(h) + 1)
    c.semilogy(kk[:r], h[:r], "o", color=BLUE, ms=4, label=f"kept ($r={r}$)")
    c.semilogy(kk[r:], h[r:], "o", color=ORANGE, ms=4, label="dropped")
    c.axvline(r + 0.5, color=GREY, lw=0.8, ls="--")
    bound = 2 * np.sum(h[r:]); peak = np.max(np.abs(d["Hf"]))
    c.set_xlabel("index $i$"); c.set_ylabel("Hankel singular value $\\sigma_i$ (m/N)")
    c.set_title("(c) the beam: where to cut", loc="left"); c.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(A.FIG, "ch09_math_bt.pdf"), bbox_inches="tight"); plt.close(fig)
    NUM["cbtMathBtR"] = str(r); NUM["cbtMathBtRp"] = str(r + 1); NUM["cbtMathBtNH"] = str(len(h))
    NUM["cbtMathBtBound"] = A.sci(bound); NUM["cbtMathBtBoundRel"] = A.sci(bound / peak)
    NUM["cbtMathBtSigOne"] = A.sci(h[0]); NUM["cbtMathBtSigR"] = A.sci(h[r - 1]); NUM["cbtMathBtSigRp"] = A.sci(h[r])
    NUM["cbtMathBtDrop"] = A.sci(h[r - 1] / h[r])
    # axes extents for the caption of (a)
    print("sigma", sig, "hsv[:10]", h[:10])


# ---------------------------------------------------------------------------------------- E2 and M2 (Krylov)
def e2():
    """One mass-spring-damper again (m=1, k=1): moments about s0=0, and the one-state two-sided Krylov model."""
    k = KS
    for c, tag in ((3.0, "Heavy"), (0.2, "Light")):
        r = KrylovROM.from_MCK(np.array([[1.0]]), np.array([[k]]), np.array([1.0]), np.array([1.0]), C=np.array([[c]]), k=1, two_sided=True)
        assert r.A_r.shape == (1, 1)
        om = np.concatenate([np.linspace(0, 4, 4001), np.logspace(np.log10(4), 3, 600)])
        H = 1 / (k - om ** 2 + 1j * om * c); Hr = r.frequency_response(om)
        assert np.allclose(Hr, 1 / (k + 1j * om * c), atol=1e-12)              # H_r(s) = 1/(k + c s) exactly
        err = np.max(np.abs(H - Hr)); peak = np.max(np.abs(H))
        NUM[f"cbtExKtwo{tag}Err"] = f"{err:.3f}"; NUM[f"cbtExKtwo{tag}ErrRel"] = f"{100 * err / peak:.0f}"
        if tag == "Heavy":
            m = [1 / k, -c / k ** 2, (c ** 2 - k) / k ** 3, -c * (c ** 2 - 2 * k) / k ** 4]
            for j, mj in enumerate(m): NUM[f"cbtExMom{'ABCD'[j]}"] = f"${mj:g}$"
            NUM["cbtExMomRedC"] = f"{c ** 2 / k ** 3:g}"


def fit_slope(x, e, lo=3e-10, hi=1e-2):
    m = (e > lo) & (e < hi)
    return np.polyfit(np.log10(x[m]), np.log10(e[m]), 1)[0], int(m.sum())


def fig_krylov(d):
    M, K, C, B, w1 = d["M"], d["K"], d["C"], d["B"], d["w"][0]
    m0 = np.mean(np.diag(M)); Ms, Ks, Cs = M / m0, K / (m0 * w1 ** 2), C / (m0 * w1)      # nondimensional: omega in units of omega_1
    x = np.logspace(-1.7, -0.2, 40)
    Hf = np.array([B @ np.linalg.solve(-o ** 2 * Ms + 1j * o * Cs + Ks, B) for o in x])
    fig = plt.figure(figsize=(9.4, 3.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    cols = {1: "#6baed6", 2: BLUE, 3: VIOLET}; slopes = {}
    for kk in (1, 2, 3):
        r = KrylovROM.from_MCK(Ms, Ks, B, B, C=Cs, s0=0.0, k=kk, two_sided=True)
        e = np.abs(r.frequency_response(x) - Hf) / np.abs(Hf)
        sl, npt = fit_slope(x, e); slopes[kk] = sl
        a.loglog(x, e, "o-", color=cols[kk], ms=2.5, lw=1.1, label=f"$k={kk}$" + (" state" if kk == 1 else " states"))
        NUM[f"cbtKrySlope{['', 'One', 'Two', 'Three'][kk]}"] = f"{sl:.1f}"
        ref = e[np.argmin(np.abs(x - 0.3))]
        xx = np.array([x[0], x[-1]]); a.loglog(xx, ref * (xx / 0.3) ** (2 * kk), ":", color="0.45", lw=0.8)
    a.axhline(1e-10, color="0.8", lw=0.6)
    a.text(x[0] * 1.05, 2.5e-10, "rounding level", fontsize=6.5, color="0.5")
    a.set_ylim(1e-11, 1e-1); a.set_xlabel("$\\omega/\\omega_1$"); a.set_ylabel("relative error $|H_r-H|/|H|$")
    a.set_title("(a) error of the two-sided model near $s_0=0$", loc="left"); a.legend(frameon=False, loc="lower right")
    a.set_xlim(0.019, 1.0)
    for kk, (xt, yt) in ((1, (0.35, 3.4e-2)), (2, (0.40, 1.6e-3)), (3, (0.42, 5e-5))):
        a.text(xt, yt, f"slope {2 * kk}", fontsize=6.5, color="0.3", ha="left")
    ks = np.array([1, 2, 3]); b.plot(ks, [slopes[k] for k in ks], "o", color=VIOLET, label="measured")
    b.plot([0.5, 3.5], [1, 7], "--", color=GREY, lw=1, label="$2k$ moments")
    b.set_xticks(ks); b.set_xlim(0.5, 3.5); b.set_ylim(0, 7.5); b.set_xlabel("states $k$ (one vector per side)"); b.set_ylabel("slope of the error (fitted)")
    b.set_title("(b) order of agreement", loc="left"); b.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(A.FIG, "ch09_math_krylov.pdf"), bbox_inches="tight"); plt.close(fig)
    NUM["cbtKryFitFloor"] = "3\\times10^{-10}"
    print("two-sided slopes", slopes)


# ---------------------------------------------------------------------------------------------- M3 (SOAR)
def asinh_img(X, lo=1e-4):
    X = np.asarray(X, float); X = X / np.max(np.abs(X))
    return np.arcsinh(X / lo) / np.arcsinh(1 / lo)


def fig_soar(d):
    M, K, C, B, w1 = d["M"], d["K"], d["C"], d["B"], d["w"][0]
    ks = 4                                                            # SOAR unknowns; the first-order model has 2*ks states
    r1 = KrylovROM.from_MCK(M, K, B, B, C=C, s0=0.0, k=2 * ks)         # one-sided first-order Galerkin at equal state count
    Ar, Er = r1.A_r.real, r1.E_r.real
    m0 = np.mean(np.diag(M)); Ms, Ks, Cs = M / m0, K / (m0 * w1 ** 2), C / (m0 * w1)
    V = soar_basis(Ms, Ks, B / np.linalg.norm(np.linalg.solve(Ks, B)), ks, C=Cs)
    assert V.shape[1] == ks
    Mr, Cr, Kr = V.T @ M @ V, V.T @ C @ V, V.T @ K @ V
    asym = lambda X: np.linalg.norm(X - X.T) / np.linalg.norm(X)
    NUM.update(cbtSoarUnk=str(ks), cbtSoarStates=str(2 * ks), cbtSoarAsymA=A.sci(asym(Ar)), cbtSoarAsymE=A.sci(asym(Er)),
               cbtSoarAsymM=A.sci(max(asym(Mr), 1e-17)), cbtSoarAsymK=A.sci(max(asym(Kr), 1e-17)))
    NUM["cbtSoarEigM"] = "positive"; assert np.linalg.eigvalsh(Mr).min() > 0 and np.linalg.eigvalsh(Kr).min() > 0 and np.linalg.eigvalsh(Cr).min() >= -1e-12 * np.linalg.eigvalsh(Cr).max()
    NUM["cbtSoarCondK"] = A.sci(np.linalg.cond(Kr)); NUM["cbtSoarMinEigCRatio"] = A.sci(np.linalg.eigvalsh(Cr).min() / np.linalg.eigvalsh(Cr).max())
    # how much of the first-order reduced pencil is the "mass-like" and "stiffness-like" block structure: none
    fig = plt.figure(figsize=(9.4, 2.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[2, 2, 1, 1, 1], wspace=0.35)
    sy = lambda X: f"asymmetry {asym(X):.1e}" if asym(X) > 1e-9 else "symmetric"
    panels = [(Ar, f"$\\mathbf{{A}}_r$ ({2 * ks}$\\times${2 * ks})", sy(Ar)), (Er, f"$\\mathbf{{E}}_r$ ({2 * ks}$\\times${2 * ks})", sy(Er)),
              (Mr, f"$\\mathbf{{M}}_r$ ({ks}$\\times${ks})", sy(Mr)), (Cr, f"$\\mathbf{{C}}_r$ ({ks}$\\times${ks})", sy(Cr)),
              (Kr, f"$\\mathbf{{K}}_r$ ({ks}$\\times${ks})", sy(Kr))]
    for i, (X, ttl, sub) in enumerate(panels):
        a = fig.add_subplot(gs[i]); a.imshow(asinh_img(X), cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=8.5); a.set_xlabel(sub, fontsize=7)
        for sp in a.spines.values(): sp.set_color(BLUE if i < 2 else VIOLET); sp.set_linewidth(1.6)
    fig.text(0.255, 1.02, f"(a) first-order Galerkin, {2 * ks} states", ha="center", fontsize=8.5, color=BLUE)
    fig.text(0.78, 1.02, f"(b) SOAR, {ks} unknowns = {2 * ks} states", ha="center", fontsize=8.5, color=VIOLET)
    fig.savefig(os.path.join(A.FIG, "ch09_math_soar.pdf"), bbox_inches="tight"); plt.close(fig)
    print({k: v for k, v in NUM.items() if k.startswith("cbtSoar")})


# ------------------------------------------------------------------------------ audit Q5, Q6 (numbers for Sec. 9.4)
def q5_q6(d):
    M, K, C, B, w1, om, Hf = d["M"], d["K"], d["C"], d["B"], d["w"][0], d["om"], d["Hf"]
    x = om / w1; sub = x <= 1.0
    ss = to_state_space(M, K, C=C, B=B, Cout=B, form="E")
    for r, nm in ((8, "Eight"), (12, "Twelve"), (16, "Sixteen")):
        for two, tag in ((False, "One"), (True, "Two")):
            m = KrylovROM.from_MCK(M, K, B, B, C=C, s0=0.0, k=r, two_sided=two)
            e = np.abs(m.frequency_response(om) - Hf) / np.abs(Hf)
            NUM[f"cbtKrySub{tag}{nm}"] = A.sci(e[sub].mean())
        V = arnoldi_basis(ss.A, ss.E, ss.B, r, s0=0.0); Wt = arnoldi_basis(ss.A.T, ss.E.T, ss.Cout.T, r, s0=0.0)
        NUM[f"cbtKryCond{nm}"] = A.sci(np.linalg.cond(Wt.T @ ss.E @ V))
    NUM["cbtKrySubUpper"] = "1"                                                       # sub-band: omega <= omega_1, the radius of convergence about s_0 = 0
    # balanced truncation: stability at every order, and the gap sigma_r / sigma_{r+1}
    Vp = d["Phi"][:, :A.N_PRE]; Mp, Kp, Cp, Bp = Vp.T @ M @ Vp, Vp.T @ K @ Vp, Vp.T @ C @ Vp, Vp.T @ B
    orders = range(1, 2 * A.N_PRE - 1); stable = 0; ratios = []
    for r in orders:
        rom = BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=r)
        stable += int(rom.is_stable()); h = rom.hsv
        if h[r] > 1e-13 * h[0]: ratios.append((h[r - 1] / h[r], r))
    NUM["cbtBtStableAll"] = str(stable); NUM["cbtBtOrdersAll"] = str(len(list(orders)))
    rmin, rat = min(ratios); NUM["cbtBtPairRatioMin"] = f"{rmin:.3f}"; NUM["cbtBtPairRatioAt"] = str(rat)
    print({k: v for k, v in NUM.items() if k.startswith(("cbtKrySub", "cbtKryCond", "cbtBtStableAll", "cbtBtOrdersAll", "cbtBtPair"))})


# ------------------------------------------------------------------------------ audit Q7: moments matched and real poles
def fmt_pole(p):
    v = float(f"{p:.3g}")
    return f"{v:.0f}" if abs(v) >= 1 else f"{v:g}"


def q7(d):
    """Moments matched and real poles of the Krylov models: table in SI units (the units of the chapter); the one-sided odd-order pole is also
    reported in nondimensional units (omega in units of omega_1, M divided by its mean diagonal), where it moves to the origin."""
    M, K, C, B, w1 = d["M"], d["K"], d["C"], d["B"], d["w"][0]

    def moments(Am, Em, Bm, Cm, nm):
        Fm = -Am; X = np.linalg.solve(Fm, Bm); out = []
        for j in range(nm):
            out.append(((-1) ** j) * (Cm @ X).item()); X = np.linalg.solve(Fm, Em @ X)
        return np.array(out)

    def real_poles(r):
        pol = eig(r.A_r, r.E_r, right=False); pol = pol[np.isfinite(pol)]
        return sorted(p.real / w1 for p in pol if abs(p.imag) < 1e-9 * max(1.0, abs(p)))

    rows = []
    for k in (1, 2, 3, 4):
        cell = []
        for two in (False, True):
            try:
                r = KrylovROM.from_MCK(M, K, B, B, C=C, s0=0.0, k=k, two_sided=two); ss = r.ss
                mf = moments(ss.A, ss.E, ss.B, ss.Cout, 2 * k + 3); mr = moments(r.A_r, r.E_r, r.B_r, r.Cout_r, 2 * k + 3)
            except np.linalg.LinAlgError:
                cell.append(("singular ($\\bm B_r=\\bm 0$)", "--")); continue
            rel = np.abs(mr - mf) / np.abs(mf); cnt = 0
            for v in rel:
                if v < 1e-6: cnt += 1
                else: break
            real = real_poles(r)
            cell.append((str(cnt), "none" if not real else ", ".join("$" + fmt_pole(p) + "$" for p in real)))
            if k == 3: NUM["cbtMomPole" + ("Two" if two else "One") + "Three"] = f"${fmt_pole(real[0])}\\,\\omega_1$"
        rows.append((k, cell))
    r5 = KrylovROM.from_MCK(M, K, B, B, C=C, s0=0.0, k=5)
    NUM["cbtMomPoleOneFive"] = f"${fmt_pole(real_poles(r5)[0])}\\,\\omega_1$"
    m0 = np.mean(np.diag(M)); Ms, Ks, Cs = M / m0, K / (m0 * w1 ** 2), C / (m0 * w1)
    rnd = KrylovROM.from_MCK(Ms, Ks, B, B, C=Cs, s0=0.0, k=3)
    NUM["cbtMomPoleOneThreeND"] = f"${fmt_pole(real_poles(rnd)[0] * w1)}\\,\\omega_1$"   # nondimensional units: poles already in units of omega_1
    lines = ["% generated by scripts/ch09_math.py -- do not edit", "\\begin{tabular}{rllll}", "\\toprule",
             " & \\multicolumn{2}{c}{one-sided} & \\multicolumn{2}{c}{two-sided} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "states $k$ & moments matched & real pole & moments matched & real pole \\\\", "\\midrule"]
    for k, cell in rows:
        lines.append(f"{k} & {cell[0][0]} & {cell[0][1]} & {cell[1][0]} & {cell[1][1]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(A.TAB, "ch09_moments.tex"), "w").write("\n".join(lines) + "\n")
    print(open(os.path.join(A.TAB, "ch09_moments.tex")).read(), {k: v for k, v in NUM.items() if k.startswith("cbtMomPole")})


# ------------------------------------------------------------------------------ audit Q10, Q11: numbers that replace vague words
def q10_11(d):
    import time
    from scipy.linalg import solve_continuous_lyapunov as lyap
    from fea_engine.material import EI_beam, Material, Section
    from fea_engine.geometry import generate_mesh
    from fea_engine.solver import FESystem
    from fea_engine.damping import RayleighDamping
    from fea_engine import elements
    Hf, om = d["Hf"], d["om"]
    NUM["cbtMathHPeakMin"] = A.sci(np.abs(Hf).max() / np.abs(Hf).min())
    def err(H):
        e = np.abs(H - Hf); return e.max() / np.abs(Hf).max(), np.mean(e / np.abs(Hf))
    def modal(r):
        V = d["Phi"][:, : r // 2]
        return A.h_second_order(V.T @ d["M"] @ V, V.T @ d["C"] @ V, V.T @ d["K"] @ V, V.T @ d["B"], V.T @ d["B"], om)
    gain = lambda r: (err(modal(r)), err(I.pre_reduced(d, r).frequency_response(om)))
    (m8, b8), (m12, b12) = gain(8), gain(12)
    NUM["cbtMathMeanGainEight"] = f"{m8[1] / b8[1]:.1f}"; NUM["cbtMathGainTwelve"] = f"{m12[0] / b12[0]:.1f}"
    for nel, nm in ((24, "NinetySix"), (96, "ThreeEightyFour"), (192, "SevenSixtyEight")):
        mesh = generate_mesh(dim=1, L=1.0, n=nel); fes = FESystem(mesh, elements.Beam2DEulerBernoulli())
        mat, sec = Material(E=210e9, nu=0.3, rho=7800.0), Section(A=0.01, I=8.33e-6)
        fes.assemble_stiffness(EI_beam(mat, sec)); fes.assemble_mass(7800.0 * 0.01)
        fes.assemble_damping(RayleighDamping(alpha=2.0, beta=1e-5)); fes.fix_dofs([0], [0, 1])
        free = np.asarray(fes.free_dofs); ix = np.ix_(free, free); Kk, Mm, Cc = fes.K[ix], fes.M[ix], fes.C[ix]
        n = len(free); Ai = np.block([[np.zeros((n, n)), np.eye(n)], [-np.linalg.solve(Mm, Kk), -np.linalg.solve(Mm, Cc)]])
        b = np.zeros(n); b[n - 2] = 1; Bi = np.concatenate([np.zeros(n), np.linalg.solve(Mm, b)])[:, None]
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); lyap(Ai, -Bi @ Bi.T); ts.append(time.perf_counter() - t0)
        assert 2 * n == {"NinetySix": 96, "ThreeEightyFour": 384, "SevenSixtyEight": 768}[nm]
        NUM["cbtMathLyap" + nm] = f"{min(ts):.2g}"
    print({k: v for k, v in NUM.items() if k.startswith(("cbtMathLyap", "cbtMathHPeak", "cbtMathMeanGain", "cbtMathGainTwelve"))})



if __name__ == "__main__":
    e = e1(); d = I.setup(); fig_bt(d, e); e2(); fig_krylov(d); fig_soar(d); q5_q6(d); q7(d); q10_11(d)
    for k_, v in NUM.items(): print(k_, v)
    with open(os.path.join(A.TAB, "numbers_ch09_math.tex"), "w") as fh:
        fh.write("% generated by scripts/ch09_math.py -- do not edit\n")
        for k_, v in NUM.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")

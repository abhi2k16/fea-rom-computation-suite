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
from scipy.linalg import solve_continuous_lyapunov, eigh, cholesky, svd
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
    c.set_xlabel("index $k$"); c.set_ylabel("Hankel singular value $\\sigma_k$ (m/N)")
    c.set_title("(c) the beam: where to cut", loc="left"); c.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(A.FIG, "ch09_math_bt.pdf"), bbox_inches="tight"); plt.close(fig)
    NUM["cbtMathBtR"] = str(r); NUM["cbtMathBtNH"] = str(len(h))
    NUM["cbtMathBtBound"] = A.sci(bound); NUM["cbtMathBtBoundRel"] = A.sci(bound / peak)
    NUM["cbtMathBtSigOne"] = A.sci(h[0]); NUM["cbtMathBtSigR"] = A.sci(h[r - 1]); NUM["cbtMathBtSigRp"] = A.sci(h[r])
    NUM["cbtMathBtDrop"] = A.sci(h[r - 1] / h[r])
    # axes extents for the caption of (a)
    print("sigma", sig, "hsv[:10]", h[:10])


if __name__ == "__main__":
    e = e1(); d = I.setup(); fig_bt(d, e)
    for k_, v in NUM.items(): print(k_, v)
    with open(os.path.join(A.TAB, "numbers_ch09_math.tex"), "w") as fh:
        fh.write("% generated by scripts/ch09_math.py -- do not edit\n")
        for k_, v in NUM.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")

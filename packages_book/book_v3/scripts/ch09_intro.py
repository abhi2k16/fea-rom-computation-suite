"""Chapter 9 introduction figures (Section 9.1): ch09_intro_port.pdf, ch09_intro_modes.pdf and tables/numbers_ch09_intro.tex.

Everything is computed on the beam of the chapter: the modal expansion of H, the Hankel singular values from the
Lyapunov equations (BalancedTruncationROM on the pre-reduced model) and their modal estimate R_k / (4 z_k w_k^2)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.linalg import eigh
import ch09_assets as A
from rom_engine.balanced_truncation import BalancedTruncationROM

BLUE, VIOLET, GREY, ORANGE, GREEN = "#08519c", "#54278f", "#636363", "#d95f0e", "#2e7d32"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7})
NUM = {}


def sci(x):
    return A.sci(x)


def setup():
    fes, free, K, M, C, B, tip = A.build()
    lam, Phi = eigh(K, M)                       # Phi^T M Phi = I
    w = np.sqrt(lam); z = A.ALPHA / (2 * w) + A.BETA * w / 2
    om = np.linspace(A.BAND_LO, A.BAND_HI, A.N_FREQ) * w[0]
    Hf = np.array([B @ np.linalg.solve(-x ** 2 * M + 1j * x * C + K, B) for x in om])
    return dict(K=K, M=M, C=C, B=B, Phi=Phi, w=w, z=z, om=om, Hf=Hf, n=len(B))


def pre_reduced(d, r):
    Vp = d["Phi"][:, :A.N_PRE]
    Mp, Kp, Cp, Bp = Vp.T @ d["M"] @ Vp, Vp.T @ d["K"] @ Vp, Vp.T @ d["C"] @ Vp, Vp.T @ d["B"]
    return BalancedTruncationROM.from_MCK(Mp, Kp, Bp, Bp, C=Cp, r=r)


def box(ax, xy, wh, text, fc, ec, fs=8, bold=False):
    p = FancyBboxPatch(xy, wh[0], wh[1], boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color="k", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.3, color=color, ls=ls))


# --------------------------------------------------------------------------------------------- I1
def fig_port(d):
    n, w, om, Hf = d["n"], d["w"], d["om"], d["Hf"]
    r = 8
    bt = pre_reduced(d, r)
    Hr = bt.frequency_response(om)
    rel = np.abs(Hr - Hf) / np.abs(Hf)
    x = om / w[0]
    fig = plt.figure(figsize=(8.8, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.3], width_ratios=[1.75, 1], hspace=0.38, wspace=0.28)
    # (a) block diagram
    a = fig.add_subplot(gs[0, :]); a.set_xlim(0, 100); a.set_ylim(0, 24); a.axis("off")
    a.set_title("(a) the rest of the system sees only a port: a force $u$ goes in, a tip displacement $y$ comes out", loc="left")
    for y0, col, ec, txt in ((13.2, "#deebf7", BLUE, f"full-order model\n{n} unknowns, {2 * n} states"),
                             (1.8, "#efe5f6", VIOLET, f"reduced-order model\n{r} states")):
        box(a, (36, y0), (28, 7.2), txt, col, ec, bold=True)
        arrow(a, (14, y0 + 3.6), (35.2, y0 + 3.6))
        arrow(a, (64.8, y0 + 3.6), (86, y0 + 3.6))
        a.text(24.5, y0 + 5.4, "force $u(t)$", ha="center", fontsize=8)
        a.text(75.5, y0 + 5.4, "tip displacement $y(t)$", ha="center", fontsize=8)
    box(a, (1, 5.5), (12.5, 8), "the rest of\nthe system\n(machine)", "#f0f0f0", GREY)
    box(a, (86.8, 5.5), (12.5, 8), "reads\n$y=H(i\\omega)\\,u$", "#f0f0f0", GREY)
    a.annotate("", xy=(50, 9.4), xytext=(50, 12.6), arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    a.text(51.5, 11.0, "same port, same $H$ to the accuracy of panel (b)", fontsize=7.5, color=GREY, va="center", ha="left")
    # (b) the frequency response and the error
    sub = gs[1, 0].subgridspec(2, 1, height_ratios=[1.6, 1], hspace=0.12)
    b1 = fig.add_subplot(sub[0]); b2 = fig.add_subplot(sub[1], sharex=b1)
    b1.semilogy(x, np.abs(Hf), "k-", lw=2.2, label=f"full order ({2 * n} states)")
    b1.semilogy(x, np.abs(Hr), "--", color=VIOLET, lw=1.5, label=f"balanced truncation, {r} states")
    for k in range(2):
        xk = w[k] / w[0]
        j = np.argmin(abs(x - xk))
        b1.annotate(f"resonance {k + 1}", (xk, np.abs(Hf[j])), (xk + 0.5, np.abs(Hf[j]) * (0.2 if k == 0 else 12)), fontsize=7, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    b1.set_ylabel("$|H(i\\omega)|$ (m/N)"); b1.legend(loc="lower left", frameon=False); plt.setp(b1.get_xticklabels(), visible=False)
    b1.set_title("(b) what the machine reads: the frequency response of the tip port", loc="left")
    b2.semilogy(x, rel, color=VIOLET, lw=1.2); b2.set_ylabel("relative error"); b2.set_xlabel("$\\omega/\\omega_1$")
    # (c) sizes to scale
    c = fig.add_subplot(gs[1, 1])
    c.barh([1, 0], [2 * n, r], color=["#9ecae1", "#bcbddc"], edgecolor=[BLUE, VIOLET], height=0.55)
    c.set_yticks([1, 0]); c.set_yticklabels(["full order", "reduced"]); c.set_xlim(0, 2 * n * 1.12); c.set_ylim(-0.6, 1.6)
    c.text(2 * n + 1, 1, f"{2 * n}", va="center", fontsize=8); c.text(r + 1, 0, f"{r}", va="center", fontsize=8)
    c.set_xlabel("states"); c.set_title("(c) size, drawn to scale", loc="left")
    c.text(2 * n * 0.5, -0.45, f"{2 * n // r}$\\times$ fewer states", ha="center", fontsize=8, color=VIOLET)
    fig.savefig(os.path.join(A.FIG, "ch09_intro_port.pdf"), bbox_inches="tight"); plt.close(fig)
    NUM["cbtIntroStatesFull"] = str(2 * n); NUM["cbtIntroStatesRed"] = str(r); NUM["cbtIntroStatesRatio"] = str(2 * n // r)
    NUM["cbtIntroErrPeak"] = sci(np.max(np.abs(Hr - Hf)) / np.max(np.abs(Hf)))


# --------------------------------------------------------------------------------------------- I2
def fig_modes(d):
    w, z, Phi, B, om, Hf, M, C, K = d["w"], d["z"], d["Phi"], d["B"], d["om"], d["Hf"], d["M"], d["C"], d["K"]
    R = (B @ Phi) ** 2                                              # residues R_k = (c^T phi_k)(phi_k^T b), collocated port
    Hm = np.array([np.sum(R / (w ** 2 - x ** 2 + 2j * z * w * x)) for x in om])
    NUM["cbtIntroModalErr"] = sci(np.max(np.abs(Hm - Hf)) / np.max(np.abs(Hf)))
    npre = A.N_PRE
    peak = R / (2 * z * w ** 2)
    est = R[:npre] / (4 * z[:npre] * w[:npre] ** 2)
    h = pre_reduced(d, 2 * npre).hsv
    pair = np.array([[h[2 * k], h[2 * k + 1]] for k in range(npre)])
    ratio = pair[:, 0] / est
    ok = z[:npre] <= 0.15
    NUM["cbtIntroZetaMax"] = f"{z[:npre][ok].max():.2f}"
    NUM["cbtIntroEstWithinPct"] = f"{100 * np.max(np.abs(ratio[ok] - 1)):.0f}"
    NUM["cbtIntroEstModes"] = str(int(ok.sum()))
    NUM["cbtIntroResidue"] = f"{R[0]:.2f}"
    NUM["cbtIntroResidueSpread"] = f"{R[:npre].max() / R[:npre].min():.2f}"
    NUM["cbtIntroZetaFirst"] = f"{z[0]:.4f}"; NUM["cbtIntroZetaThird"] = f"{z[2]:.3f}"
    nm = 8                                                           # modes drawn (the estimate is meaningless once a mode is overdamped)
    fig = plt.figure(figsize=(9.4, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.25], wspace=0.78)
    # (a) peak heights and damping ratios
    a = fig.add_subplot(gs[0]); k = np.arange(1, nm + 1)
    a.bar(k, peak[:nm], color="#9ecae1", edgecolor=BLUE)
    a.set_yscale("log"); a.set_xlabel("mode $k$"); a.set_ylabel("peak height $R_k/(2\\zeta_k\\omega_k^2)$ (m/N)")
    a.set_title("(a) how high each mode rises", loc="left")
    a2 = a.twinx(); a2.semilogy(k, z[:nm], "D-", color=ORANGE, ms=3.5, lw=1); a2.set_ylabel("damping ratio $\\zeta_k$", color=ORANGE); a2.tick_params(axis="y", colors=ORANGE)
    a.set_xticks(k)
    # (b) Hankel singular values against the estimate
    b = fig.add_subplot(gs[1])
    lo = min(est[:nm].min(), pair[:nm].min()) * 0.4; hi = est.max() * 3
    b.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--", label="equal")
    m1 = ok.copy(); m1[nm:] = False; m2 = ~ok; m2[nm:] = False
    for jj in range(2):
        b.loglog(est[m1], pair[m1, jj], "o", color=BLUE, ms=5, mfc=BLUE if jj == 0 else "white", label="$\\zeta_k\\leq0.15$" if jj == 0 else None)
        b.loglog(est[m2], pair[m2, jj], "s", color=ORANGE, ms=4, mfc=ORANGE if jj == 0 else "white", label="$\\zeta_k>0.15$" if jj == 0 else None)
    for kk in (0, 4, 7):
        b.annotate(f"$k={kk + 1}$", (est[kk], pair[kk, 0]), (est[kk] * 2.2, pair[kk, 0] * 0.28), fontsize=6.5, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    b.set_xlabel("estimate $R_k/(4\\zeta_k\\omega_k^2)$ (m/N)"); b.set_ylabel("Hankel singular values (m/N)")
    b.set_title("(b) what ranks a mode", loc="left"); b.legend(frameon=False, loc="upper left")
    b.set_xlim(lo, hi); b.set_ylim(lo, hi)
    # (c) schematic of balancing
    c = fig.add_subplot(gs[2])
    p = np.array([4.0, 0.04, 1.0]); q = np.array([0.04, 4.0, 1.0]); sg = np.sqrt(p * q)
    xs = np.arange(3); wd = 0.26
    c.bar(xs - wd, p, wd, color="#9ecae1", edgecolor=BLUE, label="excitability $p_i$")
    c.bar(xs, q, wd, color="#fdd0a2", edgecolor=ORANGE, label="visibility $q_i$")
    c.bar(xs + wd, sg, wd, color="#d9d9d9", edgecolor=GREY, label="balanced: $p_i=q_i=\\sigma_i$")
    c.set_yscale("log"); c.set_ylim(0.02, 90); c.set_xticks(xs)
    c.set_xticklabels(["state a\neasy to excite,\nhard to see", "state b\nhard to excite,\nclearly seen", "state c\nboth"], fontsize=6)
    c.set_title("(c) balancing, schematic", loc="left"); c.legend(frameon=False, loc="upper center", fontsize=6.5)
    fig.savefig(os.path.join(A.FIG, "ch09_intro_modes.pdf"), bbox_inches="tight"); plt.close(fig)
    print("ratios", np.round(ratio, 3), "z", np.round(z[:npre], 3))


if __name__ == "__main__":
    d = setup()
    fig_port(d); fig_modes(d)
    print(NUM)
    with open(os.path.join(A.TAB, "numbers_ch09_intro.tex"), "w") as fh:
        fh.write("% generated by scripts/ch09_intro.py -- do not edit\n")
        for k_, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")

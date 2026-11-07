# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch03_strongweak_fig.py -- Figure of Section 3.1.5: why a function with kinks can solve the weak form but not the strong form.

The bar of Figure "the bar problem": clamped at x = 0, uniform load q, point load P at x = L, with L = 1 m, EA = 1e6 N, q = 2 kN/m, P = 1 kN.
Four linear elements. Computed here, not typed:
  (a) exact u(x) and the finite element u_h(x), with the largest nodal difference (Galerkin is nodally exact in 1-D);
  (b) the strong-form residual -(EA u_h')' - q of u_h: it equals -q inside each element, and u_h' jumps at every node, so the second
      derivative is a point force there (a Dirac delta), which is why the strong form cannot be applied to u_h as it stands;
  (c) the weak-form residual r_a = int EA N_a' u' dx - int N_a q dx - P N_a(L) for the test functions v = N_a, for u_h, for the exact u
      and for a deliberately wrong u (0.85 u_h).

Writes: figures/ch03_strongweak.pdf, tables/numbers_ch03_strongweak.tex
Run from the book directory:   python scripts/ch03_strongweak_fig.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
BLUE, ORANGE, RED, GREY = "#2b6cb0", "#dd8a2f", "#c0392b", "0.45"
L, EA, q, P, ne = 1.0, 1.0e6, 2.0e3, 1.0e3, 4

u_ex = lambda x: (P * x + q * (L * x - 0.5 * x * x)) / EA
du_ex = lambda x: (P + q * (L - x)) / EA


def main():
    x = np.linspace(0, L, ne + 1); h = np.diff(x)
    K = np.zeros((ne + 1, ne + 1)); f = np.zeros(ne + 1)
    for e in range(ne):
        K[e:e + 2, e:e + 2] += EA / h[e] * np.array([[1, -1], [-1, 1]]); f[e:e + 2] += q * h[e] / 2        # consistent load of a constant q
    f[-1] += P
    u = np.zeros(ne + 1); u[1:] = np.linalg.solve(K[1:, 1:], f[1:])
    nodal_diff = np.max(np.abs(u - u_ex(x)))
    fscale = P + q * L                                                       # force scale of the problem
    # weak residuals r_a (free nodes a = 1..ne)
    gp, gw = np.polynomial.legendre.leggauss(4)
    def residual_exact():
        r = np.zeros(ne + 1)
        for e in range(ne):
            xe = 0.5 * (x[e] + x[e + 1]) + 0.5 * h[e] * gp; we = 0.5 * h[e] * gw
            for loc, a in enumerate((e, e + 1)):
                Na = (x[e + 1] - xe) / h[e] if loc == 0 else (xe - x[e]) / h[e]
                dNa = -1 / h[e] if loc == 0 else 1 / h[e]
                r[a] += np.sum(we * (EA * dNa * du_ex(xe) - Na * q))
        r[-1] -= P
        return r
    r_fe = (K @ u - f)[1:] / fscale
    r_ex = residual_exact()[1:] / fscale
    u_w = 0.85 * u; r_w = (K @ u_w - f)[1:] / fscale
    # strong residual of u_h: -q inside each element; point force at each interior node = -[EA u_h'] (jump of the flux)
    du_h = np.diff(u) / h; jump = EA * (du_h[1:] - du_h[:-1])                 # EA u_h' at the right minus at the left of each interior node
    spike = -jump                                                           # -(EA u_h')' = -jump * delta(x - x_i)
    # (a)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax = axs[0]; xx = np.linspace(0, L, 200)
    ax.plot(xx, 1e3 * u_ex(xx), color=BLUE, lw=1.4, label="exact $u$"); ax.plot(x, 1e3 * u, "o-", color=ORANGE, lw=1.1, ms=3.5, label=r"finite element $u_h$")
    ax.set_xlabel("$x$ (m)", fontsize=7.5); ax.set_ylabel("displacement (mm)", fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left"); ax.set_title("(a) the solution", fontsize=8)
    ax.text(0.97, 0.06, f"nodal difference\n{nodal_diff / u_ex(L):.0e} of the tip value", transform=ax.transAxes, ha="right", fontsize=6.3, color="0.3")
    # (b)
    ax = axs[1]
    for e in range(ne):
        ax.plot([x[e], x[e + 1]], [-q / 1e3, -q / 1e3], color=BLUE, lw=1.6)
    for xi, sp in zip(x[1:-1], spike):
        ax.annotate("", xy=(xi, 2.6), xytext=(xi, -q / 1e3), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2, mutation_scale=7))
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlim(-0.02, L + 0.02); ax.set_ylim(-3.2, 3.3)
    ax.set_xlabel("$x$ (m)", fontsize=7.5); ax.set_ylabel(r"$-(EA\,u_h')'-q$  (kN/m between nodes)", fontsize=7.0); ax.tick_params(labelsize=6.5)
    ax.text(0.5 * L, -2.55, "inside each element: $-q$, not zero", ha="center", fontsize=6.3, color=BLUE)
    ax.text(0.98, 2.95, f"point force of {abs(spike[0]):.0f} N at each node\n(the derivative of a kink; arrows not to scale)", ha="right", va="top", fontsize=6.0, color=RED)
    ax.set_title("(b) the strong form fails for $u_h$", fontsize=8)
    # (c)
    ax = axs[2]; idx = np.arange(1, ne + 1); w = 0.26
    ax.bar(idx - w, r_ex, w, color=BLUE, label="exact $u$"); ax.bar(idx, r_fe, w, color=ORANGE, label=r"$u_h$")
    ax.bar(idx + w, r_w, w, color=RED, label=r"wrong: $0.85\,u_h$")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(idx); ax.set_xticklabels([f"$N_{a}$" for a in idx], fontsize=7)
    ax.set_xlabel("test function $v$", fontsize=7.5); ax.set_ylabel("weak residual / $(P+qL)$", fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.set_ylim(-0.075, 0.012)
    ax.legend(fontsize=6.0, frameon=False, loc="lower left")
    ax.set_title("(c) the weak form holds for $u_h$", fontsize=8)
    ax.text(0.03, 0.44, f"exact $u$ and $u_h$:\nat most {max(np.max(np.abs(r_ex)), np.max(np.abs(r_fe))):.0e}\n(rounding)", transform=ax.transAxes, ha="left", va="center", fontsize=6.3, color="0.3")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, "ch03_strongweak.pdf"), bbox_inches="tight"); plt.close(fig)
    with open(os.path.join(TAB, "numbers_ch03_strongweak.tex"), "w") as fh:
        fh.write("% generated by scripts/ch03_strongweak_fig.py -- do not edit\n")
        def m(k, v): fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
        m("cswNe", str(ne)); m("cswNodalDiff", f"${nodal_diff:.1e}$".replace("e-", "\\times10^{-").replace("e+", "\\times10^{") + "}" if False else f"{nodal_diff:.1e}")
        m("cswResFE", f"{np.max(np.abs(r_fe)):.1e}"); m("cswResExact", f"{np.max(np.abs(r_ex)):.1e}"); m("cswResWrong", f"{np.max(np.abs(r_w)):.2f}")
        m("cswSpike", f"{np.max(np.abs(spike)):.0f}"); m("cswQ", f"{q / 1e3:.0f}")
    print("nodal diff", nodal_diff, "| max weak residual: FE", np.max(np.abs(r_fe)), "exact", np.max(np.abs(r_ex)), "wrong", np.max(np.abs(r_w)),
          "| spikes (N)", spike, "q h (N)", q * h[0])


if __name__ == "__main__":
    main()

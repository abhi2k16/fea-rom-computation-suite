# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch14_geom.py -- geometry and mesh figures of the reproduction studies of Chapter 14 (drawn from the dimensions in the
study reports; no study data are read). Writes book/figures/ch14_geom_*.pdf.

    python ch14_geom.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
BLUE, VIOLET, GREEN, ORANGE, GREY, RED = "#2b4c9b", "#6a3d9a", "#2e8b57", "#d2691e", "#555555", "#c0392b"


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name)


def clamp(ax, x, y0, y1, side, h):
    ax.plot([x, x], [y0, y1], color="k", lw=3)
    for y in np.linspace(y0, y1, 7):
        ax.plot([x, x + side * h], [y, y - 0.6 * h], color="k", lw=0.8)


def beam(ax, L, n, unit, probe=None, loadtxt=None, xlabel=None):
    x = np.linspace(0, L, n + 1)
    ax.plot(x, 0 * x, "-", color=GREY, lw=1.2); ax.plot(x, 0 * x, "o", color=BLUE, ms=3)
    h = 0.04 * L; clamp(ax, 0, -0.15 * L, 0.15 * L, -1, h); clamp(ax, L, -0.15 * L, 0.15 * L, 1, h)
    if probe is not None:
        ax.plot([probe], [0], "s", color=RED, ms=8); ax.annotate("observation point", (probe, 0), (probe, -0.13 * L), ha="center", color=RED, fontsize=8,
                                                              arrowprops=dict(arrowstyle="-", color=RED))
    if loadtxt:
        for xx in np.linspace(0.05 * L, 0.95 * L, 14):
            ax.annotate("", (xx, 0.02 * L), (xx, 0.13 * L), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
        ax.text(0.5 * L, 0.145 * L, loadtxt, ha="center", color=ORANGE, fontsize=8)
    ax.set(xlim=(-0.08 * L, 1.08 * L), ylim=(-0.2 * L, 0.2 * L), xlabel=xlabel or f"x ({unit})"); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.set_aspect("equal")


def mfs():
    fig = plt.figure(figsize=(8.8, 4.4))
    a = fig.add_subplot(2, 1, 1)
    beam(a, 228.6, 40, "mm", loadtxt="uniform pressure $P\\in[-2000,2000]$ Pa")
    a.set_title("(a) flat beam: 228.6 x 12.7 x 0.787 mm, 40 corotational beam elements, 117 free dofs", fontsize=8.5)
    b = fig.add_subplot(2, 1, 2)
    pts = [(0, -8), (220, -25), (280, -25), (280, 25), (220, 25), (0, 8)]
    b.add_patch(Polygon(pts, closed=True, fc="#dfe6f5", ec=BLUE, lw=1.4))
    for xx in np.linspace(0, 280, 15):
        b.plot([xx, xx], [-8 - (xx / 220) * 17 if xx <= 220 else -25, 8 + (xx / 220) * 17 if xx <= 220 else 25], color=BLUE, lw=0.4)
    b.plot([280, 280], [-25, 25], color="k", lw=4); b.text(284, 0, "clamped", rotation=90, va="center", fontsize=8)
    b.annotate("", (0, -34), (280, -34), arrowprops=dict(arrowstyle="<->")); b.text(140, -42, "span 280 mm", ha="center", fontsize=8)
    b.annotate("", (220, 30), (280, 30), arrowprops=dict(arrowstyle="<->")); b.text(250, 34, "60 mm", ha="center", fontsize=8)
    b.text(-4, 0, "16 mm", ha="right", va="center", fontsize=8); b.text(300, 0, "", fontsize=8); b.text(250, -31, "50 mm", ha="center", fontsize=8, color=GREY)
    b.set(xlim=(-30, 310), ylim=(-50, 45)); b.set_aspect("equal"); b.set_yticks([]); b.set_xlabel("x (mm); thickness 1.15 mm, T7050 aluminium")
    b.spines["left"].set_visible(False)
    b.set_title("(b) wing planform reconstructed from the dimensioned schematic: shell reference model, 140 elements", fontsize=8.5)
    save(fig, "ch14_geom_mfs.pdf")


def hyrom_panel():
    R, L, half = 2540.0, 508.0, 0.1
    fig = plt.figure(figsize=(7.4, 3.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    n = 10; xs = np.linspace(0, L, n + 1); th = np.linspace(-half, half, n + 1)
    X, T = np.meshgrid(xs, th, indexing="ij"); Y = R * np.sin(T); Z = R * np.cos(T)
    for i in range(n + 1):
        ax.plot(X[i], Y[i], Z[i], color=GREY, lw=0.6); ax.plot(X[:, i], Y[:, i], Z[:, i], color=GREY, lw=0.6)
    edge = np.zeros_like(X, bool); edge[0] = edge[-1] = edge[:, 0] = edge[:, -1] = True
    ax.plot(X[edge], Y[edge], Z[edge], "o", color=ORANGE, ms=3, label="hinged edge nodes (40)")
    ax.plot(X[~edge], Y[~edge], Z[~edge], ".", color=BLUE, ms=3, label="free nodes")
    c = n // 2; ax.plot(X[c - 1:c + 2, c - 1:c + 2].ravel(), Y[c - 1:c + 2, c - 1:c + 2].ravel(), Z[c - 1:c + 2, c - 1:c + 2].ravel(), "s", color=RED, ms=4, label="crown load patch (9 nodes)")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel=""); ax.set_zticks([]); ax.view_init(28, -62); ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(a) 10 x 10 mesh: 121 nodes, 726 dofs", fontsize=8.5)
    b = fig.add_subplot(1, 2, 2)
    t = np.linspace(-half, half, 60); b.plot(R * np.sin(t), R * np.cos(t) - R * np.cos(half), color=BLUE, lw=3)
    b.plot([R * np.sin(half)] * 2, [0, -60], color="k", lw=0.8); b.plot([-R * np.sin(half)] * 2, [0, -60], color="k", lw=0.8)
    b.annotate("", (-R * np.sin(half), -50), (R * np.sin(half), -50), arrowprops=dict(arrowstyle="<->")); b.text(0, -58, "$L=508$ mm", ha="center", fontsize=8)
    b.annotate("", (0, R * (1 - np.cos(half))*0 + 0.5), (0, R * (1 - np.cos(half)) * 0 + 12.7), arrowprops=dict(arrowstyle="-")); b.text(8, 6, "rise 12.7 mm", fontsize=8)
    b.annotate("crown load\n5000 N, 2.5 Hz", (0, 12.7), (0, 38), ha="center", color=RED, fontsize=8, arrowprops=dict(arrowstyle="<-", color=RED))
    b.text(0, -28, "$R=2540$ mm, $t=12.5$ mm", ha="center", fontsize=8, color=GREY)
    b.set(xlim=(-290, 290), ylim=(-70, 50)); b.set_yticks([]); b.spines["left"].set_visible(False); b.set_xlabel("section through the crown line")
    b.set_title("(b) hinged on all four edges", fontsize=8.5)
    save(fig, "ch14_geom_hyrom.pdf")


def ice():
    fig, ax = plt.subplots(figsize=(8.4, 1.9))
    beam(ax, 18.0, 36, "in", probe=4.0, loadtxt="spatially uniform random pressure, 0--1500 Hz band")
    ax.set_title("18 x 1 x 0.09 in aluminium beam, clamped at both ends, immovable ends; 36 elements, $\\Delta x=0.5$ in", fontsize=8.5)
    save(fig, "ch14_geom_ice.pdf")


def fdpinn():
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]; L = 20.0
    a.plot([0, L], [0, 0], color=BLUE, lw=5, solid_capstyle="butt")
    for x in (0, L):
        a.add_patch(Polygon([(x, -0.2), (x - 0.5, -1.1), (x + 0.5, -1.1)], closed=True, fc="white", ec="k"))
    a.annotate("", (6, 0.2), (6, 2.0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6)); a.text(6, 2.2, "$p=12$ kN at $x=vt$", ha="center", color=ORANGE, fontsize=8)
    a.annotate("", (9.5, 1.3), (6.5, 1.3), arrowprops=dict(arrowstyle="<-", color=ORANGE)); a.text(8.0, 1.6, "$v$", color=ORANGE, ha="center", fontsize=8)
    a.plot([L / 2], [0], "s", color=RED, ms=7); a.text(L / 2, -1.9, "sensor at $L/2$ (inverse problem)", ha="center", color=RED, fontsize=8)
    a.annotate("", (0, -2.9), (L, -2.9), arrowprops=dict(arrowstyle="<->")); a.text(L / 2, -3.7, "$L=20$ m, $EI=2.4\\times10^9$ N m$^2$, $m=3000$ kg/m, damping ratio 0.025", ha="center", fontsize=8)
    a.set(xlim=(-2, 22), ylim=(-4.3, 3.2)); a.set_aspect("equal"); a.axis("off"); a.set_title("(a) simply supported beam under a moving load", fontsize=8.5)
    b = ax[1]; b.set(xlim=(0, 1), ylim=(0, 1)); b.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#eef2fb", ec=BLUE))
    rng = np.random.default_rng(1); b.plot(rng.random(70), rng.random(70), ".", color=GREY, ms=3)
    b.set(xlabel="$x/L$", ylabel="$\\omega$ (band 0--5 Hz)"); b.set_xticks([0, 1]); b.set_yticks([0, 1]); b.set_yticklabels(["0", "$\\omega_{\\max}$"])
    b.set_title("(b) network domain $(x,\\omega)$, collocation points", fontsize=8.5)
    save(fig, "ch14_geom_fdpinn.pdf")


def frino():
    fig = plt.figure(figsize=(7.6, 3.4)); ax = fig.add_subplot(111, projection="3d")
    L, W, T = 60.0, 15.0, 1.5; nx, ny, nz = 14, 8, 2
    xs, ys, zs = np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1), np.linspace(0, T, nz + 1)
    for z in zs:
        for y in ys: ax.plot([0, L], [y, y], [z, z], color=GREY, lw=0.4)
        for x in xs: ax.plot([x, x], [0, W], [z, z], color=GREY, lw=0.4)
    for x in xs:
        for y in (0, W): ax.plot([x, x], [y, y], [0, T], color=GREY, lw=0.4)
    for y in ys:
        for x in (0, L): ax.plot([x, x], [y, y], [0, T], color=GREY, lw=0.4)
    yy, zz = np.meshgrid(ys, zs); ax.plot_surface(np.zeros_like(yy), yy, zz, color=BLUE, alpha=0.5)
    ax.quiver(L, W / 2, T, 0, 0, 5, color=ORANGE, lw=2, arrow_length_ratio=0.4); ax.text(L, W / 2, T + 7, "point load (transverse)", color=ORANGE, fontsize=8, ha="center")
    ax.text(0, W / 2, -5, "clamped end", color=BLUE, fontsize=8)
    ax.set(xlabel="x (mm)", ylabel="y (mm)"); ax.set_zticks([]); ax.set_yticks([0, 15]); ax.set_box_aspect((4, 1, 0.6)); ax.view_init(22, -62)
    ax.set_title("Cantilever $60\\times15\\times1.5$ mm, $14\\times8\\times2$ Hex20: 4239 dofs", fontsize=8.5)
    save(fig, "ch14_geom_frino.pdf")


def rope():
    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.8), gridspec_kw={"width_ratios": [1.7, 1]})
    a = ax[0]
    for m2, c in ((-0.05, BLUE), (0.17, VIOLET), (0.5, GREEN)):
        d = 0.2 * (np.sin(np.pi * x) - np.sin(3 * np.pi * x)) + 0.5 - m2 * (x - 0.5)
        a.plot(x, d, color=c, lw=1.3, label=f"obstacle $d$, $\\mu_2={m2}$")
    a.plot(x, 0 * x, "o", color=GREY, ms=1.8)
    a.plot([0, 1], [0, 0], color="k", lw=0.6); a.plot([0, 0, 1, 1], [0, 0.2, 0.2, 0], "k", lw=0)
    for xx in np.linspace(0.04, 0.96, 16):
        a.annotate("", (xx, 0.02), (xx, 0.12), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    a.text(0.5, 0.135, "load $f$ (towards the obstacle)", ha="center", color=ORANGE, fontsize=8)
    a.set(xlim=(0, 1), ylim=(-0.05, 1.2), xlabel="$x$", ylabel="$u$, $d$"); a.legend(fontsize=7, loc="upper center", ncol=3)
    a.set_title("(a) rope (100 nodes, $u(0)=u(1)=0$) and the parametric obstacle", fontsize=8.5)
    b = ax[1]
    for m1, c in ((0.05, BLUE), (0.15, VIOLET), (0.25, GREEN)):
        b.plot([0, 0.5, 0.5, 1], [m1, m1, 0.15, 0.15], color=c, lw=1.3, label=f"$\\mu_1={m1}$")
    b.set(xlabel="$x$", ylabel="$\\nu(x,\\mu_1)$", ylim=(0, 0.3)); b.legend(fontsize=7)
    b.set_title("(b) stiffness coefficient", fontsize=8.5)
    save(fig, "ch14_geom_rope.pdf")


if __name__ == "__main__":
    mfs(); hyrom_panel(); ice(); fdpinn(); frino(); rope()

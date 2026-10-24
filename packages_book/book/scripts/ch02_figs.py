# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch02_figs.py -- the scripted figures, tables and number macros added to Chapter 2 (meshes).

Every figure is drawn from the package's own objects (Mesh, MultiBlockMesh, the element classes, the generators):
matplotlib for 2-D and for plots, PyVista (fea_engine.visualization, run under ``xvfb-run -a``) for 3-D renderings.

Writes into book/:
    figures/ch02_anatomy.pdf       node and element numbering, connectivity table, node sets            (Sec. 2.1)
    figures/ch02_elements.pdf      the nine reference elements with node numbers                        (Sec. 2.2)
    figures/ch02_quality.pdf       Jacobian map of perturbed meshes, a re-entrant element, histograms   (Sec. 2.3)
    tables/ch02_elements.tex       nodes, dimension, Gauss points, node-order check
    tables/ch02_quality.tex        distortion level against det J, angles and patch-test error
    tables/numbers_ch02_figs.tex   every number quoted in the prose of those sections
Run from the repository root:   xvfb-run -a python packages_book/book/scripts/ch02_figs.py [step1|step2|step3|all]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.grid": False, "figure.dpi": 150})
NUM = {}
BLUE, ORANGE, RED, GREY, GREEN = "#2b6cb0", "#dd8a2f", "#c0392b", "0.45", "#2f855a"


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


from fea_engine import (FESystem, Material, D_plane_stress, Quad4PlaneStress, Tri3PlaneStress, Hex8Solid3D, mesh as M)  # noqa: E402
from fea_engine.elements import (Quad8PlaneStress, Hex20Solid3D, Tet4Solid3D, Tet10Solid3D, Tri6PlaneStress,           # noqa: E402
                                 gauss_product)


# ---------------------------------------------------------------------------------------------- Sec. 2.1 anatomy
def fig_anatomy():
    m = M.rectangle_mesh(4.0, 3.0, 4, 3)
    m.select_nodes(x=0.0, name="root"); m.select_nodes(x=4.0, name="tip"); m.select_nodes(y=3.0, name="top")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    for ax in axs[:2]:
        for e in m.elements:
            ax.add_patch(Polygon(m.nodes[e], closed=True, fc="#eaf1f8", ec=GREY, lw=0.8))
        ax.set_xlim(-0.35, 4.35); ax.set_ylim(-0.35, 3.35); ax.set_aspect("equal"); ax.axis("off")
    ax = axs[0]
    for i, p in enumerate(m.nodes):
        ax.plot(*p, "o", ms=9, mfc="white", mec=BLUE, mew=0.8)
        ax.text(p[0], p[1], str(i), ha="center", va="center", fontsize=6, color=BLUE)
    for k, e in enumerate(m.elements):
        c = m.nodes[e].mean(0)
        ax.text(c[0], c[1], f"e{k}", ha="center", va="center", fontsize=7, color=ORANGE, weight="bold")
    ax.set_title("(a) node and element numbers", fontsize=8)
    ax = axs[1]
    cols = {"root": RED, "tip": GREEN, "top": ORANGE}
    for name, col in cols.items():
        ids = m.node_set(name)
        ax.plot(m.nodes[ids, 0], m.nodes[ids, 1], "o", ms=7, color=col, label=f'"{name}" ({len(ids)})', zorder=3)
    ax.plot(m.nodes[:, 0], m.nodes[:, 1], ".", ms=2, color=GREY)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5, frameon=False, handletextpad=0.2,
              columnspacing=0.8)
    ax.set_title("(b) node sets by coordinates", fontsize=8)
    ax = axs[2]; ax.axis("off")
    rows = [f"e{k}: " + " ".join(f"{n:2d}" for n in m.elements[k]) for k in range(6)]
    ax.text(0.0, 0.97, "connectivity (first 6)", fontsize=7.5, va="top", weight="bold")
    ax.text(0.0, 0.88, "\n".join(rows), family="monospace", fontsize=7, va="top", linespacing=1.5)
    ax.text(0.0, 0.40, "nodes (first 4)\n", fontsize=7.5, va="top", weight="bold")
    ax.text(0.0, 0.33, "\n".join(f"{i:2d}: ({m.nodes[i, 0]:.0f}, {m.nodes[i, 1]:.0f})" for i in range(4)),
            family="monospace", fontsize=7, va="top", linespacing=1.5)
    ax.set_title("(c) the two arrays", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ch02_anatomy.pdf"), bbox_inches="tight"); plt.close(fig)
    num("cmeAnNodes", str(len(m.nodes))); num("cmeAnElems", str(len(m.elements)))
    num("cmeAnRoot", str(len(m.node_set("root")))); num("cmeAnTip", str(len(m.node_set("tip"))))
    num("cmeAnTop", str(len(m.node_set("top"))))


# ---------------------------------------------------------------------------------------------- Sec. 2.2 elements
def _ref_coords():
    """Natural coordinates of the nodes of each element, in the package's node order."""
    r = {}
    r["Tri3PlaneStress"] = np.array([[0, 0], [1, 0], [0, 1]], float)
    r["Quad4PlaneStress"] = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    r["Tri6PlaneStress"] = np.vstack([r["Tri3PlaneStress"], [[.5, 0], [.5, .5], [0, .5]]])
    r["Quad8PlaneStress"] = np.column_stack([Quad8PlaneStress._xi_i, Quad8PlaneStress._eta_i])
    r["Tet4Solid3D"] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    t4 = r["Tet4Solid3D"]
    r["Tet10Solid3D"] = np.vstack([t4] + [(t4[a] + t4[b]) / 2 for a, b in Tet10Solid3D._edges])
    r["Hex8Solid3D"] = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    r["Hex20Solid3D"] = np.column_stack([Hex20Solid3D._xi_i, Hex20Solid3D._eta_i, Hex20Solid3D._zeta_i])
    return r


_EDGES = {
    "Tri3PlaneStress": [(0, 1), (1, 2), (2, 0)],
    "Quad4PlaneStress": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "Tri6PlaneStress": [(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)],
    "Quad8PlaneStress": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)],
    "Tet4Solid3D": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "Tet10Solid3D": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 0), (0, 7), (7, 3), (1, 8), (8, 3), (2, 9), (9, 3)],
    "Hex8Solid3D": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
    "Hex20Solid3D": [(0, 8), (8, 1), (1, 9), (9, 2), (2, 10), (10, 3), (3, 11), (11, 0), (4, 12), (12, 5), (5, 13),
                     (13, 6), (6, 14), (14, 7), (7, 15), (15, 4), (0, 16), (16, 4), (1, 17), (17, 5), (2, 18), (18, 6),
                     (3, 19), (19, 7)],
}


def fig_elements():
    refs = _ref_coords()
    classes = [Tri3PlaneStress, Quad4PlaneStress, Tri6PlaneStress, Quad8PlaneStress, Tet4Solid3D, Hex8Solid3D,
               Tet10Solid3D, Hex20Solid3D]
    labels = {"Tri3PlaneStress": "tri3", "Quad4PlaneStress": "quad4", "Tri6PlaneStress": "tri6",
              "Quad8PlaneStress": "quad8", "Tet4Solid3D": "tet4", "Hex8Solid3D": "hex8", "Tet10Solid3D": "tet10",
              "Hex20Solid3D": "hex20"}
    rows, ok_all = [], True
    fig = plt.figure(figsize=(7.4, 5.4))
    for k, cls in enumerate(classes):
        name = cls.__name__
        X = refs[name]; el = cls(); dim = X.shape[1]
        # Kronecker check: shape function i is 1 at node i and 0 at every other node
        K = np.array([el.shape_and_derivs(tuple(x))[0] for x in X])
        ok = bool(np.abs(K - np.eye(len(X))).max() < 1e-12); ok_all &= ok
        # partition of unity at a point
        pu = abs(el.shape_and_derivs(tuple(X.mean(0)))[0].sum() - 1.0)
        if dim == 2:
            ax = fig.add_subplot(2, 4, k + 1)
        else:
            ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        for a, b in _EDGES[name]:
            ax.plot(*zip(X[a], X[b]), color=GREY, lw=1.0)
        nc = {"Tri3PlaneStress": 3, "Quad4PlaneStress": 4, "Tri6PlaneStress": 3, "Quad8PlaneStress": 4,
              "Tet4Solid3D": 4, "Hex8Solid3D": 8, "Tet10Solid3D": 4, "Hex20Solid3D": 8}[name]
        for i, x in enumerate(X):
            col = BLUE if i < nc else ORANGE
            if dim == 2:
                ax.plot(*x, "o", ms=9, mfc="white", mec=col, mew=1.0)
                ax.text(*x, str(i), ha="center", va="center", fontsize=6, color=col)
            else:
                ax.plot(*[[c] for c in x], "o", ms=8, mfc="white", mec=col, mew=1.0)
                ax.text(*(x + 0.09 * np.sign(x - X.mean(0))), str(i), fontsize=6, color=col, ha="center", va="center")
        ax.set_title(f"{labels[name]}  ({len(X)} nodes)", fontsize=8)
        if dim == 2:
            ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.18)
        else:
            ax.set_box_aspect((1, 1, 1), zoom=1.25); ax.set_axis_off(); ax.view_init(22, -62 if "Hex" in name else -28)
            lim = 1.3 if "Hex" in name else 1.15
            c0 = X.mean(0)
            ax.set_xlim(c0[0] - (1.1 if "Hex" in name else 0.7), c0[0] + (1.1 if "Hex" in name else 0.7))
            ax.set_ylim(c0[1] - (1.1 if "Hex" in name else 0.7), c0[1] + (1.1 if "Hex" in name else 0.7))
            ax.set_zlim(c0[2] - (1.1 if "Hex" in name else 0.7), c0[2] + (1.1 if "Hex" in name else 0.7))
        go = getattr(cls, "gauss_order", None)
        if go is not None:
            ng = go ** dim if "Quad" in name or "Hex" in name else None
        else:
            ng = None
        rows.append((labels[name], len(X), dim, nc, len(X) - nc, go, ng, ok, pu))
    # fig: ninth panel = legend
    ax = fig.add_subplot(2, 4, 8) if False else None
    fig.text(0.5, 0.005, "blue: corner nodes, orange: mid-edge nodes; numbers are the position in the connectivity row",
             ha="center", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(FIG, "ch02_elements.pdf"), bbox_inches="tight"); plt.close(fig)
    with open(os.path.join(TAB, "ch02_elements.tex"), "w") as f:
        f.write("% generated by scripts/ch02_figs.py\n\\begin{tabular}{@{}l r r r r l c@{}}\n\\toprule\n"
                "Element & Nodes & Dim. & Corner & Mid-edge & Gauss rule & Kronecker check \\\\\n\\midrule\n")
        for lab, n, d, c, mid, go, ng, ok, pu in rows:
            if go is None:
                rule = "simplex rule"
            else:
                rule = f"${go}^{d}={go ** d}$ points" if lab[:4] in ("quad", "hex8", "hex2") else f"order {go} simplex rule"
            f.write(f"\\texttt{{{lab}}} & {n} & {d} & {c} & {mid} & {rule} & {'pass' if ok else 'FAIL'} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    num("cmeElAllOk", "yes" if ok_all else "NO")
    num("cmeElMaxPU", sci(max(r[8] for r in rows)))


# ---------------------------------------------------------------------------------------------- Sec. 2.3 quality
def _perturbed(f, nx=14, ny=8, seed=3):
    m = M.rectangle_mesh(2.0, 1.0, nx, ny)
    rng = np.random.default_rng(seed)
    h = np.array([2.0 / nx, 1.0 / ny])
    X = m.nodes.copy()
    inner = (X[:, 0] > 1e-9) & (X[:, 0] < 2 - 1e-9) & (X[:, 1] > 1e-9) & (X[:, 1] < 1 - 1e-9)
    X[inner] += f * h * rng.uniform(-1, 1, (inner.sum(), 2))
    return M.Mesh(X, m.elements.copy(), dim=2)


def _detj(m, el):
    pts, _ = gauss_product(el.gauss_order, el.dim)
    out = np.empty((len(m.elements), len(pts)))
    for k, conn in enumerate(m.elements):
        xc = m.nodes[conn]
        for g, p in enumerate(pts):
            _, dN = el.shape_and_derivs(p)
            out[k, g] = np.linalg.det(dN @ xc)
    return out


def _angles(m):
    A = []
    for conn in m.elements:
        P = m.nodes[conn]; a = []
        for i in range(4):
            u = P[(i - 1) % 4] - P[i]; v = P[(i + 1) % 4] - P[i]
            a.append(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        A.append(a)
    return np.array(A)


def _aspect(m):
    r = []
    for conn in m.elements:
        P = m.nodes[conn]
        L = [np.linalg.norm(P[(i + 1) % 4] - P[i]) for i in range(4)]
        r.append(max(L) / min(L))
    return np.array(r)


def _patch_error(m):
    el = Quad4PlaneStress()
    D = D_plane_stress(Material(E=1.0, nu=0.3, rho=1.0))
    s = FESystem(m, el)
    s.assemble_stiffness(D, thickness=1.0)
    G_ = np.array([[1e-3, 2e-4], [3e-4, -5e-4]]); b = np.array([1e-3, -2e-3])
    exact = m.nodes @ G_.T + b
    X = m.nodes
    on_b = np.where((X[:, 0] < 1e-9) | (X[:, 0] > 2 - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > 1 - 1e-9))[0]
    for n in on_b:
        for d in range(2):
            s.fix_dofs([int(n)], [d], value=float(exact[n, d]))
    U = np.asarray(s.solve_static()).reshape(-1, 2)
    return float(np.abs(U - exact).max() / np.abs(exact).max())


def fig_quality():
    el = Quad4PlaneStress()
    levels = [0.0, 0.15, 0.30, 0.45]
    data = {}
    for f in levels:
        m = _perturbed(f)
        d = _detj(m, el)
        nominal = (2.0 / 14) * (1.0 / 8) / 4.0          # detJ of the undistorted element
        data[f] = dict(m=m, dmin=d.min(1) / nominal, ratio=d.max() / d.min() if d.min() > 0 else np.inf,
                       ang=_angles(m), asp=_aspect(m), pos=bool(d.min() > 0), err=_patch_error(m) if d.min() > 0 else np.nan)
    fig = plt.figure(figsize=(7.4, 5.6))
    gt = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.45, 1.15], left=0.04, right=0.97, top=0.93, bottom=0.55, wspace=0.12)
    gb = fig.add_gridspec(1, 3, left=0.07, right=0.97, top=0.42, bottom=0.09, wspace=0.35)
    vmin, vmax = 0.0, 2.0
    tops = []
    for k, f in enumerate([0.15, 0.45]):
        ax = fig.add_subplot(gt[0, k]); tops.append(ax)
        m = data[f]["m"]
        pc = PolyCollection([m.nodes[e] for e in m.elements], array=data[f]["dmin"], cmap="RdYlBu", edgecolors="0.3",
                            linewidths=0.4)
        pc.set_clim(vmin, vmax); ax.add_collection(pc)
        ax.set_xlim(-0.05, 2.05); ax.set_ylim(-0.08, 1.08); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"({'ab'[k]}) interior nodes moved by up to {int(f * 100)}%\nof the cell size", fontsize=7.5)
    cb = fig.colorbar(pc, ax=tops, location="bottom", fraction=0.05, pad=0.01, shrink=0.6, aspect=40)
    cb.set_label(r"$\min_g\det J\,/\,\det J_{\rm nom}$ (element)", fontsize=7); cb.ax.tick_params(labelsize=6)
    # (c) convex and re-entrant element with Gauss-point det J
    ax = fig.add_subplot(gt[0, 2])
    pts, _ = gauss_product(2, 2)
    shapes = {"convex": np.array([[0, 0], [1, 0], [1.1, 1], [0.1, 0.9]]),
              "re-entrant": np.array([[1.6, 0], [2.6, 0], [2.7, 1], [2.42, 0.45]])}
    reent = {}
    for name, P in shapes.items():
        ax.add_patch(Polygon(P, closed=True, fc="#eaf1f8" if name == "convex" else "#fbe3e0", ec=GREY, lw=1))
        dets = []
        for p in pts:
            N, dN = el.shape_and_derivs(p)
            xg = N @ P; dj = np.linalg.det(dN @ P); dets.append(dj)
            ax.plot(*xg, "o", ms=4, color=GREEN if dj > 0 else RED, zorder=3)
            ax.text(xg[0], xg[1] + 0.09, f"{dj:.2f}", ha="center", fontsize=6.5, color=GREEN if dj > 0 else RED)
        reent[name] = dets
        ax.text(P[:, 0].mean(), -0.2, name, ha="center", fontsize=7)
    ax.set_xlim(-0.2, 2.9); ax.set_ylim(-0.35, 1.25); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(r"(c) $\det J$ at the Gauss points", fontsize=7.5)
    # (d) histogram of aspect ratio, (e) min angle, (f) fraction
    ax = fig.add_subplot(gb[0, 0])
    for f, c in zip(levels, ["0.5", BLUE, ORANGE, RED]):
        ax.hist(data[f]["asp"], bins=np.linspace(1, 4, 25), histtype="step", color=c, lw=1.1, label=f"{int(f * 100)}%")
    ax.set_xlabel("aspect ratio (longest/shortest edge)", fontsize=7); ax.set_ylabel("elements", fontsize=7)
    ax.legend(fontsize=6, frameon=False, title="perturbation", title_fontsize=6); ax.set_title("(d)", fontsize=7.5)
    ax = fig.add_subplot(gb[0, 1])
    for f, c in zip(levels, ["0.5", BLUE, ORANGE, RED]):
        ax.hist(data[f]["ang"].min(1), bins=np.linspace(20, 90, 29), histtype="step", color=c, lw=1.1)
    ax.axvline(45, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("smallest interior angle (deg)", fontsize=7); ax.set_title("(e)", fontsize=7.5)
    ax = fig.add_subplot(gb[0, 2])
    mins = [data[f]["dmin"].min() for f in levels]
    ax.plot([100 * f for f in levels], mins, "o-", color=BLUE, ms=4)
    ax.axhline(0, color=RED, lw=0.8)
    ax.set_xlabel("perturbation (% of cell size)", fontsize=7)
    ax.set_ylabel(r"$\min\det J$ / nominal", fontsize=7); ax.set_title("(f)", fontsize=7.5)
    for a in fig.axes:
        a.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, "ch02_quality.pdf"), bbox_inches="tight"); plt.close(fig)
    # table
    with open(os.path.join(TAB, "ch02_quality.tex"), "w") as fh:
        fh.write("% generated by scripts/ch02_figs.py\n\\begin{tabular}{@{}r r r r r r@{}}\n\\toprule\n"
                 "Perturbation & min $\\det\\mathbf J$ (nominal $=1$) & max/min $\\det\\mathbf J$ & max aspect & min angle ($^\\circ$) & patch error \\\\\n\\midrule\n")
        for f in levels:
            d = data[f]
            fh.write(f"{int(f * 100)}\\,\\% & {d['dmin'].min():.2f} & {d['ratio']:.1f} & {d['asp'].max():.2f} & "
                     f"{d['ang'].min():.0f} & {sci(d['err']) if d['pos'] else 'n/a'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cmeQPatchMax", sci(max(data[f]["err"] for f in levels if data[f]["pos"])))
    num("cmeQMinDetHi", f"{data[0.45]['dmin'].min():.2f}")
    num("cmeQConvexMin", f"{min(reent['convex']):.2f}"); num("cmeQReentMin", f"{min(reent['re-entrant']):.2f}")
    num("cmeQReentNeg", str(sum(1 for d in reent['re-entrant'] if d <= 0)))
    # first perturbation at which an element inverts
    first = None
    for f in np.arange(0.5, 1.51, 0.05):
        if _detj(_perturbed(f), el).min() <= 0:
            first = f; break
    num("cmeQFirstBad", f"{int(round(first * 100))}" if first else "more than 150")
    print("quality:", {f: (round(data[f]['dmin'].min(), 2), data[f]['err']) for f in levels}, "first inversion", first)


# ---------------------------------------------------------------------------------------------- Sec. 2.4 generators
def _fill2d(ax, m, fc="#eaf1f8", ec="0.35", lw=0.4, vals=None, cmap="viridis", clim=None):
    polys = [m.nodes[e] for e in m.elements]
    pc = PolyCollection(polys, facecolors=fc if vals is None else None, edgecolors=ec, linewidths=lw,
                        array=vals, cmap=cmap)
    if clim is not None and vals is not None:
        pc.set_clim(*clim)
    ax.add_collection(pc)
    lo, hi = m.nodes.min(0), m.nodes.max(0); pad = 0.04 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal"); ax.axis("off")
    return pc


def _gen_cases():
    R = 0.3
    return dict(
        rect=M.rectangle_mesh(2.0, 1.0, 12, 6),
        quarter=M.rectangle_with_hole_mesh_quarter(2.0, 1.0, R, 8, 16),
        holeoff=M.hole_in_rectangle_mesh(3.0, 2.0, (1.2, 1.0), R, 6, 16),
        graded=M.hole_in_rectangle_mesh_graded(3.0, 2.0, (1.2, 1.0), R, 6, 16, h_far=0.3, growth_ratio=1.2),
        notch=M.notch_in_rectangle_mesh(2.0, 1.0, "top", 0.8, 1.2, 0.6, 8, 4, 8, 6, 6),
        fillet=M.fillet_in_rectangle_mesh(2.0, 1.0, (2.0, 1.0), R, 10, 6, 6),
        full=M.mirror_mesh(M.rectangle_with_hole_mesh_quarter(1.5, 1.0, R, 6, 12), mirror_x=True, mirror_y=True),
        line=M.line_mesh(2.0, 6))


def fig_gallery2d():
    c = _gen_cases()
    items = [("rect", "rectangle_mesh"), ("quarter", "rectangle_with_hole_\nmesh_quarter"),
             ("holeoff", "hole_in_rectangle_mesh"), ("graded", "hole_in_rectangle_\nmesh_graded (ratio 1.2)"),
             ("notch", "notch_in_rectangle_mesh"), ("fillet", "fillet_in_rectangle_mesh"),
             ("full", "mirror_mesh of the\nquarter plate"), ("line", "line_mesh")]
    fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.2))
    for ax, (k, t) in zip(axs.ravel(), items):
        m = c[k]
        if k == "line":
            x = m.nodes[:, 0]
            for a, b in m.elements:
                ax.plot([x[a], x[b]], [0, 0], color=BLUE, lw=3, solid_capstyle="butt")
            ax.plot(x, 0 * x, "o", ms=5, mfc="white", mec=ORANGE)
            ax.set_xlim(-0.2, 2.2); ax.set_ylim(-1, 1); ax.axis("off")
        else:
            _fill2d(ax, m)
        ax.set_title(t, fontsize=6.5, family="monospace")
        n = f"{len(m.nodes)} nodes, {len(m.elements)} el."
        ax.text(0.5, -0.04, n, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="0.3")
    fig.tight_layout(h_pad=1.6)
    fig.savefig(os.path.join(FIG, "ch02_gallery2d.pdf"), bbox_inches="tight"); plt.close(fig)


def _render(grid_mesh, color="#cfe0f1", edge="#333333", view=(1.0, -1.1, 0.8), window=(520, 380), highlight=None,
            scalars=None, clim=None, cmap="viridis"):
    import pyvista as pv
    from fea_engine.visualization import to_pyvista
    pv.OFF_SCREEN = True
    g = to_pyvista(grid_mesh)
    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("white")
    if scalars is not None:
        g.cell_data["h"] = scalars
        pl.add_mesh(g, scalars="h", cmap=cmap, clim=clim, show_edges=True, edge_color=edge, line_width=0.8,
                    show_scalar_bar=False)
    else:
        pl.add_mesh(g, color=color, show_edges=True, edge_color=edge, line_width=1.0)
    pl.camera_position = [tuple(np.array(g.center) + np.array(view) * g.length * 1.15), g.center, (0, 0, 1)]
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def fig_gallery3d():
    R = 0.3
    meshes = [("box_mesh", M.box_mesh(2.0, 1.0, 0.5, 8, 4, 2)),
              ("box_with_hole_mesh", M.box_with_hole_mesh(2.0, 1.0, R, 0.5, 6, 12, 2)),
              ("extrude_mesh of the fillet", M.extrude_mesh(M.fillet_in_rectangle_mesh(2.0, 1.0, (2.0, 1.0), R, 10, 6, 6), 0.5, 2))]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for ax, (t, m) in zip(axs, meshes):
        ax.imshow(_render(m)); ax.axis("off")
        ax.set_title(f"{t}\n{len(m.nodes)} nodes, {len(m.elements)} hex8", fontsize=6.8, family="monospace")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(FIG, "ch02_gallery3d.pdf"), bbox_inches="tight", dpi=200); plt.close(fig)


def _area(m):
    X = m.nodes[m.elements]; x, y = X[:, :, 0], X[:, :, 1]
    return float(0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)).sum())


def fig_arc():
    R = 0.3
    nth = np.array([4, 8, 16, 32, 64])
    cases = {
        "quarter plate with hole": (lambda k: M.rectangle_with_hole_mesh_quarter(2.0, 1.0, R, 8, k), 2.0 - np.pi * R ** 2 / 4),
        "hole off centre": (lambda k: M.hole_in_rectangle_mesh(3.0, 2.0, (1.2, 1.0), R, 6, k), 6.0 - np.pi * R ** 2),
        "fillet": (lambda k: M.fillet_in_rectangle_mesh(2.0, 1.0, (2.0, 1.0), R, 10, 6, 6, ntheta=k), 2.0 - R ** 2 * (1 - np.pi / 4)),
    }
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    orders = {}
    for (lab, (f, ex)), col, mk in zip(cases.items(), [BLUE, ORANGE, GREEN], "osd"):
        err = np.array([abs(_area(f(int(k))) / ex - 1) for k in nth])
        o = -np.polyfit(np.log(nth[1:]), np.log(err[1:]), 1)[0]
        orders[lab] = (o, err)
        ax.loglog(nth, err, mk + "-", color=col, ms=4, label=f"{lab} (order {o:.2f})")
    ax.loglog(nth, 0.5 * orders["hole off centre"][1][1] * (nth / nth[1]) ** -2.0, ":", color="0.4", label="slope $-2$")
    ax.set_xlabel("elements around the arc", fontsize=8); ax.set_ylabel("relative area error", fontsize=8)
    ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=7); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_arc.pdf")); plt.close(fig)
    num("cmeArcOrdHoleOff", f"{orders['hole off centre'][0]:.2f}")
    num("cmeArcOrdFillet", f"{orders['fillet'][0]:.2f}")
    num("cmeArcOrdQuarter", f"{orders['quarter plate with hole'][0]:.2f}")
    print("arc orders", {k: round(v[0], 2) for k, v in orders.items()})


# ---------------------------------------------------------------------------------------------- Sec. 2.5 grading
def _sizes(m):
    X = m.nodes[m.elements]; x, y = X[:, :, 0], X[:, :, 1]
    return np.sqrt(0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)))


def fig_grade_meshes():
    from fea_engine import grading as G
    R = 0.3
    cases = [("uniform ($p=1$)", M.hole_in_rectangle_mesh(3.0, 2.0, (1.2, 1.0), R, 12, 24, grade_p=1.0, extend_grade=1.0))]
    for gr in (1.1, 1.2, 1.5):
        cases.append((f"target ratio {gr}", M.hole_in_rectangle_mesh_graded(3.0, 2.0, (1.2, 1.0), R, 12, 24, h_far=0.3, growth_ratio=gr)))
    hmax = max(_sizes(m).max() for _, m in cases); hmin = min(_sizes(m).min() for _, m in cases)
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.2))
    rows = []
    for ax, (t, m) in zip(axs, cases):
        sz = _sizes(m)
        pc = _fill2d(ax, m, vals=sz, cmap="viridis_r", clim=(hmin, hmax), lw=0.3)
        nb = m.check_grading(verbose=False)
        ax.set_title(f"{t}\n$h_{{\\min}}={sz.min():.3f}$, max/min {sz.max() / sz.min():.1f}", fontsize=7)
        rows.append((t, nb, sz.max() / sz.min(), sz.min()))
    cb = fig.colorbar(pc, ax=axs, location="right", fraction=0.02, pad=0.01); cb.set_label(r"element size $\sqrt{A}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, "ch02_grade_meshes.pdf"), bbox_inches="tight"); plt.close(fig)
    num("cmeGmUniNb", f"{rows[0][1]:.2f}"); num("cmeGmUniSize", f"{rows[0][2]:.1f}")
    for k, (name, r) in zip(("A", "B", "C"), zip(("1.1", "1.2", "1.5"), rows[1:])):
        num(f"cmeGm{k}Nb", f"{r[1]:.2f}"); num(f"cmeGm{k}Size", f"{r[2]:.1f}"); num(f"cmeGm{k}Hmin", f"{r[3]:.3f}")
    num("cmeGmUniHmin", f"{rows[0][3]:.3f}")
    print("grade meshes", rows)


def fig_plan():
    from fea_engine import grading as G
    feats = [G.Hole(center=(1.2, 1.0), radius=0.3, n_ring=12),
             G.Fillet(corner=(2.0, 1.0), radius=0.3, n_ring=8),
             G.Notch(path=[(0.8, 1.0), (1.2, 1.0)], depth=0.6, n_ring=6),
             G.EdgeBias(axis=0, dense_at="start", h_first=0.02, growth_ratio=1.2)]
    P = G.plan(feats, h_far=0.3, growth_ratio=1.2)
    P2 = G.plan(feats, h_far=0.3, growth_ratio=1.8)
    rows = []
    for f in feats:
        p = P.grade_for(f); n = getattr(f, "n_ring", 20)
        real = G.growth_ratio_of_partition(G.graded_partition(0.0, 1.0, n, p, getattr(f, "dense_at", "start")))
        rows.append((type(f).__name__, n, f.h_min, p, real))
    with open(os.path.join(TAB, "ch02_plan.tex"), "w") as fh:
        fh.write("% generated by scripts/ch02_figs.py\n\\begin{tabular}{@{}l r r r r@{}}\n\\toprule\nFeature & $n$ & $h_{\\min}$ & exponent $p$ & realised ratio \\\\\n\\midrule\n")
        for nme, n, h, p, r in rows:
            fh.write(f"\\texttt{{{nme}}} & {n} & {h:.3f} & {p:.2f} & {r:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    closed = max(abs(G.grade_for_growth_ratio(n_, t_) - np.log2(1 + t_)) for n_ in (6, 12, 20) for t_ in (1.1, 1.2, 1.5, 2.0))
    num("cmeGradeClosedDiff", sci(closed)); print("closed-form p = log2(1+rho) max diff", closed)
    num("cmePlanWarnLow", str(len(P.warnings))); num("cmePlanWarnHigh", str(len(P2.warnings)))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    tg = np.linspace(1.05, 2.0, 40)
    for n, col in zip((6, 12, 20), (GREEN, BLUE, ORANGE)):
        ax.plot(tg, [G.grade_for_growth_ratio(n, t) for t in tg], color=col, label=f"$n={n}$")
    ax.axvline(1.2, color="0.6", ls=":", lw=0.8); ax.axvline(1.5, color=RED, ls=":", lw=0.8)
    ax.text(1.51, 1.1, "warning\nthreshold", fontsize=6, color=RED)
    ax.set_xlabel("target growth ratio", fontsize=8); ax.set_ylabel("exponent $p$ returned", fontsize=8)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25); ax.tick_params(labelsize=7); ax.set_title("(a)", fontsize=8)
    ax = axs[1]
    n = 12; y = 0
    for lab, p in [("uniform", 1.0)] + [(f"target {t}", G.grade_for_growth_ratio(n, t)) for t in (1.1, 1.2, 1.5, 2.0)]:
        x = G.graded_partition(0.0, 1.0, n, p)
        ax.plot(x, np.full_like(x, y), "|", ms=10, color=BLUE)
        ax.text(1.03, y, f"{lab}, $p={p:.2f}$", fontsize=6.5, va="center"); y -= 1
    ax.set_xlim(-0.03, 1.5); ax.set_ylim(y + 0.2, 0.8); ax.axis("off"); ax.set_title("(b) the 12-element partition of a hole ring", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_plan.pdf")); plt.close(fig)
    print("plan rows", rows, "warnings", len(P.warnings), len(P2.warnings))


def _Kt(nr, nth, p, a=2.0, b=1.0, R=0.3):
    m = M.rectangle_with_hole_mesh_quarter(a, b, R, nr, nth, p)
    s = FESystem(m, Quad4PlaneStress()); s.assemble_stiffness(D_plane_stress(Material(E=1.0, nu=0.3, rho=1.0)), thickness=1.0)
    s.fix_dofs(m.select_nodes(x=0.0), [0]); s.fix_dofs(m.select_nodes(y=0.0), [1])
    s.fix_dofs(m.select_nodes(x=a), [0], value=0.01)
    U = s.solve_static()
    st = np.asarray(s.stress(U)).reshape(-1, 3)
    i = m.select_nodes(x=0.0, y=R)[0]
    rx = np.asarray(s.reactions(U)).reshape(-1, 2)[m.select_nodes(x=a), 0].sum()
    return st[i, 0] / (rx / b), len(m.elements), m


def fig_payoff():
    from fea_engine import grading as G
    nrs = [4, 8, 12, 16, 24, 32]
    ref, nref, _ = _Kt(40, 80, 2.0)
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    strat = [("uniform ($p=1$)", lambda n: 1.0, "0.45"), ("ratio-limited (target 1.2)", lambda n: G.grade_for_growth_ratio(n, 1.2), BLUE),
             ("$p=2$", lambda n: 2.0, ORANGE)]
    curves = {}
    for lab, pf, col in strat:
        e = []
        for n in nrs:
            k, ne, _ = _Kt(n, 2 * n, pf(n)); e.append((ne, abs(k / ref - 1), k))
        curves[lab] = (e, col)
    for ax, (lab, pf, col) in zip(axs[:3], strat):
        m = M.rectangle_with_hole_mesh_quarter(2.0, 1.0, 0.3, 8, 16, pf(8))
        _fill2d(ax, m, lw=0.3)
        ax.set_title(f"{lab}\n{m.check_grading(verbose=False):.2f} max neighbour ratio, $n_r=8$", fontsize=6.5)
    ax = axs[3]
    for lab, (e, col) in curves.items():
        ax.loglog([x[0] for x in e], [x[1] for x in e], "o-", ms=3.5, color=col, label=lab)
    ax.set_xlabel("elements", fontsize=7.5); ax.set_ylabel(r"error of $K_t$ at the hole", fontsize=7.5)
    ax.legend(fontsize=5.8, frameon=False); ax.grid(alpha=0.25, which="both"); ax.tick_params(labelsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_payoff.pdf"), bbox_inches="tight"); plt.close(fig)
    num("cmePayRef", f"{ref:.3f}"); num("cmePayRefEl", str(nref))
    for k, (lab, (e, col)) in zip(("U", "R", "P"), curves.items()):
        num(f"cmePay{k}Err", f"{100 * e[3][1]:.1f}"); num(f"cmePay{k}El", str(e[3][0]))   # nr=16
        num(f"cmePay{k}ErrFine", f"{100 * e[-1][1]:.1f}")
    print("payoff ref", ref, {lab: [(x[0], round(x[2], 4)) for x in e] for lab, (e, c) in curves.items()})


# ---------------------------------------------------------------------------------------------- Sec. 2.6 multi-element
def _mb(shear=0.0, duplicate=False, nx=8, ny=4, Lx=2.0, Ly=1.0, seed=2):
    xs, ys = np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    jj = np.arange(len(nodes)) // (nx + 1); ii = np.arange(len(nodes)) % (nx + 1)
    rng = np.random.default_rng(seed)
    interior = [j * (nx + 1) + i for j in range(1, ny) for i in range(1, nx) if i != nx // 2]
    nodes[interior] += rng.uniform(-0.2, 0.2, (len(interior), 2)) * np.array([Lx / nx, Ly / ny])
    nodes[ii == nx // 2, 0] += shear * (nodes[ii == nx // 2, 1] - Ly / 2)
    quads, tris = [], []
    for j in range(ny):
        for i in range(nx):
            n0, n1, n2, n3 = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i
            if i < nx // 2:
                quads.append([n0, n1, n2, n3])
            else:
                tris.append([n0, n2, n3]); tris.append([n0, n1, n2])
    quads, tris = np.array(quads), np.array(tris)
    shared = np.array(sorted(set(quads.ravel()) & set(tris.ravel())))
    if duplicate:
        mp = {int(n): len(nodes) + k for k, n in enumerate(shared)}
        nodes = np.vstack([nodes, nodes[shared]])
        tris = np.vectorize(lambda n: mp.get(int(n), int(n)))(tris)
    return M.MultiBlockMesh(nodes=nodes, blocks={"quad4": quads, "tri3": tris}, dim=2), shared, Lx, Ly


def _mb_patch(mb, Lx, Ly):
    elems = {"quad4": Quad4PlaneStress(), "tri3": Tri3PlaneStress()}
    D = D_plane_stress(Material(E=2.1e5, nu=0.3, rho=1.0))
    s = FESystem(mb, elems); s.assemble_stiffness({"quad4": D, "tri3": D}, thickness=1.0)
    G_ = np.array([[1e-3, 2e-4], [3e-4, -5e-4]]); bv = np.array([1e-3, -2e-3])
    X = mb.nodes; exact = X @ G_.T + bv
    on_b = np.where((X[:, 0] < 1e-9) | (X[:, 0] > Lx - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > Ly - 1e-9))[0]
    for n in on_b:
        for d in range(2):
            s.fix_dofs([int(n)], [d], value=float(exact[n, d]))
    U = np.asarray(s.solve_static()).reshape(-1, 2)
    err = np.abs(U - exact).max() / np.abs(exact).max()
    return err, U - exact


def fig_multiblock():
    variants = [("(a) vertical interface, shared nodes", dict()), ("(b) slanted interface, shared nodes", dict(shear=0.25)),
                ("(c) interface nodes duplicated (not shared)", dict(duplicate=True))]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.3))
    errs = []
    for ax, (t, kw) in zip(axs, variants):
        mb, shared, Lx, Ly = _mb(**kw)
        ok = mb.check_quality({"quad4": Quad4PlaneStress(), "tri3": Tri3PlaneStress()}, verbose=False)
        err, d = _mb_patch(mb, Lx, Ly); errs.append(err)
        for name, col in (("quad4", "#cfe0f1"), ("tri3", "#fbe2c4")):
            for e in mb.blocks[name]:
                ax.add_patch(Polygon(mb.nodes[e], closed=True, fc=col, ec="0.35", lw=0.5))
        ax.plot(mb.nodes[shared, 0], mb.nodes[shared, 1], "o", ms=4, color=RED, zorder=3)
        if kw.get("duplicate"):
            n0 = len(mb.nodes) - len(shared)
            ax.plot(mb.nodes[n0:, 0] + 0.03, mb.nodes[n0:, 1], "s", ms=3, color=GREEN, zorder=3)
        ax.set_xlim(-0.05, Lx + 0.05); ax.set_ylim(-0.05, Ly + 0.05); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{t}\npatch error {err:.1e}", fontsize=6.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_multiblock.pdf"), bbox_inches="tight"); plt.close(fig)
    num("cmeMBVertErr", sci(errs[0])); num("cmeMBSlantErr", sci(errs[1])); num("cmeMBDupErr", f"{errs[2]:.2f}")
    print("multiblock errs", errs)


# ---------------------------------------------------------------------------------------------- Sec. 2.7 parts
def fig_parts():
    from fea_engine.grading import MeshGradingError  # noqa: F401
    R = 0.3
    q = M.rectangle_with_hole_mesh_quarter(1.5, 1.0, R, 6, 12)
    full = M.mirror_mesh(q, mirror_x=True, mirror_y=True)
    A = M.rectangle_mesh(1.0, 1.0, 5, 5); B = M.rectangle_mesh(1.0, 1.0, 5, 5, x0=1.0)
    W = M.weld_meshes(A, B)
    B3 = M.rectangle_mesh(1.0, 1.0, 3, 3, x0=1.0)
    W3 = M.weld_meshes(A, B3)
    hole = M.hole_in_rectangle_mesh(3.0, 2.0, (1.2, 1.0), R, 5, 12)
    ext = M.extrude_mesh(hole, 0.4, 2)
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.5), gridspec_kw={"width_ratios": [1.2, 1.35, 1.35, 1.2]})
    ax = axs[0]
    _fill2d(ax, full, fc="#fbe2c4", lw=0.3)
    _fill2d(ax, q, fc="#cfe0f1", lw=0.3)
    on = np.where((np.abs(full.nodes[:, 0]) < 1e-9) | (np.abs(full.nodes[:, 1]) < 1e-9))[0]
    ax.plot(full.nodes[on, 0], full.nodes[on, 1], ".", ms=2.5, color=RED, zorder=3)
    ax.set_title(f"mirror_mesh: {len(q.nodes)} nodes, {len(q.elements)} el.\n$\\rightarrow$ {len(full.nodes)} nodes, {len(full.elements)} el.", fontsize=6.5)
    ax = axs[1]
    _fill2d(ax, A, fc="#cfe0f1", lw=0.4); _fill2d(ax, B, fc="#fbe2c4", lw=0.4)
    iface = np.where(np.abs(W.nodes[:, 0] - 1.0) < 1e-9)[0]
    ax.plot(W.nodes[iface, 0], W.nodes[iface, 1], "o", ms=3, color=RED, zorder=3)
    ax.set_title(f"weld_meshes, matching: {len(A.nodes)}+{len(B.nodes)}\n$\\rightarrow$ {len(W.nodes)} nodes (6 merged)", fontsize=6.5)
    ax = axs[2]
    _fill2d(ax, A, fc="#cfe0f1", lw=0.4); _fill2d(ax, B3, fc="#fbe2c4", lw=0.4)
    xa = A.nodes[np.abs(A.nodes[:, 0] - 1.0) < 1e-9]; xb = B3.nodes[np.abs(B3.nodes[:, 0] - 1.0) < 1e-9]
    ax.plot(xa[:, 0], xa[:, 1], "o", ms=3, color=BLUE, zorder=3); ax.plot(xb[:, 0] + 0.012, xb[:, 1], "s", ms=3, color=ORANGE, zorder=3)
    ax.set_title(f"weld_meshes, 5 against 3 divisions:\n{len(A.nodes)}+{len(B3.nodes)} $\\rightarrow$ {len(W3.nodes)} nodes (only {len(A.nodes) + len(B3.nodes) - len(W3.nodes)} merged)", fontsize=6.5)
    ax = axs[3]
    ax.imshow(_render(ext, window=(420, 340), view=(0.8, -1.2, 1.0))); ax.axis("off")
    ax.set_title(f"extrude_mesh: {len(hole.elements)} quad4\n$\\rightarrow$ {len(ext.elements)} hex8, {len(ext.nodes)} nodes", fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "ch02_parts.pdf"), bbox_inches="tight", dpi=200); plt.close(fig)
    num("cmeMirQuarterNodes", str(len(q.nodes))); num("cmeMirFullNodes", str(len(full.nodes)))
    num("cmeMirQuarterEl", str(len(q.elements))); num("cmeMirFullEl", str(len(full.elements)))
    num("cmeMirAreaRatio", f"{_area(full) / _area(q):.12f}".rstrip("0").rstrip("."))
    num("cmeWeldA", str(len(A.nodes))); num("cmeWeldB", str(len(B.nodes))); num("cmeWeldOut", str(len(W.nodes)))
    num("cmeWeldMerged", str(len(A.nodes) + len(B.nodes) - len(W.nodes)))
    num("cmeWeldBadOut", str(len(W3.nodes))); num("cmeWeldBadMerged", str(len(A.nodes) + len(B3.nodes) - len(W3.nodes)))
    num("cmeWeldBadBnodes", str(len(B3.nodes)))
    num("cmeExtQuads", str(len(hole.elements))); num("cmeExtHex", str(len(ext.elements)))
    num("cmeExtNodes", str(len(ext.nodes)))
    ok = Hex8Solid3D(); pos = all(np.linalg.det(ok.shape_and_derivs(p)[1] @ ext.nodes[c]) > 0
                                  for c in ext.elements for p in gauss_product(2, 3)[0])
    num("cmeExtPos", "yes" if pos else "NO")
    sv = [np.linalg.det(Quad4PlaneStress().shape_and_derivs(p)[1] @ W.nodes[c]) > 0 for c in W.elements for p in gauss_product(2, 2)[0]]
    num("cmeWeldPos", "yes" if all(sv) else "NO")
    print("parts", len(q.nodes), len(full.nodes), _area(full) / _area(q), len(W.nodes), len(W3.nodes), pos)


# ---------------------------------------------------------------------------------------------- Sec. 2.8 sets
def _stray(m, Lx, Ly, c, R):
    """Boundary-mask nodes that are neither on the four straight edges nor on the hole circle: nodes of unshared edges
    in the interior of the domain."""
    mask = m.compute_boundary_mask(store_as=None); X = m.nodes
    ring = np.abs(np.hypot(X[:, 0] - c[0], X[:, 1] - c[1]) - R) < 1e-9
    lines = (np.abs(X[:, 0]) < 1e-9) | (np.abs(X[:, 0] - Lx) < 1e-9) | (np.abs(X[:, 1]) < 1e-9) | (np.abs(X[:, 1] - Ly) < 1e-9)
    return mask, ring, lines, int((mask & ~lines & ~ring).sum())


def fig_sets():
    R = 0.3
    Lx, Ly = 3.0, 2.0
    good = M.hole_in_rectangle_mesh(Lx, Ly, (1.2, 1.0), R, 5, 12)
    bad = M.hole_in_rectangle_mesh(Lx, Ly, (1.2, 0.9), R, 5, 12)
    m3 = M.box_with_hole_mesh(2.0, 1.0, R, 0.5, 5, 10, 2)
    mask3 = m3.compute_boundary_mask()
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    res = {}
    for ax, m, c, lab in ((axs[0], good, (1.2, 1.0), "(a) hole at (1.2, 1.0)"), (axs[1], bad, (1.2, 0.9), "(b) hole at (1.2, 0.9)")):
        mask, ring, lines, stray = _stray(m, Lx, Ly, c, R); X = m.nodes
        res[c] = (len(X), int(mask.sum()), int((mask & ring).sum()), int((mask & lines).sum()), stray)
        _fill2d(ax, m, lw=0.3)
        ax.plot(X[mask & lines, 0], X[mask & lines, 1], "o", ms=2.5, color=BLUE)
        ax.plot(X[mask & ring, 0], X[mask & ring, 1], "o", ms=2.5, color=GREEN)
        ax.plot(X[mask & ~lines & ~ring, 0], X[mask & ~lines & ~ring, 1], "o", ms=4, color=RED, zorder=3)
        ax.set_title(f"{lab}\n{int(mask.sum())} boundary nodes, {stray} stray", fontsize=7)
    ax = axs[2]
    ax.imshow(_render_points(m3, mask3)); ax.axis("off")
    ax.set_title(f"(c) 3-D box with a hole\n{int(mask3.sum())} of {len(m3.nodes)} nodes on the boundary", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_sets.pdf"), bbox_inches="tight", dpi=200); plt.close(fig)
    g = res[(1.2, 1.0)]; bd = res[(1.2, 0.9)]
    num("cmeBmNodes", str(g[0])); num("cmeBmBnd", str(g[1])); num("cmeBmRing", str(g[2])); num("cmeBmLines", str(g[3]))
    num("cmeBmStrayGood", str(g[4])); num("cmeBmStrayBad", str(bd[4])); num("cmeBmBadBnd", str(bd[1]))
    num("cmeBmBoxNodes", str(len(m3.nodes))); num("cmeBmBoxBnd", str(int(mask3.sum())))
    # conformity table over hole positions (ungraded and graded)
    rows = []
    for c in [(1.5, 1.0), (1.2, 1.0), (0.9, 1.0), (1.5, 0.9), (1.2, 0.9), (1.0, 0.7)]:
        u = M.hole_in_rectangle_mesh(Lx, Ly, c, R, 5, 12)
        gr = M.hole_in_rectangle_mesh_graded(Lx, Ly, c, R, 5, 12, h_far=0.3, growth_ratio=1.2)
        rows.append((c, _stray(u, Lx, Ly, c, R)[3], _stray(gr, Lx, Ly, c, R)[3]))
    with open(os.path.join(TAB, "ch02_conform.tex"), "w") as fh:
        fh.write("% generated by scripts/ch02_figs.py\n\\begin{tabular}{@{}l r r l@{}}\n\\toprule\nhole centre & stray (ungraded) & stray (graded) & conforming \\\\\n\\midrule\n")
        for c, a1, a2 in rows:
            fh.write(f"$({c[0]}, {c[1]})$ & {a1} & {a2} & {'yes' if a1 == a2 == 0 else 'no'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("conformity", rows)
    # annotated plot of the package: a plate with a hole, clamped left edge, load on the right edge
    m = good
    left = m.select_nodes(x=0.0); right = m.select_nodes(x=3.0); top = m.select_nodes(y=2.0)
    fig2, ax2 = M.plot_mesh_annotated(m, annotations=[
        dict(nodes=left, label=f"clamped, x=0 ({len(left)} nodes)", color=RED, hatch="left", connect=False),
        dict(nodes=right, label=f"traction, x=3 ({len(right)} nodes)", color=GREEN, arrow=(1, 0), connect=False),
        dict(nodes=top, label=f"free edge, y=2 ({len(top)} nodes)", color=ORANGE, marker="s", connect=False)], figsize=(5.2, 3.2))
    ax2.set_title("")
    fig2.savefig(os.path.join(FIG, "ch02_annot.pdf"), bbox_inches="tight"); plt.close(fig2)
    num("cmeAnnLeft", str(len(left))); num("cmeAnnRight", str(len(right))); num("cmeAnnTop", str(len(top)))


def _render_points(m, mask, window=(520, 380)):
    import pyvista as pv
    from fea_engine.visualization import to_pyvista
    pv.OFF_SCREEN = True
    g = to_pyvista(m)
    pl = pv.Plotter(off_screen=True, window_size=window); pl.set_background("white")
    pl.add_mesh(g, color="#eaf1f8", show_edges=True, edge_color="#999999", line_width=0.6)
    P = pv.PolyData(np.asarray(m.nodes)[mask])
    pl.add_mesh(P, color="#c0392b", point_size=5, render_points_as_spheres=True)
    pl.camera_position = [tuple(np.array(g.center) + np.array((0.8, -1.2, 1.0)) * g.length * 1.15), g.center, (0, 0, 1)]
    img = pl.screenshot(return_img=True); pl.close()
    return img


# ---------------------------------------------------------------------------------------------- Sec. 2.11 visualisation
def fig_viz():
    import pyvista as pv
    from fea_engine.material import D_solid3d
    from fea_engine.visualization import (plot_field, plot_deformed, slice_field, clip_field, threshold_field, to_pyvista)
    pv.OFF_SCREEN = True
    L = 10.0
    m = M.box_mesh(L, 1.0, 1.0, 20, 2, 2)
    mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
    s = FESystem(m, Hex8Solid3D()); s.assemble_stiffness(D_solid3d(mat))
    s.fix_dofs(m.select_nodes(x=0.0), [0, 1, 2])
    tip = m.select_nodes(x=L)
    s.add_nodal_force(tip, dof_index=2, total_force=-1000.0)
    U = s.solve_static()
    vm = s.von_mises(U)
    tmp = os.path.join(FIG, "_viz_tmp"); os.makedirs(tmp, exist_ok=True)
    W = (560, 300)
    paths = {}
    def snap(name, fn):
        path = os.path.join(tmp, name + ".png"); fn(path); paths[name] = path
    snap("mesh", lambda p: _save_mesh(m, p, W))
    snap("field", lambda p: plot_field(m, vm, title="von Mises stress (Pa)", window_size=W, screenshot=p, show=False, off_screen=True, cmap="viridis"))
    snap("deformed", lambda p: plot_deformed(m, U, scale="auto", field="displacement", window_size=W, screenshot=p, show=False, off_screen=True, cmap="viridis"))
    def _slice(p):
        sl = slice_field(m, vm, normal="x", origin=(L / 2, 0.5, 0.5))
        pl = pv.Plotter(off_screen=True, window_size=W); pl.set_background("white")
        g = to_pyvista(m); pl.add_mesh(g, style="wireframe", color="#999999", line_width=0.6)
        pl.add_mesh(sl, scalars=sl.array_names[0], cmap="viridis", show_edges=True, show_scalar_bar=False)
        pl.camera_position = [tuple(np.array(g.center) + np.array((0.7, -1.2, 0.6)) * g.length * 0.82), g.center, (0, 0, 1)]
        pl.add_text("slice_field at x = L/2", font_size=9, color="black"); pl.screenshot(p); pl.close()
        paths["_slice_n"] = sl.n_cells
    def _clip(p):
        cl = clip_field(m, vm, normal="y", origin=(L / 2, 0.5, 0.5))
        pl = pv.Plotter(off_screen=True, window_size=W); pl.set_background("white")
        g = to_pyvista(m)
        pl.add_mesh(cl, scalars=cl.array_names[0], cmap="viridis", show_edges=True, show_scalar_bar=False)
        pl.camera_position = [tuple(np.array(g.center) + np.array((0.7, -1.4, 0.8)) * g.length * 0.82), g.center, (0, 0, 1)]
        pl.add_text("clip_field, y < 0.5", font_size=9, color="black"); pl.screenshot(p); pl.close()
        paths["_clip_n"] = cl.n_cells
    def _thr(p):
        lim = 0.5 * float(np.max(np.asarray(vm)))
        th = threshold_field(m, vm, (lim, None), all_scalars=False)
        pl = pv.Plotter(off_screen=True, window_size=W); pl.set_background("white")
        g = to_pyvista(m)
        pl.add_mesh(g, style="wireframe", color="#bbbbbb", line_width=0.5)
        pl.add_mesh(th, scalars=th.array_names[0], cmap="viridis", show_edges=True, show_scalar_bar=False)
        pl.camera_position = [tuple(np.array(g.center) + np.array((0.7, -1.2, 0.6)) * g.length * 0.82), g.center, (0, 0, 1)]
        pl.add_text("threshold_field: a node above half the maximum", font_size=9, color="black"); pl.screenshot(p); pl.close()
        paths["_thr_n"] = th.n_cells
    snap("slice", _slice); snap("clip", _clip); snap("thr", _thr)
    from PIL import Image
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 6.0))
    names = [("mesh", "(a)"), ("field", "(b)"), ("deformed", "(c)"), ("slice", "(d)"), ("clip", "(e)"), ("thr", "(f)")]
    for ax, (n, lab) in zip(axs.ravel(), names):
        im = np.asarray(Image.open(paths[n]).convert("RGB"))
        nz = np.where((im < 245).any(2))
        im = im[max(nz[0].min() - 6, 0):nz[0].max() + 6, max(nz[1].min() - 6, 0):nz[1].max() + 6]
        ax.imshow(im); ax.axis("off")
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(os.path.join(FIG, "ch02_viz.pdf"), bbox_inches="tight", dpi=200); plt.close(fig)
    import shutil; shutil.rmtree(tmp)
    uz = np.asarray(U).reshape(-1, 3)[tip, 2].mean()
    num("cmeVizTip", f"{abs(uz):.3e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}" if False else f"{abs(uz) * 1000:.4f}")
    num("cmeVizNodes", str(len(m.nodes))); num("cmeVizElems", str(len(m.elements)))
    num("cmeVizVmMax", f"{np.max(np.asarray(vm)) / 1e3:.1f}")
    num("cmeVizSliceCells", str(paths["_slice_n"])); num("cmeVizClipCells", str(paths["_clip_n"])); num("cmeVizThrCells", str(paths["_thr_n"]))
    print("viz", uz, paths["_slice_n"], paths["_clip_n"], paths["_thr_n"])


def _save_mesh(m, path, W):
    import pyvista as pv
    from fea_engine.visualization import to_pyvista
    g = to_pyvista(m)
    pl = pv.Plotter(off_screen=True, window_size=W); pl.set_background("white")
    pl.add_mesh(g, color="#cfe0f1", show_edges=True, edge_color="#333333", line_width=0.8)
    pl.camera_position = [tuple(np.array(g.center) + np.array((0.7, -1.2, 0.6)) * g.length * 0.82), g.center, (0, 0, 1)]
    pl.add_text("to_pyvista: the mesh as an unstructured grid", font_size=9, color="black"); pl.screenshot(path); pl.close()


# ---------------------------------------------------------------------------------------------- Sec. 2.12 dispatcher
def fig_dispatch():
    from fea_engine.build_mesh import build_mesh, RectangleWithHole
    from fea_engine.grading import MeshGradingError
    from fea_engine.elements.plates import Quad4MindlinPlate
    plain = build_mesh(RectangleWithHole(3.0, 2.0), 0.15, Quad4MindlinPlate)
    holed = build_mesh(RectangleWithHole(3.0, 2.0, hole_center=(1.2, 1.0), hole_radius=0.3, n_ring=12), 0.15, Quad4MindlinPlate, nr=8)
    msg = ""
    try:
        build_mesh(RectangleWithHole(3.0, 2.0, hole_center=(1.2, 1.0), hole_radius=0.3), 0.15, Quad4PlaneStress)
    except MeshGradingError as e:
        msg = str(e)
    plain_t = build_mesh(RectangleWithHole(3.0, 2.0), 0.15, Quad4PlaneStress)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    _fill2d(axs[0], plain, lw=0.3); axs[0].set_title(f"no feature, any element\nrectangle_mesh: {len(plain.elements)} el.", fontsize=7)
    _fill2d(axs[1], holed, lw=0.3); axs[1].set_title(f"hole + Mindlin plate\nstructured, graded: {len(holed.elements)} el.", fontsize=7)
    ax = axs[2]; ax.axis("off")
    ax.text(0.0, 0.95, "hole + Quad4PlaneStress:", fontsize=7, va="top", weight="bold")
    ax.text(0.0, 0.80, "raises MeshGradingError\n(\"the unstructured grading path\nwas removed from this package ...\")",
            fontsize=6.5, va="top", family="monospace", color=RED)
    ax.set_title("refused combinations", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_dispatch.pdf"), bbox_inches="tight"); plt.close(fig)
    num("cmeDispPlainEl", str(len(plain.elements))); num("cmeDispHoleEl", str(len(holed.elements)))
    num("cmeDispPlainTransEl", str(len(plain_t.elements))); num("cmeDispErrRaised", "yes" if msg else "NO")
    print("dispatch", len(plain.elements), len(holed.elements), bool(msg))


STEPS = {"step1": [fig_anatomy, fig_elements, fig_quality],
         "step2": [fig_gallery2d, fig_gallery3d, fig_arc, fig_grade_meshes, fig_plan, fig_payoff, fig_multiblock],
         "step3": [fig_parts, fig_sets, fig_viz, fig_dispatch]}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    funcs = []
    for k in (["step1", "step2", "step3"] if which == "all" else [which]):
        funcs += STEPS.get(k, [])
    for fn in funcs:
        print("running", fn.__name__, flush=True); fn()
    path = os.path.join(TAB, "numbers_ch02_figs.tex")
    old = {}
    if os.path.exists(path):
        import re
        for ln in open(path):
            mt = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}", ln)
            if mt:
                old[mt.group(1)] = mt.group(2)
    old.update(NUM)
    with open(path, "w") as fh:
        fh.write("% generated by scripts/ch02_figs.py -- do not edit\n")
        for k, v in old.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(old), "macros")


if __name__ == "__main__":
    main()

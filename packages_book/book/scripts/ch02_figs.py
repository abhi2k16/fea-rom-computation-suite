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


STEPS = {"step1": [fig_anatomy, fig_elements, fig_quality]}


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

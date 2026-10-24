# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch02_import_assets.py -- table and number macros of the end-to-end import example of Chapter 2 (an Abaqus mesh to a result file).

The foreign file is written by fea_engine/examples/import_abaqus_mesh_demo.py (plain text, as a pre-processor would; not by meshio or
fea_engine). The checks:
  1  import, solve, and compare with the same beam built natively with box_mesh
  2  the same file with shuffled and offset node and element numbers
  3  reading with the wrong dim
  4  a wedge element (C3D6) in the file, which the package has no formulation for
  5  an element with reversed node order, which Mesh.check_quality must report
Writes into book/: tables/ch02_import.tex; tables/numbers_ch02_import.tex.
Run from the repository root:
    PYTHONPATH=fea_engine/src python packages_book/book/scripts/ch02_import_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import io
import contextlib
import os
import re
import sys
import tempfile
import warnings
import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(BOOK, "..", ".."))
TAB = os.path.join(BOOK, "tables")
sys.path.insert(0, os.path.join(ROOT, "fea_engine", "examples"))
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


from fea_engine import mesh as M, mesh_io                                    # noqa: E402
from fea_engine.elements import Hex8Solid3D                                  # noqa: E402
import import_abaqus_mesh_demo as demo                                       # noqa: E402

L, H, NX, NY, NZ = 10.0, 1.0, 20, 2, 2


def tip_and_U(mesh):
    s, U, tip = demo.solve_imported(mesh)
    return s, U, float(U[3 * tip + 2].mean())


def keyed(mesh, U):
    """displacement vector sorted by node coordinates, to compare models whose node numbering differs"""
    order = np.lexsort(np.round(mesh.nodes, 9).T[::-1])
    return U.reshape(-1, 3)[order]


def rewrite_numbers(text, offset=1000, seed=0):
    """Shuffle the order of the *NODE lines and renumber nodes and elements with an offset (node and element sets are dropped)."""
    lines = text.split("\n")
    i0 = lines.index("*NODE") + 1
    i1 = next(i for i, l in enumerate(lines) if l.startswith("*ELEMENT"))
    nodes = lines[i0:i1]
    perm = np.random.default_rng(seed).permutation(len(nodes))
    new_id = {int(l.split(",")[0]): offset + int(p) for l, p in zip(nodes, perm)}
    shuffled = [None] * len(nodes)
    for l, p in zip(nodes, perm):
        f = l.split(",")
        shuffled[int(p)] = f"{offset + int(p)}," + ",".join(f[1:])
    j1 = next(i for i, l in enumerate(lines) if l.startswith("*NSET"))
    out = lines[:i0] + [shuffled[int(np.argsort(perm)[k])] if False else l for k, l in enumerate(shuffled)] + [lines[i1]]
    for l in lines[i1 + 1:j1]:
        f = [int(v) for v in l.split(",")]
        out.append(f"{5000 + f[0]}, " + ", ".join(str(new_id[v]) for v in f[1:]))
    return "\n".join(out) + "\n"


def figure(mesh, U, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    NAVY, RED = "#1b3a5c", "#c0392b"
    vm = np.asarray(s.von_mises(U)).ravel()
    X = mesh.nodes; D = U.reshape(-1, 3)
    face = np.where(np.abs(X[:, 1]) < 1e-9)[0]
    xf, zf = X[face, 0], X[face, 2]
    xs, zs = np.unique(np.round(xf, 9)), np.unique(np.round(zf, 9))
    grid = -np.ones((len(xs), len(zs)), int)
    for loc, (a, b) in enumerate(zip(xf, zf)):
        grid[np.argmin(abs(xs - a)), np.argmin(abs(zs - b))] = loc
    scale = 0.25 * L / np.abs(D[:, 2]).max() / 4
    tris, segs0, segs1 = [], [], []
    for i in range(len(xs) - 1):
        for k in range(len(zs) - 1):
            a, b, c, d = grid[i, k], grid[i + 1, k], grid[i + 1, k + 1], grid[i, k + 1]
            tris += [[a, b, c], [a, c, d]]
    P0 = np.c_[xf, zf]; P1 = np.c_[xf + scale * D[face, 0], zf + scale * D[face, 2]]
    def edges(P):
        out = []
        for i in range(len(xs)):
            out.append(P[grid[i, :]])
        for k in range(len(zs)):
            out.append(P[grid[:, k]])
        return out
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.0), gridspec_kw={"height_ratios": [0.55, 1.35, 1.0], "hspace": 0.35})
    # (a) the model as read
    ax = axs[0]
    ax.add_collection(LineCollection(edges(P0), colors=NAVY, linewidths=0.7)); ax.set_aspect("equal")
    ax.fill_between([-0.35, 0], -0.3, 1.3, color="0.8", hatch="////", edgecolor="0.5", lw=0)
    ax.annotate("", (L, 1.0), (L, 1.7), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6)); ax.text(L - 0.1, 1.55, "P = 1000 N (downward, over the face $x=L$)", color=RED, ha="right", va="center", fontsize=7.5)
    ax.text(-0.2, -0.05, "clamped $x=0$", fontsize=7.5, ha="center", va="top", color="0.3")
    ax.set_xlim(-0.6, L + 0.4); ax.set_ylim(-0.2, 1.9); ax.axis("off")
    ax.set_title("(a) The imported mesh (face $y=0$): 20 $\\times$ 2 element faces, 10 m $\\times$ 1 m", fontsize=8.5, fontweight="bold", color=NAVY, loc="left", pad=6)
    # (b) deformed, coloured
    ax = axs[1]
    tri = mtri.Triangulation(P1[:, 0], P1[:, 1], tris)
    tc = ax.tripcolor(tri, vm[face] / 1e3, shading="gouraud", cmap="viridis")
    ax.add_collection(LineCollection(edges(P1), colors="white", linewidths=0.5, alpha=0.6))
    ax.add_collection(LineCollection(edges(P0), colors="0.55", linewidths=0.6, linestyles="--"))
    ax.set_aspect("equal"); ax.set_xlim(-0.4, L + 0.4); ax.set_ylim(P1[:, 1].min() - 0.3, 1.3); ax.axis("off")
    cax = ax.inset_axes([0.03, 0.18, 0.27, 0.06]); cb = fig.colorbar(tc, cax=cax, orientation="horizontal"); cb.ax.set_title("nodal von Mises stress (kPa)", fontsize=7.5, pad=3); cb.ax.tick_params(labelsize=7)
    ax.set_title(f"(b) Deformed shape, magnified {scale:.0f} times (dashed grey: undeformed), coloured by von Mises stress", fontsize=8.5, fontweight="bold", color=NAVY, loc="left")
    # (c) tip deflection against Euler-Bernoulli along the axis
    ax = axs[2]
    mid = face[np.abs(X[face, 2] - 0.5 * H) < 1e-9] if np.any(np.abs(X[face, 2] - 0.5 * H) < 1e-9) else face
    order = np.argsort(X[mid, 0]); xm = X[mid, 0][order]; wm = D[mid, 2][order]
    EI = 2.1e11 * H ** 4 / 12.0; Pn = 1000.0
    xx = np.linspace(0, L, 200); web = -Pn * xx ** 2 * (3 * L - xx) / (6 * EI)
    ax.plot(xx, web * 1e3, color="0.45", lw=2.2, label="Euler--Bernoulli beam theory")
    ax.plot(xm, wm * 1e3, "o", color="#2f7fc1", ms=3.5, label="imported mesh, 8-node hexahedra")
    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("vertical displacement (mm)"); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("(c) Centre line: the mesh is stiffer than the beam, as the text explains (two elements through the depth)", fontsize=8.5, fontweight="bold", color=NAVY, loc="left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(BOOK, "figures", "ch02_import.pdf"), bbox_inches="tight"); plt.close(fig)
    num("cmiScale", f"{scale:.0f}")
    num("cmiVmMax", f"{vm.max() / 1e3:.1f}")


def main():
    rows = []
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "beam.inp")
        demo.write_foreign_inp(p, L=L, h=H, nx=NX, ny=NY, nz=NZ)
        text = open(p).read()
        num("cmiNsetLines", str(sum(1 for l in text.split("\n") if l.startswith("*") and not l.startswith("**"))))
        # --- 1 import, solve, compare with a native model
        mesh = mesh_io.read_mesh(p, dim=3)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = mesh.check_quality(Hex8Solid3D())
        s, U, w = tip_and_U(mesh)
        figure(mesh, U, s)
        nat = M.box_mesh(L, H, H, NX, NY, NZ)
        _, Un, wn = tip_and_U(nat)
        a, b = keyed(mesh, U), keyed(nat, Un)
        diff = float(np.abs(a - b).max() / np.abs(b).max())
        eb = -1000.0 * L ** 3 / (3 * 2.1e11 * H ** 4 / 12)
        num("cmiNodes", str(len(mesh.nodes))); num("cmiElems", str(len(mesh.elements)))
        num("cmiTip", f"{abs(w):.4e}"); num("cmiTipNative", f"{abs(wn):.4e}"); num("cmiDiffNative", sci(diff))
        num("cmiEbRatio", f"{w / eb:.3f}"); num("cmiEb", f"{abs(eb):.4e}")
        num("cmiQualityOk", "all positive" if ok else "inverted elements")
        rows.append(("1", "import, solve, compare with \\texttt{box\\_mesh}",
                     f"{len(mesh.nodes)} nodes, {len(mesh.elements)} elements; tip deflection {abs(w):.4e}~m; "
                     f"difference to the native model {sci(diff)}"))
        # --- 2 shuffled and offset numbering
        p2 = os.path.join(td, "beam2.inp")
        open(p2, "w").write(rewrite_numbers(text))
        mesh2 = mesh_io.read_mesh(p2, dim=3)
        _, U2, w2 = tip_and_U(mesh2)
        d2 = float(np.abs(keyed(mesh2, U2) - b).max() / np.abs(b).max())
        num("cmiDiffShuffle", sci(d2))
        rows.append(("2", "node lines shuffled, node ids offset by 1000, element ids by 5000",
                     f"{len(mesh2.nodes)} nodes, {len(mesh2.elements)} elements; difference to the native model {sci(d2)}"))
        # --- 3 wrong dim
        try:
            mesh_io.read_mesh(p, dim=2)
            msg = "no error"
        except Exception as e:
            msg = type(e).__name__
        num("cmiWrongDim", msg)
        rows.append(("3", "\\texttt{dim=2} for a file with only solid elements", f"\\texttt{{{msg}}}: no cell of that dimension"))
        # --- 4 a wedge in the file
        lines = text.split("\n")
        k = lines.index("*NSET, NSET=ROOT")
        wedge = ["*ELEMENT, TYPE=C3D6, ELSET=EXTRA", "9001, 1, 2, 3, 22, 23, 24"]
        p4 = os.path.join(td, "beam4.inp")
        open(p4, "w").write("\n".join(lines[:k] + wedge + lines[k:]))
        mesh4 = mesh_io.read_mesh(p4, dim=3)
        num("cmiWedgeRead", str(len(mesh4.elements)))
        rows.append(("4", "one wedge element (\\texttt{C3D6}) added to the file",
                     f"{len(mesh4.elements)} elements read, the wedge is skipped without a message"))
        # --- 5 reversed element
        i = next(i for i, l in enumerate(lines) if l.startswith("*ELEMENT")) + 1 + 5
        f = lines[i].split(", "); n = f[1:]
        n = n[4:] + n[:4]
        lines5 = list(lines); lines5[i] = f[0] + ", " + ", ".join(n)
        p5 = os.path.join(td, "beam5.inp")
        open(p5, "w").write("\n".join(lines5))
        mesh5 = mesh_io.read_mesh(p5, dim=3)
        with contextlib.redirect_stdout(io.StringIO()):
            ok5 = mesh5.check_quality(Hex8Solid3D())
        num("cmiInvertedFlag", "False" if not ok5 else "True")
        rows.append(("5", "one element with the two faces of its node list exchanged",
                     "\\texttt{check\\_quality} returns \\texttt{" + str(ok5) + "} and reports one inverted element"))
    with open(os.path.join(TAB, "ch02_import.tex"), "w") as fh:
        fh.write("\\begin{tabularx}{\\linewidth}{@{}c >{\\raggedright\\arraybackslash}p{0.3\\linewidth} X@{}}\n\\toprule\n"
                 "case & test & result \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabularx}\n")
    with open(os.path.join(TAB, "numbers_ch02_import.tex"), "w") as fh:
        fh.write("% generated by scripts/ch02_import_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    for k, v in NUM.items():
        print(k, v)


if __name__ == "__main__":
    main()

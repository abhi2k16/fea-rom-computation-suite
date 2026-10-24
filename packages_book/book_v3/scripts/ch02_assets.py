# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch02_assets.py -- figures, tables and number macros of Chapter 2 (mesh topologies and structural domain partitioning).

Writes into book/:
    figures/ch02_meshes.pdf       four generated meshes: graded hole, notch, fillet, quarter hole
    figures/ch02_grading.pdf      the power-law grading: realised growth ratio against the exponent, and the exponent chosen for a target
    tables/ch02_generators.tex    generators: nodes, elements, area or volume against the exact value, element quality
    tables/ch02_meshio.tex        round trip through meshio for three formats
    tables/numbers_ch02.tex       every number quoted in the prose

Also runs a patch test across a triangle/quadrilateral interface of a MultiBlockMesh (blocks joined by shared nodes).
Run from the repository root:   python packages_book/book/scripts/ch02_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import platform
import tempfile
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": False, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


from fea_engine import (FESystem, Material, D_plane_stress, Quad4PlaneStress, Tri3PlaneStress, Hex8Solid3D, mesh as M,   # noqa: E402
                        grading as G)
from fea_engine.mesh import MultiBlockMesh, Mesh                                                                           # noqa: E402
from fea_engine import mesh_io                                                                                             # noqa: E402


def area2d(m):
    X = m.nodes[m.elements]
    x, y = X[:, :, 0], X[:, :, 1]
    return float(0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)).sum())


def volume3d(m):
    """Sum of detJ integrated with the element's own 2x2x2 rule (exact for the trilinear map)."""
    from fea_engine.elements import gauss_product, jacobian
    el = Hex8Solid3D()
    pts, wts = gauss_product(2, 3)
    v = 0.0
    for conn in m.elements:
        xc = m.nodes[conn]
        for p, w in zip(pts, wts):
            _, dN = el.shape_and_derivs(p)
            v += np.linalg.det(dN @ xc) * w
    return float(v)


def quality(m, el):
    pts_ok = True
    from fea_engine.elements import gauss_product
    pts, _ = gauss_product(el.gauss_order, el.dim)
    dets = []
    for conn in m.elements:
        xc = m.nodes[conn]
        for p in pts:
            _, dN = el.shape_and_derivs(p)
            dets.append(np.linalg.det(dN @ xc))
    dets = np.array(dets)
    return bool((dets > 0).all()), float(dets.max() / dets.min())


# ------------------------------------------------------------------------------------------ generators
def exp_generators():
    a, b, R, Lz = 2.0, 1.0, 0.3, 0.5
    cases = []
    cases.append(("rectangle_mesh", "rectangle $2\\times1$", M.rectangle_mesh(2.0, 1.0, 20, 10), 2.0, "quad4"))
    cases.append(("rectangle_with_hole_mesh_quarter", "quarter plate, hole $R=0.3$", M.rectangle_with_hole_mesh_quarter(a, b, R, 8, 16), a * b - np.pi * R ** 2 / 4, "quad4"))
    cases.append(("hole_in_rectangle_mesh", "hole off centre, $R=0.3$", M.hole_in_rectangle_mesh(3.0, 2.0, (1.2, 1.0), R, 6, 16), 6.0 - np.pi * R ** 2, "quad4"))
    cases.append(("notch_in_rectangle_mesh", "notch $0.4\\times0.6$", M.notch_in_rectangle_mesh(2.0, 1.0, "top", 0.8, 1.2, 0.6, 8, 4, 8, 6, 6), 2.0 - 0.4 * 0.6, "quad4"))
    cases.append(("fillet_in_rectangle_mesh", "fillet $R=0.3$", M.fillet_in_rectangle_mesh(2.0, 1.0, (2.0, 1.0), R, 10, 6, 6), 2.0 - R ** 2 * (1 - np.pi / 4), "quad4"))
    rows = []
    for name, desc, m, exact, topo in cases:
        el = Quad4PlaneStress()
        ok, ratio = quality(m, el)
        A = area2d(m)
        rows.append((name, desc, len(m.nodes), len(m.elements), A, exact, abs(A / exact - 1), ok, ratio))
        print(f"{name}: {len(m.nodes)} nodes {len(m.elements)} el, area {A:.6f} vs {exact:.6f} ({abs(A/exact-1):.1e}), positive {ok}, detJ ratio {ratio:.1f}", flush=True)
    m3 = M.box_mesh(2.0, 1.0, 0.5, 8, 4, 2)
    V = volume3d(m3)
    ok3, r3 = quality(m3, Hex8Solid3D())
    rows.append(("box_mesh", "box $2\\times1\\times0.5$", len(m3.nodes), len(m3.elements), V, 1.0, abs(V - 1.0), ok3, r3))
    m3h = M.box_with_hole_mesh(a, b, R, Lz, 6, 12, 2)
    Vh = volume3d(m3h)
    exh = (2 * a * 2 * b - np.pi * R ** 2) * Lz
    ok3h, r3h = quality(m3h, Hex8Solid3D())
    rows.append(("box_with_hole_mesh", "box, through hole $R=0.3$", len(m3h.nodes), len(m3h.elements), Vh, exh, abs(Vh / exh - 1), ok3h, r3h))
    print(f"3D: box {V} ok {ok3}; box with hole {Vh:.5f} vs {exh:.5f} ({abs(Vh/exh-1):.1e}) ok {ok3h}", flush=True)
    with open(os.path.join(TAB, "ch02_generators.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.2cm}>{\\raggedright\\arraybackslash}p{3.0cm}rrrc@{}}\n\\toprule\ngenerator & case & nodes & elements & area/volume error & $\\det\\bm J>0$ \\\\\n\\midrule\n")
        for name, desc, nn, ne, A, ex, err, ok, ratio in rows:
            fh.write(f"\\code{{{name}}} & {desc} & {nn} & {ne} & {sci(err) if err > 0 else '0'} & {'yes' if ok else 'NO'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cmeGenAllPositive", "yes" if all(r[7] for r in rows) else "no")
    num("cmeGenNRows", str(len(rows)))
    num("cmeRectErr", sci(rows[0][6])); num("cmeHoleErr", sci(rows[1][6])); num("cmeHoleOffErr", sci(rows[2][6]))
    num("cmeNotchErr", sci(rows[3][6])); num("cmeFilletErr", sci(rows[4][6]))
    num("cmeBoxHoleErr", sci(rows[6][6]))
    # polygonal arc: area error against the number of elements around the quarter circle
    nth = [8, 16, 32, 64]
    ae = [abs(area2d(M.rectangle_with_hole_mesh_quarter(a, b, R, 8, k)) / (a * b - np.pi * R ** 2 / 4) - 1) for k in nth]
    order = -np.polyfit(np.log(nth), np.log(ae), 1)[0]
    print("arc area error", ae, "order", order, flush=True)
    num("cmeArcOrder", f"{order:.1f}"); num("cmeArcErrCoarse", sci(ae[0])); num("cmeArcErrFine", sci(ae[-1]))
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 5.2))
    gm = M.hole_in_rectangle_mesh_graded(3.0, 2.0, (1.2, 1.0), R, 6, 16, h_far=0.3)
    for a_, (m, t) in zip(ax.ravel(), [(gm, "hole, graded to a growth ratio"), (cases[3][2], "notch"),
                                       (cases[4][2], "fillet"), (cases[1][2], "quarter plate with hole")]):
        M.plot_mesh_2d(m, a_); a_.set_aspect("equal"); a_.set_title(t, fontsize=9); a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch02_meshes.pdf")); plt.close(fig)
    # realised growth of the graded hole's nodes isn't a single 1-D partition: report the element size range instead
    sizes = np.sqrt([abs(np.linalg.det(Quad4PlaneStress().shape_and_derivs((0.0, 0.0))[1] @ gm.nodes[c])) * 4 for c in gm.elements])
    num("cmeGradedSizeRatio", f"{sizes.max() / sizes.min():.1f}")


# ------------------------------------------------------------------------------------------ grading law
def exp_grading():
    n = 20
    ps = np.linspace(1.0, 4.0, 31)
    gr = [G.growth_ratio_of_partition(G.graded_partition(0.0, 1.0, n, p)) for p in ps]
    targets = [1.1, 1.2, 1.3, 1.5, 2.0]
    chosen = [G.grade_for_growth_ratio(n, t) for t in targets]
    real = [G.growth_ratio_of_partition(G.graded_partition(0.0, 1.0, n, p)) for p in chosen]
    for t, p, r in zip(targets, chosen, real):
        print(f"target growth {t}: grade {p:.3f}, realised {r:.4f}", flush=True)
    # the figure itself (figures/ch02_grading.pdf) is drawn by scripts/ch02_grading.py
    num("cmeGradeForTarget", f"{chosen[1]:.2f}"); num("cmeGradeRealTarget", f"{real[1]:.3f}")
    num("cmeGradeWorstOver", f"{max(r / t - 1 for r, t in zip(real, targets)) * 100:.2f}")
    num("cmeGradeRatioPTwo", f"{gr[np.argmin(abs(ps - 2.0))]:.2f}")
    num("cmeGradeRatioPOne", f"{gr[0]:.2f}")


# ------------------------------------------------------------------------------------------ meshio round trip
def exp_meshio():
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for label, mesh, dim, ext in [("quad4, 2-D", M.rectangle_mesh(2.0, 1.0, 12, 6), 2, "vtu"),
                                      ("quad4, 2-D", M.rectangle_mesh(2.0, 1.0, 12, 6), 2, "inp"),
                                      ("hex8, 3-D", M.box_mesh(2.0, 1.0, 0.5, 6, 3, 2), 3, "vtu"),
                                      ("hex8, 3-D", M.box_mesh(2.0, 1.0, 0.5, 6, 3, 2), 3, "inp")]:
            path = os.path.join(td, f"m.{ext}")
            try:
                mesh_io.write_mesh(mesh, path)
                back = mesh_io.read_mesh(path, dim=dim)
                nn_ok = back.nodes.shape == mesh.nodes.shape and np.allclose(back.nodes[:, :dim], mesh.nodes[:, :dim], atol=1e-12)
                ne_ok = np.array_equal(np.asarray(back.elements), np.asarray(mesh.elements))
                # compare as sets of corner-node coordinates, independent of node numbering
                def key(m):
                    return sorted(tuple(np.round(m.nodes[c][:, :dim], 9).ravel().tolist()) for c in m.elements)
                set_ok = key(back) == key(mesh)
                rows.append((label, ext, nn_ok, ne_ok, set_ok, ""))
            except Exception as e:
                rows.append((label, ext, False, False, False, f"{type(e).__name__}"))
            print("meshio", rows[-1], flush=True)
    with open(os.path.join(TAB, "ch02_meshio.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llccc}\n\\toprule\nmesh & format & same nodes & same connectivity array & same element set \\\\\n\\midrule\n")
        for label, ext, a, b, c, err in rows:
            f = lambda v: "yes" if v else ("error" if err else "no")
            fh.write(f"{label} & \\texttt{{.{ext}}} & {f(a)} & {f(b)} & {f(c)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cmeMeshioAllSet", "yes" if all(r[4] for r in rows) else "no")


# ------------------------------------------------------------------------------------------ multi-block patch test
def exp_multiblock():
    nx, ny, Lx, Ly = 8, 4, 2.0, 1.0
    xs, ys = np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    rng = np.random.default_rng(2)
    interior = [j * (nx + 1) + i for j in range(1, ny) for i in range(1, nx)]
    nodes[interior] += rng.uniform(-0.2, 0.2, (len(interior), 2)) * np.array([Lx / nx, Ly / ny])
    quads, tris = [], []
    for j in range(ny):
        for i in range(nx):
            n0, n1, n2, n3 = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i
            if i < nx // 2:
                quads.append([n0, n1, n2, n3])
            else:
                tris.append([n0, n1, n2]); tris.append([n0, n2, n3])
    mesh = MultiBlockMesh(nodes=nodes, blocks={"quad4": np.array(quads), "tri3": np.array(tris)}, dim=2)
    elems = {"quad4": Quad4PlaneStress(), "tri3": Tri3PlaneStress()}
    ok, ratio = mesh.check_quality(elems, verbose=False)[:2] if isinstance(mesh.check_quality(elems, verbose=False), tuple) else (True, 0)
    D = D_plane_stress(Material(E=2.1e5, nu=0.3, rho=1.0))
    s = FESystem(mesh, elems)
    s.assemble_stiffness({"quad4": D, "tri3": D}, thickness=1.0)
    G_ = np.array([[1e-3, 2e-4], [3e-4, -5e-4]]); bvec = np.array([1e-3, -2e-3])
    exact = nodes @ G_.T + bvec
    on_b = np.where((nodes[:, 0] < 1e-9) | (nodes[:, 0] > Lx - 1e-9) | (nodes[:, 1] < 1e-9) | (nodes[:, 1] > Ly - 1e-9))[0]
    for n in on_b:
        for d in range(2):
            s.fix_dofs([int(n)], [d], value=float(exact[n, d]))
    U = np.asarray(s.solve_static()).reshape(-1, 2)
    err = np.abs(U - exact).max() / np.abs(exact).max()
    # nodes shared by the two blocks
    shared = set(np.unique(np.array(quads))) & set(np.unique(np.array(tris)))
    print(f"multiblock: {len(quads)} quads + {len(tris)} triangles, {len(shared)} shared nodes, patch error {err:.1e}", flush=True)
    num("cmeMBQuads", str(len(quads))); num("cmeMBTris", str(len(tris))); num("cmeMBShared", str(len(shared)))
    num("cmeMBErr", sci(err))


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_generators(); exp_grading(); exp_meshio(); exp_multiblock()
    num("cmeMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch02.tex"), "w") as fh:
        fh.write("% generated by scripts/ch02_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

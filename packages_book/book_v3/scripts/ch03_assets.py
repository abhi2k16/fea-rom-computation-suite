# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch03_assets.py -- figures, tables and number macros of Chapter 3 (linear continua and structural elements).

Writes into book/:
    figures/ch03_bending.pdf       pure bending of a plane-stress beam: tip-deflection ratio against mesh size, four elements
    figures/ch03_beams.pdf         produced by ch03_beams.py (beam checks)
    tables/ch03_patch.tex          patch tests on distorted meshes, nine continuum elements
    tables/ch03_locking.tex        cantilever tip deflection of the hexahedra, nu = 0.3 and 0.4999
    tables/ch03_plate.tex          Mindlin plate cantilever strip: selective-reduced against full integration, and the shell
    tables/numbers_ch03.tex        every number quoted in the prose

The helpers of fea_engine/tests (patch test, pure bending, locking cantilever) are imported, so the book quotes exactly what
the tests check. Reference values are analytic, never produced by fea_engine. Run from the repository root:

    python packages_book/book/scripts/ch03_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import platform
import warnings
import importlib.util
import numpy as np
import scipy.linalg as sl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(BOOK, "..", ".."))
TESTS = os.path.join(ROOT, "fea_engine", "tests")
sys.path.insert(0, TESTS)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from fea_engine import (FESystem, Material, D_solid3d, D_plane_stress, Quad4Laplace, Quad4MindlinPlate, Shell4MITC,   # noqa: E402
                        Beam2DEulerBernoulli, Beam2DReissner, Hex8Solid3D, Hex8SolidBbar, Hex20Solid3D, convergence as cv)
from fea_engine.geometry import generate_mesh                                                                        # noqa: E402
from fea_engine.material import EI_beam, Section, D_mindlin_plate, D_shell                                           # noqa: E402
from fea_engine.mesh import Mesh, rectangle_mesh, box_mesh                                                           # noqa: E402
from fea_engine import mesh as meshlib                                                                               # noqa: E402

T = load(os.path.join(TESTS, "test_benchmarks_convergence.py"), "bench_helpers")
HM = load(os.path.join(TESTS, "hex_meshes.py"), "hex_meshes")


# ------------------------------------------------------------------------------------------ patch tests
def exp_patch():
    classes = list(T.ELEMENTS)
    rows = []
    for cls in classes:
        u, s = T._patch(cls)
        rows.append((cls.__name__, u, s))
        print(f"patch {cls.__name__}: {u:.1e} {s:.1e}", flush=True)
    # Hex8SolidBbar: same mesh as Hex8Solid3D
    try:
        T.ELEMENTS.append(Hex8SolidBbar)
        u, s = T._patch(Hex8SolidBbar)
        rows.insert(5, (Hex8SolidBbar.__name__, u, s))
        print(f"patch Hex8SolidBbar: {u:.1e} {s:.1e}", flush=True)
    except Exception as e:                                   # report instead of hiding
        print("patch Hex8SolidBbar failed:", repr(e), flush=True)
    finally:
        T.ELEMENTS.remove(Hex8SolidBbar) if Hex8SolidBbar in T.ELEMENTS else None
    with open(os.path.join(TAB, "ch03_patch.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrr}\n\\toprule\nelement & displacement error & stress error \\\\\n\\midrule\n")
        for n, u, s in rows:
            fh.write(f"\\code{{{n}}} & {sci(u)} & {sci(s)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cecPatchWorst", sci(max(max(r[1], r[2]) for r in rows)))
    num("cecPatchCount", str(len(rows)))


# ------------------------------------------------------------------------------------------ pure bending
def exp_bending():
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    res = {}
    for cls, ns_ in [(T.Quad4PlaneStress, [4, 8, 16, 32]), (T.Tri3PlaneStress, [4, 8, 16, 32]),
                     (T.Quad8PlaneStress, [4, 8]), (T.Tri6PlaneStress, [4, 8])]:
        r = np.array([T._bending_ratio(n, cls) for n in ns_])
        res[cls.__name__] = (ns_, r)
        print(f"bending {cls.__name__}: {r}", flush=True)
    for name, (ns_, r) in res.items():
        quad = name in ("Quad8PlaneStress", "Tri6PlaneStress")
        ax.semilogx(ns_, r, "o-" if not quad else "s--", label=name)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("elements along the beam"); ax.set_ylabel("tip deflection / exact (pure bending)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch03_bending.pdf")); plt.close(fig)
    q4 = res["Quad4PlaneStress"][1]; t3 = res["Tri3PlaneStress"][1]
    num("cecBendQuadFour", f"{q4[0]:.3f}"); num("cecBendQuadThirtytwo", f"{q4[-1]:.3f}")
    num("cecBendTriFour", f"{t3[0]:.3f}"); num("cecBendTriThirtytwo", f"{t3[-1]:.3f}")
    num("cecBendQuadEightErr", sci(max(abs(res["Quad8PlaneStress"][1] - 1))))
    num("cecBendTriSixErr", sci(max(abs(res["Tri6PlaneStress"][1] - 1))))
    h = 1.0 / np.array(res["Quad4PlaneStress"][0][1:])
    num("cecBendQuadOrder", f"{cv.observed_order(h, np.abs(1 - q4[1:])):.1f}")


# ------------------------------------------------------------------------------------------ locking of hexahedra
def cantilever_ratio(cls, nu, dims, quadratic=False):
    L3, h, w, P = 1.0, 0.05, 0.05, -1000.0
    mesh = box_mesh(L3, w, h, *dims)
    if quadratic:
        mesh = HM.hex20_from_hex8(mesh)
    mat = Material(E=2.1e11, nu=nu, rho=7850.0)
    tip = mesh.nodes_on_plane(axis=0, value=L3)
    fix = mesh.nodes_on_plane(axis=0, value=0.0)
    c = tip[np.argmin(np.abs(mesh.nodes[tip, 1] - w / 2) + np.abs(mesh.nodes[tip, 2] - h / 2))]
    s = FESystem(mesh, cls())
    s.assemble_stiffness(D_solid3d(mat))
    s.add_nodal_force(tip, dof_index=2, total_force=P)
    s.fix_dofs(fix, [0, 1, 2])
    U = s.solve_static()
    return U[3 * c + 2] / (P * L3 ** 3 / (3 * mat.E * w * h ** 3 / 12)), s.n_dof


def exp_locking():
    rows = []
    for cls, dims, quad in [(Hex8Solid3D, (20, 4, 4), False), (Hex8SolidBbar, (20, 4, 4), False), (Hex20Solid3D, (10, 2, 2), True)]:
        r3, nd = cantilever_ratio(cls, 0.3, dims, quad)
        r5, _ = cantilever_ratio(cls, 0.4999, dims, quad)
        rows.append((cls.__name__, "x".join(map(str, dims)), nd, r3, r5))
        print(f"locking {cls.__name__} {dims}: {r3:.4f} {r5:.4f}", flush=True)
    with open(os.path.join(TAB, "ch03_locking.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\nelement & mesh & dofs & $\\nu=0.3$ & $\\nu=0.4999$ \\\\\n\\midrule\n")
        for n, m, nd, a, b in rows:
            fh.write(f"\\code{{{n}}} & ${m.replace('x', chr(92) + 'times ')}$ & {nd} & {a:.3f} & {b:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    names = {"Hex8Solid3D": "HexPlain", "Hex8SolidBbar": "HexBbar", "Hex20Solid3D": "HexTwenty"}
    for n, m, nd, a, b in rows:
        num(f"cecLock{names[n]}A", f"{a:.2f}"); num(f"cecLock{names[n]}B", f"{b:.2f}")


# ------------------------------------------------------------------------------------------ beams
def exp_plate():
    mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
    h, L, W, P = 0.01, 1.0, 0.2, -1000.0
    Db, Ds = D_mindlin_plate(mat, h)
    D_sh = D_shell(mat, h)
    Dpl = mat.E * h ** 3 / (12 * (1 - mat.nu ** 2))
    ref = P * L ** 3 / (3 * Dpl * W)                    # strip of plate rigidity: an approximate reference (no anticlastic effect, no shear)
    rows = []
    for nx in (4, 8, 16, 32, 64):
        ny = max(2, nx // 4)
        mesh = rectangle_mesh(L, W, nx, ny)
        left = mesh.nodes_on_plane(axis=0, value=0.0); tip = mesh.nodes_on_plane(axis=0, value=L)
        out = []
        for integ in ("sri", "full"):
            s = FESystem(mesh, Quad4MindlinPlate())
            s.assemble_stiffness((Db, Ds), integration=integ)
            s.fix_dofs(left, [0, 1, 2]); s.add_nodal_force(tip, 0, P)
            out.append(s.solve_static()[3 * tip[0]] / ref)
        nodes3 = np.column_stack([mesh.nodes, np.zeros(len(mesh.nodes))])
        ms = Mesh(nodes3, mesh.elements, dim=3)
        sh = FESystem(ms, Shell4MITC(), sparse=False)
        sh.assemble_stiffness(D_sh); sh.fix_dofs(left, [0, 1, 2, 3, 4, 5]); sh.add_nodal_force(tip, 2, P)
        out.append(sh.solve_static()[6 * tip[0] + 2] / ref)
        rows.append((nx, *out))
        print(f"plate nx={nx}: sri {out[0]:.4f} full {out[1]:.4f} mitc {out[2]:.4f}", flush=True)
    with open(os.path.join(TAB, "ch03_plate.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrr}\n\\toprule\nelements along $L$ & \\code{Quad4MindlinPlate} (SRI) & same, full integration & \\code{Shell4MITC} \\\\\n\\midrule\n")
        for nx, a, b, c in rows:
            fh.write(f"{nx} & {a:.3f} & {b:.3f} & {c:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cecPlateSriFine", f"{rows[-1][1]:.3f}"); num("cecPlateFullCoarse", f"{rows[0][2]:.3f}")
    num("cecPlateFullFine", f"{rows[-1][2]:.3f}"); num("cecPlateShellFine", f"{rows[-1][3]:.3f}")
    num("cecPlateSriCoarse", f"{rows[0][1]:.3f}"); num("cecPlateShellCoarse", f"{rows[0][3]:.3f}")
    num("cecPlateRatioFS", f"{rows[-1][3] / rows[-1][1]:.4f}")


# ------------------------------------------------------------------------------------------ element table
def exp_table():
    from fea_engine import elements as el
    rows = []
    for fam, names in [
                       ("plane", ["Tri3PlaneStress", "Tri6PlaneStress", "Quad4PlaneStress", "Quad8PlaneStress"]),
                       ("solid", ["Tet4Solid3D", "Tet10Solid3D", "Hex8Solid3D", "Hex8SolidBbar", "Hex20Solid3D"]),
                       ("beam", ["Beam2DEulerBernoulli", "Beam2DReissner", "Beam3DEulerBernoulli"]),
                       ("plate/shell", ["Quad4MindlinPlate", "Shell4MITC"])]:
        for n in names:
            c = getattr(el, n)
            rows.append((fam, n, c.n_nodes, c.dofs_per_node, c.gauss_order))
    with open(os.path.join(TAB, "ch03_elements.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llrrr}\n\\toprule\nfamily & element & nodes & dofs per node & \\code{gauss\\_order} \\\\\n\\midrule\n")
        for fam, n, nn, dp, g in rows:
            fh.write(f"{fam} & \\code{{{n}}} & {nn} & {dp} & {g} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("elements", rows, flush=True)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_table(); exp_patch(); exp_bending(); exp_locking(); exp_plate()
    num("cecMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch03.tex"), "w") as fh:
        fh.write("% generated by scripts/ch03_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

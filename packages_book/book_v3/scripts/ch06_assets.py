# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch06_assets.py -- figures, tables and number macros of Chapter 6 (vectorised assembly and batched kernels).

Writes into book/:
    figures/ch06_scaling.pdf        Quad4PlaneStressTL: force assembly time against the number of elements, loop and batched
    figures/ch06_newton.pdf         a load-controlled Newton solve, default path against batched + sparse
    tables/ch06_families.tex        four element families: kernel and system speed-ups with the agreement of the results
    tables/ch06_chunk.tex           working memory per element and chunking
    tables/numbers_ch06.tex         every number quoted in the prose

It reuses the family benchmarks of paper/experiments/e10_batched_kernels.py (imported, not run, so the stored
paper/results/batched_kernels.json is not touched). Times are from the machine that runs this script; the reference
route is timed once, the batched route is the minimum of several runs. Run from the repository root:

    python packages_book/book/scripts/ch06_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import time
import platform
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(BOOK, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "paper", "experiments"))
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


import e10_batched_kernels as e10                                   # noqa: E402
from common import best_of, load_module                              # noqa: E402
from fea_engine import FESystem, Material, MultiBlockMesh            # noqa: E402
from fea_engine.material import D_plane_stress                       # noqa: E402
from fea_engine.elements import Quad4PlaneStressTL                   # noqa: E402
from fea_engine import nonlinear_solver as nls                       # noqa: E402

GB = load_module(os.path.join(ROOT, "fea_engine", "tests", "golden_loop_baselines.py"), "golden_loop_baselines")


# ------------------------------------------------------------------------------------------ families table
def exp_families():
    rows = []
    for name, fn in [("Quad4PlaneStressTL", lambda: e10.quad_family(3)),
                     ("Hex8SolidTL", lambda: e10.hex_family("Hex8SolidTL", 3)),
                     ("Hex20SolidTL", lambda: e10.hex_family("Hex20SolidTL", 3)),
                     ("Tet10SolidTL", lambda: e10.tet_family(3))]:
        t0 = time.perf_counter()
        d = fn()
        print(f"[{name}] {time.perf_counter() - t0:.1f} s", flush=True)
        agree = max(d["kernel_force"]["max_rel_diff"], d["kernel_tangent"]["max_rel_diff"], d["system_force"]["max_rel_diff"])
        rows.append((name, d["n_elements"], d["n_dof"], d["kernel_force"]["speedup"], d["kernel_tangent"]["speedup"],
                     d["system_force"]["speedup"], agree))
        print(f"   {rows[-1]}", flush=True)
    with open(os.path.join(TAB, "ch06_families.tex"), "w") as fh:
        fh.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n& & & \\multicolumn{3}{c}{speed-up} & \\\\\n\\cmidrule(lr){4-6}\nelement & el. & dofs & force & tangent & system force & difference \\\\\n\\midrule\n")
        for n, ne, nd, a, b, c, g in rows:
            fh.write(f"\\code{{{n}}} & {ne:,} & {nd:,} & {a:.0f}$\\times$ & {b:.0f}$\\times$ & {c:.0f}$\\times$ & {sci(g)} \\\\\n".replace(",", "\\,") if False else
                     f"\\code{{{n}}} & {ne} & {nd} & {a:.0f}$\\times$ & {b:.0f}$\\times$ & {c:.0f}$\\times$ & {sci(g)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cbkAgreeWorst", sci(max(r[6] for r in rows)))
    for r in rows:
        k = r[0].replace("Quad4PlaneStressTL", "Quad").replace("Hex8SolidTL", "HexEight").replace("Hex20SolidTL", "HexTwenty").replace("Tet10SolidTL", "Tet")
        num(f"cbkKF{k}", f"{r[3]:.0f}"); num(f"cbkKT{k}", f"{r[4]:.0f}"); num(f"cbkSF{k}", f"{r[5]:.0f}")
        num(f"cbkNE{k}", f"{r[1]:,}".replace(",", "\\,")); num(f"cbkND{k}", f"{r[2]:,}".replace(",", "\\,"))


# ------------------------------------------------------------------------------------------ scaling
def quad_system(nx, ny):
    nodes, el, _, _ = GB.quad_plate(nx, ny, L=6.0, H=2.0)
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": el}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    fes.add_nodal_force(right, 1, -40.0 / (ny + 1))
    return fes, {"q": D_plane_stress(Material(E=GB.E_MOD, nu=GB.NU))}


def exp_scaling():
    sizes = [(6, 2), (12, 4), (24, 8), (36, 12), (48, 16), (72, 24)]
    ne, tl, tb = [], [], []
    for nx, ny in sizes:
        fes, mat = quad_system(nx, ny)
        u = np.random.default_rng(0).normal(size=fes.n_dof) * 0.02
        kw = dict(thickness=0.5)
        fes.assemble_internal_force(u, mat, vectorized=True, **kw)
        t0 = time.perf_counter(); f0 = fes.assemble_internal_force(u, mat, **kw); t_loop = time.perf_counter() - t0
        t_b, _ = best_of(lambda: fes.assemble_internal_force(u, mat, vectorized=True, **kw), 5)
        ne.append(nx * ny); tl.append(t_loop); tb.append(t_b)
        print(f"scaling {nx*ny:5d} el: loop {t_loop:.4f}s batched {t_b:.5f}s x{t_loop/t_b:.0f}", flush=True)
    ne, tl, tb = map(np.array, (ne, tl, tb))
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    ax[0].loglog(ne, tl, "o-", label="per-element loop"); ax[0].loglog(ne, tb, "s-", label="batched")
    ax[0].set_xlabel("elements"); ax[0].set_ylabel("force assembly time (s)"); ax[0].legend()
    ax[1].semilogx(ne, tl / tb, "o-"); ax[1].set_xlabel("elements"); ax[1].set_ylabel("speed-up (loop / batched)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_scaling.pdf")); plt.close(fig)
    num("cbkScaleSmall", f"{(tl / tb)[0]:.0f}"); num("cbkScaleLarge", f"{(tl / tb)[-1]:.0f}")
    num("cbkScaleNeSmall", str(int(ne[0]))); num("cbkScaleNeLarge", str(int(ne[-1])))
    # slope of the loop time: should be about 1 (linear in elements)
    num("cbkLoopSlope", f"{np.polyfit(np.log(ne), np.log(tl), 1)[0]:.2f}")
    num("cbkBatchSlope", f"{np.polyfit(np.log(ne[2:]), np.log(tb[2:]), 1)[0]:.2f}")


# ------------------------------------------------------------------------------------------ Newton
def exp_newton():
    nx, ny = 40, 14
    res = {}
    for label, kw in [("default", dict()), ("batched + sparse", dict(sparse_tangent=True))]:
        fes, mat = quad_system(nx, ny)
        if label != "default":
            fes.batched = True
        t0 = time.perf_counter()
        lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=0.5, **kw)
        res[label] = (time.perf_counter() - t0, np.asarray(U), fes.n_dof)
        print(f"newton {label}: {res[label][0]:.2f}s", flush=True)
    (t1, U1, nd), (t2, U2, _) = res["default"], res["batched + sparse"]
    diff = np.abs(U1 - U2).max() / np.abs(U1).max()
    tip = 2 * ny * (nx + 1) + 0                      # not used; report max displacement instead
    num("cbkNewtonN", f"{nd:,}".replace(",", "\\,")); num("cbkNewtonNE", str(nx * ny))
    num("cbkNewtonDefault", f"{t1:.1f}"); num("cbkNewtonBatched", f"{t2:.1f}"); num("cbkNewtonRatio", f"{t1 / t2:.0f}")
    num("cbkNewtonDiff", sci(diff)); num("cbkNewtonMax", f"{np.abs(U1[-1]).max():.2f}")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(["default", "batched\n+ sparse"], [t1, t2], color=["C0", "C1"]); ax.set_ylabel("wall time (s)")
    ax.set_title(f"Newton solve, {nx * ny} elements, 6 steps", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_newton.pdf")); plt.close(fig)


# ------------------------------------------------------------------------------------------ chunking and report
def exp_chunk():
    from fea_engine.elements import Hex8SolidTL, Hex20SolidTL, Tet10SolidTL
    from fea_engine import batched_kernels as bk
    rows = []
    for el in (Quad4PlaneStressTL(), Hex8SolidTL(), Hex20SolidTL(), Tet10SolidTL()):
        b = bk.bytes_per_element(el, "tangent")
        rows.append((type(el).__name__, b, b * 1e5 / 1e9))
    with open(os.path.join(TAB, "ch06_chunk.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrr}\n\\toprule\nelement & estimated bytes per element (tangent) & GB for $10^5$ elements \\\\\n\\midrule\n")
        for n, b, g in rows:
            fh.write(f"\\code{{{n}}} & {b:,.0f} & {g:.1f} \\\\\n".replace(",", "\\,"))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("chunk", rows, flush=True)
    # chunked against unchunked, on a Quad plate
    fes, mat = quad_system(24, 8)
    u = np.random.default_rng(1).normal(size=fes.n_dof) * 0.05
    K0 = fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True, thickness=0.5).toarray()
    K1 = fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True, thickness=0.5, chunk_size=7).toarray()
    num("cbkChunkDiff", sci(np.abs(K0 - K1).max() / np.abs(K0).max()))
    fes.assemble_internal_force(u, mat, vectorized=True, thickness=0.5)
    num("cbkModes", str(fes.last_assembly_modes).replace("'", "").replace("{", "").replace("}", ""))
    rep = fes.batch_support_report()
    print("report", rep, flush=True)
    # a block that cannot be batched: the plastic truss
    from fea_engine import elements as elmod
    from fea_engine.mesh import Mesh
    f2 = FESystem(Mesh(nodes=np.array([[0.0, 0.0], [1.0, 0.0]]), elements=np.array([[0, 1]]), dim=2), elmod.TrussPlastic2D())
    rep2 = f2.batch_support_report()
    print("report2", rep2, flush=True)
    num("cbkReasonPlastic", str(rep2[0][3]).replace("_", "\\_").replace("&", "\\&"))
    num("cbkChunkedPlateElems", "192")


def exp_fold():
    """Contraction length of the folded Gauss-point sum (Q times the number of strain components), and a check of the
    stacking identity sum_q c_q B_q^T D B_q = Bc^T (D B) on one random quadrilateral."""
    from fea_engine.elements import Quad4PlaneStressTL, Hex8SolidTL, Hex20SolidTL
    from fea_engine.material import D_plane_stress, Material
    q4, h8, h20 = Quad4PlaneStressTL(), Hex8SolidTL(), Hex20SolidTL()
    num("cbkContrQuad", q4.gauss_order ** 2 * 3)
    num("cbkContrHexEight", h8.gauss_order ** 3 * 6)
    num("cbkContrHexTwenty", h20.gauss_order ** 3 * 6)
    rng = np.random.default_rng(1)
    coords = np.array([[[0.0, 0.0], [1.1, 0.0], [1.0, 0.9], [0.0, 1.0]]])
    u = rng.normal(size=(1, 8)) * 0.05
    D = D_plane_stress(Material(E=2.1e11, nu=0.3))
    dN, B, S, c = q4._batch_kinematics(coords, u, D, 0.5, None)
    E_, Q = c.shape
    loop = sum(c[0, q] * B[0, q].T @ D @ B[0, q] for q in range(Q))
    Bc = (B * c[:, :, None, None]).reshape(E_, Q * 3, 8)
    DB = (D @ B).reshape(E_, Q * 3, 8)
    stacked = (Bc.transpose(0, 2, 1) @ DB)[0]
    num("cbkFoldCheck", sci(np.abs(stacked - loop).max() / np.abs(loop).max()))


def main():
    print(platform.platform(), "numpy", np.__version__, "cpus", os.cpu_count(), flush=True)
    exp_chunk(); exp_scaling(); exp_newton(); exp_families(); exp_fold()
    num("cbkMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch06.tex"), "w") as fh:
        fh.write("% generated by scripts/ch06_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

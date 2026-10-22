# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch06_torch_assets.py -- table, figure and number macros of the PyTorch section of Chapter 6.

PyTorch is not available where the book is built, so the CPU and CUDA timings are NOT measured here: they are read from the
records of the author's machine (paper/results/e6_torch_backend.json and the torch_kernels block of
paper/results/batched_kernels.json; Windows, 12 cores, NVIDIA GeForce GTX 1050, PyTorch 2.7.0 with CUDA 12.6) and are labelled as such
in the text. What runs here, without PyTorch:
    - the kernels tl_internal_force / tl_tangent, which are written once for an array namespace, are run with xp = numpy and
      compared with the element's own batched kernels (the algebra that the torch path executes);
    - the rule of backend="auto" (select_backend) on plates of two sizes and on an indefinite matrix.
Writes into book/: figures/ch06_torch.pdf; tables/ch06_torch_solve.tex, tables/ch06_torch_other.tex; tables/numbers_ch06_torch.tex.

Run from the repository root:
    PYTHONPATH=fea_engine/src python packages_book/book/scripts/ch06_torch_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(BOOK, "..", ".."))
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def ms(t):
    v = 1000.0 * t
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def ratio(x):
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


E6 = json.load(open(os.path.join(ROOT, "paper", "results", "e6_torch_backend.json")))
BK = json.load(open(os.path.join(ROOT, "paper", "results", "batched_kernels.json")))
env, rows = E6["environment"], E6["data"]["rows"]
assert not E6["data"]["skipped"]
kern = BK["data"]["torch_kernels"]


# -------------------------------------------------------------------------- the stored timings
def tables():
    r0, r3 = rows[0], rows[-1]
    num("bktGpu", env["gpu"].replace("NVIDIA GeForce ", "")); num("bktTorch", env["torch"].split("+")[0])
    num("bktCuda", env["torch_cuda_build"]); num("bktCores", str(env["cpu_count"]))
    num("bktFirstDof", f"{r0['n_free']:,}".replace(",", "\\,")); num("bktLastDof", f"{r3['n_free']:,}".replace(",", "\\,"))
    # solve table
    L = []
    for r in rows:
        cells = [f"{r['n_free']:,}".replace(",", "\\,"), ms(r["t_scipy_s"]), ms(r["t_torch_cpu_cg_s"]), ms(r["t_torch_cuda_cg_s"])]
        for k in ("t_torch_cpu_dense_s", "t_torch_cuda_dense_s"):
            cells.append(ms(r[k]) if k in r else "--")
        worst = max(r[k] for k in r if k.startswith("rel_diff_"))
        cells.append(sci(worst))
        L.append(" & ".join(cells) + r" \\")
    open(os.path.join(TAB, "ch06_torch_solve.tex"), "w").write(
        "\\begin{tabular}{@{}r r r r r r r@{}}\n\\toprule\n"
        "& \\multicolumn{5}{c}{time of the solve (ms)} & \\\\\n\\cmidrule(lr){2-6}\n"
        "free dofs & SciPy & CG, CPU & CG, CUDA & dense, CPU & dense, CUDA & worst difference \\\\\n\\midrule\n"
        + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    # recovery + kernels table
    L = []
    for r in rows:
        L.append(" & ".join([f"{r['n_elements']:,}".replace(",", "\\,"), ms(r["t_stress_numpy_s"]), ms(r["t_stress_torch_cpu_s"]),
                             ms(r["t_stress_torch_cuda_s"]),
                             sci(max(r["stress_rel_diff_cpu"], r["stress_rel_diff_cuda"]))]) + r" \\")
    rec = "\n".join(L)
    kf = [kern["numpy"]["force_s"], kern["cpu"]["force_s"], kern["cuda"]["force_s"]]
    kt = [kern["numpy"]["tangent_s"], kern["cpu"]["tangent_s"], kern["cuda"]["tangent_s"]]
    kd = max(kern["cpu"]["max_rel_diff_force"], kern["cpu"]["max_rel_diff_tangent"],
             kern["cuda"]["max_rel_diff_force"], kern["cuda"]["max_rel_diff_tangent"])
    kr = (f"force & {ms(kf[0])} & {ms(kf[1])} & {ms(kf[2])} & {sci(max(kern['cpu']['max_rel_diff_force'], kern['cuda']['max_rel_diff_force']))} \\\\\n"
          f"tangent & {ms(kt[0])} & {ms(kt[1])} & {ms(kt[2])} & {sci(max(kern['cpu']['max_rel_diff_tangent'], kern['cuda']['max_rel_diff_tangent']))} \\\\")
    open(os.path.join(TAB, "ch06_torch_other.tex"), "w").write(
        "\\begin{tabular}{@{}r r r r r@{}}\n\\toprule\n"
        "(a) elements & NumPy & torch, CPU & torch, CUDA & worst difference \\\\\n\\midrule\n" + rec + "\n\\midrule\n"
        "(b) Hex8 kernel & NumPy & torch, CPU & torch, CUDA & worst difference \\\\\n\\midrule\n" + kr + "\n\\bottomrule\n\\end{tabular}\n")
    # ratios quoted in the text
    sp = [r["t_scipy_s"] for r in rows]
    num("bktSolveSmallCpu", ratio(rows[0]["t_torch_cpu_cg_s"] / sp[0])); num("bktSolveSmallCuda", ratio(rows[0]["t_torch_cuda_cg_s"] / sp[0]))
    num("bktSolveBigCpu", ratio(sp[-1] / rows[-1]["t_torch_cpu_cg_s"])); num("bktSolveBigCuda", ratio(sp[-1] / rows[-1]["t_torch_cuda_cg_s"]))
    num("bktSolveMidCpu", ratio(sp[2] / rows[2]["t_torch_cpu_cg_s"])); num("bktSolveMidCuda", ratio(sp[2] / rows[2]["t_torch_cuda_cg_s"]))
    num("bktDenseSlow", ratio(rows[1]["t_torch_cpu_dense_s"] / sp[1])); num("bktDenseSlowCuda", ratio(rows[1]["t_torch_cuda_dense_s"] / sp[1]))
    num("bktDenseDof", f"{rows[1]['n_free']:,}".replace(",", "\\,"))
    num("bktStrSmall", ratio(rows[0]["t_stress_torch_cpu_s"] / rows[0]["t_stress_numpy_s"]))
    num("bktStrBigCpu", ratio(rows[-1]["t_stress_numpy_s"] / rows[-1]["t_stress_torch_cpu_s"]))
    num("bktStrBigCuda", ratio(rows[-1]["t_stress_numpy_s"] / rows[-1]["t_stress_torch_cuda_s"]))
    num("bktKernNe", f"{kern['n_elements']:,}".replace(",", "\\,"))
    num("bktKfCpu", ratio(kern["cpu"]["force_vs_numpy"])); num("bktKfCuda", ratio(kern["cuda"]["force_vs_numpy"]))
    num("bktKtCpu", ratio(kern["cpu"]["tangent_vs_numpy"])); num("bktKtCuda", f"{kern['cuda']['tangent_vs_numpy']:.1f}")
    num("bktCgDiffMax", sci(max(v for r in rows for k, v in r.items() if k.startswith("rel_diff_") and "cg" in k)))
    num("bktDenseDiffMax", sci(max(v for r in rows for k, v in r.items() if k.startswith("rel_diff_") and "dense" in k)))
    num("bktStrDiffMax", sci(max(v for r in rows for k, v in r.items() if k.startswith("stress_rel_diff"))))
    return rows


# -------------------------------------------------------------------------- what runs here (no torch)
def torch_free():
    from fea_engine import Material, mesh as meshlib, D_plane_stress, FESystem, Quad4PlaneStress
    from fea_engine import batched_kernels_torch as bt
    from fea_engine.elements import Hex8SolidTL
    from fea_engine.backend_dispatch import select_backend, AUTO_DOF_THRESHOLD
    m = meshlib.box_mesh(4.0, 1.0, 1.0, 24, 8, 8)
    el, mat = Hex8SolidTL(), Material(E=1.0e3, nu=0.3)
    X = m.nodes[m.elements]
    u = np.random.default_rng(0).normal(size=(len(X), 24)) * 0.02
    geo = el._reference_geometry_batch(X)
    dN, JxW = np.asarray(geo[0]), np.asarray(geo[1]); D = np.asarray(el._elasticity(mat))
    rf = el.internal_force_batch(X, u, mat, geometry=geo); rk = el.tangent_stiffness_batch(X, u, mat, geometry=geo)
    f = bt.tl_internal_force(np, dN, JxW, u, D, np.eye(3)); k = bt.tl_tangent(np, dN, JxW, u, D, np.eye(3))
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    num("bktShadowForce", sci(rel(f, rf))); num("bktShadowTangent", sci(rel(k, rk)))
    num("bktShadowNe", f"{len(X):,}".replace(",", "\\,"))
    # backend="auto"
    from fea_engine.mesh import rectangle_mesh
    MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
    picks = []
    for nx, ny in ((32, 16), (64, 32)):
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=nx, ny=ny)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=True)
        s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
        mesh.select_nodes(x=0.0, name="root"); s.fix_dofs("root", ["ux", "uy"])
        rs = s.form_linear_system()
        b, d = select_backend(rs.K)
        b2, d2 = select_backend(rs.K, dof_threshold=1000)
        picks.append((rs.n_free, b, b2, d["spd"]))
    num("bktAutoThr", f"{AUTO_DOF_THRESHOLD:,}".replace(",", "\\,"))
    num("bktAutoSmallN", f"{picks[0][0]:,}".replace(",", "\\,")); num("bktAutoBigN", f"{picks[1][0]:,}".replace(",", "\\,"))
    assert picks[0][1] == "scipy" and picks[1][1] == "scipy" and picks[0][2] == "torch" and picks[0][3]
    A = np.diag([1.0, 2.0, -3.0, 4.0])                                 # symmetric, indefinite
    b3, d3 = select_backend(A, dof_threshold=1)
    assert b3 == "scipy" and not d3["spd"]
    num("bktAutoIndefinite", "scipy")


# -------------------------------------------------------------------------- figure
def figure():
    ne = [r["n_elements"] for r in rows]; nf = [r["n_free"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax[0].loglog(nf, [r["t_scipy_s"] for r in rows], "o-", color="#444444", label="SciPy (default)")
    ax[0].loglog(nf, [r["t_torch_cpu_cg_s"] for r in rows], "s--", color="#6a3d9a", label="torch CG, CPU")
    ax[0].loglog(nf, [r["t_torch_cuda_cg_s"] for r in rows], "^--", color="#e08000", label="torch CG, CUDA")
    dn = [r for r in rows if "t_torch_cpu_dense_s" in r]
    ax[0].loglog([r["n_free"] for r in dn], [r["t_torch_cpu_dense_s"] for r in dn], "s:", color="#6a3d9a", alpha=0.6, label="torch dense, CPU")
    ax[0].loglog([r["n_free"] for r in dn], [r["t_torch_cuda_dense_s"] for r in dn], "^:", color="#e08000", alpha=0.6, label="torch dense, CUDA")
    ax[0].set_xlabel("free dofs"); ax[0].set_ylabel("time of the solve (s)"); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) linear solve, Quad4 plate", fontsize=9)
    ax[1].loglog(ne, [r["t_stress_numpy_s"] for r in rows], "o-", color="#444444", label="NumPy (default)")
    ax[1].loglog(ne, [r["t_stress_torch_cpu_s"] for r in rows], "s--", color="#6a3d9a", label="torch, CPU")
    ax[1].loglog(ne, [r["t_stress_torch_cuda_s"] for r in rows], "^--", color="#e08000", label="torch, CUDA")
    ax[1].set_xlabel("elements"); ax[1].set_ylabel("time of the recovery (s)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) stress recovery", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_torch.pdf")); plt.close(fig)


def main():
    tables(); torch_free(); figure()
    with open(os.path.join(TAB, "numbers_ch06_torch.tex"), "w") as fh:
        fh.write("% generated by scripts/ch06_torch_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    for k, v in NUM.items():
        print(k, v)


if __name__ == "__main__":
    main()

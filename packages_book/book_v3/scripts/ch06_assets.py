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


def exp_cache():
    """Cold against warm assembly, cache sizes, rebuild on mesh motion, and the memory estimate (Section 7.2)."""
    import time
    from fea_engine import batched_kernels as bk
    from fea_engine import FESystem, Material, MultiBlockMesh
    from fea_engine.material import D_plane_stress
    from fea_engine.elements import Quad4PlaneStressTL, Hex20SolidTL
    nx, ny = 72, 24
    x, y = np.meshgrid(np.linspace(0, 6.0, nx + 1), np.linspace(0, 2.0, ny + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()]); n_ = lambda i, j: j * (nx + 1) + i
    el = np.array([[n_(i, j), n_(i + 1, j), n_(i + 1, j + 1), n_(i, j + 1)] for j in range(ny) for i in range(nx)])
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": el}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=True)
    mat = {"q": D_plane_stress(Material(E=2.1e11, nu=0.3))}
    u = np.random.default_rng(0).normal(size=fes.n_dof) * 0.02
    asm = lambda: fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True, thickness=0.5)
    t0 = time.perf_counter(); asm(); cold = time.perf_counter() - t0
    warm = min((lambda t0: (asm(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(5))
    gb, pb = fes._geometry_cache.builds, fes._pattern_cache.builds
    geoB, patB = fes._geometry_cache.nbytes, fes._pattern_cache.nbytes
    posB = next(iter(fes._pattern_cache._entries.values())).pos.nbytes
    nodes_now = fes.mesh.nodes; cp = nodes_now.copy(); t0 = time.perf_counter()
    for _ in range(200): np.array_equal(cp, nodes_now)
    eq_ms = 1e3 * (time.perf_counter() - t0) / 200
    fes.mesh.nodes[5, 0] += 1e-3
    asm()
    num("cbkCachePlateElems", len(el)); num("cbkColdMs", f"{1e3 * cold:.0f}"); num("cbkWarmMs", f"{1e3 * warm:.0f}")
    num("cbkCacheGeoMB", f"{geoB / 1e6:.2f}"); num("cbkCachePatMB", f"{patB / 1e6:.2f}"); num("cbkCachePosMB", f"{posB / 1e6:.2f}")
    num("cbkCacheEqualMs", f"{eq_ms:.3f}")
    num("cbkCacheRebuildGeo", fes._geometry_cache.builds - gb); num("cbkCacheRebuildPat", fes._pattern_cache.builds - pb)
    h20 = Hex20SolidTL()
    bpe = bk.bytes_per_element(h20, "tangent")
    fmt = lambda v: f"{v:,}".replace(",", "\\,")
    num("cbkBytesHexTwenty", fmt(bpe)); num("cbkBudgetMB", f"{bk.DEFAULT_BUDGET_BYTES / 2 ** 20:.0f}")
    num("cbkChunkQuad", fmt(bk.chunk_size_for(bk.bytes_per_element(Quad4PlaneStressTL(), "tangent"))))
    num("cbkChunkHexTwenty", fmt(bk.chunk_size_for(bpe)))
    num("cbkCachePosBigGB", f"{1e5 * 60 * 60 * 4 / 1e9:.1f}")


def exp_figs72():
    """Figures of Section 7.2: axes and fold (schematic), the four kernel variants, the scatter-add, and chunking."""
    import time, tracemalloc
    import matplotlib.patches as mp
    from fea_engine.elements import Quad4PlaneStressTL, Hex8SolidTL
    from fea_engine.material import D_plane_stress, Material
    from fea_engine import batched_kernels as bk

    # ------------------------------------------------------------ figure 1: schematic
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 6.2))
    a = ax[0]; a.set_xlim(0, 14); a.set_ylim(-0.5, 6.6); a.axis("off"); a.grid(False)
    nE, nQ = 5, 4
    for off, title in ((0.4, "loop: one Python call chain per cell"), (7.6, "batched: one array, one call")):
        a.text(off + 2.9, 6.2, title, ha="center", fontsize=9, weight="bold")
        for e in range(nE):
            for q in range(nQ):
                a.add_patch(mp.Rectangle((off + 0.8 + q * 1.2, 4.9 - e * 0.9), 1.0, 0.7, fc="#eaf2fb", ec="#5b8fc7", lw=0.8))
            a.text(off + 0.55, 5.25 - e * 0.9, f"$e={e + 1}$" if e < nE - 1 else "$e=E$", ha="right", va="center", fontsize=7)
        for q in range(nQ):
            a.text(off + 1.3 + q * 1.2, 5.75, f"$q={q + 1}$" if q < nQ - 1 else "$q=n_g$", ha="center", fontsize=7)
    # loop arrows
    for e in range(nE):
        for q in range(nQ - 1):
            a.annotate("", xy=(0.4 + 0.8 + (q + 1) * 1.2, 5.25 - e * 0.9), xytext=(0.4 + 1.8 + q * 1.2, 5.25 - e * 0.9),
                       arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    a.text(3.3, 0.35, "$E\\,n_g$ steps, each a few NumPy calls\non a $2\\times2$ or $3\\times3$ array", ha="center", fontsize=8, color="tab:red")
    a.add_patch(mp.FancyBboxPatch((7.6 + 0.7, 1.35), 5.0, 4.4, boxstyle="round,pad=0.05", fc="none", ec="tab:green", lw=1.8))
    a.text(10.5, 0.1, "$F_{eqij}=\\delta_{ij}+\\sum_a u_{eai}\\,(\\partial N_a/\\partial X_j)_{eq}$\none call over the axes $e$ and $q$", ha="center", fontsize=8, color="tab:green")
    a.text(0.0, 6.5, "(a)", fontsize=10, weight="bold")
    b = ax[1]; b.set_xlim(0, 14); b.set_ylim(0, 6.0); b.axis("off"); b.grid(False)
    b.text(0.0, 5.85, "(b)", fontsize=10, weight="bold")
    b.text(3.0, 5.5, "sum of $n_g$ small products", ha="center", fontsize=9, weight="bold")
    for q in range(4):
        x0 = 0.3 + q * 1.5
        b.add_patch(mp.Rectangle((x0, 3.2), 0.5, 1.5, fc="#fde9d9", ec="#c0504d", lw=0.8))   # B_q^T: n_dof x s  (drawn 8x3)
        b.add_patch(mp.Rectangle((x0 + 0.6, 3.9), 0.5, 0.8, fc="#fff2cc", ec="#bf9000", lw=0.8))  # D (s x s)
        b.add_patch(mp.Rectangle((x0 + 1.2, 3.9), 0.0, 0.0))
        b.text(x0 + 0.25, 2.95, f"$B_{q + 1}^T$", ha="center", fontsize=7); b.text(x0 + 0.85, 3.65, "$D$", ha="center", fontsize=7)
    b.text(3.0, 1.3, "$K_e=\\sum_{g=1}^{n_g}c_g\\,B_g^T D B_g$\n$n_g$ products, each with contraction length $s$", ha="center", fontsize=8)
    b.text(7.0, 4.1, "$\\Rightarrow$", ha="center", fontsize=16)
    b.text(10.4, 5.5, "one stacked product", ha="center", fontsize=9, weight="bold")
    Qd, sd = 4, 3
    for q in range(Qd):
        b.add_patch(mp.Rectangle((8.0, 3.0 + (Qd - 1 - q) * 0.45), 0.9, 0.42, fc="#fde9d9", ec="#c0504d", lw=0.7))
        b.add_patch(mp.Rectangle((9.8, 3.0 + (Qd - 1 - q) * 0.45), 0.9, 0.42, fc="#fff2cc", ec="#bf9000", lw=0.7))
    b.text(8.45, 4.95, "$B_c^T$", ha="center", fontsize=8); b.text(10.25, 4.95, "$DB$", ha="center", fontsize=8)
    b.text(9.35, 3.75, "$\\times$", ha="center", fontsize=12)
    b.text(8.45, 2.75, "stacked $c_gB_g$\n$(n_g s)\\times n_{dof}$", ha="center", fontsize=7)
    b.text(10.25, 2.75, "stacked $DB_g$\n$(n_g s)\\times n_{dof}$", ha="center", fontsize=7)
    b.add_patch(mp.Rectangle((11.7, 3.3), 1.2, 1.2, fc="#e2f0d9", ec="#548235", lw=0.8)); b.text(11.3, 3.85, "$=$", fontsize=12, ha="center")
    b.text(12.3, 4.7, "$K_e$", ha="center", fontsize=8); b.text(12.3, 2.9, "$n_{dof}\\times n_{dof}$", ha="center", fontsize=7)
    b.text(10.4, 1.5, "one batched matrix product, contraction length $n_g\\,s$\n(drawn for $n_g=4$, $s=3$: the quadrilateral, length 12)", ha="center", fontsize=8, color="tab:green")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_axes_fold.pdf")); plt.close(fig)

    # ------------------------------------------------------------ figure 2: four variants
    rng = np.random.default_rng(0)
    def tmin(fn, n=5):
        best = 1e9
        for _ in range(n):
            t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
        return best
    res = {}
    # quadrilateral
    q4 = Quad4PlaneStressTL(); Eq = 1728
    cq = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), (Eq, 1, 1)) + rng.normal(size=(Eq, 4, 2)) * 0.02
    uq = rng.normal(size=(Eq, 8)) * 0.02; Dq = D_plane_stress(Material(E=2.1e11, nu=0.3))
    def q_loop(): [q4.tangent_stiffness(cq[e], uq[e], Dq, thickness=0.5) for e in range(Eq)]
    def q_one(): [q4.tangent_stiffness_batch(cq[e:e + 1], uq[e:e + 1], Dq, thickness=0.5) for e in range(Eq)]
    def q_gp():
        dN, B, S, c = q4._batch_kinematics(cq, uq, Dq, 0.5, None); Q = c.shape[1]
        K = np.zeros((Eq, 8, 8))
        for q in range(Q):
            K += c[:, q, None, None] * (B[:, q].transpose(0, 2, 1) @ (Dq @ B[:, q]))
        S2 = np.empty((Eq, Q, 2, 2)); S2[..., 0, 0], S2[..., 1, 1], S2[..., 0, 1], S2[..., 1, 0] = S[..., 0], S[..., 1], S[..., 2], S[..., 2]
        Kg = np.einsum('eq,eqai,eqij,eqbj->eab', c, dN, S2, dN, optimize=True); K[:, 0::2, 0::2] += Kg; K[:, 1::2, 1::2] += Kg
        return K
    def q_fold(): return q4.tangent_stiffness_batch(cq, uq, Dq, thickness=0.5)
    res["Quad"] = [tmin(q_loop, 1), tmin(q_one, 1), tmin(q_gp), tmin(q_fold)]
    agree_q = np.abs(q_gp() - q_fold()).max() / np.abs(q_fold()).max()
    # hexahedron
    h8 = Hex8SolidTL(); Eh = 1536
    b8 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    ch = np.tile(b8, (Eh, 1, 1)) + rng.normal(size=(Eh, 8, 3)) * 0.02; uh = rng.normal(size=(Eh, 24)) * 0.02; mh = Material(E=2.1e11, nu=0.3)
    def h_loop(): [h8.tangent_stiffness(ch[e], uh[e], mh) for e in range(Eh)]
    def h_one(): [h8.tangent_stiffness_batch(ch[e:e + 1], uh[e:e + 1], mh) for e in range(Eh)]
    def h_gp():
        D, dN, B, S, c = h8._batch_kinematics(ch, uh, mh, None); Q = c.shape[1]; n = 8
        K = np.zeros((Eh, 24, 24))
        for q in range(Q):
            K += c[:, q, None, None] * (B[:, q].transpose(0, 2, 1) @ (D @ B[:, q]))
        S3 = np.empty((Eh, Q, 3, 3)); S3[..., 0, 0], S3[..., 1, 1], S3[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
        S3[..., 0, 1] = S3[..., 1, 0] = S[..., 3]; S3[..., 1, 2] = S3[..., 2, 1] = S[..., 4]; S3[..., 0, 2] = S3[..., 2, 0] = S[..., 5]
        SdN = S3 @ dN.transpose(0, 1, 3, 2); A = (dN * c[:, :, None, None]).transpose(0, 1, 3, 2).reshape(Eh, Q * 3, n)
        Kg = A.transpose(0, 2, 1) @ SdN.reshape(Eh, Q * 3, n)
        for i in range(3): K[:, i::3, i::3] += Kg
        return K
    def h_fold(): return h8.tangent_stiffness_batch(ch, uh, mh)
    res["Hex"] = [tmin(h_loop, 1), tmin(h_one, 1), tmin(h_gp), tmin(h_fold)]
    agree_h = np.abs(h_gp() - h_fold()).max() / np.abs(h_fold()).max()
    labels = ["element\\nloop", "batch\\nof one", "one call,\\nGauss loop", "one call,\\nfolded"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=False)
    for a_, key, name in ((ax[0], "Quad", "Quad4PlaneStressTL"), (ax[1], "Hex", "Hex8SolidTL")):
        t = np.array(res[key]) * 1e3
        a_.bar(range(4), t, color=["tab:gray", "tab:orange", "tab:blue", "tab:green"]); a_.set_yscale("log")
        for i, v in enumerate(t): a_.text(i, v * 1.15, f"{v:.0f} ms\n$\\times${t[0] / v:.0f}" if i else f"{v:.0f} ms", ha="center", fontsize=7)
        a_.set_xticks(range(4)); a_.set_xticklabels([l.replace("\\n", "\n") for l in labels], fontsize=7, rotation=0)
        a_.set_title(name, fontsize=9); a_.set_ylim(t.min() / 2, t.max() * 6); a_.grid(axis="x", visible=False)
    ax[0].set_ylabel("time of the tangent kernel (ms)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_variants.pdf")); plt.close(fig)
    for k, tag in (("Quad", "Quad"), ("Hex", "Hex")):
        t = np.array(res[k]) * 1e3
        for i, L in enumerate("ABCD"):
            num(f"cbkVar{tag}{L}", f"{t[i]:.0f}")
        num(f"cbkVar{tag}RatioOne", f"{t[0] / t[1]:.0f}"); num(f"cbkVar{tag}RatioBatch", f"{t[1] / t[2]:.0f}")
        num(f"cbkVar{tag}RatioFold", f"{t[2] / t[3]:.1f}"); num(f"cbkVar{tag}RatioAll", f"{t[0] / t[3]:.0f}")
    num("cbkVarQuadN", Eq); num("cbkVarHexN", Eh); num("cbkVarAgree", sci(max(agree_q, agree_h)))

    # ------------------------------------------------------------ figure 3: scatter-add schematic
    two = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    pat = bk.build_sparse_pattern(two, 6)                      # one degree of freedom per node
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.7]})
    m = ax[0]; m.axis("off"); m.grid(False); m.set_xlim(-0.3, 4.3); m.set_ylim(-0.7, 2.6)
    xy = {0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (0, 2), 4: (2, 2), 5: (4, 2)}
    for el, col in ((0, "#5b8fc7"), (1, "#c0504d")):
        pts = np.array([xy[i] for i in two[el]]); order = [0, 1, 2, 3]
        m.add_patch(mp.Polygon(pts[order], closed=True, fc=col, alpha=0.18, ec=col, lw=1.2))
    for i, (x, y) in xy.items():
        m.plot(x, y, "ko", ms=5); m.text(x + (0.12 if i % 3 != 2 else 0.12), y + (-0.3 if y == 0 else 0.15), f"node {i + 1}", fontsize=7)
    m.text(1.0, 1.0, "element 1", ha="center", fontsize=8, color="#2f5f9f"); m.text(3.0, 1.0, "element 2", ha="center", fontsize=8, color="#a03030")
    m.text(2.0, -0.55, "shared nodes 2 and 5", ha="center", fontsize=7)
    m.set_title("two elements, one degree of freedom per node", fontsize=8)
    r = ax[1]; r.axis("off"); r.grid(False); r.set_xlim(0, 10); r.set_ylim(-0.8, 9.2)
    shared = [(1, 1), (1, 4), (4, 1), (4, 4)]               # global (i,j), 0-based, shared by both elements
    pos_of = {}
    for el in (0, 1):
        for a_i in range(4):
            for b_i in range(4):
                gi, gj = two[el][a_i], two[el][b_i]
                pos_of[(el, gi, gj)] = int(pat.pos[el, a_i, b_i])
    r.text(1.4, 8.8, "entries of the\nelement matrices", ha="center", fontsize=8, weight="bold"); r.text(8.4, 8.8, "slots of the CSR\ndata array", ha="center", fontsize=8, weight="bold")
    slots = sorted({pos_of[(0, gi, gj)] for gi, gj in shared})
    ys_slot = {sp_: 7.2 - k * 1.4 for k, sp_ in enumerate(slots)}
    k = 0
    for el, col in ((0, "#5b8fc7"), (1, "#c0504d")):
        for gi, gj in shared:
            y = 7.7 - k * 0.95; k += 1
            r.add_patch(mp.FancyBboxPatch((0.2, y - 0.3), 2.5, 0.6, boxstyle="round,pad=0.02", fc=col, alpha=0.25, ec=col))
            r.text(1.45, y, f"element {el + 1}: ({gi + 1},{gj + 1})", ha="center", va="center", fontsize=7)
            sp_ = pos_of[(el, gi, gj)]
            r.annotate("", xy=(7.0, ys_slot[sp_]), xytext=(2.75, y), arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    for sp_, y in ys_slot.items():
        r.add_patch(mp.Rectangle((7.0, y - 0.38), 2.8, 0.76, fc="#e2f0d9", ec="#548235", lw=1.0))
        gi, gj = [(a_, b_) for a_, b_ in shared if pos_of[(0, a_, b_)] == sp_][0]
        r.text(8.4, y, f"slot {sp_}: $K_{{{gi + 1}{gj + 1}}}$", ha="center", va="center", fontsize=8)
    r.text(5.0, -0.5, "one bincount adds all arrows that end in the same slot", ha="center", fontsize=8, color="tab:green")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_scatter.pdf")); plt.close(fig)
    num("cbkScatterSlots", len(slots)); num("cbkScatterEntries", 2 * len(shared))

    # ------------------------------------------------------------ figure 4: chunk size
    bpe = bk.bytes_per_element(h8, "tangent")
    rows = []
    for cs in (1, 4, 16, 64, 256, 1536):
        def runc():
            out = np.empty((Eh, 24, 24))
            for s_ in bk.chunk_slices(Eh, chunk_size=cs): out[s_] = h8.tangent_stiffness_batch(ch[s_], uh[s_], mh)
            return out
        runc(); t = tmin(runc, 3)
        tracemalloc.start(); runc(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append((cs, t * 1e3, (peak - Eh * 24 * 24 * 8) / 1e6, cs * bpe / 1e6))
    R = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].loglog(R[:, 0], R[:, 2], "o-", ms=3, label="measured peak (\\texttt{tracemalloc})".replace("\\texttt{tracemalloc}", "tracemalloc"))
    ax[0].loglog(R[:, 0], R[:, 3], "s--", ms=3, label="estimate, equation of the text")
    ax[0].set_xlabel("chunk size (elements)"); ax[0].set_ylabel("working memory (MB)"); ax[0].legend(fontsize=7)
    ax[1].semilogx(R[:, 0], R[:, 1], "o-", ms=3, color="tab:green"); ax[1].set_xlabel("chunk size (elements)"); ax[1].set_ylabel("time of the tangent kernel (ms)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_chunk_fig.pdf")); plt.close(fig)
    num("cbkChunkFigN", Eh); num("cbkChunkFigRatio", f"{R[-1, 2] / R[-1, 3]:.2f}"); num("cbkChunkFigRatioSmall", f"{R[1, 2] / R[1, 3]:.2f}")
    num("cbkChunkTimeOne", f"{R[0, 1]:.0f}"); num("cbkChunkTimeBest", f"{R[:, 1].min():.0f}")
    num("cbkChunkTimeFlatFrom", int(R[np.argmax(R[:, 1] <= 1.25 * R[:, 1].min()), 0]))
    num("cbkChunkMemFull", f"{R[-1, 2]:.0f}")


def main():
    print(platform.platform(), "numpy", np.__version__, "cpus", os.cpu_count(), flush=True)
    exp_chunk(); exp_scaling(); exp_newton(); exp_families(); exp_fold(); exp_cache(); exp_figs72()
    num("cbkMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch06.tex"), "w") as fh:
        fh.write("% generated by scripts/ch06_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

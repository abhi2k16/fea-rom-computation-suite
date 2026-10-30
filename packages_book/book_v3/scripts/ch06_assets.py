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

    # ------------------------------------------------------------ figure 1: schematic (axes and fold)
    with plt.rc_context({"font.size": 8, "mathtext.fontset": "dejavusans"}):
        RED, REDF = "#c0504d", "#fde9d9"; YEL, YELF = "#bf9000", "#fff2cc"; GRN, GRNF = "#548235", "#e2f0d9"
        BLU, BLUF = "#5b8fc7", "#eaf2fb"; TR = "#c00000"
        fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 7.3), gridspec_kw={"height_ratios": [4.3, 7.0], "hspace": 0.05})
        for ax in (a, b): ax.axis("off")
        b.set_aspect("equal")

        # ------------------------------------------------ (a)
        a.set_xlim(0, 14); a.set_ylim(-1.6, 6.9)
        a.text(0.0, 6.75, "(a)", fontsize=10, weight="bold", va="center")
        nE, nQ, cw, ch, gx, gy = 5, 4, 1.0, 0.62, 0.22, 0.28
        def grid(x0, fc, ec, numbers=False):
            n = 0
            for e in range(nE):
                for q in range(nQ):
                    x = x0 + q * (cw + gx); y = 4.45 - e * (ch + gy)
                    hl = numbers and e == 1 and q == 1
                    a.add_patch(mp.Rectangle((x, y), cw, ch, fc="#ffd9d9" if hl else fc, ec=TR if hl else ec, lw=1.6 if hl else 0.8))
                    if numbers:
                        lab = "$En_g$" if (e == nE - 1 and q == nQ - 1) else ("$\\cdots$" if (e == nE - 2 and q == nQ - 1) else str(n + 1))
                        a.text(x + cw / 2, y + ch / 2, lab, ha="center", va="center", fontsize=7, color=TR if hl else "#444")
                    n += 1
            for q in range(nQ):
                a.text(x0 + q * (cw + gx) + cw / 2, 5.4, f"$q={q+1}$" if q < nQ - 1 else "$q=n_g$", ha="center", fontsize=7)
            for e in range(nE):
                a.text(x0 - 0.15, 4.45 - e * (ch + gy) + ch / 2, f"$e={e+1}$" if e < nE - 1 else "$e=E$", ha="right", va="center", fontsize=7)
        gl, gr = 1.0, 8.4
        grid(gl, BLUF, BLU, numbers=True); grid(gr, GRNF, GRN)
        a.text(gl + 2.6, 6.1, "loop: one cell at a time", ha="center", va="center", fontsize=9, weight="bold")
        a.text(gr + 2.6, 6.1, "batched: all cells in one array", ha="center", va="center", fontsize=9, weight="bold")
        a.text(0.4, 0.2, "$E\\,n_g$ steps, each a separate Python call chain", ha="left", va="center", fontsize=8, color=TR)
        a.text(0.4, -0.3, "highlighted: step 6, a few NumPy calls", ha="left", va="center", fontsize=7.5, color=TR)
        x1, x2 = gr - 0.12, gr + 4 * cw + 3 * gx + 0.12
        a.add_patch(mp.FancyBboxPatch((x1, 4.45 - 4 * (ch + gy) - 0.12), x2 - x1, 5 * ch + 4 * gy + 0.24, boxstyle="round,pad=0.03", fc="none", ec=GRN, lw=1.8))
        a.text(gr + 2.6, 0.2, "one array of shape $(E,\\,n_g,\\,\\ldots)$, one call", ha="center", va="center", fontsize=8, color=GRN)
        a.text(gr + 2.6, -1.0, "$F_{eqij}=\\delta_{ij}+\\sum_a u_{eai}\\,(\\partial N_a/\\partial X_j)_{eq}$", ha="center", va="center", fontsize=8, color=GRN)

        # ------------------------------------------------ (b)
        b.set_xlim(0, 14); b.set_ylim(0, 8.3)
        b.text(0.0, 8.1, "(b)", fontsize=10, weight="bold", va="center")
        b.text(7.0, 8.1, "the Gauss-point sum becomes one product", ha="center", va="center", fontsize=9, weight="bold")
        def R(x, y, w, h, fc, ec, lw=0.8): b.add_patch(mp.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
        def T(x, y, s, **k): b.text(x, y, s, ha=k.pop("ha", "center"), va=k.pop("va", "center"), fontsize=k.pop("fontsize", 7.5), **k)
        # row 1
        y1, H = 5.9, 1.1
        def triple(x0, g):
            R(x0, y1, 0.42, H, REDF, RED); R(x0 + 0.57, y1 + H - 0.42, 0.42, 0.42, YELF, YEL); R(x0 + 1.14, y1 + H - 0.42, H, 0.42, REDF, RED)
            return x0 + 1.14 + H
        xs = [0.3, 3.3, 7.7]
        for x0, lab in zip(xs, ("1", "2", "n_g")):
            xe = triple(x0, lab)
            T(x0 + 0.21, y1 - 0.3, f"$B_{{{lab}}}^T$"); T(x0 + 0.78, y1 + H - 0.42 - 0.3, "$D$"); T(x0 + 1.14 + H / 2, y1 + H - 0.42 - 0.3, f"$B_{{{lab}}}$")
        T(2.9, y1 + H / 2 + 0.15, "$+$", fontsize=11); T(6.6, y1 + H / 2 + 0.15, "$+\\ \\cdots\\ +$", fontsize=11)
        T(10.3, y1 + H / 2 + 0.15, "$=$", fontsize=11); R(10.75, y1, H, H, GRNF, GRN); T(10.75 + H / 2, y1 - 0.3, "$K_e$")
        T(12.15, y1 + H / 2, "$K_e=\\sum_{g=1}^{n_g} c_g\\,B_g^T D\\,B_g$", fontsize=8, ha="left")
        T(5.0, y1 + H + 0.45, "$n_g$ small products, each contracting over $s$ entries", color="#444", fontsize=8)
        # fold arrow
        b.annotate("", xy=(3.2, 3.55), xytext=(3.2, 4.9), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
        T(3.5, 4.25, "fold: stack the $n_g$ blocks", ha="left", fontsize=8)
        # row 2
        ng, bw = 4, 0.55
        yb = 0.8; Hb = ng * bw
        xr = 0.6
        yc = yb + Hb / 2 - H / 2
        for q in range(ng):
            R(xr + q * bw, yc, bw, H, REDF, RED, 0.9)
        R(xr, yc, ng * bw, H, "none", RED, 1.4)
        T(xr + ng * bw / 2, yc + H + 0.3, "$B_c^T$ ($c_g$ included)"); T(xr + ng * bw / 2, yc - 0.3, "$n_{dof}\\times n_gs$")
        xd = xr + ng * bw + 0.7
        T(xd - 0.35, yb + Hb / 2, "$\\times$", fontsize=12)
        for q in range(ng):
            R(xd, yb + (ng - 1 - q) * bw, H, bw, YELF, YEL, 0.9)
        R(xd, yb, H, Hb, "none", YEL, 1.4)
        T(xd + H / 2, yb + Hb + 0.3, "$DB$"); T(xd + H / 2, yb - 0.3, "$n_gs\\times n_{dof}$")
        xk = xd + H + 0.7
        T(xk - 0.35, yb + Hb / 2, "$=$", fontsize=12); R(xk, yc, H, H, GRNF, GRN, 1.4)
        T(xk + H / 2, yc + H + 0.3, "$K_e$"); T(xk + H / 2, yc - 0.3, "$n_{dof}\\times n_{dof}$")
        T(xk + H + 0.9, yb + Hb / 2, "one batched matrix product,\ncontraction length $n_g\\,s$\n(drawn for the quadrilateral:\n$n_g=4$, $s=3$, length 12)", ha="left", color=GRN, fontsize=8)
        fig.savefig(os.path.join(FIG, "ch06_axes_fold.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)

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

    # ------------------------------------------------------------ figure 3: scatter-add schematic (local to global, position map, sum)
    two = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    pat = bk.build_sparse_pattern(two, 6)                      # one degree of freedom per node
    with plt.rc_context({"font.size": 8, "mathtext.fontset": "dejavusans"}):
        fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0, 15); ax.set_ylim(-0.2, 9.2)
        C1, C2 = "#5b8fc7", "#c0504d"
        slotcol = ["#e08a1e", "#2e8b57", "#8e44ad", "#c0392b"]           # one colour per shared slot
        # ---------------- mesh with local numbering
        xy = {0: (0.3, 2.2), 1: (1.9, 2.2), 2: (3.5, 2.2), 3: (0.3, 4.4), 4: (1.9, 4.4), 5: (3.5, 4.4)}
        for el, col in ((0, C1), (1, C2)):
            pts = np.array([xy[i] for i in two[el]]); ax.add_patch(mp.Polygon(pts, closed=True, fc=col, alpha=0.18, ec=col, lw=1.2))
        for i, (x, y) in xy.items():
            ax.plot(x, y, "ko", ms=4); ax.text(x, y + (-0.45 if y < 3 else 0.35), f"{i + 1}", ha="center", va="center", fontsize=8, weight="bold")
        for el, col, dx in ((0, C1, 0.28), (1, C2, 0.28)):
            for a_i, g in enumerate(two[el]):
                x, y = xy[g]; sx = 1 if a_i in (0, 3) and el == 0 or a_i in (0, 3) and el == 1 else -1
                # inward offset: towards the element centre
                cx = np.mean([xy[k][0] for k in two[el]]); cy = np.mean([xy[k][1] for k in two[el]])
                ax.text(x + 0.38 * np.sign(cx - x), y + 0.38 * np.sign(cy - y), f"{a_i + 1}", ha="center", va="center", fontsize=7, color=col, weight="bold")
        ax.text(1.1, 3.3, "element 1", ha="center", fontsize=8, color=C1, rotation=90); ax.text(2.7, 3.3, "element 2", ha="center", fontsize=8, color=C2, rotation=90)
        ax.text(1.9, 6.0, "mesh: global node numbers (black),\nlocal numbers (coloured)", ha="center", va="center", fontsize=7.5)
        ax.text(1.9, 1.2, "nodes 2 and 5 are shared", ha="center", fontsize=7.5)
        # ---------------- element matrices
        s = 0.52
        def matrix(x0, ytop, el, col, hl):
            for r in range(4):
                for c in range(4):
                    h = (r, c) in hl
                    ax.add_patch(mp.Rectangle((x0 + c * s, ytop - (r + 1) * s), s, s, fc=slotcol[hl[(r, c)]] if h else "#f2f2f2", ec="white" if not h else "k", lw=0.8 if not h else 0.9, alpha=0.9 if h else 1))
            ax.add_patch(mp.Rectangle((x0, ytop - 4 * s), 4 * s, 4 * s, fc="none", ec=col, lw=1.6))
            for k in range(4):
                ax.text(x0 + k * s + s / 2, ytop + 0.18, str(k + 1), ha="center", va="center", fontsize=6.5, color="#555")
                ax.text(x0 - 0.18, ytop - k * s - s / 2, str(k + 1), ha="center", va="center", fontsize=6.5, color="#555")
            ax.text(x0 + 2 * s, ytop + 0.55, f"element {el + 1}: $k^{{({el + 1})}}$", ha="center", fontsize=8, color=col, weight="bold")
        x0m = 6.0
        hl1 = {(1, 1): 0, (1, 2): 1, (2, 1): 2, (2, 2): 3}          # element 1: local nodes 2,3 are global 2,5
        hl2 = {(0, 0): 0, (0, 3): 1, (3, 0): 2, (3, 3): 3}          # element 2: local nodes 1,4 are global 2,5
        y1, y2 = 8.0, 4.6
        matrix(x0m, y1, 0, C1, hl1); matrix(x0m, y2, 1, C2, hl2)
        ax.text(x0m + 2 * s, 1.7, "same colour: same global\nentry, hence the same slot", ha="center", va="center", fontsize=7.5)
        # ---------------- slots
        pos_of = {}
        for el in (0, 1):
            for a_i in range(4):
                for b_i in range(4): pos_of[(el, a_i, b_i)] = int(pat.pos[el, a_i, b_i])
        names = [(1, 1), (1, 4), (4, 1), (4, 4)]
        slot_p = [pos_of[(0, 1, 1)], pos_of[(0, 1, 2)], pos_of[(0, 2, 1)], pos_of[(0, 2, 2)]]
        assert slot_p == [pos_of[(1, 0, 0)], pos_of[(1, 0, 3)], pos_of[(1, 3, 0)], pos_of[(1, 3, 3)]]
        xs0 = 10.2; ys = [7.6, 5.7, 3.8, 1.9]
        ax.text(xs0 + 2.4, 8.95, "CSR data array", ha="center", fontsize=8.5, weight="bold")
        for k, (p, (gi, gj)) in enumerate(zip(slot_p, names)):
            ax.add_patch(mp.FancyBboxPatch((xs0, ys[k] - 0.4), 4.5, 0.8, boxstyle="round,pad=0.02", fc=slotcol[k], alpha=0.25, ec=slotcol[k], lw=1.4))
            ax.text(xs0 + 0.15, ys[k] + 0.17, f"data[{p}]", fontsize=7, va="center", color="#333")
        lab = [("22", "22", "11"), ("25", "23", "14"), ("52", "32", "41"), ("55", "33", "44")]
        for k, (K, a, b) in enumerate(lab):
            ax.text(xs0 + 4.35, ys[k] - 0.14, f"$K_{{{K}}}=k^{{(1)}}_{{{a}}}+k^{{(2)}}_{{{b}}}$", fontsize=8, ha="right", va="center")
        # arrows
        for k in range(4):
            for el, hl, ytop in ((0, hl1, y1), (1, hl2, y2)):
                (r, c) = [rc for rc, v in hl.items() if v == k][0]
                ax.annotate("", xy=(xs0 - 0.02, ys[k] + (0.2 if el == 0 else -0.2)), xytext=(x0m + 4 * s + 0.05, ytop - r * s - s / 2 if True else 0),
                            arrowprops=dict(arrowstyle="->", color=slotcol[k], lw=1.1, shrinkA=0, shrinkB=0))
        ax.text(9.35, 9.0, "position map", ha="center", va="center", fontsize=8, style="italic")
        ax.text(xs0 + 2.25, 0.5, "one bincount adds all arrows\nthat end in the same slot", ha="center", va="center", fontsize=8, color="#2e6b2e")
        fig.savefig(os.path.join(FIG, "ch06_scatter.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    slots = slot_p; shared = names
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

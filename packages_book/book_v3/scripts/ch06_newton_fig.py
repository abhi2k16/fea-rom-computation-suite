# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch06_newton_fig.py -- figures/ch06_newton.pdf: one load-controlled Newton solve, timed in four configurations
(default, batched assembly only, sparse tangent only, both) and the time of the default and the batched + sparse path
against the number of elements. It also rewrites the Newton macros of tables/numbers_ch06.tex so that the text and the
figure come from the same run. Run from the repository root:
    PYTHONPATH=fea_engine/src python packages_book/book_v3/scripts/ch06_newton_fig.py
"""
import os, re, sys, time, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch06_assets as A                                  # noqa: E402  (imports the helpers only; main() is not run)
from fea_engine import nonlinear_solver as nls           # noqa: E402

NAVY, BLUE, ORG, GRN, RED, GREY = "#1b3a5c", "#2f7fc1", "#d9822b", "#2e8b57", "#c0392b", "#8a96a3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})


def run(nx, ny, batched, sparse, repeats=1):
    best, U = None, None
    for _ in range(repeats):
        fes, mat = A.quad_system(nx, ny)
        if batched:
            fes.batched = True
        t0 = time.perf_counter()
        _, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=0.5, sparse_tangent=sparse)
        t = time.perf_counter() - t0
        best = t if best is None else min(best, t)
    return best, np.asarray(U), fes.n_dof


def main():
    nx, ny = 40, 14
    variants = [("default", False, False), ("batched\nassembly only", True, False),
                ("sparse\ntangent only", False, True), ("batched +\nsparse", True, True)]
    times, sols = [], []
    for lab, b, s in variants:
        t, U, nd = run(nx, ny, b, s, repeats=1 if not (b and s) else 3)
        times.append(t); sols.append(U); print(f"{lab!r}: {t:.2f} s", flush=True)
    diff = max(np.abs(U - sols[0]).max() / np.abs(sols[0]).max() for U in sols[1:])
    sizes = [(10, 4), (20, 8), (30, 11), (40, 14)]
    ne, t_def, t_fast = [], [], []
    for (a, b) in sizes:
        td, _, _ = run(a, b, False, False); tf, _, _ = run(a, b, True, True, repeats=3)
        ne.append(a * b); t_def.append(td); t_fast.append(tf); print(f"{a*b} el: default {td:.2f}  fast {tf:.3f}", flush=True)
    t_def[-1], t_fast[-1] = times[0], times[3]            # the largest model is the run of panel (a)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.32})
    plt.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.2)
    ax = axs[0]
    cols = [BLUE, GREY, GREY, GRN]
    ax.bar(range(4), times, color=cols, width=0.62)
    ax.set_yscale("log"); ax.set_ylim(min(times) / 3, max(times) * 3)
    for i, t in enumerate(times):
        ax.text(i, t * 1.25, f"{t:.2f} s" if t < 10 else f"{t:.1f} s", ha="center", fontsize=7.5, color=NAVY)
        if i > 0: ax.text(i, t / 1.9, f"{times[0] / t:.1f}$\\times$", ha="center", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([v[0] for v in variants], fontsize=6.8); ax.set_ylabel("wall time (s, log scale)")
    ax.set_title(f"(a) Where the time goes ({nx * ny} elements, 6 steps)", fontsize=8.5, fontweight="bold", color=NAVY, loc="left")
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    ax = axs[1]
    ax.loglog(ne, t_def, "o-", color=BLUE, lw=1.8, label="default")
    ax.loglog(ne, t_fast, "s-", color=GRN, lw=1.8, label="batched + sparse")
    ax.set_xlabel("elements"); ax.set_ylabel("wall time (s)"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(ne[-1], np.sqrt(t_def[-1] * t_fast[-1]), f"{t_def[-1] / t_fast[-1]:.0f}$\\times$", color=NAVY, fontsize=8, ha="right", va="center", fontweight="bold")
    ax.set_title("(b) Growth with the model size", fontsize=8.5, fontweight="bold", color=NAVY, loc="left")
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    fig.savefig(os.path.join(os.path.dirname(HERE), "figures", "ch06_newton.pdf"))
    # keep the macros of the text in step with this run
    p = os.path.join(os.path.dirname(HERE), "tables", "numbers_ch06.tex"); s = open(p).read()
    def setm(name, val):
        nonlocal s
        line = "\\newcommand{\\%s}{%s\\xspace}" % (name, val)
        pat = r"\\newcommand\{\\%s\}\{[^\n]*\}" % name
        if re.search(pat, s): s = re.sub(pat, lambda m: line, s)
        else: s += line + "\n"
    setm("cbkNewtonDefault", f"{times[0]:.1f}"); setm("cbkNewtonBatched", f"{times[3]:.2f}"); setm("cbkNewtonRatio", f"{times[0] / times[3]:.0f}")
    setm("cbkNewtonDiff", A.sci(diff))
    setm("cbkNewtonBatchedOnly", f"{times[1]:.1f}"); setm("cbkNewtonSparseOnly", f"{times[2]:.1f}")
    setm("cbkNewtonBatchedOnlyRatio", f"{times[0] / times[1]:.1f}"); setm("cbkNewtonDenseCost", f"{times[1] - times[3]:.1f}")
    open(p, "w").write(s)
    print("times", times, "diff", diff)


if __name__ == "__main__":
    main()

"""Plot the results written by system_driven_bases_timing.py.

Usage: python plot_system_driven_bases_timing.py [results_dir] [out.png]
Reads results/timing_*.json (default) and draws, per case, the speed-up
(log scale), the error (log scale) and the speed-up including offline cost
for the first 8 queries of a sweep (or the single transient run).
"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(rdir, "system_driven_bases_timing.png")

cases = []
for f in sorted(glob.glob(os.path.join(rdir, "timing_*.json"))):
    d = json.load(open(f))
    for kind in ("static", "transient"):
        if kind in d and d[kind].get("models"):
            cases.append((f"{d['model']} {d['size']} {kind}", d[kind]["models"]))

names = ["Galerkin, 3 modes (full force)", "Galerkin, modal derivatives (full force)",
         "polynomial force (EED)", "quadratic manifold + ECSW"]
short = ["Galerkin 3", "Galerkin MD", "Poly (EED)", "Manifold+ECSW"]
colors = ["#9aa5b1", "#4c78a8", "#e8a33d", "#2a9d5c"]

fig, ax = plt.subplots(3, 1, figsize=(max(9, 1.6 * len(cases)), 11), sharex=True)
x = np.arange(len(cases)); w = 0.2
keys = ["speedup", "err", "effective_speedup"]
titles = ["Per-query speed-up vs full order (x, log)", "Worst relative error (log)",
          "Effective speed-up incl. offline cost (x, log; 1 = no gain)"]
for a, key, title in zip(ax, keys, titles):
    for j, n in enumerate(names):
        for i, (_, m) in enumerate(cases):
            r = m.get(n) or m.get(n.replace("modal derivatives", "modal derivatives"))
            if r is None or "failed" in r:
                a.text(i + (j - 1.5) * w, a.get_ylim()[0] if False else 1e-3 if key == "err" else 0.5,
                       "x", ha="center", color="red", fontsize=9)
                continue
            v = max(r[key] if r[key] is not None else np.nan, 1e-6)
            a.bar(i + (j - 1.5) * w, v, w, color=colors[j], label=short[j] if i == 0 else None)
    a.set_yscale("log"); a.set_title(title, fontsize=10, loc="left")
    a.grid(axis="y", alpha=0.3, which="both")
    if key != "err":
        a.axhline(1, color="k", lw=0.8)
ax[0].legend(ncol=4, fontsize=8, loc="upper left")
ax[2].set_xticks(x); ax[2].set_xticklabels([c for c, _ in cases], rotation=25, ha="right")
fig.text(0.01, 0.005, "red x = Newton did not converge. 2 CPU cores; plate mesh is coarse; "
         "6x6 plate errors (~7.5%) unexplained.", fontsize=7)
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig(out, dpi=150)
print("wrote", out)

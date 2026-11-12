r"""Chapter 14 figure: the membrane benchmark (fig:rc-membrane) and macros tables/numbers_ch14_fig.tex (prefix cfg).
Run from the repository root with PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests:fea_engine/tests."""
import os, warnings, numpy as np, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
warnings.simplefilter("ignore")
from contact_fixtures import ObstacleMembrane
from fea_engine.contact_solvers import solve_contact_dynamics
H, NS = 0.005, 120
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
mem = ObstacleMembrane(18); n = mem.n; N = mem.n_dof
M = sp.diags(np.full(N, mem.h ** 2)).tocsr(); force = lambda t: mem.f
train = [(0.35, 0.10), (0.45, 0.12), (0.40, 0.18), (0.40, 0.12)]
tests = {"inside": (0.40, 0.15), "outside": (0.55, 0.05)}
fom = {}
for g in train + list(tests.values()):
    c = mem.obstacle(g); fom[g] = (c,) + solve_contact_dynamics(M, mem.K, force, mem.B, c, H, NS, scheme="bdf2")
fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
# (a) field at the step of most contact, outside obstacle
g = tests["outside"]; c, U, L = fom[g]; k = int((L > 1e-9).sum(1).argmax())
u = U[k].reshape(n, n); act = (L[k] > 1e-9).reshape(n, n)
a = ax[0]; im = a.pcolormesh(mem.X, mem.Y, u, shading="nearest", cmap="viridis"); plt.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="displacement")
a.contour(mem.X, mem.Y, c.reshape(n, n), levels=[-0.29, -0.25, -0.2, -0.15], colors="w", linewidths=0.7)
xi, yi = np.nonzero(act); a.plot(mem.X[xi, yi], mem.Y[xi, yi], "r.", ms=4, label="in contact")
a.set_aspect("equal"); a.grid(False); a.set_xlabel("x"); a.set_ylabel("y"); a.set_title("(a) held-out obstacle, most contact", fontsize=8); a.legend(fontsize=6.5, loc="lower left")
# (b) parameter plane
b = ax[1]; tp = np.array(train)
b.add_patch(Rectangle((tp[:, 0].min(), tp[:, 1].min()), np.ptp(tp[:, 0]), np.ptp(tp[:, 1]), color="C0", alpha=0.12, label="training range"))
b.plot(tp[:, 0], tp[:, 1], "ko", label="training")
b.plot(*tests["inside"], "s", color="C2", label="held out, inside"); b.plot(*tests["outside"], "^", color="C3", label="held out, outside")
b.set_xlabel("position of bump 1"); b.set_ylabel("height of bump 2"); b.set_title("(b) the obstacles", fontsize=8); b.legend(fontsize=6.5)
# (c) active constraints in time
cc = ax[2]; t = np.arange(NS + 1) * H
cc.plot(t, (fom[train[3]][2] > 1e-9).sum(1), color="k", label="training obstacle")
cc.plot(t, (fom[tests["inside"]][2] > 1e-9).sum(1), color="C2", label="inside")
cc.plot(t, (fom[tests["outside"]][2] > 1e-9).sum(1), color="C3", label="outside")
cc.set_xlabel("t"); cc.set_ylabel("active constraints (of %d)" % N); cc.set_title("(c) active set in time", fontsize=8); cc.legend(fontsize=6.5)
plt.tight_layout(); B = "packages_book/book_v3"
plt.savefig(os.path.join(B, "figures", "ch14_membrane.pdf")); plt.savefig(os.path.join(B, "preview_figs", "ch14_f1_preview.png"), dpi=150)
d = dict(cfgStep=str(k), cfgTime=f"{k * H:.2f}", cfgPeakOut=str(int((fom[tests["outside"]][2] > 1e-9).sum(1).max())),
         cfgPeakInLo=str(min(int((fom[q][2] > 1e-9).sum(1).max()) for q in train + [tests["inside"]])),
         cfgPeakInHi=str(max(int((fom[q][2] > 1e-9).sum(1).max()) for q in train + [tests["inside"]])),
         cfgNSteps=str(NS), cfgLateMax=str(int((fom[tests["outside"]][2][int(round(0.4 / H)):] > 1e-9).sum(1).max())))
with open(os.path.join(B, "tables", "numbers_ch14_fig.tex"), "w") as fh:
    for kk, v in d.items(): fh.write(f"\\newcommand{{\\{kk}}}{{{v}\\xspace}}\n")
print(d)

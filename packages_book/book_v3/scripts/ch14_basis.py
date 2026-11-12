r"""Chapter 14 figure: dual basis vectors, SVD against NMF (fig:rc-basis), macros prefix cfb in tables/numbers_ch14_basis.tex.
Run from the repository root with PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests:fea_engine/tests."""
import os, warnings, numpy as np, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
from contact_fixtures import ObstacleMembrane
from fea_engine.contact_solvers import solve_contact_dynamics
from rom_engine import nonneg_dual_basis, active_snapshot_columns
from rom_engine.contact_dual_basis import svd_dual_basis
H, NS = 0.005, 120; PL = 10; NSHOW = 4
plt.rcParams.update({"font.size": 8})
mem = ObstacleMembrane(18); n = mem.n; N = mem.n_dof
M = sp.diags(np.full(N, mem.h ** 2)).tocsr(); force = lambda t: mem.f
train = [(0.35, 0.10), (0.45, 0.12), (0.40, 0.18), (0.40, 0.12)]
Ltr = np.hstack([solve_contact_dynamics(M, mem.K, force, mem.B, mem.obstacle(g), H, NS, scheme="bdf2")[1][1:].T for g in train])
X = Ltr[:, active_snapshot_columns(Ltr)]
Us = svd_dual_basis(X, PL); nm = nonneg_dual_basis(X, PL); Un = nm.U
order = np.argsort(-(np.linalg.norm(nm.V, axis=1)))          # NMF columns carry no order: rank them by their weight in the fit
Un = Un[:, order]
negfrac = lambda U: [(U[:, k] < -1e-12 * np.abs(U[:, k]).max()).mean() for k in range(U.shape[1])]
supp = lambda U: [(np.abs(U[:, k]) > 1e-3 * np.abs(U[:, k]).max()).mean() for k in range(U.shape[1])]
fig, ax = plt.subplots(2, NSHOW, figsize=(8.4, 4.0))
for j in range(NSHOW):
    for i, (U, name, cmap) in enumerate(((Us, "SVD", "RdBu_r"), (Un, "NMF", "viridis"))):
        v = U[:, j].reshape(n, n); a = ax[i, j]
        if name == "SVD": m = np.abs(v).max(); im = a.pcolormesh(mem.X, mem.Y, v, shading="nearest", rasterized=True, cmap=cmap, vmin=-m, vmax=m)
        else: im = a.pcolormesh(mem.X, mem.Y, v, shading="nearest", rasterized=True, cmap=cmap, vmin=0, vmax=v.max())
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{name} vector {j + 1}\n{100 * negfrac(U)[j]:.0f} % negative entries", fontsize=7.5)
plt.tight_layout(); B = "packages_book/book_v3"
plt.savefig(os.path.join(B, "figures", "ch14_basis.pdf"), dpi=300); plt.savefig(os.path.join(B, "preview_figs", "ch14_f2_preview.png"), dpi=150)
shown = negfrac(Us)[:NSHOW]
d = dict(cfbPLam=str(PL), cfbNShow=str(NSHOW), cfbNegLo=f"{100 * min(shown[1:]):.0f}", cfbNegHi=f"{100 * max(shown[1:]):.0f}")
with open(os.path.join(B, "tables", "numbers_ch14_basis.tex"), "w") as fh:
    for kk, v in d.items(): fh.write(f"\\newcommand{{\\{kk}}}{{{v}\\xspace}}\n")
print(d)

r"""Chapter 14 introduction: figure ch14_intro.pdf and macros tables/numbers_ch14_intro.tex (prefix crx).
Run from the repository root with PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests:fea_engine/tests."""
import os, sys, warnings
import numpy as np, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
from contact_fixtures import ObstacleMembrane, NonlinearChainContact
from fea_engine.contact_solvers import solve_contact_dynamics
from rom_engine import ContactROM, nonneg_dual_basis, active_snapshot_columns
from rom_engine.contact_dual_basis import svd_dual_basis
from rom_engine.contact_hyper_reduction import rank_deficiency, enrich_basis_for_constraints
H, NS = 0.005, 120
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
mem = ObstacleMembrane(18); n = mem.n_dof
M = sp.diags(np.full(n, mem.h ** 2)).tocsr(); force = lambda t: mem.f
train = [(0.35, 0.10), (0.45, 0.12), (0.40, 0.18), (0.40, 0.12)]; test = (0.55, 0.05)
fom = {}
for g in train + [test]:
    c = mem.obstacle(g); fom[g] = (c,) + solve_contact_dynamics(M, mem.K, force, mem.B, c, H, NS, scheme="bdf2")
Utr = np.hstack([fom[g][1][1:].T for g in train]); Ltr = np.hstack([fom[g][2][1:].T for g in train])
act = active_snapshot_columns(Ltr); Ux = np.linalg.svd(Utr, full_matrices=False)[0]
c, Uf, Lf = fom[test]
sizes = (10, 20, 30, 40); err = {"nmf": [], "svd": []}; kkt = {"nmf": [], "svd": []}; lifts = {}
for p in sizes:
    for name, Ul in (("nmf", nonneg_dual_basis(Ltr[:, act], p).U), ("svd", svd_dual_basis(Ltr[:, act], p))):
        r = ContactROM(mem.K, mem.B, Ux[:, :p], Ul, M=M).solve_dynamic(force, c, H, NS)
        err[name].append(np.sum((r.u - Uf) ** 2) / np.sum(Uf ** 2)); kkt[name].append(float(np.max(r.kkt_residual)))
        if p == 20: lifts[name] = r.lam
# (a) two multipliers with largest peak, plane (lam_i, lam_j)
pk = Lf.max(0); i, j = np.argsort(pk)[-2:]
# (c) chain rank
m = NonlinearChainContact()
train_c = [(s, g) for s in (0.5, 1, 1.5, 2, 2.5, 3) for g in (-0.08, -0.15, -0.25)]
Xu = np.array([m.solve(*q)[0] for q in train_c]).T
V3 = np.linalg.svd(Xu, full_matrices=False)[0][:, :3]; V3e, added = enrich_basis_for_constraints(V3, m.B)
rows = m.B.shape[0]; ks = list(range(len(added) + 1))
rk = [rows - rank_deficiency(m.B, V3e[:, :3 + k]) for k in ks]

fig, ax = plt.subplots(1, 3, figsize=(8.6, 3.0))
a = ax[0]
t = np.arange(Lf.shape[0]) * H
mn = lambda L: L.min(1)
lo = mn(lifts["svd"]).min() * 1.08
a.axhspan(lo, 0, color="#f4cccc", alpha=0.5)
a.plot(t, mn(Lf), "k", lw=2.2, label="full order")
a.plot(t, mn(lifts["nmf"]), "--", color="C0", lw=1.6, label="NMF lift")
a.plot(t, mn(lifts["svd"]), "-", color="C3", lw=1.2, label="SVD lift")
a.set_ylim(lo, -lo * 0.08)
a.text(0.02, 0.06, r"inadmissible: $\lambda<0$", transform=a.transAxes, fontsize=6.5, color="C3")
a.set_xlabel("t"); a.set_ylabel("smallest lifted contact force")
a.set_title("(a) sign of the lifted force, $p_\\lambda=20$"); a.legend(fontsize=6.5, loc="center right")
b = ax[1]
b.semilogy(sizes, err["nmf"], "o-", color="C0", label="error, NMF"); b.semilogy(sizes, err["svd"], "s-", color="C3", label="error, SVD")
b.semilogy(sizes, np.maximum(kkt["nmf"], 1e-17), "o--", color="C0", mfc="w", label="reduced KKT, NMF")
b.semilogy(sizes, np.maximum(kkt["svd"], 1e-17), "s--", color="C3", mfc="w", label="reduced KKT, SVD")
b.set_xlabel(r"$p=p_\lambda$"); b.set_title("(b) the reduced check cannot see it"); b.legend(fontsize=6.5)
c_ = ax[2]
c_.step(ks, rk, "o-", where="post", color="C2"); c_.axhline(rows, color="k", ls="--", lw=.8)
c_.text(0.2, rows + 0.4, f"{rows} constraints (full row rank)", fontsize=6.5)
c_.set_xlabel("unit vectors added to the 3-mode basis"); c_.set_ylabel(r"rank of the reduced constraints")
c_.set_title("(c) rank repair"); c_.set_ylim(0, rows + 3)
plt.tight_layout()
B = "packages_book/book_v3"
plt.savefig(os.path.join(B, "figures", "ch14_intro.pdf")); plt.savefig(os.path.join(B, "preview_figs", "ch14_intro_preview.png"), dpi=150)
def sci(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
srow = {p: ContactROM(mem.K, mem.B, Ux[:, :p], svd_dual_basis(Ltr[:, act], p), M=M).solve_dynamic(force, c, H, NS).lam.min() for p in sizes}
d = dict(crxNmfOutMin=sci(min(err["nmf"])), crxNmfOutMax=sci(max(err["nmf"])), crxSvdOutMin=sci(min(err["svd"])),
         crxSvdMinLamWorst=sci(min(srow.values())), crxKktMax=sci(max(max(kkt["nmf"]), max(kkt["svd"]))),
         crxRankBefore=str(rk[0]), crxBasisModes="3", crxPLam="20")
with open(os.path.join(B, "tables", "numbers_ch14_intro.tex"), "w") as fh:
    for k, v in d.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
print(d)

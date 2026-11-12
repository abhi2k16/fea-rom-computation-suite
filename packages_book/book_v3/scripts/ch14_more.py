r'''Chapter 14 figures: where the greedy loop samples (fig:rc-picks) and the indicator kappa_e (fig:rc-kappa); macros prefix cfm in tables/numbers_ch14_more.tex.
Run from the repository root with PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests:fea_engine/tests.'''
import os, warnings, numpy as np, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
from contact_fixtures import ObstacleMembrane, NonlinearChainContact
from rom_engine import ContactROM, ContactConditionIndicator, greedy_contact_sampling
from rom_engine.domain_partition import nonlinearity_indicator
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
B = "packages_book/book_v3"; PRE = B + "/preview_figs/"; FIGD = B + "/figures/"; D = {}
KB = 6      # size at which the greedy choice does not beat the best random draw (Table 14.3)
# ---------------- F4
mem = ObstacleMembrane(18)
cands = [(a, b) for a in np.linspace(0.3, 0.6, 6) for b in np.linspace(0.0, 0.1, 6)]
test = [(0.33, 0.02), (0.41, 0.07), (0.47, 0.04), (0.58, 0.09), (0.36, 0.095), (0.55, 0.01)]
cache = {}
def sol(i):
    if i not in cache: cache[i] = mem.solve(cands[i])
    return cache[i]
ref = [mem.solve(g) for g in test]
def make_rom(U, L):
    Q, _ = np.linalg.qr(U); Ul = L / np.maximum(np.linalg.norm(L, axis=0), 1e-300)
    rom = ContactROM(mem.K, mem.B, Q, Ul); return lambda p: rom.solve(mem.f, mem.obstacle(p)).u
test_err = lambda rom: np.mean([np.linalg.norm(rom(g) - r[0]) / np.linalg.norm(r[0]) for g, r in zip(test, ref)])
indic = ContactConditionIndicator(mem.K, mem.B, mem.f, weights=(1, 0, 0)); ind = lambda p, u: indic(u, mem.obstacle(p))
rom8, hist = greedy_contact_sampling(cands, lambda p: mem.solve(p)[:2], make_rom, ind, 8)
sel = list(hist.selected); print("greedy picks", sel, [cands[i] for i in sel])
rng = np.random.default_rng(0); draws = {}
for k in (2, 4, 6, 8):
    rr = []
    for _ in range(20):
        idx = rng.choice(len(cands), k, replace=False); S = [sol(i) for i in idx]
        rr.append((test_err(make_rom(np.array([s[0] for s in S]).T, np.array([s[1] for s in S]).T)), list(idx)))
    draws[k] = rr
best = min(draws[KB])[1]; worst = max(draws[KB])[1]
border = lambda i: (abs(A_[i, 0] - 0.3) < 1e-9 or abs(A_[i, 0] - 0.6) < 1e-9 or abs(A_[i, 1]) < 1e-9 or abs(A_[i, 1] - 0.1) < 1e-9)
A_ = np.array(cands)
S = [sol(sel[0])]; rom1 = make_rom(np.array([S[0][0]]).T, np.array([S[0][1]]).T)
I1 = np.array([ind(p, rom1(p)) for p in cands]); I1[sel[0]] = np.nan; I1 = I1.reshape(6, 6)
A = A_
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
a = ax[0]
im = a.pcolormesh(np.linspace(0.3, 0.6, 6), np.linspace(0, 0.1, 6), np.ma.masked_invalid(I1.T), shading="nearest", cmap="Blues", rasterized=True); plt.colorbar(im, ax=a, fraction=0.046, label="indicator after round 1")
a.plot(*np.array(test).T, "k^", ms=6, mfc="none", label="test obstacles")
for j, i in enumerate(sel): a.annotate(str(j + 1), A[i], color="C3", fontsize=8, ha="center", va="center", weight="bold", bbox=dict(boxstyle="circle,pad=0.15", fc="w", ec="C3"))
a.set_xlabel("position of bump 1"); a.set_ylabel("height of bump 2"); a.set_title("(a) greedy picks, in order", fontsize=8); a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2)); a.grid(False)
b = ax[1]
b.plot(A[:, 0], A[:, 1], ".", color="0.7", ms=4)
b.plot(*A[worst].T, "s", color="C1", mfc="none", ms=9, label=f"worst random draw ({KB})"); b.plot(*A[best].T, "o", color="C2", mfc="none", ms=9, label=f"best random draw ({KB})")
b.plot(*A[sel[:KB]].T, "x", color="C3", ms=7, label=f"greedy ({KB})"); b.plot(*np.array(test).T, "k^", ms=5, mfc="none", label="test obstacles")
b.set_xlabel("position of bump 1"); b.set_title(f"(b) the first {KB} picks against two random draws", fontsize=8); b.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); b.set_ylim(-0.005, 0.112)
plt.tight_layout(); plt.savefig(FIGD + "ch14_picks.pdf", dpi=300); plt.savefig(PRE + "ch14_f4_preview.png", dpi=150); plt.close()
D.update(cfmNPicks=str(len(sel)), cfmBorder=str(sum(border(i) for i in sel)), cfmBorderK=str(sum(border(i) for i in sel[:KB])), cfmKBorder=str(KB),
         cfmBestInterior=str(sum(not border(i) for i in best)), cfmWorstInterior=str(sum(not border(i) for i in worst)), cfmGrid="6")
print("picks border", D["cfmBorder"], "best interior", D["cfmBestInterior"], "worst interior", D["cfmWorstInterior"])
# ---------------- F5
E0 = 24
m = NonlinearChainContact(n=60, linear_below=E0)
states = np.array([m.solve(s_, -0.15)[0] for s_ in (0.5, 1.0, 2.0, 3.0)]).T
kappa = nonlinearity_indicator(m.elem_dofs, m.elem_force, m.elem_tangent, states)
obs = set(m.obs_nodes.tolist())
held = [e for e in range(E0, m.n) if len(m.elem_dofs[e]) == 2 and set(m.elem_dofs[e].tolist()) <= obs]
elong = max(abs(states[m.elem_dofs[e][1], j] - states[m.elem_dofs[e][0], j]) for e in held for j in range(states.shape[1]))
fig, ax = plt.subplots(figsize=(6.2, 3.1))
x = np.arange(m.n); kk = np.maximum(kappa, 1e-17)
ax.axvspan(-0.5, E0 - 0.5, color="C0", alpha=0.12, label="linear zone")
ax.semilogy(x, kk, "o-", ms=3, color="k", lw=0.8)
ax.semilogy(held, kk[held], "o", ms=7, mfc="none", mec="C3", label="between obstacle nodes (elongation at rounding level)")
ax.axhline(1e-6, color="C1", ls="--", lw=0.8, label="selection threshold")
ax.text(2, 2e-17, "zero (drawn at the floor)", fontsize=6.5, va="bottom")
ax.set_xlabel("element"); ax.set_ylabel(r"$\kappa_e$"); ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
plt.tight_layout(); plt.savefig(FIGD + "ch14_kappa.pdf"); plt.savefig(PRE + "ch14_f5_preview.png", dpi=150); plt.close()
zero = [e for e in held if kappa[e] == 0]; noise = [e for e in held if kappa[e] > 1e-6]
sel_nl = [e for e in range(E0, m.n) if kappa[e] > 1e-6]
sci = lambda v: "$%s\\times10^{%d}$" % tuple(f"{v:.1e}".split("e")[0:1] + [int(f"{v:.1e}".split("e")[1])])
D.update(cfmHeld=str(len(held)), cfmHeldZero=str(len(zero)), cfmHeldNoise=str(len(noise)), cfmElong=sci(elong),
         cfmKLo=f"{min(kappa[e] for e in noise):.2g}", cfmKHi=f"{max(kappa[e] for e in noise):.2g}", cfmHardSel=str(len(sel_nl)), cfmHard=str(m.n - E0))
# --- numbers derived from the Chapter 13/14 macro file (no digits typed in the text)
import re
_T = open(B + "/tables/numbers_ch13.tex").read()
def old(k):
    v = re.search(r"\\newcommand\{\\" + k + r"\}\{(.*?)\\xspace\}", _T).group(1)
    mm = re.match(r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$", v)
    return float(mm.group(1)) * 10 ** int(mm.group(2)) if mm else float(v)
rat = [old(f"cgr{k}Med") / old(f"cgr{k}Greedy") for k in ("Two", "Four", "Six", "Eight")]
car = [old(f"cecPlainNmf{k}Err") / old(f"cecCaNmf{k}Err") for k in ("In", "InB", "InC", "Out", "OutB")]
out = [4.0 / 3.0, 5.0 / 3.0, 0.30 / 0.25, 0.40 / 0.25]      # out-of-range load factors and depths over the training maxima (3 and 0.25) of Chapter 13
rid = max(old("crdFullErr"), old("crdThreeErr"), old("crdOneErr"))
D.update(cfmFacOutIn=f"{old('chdNmfThirtyOut') / old('chdNmfThirtyIn'):.1f}", cfmGreedyLo=f"{min(rat):.1f}", cfmGreedyHi=f"{max(rat):.1f}",
         cfmOutLo=f"{min(out):.1f}", cfmOutHi=f"{max(out):.1f}", cfmCaRatio=f"{max(car):.1f}",
         cfmRidErr="$%.1f\\times10^{%d}$" % (rid / 10 ** int(np.floor(np.log10(rid))), int(np.floor(np.log10(rid)))))
print(D)
with open(B + "/tables/numbers_ch14_more.tex", "w") as fh:
    for kk_, v in D.items(): fh.write(f"\\newcommand{{\\{kk_}}}{{{v}\\xspace}}\n")

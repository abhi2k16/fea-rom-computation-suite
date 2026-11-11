"""Figures of Chapter 13 (hyper-reduction): introduction figure, path of the fit, extrapolation in load.
Run from the repository root:
  PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests python packages_book/book_v3/scripts/ch13_figs.py
Writes figures/ch13_intro.pdf, ch13_path.pdf, ch13_extrap.pdf and tables/numbers_ch13_intro.tex (prefix chy), numbers_ch13_fig.tex (prefix chf).
No digits in macro names. Every value that the text quotes is asserted against tables/numbers_ch13.tex where that file has it."""
import os, re, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
from contact_fixtures import NonlinearChainContact
from rom_engine.hyper_reduction import ECSW, ecsw_training_matrix, ecsw_weights
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures"); TAB = os.path.join(HERE, "..", "tables")
plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": 0.3})
NI, NF = {}, {}
def sci(x):
    mnt, e = f"{x:.1e}".split("e"); return f"${mnt}\\times10^{{{int(e)}}}$"
_T = open(os.path.join(TAB, "numbers_ch13.tex")).read()
def old(k): return re.search(r"\\newcommand\{\\" + k + r"\}\{(.*?)\\xspace\}", _T).group(1)

m = NonlinearChainContact(); nel = len(m.elem_dofs)
SS, GG = (0.5, 1, 1.5, 2, 2.5, 3), (-0.08, -0.15, -0.25)
train = [(s, g) for s in SS for g in GG]
cache = {}
def solve(p):
    if p not in cache: cache[p] = m.solve(*p)[0]
    return cache[p]
Xu = np.array([solve(p) for p in train]).T
held = [(1.25, -0.12), (2.2, -0.2), (0.8, -0.1)]; out = [(4.0, -0.30), (5.0, -0.40)]
req = [e for e, g in enumerate(m.elem_dofs) if set(g.tolist()) & set(m.obs_nodes.tolist())]
def basis(X, r): return np.linalg.svd(X, full_matrices=False)[0][:, :r]
def mat(U, X): return ecsw_training_matrix(U, X, m.elem_dofs, m.elem_force)
def mk(U, w, info): return ECSW(U, m.elem_dofs, m.elem_force, w, m.elem_tangent, info)
def ferr(ec, U, p):
    q = U.T @ solve(p); ref = U.T @ m.internal_force(U @ q)
    return np.linalg.norm(ec.reduced_force(q) - ref) / np.linalg.norm(ref)
R = int(old("cecRank")); U = basis(Xu, R); G, b = mat(U, Xu)
w4, i4 = ecsw_weights(G, b, tol=1e-4); sel = np.nonzero(w4)[0]
assert len(sel) == int(old("cecFourEN")) and len(train) == int(old("cecNTrain")) and nel == int(old("cecNElem")) and len(req) == int(old("cecNReq"))

# ---------------- F0: selection, count against r, tolerance
tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]; nsel, res, eu, eo = [], [], [], []
for t in tols:
    w, info = ecsw_weights(G, b, tol=t); ec = mk(U, w, info); nsel.append(ec.n_selected); res.append(np.linalg.norm(G @ w - b) / np.linalg.norm(b))
    eu.append(max(ferr(ec, U, p) for p in held)); eo.append(max(ferr(ec, U, p) for p in out))
assert sci(eu[3]) == old("cecFourEErr") and sci(eo[3]) == old("cecFourEErrOut") and sci(eu[5]) == old("cecEightEErr") and sci(eo[5]) == old("cecEightEErrOut")
rs = [2, 3, 4, 6, 8, 10, 12]; cnt = []
for r in rs:
    Ur = basis(Xu, r); Gr, br = mat(Ur, Xu); wr, _ = ecsw_weights(Gr, br, tol=1e-4); cnt.append(int((wr > 0).sum()))
assert cnt[rs.index(R)] == len(sel)
cn = np.abs(G).max(axis=0); zero = [e for e in range(nel) if cn[e] < 1e-10]
rankG = int(np.linalg.matrix_rank(G, tol=1e-10 * np.linalg.norm(G, 2)))
sv = np.linalg.svd(G, compute_uv=False); sv = sv / sv[0]
NI.update(chyNZeroCols=str(len(zero)), chyNReqSel=str(len(set(sel) & set(req))), chyRankG=str(rankG), chyKne=str(len(train) * nel),
          chyCountLo=str(cnt[0]), chyCountHi=str(cnt[-1]), chyRLo=str(rs[0]), chyRHi=str(rs[-1]),
          chyResTolHigh=sci(res[0]), chyResFour=sci(res[3]), chyResEight=sci(res[5]), chyUnseenTolHigh=sci(eu[0]),
          chyOutLoads=" and ".join(f"{p[0]:g}" for p in out), chyOutDepths=" and ".join(f"{-p[1]:g}" for p in out),
          chyTolHigh="10^{-1}", chyTolFour="10^{-4}", chyTolEight="10^{-8}")
assert zero == list(range(zero[0], zero[-1] + 1)) and set(zero) <= set(req)
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.4))
a = ax[0]; a.axvspan(min(req) - .5, max(req) + .5, color="0.85", label="elements touching the obstacle")
a.plot(range(nel), np.ones(nel), ".", color="0.55", ms=2.5, label=f"naive: all {nel}, weight 1")
a.vlines(sel, 0, w4[sel], color="C0"); a.plot(sel, w4[sel], "o", color="C0", ms=3.5, label=f"ECSW: {len(sel)} elements")
a.set_ylim(0, 11.5); a.set_xlabel("element"); a.set_ylabel("weight"); a.set_title("(a) selected elements", fontsize=7); a.legend(fontsize=5.5, loc="upper right")
b_ = ax[1]; b_.plot(rs, [nel] * len(rs), "k--", label="naive"); b_.plot(rs, cnt, "o-", color="C0", label=r"ECSW, tolerance $10^{-4}$"); b_.set_ylim(0, 70)
for x, y in zip(rs, cnt): b_.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6)
b_.set_xlabel("reduced unknowns $r$"); b_.set_ylabel("elements evaluated per force"); b_.set_title("(b) cost of one force", fontsize=7); b_.legend(fontsize=6, loc="center right")
c = ax[2]; c.loglog(tols, res, "s-", color="C3", ms=3, label="training residual"); c.loglog(tols, eu, "o-", color="C0", ms=3, label="unseen states"); c.loglog(tols, eo, "^--", color="C1", ms=3, label="out of range")
c.invert_xaxis(); c.set_xlabel("tolerance"); c.set_ylabel("relative error"); c.set_title("(c) the tolerance", fontsize=7); c.legend(fontsize=5.5, loc="lower left")
sx = c.secondary_xaxis("top"); sx.set_xticks(tols); sx.set_xticklabels([str(n_) for n_ in nsel], fontsize=5.5); sx.set_xlabel("elements selected", fontsize=6, labelpad=1); sx.minorticks_off()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch13_intro.pdf"))

# ---------------- F1: path of the fit, training matrix, singular values
path = []; npath = []
for k in range(1, 16):
    w, info = ecsw_weights(G, b, tol=1e-14, max_elements=k); path.append(np.linalg.norm(G @ w - b) / np.linalg.norm(b)); npath.append(int((w > 0).sum()))
nend = max(npath)
assert [npath[i] for i in (8, 9, 11)] == [int(old("cecTwoEN")), int(old("cecThreeEN")), int(old("cecFourEN"))] and nend == int(old("cecEightEN"))
NF.update(chfPathEight=sci(path[7]), chfPathNine=sci(path[8]), chfPathTwelve=sci(path[11]), chfPathEnd=sci(path[-1]), chfNPathEnd=str(nend),
          chfSvLast=sci(sv[rankG - 1]), chfSvBeyond=sci(sv[rankG + 1]), chfIdxBeyond=str(rankG + 2), chfNNonzeroCols=str(nel - len(zero)),
          chfPathAbove=f"{0.1:g}")
assert path[6] > 0.1 > path[7] or path[7] < 0.1
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), gridspec_kw=dict(width_ratios=[1, 1.35, 0.9]))
a = ax[0]; a.semilogy(range(1, 16), path, "o-", ms=3)
for t, cn_, col in ((1e-2, 9, "C1"), (1e-3, 10, "C2"), (1e-4, 12, "C3"), (1e-8, 13, "C4")):
    a.axhline(t, color=col, ls=":", lw=.8); a.text(1.2, t, f"{t:g}", fontsize=5.5, va="bottom", ha="left", color=col)
a.set_xlabel("elements selected"); a.set_ylabel(r"$\|\mathbf{G}\mathbf{w}-\mathbf{b}\|/\|\mathbf{b}\|$"); a.set_title("(a) path of the fit", fontsize=7)
bb = ax[1]; bb.imshow(np.log10(np.abs(G) + 1e-12), aspect="auto", cmap="viridis", vmin=-8, vmax=np.log10(np.abs(G).max()), interpolation="nearest")
bb.plot(sel, np.full(len(sel), -2.5), "v", color="C3", ms=3, clip_on=False); bb.set_xlabel("element (column)"); bb.set_ylabel("row (training state, mode)")
bb.set_title(r"(b) $\log_{10}|\mathbf{G}|$, selected columns marked", fontsize=7); bb.grid(False)
cc = ax[2]; cc.semilogy(range(1, len(sv) + 1), np.maximum(sv, 1e-25), ".-", ms=3); cc.axvline(rankG + .5, color="C3", ls=":", lw=.8)
cc.set_xlabel("index"); cc.set_ylabel(r"$\sigma_k/\sigma_1$"); cc.set_title(r"(c) singular values of $\mathbf{G}$", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch13_path.pdf"))

# ---------------- F2: extrapolation in load factor at depth 0.15
S = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]; DEPTH = -0.15; Y = {}
fig, a = plt.subplots(figsize=(3.6, 2.5))
for tol, mk_, ls in ((1e-4, "o", "-"), (1e-8, "s", "--")):
    w, info = ecsw_weights(G, b, tol=tol); ec = mk(U, w, info); Y[tol] = [ferr(ec, U, (s, DEPTH)) for s in S]
    a.semilogy(S, Y[tol], mk_ + ls, ms=3, label=f"tolerance {tol:g}, {ec.n_selected} elements")
a.axvspan(0, max(SS), color="0.88", label="training range"); a.set_xlabel("load factor (obstacle depth 0.15)"); a.set_ylabel("error of the reduced force"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch13_extrap.pdf"))
y = dict(zip(S, Y[1e-4])); inside = [y[s] for s in S if s <= max(SS)]
NF.update(chfInLo=sci(min(inside)), chfInHi=sci(max(inside)), chfAtEdge=sci(y[3]), chfAtHalf=sci(y[3.5]), chfJump=f"{y[3.5] / y[3]:.0f}", chfAtFour=sci(y[4]), chfAtEight=sci(y[8]),
          chfDoubling=f"{y[8] / y[4]:.0f}", chfLoadMax=f"{max(S):g}", chfLoadTrainMax=f"{max(SS):g}", chfLoadRatio=f"{max(S) / max(SS):.1f}", chfDepth=f"{-DEPTH:g}",
          chfAtHalfLoad=f"{S[S.index(3.5)]:g}", chfAtFourLoad="4")
# Exercise 2: train to load factor 2 only, test at load factor 3
tr2 = [(s, g) for s in SS if s <= 2 for g in GG]; X2 = np.array([solve(p) for p in tr2]).T; U2 = basis(X2, R); G2, b2 = mat(U2, X2)
w2, i2 = ecsw_weights(G2, b2, tol=1e-4); ec2 = mk(U2, w2, i2); sel2 = set(np.nonzero(w2)[0].tolist())
NF.update(chfExTrainMax="2", chfExTest="3", chfExErr=sci(ferr(ec2, U2, (3, DEPTH))), chfExNSel=str(len(sel2)), chfExNew=str(len(set(sel.tolist()) - sel2)), chfExDropped=str(len(sel2 - set(sel.tolist()))),
          chfExRefErr=sci(y[3]), chfExNTrain=str(len(tr2)))
print({**NI, **NF})
for name, d in (("numbers_ch13_intro.tex", NI), ("numbers_ch13_fig.tex", NF)):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write("% generated by scripts/ch13_figs.py -- do not edit\n")
        for k, v in d.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")

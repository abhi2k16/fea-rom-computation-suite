"""Figures and numbers of Chapter 13 (hyper-reduction) from the lifted-state protocol of Sec. 13.2.
Run from the repository root:
  PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests python packages_book/book_v3/scripts/ch13_figs.py
Writes figures/ch13_intro.pdf, ch13_path.pdf, ch13_extrap.pdf, ch13_basis.pdf, ch13_train.pdf, tables/ch13_try_output.txt,
tables/numbers_ch13_intro.tex (prefix chy) and numbers_ch13_fig.tex (prefix chf). No digits in macro names.
ECSW is trained on the LIFTED states U U^T x_k (what the reduced model evaluates online), not on the full-order snapshots x_k.
Run scripts/ch13_assets.py --ecsw-only first, so that numbers_ch13.tex holds the macros asserted here."""
import os, re, io, contextlib, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import qmc
warnings.simplefilter("ignore")
from contact_fixtures import NonlinearChainContact
from rom_engine.hyper_reduction import ECSW, DEIM, ecsw_training_matrix, ecsw_weights
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
def lifted(U, X): return U @ (U.T @ X)
def mat(U, X): return ecsw_training_matrix(U, lifted(U, X), m.elem_dofs, m.elem_force)
def mk(U, w, info): return ECSW(U, m.elem_dofs, m.elem_force, w, m.elem_tangent, info)
def fref(U, q): return U.T @ m.internal_force(U @ q)
def rel(a, ref): return np.linalg.norm(a - ref) / np.linalg.norm(ref)
def ferr(ec, U, p):
    q = U.T @ solve(p); return rel(ec.reduced_force(q), fref(U, q))
def fit(tr, r, tol=1e-4):
    X = np.array([solve(p) for p in tr]).T; U = basis(X, r); G, b = mat(U, X); w, info = ecsw_weights(G, b, tol=tol)
    return U, G, b, w, mk(U, w, info), np.linalg.norm(G @ w - b) / np.linalg.norm(b)
R = int(old("cecRank")); U = basis(Xu, R); G, b = mat(U, Xu)
w4, i4 = ecsw_weights(G, b, tol=1e-4); sel = np.nonzero(w4)[0]
assert len(sel) == int(old("cecFourEN")) and len(train) == int(old("cecNTrain")) and nel == int(old("cecNElem")) and len(req) == int(old("cecNReq"))
projres = np.linalg.norm(Xu - lifted(U, Xu)) / np.linalg.norm(Xu)

# ---------------- tolerance sweep and basis size (F0)
tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]; nsel, res, eu, eo = [], [], [], []
for t in tols:
    w, info = ecsw_weights(G, b, tol=t); ec = mk(U, w, info); nsel.append(ec.n_selected); res.append(np.linalg.norm(G @ w - b) / np.linalg.norm(b))
    eu.append(max(ferr(ec, U, p) for p in held)); eo.append(max(ferr(ec, U, p) for p in out))
for k, i in (("TwoE", 1), ("ThreeE", 2), ("FourE", 3), ("SixE", 4), ("EightE", 5)):
    assert nsel[i] == int(old("cec" + k + "N")) and sci(eu[i]) == old("cec" + k + "Err") and sci(eo[i]) == old("cec" + k + "ErrOut"), k
rs = [2, 3, 4, 6, 8, 10, 12]; cnt = []; errr = {}
for r in rs:
    Ur, Gr, br, wr, ecr, _ = fit(train, r); cnt.append(ecr.n_selected); errr[r] = max(ferr(ecr, Ur, p) for p in held)
assert cnt[rs.index(R)] == len(sel) and sci(errr[R]) == old("cecFourEErr")
cn = np.abs(G).max(axis=0); zero = [e for e in range(nel) if cn[e] < 1e-10]
rankG = int(np.linalg.matrix_rank(G, tol=1e-10 * np.linalg.norm(G, 2))); sv = np.linalg.svd(G, compute_uv=False); sv = sv / sv[0]
assert zero == list(range(zero[0], zero[-1] + 1)) and set(zero) <= set(req)
NI.update(chyNZeroCols=str(len(zero)), chyNReqSel=str(len(set(sel) & set(req))), chyRankG=str(rankG), chyKne=str(len(train) * nel), chyNRows=str(G.shape[0]),
          chyErrRTen=sci(errr[10]), chyErrRTwelve=sci(errr[12]), chyRTen="10", chyRTwelve="12", chyCountLo=str(cnt[0]), chyCountHi=str(cnt[-1]), chyRLo=str(rs[0]), chyRHi=str(rs[-1]),
          chyResTolHigh=sci(res[0]), chyResFour=sci(res[3]), chyResEight=sci(res[5]), chyUnseenTolHigh=sci(eu[0]),
          chyOutLoads=" and ".join(f"{p[0]:g}" for p in out), chyOutDepths=" and ".join(f"{-p[1]:g}" for p in out),
          chyTolHigh="10^{-1}", chyTolFour="10^{-4}", chyTolEight="10^{-8}", chyProjRes=sci(projres))
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.8))
a = ax[0]; a.axvspan(min(req) - .5, max(req) + .5, color="0.85", label="elements touching the obstacle")
a.plot(range(nel), np.ones(nel), ".", color="0.55", ms=2.5, label=f"naive: all {nel}, weight 1")
a.vlines(sel, 0, w4[sel], color="C0"); a.plot(sel, w4[sel], "o", color="C0", ms=3.5, label=f"ECSW: {len(sel)} elements")
a.set_ylim(0, 11.5); a.set_xlabel("element"); a.set_ylabel("weight"); a.set_title("(a) selected elements", fontsize=7); a.legend(fontsize=5.5, loc="upper right")
b_ = ax[1]; b_.plot(rs, [nel] * len(rs), "k--", label="naive"); b_.plot(rs, cnt, "o-", color="C0", label=r"ECSW, tolerance $10^{-4}$"); b_.set_ylim(0, 70)
for x, y in zip(rs, cnt): b_.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6)
b_.set_xlabel("reduced unknowns $r$"); b_.set_ylabel("elements evaluated per force"); b_.set_title("(b) cost of one force", fontsize=7); b_.legend(fontsize=6, loc="center right")
c = ax[2]; c.loglog(tols, res, "s-", color="C3", ms=3, label="training residual"); c.loglog(tols, eu, "o-", color="C0", ms=3, label="unseen states"); c.loglog(tols, eo, "^--", color="C1", ms=3, label="out of range")
c.invert_xaxis(); c.set_xlabel("tolerance"); c.set_ylabel("relative error"); c.set_title("(c) the tolerance", fontsize=7, pad=16); c.legend(fontsize=5.5, loc="lower left")
sx = c.secondary_xaxis("top"); sx.set_xticks(tols); sx.set_xticklabels([str(n_) for n_ in nsel], fontsize=5.5); sx.set_xlabel("elements selected", fontsize=6, labelpad=1); sx.minorticks_off()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch13_intro.pdf"))

# ---------------- F1: path of the fit, training matrix, singular values
path, npath = [], []
for k in range(1, 22):
    w, info = ecsw_weights(G, b, tol=1e-14, max_elements=k); path.append(np.linalg.norm(G @ w - b) / np.linalg.norm(b)); npath.append(int((w > 0).sum()))
nend = max(npath); kend = npath.index(nend) + 1
for k, i in (("TwoE", 1), ("ThreeE", 2), ("FourE", 3)): assert npath[int(old("cec" + k + "N")) - 1] == int(old("cec" + k + "N"))
assert nend == int(old("cecEightEN"))
k0 = next(i for i, v in enumerate(path) if v < 0.1) + 1               # first number of elements with residual below 0.1
NF.update(chfKAbove=str(k0 - 1), chfKBelow=str(k0), chfPathKBelow=sci(path[k0 - 1]), chfPathTwoE=sci(path[int(old("cecTwoEN")) - 1]), chfPathFourE=sci(path[int(old("cecFourEN")) - 1]),
          chfNPathEnd=str(nend), chfPathEnd=sci(path[kend - 1]), chfSvLast=sci(sv[rankG - 1]), chfSvBeyond=sci(sv[rankG + 1]), chfIdxBeyond=str(rankG + 2), chfPathAbove="0.1")
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), gridspec_kw=dict(width_ratios=[1, 1.35, 0.9]))
a = ax[0]; a.semilogy(range(1, len(path) + 1), path, "o-", ms=3)
for t, cn_, col in ((1e-2, "TwoE", "C1"), (1e-3, "ThreeE", "C2"), (1e-4, "FourE", "C3"), (1e-8, "EightE", "C4")):
    a.axhline(t, color=col, ls=":", lw=.8); a.text(len(path) - .3, t, f"{t:g}", fontsize=5.5, va="bottom", ha="right", color=col)
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
          chfDoubling=f"{y[8] / y[4]:.0f}", chfLoadMax=f"{max(S):g}", chfLoadMin=f"{min(S):g}", chfLoadTrainMax=f"{max(SS):g}", chfLoadRatio=f"{max(S) / max(SS):.1f}", chfDepth=f"{-DEPTH:g}",
          chfAtHalfLoad="3.5", chfAtFourLoad="4")
# Exercise 2: train to load factor 2 only, test at load factor 3
tr2 = [(s, g) for s in SS if s <= 2 for g in GG]; U2, G2, b2, w2, ec2, _ = fit(tr2, R); sel2 = set(np.nonzero(w2)[0].tolist())
NF.update(chfExTrainMax="2", chfExTest="3", chfExErr=sci(ferr(ec2, U2, (3, DEPTH))), chfExNSel=str(len(sel2)), chfExNew=str(len(set(sel.tolist()) - sel2)),
          chfExDropped=str(len(sel2 - set(sel.tolist()))), chfExRefErr=sci(y[3]), chfExNTrain=str(len(tr2)))

# ---------------- F3: training states K, basis size r, error against the training residual at the stop
grid = lambda ss, gg: [(s, g) for s in ss for g in gg]
sets = {6: grid((1, 2, 3), (-0.08, -0.25)), 9: grid((1, 2, 3), GG), 12: grid((0.5, 1.5, 2.5, 3), GG), 18: train, 36: grid(SS, (-0.05, -0.08, -0.11, -0.15, -0.2, -0.25))}
assert len(sets[18]) == len(train)
resK = {}
for r in (5, 8):
    for K in sorted(sets):
        if K >= r: Ur, _, _, _, ecr, rr_ = fit(sets[K], r); resK[(r, K)] = (ecr.n_selected, max(ferr(ecr, Ur, p) for p in held), rr_)
resR = {}
for K in (18, 36):
    for r in range(2, 13): Ur, _, _, _, ecr, rr_ = fit(sets[K], r); resR[(K, r)] = (ecr.n_selected, max(ferr(ecr, Ur, p) for p in held), rr_)
pairs = [(v[2], v[1]) for v in resR.values()]; good = [(x, e) for x, e in pairs if x > 1e-9]
ratio = np.array([e / x for x, e in good])
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5))
for r, mk_ in ((5, "o-"), (8, "s--")):
    ks = [K for K in sorted(sets) if (r, K) in resK]; ax[0].loglog(ks, [resK[(r, K)][1] for K in ks], mk_, ms=3, label=f"$r={r}$")
ax[0].set_xlabel("training states $K$"); ax[0].set_ylabel("error at unseen states"); ax[0].legend(fontsize=6); ax[0].set_title("(a) training states", fontsize=7)
rr = list(range(2, 13))
for K, mk_ in ((18, "o-"), (36, "s--")): ax[1].semilogy(rr, [resR[(K, r)][1] for r in rr], mk_, ms=3, label=f"$K={K}$")
ax[1].set_xlabel("reduced unknowns $r$"); ax[1].set_ylabel("error at unseen states"); ax[1].legend(fontsize=6); ax[1].set_title("(b) basis size", fontsize=7)
for K, mk_ in ((18, "o"), (36, "s")): ax[2].loglog([max(resR[(K, r)][2], 1e-16) for r in rr], [resR[(K, r)][1] for r in rr], mk_, ms=3.5, label=f"$K={K}$")
lim = [1e-8, 1e-3]; ax[2].plot(lim, lim, "k:", lw=.8); ax[2].set_xlim(1e-16, 1e-3); ax[2].set_xlabel("training residual at the stop"); ax[2].set_ylabel("error at unseen states"); ax[2].legend(fontsize=6, loc="lower right"); ax[2].set_title("(c) error against residual", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch13_basis.pdf"))
NF.update(chfFixR="5", chfFixKLo=str(6), chfFixKHi=str(36), chfFixErrLo=sci(resK[(5, 6)][1]), chfFixErrHi=sci(resK[(5, 36)][1]),
          chfEightKA="9", chfEightKB="12", chfEightKC="18", chfEightKD="36", chfEightErrA=sci(resK[(8, 9)][1]), chfEightErrB=sci(resK[(8, 12)][1]),
          chfEightErrC=sci(resK[(8, 18)][1]), chfEightErrD=sci(resK[(8, 36)][1]), chfEightRes=sci(resR[(18, 8)][2]),
          chfRatioMin=f"{ratio.min():.1f}", chfRatioMax=f"{ratio.max():.1f}", chfNFitsGood=str(len(good)), chfNFitsAll=str(len(pairs)), chfNFitsExact=str(len(pairs) - len(good)),
          chfExactTol="10^{-9}")

# ---------------- F6: which states to train on (consistent protocol, tolerance 1e-4)
Q = U.T @ Xu; lo, hi = Q.min(axis=1), Q.max(axis=1); span = hi - lo
def pert(amp, reps, seed):
    g = np.random.default_rng(seed); return [Q[:, k] * (1 + amp * g.standard_normal(R)) for k in range(Q.shape[1]) for _ in range(reps)]
eqs = [Q[:, k] for k in range(Q.shape[1])]
SETS = {"Eq": ("equilibria", eqs), "PertLow": ("equilibria + 3 % perturbed", eqs + pert(0.03, 2, 1)), "PertHigh": ("equilibria + 10 % perturbed", eqs + pert(0.10, 2, 2)),
        "Box": ("simulation-free box", [lo - 0.1 * span + 1.2 * span * s_ for s_ in qmc.Sobol(d=R, scramble=True, seed=0).random(64)])}
def Gq(qs): return ecsw_training_matrix(U, np.array([U @ q for q in qs]).T, m.elem_dofs, m.elem_force)
AMPS = [0.0, 0.003, 0.01, 0.03, 0.1]
TEST = {a_: [U.T @ solve(p) * (1 + a_ * np.random.default_rng(100 + i).standard_normal(R)) for p in held for _ in range(5)] for i, a_ in enumerate(AMPS)}
OUTQ = [U.T @ solve(p) for p in out]
def qerr(f, qs): return max(rel(f(q), fref(U, q)) for q in qs)
assert sci(qerr(lambda q: mk(U, w4, i4).reduced_force(q), TEST[0.0])) == old("cecFourEErr")
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6)); TT = {}
for (key, (label, qs)), mk_ in zip(SETS.items(), ("o-", "s-", "^-", "d--")):
    Gs, bs = Gq(qs); w, info = ecsw_weights(Gs, bs, tol=1e-4); ec = mk(U, w, info); e_ = [qerr(ec.reduced_force, TEST[a_]) for a_ in AMPS]; eout = qerr(ec.reduced_force, OUTQ)
    TT[key] = dict(states=len(qs), n=ec.n_selected, err=e_, out=eout)
    NF.update({f"chfTr{key}States": str(len(qs)), f"chfTr{key}N": str(ec.n_selected), f"chfTr{key}ErrZero": sci(e_[0]), f"chfTr{key}ErrOne": sci(e_[2]), f"chfTr{key}ErrTen": sci(e_[4]), f"chfTr{key}ErrOut": sci(eout)})
    ax[0].semilogy(range(len(AMPS)), e_, mk_, ms=3, label=f"{label}: {ec.n_selected} el.")
    ns, es = [], []
    for tol in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
        w, info = ecsw_weights(Gs, bs, tol=tol); e2 = mk(U, w, info); ns.append(e2.n_selected); es.append(qerr(e2.reduced_force, TEST[0.1]))
    ax[1].semilogy(ns, es, mk_, ms=3, label=label)
ax[0].set_xticks(range(len(AMPS))); ax[0].set_xticklabels(["0", "0.3 %", "1 %", "3 %", "10 %"]); ax[0].set_xlabel("perturbation of the test states"); ax[0].set_ylabel("error of the reduced force")
ax[0].legend(fontsize=5.2); ax[0].set_title(r"(a) tolerance $10^{-4}$", fontsize=7); ax[1].set_xlabel("elements selected"); ax[1].set_ylabel("error at 10 % perturbed states"); ax[1].set_title("(b) tolerance sweep", fontsize=7); ax[1].legend(fontsize=5.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch13_train.pdf"))
NF.update(chfRatioEqTen=f"{TT['Eq']['err'][4] / TT['Eq']['err'][0]:.0f}", chfPertOne="1", chfPertTen="10", chfPertLow="3", chfPertHigh="10", chfBoxSamples=str(len(SETS["Box"][1])), chfBoxWiden="10", chfPertCopies="2")
# DEIM / QDEIM comparison on the same test states (force basis from the equilibrium force snapshots)
Fs = np.array([m.internal_force(U @ q) for q in eqs]).T; fsv = np.linalg.svd(Fs, compute_uv=False)
NF.update(chfFRank=str(int((fsv / fsv[0] > 1e-10).sum())), chfFSvTwo=sci(fsv[1] / fsv[0]))
NPT = 4
def touched(idx): return len({e for e, g in enumerate(m.elem_dofs) if set(g.tolist()) & set(int(i) for i in idx)})
for meth in ("deim", "qdeim"):
    d = DEIM.from_snapshots(Fs, n_points=NPT, method=meth); f = lambda q: d.approximate_reduced(U, m.internal_force(U @ q)[d.indices])
    e_ = [qerr(f, TEST[a_]) for a_ in AMPS]; tag = meth.capitalize()
    NF.update({f"chf{tag}Pts": str(NPT), f"chf{tag}Elems": str(touched(d.indices)), f"chf{tag}ErrZero": sci(e_[0]), f"chf{tag}ErrOne": sci(e_[2]), f"chf{tag}ErrTen": sci(e_[4])})

# training on the full-order snapshots themselves (not lifted): the mismatch the chapter warns about
Gx, bx = ecsw_training_matrix(U, Xu, m.elem_dofs, m.elem_force); wx, ix = ecsw_weights(Gx, bx, tol=1e-8); ecx = mk(U, wx, ix)
NF.update(chfSnapN=str(ecx.n_selected), chfSnapErr=sci(max(ferr(ecx, U, p_) for p_ in held)))
# ---------------- the listing of Sec. 13.4 is run exactly as printed in ch13.tex; its output is what the chapter shows
_tex = open(os.path.join(HERE, "..", "chapters", "ch13.tex"), encoding="utf-8").read()
_m = re.search(r"\\begin\{lstlisting\}\[caption=\{ECSW on the hardening chain[^\n]*\n(.*?)\\end\{lstlisting\}", _tex, re.S)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf): exec(compile(_m.group(1), "listing", "exec"), {"__name__": "listing"})
_lines = _buf.getvalue().strip().split("\n"); assert len(_lines) == 3
for ln, k in zip(_lines, ("TwoE", "FourE", "EightE")):
    mm = re.match(r"tol .*: +(\d+) of (\d+) elements, worst force error (\S+)", ln)
    assert int(mm.group(1)) == int(old("cec" + k + "N")) and float(mm.group(3)) == float(re.search(r"([\d.]+)\\times10\^\{(-?\d+)\}", old("cec" + k + "Err")).expand(r"\1e\2"))
open(os.path.join(TAB, "ch13_try_output.txt"), "w").write(_buf.getvalue())

print({**NI, **NF})
for name, d in (("numbers_ch13_intro.tex", NI), ("numbers_ch13_fig.tex", NF)):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write("% generated by scripts/ch13_figs.py -- do not edit\n")
        for k, v in d.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")

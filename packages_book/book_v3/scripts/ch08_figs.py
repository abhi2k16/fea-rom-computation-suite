# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""ch08_figs.py -- additional figures of Chapter 8 (see PLAN_ch08_figures.md). Uses the beam and designs of ch08_assets.py.
Writes figures/ch08_cloud.pdf (N2: the snapshot cloud, plane for a fixed load, curved sheet for a moving load) and tables/numbers_ch08_fig.tex."""
import os, sys, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("ch08_assets", os.path.join(HERE, "ch08_assets.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
FIG, TAB = A.FIG, A.TAB
NUM = {}


def sci(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"


def fig_cloud(b):
    mus3 = A.design(60, "olhs", 0)
    S3 = b.snapshots(mus3)
    S2 = np.column_stack([np.linalg.solve(m[0] * b.K1 + m[1] * b.K2, b.F) for m in mus3[:, :2]])
    fig, ax = plt.subplots(1, 3, figsize=(8.8, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.95]})
    sv = []
    for k, (S, col, lab, title) in enumerate(((S2, np.log(mus3[:, 0] / mus3[:, 1]), r"$\ln(\mu_1/\mu_2)$", "(a) fixed tip load"),
                                              (S3, mus3[:, 2], r"load position $\xi$", "(b) moving load"))):
        Sn = S / np.linalg.norm(S, axis=0)              # each snapshot scaled to unit length: the relative error is what matters
        U, s, _ = np.linalg.svd(Sn, full_matrices=False)
        Q = U[:, :3].T @ Sn
        sv.append(s / s[0])
        sc = ax[k].scatter(Q[1], Q[2], c=col, cmap="viridis", s=16)
        ax[k].set_xlim(-0.28, 0.28); ax[k].set_ylim(-0.14, 0.14); ax[k].axhline(0, color="0.6", lw=0.6)
        ax[k].set_xlabel("coordinate on mode 2"); ax[k].set_title(title, fontsize=9)
        cb = fig.colorbar(sc, ax=ax[k], pad=0.02); cb.set_label(lab, fontsize=8)
        NUM[f"cpdCloudThirdMax{'AB'[k]}"] = sci(float(np.abs(Q[2]).max()))
    ax[0].set_ylabel("coordinate on mode 3")
    for s_, st, lab in ((sv[0], "o-", "fixed tip load"), (sv[1], "s-", "moving load")):
        ax[2].semilogy(np.arange(1, 13), s_[:12], st, ms=4, label=lab)
    ax[2].set_xlabel("mode $k$"); ax[2].set_ylabel(r"$\sigma_k/\sigma_1$"); ax[2].set_title("(c) singular values", fontsize=9); ax[2].legend(fontsize=7)
    NUM["cpdCloudSvThreeA"] = sci(float(sv[0][2])); NUM["cpdCloudSvThreeB"] = f"{sv[1][2]:.3f}"
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch08_cloud.pdf")); fig.savefig("/tmp/ch08_cloud.png", dpi=120); plt.close(fig)
    print(NUM)


def fig_error(b):
    from rom_engine import PodBasis
    mus = A.design(60, "olhs", 0)
    S = b.snapshots(mus)
    test = A.design(40, "rand", 7)
    pod = PodBasis().fit(S, n_modes=24)
    sv = pod.singular_values
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    ranks = np.arange(1, 25)
    ge = np.array([A.tip_err(b, pod.V[:, :r], test) for r in ranks])
    # a representative unseen parameter: the one whose error at r = 6 is the median over the test set
    def err_at(mu, r):
        V = pod.V[:, :r]; x = b.solve(mu)
        q = np.linalg.solve(mu[0] * V.T @ b.K1 @ V + mu[1] * V.T @ b.K2 @ V, V.T @ b.load_at(mu[2]))
        return np.linalg.norm(V @ q - x) / np.linalg.norm(x)
    e6 = np.array([err_at(m, 6) for m in test]); k = int(np.argsort(e6)[len(e6) // 2]); mu = test[k]
    ref = b.solve(mu); xs = np.arange(1, b.n + 1) / b.n; tr = 2 * np.arange(b.n)
    fig = plt.figure(figsize=(8.8, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.3, 1])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    scale = np.abs(ref).max()
    a1.plot(xs, ref[tr] / scale, "k-", lw=2.2, label="full model")
    for r, st in ((2, "--"), (6, "-."), (12, ":")):
        V = pod.V[:, :r]
        q = np.linalg.solve(mu[0] * V.T @ b.K1 @ V + mu[1] * V.T @ b.K2 @ V, V.T @ b.load_at(mu[2]))
        x = V @ q
        a1.plot(xs, x[tr] / scale, st, lw=1.5, label=f"$r={r}$")
        a2.semilogy(xs, np.abs(x[tr] - ref[tr]) / scale + 1e-16, st, lw=1.5)
        NUM[f"cpdErrPanel{'Two Six Twelve'.split()[(2, 6, 12).index(r)]}"] = sci(err_at(mu, r))
    a1.set_ylabel("deflection / peak"); a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False); a1.set_xticklabels([])
    a1.set_title(f"(a) one unseen parameter: $\\mu_1={mu[0]:.1f}$, $\\mu_2={mu[1]:.1f}$, $\\xi={mu[2]:.2f}$", fontsize=8.5)
    a2.set_xlabel("position along the span $x/L$"); a2.set_ylabel("|error| / peak"); a2.set_ylim(1e-9, 1)
    a3.loglog(1 - energy[:24], ge, "o-", ms=4)
    for r in (1, 2, 3, 4, 6, 12, 24):
        a3.annotate(str(r), (1 - energy[r - 1], ge[r - 1]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    thr = 1e-4; a3.axvline(thr, color="C3", ls="--", lw=1)
    a3.text(thr * 1.3, ge.min() * 1.6, "99.99 % rule", color="C3", fontsize=7, rotation=90, va="bottom")
    a3.invert_xaxis()
    a3.set_xlabel("1 $-$ energy captured by $r$ modes"); a3.set_ylabel("worst error on 40 unseen parameters")
    a3.set_title("(b) the energy criterion against the unseen error (labels: $r$)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch08_error.pdf")); fig.savefig("/tmp/ch08_error.png", dpi=120); plt.close(fig)
    nrm2 = np.sum(S ** 2, axis=0); top = np.sort(nrm2)[::-1]
    NUM["cpdTopShareN"] = "5"; NUM["cpdTopShare"] = f"{100 * top[:5].sum() / top.sum():.0f}"
    V2 = pod.V[:, :2]; q2 = np.linalg.solve(mu[0] * V2.T @ b.K1 @ V2 + mu[1] * V2.T @ b.K2 @ V2, V2.T @ b.load_at(mu[2]))
    NUM["cpdPointMaxTwo"] = sci(float(np.abs((V2 @ q2)[tr] - ref[tr]).max() / scale))
    r_rule = int(np.argmax(1 - energy <= thr)) + 1
    NUM["cpdRuleRank"] = str(r_rule); NUM["cpdRuleEnergyGap"] = sci(1 - energy[r_rule - 1]); NUM["cpdRuleErr"] = sci(ge[r_rule - 1])
    NUM["cpdPanelMuA"] = f"{mu[0]:.1f}"; NUM["cpdPanelMuB"] = f"{mu[1]:.1f}"; NUM["cpdPanelXi"] = f"{mu[2]:.2f}"
    print({k_: v for k_, v in NUM.items() if "Rule" in k_ or "Panel" in k_ or "Err" in k_})


def fig_beam(b):
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig = plt.figure(figsize=(8.8, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1], hspace=0.5, wspace=0.3)
    a0, a1, a2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # (a) schematic
    a0.add_patch(Rectangle((0, -0.05), 0.5, 0.1, fc="#9ecae1", ec="k")); a0.add_patch(Rectangle((0.5, -0.05), 0.5, 0.1, fc="#fdae6b", ec="k"))
    a0.plot([0, 0], [-0.22, 0.22], "k", lw=3)
    for y in np.linspace(-0.2, 0.2, 7): a0.plot([-0.03, 0], [y - 0.03, y], "k", lw=0.8)
    a0.text(0.25, 0, r"region 1: $\mu_1\mathbf{K}_1$", ha="center", va="center", fontsize=9); a0.text(0.75, 0, r"region 2: $\mu_2\mathbf{K}_2$", ha="center", va="center", fontsize=9)
    xi = 0.61
    a0.add_patch(FancyArrowPatch((xi, 0.27), (xi, 0.06), arrowstyle="-|>", mutation_scale=12, color="C3", lw=1.6)); a0.text(xi + 0.015, 0.2, r"unit load at $\xi$", color="C3", fontsize=9)
    a0.add_patch(FancyArrowPatch((1.0, 0.27), (1.0, 0.06), arrowstyle="-|>", mutation_scale=12, color="0.4", lw=1.2)); a0.text(0.965, 0.3, "fixed tip load", color="0.4", fontsize=8, ha="right")
    a0.annotate("", (0, -0.3), (1, -0.3), arrowprops=dict(arrowstyle="<->", lw=0.8)); a0.text(0.5, -0.4, r"span $L=1$, $n=120$ elements, two equal regions", ha="center", fontsize=8.5)
    a0.set_xlim(-0.08, 1.05); a0.set_ylim(-0.5, 0.4); a0.axis("off"); a0.set_title(r"(a) clamped Euler–Bernoulli beam, $\mathbf{K}(\mu)=\mu_1\mathbf{K}_1+\mu_2\mathbf{K}_2$, $\mu_1,\mu_2\in[0.5,5]$, $\xi\in[0.1,0.9]$", fontsize=9)
    # (b) fixed tip load: closed form against the solve
    g = np.linspace(A.LO, A.HI, 60); M1, M2 = np.meshgrid(g, g)
    P, L = abs(A.TIP_LOAD), 1.0
    delta = P * L ** 3 / 24 * (7 / M1 + 1 / M2)
    cs = a1.contourf(M1, M2, delta, levels=14, cmap="viridis"); a1.contour(M1, M2, delta, levels=14, colors="w", linewidths=0.4)
    cb = fig.colorbar(cs, ax=a1); cb.set_label("tip deflection", fontsize=8)
    a1.set_xlabel(r"$\mu_1$"); a1.set_ylabel(r"$\mu_2$"); a1.set_title("(b) fixed tip load: smooth and explicit", fontsize=9)
    rng = np.random.default_rng(5); pts = rng.uniform(A.LO, A.HI, size=(30, 2))
    num_ = np.array([abs(b.solve(m)[b.tip]) for m in pts]) if False else np.array([abs(np.linalg.solve(m[0] * b.K1 + m[1] * b.K2, b.F)[b.tip]) for m in pts])
    ex_ = P * L ** 3 / 24 * (7 / pts[:, 0] + 1 / pts[:, 1])
    NUM["cpdBeamClosedErr"] = sci(float(np.max(np.abs(num_ - ex_) / ex_)))
    # (c) moving load
    xs = np.linspace(0.1, 0.9, 41)
    for (m1, m2), st in (((1.0, 1.0), "-"), ((5.0, 1.0), "--"), ((1.0, 5.0), ":"), ((5.0, 5.0), "-.")):
        a2.plot(xs, [abs(b.solve((m1, m2, x))[b.tip]) for x in xs], st, lw=1.6, label=f"$\\mu_1={m1:g},\\mu_2={m2:g}$")
    a2.set_xlabel(r"load position $\xi$"); a2.set_ylabel("tip deflection under a unit load"); a2.set_title("(c) moving load: smooth in $\\xi$", fontsize=9); a2.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "ch08_beam.pdf"), bbox_inches="tight"); fig.savefig("/tmp/ch08_beam.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print({k: v for k, v in NUM.items() if "Beam" in k})


def fig_svd(b):
    from matplotlib.patches import Rectangle
    from rom_engine import PodBasis
    mus = A.design(60, "olhs", 0); S = b.snapshots(mus)
    pod = PodBasis().fit(S, n_modes=8); sv = pod.singular_values
    tr = 2 * np.arange(b.n); rt = tr + 1
    for k_, nm in ((1, "Two"), (2, "Three"), (3, "Four")):
        NUM["cpdModeSv" + nm] = f"{sv[k_] / sv[0]:.3f}"
    for k_ in range(4):
        NUM["cpdRotShareMode" + "One Two Three Four".split()[k_]] = f"{100 * np.sum(pod.V[rt, k_] ** 2):.0f}"
    NUM["cpdRotShareS"] = f"{100 * np.sum(S[rt] ** 2) / np.sum(S ** 2):.0f}"
    CT, CTl, CR, CRl = "#6baed6", "#deebf7", "#fd8d3c", "#feedde"
    fig = plt.figure(figsize=(9.0, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], wspace=0.38, hspace=0.12)
    a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)

    def blk(x, y, w, h, fc, ec="k", lw=0.9):
        a0.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    H, y0, hh = 2.0, 1.45, 1.0
    # S: translation rows above rotation rows
    blk(0.0, y0 + hh, 0.8, hh, CTl); blk(0.0, y0, 0.8, hh, CRl)
    a0.text(0.4, y0 + 1.5, r"$\mathbf{S}$", ha="center", va="center", fontsize=11)
    a0.text(-0.07, y0 + 1.5, "translations $v$\n(length)", ha="right", va="center", fontsize=7.5, color="#08519c")
    a0.text(-0.07, y0 + 0.5, "rotations $\\theta$\n(radian)", ha="right", va="center", fontsize=7.5, color="#a63603")
    a0.text(0.4, y0 - 0.17, r"$n\times N$", ha="center", fontsize=8)
    a0.text(0.4, y0 + H + 0.12, "one column\nper snapshot", ha="center", fontsize=7.5)
    a0.text(1.0, y0 + 1.0, "=", fontsize=14, ha="center", va="center")
    ux, kept, wU = 1.2, 0.34, 0.8
    blk(ux, y0 + hh, kept, hh, CT); blk(ux, y0, kept, hh, CR)
    blk(ux + kept, y0 + hh, wU - kept, hh, CTl); blk(ux + kept, y0, wU - kept, hh, CRl)
    a0.text(ux + kept / 2, y0 + 1.5, r"$\mathbf{V}$", ha="center", va="center", fontsize=10)
    a0.text(ux + kept + (wU - kept) / 2, y0 + 1.5, r"$\mathbf{U}$", ha="center", va="center", fontsize=10)
    a0.text(ux + wU / 2, y0 - 0.17, r"$n\times N$", ha="center", fontsize=8)
    a0.text(ux + wU / 2, y0 + H + 0.12, "mode shapes\n(columns)", ha="center", fontsize=7.5)
    a0.annotate("kept: first $r$\ncolumns of U", (ux + kept / 2, y0 + H), (ux - 0.1, y0 + H + 0.75), fontsize=8, ha="center", color="C0", arrowprops=dict(arrowstyle="->", lw=0.8))
    sx, sz = ux + wU + 0.12, 0.8
    blk(sx, y0 + H - sz, sz, sz, "white")
    for k in range(5):
        a0.plot([sx + 0.11 + k * 0.14], [y0 + H - 0.11 - k * 0.14], "ko", ms=6 - k)
    a0.text(sx + 0.1, y0 + H - sz + 0.12, r"$\mathbf{\Sigma}$", fontsize=10, color="0.3")
    a0.text(sx + sz / 2, y0 + H - sz - 0.17, r"$N\times N$", ha="center", fontsize=8)
    a0.text(sx + sz / 2, y0 + H + 0.12, "singular\nvalues", ha="center", fontsize=7.5)
    wx = sx + sz + 0.1
    blk(wx, y0 + H - sz, sz, sz, "#f0f0f0"); a0.text(wx + sz / 2, y0 + H - sz / 2, r"$\mathbf{W}^T$", ha="center", va="center", fontsize=10)
    a0.text(wx + sz / 2, y0 + H - sz - 0.17, r"$N\times N$", ha="center", fontsize=8)
    a0.text(wx + sz / 2, y0 + H + 0.12, "mode mix per\nsnapshot", ha="center", fontsize=7.5)
    a0.text(0.0, 0.78, "Eckart–Young: $\\mathbf{V}\\mathbf{V}^T\\mathbf{S}$ is the best rank-$r$ approximation,\nwith $\\|\\mathbf{S}-\\mathbf{V}\\mathbf{V}^T\\mathbf{S}\\|_F^2=\\sum_{k>r}\\sigma_k^2$.", fontsize=8, va="center")
    a0.text(0.0, 0.12, "Rows are drawn grouped by type; in the code they\nalternate $(v,\\theta)$ node by node. The Euclidean SVD\nadds length$^2$ and radian$^2$ in one sum (weighted POD).", fontsize=7.5, va="center", color="0.3")
    a0.set_xlim(-0.75, wx + sz + 0.05); a0.set_ylim(-0.3, y0 + H + 1.2); a0.axis("off"); a0.set_title("(a) the SVD of the snapshot matrix, by type of unknown", fontsize=9)
    # (b) both parts of the first four modes (unit-norm modes, true component values)
    xs = np.arange(1, b.n + 1) / b.n
    for k in range(4):
        v = pod.V[tr, k]; t = pod.V[rt, k]
        sg = 1.0 if v[np.argmax(np.abs(v))] > 0 else -1.0
        lab = f"mode {k + 1}"
        a1.plot(xs, sg * v, lw=1.5, label=lab); a2.plot(xs, sg * t, lw=1.5)
    a1.set_ylabel("translation part $v$"); a2.set_ylabel("rotation part $\\theta$"); a2.set_xlabel("position along the span $x/L$")
    plt.setp(a1.get_xticklabels(), visible=False)
    for a_ in (a1, a2): a_.axvline(0.5, color="0.6", ls=":", lw=1); a_.axhline(0, color="0.85", lw=0.6)
    a1.set_title("(b) the first four POD modes, by type of unknown", fontsize=9, pad=22)
    a1.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.0, handlelength=1.2)
    a2.annotate("kink at the\nregion interface", (0.5, 0.06), (0.62, 0.15), fontsize=7, color="0.35", arrowprops=dict(arrowstyle="->", lw=0.6, color="0.5"))
    fig.savefig(os.path.join(FIG, "ch08_svd.pdf"), bbox_inches="tight"); fig.savefig("/tmp/ch08_svd.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_designs(b):
    import itertools
    from scipy.spatial.distance import pdist
    from matplotlib.patches import Rectangle
    N = 16
    g = (np.arange(4) + 0.5) / 4
    grid = np.array([(x, y) for x in g for y in g])
    rnd = np.random.default_rng(11).uniform(size=(N, 2))
    olhs = A.optimal_lhs(N, 2, rng=np.random.default_rng(0))
    top = ((r"(a) grid, $4\times4$", grid), ("(b) uniform random", rnd), ("(c) optimal Latin hypercube", olhs))
    fig = plt.figure(figsize=(8.8, 6.6))
    outer = fig.add_gridspec(2, 3, height_ratios=[1.12, 1], hspace=0.5, wspace=0.28)
    for k, (title, P) in enumerate(top):
        sub = outer[0, k].subgridspec(2, 2, width_ratios=[1, 0.09], height_ratios=[1, 0.09], wspace=0.04, hspace=0.04)
        a = fig.add_subplot(sub[0, 0]); sx = fig.add_subplot(sub[1, 0]); sy = fig.add_subplot(sub[0, 1])
        Q = A.LO + (A.HI - A.LO) * P
        a.scatter(Q[:, 0], Q[:, 1], s=24, color="C%d" % k, zorder=3)
        for t in np.linspace(A.LO, A.HI, N + 1)[1:-1]:
            a.axvline(t, color="0.92", lw=0.5, zorder=0); a.axhline(t, color="0.92", lw=0.5, zorder=0)
        a.set_xlim(A.LO, A.HI); a.set_ylim(A.LO, A.HI); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=9)
        bins = np.minimum((P * N).astype(int), N - 1)
        used = [set(bins[:, 0]), set(bins[:, 1])]
        for ax_, u, horiz in ((sx, used[0], True), (sy, used[1], False)):
            for c in range(N):
                xy, w, h = ((c, 0), 1, 1) if horiz else ((0, c), 1, 1)
                ax_.add_patch(Rectangle(xy, w, h, fc="C%d" % k if c in u else "white", ec="0.6", lw=0.5))
            ax_.set_xlim(0, N if horiz else 1); ax_.set_ylim(0, 1 if horiz else N); ax_.axis("off")
        sx.text(N / 2, -0.9, f"$\\mu_1$ bins used: {len(used[0])} of {N}", ha="center", va="top", fontsize=7.5, transform=sx.transData)
        sy.text(1.3, N / 2, f"$\\mu_2$: {len(used[1])} of {N}", rotation=90, ha="left", va="center", fontsize=7.5)
        tag = ("Grid", "Rand", "Olhs")[k]
        NUM["cpdDes" + tag + "UsedA"] = str(len(used[0])); NUM["cpdDes" + tag + "UsedB"] = str(len(used[1])); NUM["cpdDes" + tag + "MinDist"] = f"{float(pdist(P).min()):.2f}"
    # bottom row: how a Latin hypercube is built, on a 5 x 5 board
    M = 5
    def board(a, perm, title, note, col):
        for t in range(M + 1):
            a.axvline(t, color="0.7", lw=0.8); a.axhline(t, color="0.7", lw=0.8)
        pts = np.array([(r + 0.5, perm[r] + 0.5) for r in range(M)])
        a.scatter(pts[:, 0], pts[:, 1], s=60, color=col, zorder=3)
        D = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(M) * 9; i, j = np.unravel_index(np.argmin(D), D.shape)
        a.plot(pts[[i, j], 0], pts[[i, j], 1], "-", color="k", lw=1.4, zorder=2)
        a.set_xlim(0, M); a.set_ylim(0, M); a.set_xticks(np.arange(M) + 0.5); a.set_xticklabels(range(1, M + 1), fontsize=7); a.set_yticks(np.arange(M) + 0.5); a.set_yticklabels(range(1, M + 1), fontsize=7); a.tick_params(length=0); a.set_xlabel("bin of $\\mu_1$", fontsize=8, labelpad=2); a.set_ylabel("bin of $\\mu_2$", fontsize=8, labelpad=2); a.set_aspect("equal")
        a.set_title(title, fontsize=9); a.text(M / 2, -1.25, note, ha="center", va="top", fontsize=8)
        return D.min() / M
    bestd, bestp = -1, None
    for perm in itertools.permutations(range(M)):
        pts = np.array([(r + 0.5, perm[r] + 0.5) for r in range(M)])
        d = pdist(pts).min()
        if d > bestd + 1e-12: bestd, bestp = d, perm
    a1 = fig.add_subplot(outer[1, 0]); a2 = fig.add_subplot(outer[1, 1]); a3 = fig.add_subplot(outer[1, 2])
    d1 = board(a1, (2, 4, 1, 3, 0), "(d) the rule, $N=5$ points", "one point in every row and\nevery column of the board", "C2")
    d2 = board(a2, (0, 1, 2, 3, 4), "(e) allowed, but poor", f"all on a diagonal: $\\mu_1$ and $\\mu_2$ move\ntogether; closest pair (line) {np.sqrt(2) / M:.2f}", "C3")
    d3 = board(a3, bestp, "(f) the best of the 120 allowed", f"closest pair (line) {bestd / M:.2f}:\nthe 'maximin' choice", "C2")
    NUM["cpdBoardN"] = str(M); NUM["cpdBoardAll"] = str(len(list(itertools.permutations(range(M))))); NUM["cpdBoardDiag"] = f"{np.sqrt(2) / M:.2f}"; NUM["cpdBoardBest"] = f"{bestd / M:.2f}"
    fig.savefig(os.path.join(FIG, "ch08_designs.pdf"), bbox_inches="tight"); fig.savefig("/tmp/ch08_designs.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print({k_: v for k_, v in NUM.items() if "Des" in k_ or "Board" in k_})


def fig_intro(b):
    import re
    from matplotlib.patches import Rectangle, FancyArrowPatch
    rows = [[float(x) for x in re.findall(r"[\d.]+", ln.split("\\\\")[0])] for ln in open(os.path.join(TAB, "ch08_cost.tex")).read().split("\n") if re.match(r"\d+ &", ln)]
    off_ms, red_us, full_us = rows[0][2], rows[0][3], rows[0][4]
    nf = b.nf; r = A.COST_MODES
    NUM["cpdTabOffMs"] = f"{off_ms:.0f}"; NUM["cpdTabRedUs"] = f"{red_us:.0f}"; NUM["cpdTabFullUs"] = f"{full_us:.0f}"
    NUM["cpdEntriesFull"] = f"{nf * nf:,}".replace(",", "\\,"); NUM["cpdEntriesRed"] = str(r * r); NUM["cpdSizeRatio"] = f"{nf / r:.0f}"
    fig = plt.figure(figsize=(9.0, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.2, 1], hspace=0.28, wspace=0.12)
    a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0], projection="3d"); a2 = fig.add_subplot(gs[1, 1])
    # (a) workflow
    def box(x, y, w, h, fc, txt, fs=8, ec="k"):
        a0.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0)); a0.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)
    def arr(x0, y0_, x1, y1_):
        a0.add_patch(FancyArrowPatch((x0, y0_), (x1, y1_), arrowstyle="-|>", mutation_scale=9, color="0.25", lw=1.1))
    a0.text(0.0, 4.55, "without reduction: every query is a full solve", fontsize=8.5, fontweight="bold", va="center")
    box(0.0, 3.5, 1.5, 0.7, "#f0f0f0", r"parameters $\mu_i$")
    for k in range(3):
        a0.add_patch(Rectangle((2.2 + 0.12 * (2 - k), 3.5 + 0.1 * (2 - k)), 2.8, 0.7, fc="#deebf7", ec="#08519c", lw=1.0, zorder=2 - k))
    a0.text(3.72, 3.95, "full solve\n" + r"$\mathbf{K}(\mu_i)\mathbf{u}_i=\mathbf{f}$", ha="center", va="center", fontsize=8, zorder=5)
    arr(1.5, 3.85, 2.2, 3.85); arr(5.3, 3.85, 6.3, 3.85); box(6.3, 3.5, 1.5, 0.7, "#f0f0f0", r"response $\mathbf{u}_i$")
    a0.text(8.1, 3.85, f"{full_us:.0f} µs per query,\nrepeated $N_{{\\mathrm{{q}}}}$ times", fontsize=8, va="center", color="#08519c")
    a0.text(0.0, 2.7, "with a reduced-order model: build once, then every query is a small solve", fontsize=8.5, fontweight="bold", va="center")
    box(2.2, 1.85, 2.8, 0.5, "#e6e0f3", f"offline build, once: {off_ms:.0f} ms\n(sample, SVD, project)", fs=7.5, ec="#54278f")
    arr(3.6, 1.85, 3.6, 1.45)
    box(0.0, 0.6, 1.5, 0.7, "#f0f0f0", r"parameters $\mu_i$")
    for k in range(3):
        a0.add_patch(Rectangle((2.2 + 0.12 * (2 - k) + 0.2, 0.6 + 0.1 * (2 - k)), 2.4, 0.7, fc="#e6e0f3", ec="#54278f", lw=1.0, zorder=2 - k))
    a0.text(3.72 + 0.2, 1.05, f"small solve\n${r}\\times{r}$ system", ha="center", va="center", fontsize=8, zorder=5)
    arr(1.5, 0.95, 2.4, 0.95); arr(5.3, 0.95, 6.3, 0.95); box(6.3, 0.6, 1.5, 0.7, "#f0f0f0", r"response $\mathbf{u}_i$")
    a0.text(8.1, 0.95, f"{red_us:.0f} µs per query,\nrepeated $N_{{\\mathrm{{q}}}}$ times", fontsize=8, va="center", color="#54278f")
    a0.set_xlim(-0.1, 10.2); a0.set_ylim(0.2, 4.9); a0.axis("off"); a0.set_title("(a) many queries: the same question asked $N_{\\mathrm{q}}$ times", fontsize=9)
    # (b) schematic: the set of solutions, the subspace of the basis, and the error of one solution
    a1.view_init(elev=20, azim=-64)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
    a1.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color="C4", shade=False, linewidth=0)
    f_ = lambda x_, y_: 0.20 + 0.30 * x_ ** 2 + 0.10 * y_
    th = np.linspace(-0.9, 0.9, 30); gx, gy = np.meshgrid(th, th)
    a1.plot_surface(gx, gy, f_(gx, gy), alpha=0.45, color="C0", shade=True, linewidth=0)
    rng_ = np.random.default_rng(3); px, py = rng_.uniform(-0.85, 0.85, 16), rng_.uniform(-0.85, 0.85, 16)
    a1.scatter(px, py, f_(px, py), s=12, color="k", depthshade=False)
    ux_, uy_ = 0.7, 0.05; uz_ = f_(ux_, uy_)
    a1.plot([ux_, ux_], [uy_, uy_], [uz_, 0], color="C3", lw=1.8); a1.scatter([ux_], [uy_], [uz_], s=50, color="C3", depthshade=False); a1.scatter([ux_], [uy_], [0], s=36, color="C4", depthshade=False)
    a1.text(ux_ + 0.05, uy_, uz_ + 0.08, r"$\mathbf{u}(\mu)$", fontsize=9, color="C3")
    a1.text(ux_ + 0.1, uy_ + 0.12, -0.02, r"$\mathbf{V}\mathbf{q}$", fontsize=9, color="C4")
    a1.text(ux_ + 0.12, uy_ - 0.05, uz_ / 2, "error", fontsize=8, color="C3")
    a1.text(-0.95, -0.95, -0.02, "plane = span of the\nreduced basis $\\mathbf{V}$", fontsize=7.5, color="C4", ha="left", va="top")
    a1.text(-0.9, 0.3, 0.62, "all solutions $\\{\\mathbf{u}(\\mu)\\}$\n(dots: snapshots)", fontsize=7.5, color="C0", ha="left")
    a1.set_axis_off(); a1.set_xlim(-1, 1); a1.set_ylim(-1, 1); a1.set_zlim(-0.1, 0.7)
    a1.set_title("(b) the solutions lie near a small set (schematic)", fontsize=9, pad=-6); a1.set_box_aspect((1, 1, 0.6))
    # (c) sizes to scale
    a2.add_patch(Rectangle((0, 0), 1, 1, fc="#deebf7", ec="#08519c", lw=1.2)); a2.text(0.5, 0.5, f"{nf}$\\times${nf}\n" + r"$\mathbf{K}\mathbf{u}=\mathbf{f}$" + f"\n{nf * nf:,} entries", ha="center", va="center", fontsize=9)
    sc = r / nf
    a2.add_patch(Rectangle((1.25, 0.5 - sc / 2), sc, sc, fc="#e6e0f3", ec="#54278f", lw=1.2))
    a2.annotate("", (1.25, 0.5), (1.02, 0.5), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    a2.text(1.25 + sc / 2, 0.5 - sc / 2 - 0.05, f"{r}$\\times${r}\n{r * r} entries", ha="center", va="top", fontsize=8.5, color="#54278f")
    a2.text(0.5, -0.12, f"to scale: {nf // r}$\\times$ fewer rows and columns,\n{nf * nf // (r * r):,}$\\times$ fewer entries", ha="center", va="top", fontsize=8)
    a2.set_xlim(-0.05, 1.7); a2.set_ylim(-0.4, 1.12); a2.set_aspect("equal"); a2.axis("off"); a2.set_title("(c) the size of the system", fontsize=9)
    NUM["cpdEntryRatio"] = f"{nf * nf // (r * r):,}".replace(",", "\\,")
    fig.savefig(os.path.join(FIG, "ch08_intro.pdf"), bbox_inches="tight"); fig.savefig("/tmp/ch08_intro.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print({k_: v for k_, v in NUM.items() if k_.startswith("cpdTab") or "Entr" in k_ or "SizeRatio" in k_})


def fig_projection(b):
    from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
    from rom_engine import PodBasis
    NAVY, ORG, GRN, VIO = "#1b3a5c", "#d9822b", "#2e8b57", "#6a3d9a"
    fig = plt.figure(figsize=(9.0, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.2, width_ratios=[1.2, 1], height_ratios=[1.3, 1])
    a, bb, c, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # ---- (a) geometry: error and residual
    K = np.array([[4.0, 3.2], [3.2, 3.0]]); th = np.radians(8); v = np.array([np.cos(th), np.sin(th)])
    u = np.array([2.2, 2.1]); f = K @ u
    E = v * (v @ u); G = v * (v @ f) / (v @ K @ v)
    eE, eG = u - E, u - G; rG = f - K @ G
    a.set_aspect("equal"); a.axis("off")
    t = np.array([-1.4, 5.6]); a.plot(t * v[0], t * v[1], color=VIO, lw=2.4); a.text(5.6 * v[0] + 0.05, 5.6 * v[1] + 0.1, r"span($\mathbf{V}$)", color=VIO, fontsize=8.5, ha="left")
    a.plot([u[0], E[0]], [u[1], E[1]], color=ORG, lw=2); a.plot([u[0], G[0]], [u[1], G[1]], color=GRN, lw=2)
    nn = eE / np.linalg.norm(eE); s_ = 0.15
    a.add_patch(Polygon([E + s_ * nn, E + s_ * nn + s_ * v, E + s_ * v], closed=False, fill=False, ec=ORG, lw=0.9))
    rr = rG / np.linalg.norm(rG) * 1.1
    a.add_patch(FancyArrowPatch(tuple(G), tuple(G + rr), arrowstyle="-|>", mutation_scale=11, color=GRN, lw=1.6, ls="--"))
    a.add_patch(Polygon([G + s_ * rr / 1.1, G + s_ * rr / 1.1 + s_ * v, G + s_ * v], closed=False, fill=False, ec=GRN, lw=0.9))
    for pt, col in ((u, NAVY), (E, ORG), (G, GRN)): a.plot(*pt, "o", color=col, ms=6, zorder=5)
    a.text(u[0] - 0.12, u[1] + 0.05, r"full solution $\mathbf{u}$", color=NAVY, fontsize=9, ha="right", fontweight="bold")
    a.annotate("Euclidean projection $\\mathbf{V}\\mathbf{V}^T\\mathbf{u}$\nerror $\\perp$ subspace\n(needs $\\mathbf{u}$)", E, (E[0] - 2.3, E[1] - 2.0), color=ORG, fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=ORG, lw=0.8))
    a.annotate("Galerkin solution $\\mathbf{V}\\mathbf{q}$\nresidual $\\mathbf{r}=\\mathbf{f}-\\mathbf{K}\\mathbf{V}\\mathbf{q}\\perp$ subspace\n(needs only $\\mathbf{K},\\mathbf{f}$)", G, (G[0] + 0.9, G[1] - 2.2), color=GRN, fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=GRN, lw=0.8))
    a.text(G[0] + rr[0] + 0.08, G[1] + rr[1] + 0.02, "r (scaled)", color=GRN, fontsize=8, ha="left")
    a.set_xlim(-3.4, 6.6); a.set_ylim(-3.2, 3.2); a.set_title("(a) two projections of $\\mathbf{u}$ onto the same subspace", fontsize=9, loc="left")
    # ---- (b) the two errors along the subspace
    q = np.linspace(-0.5, 4.6, 300); qE, qG = v @ u, (v @ f) / (v @ K @ v)
    eK = np.array([0.5 * (u - v * x) @ K @ (u - v * x) for x in q]); e2 = np.array([0.5 * (u - v * x) @ (u - v * x) for x in q])
    bb.plot(q, e2, color=ORG, lw=2, label=r"$\frac{1}{2}\|\mathbf{u}-\mathbf{V}\mathbf{q}\|_2^2$ (Euclidean)"); bb.plot(q, eK, color=GRN, lw=2, label=r"$\frac{1}{2}\|\mathbf{u}-\mathbf{V}\mathbf{q}\|_K^2$ (energy)")
    bb.axvline(qE, color=ORG, ls=":", lw=1); bb.axvline(qG, color=GRN, ls=":", lw=1)
    bb.plot([qE], [0.5 * (u - v * qE) @ (u - v * qE)], "o", color=ORG); bb.plot([qG], [0.5 * (u - v * qG) @ K @ (u - v * qG)], "o", color=GRN)
    bb.text(qE - 0.04, bb.get_ylim()[1] * 0.0 + 0.25, "best in\nEuclidean norm", color=ORG, fontsize=8, ha="right", va="bottom"); bb.text(qG - 0.08, 2.2, "best in\nenergy norm", color=GRN, fontsize=8, ha="right", va="bottom")
    bb.set_xlabel(r"coefficient $q$ along the subspace ($\mathbf{V}\mathbf{q}=q\,\mathbf{v}$)"); bb.set_ylabel("squared error / 2"); bb.set_ylim(-0.3, 12); bb.legend(fontsize=7.5, loc="upper center", frameon=False)
    bb.set_title("(b) each error is smallest at a different point", fontsize=9, loc="left")
    # ---- (c) what each needs
    c.axis("off"); c.set_xlim(0, 10); c.set_ylim(0, 6)
    def card(y, col, title, body):
        c.add_patch(FancyBboxPatch((0.2, y), 9.6, 2.45, boxstyle="round,pad=0.05,rounding_size=0.2", fc=col + "1a", ec=col, lw=1.4))
        c.text(0.5, y + 2.0, title, fontsize=9, fontweight="bold", color=col, va="center"); c.text(0.5, y + 0.9, body, fontsize=8, va="center", linespacing=1.45)
    card(3.3, ORG, "Euclidean projection", r"$\hat{\mathbf{u}}=\mathbf{V}\mathbf{V}^T\mathbf{u}$ needs the full solution $\mathbf{u}$:" + "\nit cannot be computed online. It is the best\nthat the basis $\\mathbf{V}$ could do (the POD error).")
    card(0.4, GRN, "Galerkin projection", r"$(\mathbf{V}^T\mathbf{K}\mathbf{V})\,\mathbf{q}=\mathbf{V}^T\mathbf{f}$ needs only $\mathbf{K}$ and $\mathbf{f}$:" + "\nan $r\\times r$ solve per query. For symmetric positive\ndefinite $\\mathbf{K}$ it minimises the energy error.")
    c.set_title("(c) what each projection needs", fontsize=9, loc="left")
    # ---- (d) measured on the beam
    mus = A.design(60, "olhs", 0); S = b.snapshots(mus); test = A.design(40, "rand", 7)
    r = 6; V = PodBasis().fit(S, n_modes=24).V[:, :r]
    e2p, e2g, eKp, eKg = [], [], [], []
    for mu in test:
        x = b.solve(mu); Kf = mu[0] * b.K1 + mu[1] * b.K2
        q_ = np.linalg.solve(V.T @ Kf @ V, V.T @ (Kf @ x)); xg = V @ q_; xp = V @ (V.T @ x)
        nk = lambda e: float(np.sqrt(e @ Kf @ e))
        e2p.append(np.linalg.norm(x - xp) / np.linalg.norm(x)); e2g.append(np.linalg.norm(x - xg) / np.linalg.norm(x))
        eKp.append(nk(x - xp) / nk(x)); eKg.append(nk(x - xg) / nk(x))
    e2p, e2g, eKp, eKg = map(np.array, (e2p, e2g, eKp, eKg))
    xi = test[:, 2]
    d.axhline(1.0, color="0.4", lw=1)
    d.scatter(xi, e2g / e2p, s=18, color=ORG, label="Euclidean norm", zorder=3); d.scatter(xi, eKg / eKp, s=18, color=GRN, marker="s", label="energy norm", zorder=3)
    d.set_yscale("log"); d.set_yticks([0.6, 0.8, 1, 1.4, 1.8]); d.set_yticklabels(["0.6", "0.8", "1", "1.4", "1.8"]); d.minorticks_off(); d.set_xlabel(r"load position $\xi$ of the unseen parameter"); d.set_ylabel("error of Galerkin / error of projection")
    d.legend(fontsize=7.5, loc="lower right", frameon=False); d.text(0.1, 1.75, r"$\uparrow$ Galerkin worse", fontsize=7.5, color="0.35"); d.text(0.1, 0.66, r"$\downarrow$ Galerkin better", fontsize=7.5, color="0.35")
    d.set_title(f"(d) the beam, {len(test)} unseen parameters, $r={r}$", fontsize=9, loc="left")
    NUM["cpdProjR"] = str(r); NUM["cpdProjNTest"] = str(len(test)); NUM["cpdProjEnMin"] = f"{np.min(eKg / eKp):.2f}"; NUM["cpdProjWorstRatio"] = f"{np.max(e2g) / np.max(e2p):.2f}"
    NUM["cpdProjEucRatio"] = f"{np.max(e2g / e2p):.2f}"; NUM["cpdProjEnWins"] = str(int(np.sum(eKg <= eKp * (1 + 1e-9)))); NUM["cpdProjEucWorse"] = str(int(np.sum(e2g >= e2p * (1 - 1e-9))))
    fig.savefig(os.path.join(FIG, "ch08_projection.pdf"), bbox_inches="tight"); fig.savefig("/tmp/ch08_projection.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print({k_: v_ for k_, v_ in NUM.items() if "Proj" in k_})


def fig_kappa(b):
    """Condition number of K(2,1): the Cea bound sqrt(kappa) on the Galerkin/Euclidean-projection error ratio."""
    w = np.linalg.eigvalsh(2.0 * b.K1 + 1.0 * b.K2)
    kap = float(w[-1] / w[0])
    NUM["cpdKappa"] = "$" + A.sci(kap).strip("$") + "$" if not A.sci(kap).startswith("$") else A.sci(kap)
    NUM["cpdSqrtKappa"] = A.sci(float(np.sqrt(kap)))


def write_numbers():
    with open(os.path.join(TAB, "numbers_ch08_fig.tex"), "w") as fh:
        fh.write("% generated by scripts/ch08_figs.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")


if __name__ == "__main__":
    b = A.Beam(120)
    fig_cloud(b)
    fig_error(b)
    fig_beam(b)
    fig_svd(b)
    fig_designs(b)
    fig_intro(b)
    fig_projection(b)
    fig_kappa(b)
    write_numbers()

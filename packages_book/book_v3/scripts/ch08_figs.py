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
    write_numbers()

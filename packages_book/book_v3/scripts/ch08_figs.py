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
    a1.set_ylabel("deflection / peak"); a1.legend(fontsize=7, loc="lower left"); a1.set_xticklabels([])
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


def write_numbers():
    with open(os.path.join(TAB, "numbers_ch08_fig.tex"), "w") as fh:
        fh.write("% generated by scripts/ch08_figs.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")


if __name__ == "__main__":
    b = A.Beam(120)
    fig_cloud(b)
    fig_error(b)
    write_numbers()

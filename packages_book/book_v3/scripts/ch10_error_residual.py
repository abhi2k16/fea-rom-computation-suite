"""E1 (PLAN_ch10_math.md): why the residual ranks the error and why the certified bound is loose.
5-column greedy model of the chapter (real basis, projection V^T). The error e = x - x_r and the residual rho = F - A x_r
are computed in 40-digit arithmetic (mpmath): in double precision the cancellation in rho (||A|| = 6.7e11 against
||rho|| down to 1e-5) is of the size of rho itself. Output: figures/ch10_math_residual.pdf, tables/numbers_ch10_error_residual.tex."""
import os, warnings, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.linalg import eigh
from scipy.stats import spearmanr
from greedy_frequency_training import build_damped_cantilever
from rom_engine import greedy_train_frequency_basis
warnings.simplefilter("ignore"); mp.mp.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = os.path.dirname(HERE)
BLUE, VIOLET, GREY, ORANGE, BLACK = "#08519c", "#54278f", "#636363", "#d95f0e", "#000000"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3})
NUM = {}
num = lambda k, v: NUM.__setitem__(k, v)


def tex(v, p=1):
    m, e = f"{v:.{p}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


s, al, be = build_damped_cantilever(n=20)
fr = np.asarray(s.free_dofs); ix = np.ix_(fr, fr)
K, M, C = s.K[ix], s.M[ix], s.C[ix]
lam, Phi = eigh(K, M); wn = np.sqrt(lam); w1 = wn[0]; zeta = 0.5 * (al / wn + be * wn)
F = np.zeros(s.n_dof); F[s.n_dof - 2] = 1000.0; Ff = F[fr]; n = len(Ff)
band = np.linspace(0.1, 20.0, 400) * w1
g, _ = greedy_train_frequency_basis(band, M, K, Ff, rayleigh=(al, be), n_seed=3, tol=1e-12, max_modes=5)
V = np.real(g.V)
wid = 2 * zeta[:3] * wn[:3] / w1
extra = np.concatenate([wn[k] / w1 + np.linspace(-3, 3, 25) * wid[k] for k in range(3)])
grid = np.unique(np.concatenate([np.linspace(0.1, 20.0, 400), extra])) * w1
Mm, Km, Cm, Fm, Vm = (mp.matrix(a.tolist()) for a in (M, K, C, Ff, V))
rows, modal = [], []
t0 = time.time()
CACHE = os.path.exists('/tmp/e1_R.npy') and os.path.exists('/tmp/e1_Mo.npy') and len(np.load('/tmp/e1_R.npy')) == len(grid)
for w in ([] if CACHE else grid):
    wm = mp.mpf(float(w)); A = -wm * wm * Mm + 1j * wm * Cm + Km
    x = mp.lu_solve(A, Fm); q = mp.lu_solve(Vm.T * A * Vm, Vm.T * Fm); xr = Vm * q
    e = x - xr; rho = Fm - A * xr
    ed = np.array([complex(e[k]) for k in range(n)]); rd = np.array([complex(rho[k]) for k in range(n)])
    Ad = -w * w * M + 1j * w * C + K; U, S, _ = np.linalg.svd(Ad)
    D = wn ** 2 - w * w + 2j * zeta * wn * w; c = Phi.T @ rd; sh = np.abs(c / D) ** 2; shr = np.abs(c) ** 2
    rows.append((w / w1, np.linalg.norm(ed), np.linalg.norm(rd), 1 / S[-1], 1 / S[0], abs(np.vdot(U[:, -1], rd)) / np.linalg.norm(rd),
                 np.linalg.norm(V.T @ rd) / np.linalg.norm(rd)))
    modal.append((sh[:3].sum() / sh.sum(), sh[3:6].sum() / sh.sum(), shr[:3].sum() / shr.sum(), int(np.argmax(sh)) + 1))
print("E1 time %.0fs, %d frequencies" % (time.time() - t0, len(grid)), flush=True)
R = np.load('/tmp/e1_R.npy') if CACHE else np.array(rows); Mo = np.load('/tmp/e1_Mo.npy') if CACHE else np.array(modal)
np.save('/tmp/e1_R.npy', R); np.save('/tmp/e1_Mo.npy', Mo)
keep = (R[:, 0] >= 0.1 - 1e-9) & (R[:, 0] <= 20.0 + 1e-9); R, Mo = R[keep], Mo[keep]
f, e_, r_, imin, imax, cosu, orth = R.T
amp = e_ / r_
fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.8), constrained_layout=True)
a = ax[0]
a.fill_between(f, imax, imin, color=BLUE, alpha=0.15, lw=0, label=r"$1/\sigma_{\max}$ to $1/\sigma_{\min}$")
a.semilogy(f, imin, color=BLUE, lw=0.8, label=r"$1/\sigma_{\min}$")
a.semilogy(f, amp, color=ORANGE, lw=1.4, label=r"$\|\mathbf{e}\|/\|\rho\|$")
a.set_xlabel(r"$\omega/\omega_1$"); a.set_ylim(imax.min() * 0.3, imin.max() * 3); a.legend(fontsize=6, loc="upper right")
a.set_title("(a) amplification", loc="left")
a = ax[1]
a.loglog(r_, e_, ".", color=VIOLET, ms=2.5)
xs = np.array([r_.min(), r_.max()]); a.loglog(xs, np.median(amp) * xs, color=ORANGE, lw=0.8)
a.set_xlabel(r"$\|\rho\|$"); a.set_ylabel(r"$\|\mathbf{e}\|$"); a.set_title("(b) error against residual", loc="left")
a = ax[2]
a.semilogy(f, np.maximum(cosu, 1e-8), color=BLUE, lw=0.9)
a.set_xlabel(r"$\omega/\omega_1$"); a.set_ylabel(r"$|\mathbf{u}_{\min}^H\rho|/\|\rho\|$")
a.set_title(r"(c) residual along $\mathbf{u}_{\min}$", loc="left")
fig.savefig(os.path.join(BOOK, "figures", "ch10_math_residual.pdf"))
rat = amp * imin ** -1
num("cfrEoneNFreq", str(len(R)))
num("cfrEoneAmpLo", tex(amp.min())); num("cfrEoneAmpHi", tex(amp.max())); num("cfrEoneAmpFactor", f"{amp.max() / amp.min():.2f}")
num("cfrEoneInvLo", tex(imin.min())); num("cfrEoneInvHi", tex(imin.max())); num("cfrEoneInvFactor", f"{imin.max() / imin.min():.0f}")
num("cfrEoneSpear", f"{spearmanr(e_, r_)[0]:.4f}")
num("cfrEoneRatioMed", tex(np.median(rat))); num("cfrEoneRatioMin", tex(rat.min())); num("cfrEoneRatioMax", tex(rat.max()))
num("cfrEoneRatioInv", f"{1 / np.median(rat):.0f}")
num("cfrEoneCosMed", tex(np.median(cosu))); num("cfrEoneCosMax", tex(cosu.max()))
num("cfrEoneOrth", tex(orth.max()))
num("cfrEoneShareLow", tex(np.median(Mo[:, 0]))); num("cfrEoneShareMid", f"{100 * np.median(Mo[:, 1]):.1f}")
num("cfrEoneResLow", tex(np.median(Mo[:, 2]))); num("cfrEoneDomMode", str(int(np.bincount(Mo[:, 3].astype(int)).argmax())))
i0 = 0; num("cfrEoneRefAmp", tex(amp[i0])); num("cfrEoneRefSigma", f"{1 / imin[i0]:.0f}"); num("cfrEoneRefRatio", f"{1 / (amp[i0] * (1 / imin[i0])):.0f}")

# ---- the constant e/rho for models of 4 to 8 columns (same arithmetic, 80 frequencies incl. the resonance neighbourhoods)
extra2 = np.concatenate([wn[k] / w1 + np.linspace(-3, 3, 7) * wid[k] for k in range(3)])
grid2 = np.unique(np.concatenate([np.linspace(0.1, 20.0, 60), extra2])); grid2 = grid2[(grid2 >= 0.1) & (grid2 <= 20.0)] * w1
CK = '/tmp/e1_cols.npy'
if os.path.exists(CK) and np.load(CK).shape == (5, 2):
    cols = np.load(CK)
else:
    cols = []
    for k in (4, 5, 6, 7, 8):
        Vk = np.real(greedy_train_frequency_basis(band, M, K, Ff, rayleigh=(al, be), n_seed=3, tol=1e-12, max_modes=k)[0].V); Vkm = mp.matrix(Vk.tolist()); am = []
        for w in grid2:
            wm = mp.mpf(float(w)); A = -wm * wm * Mm + 1j * wm * Cm + Km
            x = mp.lu_solve(A, Fm); q = mp.lu_solve(Vkm.T * A * Vkm, Vkm.T * Fm); xr = Vkm * q
            am.append(float(mp.norm(x - xr) / mp.norm(Fm - A * xr)))
        am = np.array(am); cols.append((np.median(am), am.max() / am.min()))
    cols = np.array(cols); np.save(CK, cols)
# medians are reproducible for 4 to 8 columns; the spread is not for 7 and 8 columns (their last POD directions sit at the noise
# level of the snapshot matrix: 8 columns gave 2.4 and 10.9 in two runs), so it is only reported for 4 to 6 columns
for (m_, f_), nm in zip(cols, ("Four", "Five", "Six", "Seven", "Eight")):
    num(f"cfrEoneConst{nm}", tex(m_))
    if nm in ("Four", "Five", "Six"):
        num(f"cfrEoneSpread{nm}", f"{f_:.2f}")
num("cfrEoneConstFall", f"{cols[0, 0] / cols[-1, 0]:.1f}"); num("cfrEoneAmpPct", f"{100 * (amp.max() / amp.min() - 1):.0f}")
with open(os.path.join(BOOK, "tables", "numbers_ch10_error_residual.tex"), "w") as fh:
    fh.write("% generated by scripts/ch10_error_residual.py -- do not edit\n")
    for k_, v in NUM.items():
        fh.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")
print(NUM)

"""Chapter 10 explanations (PLAN_ch10_math.md): M0 (Galerkin orthogonality of the reduced model) and R1
(random vibration: figures/ch10_math_psd.pdf and an independent modal closed form). Macros in tables/numbers_ch10_math.tex."""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from greedy_frequency_training import build_damped_cantilever
from rom_engine import FrequencyROM, build_pod_basis_from_frf_snapshots, greedy_train_frequency_basis
from rom_engine.random_vibration import psd_response
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = os.path.dirname(HERE)
BLUE, VIOLET, GREY, ORANGE = "#08519c", "#54278f", "#636363", "#d95f0e"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.grid": True, "grid.alpha": 0.3})
NUM = {}
num = lambda k, v: NUM.__setitem__(k, v)


def tex(v, p=1):
    m, e = f"{v:.{p}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


s, al, be = build_damped_cantilever(n=20)
fr = np.asarray(s.free_dofs); ix = np.ix_(fr, fr)
K, M, C = s.K[ix], s.M[ix], s.C[ix]
lam, Phi = eigh(K, M); wn = np.sqrt(lam); w1 = wn[0]; zeta = 0.5 * (al / wn + be * wn)
F = np.zeros(s.n_dof); F[s.n_dof - 2] = 1000.0; Ff = F[fr]; tip = len(fr) - 2
band = np.linspace(0.1, 20.0, 400) * w1

# ---------------------------------------------------------------- M0: orthogonality of the residual
gb, _ = greedy_train_frequency_basis(band, M, K, Ff, rayleigh=(al, be), n_seed=3, tol=1e-12, max_modes=5)
ub = build_pod_basis_from_frf_snapshots(np.linspace(band[0], band[-1], 5), M, K, Ff, C=C, n_modes=5)
out = {}
for nm, b in (("greedy", gb), ("uniform", ub)):
    V = b.V; rom = FrequencyROM.from_MCK(M, K, V, rayleigh=(al, be)); mT = mH = 0.0
    for r in np.linspace(0.1, 20.0, 100):
        w = r * w1; A = -w * w * M + 1j * w * C + K
        rho = Ff - A @ rom.frequency_response([w], Ff)[0]
        mT = max(mT, np.linalg.norm(V.T @ rho) / np.linalg.norm(Ff)); mH = max(mH, np.linalg.norm(V.conj().T @ rho) / np.linalg.norm(Ff))
    out[nm] = (mT, mH, np.linalg.norm(V.imag) / np.linalg.norm(V))
num("cfrMathOrthT", tex(max(out["greedy"][0], out["uniform"][0])))
num("cfrMathOrthHUni", tex(out["uniform"][1]))
num("cfrMathImagGreedy", f"{out['greedy'][2]:.0f}")
num("cfrMathImagUni", f"{100 * out['uniform'][2]:.0f}")


# ---------------------------------------------------------------- Q15: snapshots damped or undamped, V^T or V^H
rng = np.random.default_rng(0)
test = np.concatenate([wn[i] * (1.0 + 0.05 * rng.standard_normal(20)) for i in range(3)])
truth = {w: s.solve_harmonic(w, F)[fr] for w in test}


def near_errors(V, herm):
    P = V.conj().T if herm else V.T; e = []
    for w in test:
        A = -w * w * M + 1j * w * C + K
        x = V @ np.linalg.solve(P @ A @ V, P @ Ff)
        e.append(np.linalg.norm(x - truth[w]) / np.linalg.norm(truth[w]))
    return np.array(e)


eT, eH = near_errors(ub.V, False), near_errors(ub.V, True)
num("cfrMathProjDiff", tex(np.max(np.abs(eT - eH) / eT)))
gd, hd = greedy_train_frequency_basis(band, M, K, Ff, C=C, n_seed=3, tol=1e-12, max_modes=5)
ed = near_errors(gd.V, False)
num("cfrMathGreedyDampMed", tex(np.median(ed))); num("cfrMathGreedyDampMax", tex(np.max(ed)))
num("cfrMathGreedyDampSel", ", ".join(f"{o / w1:.2f}" for o, _, _ in hd))

# ---------------------------------------------------------------- R1: random vibration
f = np.linspace(0.1 * w1 / (2 * np.pi), 20 * w1 / (2 * np.pi), 6000); psd = np.ones_like(f)
full = FrequencyROM.from_MCK(M, K, np.eye(len(fr)), rayleigh=(al, be))
S_out, rms = psd_response(full, f, psd, Ff, tip)
H2 = S_out / psd
Gam = Phi[tip, :] * (Phi.T @ Ff)
var = Gam ** 2 / (8 * zeta * wn ** 3)                   # S_0 Gamma_l^2 / (8 zeta_l w_l^3), S_0 = 1 per Hz
rms_cf = np.sqrt(var.sum())
cum = np.concatenate([[0], np.cumsum(0.5 * (S_out[1:] + S_out[:-1]) * np.diff(f))]); cum /= cum[-1]
num("cfrMathRmsNum", tex(rms, 4)); num("cfrMathRmsCF", tex(rms_cf, 4))
num("cfrMathRmsDiff", tex(abs(rms_cf - rms) / rms))
num("cfrMathShareOne", f"{100 * var[0] / var.sum():.2f}"); num("cfrMathShareRest", f"{100 * (1 - var[0] / var.sum()):.2f}")
num("cfrMathRmsOnly", tex(np.sqrt(var[0]), 4))
num("cfrMathEffPSD", tex(1000.0 ** 2, 0))
num("cfrMathBandLoHz", f"{f[0]:.1f}"); num("cfrMathBandHiHz", f"{f[-1]:.0f}")
fn = wn[:3] / (2 * np.pi)
fig, ax = plt.subplots(2, 2, figsize=(6.2, 4.4), constrained_layout=True)
a = ax[0, 0]; a.semilogx(f, psd, color=GREY); a.set_ylim(0, 1.5)
a.set_ylabel(r"$S_{\rm in}$ (Hz$^{-1}$)"); a.set_title("(a) input: flat spectrum", loc="left")
a = ax[0, 1]; a.loglog(f, H2, color=BLUE, lw=1.0)
a.set_ylabel(r"$|H|^2$ (m$^2$)"); a.set_title("(b) transfer function, load to tip", loc="left")
a = ax[1, 0]; a.loglog(f, S_out, color=VIOLET, lw=1.0)
a.set_ylabel(r"$S_{\rm out}=|H|^2S_{\rm in}$ (m$^2$/Hz)"); a.set_title("(c) output spectrum", loc="left")
a = ax[1, 1]; a.semilogx(f, cum, color=ORANGE)
for k in range(3):
    a.axvline(fn[k], color=GREY, lw=0.7, ls=":")
a.set_ylabel(r"share of $\sigma_{\rm rms}^2$ below $f$"); a.set_ylim(0, 1.05)
a.set_title(r"(d) where the variance $\sigma_{\rm rms}^2=\int S_{\rm out}\,df$ comes from", loc="left")
for a_ in ax.ravel():
    a_.set_xlabel("$f$ (Hz)")
for a_ in (ax[0, 1], ax[1, 0]):
    for k in range(3):
        a_.axvline(fn[k], color=GREY, lw=0.7, ls=":")
fig.savefig(os.path.join(BOOK, "figures", "ch10_math_psd.pdf"))
num("cfrMathShareAtSecond", f"{100 * np.interp(fn[1], f, cum):.2f}")
with open(os.path.join(BOOK, "tables", "numbers_ch10_math.tex"), "w") as fh:
    fh.write("% generated by scripts/ch10_math.py -- do not edit\n")
    for k_, v in NUM.items():
        fh.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")
print(NUM)

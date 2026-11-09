"""G1 (PLAN_ch10_math.md): the greedy loop round by round. figures/ch10_math_greedy.pdf and
tables/numbers_ch10_greedy_rounds.tex. The models and the picks come from greedy_train_frequency_basis itself
(history of a long run, bases of runs with max_modes = 3, 4, 5, 6); nothing of the loop is re-implemented except
the evaluation of the indicators on the candidate pool, with the package's own methods."""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from greedy_frequency_training import build_damped_cantilever
from rom_engine import FrequencyROM, greedy_train_frequency_basis
warnings.simplefilter("ignore")
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
K, M = s.K[ix], s.M[ix]
wn = np.sqrt(eigh(K, M, eigvals_only=True)); w1 = wn[0]
F = np.zeros(s.n_dof); F[s.n_dof - 2] = 1000.0; Ff = F[fr]
band = np.linspace(0.1, 20.0, 400) * w1
truth = np.array([s.solve_harmonic(w, F)[fr] for w in band])
tn = np.linalg.norm(truth, axis=1)


def run(k):
    return greedy_train_frequency_basis(band, M, K, Ff, rayleigh=(al, be), n_seed=3, tol=1e-12, max_modes=k)


_, hist = run(7)                                  # picks 1..4 (and a fifth) of the loop
seed_idx = np.linspace(0, len(band) - 1, 3).astype(int)
picks = [h[0] for h in hist]
roms = {}
for k in (3, 4, 5, 6, 7):
    b, _ = run(k); roms[k] = FrequencyROM.from_MCK(M, K, b.V, rayleigh=(al, be))
    hk = run(k)[1]
    assert [h[0] for h in hk] == picks[:len(hk)], "history of shorter run differs"
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.0), constrained_layout=True)
sel = list(band[seed_idx])
idx_of = lambda w: int(np.argmin(np.abs(band - w)))
rows = []
for rnd in range(4):
    k = 3 + rnd; cur = roms[k]; prev = roms[k - 1] if rnd > 0 else None
    mask = np.ones(len(band), bool); mask[[idx_of(w) for w in sel]] = False
    cand = np.where(mask)[0]
    if prev is None:
        ind = np.array([cur.residual_norm(band[i], Ff) / np.linalg.norm(Ff) for i in cand]); nm = "residual"
    else:
        ind = np.array([cur.hierarchical_error_indicator(band[i], Ff, comparison_rom=prev)
                        / np.linalg.norm(cur.frequency_response([band[i]], Ff)[0]) for i in cand]); nm = "hierarchical"
    err_all = np.array([np.linalg.norm(cur.frequency_response([band[i]], Ff)[0] - truth[i]) / tn[i] for i in range(len(band))])
    pick = picks[rnd]; ipk = idx_of(pick)
    assert cand[int(np.argmax(ind))] == ipk, "pick is not the argmax of the indicator"
    gamma = err_all[ipk] / err_all[cand].max()
    e_res = [err_all[idx_of(wn[j])] for j in range(3)]
    rows.append((pick / w1, gamma, e_res, err_all[cand].max(), band[cand[int(np.argmax(err_all[cand]))]] / w1))
    a = ax.ravel()[rnd]
    a.semilogy(band / w1, err_all, color=BLACK, lw=0.9)
    scale = err_all[cand].max() / ind.max()
    a.semilogy(band[cand] / w1, ind * scale, color=BLUE, lw=0.9, ls="--")
    for w in sel:
        a.axvline(w / w1, color=GREY, lw=0.8, ls=":")
    a.plot([pick / w1], [ind.max() * scale], "v", color=ORANGE, ms=6, zorder=5)
    for j in range(3):
        a.axvline(wn[j] / w1, color=ORANGE, lw=0.5, alpha=0.35)
    a.set_title(f"({'abcd'[rnd]}) {k} columns, pick {rnd + 1} by the {nm}", loc="left")
    a.set_xlabel(r"$\omega/\omega_1$"); a.set_ylabel("relative error")
    a.set_ylim(max(err_all.min() * 0.5, 1e-14), err_all.max() * 8)
    sel.append(pick)
from matplotlib.lines import Line2D
hd = [Line2D([], [], color=BLACK, lw=0.9, label="true error of the model"),
      Line2D([], [], color=BLUE, lw=0.9, ls="--", label="indicator, scaled to the peak of the error"),
      Line2D([], [], color=GREY, lw=0.8, ls=":", label="snapshots so far"),
      Line2D([], [], color=ORANGE, lw=0.8, alpha=0.5, label="natural frequencies"),
      Line2D([], [], color=ORANGE, marker="v", ls="", ms=6, label="next snapshot")]
fig.legend(handles=hd, loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
fig.savefig(os.path.join(BOOK, "figures", "ch10_math_greedy.pdf"))
for r, (p, g, er, mx, at) in enumerate(rows, 1):
    num(f"cfrGrPick{'ABCD'[r - 1]}", f"{p:.2f}")
    num(f"cfrGrGamma{'ABCD'[r - 1]}", f"{g:.2f}")
    for j in range(3):
        num(f"cfrGrErr{'ABCD'[r - 1]}R{j + 1}", tex(er[j]))
    num(f"cfrGrMax{'ABCD'[r - 1]}", tex(mx)); num(f"cfrGrMaxAt{'ABCD'[r - 1]}", f"{at:.1f}")
num("cfrGrThirdRes", f"{wn[2] / w1:.1f}"); num("cfrGrSecondRes", f"{wn[1] / w1:.2f}")
with open(os.path.join(BOOK, "tables", "numbers_ch10_greedy_rounds.tex"), "w") as fh:
    fh.write("% generated by scripts/ch10_greedy_rounds.py -- do not edit\n")
    for k_, v in NUM.items():
        fh.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")
for r in rows: print("ROW", [round(float(x), 4) if np.isscalar(x) else [float('%.2e' % y) for y in x] for x in r])

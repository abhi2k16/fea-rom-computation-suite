"""Introduction figure of Chapter 11 (Sec. 11.1): the beam cut at midspan, the frequency ladder, Guyan and Craig-Bampton at equal size.
Run: PYTHONPATH=fea_engine/src:rom_engine/src python packages_book/book_v3/scripts/ch11_intro.py
Writes figures/ch11_intro.pdf and tables/numbers_ch11_intro.tex (macro prefix ccm, no digits in names)."""
import os, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures"); TAB = os.path.join(HERE, "..", "tables")
spec = importlib.util.spec_from_file_location("a", os.path.join(HERE, "ch11_assets.py")); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
from rom_engine.cms import couple, craig_bampton, guyan
plt.rcParams.update({"font.size": 8, "axes.grid": False})
NEL = a.NEL; NUM = {}
def sci(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
hz = lambda w2: np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi)
fr, K, M = a.beam(1.0, 2 * NEL, [0, 2 * NEL]); f_full = hz(eigh(K, M, eigvals_only=True)[:8])
frL, KL, ML = a.beam(0.5, NEL, [0]); frR, KR, MR = a.beam(0.5, NEL, [NEL])
pos = lambda f_, d: int(np.where(f_ == d)[0][0])
ibL = [pos(frL, 2 * NEL), pos(frL, 2 * NEL + 1)]; ibR = [pos(frR, 0), pos(frR, 1)]
nint = KL.shape[0] - len(ibL)
cbL = craig_bampton(KL, ML, ibL, 5); cbR = craig_bampton(KR, MR, ibR, 5)
cp = couple([cbL, cbR], [[0, 1], [0, 1]])
f_cb, _ = cp.solve_modal(8); f_cb = np.asarray(f_cb)
vd = np.arange(0, K.shape[0], 2)
idx = vd[np.round(np.linspace(0, len(vd) - 1, 14)[1:-1]).astype(int)]
g = guyan(K, M, idx); f_g = hz(eigh(g.K, g.M, eigvals_only=True)[:8])
# fixed-interface frequencies of one half
keep = np.setdiff1d(np.arange(KL.shape[0]), ibL)
f_fix = hz(eigh(KL[np.ix_(keep, keep)], ML[np.ix_(keep, keep)], eigvals_only=True)[:4])
e_cb = np.abs(f_cb - f_full) / f_full; e_g = np.abs(f_g - f_full) / f_full
ratio = f_full / f_fix[0]
words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
NUM["ccmItElem"] = str(2 * NEL); NUM["ccmItFull"] = str(K.shape[0]); NUM["ccmItIface"] = str(len(ibL)); NUM["ccmItInt"] = str(nint)
NUM["ccmItFreqOne"] = f"{f_full[0]:.0f}"; NUM["ccmItFreqThree"] = f"{f_full[2]:.0f}"
NUM["ccmItFixOne"] = f"{f_fix[0]:.0f}"
NUM["ccmItRatioOne"] = f"{ratio[0]:.2f}"; NUM["ccmItRatioTwo"] = f"{ratio[1]:.2f}"; NUM["ccmItRatioThree"] = f"{ratio[2]:.2f}"
NUM["ccmItMasters"] = str(len(idx)); NUM["ccmItUnk"] = str(cp.K.shape[0]); NUM["ccmItModes"] = "5"
for k in (0, 2, 7): NUM[f"ccmItGuy{words[k]}"] = sci(e_g[k])
NUM["ccmItCbLo"] = sci(e_cb.min()); NUM["ccmItCbHi"] = sci(e_cb.max())
NUM["ccmItCbOne"] = sci(e_cb[0])
assert cp.K.shape[0] == len(idx), "equal size expected"
# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(6.2, 2.7), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
ax = [fig.add_subplot(gs[0, j]) for j in range(3)]
# (a) the cut beam
A = ax[0]; A.set_axis_off(); A.set_xlim(0, 10); A.set_ylim(-4.5, 4.5)
A.add_patch(plt.Rectangle((0.2, 0.6), 4.4, 1.0, fc="#cfe0f5", ec="C0", lw=1)); A.add_patch(plt.Rectangle((5.4, 0.6), 4.4, 1.0, fc="#fbe0c6", ec="C1", lw=1))
for x in (0.2, 9.8): A.plot([x, x], [0.2, 2.0], "k", lw=2.0)
A.plot([5.0], [1.1], "o", color="C3", ms=6); A.annotate("", xy=(4.65, 1.1), xytext=(5.35, 1.1), arrowprops=dict(arrowstyle="<->", color="C3", lw=0.8))
A.text(2.4, 1.1, "half A", ha="center", va="center", fontsize=7); A.text(7.6, 1.1, "half B", ha="center", va="center", fontsize=7)
A.text(2.4, -0.2, f"{nint} internal\nunknowns", ha="center", va="top", fontsize=6.5); A.text(7.6, -0.2, f"{nint} internal\nunknowns", ha="center", va="top", fontsize=6.5)
A.text(5.0, 3.1, f"{len(ibL)} interface unknowns\n(deflection, rotation)", ha="center", va="center", fontsize=6.5, color="C3"); A.annotate("", xy=(5.0, 1.5), xytext=(5.0, 2.55), arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
A.set_title("(a) the beam cut at midspan", loc="left", fontsize=7)
# (b) ladder
B = ax[1]
for k in range(8): B.hlines(f_full[k], 0.1, 0.9, color="C0", lw=1.4)
for k in range(4): B.hlines(f_fix[k], 1.1, 1.9, color="C1", lw=1.4)
B.axhline(f_fix[0], color="0.5", ls="--", lw=0.8)
B.set_yscale("log"); B.set_xlim(0, 2); B.set_xticks([0.5, 1.5]); B.set_xticklabels(["coupled\nbeam", "half, interface\nheld"], fontsize=6.5)
B.set_ylabel("frequency (Hz)"); B.set_title("(b) frequency ladder", loc="left", fontsize=7); B.grid(True, axis="y", alpha=0.3)
# (c) errors
C = ax[2]; ks = np.arange(1, 9)
C.semilogy(ks, e_g, "s--", color="C1", ms=3.5, label=f"Guyan, {len(idx)} masters"); C.semilogy(ks, e_cb, "o-", color="C0", ms=3.5, label=f"Craig--Bampton, {cp.K.shape[0]} unknowns")
C.set_xlabel("mode number"); C.set_ylabel("relative frequency error"); C.set_title("(c) equal size", loc="left", fontsize=7); C.legend(fontsize=5.5, loc="upper left"); C.grid(True, alpha=0.3)
fig.savefig(os.path.join(FIG, "ch11_intro.pdf")); fig.savefig("/tmp/ch11_intro.png", dpi=140)
with open(os.path.join(TAB, "numbers_ch11_intro.tex"), "w") as fh:
    fh.write("% generated by scripts/ch11_intro.py -- do not edit\n")
    for k, v in NUM.items(): fh.write("\\newcommand{\\%s}{%s\\xspace}\n" % (k, v))
print(NUM); print(f_full, f_fix, e_g, e_cb, ratio, sep="\n")

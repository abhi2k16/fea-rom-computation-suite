"""Numbers quoted in the prose of Chapter 11, Sec. 11.5 (audit item U6), and the output of Listing 11.3.
Run from the repository root: PYTHONPATH=fea_engine/src:rom_engine/src python packages_book/book_v3/scripts/ch11_claims.py
Writes tables/ch11_try_output.txt (the printed output of Listing 11.3, run exactly as printed in the chapter) and
tables/numbers_ch11_claims.tex (macro prefix ccm, no digits in names)."""
import os, re, importlib.util, io, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TAB = os.path.join(HERE, "..", "tables"); CH = os.path.join(HERE, "..", "chapters")
tex = open(os.path.join(CH, "ch11.tex")).read()
NUM = {}
def sci(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
# ---- Listing 11.3, executed as printed
m = re.search(r"label=\{lst:cm-try\}\]\n(.*?)\\end\{lstlisting\}", tex, re.S)
buf = io.StringIO(); ns = {"__name__": "listing"}
with contextlib.redirect_stdout(buf):
    exec(compile(m.group(1), "listing", "exec"), ns)
open(os.path.join(TAB, "ch11_try_output.txt"), "w").write(buf.getvalue())
NUM["ccmClRatio"] = f"{ns['err_g'] / ns['err_c']:.0f}"
# ---- Rayleigh-Ritz properties of the Craig-Bampton beam: each of the first four frequencies is an upper bound and non-increasing in n_m
spec0 = importlib.util.spec_from_file_location("a0", os.path.join(HERE, "ch11_assets.py")); a0 = importlib.util.module_from_spec(spec0); spec0.loader.exec_module(a0)
from rom_engine.cms import couple as _couple, craig_bampton as _cb
fr0, K0, M0 = a0.beam(1.0, 2 * a0.NEL, [0, 2 * a0.NEL]); f0 = a0.freqs_hz(K0, M0, 4)
frL0, KL0, ML0 = a0.beam(0.5, a0.NEL, [0]); frR0, KR0, MR0 = a0.beam(0.5, a0.NEL, [a0.NEL])
pos0 = lambda f_, d: int(np.where(f_ == d)[0][0])
ibL0 = [pos0(frL0, 2 * a0.NEL), pos0(frL0, 2 * a0.NEL + 1)]; ibR0 = [pos0(frR0, 0), pos0(frR0, 1)]
prev, ub, mono = None, True, True
for nm in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30):
    cp0 = _couple([_cb(KL0, ML0, ibL0, nm), _cb(KR0, MR0, ibR0, nm)], [[0, 1], [0, 1]])
    e0 = (np.asarray(cp0.solve_modal(4)[0]) - f0) / f0
    ub &= bool(np.all(e0 >= -1e-9)); mono &= prev is None or bool(np.all(e0 <= prev * (1 + 1e-9) + 1e-14)); prev = e0
NUM["ccmClUpperBound"] = "yes" if ub else "no"; NUM["ccmClMonoEach"] = "yes" if mono else "no"
# ---- constants of the experiment script
src = open(os.path.join(HERE, "ch11_assets.py")).read()
NUM["ccmClMastersMax"] = str(max(int(x) for x in re.search(r"for m in \(([\d, ]+)\):", src).group(1).split(",")))
NUM["ccmClModes"] = re.search(r"^NMODES = (\d+)", src, re.M).group(1)
# ---- interface reduction: smallest k that equals the unreduced interface
spec = importlib.util.spec_from_file_location("a", os.path.join(HERE, "ch11_assets.py")); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
from rom_engine.cms import craig_bampton, reduce_interface
from rom_engine.interface_reduction import natural_frequency_errors
K, M, iface = a.membrane(16, 21); cb = craig_bampton(K, M, iface, 40); nmod = int(NUM["ccmClModes"])
plain = np.abs(natural_frequency_errors(K, M, cb.T, nmod)).max()
kmin = next(k for k in range(1, len(iface) + 1) if abs(np.abs(natural_frequency_errors(K, M, reduce_interface(cb, k).T, nmod)).max() - plain) <= 1e-6 * plain)
NUM["ccmClCcMinK"] = str(kmin); NUM["ccmClCcRedundant"] = str(len(iface) - kmin)
# ---- Table 11.3: errors of the frequencies above the retained ones (Craig-Bampton and Rubin rows)
vals = []
for line in open(os.path.join(TAB, "ch11_free.tex")):
    cells = [c.strip() for c in line.rstrip("\\\n ").split("&")]
    if len(cells) == 6 and ("Craig" in cells[1] or "Rubin" in cells[1]):
        mm = re.search(r"\$(\d+\.?\d*)\\times10\^\{(-?\d+)\}\$", cells[4])
        if mm: vals.append(float(mm.group(1)) * 10 ** int(mm.group(2)))
NUM["ccmClTopLo"] = f"{min(vals):.2f}"; NUM["ccmClTopHi"] = f"{max(vals):.2f}"
# ---- static error at the loaded mass for a load at an internal mass of the chain (Sec. 11.5.3, audit item U13)
from rom_engine.cms import free_interface_cms
from scipy.linalg import eigh
Kc, Mc = a.chain(); nc = Kc.shape[0]; ifc = [39, 20]; node = 10
Fc = np.zeros(nc); Fc[node] = 1.0; uc = np.linalg.solve(Kc, Fc)
def tip(u_r): return abs(u_r[node] - uc[node]) / abs(uc[node])
def models(nm):
    out = {}
    cbm = craig_bampton(Kc, Mc, ifc, nm); out["cb"] = tip(cbm.T @ np.linalg.solve(cbm.K, cbm.T.T @ Fc))
    for meth in ("rubin", "macneal"):
        fm = free_interface_cms(Kc, Mc, ifc, nm, meth); out[meth] = tip(fm.T @ np.linalg.solve(fm.K, fm.T.T @ Fc))
    _, V = eigh(Kc, Mc); Vm = V[:, :nm]; out["modal"] = tip(Vm @ np.linalg.solve(Vm.T @ Kc @ Vm, Vm.T @ Fc))
    return out
m3, m10 = models(3), models(10)
assert abs(m3["macneal"] - m3["modal"]) <= 1e-9 * m3["modal"]
NUM["ccmClIntNode"] = str(node)
NUM["ccmClIntCbThree"] = sci(m3["cb"]); NUM["ccmClIntRubinThree"] = sci(m3["rubin"]); NUM["ccmClIntModalThree"] = sci(m3["modal"])
NUM["ccmClIntLoTen"] = sci(min(m10.values())); NUM["ccmClIntHiTen"] = sci(max(m10.values()))
with open(os.path.join(TAB, "numbers_ch11_claims.tex"), "w") as fh:
    fh.write("% generated by scripts/ch11_claims.py -- do not edit\n")
    for k, v in NUM.items(): fh.write("\\newcommand{\\%s}{%s\\xspace}\n" % (k, v))
print(NUM); print(buf.getvalue())

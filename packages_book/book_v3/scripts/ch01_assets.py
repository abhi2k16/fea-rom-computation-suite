# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch01_assets.py -- figures, tables and number macros of Chapter 1 (architectural design of the packages).

Writes into book/:
    figures/ch06_dispatch.pdf     solve time with backend="scipy" and backend="auto" against the problem size
    tables/ch01_registry.tex      interface conformance of every entry of ELEMENT_REGISTRY, by family
    tables/ch01_packages.tex      size of the two packages (modules, lines) and the import rule
    tables/numbers_ch01.tex       every number quoted in the prose

It also runs the extension example of the chapter (a new bar element added by a subclass and one registry line) and
records its result. Run from the repository root:   python packages_book/book/scripts/ch01_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import ast
import os
import platform
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(BOOK, "..", ".."))
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
NUM = {}


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


import fea_engine                                                                    # noqa: E402
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, elements as elmod   # noqa: E402
from fea_engine.elements import ELEMENT_REGISTRY, Element                            # noqa: E402
from fea_engine.mesh import rectangle_mesh, line_mesh                                # noqa: E402


# ------------------------------------------------------------------------------------------ registry conformance
FAMILY = [("scalar", ("laplace",)), ("beam, truss", ("beam", "truss")), ("plate, shell", ("plate", "shell")),
          ("contact", ("contact",)), ("plane stress / strain", ("quad4_plane", "tri3_plane", "quad8_plane", "tri6_plane", "plastic_j2_plane")),
          ("solid", ("hex", "tet"))]


def family_of(name):
    for fam, keys in FAMILY:
        if any(k in name for k in keys):
            return fam
    return "other"


def check_element(cls):
    """Interface test that needs no material: class attributes and the shape functions at the natural origin."""
    issues = []
    if not (isinstance(cls, type) and issubclass(cls, Element)):
        return ["not an Element subclass"]
    for a in ("n_nodes", "dofs_per_node", "dim", "gauss_order"):
        v = getattr(cls, a, None)
        if not isinstance(v, (int, np.integer)) or v <= 0:
            issues.append(f"attribute {a} missing or not a positive integer")
    if issues:
        return issues
    if cls.__name__.startswith(("Gap", "NodeTo")):               # contact elements: no interpolation, force/tangent interface
        miss = [m for m in ("internal_force", "tangent_stiffness") if not callable(getattr(cls, m, None))]
        return [f"missing {m}" for m in miss]
    try:
        el = cls()
        N, dN = el.shape_and_derivs(tuple(np.zeros(cls.dim)))
        if abs(float(np.sum(N)) - 1.0) > 1e-12 and not (cls.__name__.startswith(("Beam", "Shell", "Gap", "NodeTo"))):
            issues.append("shape functions do not sum to 1")
        if np.shape(dN) != (cls.dim, cls.n_nodes) and not (cls.__name__.startswith(("Gap", "NodeTo"))):
            issues.append(f"derivative array shape {np.shape(dN)}")
    except Exception as e:                                           # reported, not hidden
        issues.append(f"{type(e).__name__}: {str(e)[:60]}")
    return issues


def exp_registry():
    rows, bad = {}, []
    for name, cls in ELEMENT_REGISTRY.items():
        iss = check_element(cls)
        fam = family_of(name)
        r = rows.setdefault(fam, [0, 0])
        r[0] += 1
        r[1] += 0 if iss else 1
        if iss:
            bad.append((name, iss))
    print("registry:", {k: tuple(v) for k, v in rows.items()}, flush=True)
    for b in bad:
        print("   non-conforming:", b, flush=True)
    with open(os.path.join(TAB, "ch01_registry.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrr}\n\\toprule\nfamily & registered & pass the interface test \\\\\n\\midrule\n")
        for fam, (n, p) in rows.items():
            fh.write(f"{fam} & {n} & {p} \\\\\n")
        tot = sum(v[0] for v in rows.values()); tp = sum(v[1] for v in rows.values())
        fh.write(f"\\midrule\ntotal & {tot} & {tp} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    num("cdeRegTotal", str(sum(v[0] for v in rows.values()))); num("cdeRegPass", str(sum(v[1] for v in rows.values())))
    num("cdeRegBad", ", ".join(f"\\code{{{n}}}" for n, _ in bad) if bad else "none")


# ------------------------------------------------------------------------------------------ package sizes and import rule
def count(pkg_dir):
    files = [os.path.join(d, f) for d, _, fs in os.walk(pkg_dir) for f in fs if f.endswith(".py") and "__pycache__" not in d]
    lines = sum(sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) for f in files)
    return len(files), lines, files


def imports(files, name):
    hits = []
    for f in files:
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import) and any(a.name.split(".")[0] == name for a in n.names):
                hits.append(f)
            if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == name:
                hits.append(f)
    return hits


def exp_packages():
    nf, lf, ff = count(os.path.join(ROOT, "fea_engine", "src", "fea_engine"))
    nr, lr, fr = count(os.path.join(ROOT, "rom_engine", "src", "rom_engine"))
    tr_f, tr_r = count(os.path.join(ROOT, "fea_engine", "tests")), count(os.path.join(ROOT, "rom_engine", "tests"))
    rom_in_fea = imports(ff, "rom_engine")
    fea_in_rom = imports(fr, "fea_engine")
    fea_in_rom_tests = imports(tr_r[2], "fea_engine")
    print(f"packages: fea {nf} modules {lf} lines; rom {nr} modules {lr} lines; fea imports rom in {len(rom_in_fea)} files; rom imports fea in {len(fea_in_rom)} files; rom tests import fea in {len(fea_in_rom_tests)}", flush=True)
    with open(os.path.join(TAB, "ch01_packages.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\n & source modules & source lines & test files & test lines \\\\\n\\midrule\n")
        fh.write(f"\\code{{fea\\_engine}} & {nf} & {lf:,} & {tr_f[0]} & {tr_f[1]:,} \\\\\n".replace(",", "\\,"))
        fh.write(f"\\code{{rom\\_engine}} & {nr} & {lr:,} & {tr_r[0]} & {tr_r[1]:,} \\\\\n".replace(",", "\\,"))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    num("cdeFeaImportsRom", str(len(rom_in_fea))); num("cdeRomImportsFea", str(len(fea_in_rom)))
    num("cdeRomTestsImportFea", str(len(fea_in_rom_tests)))
    num("cdeRomTestFiles", str(tr_r[0]))


# ------------------------------------------------------------------------------------------ backend dispatch overhead
def plate(n):
    m = rectangle_mesh(Lx=1.0, Ly=1.0, nx=n, ny=n)
    return m


def solve_time(n, backend, reps=3):
    m = plate(n)
    D = D_plane_stress(Material(E=2.1e11, nu=0.3, rho=7850.0))
    best = 1e9
    for _ in range(reps):
        s = FESystem(m, Quad4PlaneStress(), sparse=True, backend=backend)
        s.assemble_stiffness(D)
        s.fix_dofs(m.nodes_on_line(axis=0, value=0.0), [0, 1])
        tip = m.nodes_on_line(axis=0, value=1.0)
        s.add_nodal_force(tip, dof_index=1, total_force=-1.0e4)
        t0 = time.perf_counter()
        s.solve_static()
        best = min(best, time.perf_counter() - t0)
    return best, s.n_dof


def exp_dispatch():
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:
        has_torch = False
    sizes = [8, 16, 24, 32, 40]                # up to about 3,400 dofs: "auto" stays on the scipy path (threshold 5,000 free dofs)
    ts, ta, nd = [], [], []
    for n in sizes:
        a, d = solve_time(n, "scipy"); b, _ = solve_time(n, "auto")
        ts.append(a); ta.append(b); nd.append(d)
        print(f"dispatch n={n} ({d} dofs): scipy {a*1e3:.2f} ms auto {b*1e3:.2f} ms", flush=True)
    ts, ta, nd = map(np.array, (ts, ta, nd))
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    ax[0].loglog(nd, ts * 1e3, "o-", label="backend='scipy'"); ax[0].loglog(nd, ta * 1e3, "s--", label="backend='auto'")
    ax[0].set_xlabel("dofs"); ax[0].set_ylabel("solve_static time (ms)"); ax[0].legend()
    ax[1].semilogx(nd, (ta - ts) * 1e3, "o-"); ax[1].set_xlabel("dofs"); ax[1].set_ylabel("extra time of 'auto' (ms)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ch06_dispatch.pdf")); plt.close(fig)
    sl = np.polyfit(np.log(nd[1:]), np.log(ta[1:] - ts[1:]), 1)[0]
    num("cdeDispSlope", f"{sl:.1f}")
    num("cdeDenseFiveK", f"{8 * 5000 ** 2 / 1e9:.1f}"); num("cdeDenseFiftyK", f"{8 * 50000 ** 2 / 1e9:.1f}")
    num("cdeDispMaxDof", f"{int(nd.max()):,}".replace(",", "\\,")); num("cdeDispThreshold", "5\\,000")
    num("cdeDispExtraMs", f"{(ta - ts).max() * 1e3:.1f}"); num("cdeDispExtraFrac", f"{((ta - ts) / ts).max() * 100:.0f}")
    num("cdeDispExtraFracSmall", f"{((ta - ts) / ts)[0] * 100:.0f}")
    num("cdeDispExtraFracLarge", f"{((ta - ts) / ts)[-1] * 100:.0f}")
    num("cdeHasTorch", "installed" if has_torch else "not installed")


# ------------------------------------------------------------------------------------------ extension example
class Bar1D(Element):                                       # the new element: nothing existing is edited
    n_nodes, dofs_per_node, dim, gauss_order = 2, 1, 1, 1
    dof_names = ("u",)

    def shape_and_derivs(self, natural_coords):
        (xi,) = natural_coords
        return np.array([(1 - xi) / 2, (1 + xi) / 2]), np.array([[-0.5, 0.5]])

    def B_matrix(self, natural_coords, elem_coords):
        from fea_engine.elements import jacobian
        _, dN = self.shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN, elem_coords)
        return np.linalg.solve(J, dN), detJ


def exp_extension():
    ELEMENT_REGISTRY["bar1d"] = Bar1D                         # the one registry line
    E, A, rho, L, P = 210e9, 1.0e-4, 7800.0, 2.0, 1000.0
    errs, freqs = [], []
    for n in (1, 2, 4, 8, 16, 32):
        m = line_mesh(L, n)
        s = FESystem(m, ELEMENT_REGISTRY["bar1d"]())
        s.assemble_stiffness(np.array([[E * A]]))
        s.assemble_mass(np.array([[rho * A]]))
        s.fix_dofs([0], [0])
        s.add_nodal_force([n], 0, P)
        U = s.solve_static()
        errs.append(abs(U[n] / (P * L / (E * A)) - 1))
        f, _ = s.solve_modal(n_modes=1)
        freqs.append(f[0])
    f_exact = np.sqrt(E / rho) / (4 * L)
    ferr = [abs(f / f_exact - 1) for f in freqs]
    h = 1 / np.array([1, 2, 4, 8, 16, 32])
    order = np.polyfit(np.log(h[1:]), np.log(ferr[1:]), 1)[0]
    print(f"extension: static error {max(errs):.1e}; frequency errors {ferr}; order {order:.2f}", flush=True)
    num("cdeExtStatic", sci(max(errs))); num("cdeExtFreqErr", sci(ferr[-1])); num("cdeExtFreqOrder", f"{order:.1f}")
    num("cdeExtFreqCoarse", f"{ferr[0] * 100:.0f}\\%")
    ELEMENT_REGISTRY.pop("bar1d", None)


def main():
    print(platform.platform(), "numpy", np.__version__, flush=True)
    exp_registry(); exp_packages(); exp_extension(); exp_dispatch()
    num("cdeMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch01.tex"), "w") as fh:
        fh.write("% generated by scripts/ch01_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()

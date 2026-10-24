# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
appA_assets.py -- figures and number macros of Appendix A (differentiable assembly and autograd tangents).

Everything here runs WITHOUT PyTorch, on the torch-free parts of the packages:
  A  tangent of Tet4NeoHookean by central differences against a complex-step reference, as a function of the step h
     (the floor that an exact autograd tangent does not have)
  B  Newton solve of the corrected residual, with and without the Jacobian of the correction in the tangent
  C  gradient of a loss through a converged corrected Newton solve by the adjoint formula (adjoint_gradient) against
     central differences of the whole solve, on a one-dof and a two-dof truss
  D  metric identity of MetricScaledField (Euclidean step in the reparametrised variable = mass-metric step)
Quantities that need torch (autograd tangents, implicit layer, trained corrections) are NOT regenerated here; the
chapter quotes them from the developer-machine record and says so.
Writes into book/: figures/appA_checks.pdf; tables/numbers_appA.tex.

Run from the repository root:
    PYTHONPATH=fea_engine/src:rom_engine/src:fea_engine/tests python packages_book/book/scripts/appA_assets.py
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
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


from fea_engine.elements.nonlinear_solids import Tet4NeoHookean                           # noqa: E402
from fea_engine.material import NeoHookeanMaterial                                         # noqa: E402
from fea_engine.differentiable import (AdditiveCorrection, _newton_equilibrium, corrected_residual,   # noqa: E402
                                       corrected_tangent, adjoint_gradient, mass_cholesky_factor,
                                       metric_step_matches_euclidean_reparametrized_step)
from fea_engine import elements as elmod                                                    # noqa: E402
from fea_engine.mesh import Mesh                                                            # noqa: E402
from fea_engine.solver import FESystem                                                      # noqa: E402


# ---------------------------------------------------------------- A: FD floor of the Tet4 tangent
def experiment_A():
    coords = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    mat = NeoHookeanMaterial(E=1.0, nu=0.3)
    el = Tet4NeoHookean()
    rng = np.random.default_rng(1)
    u = 0.15 * rng.standard_normal(12)

    def f_cs(v):
        return el.internal_force(coords, v, mat)
    # complex-step reference: K[:, j] = Im f(u + i h e_j) / h
    hc = 1e-30
    Kref = np.zeros((12, 12))
    for j in range(12):
        d = np.zeros(12, dtype=complex); d[j] = 1j * hc
        Kref[:, j] = np.imag(f_cs(u.astype(complex) + d)) / hc
    asym = np.max(np.abs(Kref - Kref.T)) / np.max(np.abs(Kref))
    hs = np.logspace(-12, -1, 23)
    err = []
    for h in hs:
        K = el.tangent_stiffness(coords, u, mat, h=h)
        err.append(np.max(np.abs(K - Kref)) / np.max(np.abs(Kref)))
    err = np.array(err)
    h_best = hs[np.argmin(err)]
    K_def = el.tangent_stiffness(coords, u, mat)           # default h = 1e-6
    e_def = np.max(np.abs(K_def - Kref)) / np.max(np.abs(Kref))
    num("adaFdBest", sci(err.min())); num("adaFdBestH", sci(h_best)); num("adaFdDefault", sci(e_def))
    num("adaFdAtLarge", sci(err[-1])); num("adaFdAtSmall", sci(err[0])); num("adaCsAsym", sci(asym))
    return hs, err


# ---------------------------------------------------------------- shared fixtures (as in the package's tests)
def single_bar(E=200e9, A=1e-4, L0=2.0, F=5.0e3):
    nodes = np.array([[0.0, 0.0], [L0, 0.0]]); elements = np.array([[0, 1]], dtype=int)
    fes = FESystem(Mesh(nodes=nodes, elements=elements, dim=2), elmod.TrussTL2D())
    fes.fix_dofs([0], [0, 1]); fes.fix_dofs([1], [1]); fes.add_nodal_force([1], 0, F)
    return fes, (E, A), fes.F.copy()


def two_bar(a=1.0, h0=0.6, E=210e9, A=2e-4, F=2.0e3):
    nodes = np.array([[-a, 0.0], [0.0, h0], [a, 0.0]]); elements = np.array([[0, 1], [1, 2]], dtype=int)
    fes = FESystem(Mesh(nodes=nodes, elements=elements, dim=2), elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1]); fes.add_nodal_force([1], 1, -F)
    return fes, (E, A), fes.F.copy()


class ToyCorrection(AdditiveCorrection):
    """f_theta(u) = theta * (A @ u), as in tests/test_differentiable.py."""
    def __init__(self, theta, A):
        self.theta, self.A = theta, A

    def value(self, u):
        return self.theta * (self.A @ u)

    def jacobian(self, u):
        return self.theta * self.A

    def dvalue_dtheta(self, u):
        return self.A @ u


class LazyJacobian(ToyCorrection):
    def jacobian(self, u):
        return np.zeros((len(u), len(u)))


def newton_history(fes, mat, corr, F_ext, n_it=16):
    free = fes.free_dofs
    u = np.zeros(fes.n_dof)
    hist = []
    for _ in range(n_it):
        R, _f = corrected_residual(fes, u, mat, corr, F_ext=F_ext)
        hist.append(np.linalg.norm(R))
        K = corrected_tangent(fes, u, mat, corr)
        u[free] += np.linalg.solve(K, R)
    return np.array(hist)


# ---------------------------------------------------------------- B: Newton with / without the correction Jacobian
def experiment_B():
    fes1, mat, F_ext = two_bar(); fes2, _, _ = two_bar()
    n = len(fes1.free_dofs)
    theta = 3.0e6
    A = np.array([[1.0, 0.3], [0.3, 1.0]])
    full = newton_history(fes1, mat, ToyCorrection(theta, A), F_ext)
    lazy = newton_history(fes2, mat, LazyJacobian(theta, A), F_ext)
    thresh = 1e-10 * np.linalg.norm(F_ext[fes1.free_dofs])
    def count(h):
        idx = np.where(h < thresh)[0]
        return int(idx[0]) if len(idx) else len(h)
    num("adaItFull", str(count(full))); num("adaItLazy", str(count(lazy)))
    num("adaTheta", sci(theta))
    return full, lazy


# ---------------------------------------------------------------- C: adjoint against finite differences
def solve_state(builder, theta, A):
    fes, mat, F_ext = builder()
    corr = ToyCorrection(theta, A)
    u_full, u_free, K_eff, F_int, conv, Rn = _newton_equilibrium(fes, mat, corr, F_ext, tol=1e-10, max_iter=50)
    assert conv, Rn
    return u_free, K_eff, corr


def experiment_C():
    out = {}
    cases = {"one dof": (single_bar, np.eye(1), np.array([1.0])),
             "two dof": (two_bar, np.array([[0.8, -0.4], [0.3, 1.1]]), np.array([1.3, -0.7]))}
    rel_eps = np.logspace(-8, -1, 15)
    for name, (builder, A, w) in cases.items():
        theta0 = 1.0e5
        u0, K0, c0 = solve_state(builder, theta0, A)
        grad_w = w * (1.0 if len(w) > 1 else 1.0)
        if len(w) == 1:
            grad_w = u0.copy()                  # L = 0.5 u^2

        def loss(u):
            return 0.5 * np.sum(u ** 2) if len(w) == 1 else float(np.dot(w, u))
        g_adj = adjoint_gradient(K0, grad_w, lambda lam: np.array([np.dot(lam, c0.dvalue_dtheta(u0))]))[0]
        errs = []
        for re in rel_eps:
            e = theta0 * re
            up, _, _ = solve_state(builder, theta0 + e, A)
            um, _, _ = solve_state(builder, theta0 - e, A)
            g_fd = (loss(up) - loss(um)) / (2 * e)
            errs.append(abs(g_adj - g_fd) / abs(g_fd))
        out[name] = (np.array(errs), g_adj)
        key = "One" if name == "one dof" else "Two"
        num(f"ada{key}MinErr", sci(min(errs)))
        num(f"ada{key}Grad", sci(abs(g_adj)))
        i3 = int(np.argmin(np.abs(np.log10(rel_eps) + 3)))
        num(f"ada{key}ErrMid", sci(errs[i3]))
    return rel_eps, out


# ---------------------------------------------------------------- D: metric identity
def experiment_D():
    worst = 0.0
    rng = np.random.default_rng(0)
    for n in (3, 5, 20, 50):
        Bm = rng.standard_normal((n, n)); M = Bm @ Bm.T + n * np.eye(n)
        N = mass_cholesky_factor(M)
        g = rng.standard_normal(n)
        d1, d2 = metric_step_matches_euclidean_reparametrized_step(N, g, lr=0.1)
        worst = max(worst, np.max(np.abs(d1 - d2)) / np.max(np.abs(d2)))
    num("adaMetricErr", sci(worst))
    # mesh-density illustration: 1-D lumped mass of a graded mesh
    h = np.concatenate([np.full(10, 0.01), np.full(10, 0.1)])
    M = np.diag(h)
    g = np.ones(20)
    step_eucl = 0.1 * g                     # plain step on theta: same at every node
    step_metric = 0.1 * (M @ g)             # step through the reparametrisation: scales with the cell size
    num("adaMeshRatio", f"{step_metric[-1] / step_metric[0]:.0f}")


def main():
    hs, errA = experiment_A()
    full, lazy = experiment_B()
    rel_eps, outC = experiment_C()
    experiment_D()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].loglog(hs, errA, "o-", ms=3, color="#1f4e9c")
    ax[0].set_xlabel("difference step $h$"); ax[0].set_ylabel("rel. error of tangent")
    ax[0].set_title("(a) Tet4NeoHookean: central difference", fontsize=9)
    ax[1].semilogy(range(len(full)), np.maximum(full, 1e-18), "o-", ms=3, color="#6a3d9a", label="with $\\partial f_\\theta/\\partial u$")
    ax[1].semilogy(range(len(lazy)), np.maximum(lazy, 1e-18), "s--", ms=3, color="#e08000", label="zero Jacobian")
    ax[1].set_xlabel("Newton iteration"); ax[1].set_ylabel("$\\|R\\|$ (N)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Newton with a correction", fontsize=9)
    for name, (errs, g) in outC.items():
        ax[2].loglog(rel_eps, errs, "o-", ms=3, label=name)
    ax[2].set_xlabel("relative step of $\\theta$ in the difference"); ax[2].set_ylabel("rel. diff. adjoint vs FD")
    ax[2].legend(fontsize=7); ax[2].set_title("(c) gradient through the solve", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "appA_checks.pdf")); plt.close(fig)
    with open(os.path.join(TAB, "numbers_appA.tex"), "w") as fh:
        fh.write("% generated by scripts/appA_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    for k, v in NUM.items():
        print(k, v)


if __name__ == "__main__":
    main()

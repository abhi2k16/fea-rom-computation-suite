# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
rom_07_contact_rom_dynamics.py -- Example Gallery: contact reduced-order models. A membrane is pushed onto a parametrised rigid
obstacle (a flat floor with two bumps), as in Balajewicz, Amsallem and Farhat (2016), on a much smaller mesh (18 x 18 interior nodes).

Panel 1: dynamic contact. The full-order BDF2 trajectory (`fea_engine.solve_contact_dynamics(..., scheme="bdf2")`) against
         `ContactROM.solve_dynamic` (POD displacement basis, nonnegative-matrix-factorisation dual basis, p = p_lambda = 10 / 30 / 40):
         centre-node displacement against time.
Panel 2: relative error of the reduced trajectory against the number of basis vectors, and the largest KKT violation of the reduced
         contact conditions per step.
Panel 3: greedy snapshot selection (`greedy_contact_sampling` with `ContactConditionIndicator`) against random selection of the same
         number of snapshots, for the STATIC problem: mean relative error over six held-out obstacles.

Everything is run here and printed; only fea_engine (full-order reference) and rom_engine are used.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fea_engine import solve_contact_dynamics, solve_contact_static
from rom_engine import (ContactROM, ContactConditionIndicator, greedy_contact_sampling, nonneg_dual_basis, active_snapshot_columns)

HERE = os.path.dirname(os.path.abspath(__file__))
H, N_STEPS = 0.005, 120


class ObstacleMembrane:
    def __init__(self, n=18, load=10.0):
        self.n, self.h = n, 1.0 / (n + 1)
        x = np.linspace(self.h, 1 - self.h, n)
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        I = sp.identity(n)
        T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
        self.K = (sp.kron(T, I) + sp.kron(I, T)).tocsr()
        self.n_dof = n * n
        self.f = -load * self.h ** 2 * np.ones(self.n_dof)
        self.B = sp.identity(self.n_dof, format="csr")

    def obstacle(self, gamma):
        g1, g2 = gamma
        g = (-0.30 + 0.12 * np.exp(-60 * ((self.X - g1) ** 2 + (self.Y - 0.5) ** 2))
             + g2 * np.exp(-60 * ((self.X - 0.7) ** 2 + (self.Y - 0.5) ** 2)))
        return g.ravel()

    def solve(self, gamma):
        c = self.obstacle(gamma)
        r = solve_contact_static(self.K, self.f, self.B, c)
        return r.u, r.lam, c


def main():
    mem = ObstacleMembrane(18)
    n = mem.n_dof
    M = sp.diags(np.full(n, mem.h ** 2)).tocsr()
    c = mem.obstacle((0.4, 0.15))
    force = lambda t: mem.f
    Uf, Lf = solve_contact_dynamics(M, mem.K, force, mem.B, c, H, N_STEPS, scheme="bdf2")
    print(f"full-order BDF2: {n} DOFs, {N_STEPS} steps, up to {(Lf > 1e-9).sum(1).max()} active constraints in a step")

    Ux_full = np.linalg.svd(Uf[1:].T, full_matrices=False)[0]
    act = active_snapshot_columns(Lf[1:].T)
    sizes, errs, kkt, traj = (10, 30, 40), [], [], {}
    for p in sizes:
        Ux = Ux_full[:, :p]
        Ul = nonneg_dual_basis(Lf[1:].T[:, act], p).U
        rom = ContactROM(mem.K, mem.B, Ux, Ul, M=M)
        r = rom.solve_dynamic(force, c, H, N_STEPS)
        e = 100 * np.sum((r.u - Uf) ** 2) / np.sum(Uf ** 2) if r.ok else np.nan
        errs.append(e); kkt.append(r.kkt_residual.max() if r.ok else np.nan); traj[p] = r.u[:, n // 2] if r.ok else None
        print(f"p = p_lambda = {p:3d}: ok {r.ok}, relative error {e:.3g} %, max KKT residual {kkt[-1]:.1e}")

    # ---- static greedy against random
    cands = [(a, b) for a in np.linspace(0.3, 0.6, 6) for b in np.linspace(0.0, 0.1, 6)]
    test = [(0.33, 0.02), (0.41, 0.07), (0.47, 0.04), (0.58, 0.09), (0.36, 0.095), (0.55, 0.01)]
    ref = [mem.solve(g) for g in test]

    def make_rom(U, L):
        Q, _ = np.linalg.qr(U)
        Ul = L / np.maximum(np.linalg.norm(L, axis=0), 1e-300)
        rom = ContactROM(mem.K, mem.B, Q, Ul)
        return lambda p: rom.solve(mem.f, mem.obstacle(p)).u

    test_err = lambda rom: 100 * np.mean([np.linalg.norm(rom(g) - r[0]) / np.linalg.norm(r[0]) for g, r in zip(test, ref)])
    indic = ContactConditionIndicator(mem.K, mem.B, mem.f, weights=(1, 0, 0))
    ind = lambda p, u: indic(u, mem.obstacle(p))
    rng = np.random.default_rng(0)
    ns = (2, 4, 6, 8)
    e_greedy, e_rand_med, e_rand_max = [], [], []
    for k in ns:
        rom, hist = greedy_contact_sampling(cands, lambda p: mem.solve(p)[:2], make_rom, ind, k)
        e_greedy.append(test_err(rom))
        rr = []
        for _ in range(8):
            idx = rng.choice(len(cands), k, replace=False)
            sols = [mem.solve(cands[i]) for i in idx]
            rr.append(test_err(make_rom(np.array([s[0] for s in sols]).T, np.array([s[1] for s in sols]).T)))
        e_rand_med.append(np.median(rr)); e_rand_max.append(np.max(rr))
        print(f"{k} snapshots: greedy {e_greedy[-1]:.3g} %, random median {e_rand_med[-1]:.3g} % (worst of 8: {e_rand_max[-1]:.3g} %)")

    t = np.arange(N_STEPS + 1) * H
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax[0].plot(t, Uf[:, n // 2], "k", lw=2.5, label="full order (BDF2)")
    for p, st in zip(sizes, ("--", "-.", ":")):
        if traj[p] is not None:
            ax[0].plot(t, traj[p], st, label=f"ROM p = {p}")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("centre displacement"); ax[0].set_title("dynamic contact ROM vs full order"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].semilogy(sizes, errs, "o-")
    for p_, e_ in zip(sizes, errs):
        ax[1].annotate(f"{e_:.3g} %", (p_, e_), textcoords="offset points", xytext=(6, 6))
    ax[1].set_xlabel("p = p_lambda"); ax[1].set_ylabel("relative error (%)")
    ax[1].set_title(f"reduced trajectory error (max KKT residual {max(kkt):.0e})"); ax[1].grid(True, which="both", alpha=0.3)
    ax[2].semilogy(ns, e_greedy, "o-", label="greedy (ContactConditionIndicator)")
    ax[2].semilogy(ns, e_rand_med, "s--", label="random, median of 8")
    ax[2].semilogy(ns, e_rand_max, "^:", label="random, worst of 8")
    ax[2].set_xlabel("snapshots"); ax[2].set_ylabel("mean test error (%)"); ax[2].set_title("static ROM: greedy vs random snapshots"); ax[2].grid(True, which="both", alpha=0.3); ax[2].legend()
    fig.tight_layout()
    out = os.path.join(HERE, "rom_07_contact_rom_dynamics.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_10_bdf2_contact_dynamics.py -- Example Gallery: second-order contact time stepping, `solve_contact_dynamics(..., scheme="bdf2")`
against the default first-order `scheme="two_step"`.

Panel 1: error at t = 2 of a free membrane (Quad4Laplace, 15 x 15 elements, contact inactive) against the exact modal solution, for
         dt = 0.01 / 0.005 / 0.0025 -- order about 2 (BDF2) against about 1 (two_step).
Panel 2: an elastic bar (40 nodes) moving at unit speed into a rigid wall: contact force at the wall, both schemes.
Panel 3: total energy during that impact. The default scheme never gains energy; BDF2 is NOT contact-stabilised and gains a little
         (printed). `return_info=True` returns the velocities BDF2 computes (ContactDynamicsInfo.V).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fea_engine import FESystem, Quad4Laplace, solve_contact_dynamics
from fea_engine import mesh as meshlib

HERE = os.path.dirname(os.path.abspath(__file__))


def free_membrane(ne=15):
    m = meshlib.rectangle_mesh(1.0, 1.0, ne, ne)
    fes = FESystem(m, Quad4Laplace(), sparse=True)
    fes.assemble_stiffness(1.0)
    fes.assemble_lumped_mass(1.0)
    X = m.nodes
    edge = np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > 1 - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > 1 - 1e-9))
    fes.fix_dofs(edge, ["u"])
    free = np.asarray(fes.free_dofs)
    K = sp.csr_matrix(fes.K)[free][:, free].toarray()
    M = np.diag(np.asarray(sp.csr_matrix(fes.M_lumped).diagonal())[free])
    f = -10.0 * (M @ np.ones(len(free)))
    return K, M, f


def bar_impact(h, T=1.5):
    n, dx = 40, 1.0 / 39
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i:i + 2, i:i + 2] += (1 / dx) * np.array([[1, -1], [-1, 1]])
    K += np.eye(n) * 1e-3 / n                         # weak ground springs: K positive definite
    M = np.diag(np.full(n, dx))
    M[0, 0] = M[-1, -1] = dx / 2
    B = np.zeros((1, n)); B[0, 0] = 1.0
    c = np.array([-0.02])                             # wall: u_0 >= -0.02
    v0 = -np.ones(n)
    N = int(round(T / h))
    Ub, Lb, ib = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0, scheme="bdf2", return_info=True)
    U2, L2 = solve_contact_dynamics(M, K, lambda t: np.zeros(n), B, c, h, N, v0=v0)
    V2 = np.zeros_like(U2)
    V2[0] = v0
    V2[1:-1] = (U2[2:] - U2[:-2]) / (2 * h)
    V2[-1] = (U2[-1] - U2[-2]) / h
    energy = lambda U, V: np.array([0.5 * V[k] @ M @ V[k] + 0.5 * U[k] @ K @ U[k] for k in range(len(U))])
    return dict(t=np.arange(N + 1) * h, Lb=Lb[:, 0], L2=L2[:, 0], Eb=energy(Ub, ib.V), E2=energy(U2, V2))


def main():
    K, M, f = free_membrane(15)
    n = K.shape[0]
    w2, V = sl.eigh(K, M)
    a = V.T @ f
    exact = V @ ((a / w2) * (1 - np.cos(np.sqrt(w2) * 2.0)))
    B, c = np.eye(n), np.full(n, -100.0)             # obstacle far away: contact inactive
    dts = (0.01, 0.005, 0.0025)
    eb, et = [], []
    for h in dts:
        N = int(round(2 / h))
        U2, _ = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N, scheme="bdf2")
        U1, _ = solve_contact_dynamics(M, K, lambda t: f, B, c, h, N)
        eb.append(np.abs(U2[-1] - exact).max()); et.append(np.abs(U1[-1] - exact).max())
    print("dt          ", dts)
    print("error two_step", ["%.3e" % e for e in et])
    print("error bdf2    ", ["%.3e" % e for e in eb])
    print("ratio two_step / bdf2", ["%.0f" % (x / y) for x, y in zip(et, eb)])
    print("observed order two_step", np.round(np.log2(np.array(et[:-1]) / np.array(et[1:])), 2),
          " bdf2", np.round(np.log2(np.array(eb[:-1]) / np.array(eb[1:])), 2))

    imp = bar_impact(0.002)
    E0 = imp["Eb"][0]
    print(f"impact (h = 0.002): BDF2 energy max/initial {imp['Eb'].max() / E0:.4f}, final/initial {imp['Eb'][-1] / E0:.4f}; "
          f"two_step max/initial {imp['E2'].max() / E0:.4f}, final/initial {imp['E2'][-1] / E0:.4f}")
    print(f"min contact force: BDF2 {imp['Lb'].min():.2e}, two_step {imp['L2'].min():.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
    ax[0].loglog(dts, et, "o-", label="two_step (default)")
    ax[0].loglog(dts, eb, "s-", label='scheme="bdf2"')
    d = np.array(dts)
    ax[0].loglog(d, et[0] * (d / d[0]), "k:", lw=1, label="slope 1")
    ax[0].loglog(d, eb[0] * (d / d[0]) ** 2, "k--", lw=1, label="slope 2")
    ax[0].set_xlabel("time step dt"); ax[0].set_ylabel("max error at t = 2")
    ax[0].set_title("free membrane against the exact modal solution"); ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
    ax[1].plot(imp["t"], imp["L2"], label="two_step"); ax[1].plot(imp["t"], imp["Lb"], label="bdf2")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("wall force"); ax[1].set_title("bar impact: contact force (h = 0.002)")
    ax[1].grid(alpha=0.3); ax[1].legend()
    ax[2].plot(imp["t"], imp["E2"] / E0, label="two_step"); ax[2].plot(imp["t"], imp["Eb"] / E0, label="bdf2")
    ax[2].axhline(1.0, color="k", lw=0.8)
    ax[2].set_xlabel("t"); ax[2].set_ylabel("energy / initial energy"); ax[2].set_title("bar impact: energy (BDF2 not contact-stabilised)")
    ax[2].grid(alpha=0.3); ax[2].legend()
    fig.tight_layout()
    out = os.path.join(HERE, "fea_10_bdf2_contact_dynamics.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Contact benchmark fixtures shared by the contact ROM tests. Plain arrays only (the full-order solve uses
fea_engine.contact_solvers, which tests are allowed to import; the library code never does).

ObstacleMembrane: a square membrane (five-point Laplacian, zero Dirichlet edge) pushed down by a uniform load onto
a parametrised obstacle below it, the model problem of Balajewicz et al. 2015 and Fauque et al. 2018. Meshes are
much smaller than theirs (n x n interior nodes, n = 18 by default)."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp

from fea_engine.contact_solvers import solve_contact_static


class ObstacleMembrane:
    def __init__(self, n=18, load=10.0):
        self.n = n
        self.h = 1.0 / (n + 1)
        x = np.linspace(self.h, 1 - self.h, n)
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        I = sp.identity(n)
        T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
        self.K = (sp.kron(T, I) + sp.kron(I, T)).tocsr()              # -Laplacian * h^2
        self.n_dof = n * n
        self.f = -load * self.h ** 2 * np.ones(self.n_dof)            # membrane pushed down
        self.B = sp.identity(self.n_dof, format="csr")                # every node may touch the obstacle

    def obstacle(self, gamma):
        """u >= g(gamma), g a flat floor with two bumps; gamma = (position of bump 1, height of bump 2)."""
        g1, g2 = gamma
        g = (-0.30 + 0.12 * np.exp(-60 * ((self.X - g1) ** 2 + (self.Y - 0.5) ** 2))
             + g2 * np.exp(-60 * ((self.X - 0.7) ** 2 + (self.Y - 0.5) ** 2)))
        return g.ravel()

    def solve(self, gamma):
        c = self.obstacle(gamma)
        res = solve_contact_static(self.K, self.f, self.B, c)
        return res.u, res.lam, c

    def training_set(self, gammas):
        U, L = [], []
        for g in gammas:
            u, lam, _ = self.solve(g)
            U.append(u)
            L.append(lam)
        return np.array(U).T, np.array(L).T


class NonlinearChainContact:
    """A chain of hardening springs (force k (d + k3 d^3) for elongation d), node 0 grounded, uniform downward load,
    with a rigid obstacle below a block of nodes. Element e joins free DOF e-1 and e (the first element joins the
    ground and DOF 0), so each element has one or two DOFs. The full-order solve is fea_engine's sequential-
    linearisation contact Newton. Used to test hyper-reduction with contact on a model small enough to run fast."""

    def __init__(self, n=60, k=50.0, k3=8.0, load=1.0, obstacle_nodes=None, linear_below=0):
        self.n, self.k, self.k3, self.load = n, k, k3, load
        self.linear_below = linear_below                      # elements with index < linear_below are linear (k3 = 0)
        self.elem_dofs = [np.array([0])] + [np.array([i - 1, i]) for i in range(1, n)]
        self.f = -load * np.ones(n) / n * 5.0
        self.obs_nodes = np.arange(int(0.55 * n), int(0.8 * n)) if obstacle_nodes is None else np.asarray(obstacle_nodes)
        self.B = np.zeros((len(self.obs_nodes), n))
        self.B[np.arange(len(self.obs_nodes)), self.obs_nodes] = 1.0

    def elem_force(self, e, ul):
        d = ul[0] if len(ul) == 1 else ul[1] - ul[0]
        g = self.k * (d + (self.k3 if e >= self.linear_below else 0.0) * d ** 3)
        return np.array([g]) if len(ul) == 1 else np.array([-g, g])

    def elem_tangent(self, e, ul):
        d = ul[0] if len(ul) == 1 else ul[1] - ul[0]
        kt = self.k * (1 + 3 * (self.k3 if e >= self.linear_below else 0.0) * d ** 2)
        return np.array([[kt]]) if len(ul) == 1 else kt * np.array([[1.0, -1.0], [-1.0, 1.0]])

    def internal_force(self, u):
        f = np.zeros(self.n)
        for e, g in enumerate(self.elem_dofs):
            f[g] += self.elem_force(e, u[g])
        return f

    def tangent(self, u):
        K = np.zeros((self.n, self.n))
        for e, g in enumerate(self.elem_dofs):
            K[np.ix_(g, g)] += self.elem_tangent(e, u[g])
        return K

    def solve(self, scale=1.0, gap=-0.12):
        """Full-order contact solution for a load factor `scale` and obstacle height `gap` (u >= gap at the block)."""
        from fea_engine.contact_solvers import solve_contact_newton
        fe = scale * self.f
        c = np.full(len(self.obs_nodes), gap)
        res = solve_contact_newton(lambda u: self.internal_force(u) - fe, self.tangent, self.B, c, np.zeros(self.n))
        return res.u, res.lam, c, fe

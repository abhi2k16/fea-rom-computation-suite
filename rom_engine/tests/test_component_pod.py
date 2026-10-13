# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Component-wise POD with mortar coupling on two non-matching nonlinear scalar membranes."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp

from fea_engine.mortar import MortarTie2D
from rom_engine.component_pod import (legendre_interface_parametrisation, component_snapshots, ComponentBasis,
                                      ReducedAssembly)

K3 = 4.0


class Grid:
    def __init__(self, nx, ny, x0, x1, interface_side):
        xs, ys = np.linspace(x0, x1, nx), np.linspace(0, 1, ny)
        self.coords = np.array([[x, y] for x in xs for y in ys])
        self.n = nx * ny
        idx = lambda i, j: i * ny + j
        K = np.zeros((self.n, self.n))
        for i in range(nx):
            for j in range(ny):
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < nx and j + dj < ny:
                        a, b = idx(i, j), idx(i + di, j + dj)
                        K[a, a] += 1; K[b, b] += 1; K[a, b] -= 1; K[b, a] -= 1
        K += 0.5 * np.eye(self.n)
        outer = 0 if interface_side == "right" else nx - 1
        self.outer = np.array([idx(outer, j) for j in range(ny)])
        self.interface = np.array([idx(nx - 1 if interface_side == "right" else 0, j) for j in range(ny)])
        for d in self.outer:
            K[d, d] += 1e4                             # clamped outer edge (penalty keeps K positive definite)
        self.K = K
        self.f = np.ones(self.n)
        self.ys = ys

    def resid(self, u, lam):
        return self.K @ u + K3 * u ** 3 - lam * self.f

    def tang(self, u):
        return self.K + np.diag(3 * K3 * u ** 2)

    def solve_with_interface(self, g, lam):
        u = np.zeros(self.n)
        u[self.interface] = g
        free = np.setdiff1d(np.arange(self.n), self.interface)
        for _ in range(40):
            r = self.resid(u, lam)[free]
            if np.linalg.norm(r) < 1e-11:
                break
            u[free] -= np.linalg.solve(self.tang(u)[np.ix_(free, free)], r)
        return u


@pytest.fixture(scope="module")
def setup():
    A = Grid(9, 9, 0.0, 1.0, "right")
    B = Grid(9, 7, 1.0, 2.0, "left")
    na = A.n
    coords = np.vstack([A.coords, B.coords])
    tie = MortarTie2D(na + B.interface, A.interface, coords, na + B.n, dofs_per_node=1)
    T = tie.transformation()
    lam_ref = 1.5

    def resid_full(u):
        return np.concatenate([A.resid(u[:na], lam_ref), B.resid(u[na:], lam_ref)])

    def tang_full(u):
        return sp.block_diag([A.tang(u[:na]), B.tang(u[na:])]).tocsr()
    x = np.zeros(T.shape[1])
    for _ in range(30):
        u = T @ x
        r = T.T @ resid_full(u)
        if np.linalg.norm(r) < 1e-11:
            break
        x -= np.linalg.solve((T.T @ tang_full(u) @ T).toarray(), r)
    u_ref = T @ x
    G_A = legendre_interface_parametrisation(A.ys, 3)
    G_B = legendre_interface_parametrisation(B.ys, 3)
    return dict(A=A, B=B, tie=tie, P=tie.P, u_ref=u_ref, resid=resid_full, tang=tang_full, G_A=G_A, G_B=G_B, na=na)


def _basis(setup, rank_a, rank_b, n_samples=30):
    A, B = setup["A"], setup["B"]
    fa = lambda p: A.solve_with_interface(setup["G_A"] @ p[:4], p[4])
    fb = lambda p: B.solve_with_interface(setup["G_B"] @ p[:4], p[4])
    # scale the load coordinate around the target value by shifting the sampling box: load in [0, 3]
    wrap = lambda f: (lambda p: f(np.concatenate([p[:4], [1.5 + 1.5 * p[4]]])))
    Sa, _ = component_snapshots(wrap(fa), 5, n_samples, scale=0.5, rng=1)
    Sb, _ = component_snapshots(wrap(fb), 5, n_samples, scale=0.5, rng=2)
    ba = ComponentBasis(Sa, A.interface, rank=rank_a)
    bb = ComponentBasis(Sb, B.interface, rank=rank_b, unreduced_interface=True)
    return ba, bb


def test_reduced_assembly_matches_the_full_tied_solution(setup):
    ba, bb = _basis(setup, 12, 12)
    asm = ReducedAssembly(ba, bb, setup["P"])
    res = asm.solve(setup["resid"], setup["tang"])
    assert res.converged
    err = np.linalg.norm(res.u - setup["u_ref"]) / np.linalg.norm(setup["u_ref"])
    assert err < 1e-2
    assert res.n_reduced == 24 and res.n_full == setup["A"].n + setup["B"].n
    # the slave interface of the reduced solution obeys the mortar tie exactly
    na, B, A = setup["na"], setup["B"], setup["A"]
    assert np.allclose(res.u[na + B.interface], setup["P"] @ res.u[A.interface], atol=1e-12)


def test_error_decreases_with_basis_size(setup):
    errs = []
    for r in (3, 6, 12):
        ba, bb = _basis(setup, r, r)
        res = ReducedAssembly(ba, bb, setup["P"]).solve(setup["resid"], setup["tang"])
        errs.append(np.linalg.norm(res.u - setup["u_ref"]) / np.linalg.norm(setup["u_ref"]))
    assert errs[0] > errs[1] > errs[2]


def test_one_master_basis_is_reused_against_a_different_slave_mesh(setup):
    ba, bb = _basis(setup, 10, 10)
    A = setup["A"]
    B2 = Grid(9, 5, 1.0, 2.0, "left")                   # a coarser, different slave component
    na = A.n
    coords = np.vstack([A.coords, B2.coords])
    tie2 = MortarTie2D(na + B2.interface, A.interface, coords, na + B2.n, dofs_per_node=1)
    T2 = tie2.transformation()
    resid = lambda u: np.concatenate([A.resid(u[:na], 1.5), B2.resid(u[na:], 1.5)])
    tang = lambda u: sp.block_diag([A.tang(u[:na]), B2.tang(u[na:])]).tocsr()
    x = np.zeros(T2.shape[1])
    for _ in range(30):
        u = T2 @ x
        r = T2.T @ resid(u)
        if np.linalg.norm(r) < 1e-11:
            break
        x -= np.linalg.solve((T2.T @ tang(u) @ T2).toarray(), r)
    u_ref2 = T2 @ x
    G_B2 = legendre_interface_parametrisation(B2.ys, 3)
    wrap = lambda p: B2.solve_with_interface(G_B2 @ p[:4], 1.5 + 1.5 * p[4])
    Sb2, _ = component_snapshots(wrap, 5, 30, scale=0.5, rng=3)
    bb2 = ComponentBasis(Sb2, B2.interface, rank=10, unreduced_interface=True)
    res = ReducedAssembly(ba, bb2, tie2.P).solve(resid, tang)           # same master basis ba as before
    assert res.converged
    assert np.linalg.norm(res.u - u_ref2) / np.linalg.norm(u_ref2) < 2e-2


def test_parametrisation_and_snapshots_shapes():
    G = legendre_interface_parametrisation(np.linspace(0, 1, 5), 2, dofs_per_node=2)
    assert G.shape == (10, 6) and np.allclose(G[0::2, 0], 1) and np.allclose(G[1::2, :3], 0)
    S, D = component_snapshots(lambda p: np.array([p.sum(), (p ** 2).sum()]), 3, 5, rng=0)
    assert S.shape == (2, 5 + 1 + 6) and D.shape == (12, 3)

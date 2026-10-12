# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
nonlinear_beam_fixtures.py -- shared fixtures for the system-driven basis and quadratic-manifold tests.

Two kinds of model, both returned as plain arrays and callables so the library modules under test never see an
fea_engine object:

  synthetic_cubic_model()  -- a small system with an EXACT quadratic-plus-cubic internal force built from a random
      symmetric third-order tensor. Every derivative the library has to compute is known in closed form here, so
      this is the strict correctness check.
  nonlinear_beam_model()   -- the real geometrically nonlinear clamped-clamped Beam2DCorotational chain from
      fea_fixtures.py, restricted to its free DOFs. This is the realistic check: coupled bending and stretching,
      element-level internal forces, a real nonlinear static solver as the full-order reference.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.optimize import root

from fea_fixtures import clamped_clamped_nonlinear_beam_system


def synthetic_cubic_model(n=12, seed=3, scale=40.0):
    """f(u) = K0 u + 1/2 T(u, u) + 1/6 S(u, u, u) with symmetric T (n,n,n) and S (n,n,n,n). The tangent is
    K0 + T(u) + 1/2 S(u, u), so dK/du_k at the origin is exactly T[:, :, k]."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    K0 = A @ A.T + n * np.eye(n)
    B = rng.standard_normal((n, n))
    M = B @ B.T + n * np.eye(n)
    T = rng.standard_normal((n, n, n)) * scale
    T = sum(np.transpose(T, p) for p in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6.0
    S = rng.standard_normal((n, n, n, n)) * scale
    import itertools
    S = sum(np.transpose(S, p) for p in itertools.permutations(range(4))) / 24.0

    def f_int(u):
        return K0 @ u + 0.5 * np.einsum("ijk,j,k->i", T, u, u) + (1.0 / 6.0) * np.einsum("ijkl,j,k,l->i", S, u, u, u)

    def K_t(u):
        return K0 + np.einsum("ijk,k->ij", T, u) + 0.5 * np.einsum("ijkl,k,l->ij", S, u, u)

    return dict(n=n, K0=K0, M=M, T=T, S=S, f_int=f_int, K_t=K_t)


def nonlinear_beam_model(n_elem=20, n_modes=4, **kwargs):
    """Clamped-clamped geometrically nonlinear beam on its free DOFs.

    Returns a dict with: n_free, free, n (all DOFs), K0 and M (free-DOF matrices), Phi (n_free, n_modes
    M-orthonormal vibration modes), omega2, f_int(u_free), K_t(u_free), embed(u_free) -> full vector, the
    full-DOF element data for ECSW (elem_dofs, elem_force, elem_tangent) and `static(F_free)`, the real
    nonlinear static solution (Newton, load ramped in steps) used as the full-order reference.
    """
    d = clamped_clamped_nonlinear_beam_system(n_elem=n_elem, n_modes=n_modes, **kwargs)
    fes, mat = d["sys"], d["mat"]
    free = np.asarray(d["free_dofs"])
    n = d["n_dof"]
    nf = len(free)

    def embed(u):
        full = np.zeros(n)
        full[free] = u
        return full

    def f_int(u):
        return np.asarray(fes.assemble_internal_force(embed(u), mat))[free]

    def K_t(u):
        K = fes.assemble_tangent_stiffness(embed(u), mat)
        return np.asarray(K)[np.ix_(free, free)]

    M = np.asarray(fes.M)[np.ix_(free, free)]
    K0 = K_t(np.zeros(nf))
    from scipy.linalg import eigh
    w2, Phi = eigh(K0, M)
    Phi = Phi[:, :n_modes]
    conn = fes.mesh.elements
    elem = fes._blocks[0][1]
    elem_dofs = [np.asarray(fes._global_dofs(c)) for c in conn]
    elem_force = lambda e, u: elem.internal_force(fes.mesh.nodes[conn[e]], u, mat)
    elem_tangent = lambda e, u: elem.tangent_stiffness(fes.mesh.nodes[conn[e]], u, mat)
    return dict(d=d, fes=fes, mat=mat, free=free, n=n, n_free=nf, K0=K0, M=M, Phi=Phi, omega2=w2[:n_modes],
                f_int=f_int, K_t=K_t, embed=embed, elem_dofs=elem_dofs, elem_force=elem_force,
                elem_tangent=elem_tangent, static=d["fom_solver"], L=d["L"], mode_peaks=d["mode_shape_peaks"])


def reduced_static_solve(V, f_int, K_t, F, q0=None):
    """Galerkin static solution V^T (f_int(V q) - F) = 0 by Newton with the analytic reduced tangent."""
    m = V.shape[1]
    q0 = np.zeros(m) if q0 is None else q0
    sol = root(lambda q: V.T @ (f_int(V @ q) - F), q0, jac=lambda q: V.T @ K_t(V @ q) @ V, method="hybr",
               tol=1e-12)
    if not sol.success:
        raise RuntimeError(f"reduced static solve did not converge: {sol.message}")
    return sol.x

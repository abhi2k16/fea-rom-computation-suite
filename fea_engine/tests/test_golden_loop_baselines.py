# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_golden_loop_baselines.py -- the current loop assembly still reproduces the frozen results in
tests/data/golden_loop_baselines.npz (see golden_loop_baselines.py), and the conformance helpers in
batch_conformance.py accept a correct kernel and reject a wrong one (negative controls).

The golden values are loop results, not validated physics; the physics tests live in test_quad4_tl.py and
test_tet10_geometric_nonlinear.py. Force and tangent are compared to 1e-12 relative (same code, same machine
family, so differences come only from BLAS/summation order); Newton paths to 1e-9.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_loop_baselines as gb
import batch_conformance as bc
import fea_engine.nonlinear_solver as nls
from fea_engine import Material
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL

pytestmark = pytest.mark.skipif(not os.path.exists(gb.GOLDEN_PATH), reason="golden file not generated")


@pytest.fixture(scope="module")
def golden():
    return dict(np.load(gb.GOLDEN_PATH))


@pytest.fixture(scope="module")
def quad():
    return gb.quad_case()


@pytest.fixture(scope="module")
def tet():
    return gb.tet_case()


@pytest.mark.parametrize("k", range(len(gb.QUAD_SCALES)))
def test_quad_force_and_tangent(golden, quad, k):
    fes, mat, t = quad
    u = golden[f"quad_u{k}"]
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, thickness=t), golden[f"quad_f{k}"], label="force")
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, thickness=t), golden[f"quad_K{k}"], label="tangent")


@pytest.mark.parametrize("k", range(len(gb.TET_SCALES)))
def test_tet10_force_and_tangent(golden, tet, k):
    fes, mat = tet
    u = golden[f"tet_u{k}"]
    bc.equal_to_loop(fes.assemble_internal_force(u, mat), golden[f"tet_f{k}"], label="force")
    # Tet10SolidTL's tangent is analytic and deterministic; same rtol as the force
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat), golden[f"tet_K{k}"], label="tangent")


def test_quad_newton_path(golden):
    fes, mat, t = gb.quad_case()
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=t)
    assert lf[-1] == pytest.approx(1.0)
    bc.same_newton_path(np.asarray(U), golden["quad_newton_U"])


def test_tet10_newton_path(golden):
    fes, mat = gb.tet_case()
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-5, max_iter=40)
    assert lf[-1] == pytest.approx(1.0)
    # tol=1e-5 residual plateau (documented for Tet10SolidTL) -> compare at 1e-6, not 1e-9
    bc.same_newton_path(np.asarray(U), golden["tet_newton_U"], rtol=1e-6)


def test_golden_states_are_nontrivial(golden):
    """Guard against a degenerate golden file: forces are nonzero and the tangents are symmetric."""
    for name in ("quad", "tet"):
        for key in [k for k in golden if k.startswith(f"{name}_f")]:
            assert np.max(np.abs(golden[key])) > 1e-6
        for key in [k for k in golden if k.startswith(f"{name}_K")]:
            K = golden[key]
            assert np.max(np.abs(K - K.T)) < 1e-9 * np.max(np.abs(K))


# ----------------------------------------------------------------------------- helper negative controls
X = np.array([[0.0, 0.0], [2.0, 0.1], [2.2, 1.1], [-0.1, 0.9]])
D = D_plane_stress(Material(E=1.0e3, nu=0.3))
el = Quad4PlaneStressTL()


def test_helpers_accept_the_loop_element():
    f = lambda u: el.internal_force(X, u, D, 0.5)
    K = lambda u: el.tangent_stiffness(X, u, D, 0.5)
    u = np.random.default_rng(1).normal(size=8) * 0.05
    scale = float(np.max(np.abs(f(0.01 * (X.ravel() - X.ravel().mean())))))
    bc.objective(f, X, scale)
    bc.fd_tangent(f, K, u)
    # chunking negative-free case: a function that sums per-chunk partial sums of a fixed vector
    v = np.random.default_rng(2).normal(size=40)
    bc.chunk_invariant(lambda c: np.array([v.sum()]) if c is None else
                       np.array([sum(v[i:i + c].sum() for i in range(0, 40, c))]), 40)


def test_helpers_reject_wrong_kernels():
    u = np.random.default_rng(1).normal(size=8) * 0.05
    f = lambda u: el.internal_force(X, u, D, 0.5)
    K = lambda u: el.tangent_stiffness(X, u, D, 0.5)
    with pytest.raises(AssertionError):                                   # perturbed result
        bc.equal_to_loop(f(u) * (1 + 1e-9), f(u))
    with pytest.raises(AssertionError):                                   # a "linearised" force is not objective
        bc.objective(lambda w: el.internal_force(X, 0 * w, D, 0.5) + 1.0, X, 1.0)
    with pytest.raises(AssertionError):                                   # wrong tangent (scaled) fails the FD check
        bc.fd_tangent(f, lambda w: 1.01 * K(w), u)
    with pytest.raises(AssertionError):                                   # chunk-dependent result
        bc.chunk_invariant(lambda c: np.array([1.0 if c is None else 1.0 + 1e-6 * c]), 10)

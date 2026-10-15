# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_quad4_tl_batched.py -- batched Quad4PlaneStressTL kernels and FESystem's vectorized=True assembly.

Evidence (each against the per-element loop code, which is left untouched and acts as the reference):
  element level   internal_force_batch / tangent_stiffness_batch equal the per-element methods to 1e-12 relative for
                  Gauss orders 1, 2, 3, several thicknesses and displacement scales; the batched force is objective
                  (zero at a rigid rotation), the batched tangent equals central differences of the batched force
                  and is symmetric; with and without a precomputed geometry argument;
  system level    vectorized=True reproduces the frozen golden loop force/tangent (tests/data) to 1e-12, equals the
                  live loop, is independent of the chunk size (1, small, larger than the block), and leaves the default
                  path (vectorized not given) on the loop;
  fallbacks       blocks without batch kernels, blocks with path-dependent state, contact elements and springs are
                  assembled by the loop inside a vectorized assembly, and last_assembly_modes says which;
  Newton path     solve_nonlinear_static with fes.batched = True, and with vectorized=True forwarded, gives the golden
                  loop displacements at every load step to 1e-9.
Speed is not asserted here; tests/batch_timing.py reports it.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_conformance as bc
import golden_loop_baselines as gb
import fea_engine.nonlinear_solver as nls
from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL, Quad4PlaneStress
from fea_engine.elements.contact import GapContactPenalty

D = D_plane_stress(Material(E=gb.E_MOD, nu=gb.NU))
el = Quad4PlaneStressTL()
X1 = np.array([[0.0, 0.0], [2.0, 0.1], [2.2, 1.1], [-0.1, 0.9]])
GOLDEN = os.path.exists(gb.GOLDEN_PATH)


def _loop(el_, X, u, mat, t, what):
    fn = el_.internal_force if what == "force" else el_.tangent_stiffness
    return np.array([fn(x, w, mat, t) for x, w in zip(X, u)])


# ----------------------------------------------------------------------------- element level
@pytest.mark.parametrize("gauss_order", [1, 2, 3])
@pytest.mark.parametrize("scale", [0.02, 0.3])
@pytest.mark.parametrize("thickness", [0.5, 2.0])
def test_batch_equals_loop(gauss_order, scale, thickness):
    e = Quad4PlaneStressTL()
    e.gauss_order = gauss_order
    nodes, elems, _, _ = gb.quad_plate(6, 4)
    X = nodes[elems]
    u = np.random.default_rng(10 + gauss_order).normal(size=(len(elems), 8)) * scale
    bc.equal_to_loop(e.internal_force_batch(X, u, D, thickness), _loop(e, X, u, D, thickness, "force"), label="force")
    bc.equal_to_loop(e.tangent_stiffness_batch(X, u, D, thickness), _loop(e, X, u, D, thickness, "tangent"),
                     label="tangent")


def test_batch_with_and_without_precomputed_geometry():
    nodes, elems, _, _ = gb.quad_plate(5, 3)
    X = nodes[elems]
    u = np.random.default_rng(1).normal(size=(len(elems), 8)) * 0.1
    geo = el._reference_geometry_batch(X)
    bc.equal_to_loop(el.internal_force_batch(X, u, D, 0.5, geometry=geo), el.internal_force_batch(X, u, D, 0.5))
    bc.equal_to_loop(el.tangent_stiffness_batch(X, u, D, 0.5, geometry=geo), el.tangent_stiffness_batch(X, u, D, 0.5))


def test_batched_force_is_objective_and_tangent_is_its_derivative():
    X = X1[None]
    f = lambda w: el.internal_force_batch(X, w.reshape(1, 8), D, 0.5)[0]
    K = lambda w: el.tangent_stiffness_batch(X, w.reshape(1, 8), D, 0.5)[0]
    scale = float(np.max(np.abs(f(0.01 * (X1.ravel() - X1.ravel().mean())))))
    bc.objective(f, X1, scale)
    for s in (0.02, 0.2):
        bc.fd_tangent(f, K, np.random.default_rng(1).normal(size=8) * s)


def test_single_element_and_empty_batches():
    u = np.random.default_rng(2).normal(size=(1, 8)) * 0.1
    assert el.internal_force_batch(X1[None], u, D, 1.0).shape == (1, 8)
    assert el.tangent_stiffness_batch(X1[None], u, D, 1.0).shape == (1, 8, 8)
    bc.equal_to_loop(el.internal_force_batch(X1[None], u, D, 1.0)[0], el.internal_force(X1, u[0], D, 1.0))


# ----------------------------------------------------------------------------- system level
@pytest.fixture(scope="module")
def golden():
    return dict(np.load(gb.GOLDEN_PATH)) if GOLDEN else None


@pytest.mark.skipif(not GOLDEN, reason="golden file not generated")
@pytest.mark.parametrize("k", range(len(gb.QUAD_SCALES)))
def test_vectorized_system_assembly_matches_golden_loop(golden, k):
    fes, mat, t = gb.quad_case()
    u = golden[f"quad_u{k}"]
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, thickness=t, vectorized=True), golden[f"quad_f{k}"], label="force")
    assert fes.last_assembly_modes == {"q": "batched"}
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, thickness=t, vectorized=True), golden[f"quad_K{k}"],
                     label="tangent")
    assert fes.last_assembly_modes == {"q": "batched"}


def test_vectorized_equals_live_loop_and_default_stays_on_the_loop():
    fes, mat, t = gb.quad_case()
    u = gb.random_state(fes.n_dof, 0.1, 5)
    f_loop = fes.assemble_internal_force(u, mat, thickness=t)
    assert fes.last_assembly_modes["q"].startswith("loop") and "not requested" in fes.last_assembly_modes["q"]
    K_loop = fes.assemble_tangent_stiffness(u, mat, thickness=t)
    assert fes.last_assembly_modes["q"].startswith("loop")
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, thickness=t, vectorized=True), f_loop)
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, thickness=t, vectorized=True), K_loop)
    fes.batched = True                                                   # system-wide switch
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, thickness=t), f_loop)
    assert fes.last_assembly_modes == {"q": "batched"}
    fes.assemble_internal_force(u, mat, thickness=t, vectorized=False)   # an explicit argument overrides the switch
    assert fes.last_assembly_modes["q"].startswith("loop")


def test_system_assembly_is_chunk_invariant():
    fes, mat, t = gb.quad_case()
    n = len(fes._blocks[0][2])
    u = gb.random_state(fes.n_dof, 0.1, 6)
    bc.chunk_invariant(lambda c: fes.assemble_internal_force(u, mat, thickness=t, vectorized=True, chunk_size=c), n)
    bc.chunk_invariant(lambda c: fes.assemble_tangent_stiffness(u, mat, thickness=t, vectorized=True, chunk_size=c), n)


def test_geometry_cache_is_reused_across_calls_and_follows_the_mesh():
    fes, mat, t = gb.quad_case()
    u = gb.random_state(fes.n_dof, 0.05, 7)
    fes.assemble_internal_force(u, mat, thickness=t, vectorized=True)
    fes.assemble_tangent_stiffness(u, mat, thickness=t, vectorized=True)
    assert fes._geometry_cache.builds == 1
    f_before = fes.assemble_internal_force(u, mat, thickness=t, vectorized=True)
    fes.mesh.nodes = fes.mesh.nodes * 1.1                                # moving the mesh must not leave a stale cache
    f_after = fes.assemble_internal_force(u, mat, thickness=t, vectorized=True)
    assert fes._geometry_cache.builds == 2
    bc.equal_to_loop(f_after, fes.assemble_internal_force(u, mat, thickness=t, vectorized=False))
    assert bc.rel_err(f_after, f_before) > 1e-3


# ----------------------------------------------------------------------------- fallbacks
class Quad4TLLoopOnly(Quad4PlaneStressTL):
    supports_batch = False


def _two_block_system(second=Quad4TLLoopOnly):
    nodes, elems, nx, ny = gb.quad_plate(8, 3)
    half = len(elems) // 2
    blocks = {"a": elems[:half], "b": elems[half:]}
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks=blocks, dim=2),
                   {"a": Quad4PlaneStressTL(), "b": second()}, sparse=False)
    return fes, {"a": D, "b": D}


def test_mixed_blocks_use_batch_where_available_and_loop_elsewhere():
    fes, mat = _two_block_system()
    u = gb.random_state(fes.n_dof, 0.1, 8)
    f_ref = fes.assemble_internal_force(u, mat, thickness=0.5)
    K_ref = fes.assemble_tangent_stiffness(u, mat, thickness=0.5)
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, thickness=0.5, vectorized=True), f_ref)
    assert fes.last_assembly_modes["a"] == "batched"
    assert fes.last_assembly_modes["b"] == "loop (Quad4TLLoopOnly does not declare supports_batch)"
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=True), K_ref)
    assert fes.last_assembly_modes["a"] == "batched" and fes.last_assembly_modes["b"].startswith("loop")


def test_path_dependent_state_forces_the_loop():
    fes, mat, t = gb.quad_case()
    u = gb.random_state(fes.n_dof, 0.05, 9)
    ref = fes.assemble_internal_force(u, mat, thickness=t)
    fes.state = {"q": [object()] * len(fes._blocks[0][2])}               # any non-None entry marks a stateful element
    out = fes.assemble_internal_force(u, mat, thickness=t, vectorized=True)
    assert fes.last_assembly_modes["q"] == "loop (path-dependent element state in use)"
    bc.equal_to_loop(out, ref)


def test_contact_elements_and_springs_still_added_in_a_vectorized_assembly():
    fes, mat, t = gb.quad_case()
    fes.add_spring([5], 1, 50.0)
    fes.add_contact_element(GapContactPenalty(), [30], (1.0e4, 0.0, np.array([0.0, 1.0])))
    u = gb.random_state(fes.n_dof, 0.1, 11)
    u[2 * 30 + 1] = 0.05                                                 # penetrate the gap so the contact is active
    f_loop = fes.assemble_internal_force(u, mat, thickness=t)
    K_loop = fes.assemble_tangent_stiffness(u, mat, thickness=t)
    f_vec = fes.assemble_internal_force(u, mat, thickness=t, vectorized=True)
    K_vec = fes.assemble_tangent_stiffness(u, mat, thickness=t, vectorized=True)
    bc.equal_to_loop(f_vec, f_loop)
    bc.equal_to_loop(K_vec, K_loop)
    plain = gb.quad_case()[0].assemble_internal_force(u, mat, thickness=t)
    assert bc.rel_err(f_vec, plain) > 1e-3                               # the spring/contact terms are really present


# ----------------------------------------------------------------------------- Newton path
@pytest.mark.skipif(not GOLDEN, reason="golden file not generated")
def test_newton_path_with_system_switch(golden):
    fes, mat, t = gb.quad_case()
    fes.batched = True
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=t)
    assert fes.last_assembly_modes == {"q": "batched"}
    assert lf[-1] == pytest.approx(1.0)
    bc.same_newton_path(np.asarray(U), golden["quad_newton_U"])


@pytest.mark.skipif(not GOLDEN, reason="golden file not generated")
def test_newton_path_with_vectorized_forwarded_by_the_driver(golden):
    fes, mat, t = gb.quad_case()
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=t, vectorized=True)
    assert fes.last_assembly_modes == {"q": "batched"}
    bc.same_newton_path(np.asarray(U), golden["quad_newton_U"])

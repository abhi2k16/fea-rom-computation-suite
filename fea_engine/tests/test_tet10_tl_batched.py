# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_tet10_tl_batched.py -- batched Tet10SolidTL kernels and their FESystem integration.

Oracles: the element's own per-point code (``internal_force``, analytic ``tangent_stiffness``), its complex-step tangent
(``_tangent_stiffness_complex_step``), and the stored golden arrays of M0 (``golden_loop_baselines``). Evidence:
  element level   batched force and tangent equal the per-point ones to 1e-12 relative on stacks of distorted elements with
                  curved edges (displacement scales 0.02 and 0.3), with and without a precomputed geometry argument, Material or
                  6 x 6 matrix; the tangent equals the complex-step tangent; the force is objective; the tangent is symmetric;
  geometry        the cache holds the element's OWN 4-point rule (JxW has 4 points, summing to the element volume), not the 8-point
                  simplex rule of MeshTransformation; an element without ``own_batch_geometry`` is unchanged;
  system level    vectorized=True assembly (dense and CSR) equals the loop and the golden arrays, is chunk invariant, falls back for a
                  loop-only block, and gives the same Newton path (golden cantilever, tol 1e-5).
Speed is not asserted; tests/batch_timing_tet10.py reports it.
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
from fea_engine import FESystem, Material, Mesh, MultiBlockMesh
from fea_engine import batched_kernels as bk
from fea_engine.elements import Tet10SolidTL, Hex8SolidTL
from fea_engine.material import D_solid3d

MAT = Material(E=gb.E_MOD, nu=gb.NU)
DMAT = D_solid3d(MAT)
EL = Tet10SolidTL()


def _stack(n=6, amp0=0.02):
    """Tet10 elements from the structured box, every node (mid-edge nodes too) displaced: curved edges, different shapes."""
    nodes, elems = gb.tet10_box(2, 1, 1)
    X = nodes[elems][:n].copy()
    for k in range(len(X)):
        X[k] += np.random.default_rng(10 + k).normal(size=X[k].shape) * (amp0 + 0.01 * k)
    return X


def _loop(what, X, u, mat=MAT):
    fn = EL.internal_force if what == "force" else EL.tangent_stiffness
    return np.array([fn(x, w, mat) for x, w in zip(X, u)])


@pytest.mark.parametrize("scale", [0.02, 0.3])
def test_batch_equals_the_per_point_code(scale):
    X = _stack()
    u = np.random.default_rng(1).normal(size=(len(X), 30)) * scale
    bc.equal_to_loop(EL.internal_force_batch(X, u, MAT), _loop("force", X, u), label="force")
    bc.equal_to_loop(EL.tangent_stiffness_batch(X, u, MAT), _loop("tangent", X, u), label="tangent")


def test_batched_tangent_equals_the_complex_step_reference():
    X = _stack(3)
    u = np.random.default_rng(2).normal(size=(3, 30)) * 0.2
    ref = np.array([EL._tangent_stiffness_complex_step(x, w, MAT) for x, w in zip(X, u)])
    bc.equal_to_loop(EL.tangent_stiffness_batch(X, u, MAT), ref, rtol=1e-9, label="tangent vs complex step")


def test_geometry_argument_and_matrix_input_and_single_element():
    X = _stack(4)
    u = np.random.default_rng(3).normal(size=(4, 30)) * 0.1
    geo = EL._reference_geometry_batch(X)
    assert geo[0].shape == (4, 4, 10, 3) and geo[1].shape == (4, 4)
    ref = EL.internal_force_batch(X, u, MAT)
    bc.equal_to_loop(EL.internal_force_batch(X, u, MAT, geometry=geo), ref)
    bc.equal_to_loop(EL.internal_force_batch(X, u, DMAT), ref)
    bc.equal_to_loop(EL.tangent_stiffness_batch(X, u, DMAT, geometry=geo), EL.tangent_stiffness_batch(X, u, MAT))
    assert EL.internal_force_batch(X[:1], u[:1], MAT).shape == (1, 30)
    bc.equal_to_loop(EL.internal_force_batch(X[:1], u[:1], MAT)[0], EL.internal_force(X[0], u[0], MAT))
    with pytest.raises(ValueError, match="6 x 6"):
        EL.internal_force_batch(X, u, np.eye(3))


def test_objective_symmetric_and_derivative_of_the_force():
    X = _stack(1)[0]
    f = lambda w: EL.internal_force_batch(X[None], w.reshape(1, 30), MAT)[0]
    K = lambda w: EL.tangent_stiffness_batch(X[None], w.reshape(1, 30), MAT)[0]
    scale = float(np.max(np.abs(f(0.01 * (X.ravel() - np.tile(X.mean(axis=0), 10))))))
    bc.objective(f, X, scale, theta=0.7)
    bc.fd_tangent(f, K, np.random.default_rng(4).normal(size=30) * 0.1)


def test_protocol_and_memory_estimate():
    assert bk.supports_batch(EL) and bk.batch_unavailable_reason(EL) is None
    assert EL.batch_bytes_per_element("tangent") == bk.default_bytes_per_element(4, 30, 3, "tangent")
    assert EL.batch_bytes_per_element("force") < EL.batch_bytes_per_element("tangent")
    assert bk.bytes_per_element(EL, "tangent") == EL.batch_bytes_per_element("tangent")


# ----------------------------------------------------------------------------- geometry cache
def _box_system(nx=2, ny=1, nz=1):
    nodes, elems = gb.tet10_box(nx, ny, nz)
    return FESystem(Mesh(nodes, elems, 3), Tet10SolidTL(), sparse=False), nodes, elems


def test_cache_uses_the_elements_own_four_point_rule():
    fes, nodes, elems = _box_system()
    geo = fes.block_geometry(None)
    assert geo.JxW.shape == (len(elems), 4) and geo.shape_grad.shape == (len(elems), 4, 10, 3)
    vol = np.abs(np.linalg.det(np.stack([np.stack([nodes[e[1]] - nodes[e[0]], nodes[e[2]] - nodes[e[0]],
                                                    nodes[e[3]] - nodes[e[0]]]) for e in elems]))) / 6.0   # straight-sided tets
    assert np.allclose(geo.JxW.sum(axis=1), vol, rtol=1e-12)           # the weights integrate the element volume
    assert geo.JxW.sum() == pytest.approx(2.0 * 1.0 * 1.0, rel=1e-12)  # the 2 x 1 x 1 box
    u = np.random.default_rng(0).normal(size=fes.n_dof) * 0.05
    fes.assemble_internal_force(u, MAT, vectorized=True)
    fes.assemble_tangent_stiffness(u, MAT, vectorized=True, sparse=True)
    assert fes._geometry_cache.builds == 1 and fes._pattern_cache.builds == 1


def test_elements_without_the_flag_still_use_mesh_transformation():
    assert not getattr(Hex8SolidTL, "own_batch_geometry", False)
    assert Tet10SolidTL.own_batch_geometry is True


# ----------------------------------------------------------------------------- system level
@pytest.fixture(scope="module")
def golden():
    return np.load(gb.GOLDEN_PATH)


@pytest.mark.parametrize("k", range(len(gb.TET_SCALES)))
def test_vectorized_assembly_equals_loop_and_golden(k, golden):
    fes, mat = gb.tet_case()
    u = golden[f"tet_u{k}"]
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, vectorized=True), golden[f"tet_f{k}"], label="force vs golden")
    assert fes.last_assembly_modes == {None: "batched"}
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, vectorized=True), golden[f"tet_K{k}"], label="tangent vs golden")
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True), golden[f"tet_K{k}"], label="CSR vs golden")
    bc.equal_to_loop(fes.assemble_internal_force(u, mat), golden[f"tet_f{k}"], label="loop vs golden")


def test_system_assembly_is_chunk_invariant():
    fes, nodes, elems = _box_system(2, 2, 1)
    u = np.random.default_rng(6).normal(size=fes.n_dof) * 0.05
    n_el = len(elems)
    bc.chunk_invariant(lambda c: fes.assemble_internal_force(u, MAT, vectorized=True, chunk_size=c), n_el)
    bc.chunk_invariant(lambda c: fes.assemble_tangent_stiffness(u, MAT, vectorized=True, chunk_size=c), n_el)
    bc.chunk_invariant(lambda c: fes.assemble_tangent_stiffness(u, MAT, vectorized=True, sparse=True, chunk_size=c), n_el)
    bc.equal_to_loop(fes.assemble_internal_force(u, MAT, vectorized=True), fes.assemble_internal_force(u, MAT))


class _LoopOnlyTet(Tet10SolidTL):
    supports_batch = False


def test_mixed_batched_and_loop_blocks():
    nodes, elems = gb.tet10_box(2, 1, 1)
    h = len(elems) // 2
    mb = MultiBlockMesh(nodes=nodes, blocks={"a": elems[:h], "b": elems[h:]}, dim=3)
    fes = FESystem(mb, {"a": Tet10SolidTL(), "b": _LoopOnlyTet()}, sparse=False)
    mat = {"a": MAT, "b": MAT}
    u = np.random.default_rng(8).normal(size=fes.n_dof) * 0.05
    f0, K0 = fes.assemble_internal_force(u, mat), fes.assemble_tangent_stiffness(u, mat)
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, vectorized=True), f0)
    assert fes.last_assembly_modes["a"] == "batched" and fes.last_assembly_modes["b"].startswith("loop")
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True), K0)


def test_element_response_hooks_work_for_tet10():
    fes, mat = gb.tet_case()
    u = np.random.default_rng(9).normal(size=fes.n_dof) * 0.05
    F = fes.element_forces(u, mat, vectorized=True)
    assert fes.last_assembly_modes == {None: "batched"}
    bc.equal_to_loop(bk.scatter_add_vector(fes.n_dof, fes.element_dofs(), F), fes.assemble_internal_force(u, mat))
    sel = np.array([5, 0, 3])
    bc.equal_to_loop(fes.element_tangents(u, mat, elements=sel, vectorized=True), fes.element_tangents(u, mat, elements=sel))


def test_newton_path_equals_the_golden_path(golden):
    """The golden cantilever of M0 (4 load steps, tol 1e-5) through the batched dense and the batched sparse paths."""
    for kw in (dict(vectorized=True), dict(vectorized=True, sparse_tangent=True)):
        fes, mat = gb.tet_case()
        lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-5, max_iter=40, **kw)
        assert fes.last_assembly_modes == {None: "batched"}
        bc.same_newton_path(np.asarray(U), golden["tet_newton_U"], rtol=1e-6)

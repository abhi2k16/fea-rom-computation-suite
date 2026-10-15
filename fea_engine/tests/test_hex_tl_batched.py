# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_hex_tl_batched.py -- batched Hex8SolidTL / Hex20SolidTL kernels and their FESystem integration.

The per-Gauss-point reference elements (test_hex_tl.py) are the oracle. Evidence:
  element level   internal_force_batch / tangent_stiffness_batch equal the reference to 1e-12 relative for 1, 2 and 3
                  points per direction, on stacks of differently distorted (Hex20: curved-edge) elements, with and without
                  a precomputed geometry argument; the batched force is objective, the batched tangent equals central
                  differences of the batched force and is symmetric;
  system level    vectorized=True assembly of a Hex8 and a Hex20 mesh equals the loop assembly (dense and CSR), is chunk
                  invariant, reports "batched", falls back to the loop for a loop-only block, and the geometry cache holds
                  one entry per block with the element's own Gauss rule;
  Newton path     a 3-D cantilever through solve_nonlinear_static gives the same displacements with the batched path,
                  with the batched sparse path, and with the reference loop (1e-9), for both elements.
Speed is not asserted; tests/batch_timing_hex.py reports it.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_conformance as bc
import hex_meshes
import test_hex_tl as T
import fea_engine.nonlinear_solver as nls
from fea_engine import FESystem, MultiBlockMesh
from fea_engine import batched_kernels as bk
from fea_engine import mesh as meshlib
from fea_engine.elements import Hex8SolidTL, Hex20SolidTL

CLASSES = [Hex8SolidTL, Hex20SolidTL]
MESH_DIMS = {Hex8SolidTL: (4, 2, 2), Hex20SolidTL: (3, 1, 1)}


def _stack(cls, n_el=6):
    return np.stack([T.distorted(cls, amp=0.02 + 0.012 * k) for k in range(n_el)])


def _loop(el, X, u, what):
    fn = el.internal_force if what == "force" else el.tangent_stiffness
    return np.array([fn(x, w, T.D) for x, w in zip(X, u)])


# ----------------------------------------------------------------------------- element level
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("scale", [0.02, 0.3])
def test_batch_equals_reference(cls, order, scale):
    el = cls(gauss_order=order)
    X = _stack(cls)
    u = np.random.default_rng(order).normal(size=(len(X), 3 * el.n_nodes)) * scale
    bc.equal_to_loop(el.internal_force_batch(X, u, T.D), _loop(el, X, u, "force"), label="force")
    bc.equal_to_loop(el.tangent_stiffness_batch(X, u, T.D), _loop(el, X, u, "tangent"), label="tangent")


@pytest.mark.parametrize("cls", CLASSES)
def test_batch_with_precomputed_geometry_and_material_input(cls):
    el = cls()
    X = _stack(cls, 4)
    u = np.random.default_rng(1).normal(size=(4, 3 * el.n_nodes)) * 0.1
    geo = el._reference_geometry_batch(X)
    assert geo[0].shape == (4, el.gauss_order ** 3, el.n_nodes, 3) and geo[1].shape == (4, el.gauss_order ** 3)
    ref = el.internal_force_batch(X, u, T.D)
    bc.equal_to_loop(el.internal_force_batch(X, u, T.D, geometry=geo), ref)
    bc.equal_to_loop(el.internal_force_batch(X, u, T.MAT), ref)
    bc.equal_to_loop(el.tangent_stiffness_batch(X, u, T.MAT, geometry=geo), el.tangent_stiffness_batch(X, u, T.D))


@pytest.mark.parametrize("cls", CLASSES)
def test_batched_force_is_objective_and_tangent_is_its_derivative(cls):
    el = cls()
    X = T.distorted(cls)
    n = 3 * el.n_nodes
    f = lambda w: el.internal_force_batch(X[None], w.reshape(1, n), T.D)[0]
    K = lambda w: el.tangent_stiffness_batch(X[None], w.reshape(1, n), T.D)[0]
    scale = float(np.max(np.abs(f(0.01 * (X.ravel() - np.tile(X.mean(axis=0), el.n_nodes))))))
    bc.objective(f, X, scale, theta=0.7)
    bc.fd_tangent(f, K, np.random.default_rng(2).normal(size=n) * 0.1)


@pytest.mark.parametrize("cls", CLASSES)
def test_single_element_batch_and_protocol(cls):
    el = cls()
    assert bk.supports_batch(el) and bk.batch_unavailable_reason(el) is None
    X = T.distorted(cls)[None]
    u = np.random.default_rng(3).normal(size=(1, 3 * el.n_nodes)) * 0.1
    assert el.internal_force_batch(X, u, T.D).shape == (1, 3 * el.n_nodes)
    assert el.tangent_stiffness_batch(X, u, T.D).shape == (1, 3 * el.n_nodes, 3 * el.n_nodes)
    bc.equal_to_loop(el.internal_force_batch(X, u, T.D)[0], el.internal_force(X[0], u[0], T.D))


def test_memory_estimate_follows_the_gauss_rule_and_node_count():
    h20 = Hex20SolidTL()                                       # 27 points, 60 DOFs
    assert h20.batch_bytes_per_element("tangent") == bk.default_bytes_per_element(27, 60, 3, "tangent")
    assert h20.batch_bytes_per_element("force") < h20.batch_bytes_per_element("tangent")
    assert Hex20SolidTL(gauss_order=2).batch_bytes_per_element("tangent") < h20.batch_bytes_per_element("tangent")
    assert bk.bytes_per_element(h20, "tangent") == h20.batch_bytes_per_element("tangent")   # uses the element's own
    assert h20.batch_bytes_per_element("tangent") > 77760 + 28800                           # B array + element matrix


# ----------------------------------------------------------------------------- system level
def _system(cls, second=None):
    dims = MESH_DIMS[cls]
    m = T._box(cls, *dims)
    return FESystem(m, cls(), sparse=False), m


@pytest.mark.parametrize("cls", CLASSES)
def test_vectorized_system_assembly_equals_loop_dense_and_sparse(cls):
    fes, m = _system(cls)
    u = np.random.default_rng(5).normal(size=fes.n_dof) * 0.05
    f0 = fes.assemble_internal_force(u, T.D)
    assert fes.last_assembly_modes[None].startswith("loop")
    K0 = fes.assemble_tangent_stiffness(u, T.D)
    bc.equal_to_loop(fes.assemble_internal_force(u, T.D, vectorized=True), f0, label="force")
    assert fes.last_assembly_modes == {None: "batched"}
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, T.D, vectorized=True), K0, label="tangent")
    Ks = fes.assemble_tangent_stiffness(u, T.D, vectorized=True, sparse=True)
    bc.equal_to_loop(Ks, K0, label="sparse tangent")
    assert Ks.nnz < K0.size


@pytest.mark.parametrize("cls", CLASSES)
def test_system_assembly_is_chunk_invariant_and_cache_uses_the_elements_rule(cls):
    fes, m = _system(cls)
    n_el = len(m.elements)
    u = np.random.default_rng(6).normal(size=fes.n_dof) * 0.05
    bc.chunk_invariant(lambda c: fes.assemble_internal_force(u, T.D, vectorized=True, chunk_size=c), n_el)
    bc.chunk_invariant(lambda c: fes.assemble_tangent_stiffness(u, T.D, vectorized=True, chunk_size=c), n_el)
    bc.chunk_invariant(lambda c: fes.assemble_tangent_stiffness(u, T.D, vectorized=True, sparse=True, chunk_size=c), n_el)
    geo = fes.block_geometry(None)
    q = fes._blocks[0][1].gauss_order ** 3
    assert geo.shape_grad.shape == (n_el, q, fes._blocks[0][1].n_nodes, 3) and geo.JxW.shape == (n_el, q)
    assert fes._geometry_cache.builds == 1 and fes._pattern_cache.builds == 1


def test_hex20_second_order_rule_gives_a_separate_cache_entry_and_still_matches():
    m = T._box(Hex20SolidTL, 3, 1, 1)
    fes = FESystem(m, Hex20SolidTL(gauss_order=2), sparse=False)
    u = np.random.default_rng(7).normal(size=fes.n_dof) * 0.05
    bc.equal_to_loop(fes.assemble_internal_force(u, T.D, vectorized=True), fes.assemble_internal_force(u, T.D))
    assert fes.block_geometry(None).JxW.shape[1] == 8


class _Hex8LoopOnly(Hex8SolidTL):
    supports_batch = False


def test_mixed_blocks_hex_batched_and_hex_loop():
    m8 = meshlib.box_mesh(4.0, 1.0, 1.0, 4, 1, 1)
    h = len(m8.elements) // 2
    mb = MultiBlockMesh(nodes=m8.nodes, blocks={"a": m8.elements[:h], "b": m8.elements[h:]}, dim=3)
    fes = FESystem(mb, {"a": Hex8SolidTL(), "b": _Hex8LoopOnly()}, sparse=False)
    mat = {"a": T.D, "b": T.D}
    u = np.random.default_rng(8).normal(size=fes.n_dof) * 0.05
    f0, K0 = fes.assemble_internal_force(u, mat), fes.assemble_tangent_stiffness(u, mat)
    bc.equal_to_loop(fes.assemble_internal_force(u, mat, vectorized=True), f0)
    assert fes.last_assembly_modes["a"] == "batched" and fes.last_assembly_modes["b"].startswith("loop")
    bc.equal_to_loop(fes.assemble_tangent_stiffness(u, mat, vectorized=True, sparse=True), K0)


# ----------------------------------------------------------------------------- Newton path
def _cantilever_fes(cls, nx, ny, nz, load):
    m = T._box(cls, nx, ny, nz)
    fes = FESystem(m, cls(), sparse=False)
    fes.fix_dofs(np.where(np.abs(m.nodes[:, 0]) < 1e-9)[0], [0, 1, 2])
    fes.add_nodal_force(np.where(np.abs(m.nodes[:, 0] - 4.0) < 1e-9)[0], 1, -load)
    return fes


@pytest.mark.parametrize("cls,dims", [(Hex8SolidTL, (4, 1, 1)), (Hex20SolidTL, (3, 1, 1))])
def test_newton_path_loop_batched_and_batched_sparse_agree(cls, dims):
    load = 8.0
    runs = {}
    for name, kw in (("loop", {}), ("batched", dict(vectorized=True)),
                     ("sparse", dict(vectorized=True, sparse_tangent=True))):
        fes = _cantilever_fes(cls, *dims, load)
        lf, U = nls.solve_nonlinear_static(fes, T.D, n_steps=6, tol=1e-9, max_iter=40, **kw)
        runs[name] = np.asarray(U)
        if name != "loop":
            assert fes.last_assembly_modes == {None: "batched"}
    assert np.max(np.abs(runs["loop"][-1])) > 0.5                          # a genuinely large deflection
    bc.same_newton_path(runs["batched"], runs["loop"])
    bc.same_newton_path(runs["sparse"], runs["loop"])

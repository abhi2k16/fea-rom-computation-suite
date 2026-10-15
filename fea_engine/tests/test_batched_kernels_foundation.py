# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_batched_kernels_foundation.py -- the batch-method protocol, chunking, DOF map/scatter and the per-block geometry
cache (fea_engine.batched_kernels, FESystem.block_geometry / batch_support_report).

No element implements the batch methods yet, so a small test element (a batch version of a plane-stress Quad4 that
simply calls the per-element code row by row) exercises the protocol and the conformance helpers in
batch_conformance.py. Evidence: chunk slices tile the range exactly; chunked evaluation agrees with unchunked to
1e-12; the DOF map equals FESystem._global_dofs; bincount scatter equals the loop's fancy-index scatter; the cached
geometry equals a per-element recomputation (dN/dX, det(J) * w from the element's own shape_and_derivs); the cache
is rebuilt after the nodes, connectivity or Gauss order change and not otherwise; the byte estimate reproduces the
sizes measured in the planning study.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_conformance as bc
import golden_loop_baselines as gb
from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL, Quad4PlaneStress, TrussTL2D
from fea_engine import batched_kernels as bk


class Quad4TLRowwiseBatch(Quad4PlaneStressTL):
    """Test element: satisfies the batch protocol by calling the per-element methods row by row."""
    supports_batch = True

    def internal_force_batch(self, coords, u, mat, thickness=1.0, **kw):
        return np.array([self.internal_force(c, w, mat, thickness, **kw) for c, w in zip(coords, u)])

    def tangent_stiffness_batch(self, coords, u, mat, thickness=1.0, **kw):
        return np.array([self.tangent_stiffness(c, w, mat, thickness, **kw) for c, w in zip(coords, u)])


D = D_plane_stress(Material(E=gb.E_MOD, nu=gb.NU))


def _system(el=None):
    nodes, elems, nx, ny = gb.quad_plate(6, 3)
    el = el or Quad4PlaneStressTL()
    return FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": el}, sparse=False), nodes, elems


# ----------------------------------------------------------------------------- protocol
def test_protocol_detection():
    assert not bk.supports_batch(Quad4PlaneStress())
    assert bk.supports_batch(Quad4TLRowwiseBatch())
    assert "does not declare" in bk.batch_unavailable_reason(Quad4PlaneStress())

    class Half(Quad4PlaneStress):
        supports_batch = True
    assert "lacks" in bk.batch_unavailable_reason(Half())
    assert bk.batch_unavailable_reason(Quad4TLRowwiseBatch()) is None


def test_batch_support_report_names_each_block_and_reason():
    fes, _, elems = _system(Quad4PlaneStress())
    rep = fes.batch_support_report()
    assert rep == [("q", "Quad4PlaneStress", len(elems), bk.batch_unavailable_reason(Quad4PlaneStress()))]
    assert rep[0][3] is not None
    fes2, _, _ = _system(Quad4TLRowwiseBatch())
    assert fes2.batch_support_report()[0][3] is None
    assert _system()[0].batch_support_report()[0][3] is None                       # Quad4PlaneStressTL is batched


# ----------------------------------------------------------------------------- chunking
@pytest.mark.parametrize("n,c", [(10, 1), (10, 3), (10, 10), (10, 99), (1, 5)])
def test_chunk_slices_tile_the_range(n, c):
    sl = bk.chunk_slices(n, chunk_size=c)
    idx = np.concatenate([np.arange(n)[s] for s in sl])
    assert np.array_equal(idx, np.arange(n))
    assert all(s.stop - s.start <= c for s in sl)


def test_chunk_slices_budget_and_edges():
    assert bk.chunk_slices(0, 3) == []
    assert len(bk.chunk_slices(7)) == 1                                        # no information -> one slice
    assert bk.chunk_size_for(1000, budget_bytes=10_000) == 10
    assert bk.chunk_size_for(10 ** 9, budget_bytes=10) == 1                    # never below one element
    sl = bk.chunk_slices(100, bytes_per_elem=1000, budget_bytes=10_000)
    assert len(sl) == 10 and sl[0] == slice(0, 10)
    with pytest.raises(ValueError):
        bk.chunk_slices(5, chunk_size=0)
    with pytest.raises(ValueError):
        bk.chunk_size_for(0)


def test_map_chunks_matches_unchunked_for_the_batch_kernel():
    fes, nodes, elems = _system(Quad4TLRowwiseBatch())
    el = fes._blocks[0][1]
    geo = fes.block_geometry("q")
    u = np.random.default_rng(3).normal(size=fes.n_dof) * 0.05

    def forces(chunk_size):
        def one(s):
            return el.internal_force_batch(nodes[elems[s]], u[geo.dofs[s]], D, 0.5)
        return bk.map_chunks(one, len(elems), chunk_size)
    res = bc.chunk_invariant(forces, len(elems))
    assert set(res) == {"chunk_1", f"chunk_{len(elems) // 3}", f"chunk_{len(elems) + 7}"}


def test_byte_estimate_matches_planning_measurements():
    # Hex20, 3x3x3 rule: B 77.8 kB, element tangent 28.8 kB (measured in the planning study)
    g, ndof, dim = 27, 60, 3
    force = bk.default_bytes_per_element(g, ndof, dim, "force")
    tang = bk.default_bytes_per_element(g, ndof, dim, "tangent")
    assert 8 * g * 6 * ndof == 77760 and 8 * ndof * ndof == 28800
    assert force > 77760 and tang > force + 28800          # dominated by those arrays, plus work arrays
    assert tang - force == 8 * (2 * g * 6 * ndof + ndof * ndof)
    with pytest.raises(ValueError):
        bk.default_bytes_per_element(g, ndof, dim, "bogus")


def test_bytes_per_element_uses_the_elements_own_estimate_when_present():
    class Own(Quad4PlaneStressTL):
        def batch_bytes_per_element(self, kind):
            return 1234 if kind == "tangent" else 99
    assert bk.bytes_per_element(Own(), "tangent") == 1234 and bk.bytes_per_element(Own(), "force") == 99
    generic = bk.bytes_per_element(Quad4PlaneStressTL(), "tangent")
    assert generic == bk.default_bytes_per_element(4, 8, 2, "tangent")


# ----------------------------------------------------------------------------- DOF map and scatter
def test_dof_map_equals_fesystem_global_dofs():
    fes, _, elems = _system()
    dm = bk.element_dof_map(elems, fes.npn)
    for e in (0, 5, len(elems) - 1):
        assert np.array_equal(dm[e], fes._global_dofs(elems[e]))


def test_bincount_scatter_equals_fancy_index_loop():
    fes, nodes, elems = _system()
    dm = bk.element_dof_map(elems, fes.npn)
    fe = np.random.default_rng(4).normal(size=dm.shape)
    ref = np.zeros(fes.n_dof)
    for g, f in zip(dm, fe):
        ref[g] += f
    assert bc.rel_err(bk.scatter_add_vector(fes.n_dof, dm, fe), ref) < 1e-14


# ----------------------------------------------------------------------------- geometry cache
def test_cached_geometry_equals_per_element_recomputation():
    fes, nodes, elems = _system()
    geo = fes.block_geometry("q")
    el = fes._blocks[0][1]
    from fea_engine.elements.base import gauss_product
    pts, wts = gauss_product(el.gauss_order, 2)
    assert geo.shape_grad.shape == (len(elems), len(pts), 4, 2) and geo.JxW.shape == (len(elems), len(pts))
    for e in (0, 7, len(elems) - 1):
        X = nodes[elems[e]]
        for q, (p, w) in enumerate(zip(pts, wts)):
            N, dN = el._cached_shape_and_derivs(tuple(p))
            J = dN @ X
            assert np.allclose(geo.shape_grad[e, q], np.linalg.solve(J, dN).T, rtol=1e-12, atol=1e-14)
            assert geo.JxW[e, q] == pytest.approx(np.linalg.det(J) * w, rel=1e-12)
    # the reference area is recovered by summing JxW
    area = geo.JxW.sum()
    x, y = nodes[elems[:, [0, 1, 2, 3]]][:, :, 0], nodes[elems[:, [0, 1, 2, 3]]][:, :, 1]
    shoelace = 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)).sum()
    assert area == pytest.approx(shoelace, rel=1e-12)


def test_cache_reuse_and_invalidation():
    fes, nodes, elems = _system()
    g1 = fes.block_geometry("q")
    assert fes._geometry_cache.builds == 1
    assert fes.block_geometry("q") is g1 and fes._geometry_cache.builds == 1        # reused, not rebuilt
    fes.mesh.nodes = fes.mesh.nodes * 1.5                                           # move the mesh
    g2 = fes.block_geometry("q")
    assert g2 is not g1 and fes._geometry_cache.builds == 2
    assert g2.JxW.sum() == pytest.approx(2.25 * g1.JxW.sum(), rel=1e-12)            # area scales by 1.5**2
    fes.block_geometry("q", gauss_order=3)                                          # different Gauss order -> rebuild
    assert fes._geometry_cache.builds == 3
    assert fes.block_geometry("q", gauss_order=3).JxW.shape[1] == 9
    fes.clear_block_geometry()
    assert len(fes._geometry_cache) == 0 and fes._geometry_cache.nbytes == 0
    with pytest.raises(KeyError):
        fes.block_geometry("nope")


def test_cache_detects_a_changed_connectivity():
    fes, nodes, elems = _system()
    fes.block_geometry("q")
    name, f, conn = fes._blocks[0]
    new = conn.copy()
    new[0] = new[0][[1, 2, 3, 0]]                                                   # same nodes, rotated numbering
    fes._blocks[0] = (name, f, new)
    fes.block_geometry("q")
    assert fes._geometry_cache.builds == 2


def test_unsupported_elements_are_reported_not_hidden():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"t": np.array([[0, 1], [1, 2]])}, dim=2),
                   {"t": TrussTL2D()}, sparse=False)
    assert fes.batch_support_report()[0][3] is not None                              # loop only
    with pytest.raises(ValueError, match="does not support TrussTL2D"):                # out of scope: says so
        fes.block_geometry("t")


def test_single_block_mesh_uses_none_as_the_block_key():
    nodes, elems, _, _ = gb.quad_plate(3, 2)
    from fea_engine import Mesh
    fes = FESystem(Mesh(nodes, elems, 2), Quad4PlaneStress(), sparse=False)
    g = fes.block_geometry(None)
    assert g.n_elements == len(elems) and fes.batch_support_report()[0][0] is None

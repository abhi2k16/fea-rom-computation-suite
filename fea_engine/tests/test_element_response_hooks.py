# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_element_response_hooks.py -- FESystem.element_forces / element_tangents / element_dofs (+ *_local).

These are the hooks hyper-reduction needs: the response of selected elements, one row per element, without
assembling a global vector. Evidence:
  consistency   scattering the rows through element_dofs reproduces assemble_internal_force / assemble_tangent_stiffness
                (no contact, no springs), for the loop route and for the batched route, to 1e-12 relative;
  routes        batched and loop rows agree; the batched route is really taken (last_assembly_modes);
  selection     arbitrary order, repeats and an empty selection give the rows of the full result; the *_local variants
                equal the global-gather ones; element numbers run over the blocks in order;
  mixed         a batched block next to a loop-only block, and a Quad4 block next to a Tri3 block, are handled;
  guards        bad element numbers, wrong U_local shape and unequal DOF counts raise.
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
from fea_engine import batched_kernels as bk
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL, Tri3PlaneStress

D = D_plane_stress(Material(E=gb.E_MOD, nu=gb.NU))
KW = dict(thickness=0.5)


class _LoopOnlyQuad(Quad4PlaneStressTL):
    supports_batch = False


def _plate(nx=6, ny=3, loop_second_half=False):
    nodes, elems, _, _ = gb.quad_plate(nx, ny, L=nx / 2.0, H=ny / 4.0)
    if loop_second_half:
        h = len(elems) // 2
        mesh = MultiBlockMesh(nodes=nodes, blocks={"a": elems[:h], "b": elems[h:]}, dim=2)
        fes = FESystem(mesh, {"a": Quad4PlaneStressTL(), "b": _LoopOnlyQuad()}, sparse=False)
        mat = {"a": D, "b": D}
    else:
        mesh = MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2)
        fes = FESystem(mesh, {"q": Quad4PlaneStressTL()}, sparse=False)
        mat = {"q": D}
    u = np.random.default_rng(1).normal(size=fes.n_dof) * 0.03
    return fes, mat, u, len(elems)


def _scatter_force(fes, F, dofs):
    return bk.scatter_add_vector(fes.n_dof, dofs, F)


def _scatter_tangent(fes, Ke, dofs):
    K = np.zeros((fes.n_dof, fes.n_dof))
    bk.scatter_add_matrix(K, dofs, Ke)
    return K


@pytest.mark.parametrize("vectorized", [False, True])
def test_rows_scatter_to_the_assembled_force_and_tangent(vectorized):
    fes, mat, u, ne = _plate()
    assert fes.n_elements == ne
    dofs = fes.element_dofs()
    assert dofs.shape == (ne, 8)
    F = fes.element_forces(u, mat, vectorized=vectorized, **KW)
    assert fes.last_assembly_modes == {"q": "batched" if vectorized else "loop (vectorized not requested)"}
    bc.equal_to_loop(_scatter_force(fes, F, dofs), fes.assemble_internal_force(u, mat, **KW), label="force")
    Ke = fes.element_tangents(u, mat, vectorized=vectorized, **KW)
    assert Ke.shape == (ne, 8, 8)
    bc.equal_to_loop(_scatter_tangent(fes, Ke, dofs), fes.assemble_tangent_stiffness(u, mat, **KW), label="tangent")


def test_batched_rows_equal_loop_rows_and_chunk_size_does_not_matter():
    fes, mat, u, ne = _plate()
    ref_f = fes.element_forces(u, mat, **KW)
    ref_k = fes.element_tangents(u, mat, **KW)
    for c in (1, 5, ne + 10):
        bc.equal_to_loop(fes.element_forces(u, mat, vectorized=True, chunk_size=c, **KW), ref_f, label=f"force c={c}")
        bc.equal_to_loop(fes.element_tangents(u, mat, vectorized=True, chunk_size=c, **KW), ref_k,
                         label=f"tangent c={c}")
    fes.batched = True                                         # the system-wide switch is honoured
    fes.element_forces(u, mat, **KW)
    assert fes.last_assembly_modes == {"q": "batched"}


def test_selection_order_repeats_and_empty():
    fes, mat, u, ne = _plate()
    full_f = fes.element_forces(u, mat, vectorized=True, **KW)
    full_k = fes.element_tangents(u, mat, vectorized=True, **KW)
    full_d = fes.element_dofs()
    sel = np.array([7, 2, 2, 11, 0])
    for vec in (False, True):
        bc.equal_to_loop(fes.element_forces(u, mat, elements=sel, vectorized=vec, **KW), full_f[sel])
        bc.equal_to_loop(fes.element_tangents(u, mat, elements=sel, vectorized=vec, **KW), full_k[sel])
    assert np.array_equal(fes.element_dofs(sel), full_d[sel])
    empty = fes.element_forces(u, mat, elements=np.array([], dtype=int), vectorized=True, **KW)
    assert empty.shape[0] == 0


@pytest.mark.parametrize("vectorized", [False, True])
def test_local_variants_equal_the_global_gather(vectorized):
    fes, mat, u, ne = _plate()
    sel = np.array([3, 9, 4])
    U = u[fes.element_dofs(sel)]
    bc.equal_to_loop(fes.element_forces_local(sel, U, mat, vectorized=vectorized, **KW),
                     fes.element_forces(u, mat, elements=sel, vectorized=vectorized, **KW))
    bc.equal_to_loop(fes.element_tangents_local(sel, U, mat, vectorized=vectorized, **KW),
                     fes.element_tangents(u, mat, elements=sel, vectorized=vectorized, **KW))


def test_blocks_number_elements_in_order_and_mix_batched_with_loop():
    fes, mat, u, ne = _plate(loop_second_half=True)
    F = fes.element_forces(u, mat, vectorized=True, **KW)
    assert fes.last_assembly_modes["a"] == "batched" and fes.last_assembly_modes["b"].startswith("loop")
    bc.equal_to_loop(_scatter_force(fes, F, fes.element_dofs()), fes.assemble_internal_force(u, mat, **KW))
    # elements chosen across the block boundary, in reverse order
    h = ne // 2
    sel = np.array([h + 1, h - 1, 0, ne - 1])
    bc.equal_to_loop(fes.element_forces(u, mat, elements=sel, vectorized=True, **KW), F[sel])
    Ke = fes.element_tangents(u, mat, vectorized=True, **KW)
    bc.equal_to_loop(_scatter_tangent(fes, Ke, fes.element_dofs()), fes.assemble_tangent_stiffness(u, mat, **KW))


def test_unequal_dof_counts_raise_unless_one_block_is_selected():
    nodes, quads, nx, ny = gb.quad_plate(4, 2, L=2.0, H=0.5)
    keep = quads[:-2]
    last = quads[-2:]                                          # the last column: split each quad into two triangles
    tris = np.array([t for q in last for t in ([q[0], q[1], q[2]], [q[0], q[2], q[3]])])
    mesh = MultiBlockMesh(nodes=nodes, blocks={"q": keep, "t": tris}, dim=2)
    fes = FESystem(mesh, {"q": Quad4PlaneStressTL(), "t": Tri3PlaneStress()}, sparse=False)
    with pytest.raises(ValueError, match="different numbers of DOFs"):
        fes.element_dofs()
    nq = len(keep)
    assert fes.element_dofs(np.arange(nq)).shape == (nq, 8)
    assert fes.element_dofs(np.arange(nq, nq + len(tris))).shape == (len(tris), 6)


def test_argument_guards():
    fes, mat, u, ne = _plate()
    with pytest.raises(IndexError):
        fes.element_forces(u, mat, elements=np.array([ne]), **KW)
    with pytest.raises(IndexError):
        fes.element_forces(u, mat, elements=np.array([-1]), **KW)
    with pytest.raises(ValueError, match="integer"):
        fes.element_forces(u, mat, elements=np.array([0.5]), **KW)
    with pytest.raises(ValueError, match="U_local"):
        fes.element_forces_local(np.array([0, 1]), np.zeros((3, 8)), mat, **KW)

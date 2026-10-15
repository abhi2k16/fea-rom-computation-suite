# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_sparse_newton.py -- sparse tangent assembly (FESystem.assemble_tangent_stiffness(sparse=True)) and the sparse
Newton path of the nonlinear drivers (sparse_tangent=, lu_ordering=, reuse_factorization=).

Evidence: the CSR tangent equals the dense one (batched blocks, loop blocks, mixed blocks, contact elements, springs) to
1e-12; the CSR pattern is built once per block and survives mesh motion but not a connectivity change; sparse and
dense Newton give the same converged displacements (static, displacement control, arc length, transient) to 1e-9;
modified Newton (reuse_factorization) reaches the same solution with fewer tangent assemblies; every lu_ordering is
accepted and gives the same solution; invalid orderings raise; NewtonOptions carries the new fields; and a model whose
dense tangent would need 3.3 GB assembles and solves in a small fraction of that memory (tracemalloc peak).
No speed is asserted; tests/batch_timing.py reports it.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import tracemalloc
import numpy as np
import pytest
import scipy.sparse as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_conformance as bc
import golden_loop_baselines as gb
import fea_engine.nonlinear_solver as nls
from fea_engine import FESystem, Material, MultiBlockMesh, RayleighDamping
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL
from fea_engine.elements.contact import GapContactPenalty
from fea_engine.newton_options import NewtonOptions
from fea_engine import batched_kernels as bk

GOLDEN = os.path.exists(gb.GOLDEN_PATH)
D = D_plane_stress(Material(E=gb.E_MOD, nu=gb.NU))


class LoopOnly(Quad4PlaneStressTL):
    supports_batch = False


def _plate_system(nx=8, ny=3, second=None):
    nodes, elems, nx, ny = gb.quad_plate(nx, ny)
    if second is None:
        fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
        mat = {"q": D}
    else:
        h = len(elems) // 2
        fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"a": elems[:h], "b": elems[h:]}, dim=2),
                       {"a": Quad4PlaneStressTL(), "b": second()}, sparse=False)
        mat = {"a": D, "b": D}
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    fes.add_nodal_force(right, 1, -40.0 / (ny + 1))
    return fes, mat, right


# ----------------------------------------------------------------------------- sparse tangent
@pytest.mark.parametrize("vectorized", [False, True])
def test_sparse_tangent_equals_dense(vectorized):
    fes, mat, _ = _plate_system()
    u = gb.random_state(fes.n_dof, 0.1, 1)
    Kd = fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=vectorized)
    Ks = fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=vectorized, sparse=True)
    assert sps.isspmatrix_csr(Ks) and Ks.shape == Kd.shape
    bc.equal_to_loop(Ks, Kd, label="sparse tangent")
    assert abs(Ks - Ks.T).max() < 1e-10 * abs(Kd).max()
    assert Ks.nnz < 0.2 * Kd.size                                     # genuinely sparse
    assert fes.last_assembly_modes["q"] == ("batched" if vectorized else "loop (vectorized not requested)")


def test_sparse_tangent_mixed_blocks_contact_and_springs():
    fes, mat, _ = _plate_system(second=LoopOnly)
    fes.add_spring([5], 1, 50.0)
    fes.add_contact_element(GapContactPenalty(), [30], (1.0e4, 0.0, np.array([0.0, 1.0])))
    u = gb.random_state(fes.n_dof, 0.1, 2)
    u[2 * 30 + 1] = 0.05
    Kd = fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=True)
    Ks = fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=True, sparse=True)
    bc.equal_to_loop(Ks, Kd, label="sparse tangent, mixed")
    assert fes.last_assembly_modes["a"] == "batched" and fes.last_assembly_modes["b"].startswith("loop")


def test_default_tangent_is_still_a_dense_array():
    fes, mat, _ = _plate_system()
    K = fes.assemble_tangent_stiffness(np.zeros(fes.n_dof), mat, thickness=0.5)
    assert isinstance(K, np.ndarray) and K.shape == (fes.n_dof, fes.n_dof)


def test_pattern_is_cached_survives_mesh_motion_and_follows_connectivity():
    fes, mat, _ = _plate_system()
    u = gb.random_state(fes.n_dof, 0.05, 3)
    for _ in range(3):
        fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=True, sparse=True)
    assert fes._pattern_cache.builds == 1
    fes.mesh.nodes = fes.mesh.nodes * 1.2                             # geometry changes, pattern does not
    fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=True, sparse=True)
    assert fes._pattern_cache.builds == 1
    name, f, conn = fes._blocks[0]
    new = conn.copy()
    new[0], new[1] = conn[1].copy(), conn[0].copy()                   # different element-to-DOF map
    fes._blocks[0] = (name, f, new)
    fes.assemble_tangent_stiffness(u, mat, thickness=0.5, vectorized=True, sparse=True)
    assert fes._pattern_cache.builds == 2


def test_pattern_positions_reproduce_a_scipy_coo_sum():
    nodes, elems, _, _ = gb.quad_plate(5, 4)
    dofs = bk.element_dof_map(elems, 2)
    n = 2 * len(nodes)
    Ke = np.random.default_rng(0).normal(size=(len(elems), 8, 8))
    pat = bk.build_sparse_pattern(dofs, n)
    data = np.zeros(pat.nnz)
    bk.scatter_add_csr_data(data, pat, slice(0, len(elems)), Ke)
    ref = sps.coo_matrix((Ke.ravel(), ((dofs[:, :, None] + 0 * dofs[:, None, :]).ravel(),
                                         (dofs[:, None, :] + 0 * dofs[:, :, None]).ravel())), shape=(n, n)).tocsr()
    ours = sps.csr_matrix((data, pat.indices, pat.indptr), shape=(n, n))
    assert abs(ours - ref).max() < 1e-12
    assert np.all(np.diff(pat.indptr) >= 0) and pat.indptr[-1] == pat.nnz
    # chunked accumulation equals one pass
    data2 = np.zeros(pat.nnz)
    for s in bk.chunk_slices(len(elems), 7):
        bk.scatter_add_csr_data(data2, pat, s, Ke[s])
    assert bc.rel_err(data2, data) < 1e-12


# ----------------------------------------------------------------------------- sparse Newton
@pytest.mark.skipif(not GOLDEN, reason="golden file not generated")
@pytest.mark.parametrize("kw", [dict(sparse_tangent=True), dict(sparse_tangent=True, vectorized=True)])
def test_static_sparse_matches_golden_loop_path(kw):
    golden = np.load(gb.GOLDEN_PATH)
    fes, mat, t = gb.quad_case()
    lf, U = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-9, max_iter=40, thickness=t, **kw)
    bc.same_newton_path(np.asarray(U), golden["quad_newton_U"])


def _cantilever_case():
    fes, mat, right = _plate_system()
    return fes, mat, right


def test_displacement_control_sparse_matches_dense():
    fes, mat, right = _cantilever_case()
    cdof = 2 * right[-1] + 1
    fes.F[:] = 0.0
    targets = np.linspace(-0.4, -1.6, 4)
    ud, rd = nls.solve_nonlinear_displacement_control(fes, mat, cdof, targets, tol=1e-9, thickness=0.5)
    us, rs = nls.solve_nonlinear_displacement_control(fes, mat, cdof, targets, tol=1e-9, thickness=0.5,
                                                      sparse_tangent=True, vectorized=True)
    bc.same_newton_path(us, ud)
    assert bc.rel_err(rs, rd) < 1e-8


def test_arc_length_sparse_matches_dense():
    fes, mat, _ = _cantilever_case()
    ld, Ud = nls.solve_nonlinear_arc_length(fes, mat, 0.6, n_steps=4, tol=1e-9, thickness=0.5)[:2]
    ls, Us = nls.solve_nonlinear_arc_length(fes, mat, 0.6, n_steps=4, tol=1e-9, thickness=0.5,
                                            sparse_tangent=True, vectorized=True)[:2]
    bc.same_newton_path(np.asarray(Us), np.asarray(Ud))
    assert bc.rel_err(np.asarray(ls), np.asarray(ld)) < 1e-8


def test_transient_sparse_matches_dense():
    fes, mat, right = _cantilever_case()
    fes.F[:] = 0.0
    fes.assemble_mass(7800.0 * np.eye(2), thickness=0.5)
    fes.assemble_damping(RayleighDamping.calibrate(2 * np.pi * 5.0, 2 * np.pi * 30.0, 0.02))
    load = TimeHistoryLoad(LoadPattern(node_ids=right, dof_index=1), time_fn=lambda tt: -5.0)
    Ud = nls.solve_nonlinear_transient(fes, mat, load, 0.05, 0.005, tol=1e-9, thickness=0.5)[1]
    Us = nls.solve_nonlinear_transient(fes, mat, load, 0.05, 0.005, tol=1e-9, thickness=0.5,
                                       sparse_tangent=True, vectorized=True)[1]
    assert np.max(np.abs(Ud)) > 0
    bc.same_newton_path(Us, Ud)


def test_transient_sparse_rejects_the_torch_backend():
    fes, mat, right = _cantilever_case()
    fes.assemble_mass(7800.0 * np.eye(2), thickness=0.5)
    fes.assemble_damping(RayleighDamping.calibrate(30.0, 200.0, 0.02))
    load = TimeHistoryLoad(LoadPattern(node_ids=right, dof_index=1), time_fn=lambda tt: -5.0)
    with pytest.raises(ValueError, match="sparse_tangent"):
        nls.solve_nonlinear_transient(fes, mat, load, 0.01, 0.005, sparse_tangent=True, backend="torch", thickness=0.5)


# ----------------------------------------------------------------------------- options
@pytest.mark.parametrize("ordering", ["NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD", "symmetric", None])
def test_every_lu_ordering_gives_the_same_solution(ordering):
    fes, mat, _ = _cantilever_case()
    ref = nls.solve_nonlinear_static(fes, mat, n_steps=3, tol=1e-9, max_iter=40, thickness=0.5)[1]
    out = nls.solve_nonlinear_static(fes, mat, n_steps=3, tol=1e-9, max_iter=40, thickness=0.5,
                                     sparse_tangent=True, lu_ordering=ordering)[1]
    bc.same_newton_path(out, ref)


def test_invalid_lu_ordering_is_rejected():
    fes, mat, _ = _cantilever_case()
    with pytest.raises(ValueError, match="lu_ordering"):
        nls.solve_nonlinear_static(fes, mat, n_steps=1, thickness=0.5, sparse_tangent=True, lu_ordering="bogus")
    with pytest.raises(ValueError, match="lu_ordering"):
        NewtonOptions(lu_ordering="bogus")


def test_newton_options_carries_the_new_fields():
    fes, mat, _ = _cantilever_case()
    ref = nls.solve_nonlinear_static(fes, mat, n_steps=3, tol=1e-9, max_iter=40, thickness=0.5)[1]
    opts = NewtonOptions(tol=1e-9, max_iter=40, sparse_tangent=True, lu_ordering="COLAMD")
    out = nls.solve_nonlinear_static(fes, mat, n_steps=3, options=opts, thickness=0.5, vectorized=True)[1]
    assert fes.last_assembly_modes == {"q": "batched"}
    bc.same_newton_path(out, ref)
    nls.solve_nonlinear_koiter_newton  # drivers without the fields must simply ignore them (no error on options)
    assert NewtonOptions(sparse_tangent=True).as_kwargs() == {"sparse_tangent": True}


def test_reuse_factorization_reaches_the_same_solution_with_fewer_assemblies():
    fes, mat, _ = _cantilever_case()
    counts = {"n": 0}
    orig = fes.assemble_tangent_stiffness

    def counting(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)
    fes.assemble_tangent_stiffness = counting
    ref = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-9, max_iter=60, thickness=0.5)[1]
    n_full = counts["n"]
    counts["n"] = 0
    out = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-9, max_iter=60, thickness=0.5,
                                     reuse_factorization=True)[1]
    n_reuse = counts["n"]
    assert bc.rel_err(out, ref) < 1e-7
    assert n_reuse < n_full, (n_reuse, n_full)
    counts["n"] = 0
    out_s = nls.solve_nonlinear_static(fes, mat, n_steps=4, tol=1e-9, max_iter=60, thickness=0.5,
                                       reuse_factorization=True, sparse_tangent=True)[1]
    assert bc.rel_err(out_s, ref) < 1e-7


# ----------------------------------------------------------------------------- memory
def test_large_model_never_forms_a_dense_tangent():
    n = 100                                                    # 10,000 Quad4TL elements, 20,402 DOFs
    nodes, elems, nx, ny = gb.quad_plate(n, n, L=float(n), H=float(n))
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": elems}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    left = [j * (nx + 1) for j in range(ny + 1)]
    fes.fix_dofs(left, [0, 1])
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    fes.add_nodal_force(right, 1, -1.0)
    mat = {"q": D}
    dense_bytes = fes.n_dof ** 2 * 8
    assert dense_bytes > 3.0e9                                 # the dense tangent would need 3.3 GB
    free = fes.free_dofs
    tracemalloc.start()
    K = fes.assemble_tangent_stiffness(np.zeros(fes.n_dof), mat, thickness=0.5, vectorized=True, sparse=True)
    Kff = K.tocsr()[free][:, free]
    import scipy.sparse.linalg as spla
    x = spla.splu(Kff.tocsc()).solve(fes.F[free])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 0.05 * dense_bytes, (peak, dense_bytes)
    assert np.linalg.norm(Kff @ x - fes.F[free]) < 1e-8 * np.linalg.norm(fes.F[free])
    u = np.zeros(fes.n_dof)
    u[free] = x
    assert u[2 * right[-1] + 1] < 0.0                          # the clamped plate deflects in the load direction

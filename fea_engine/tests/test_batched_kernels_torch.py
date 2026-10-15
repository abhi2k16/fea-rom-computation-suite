# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_batched_kernels_torch.py -- the torch backend of the batched total-Lagrangian kernels (batched_kernels_torch.py).

Two groups, because torch is an optional dependency:
  * ALGEBRA (always runs): the kernels ``tl_internal_force`` / ``tl_tangent`` are written once against an array namespace; here they run
    with ``xp=numpy`` and are compared with the NumPy batch kernels of the four element classes (Quad4PlaneStressTL, Hex8SolidTL,
    Hex20SolidTL, Tet10SolidTL) to 1e-12. This checks the algebra, the Voigt conventions and the DOF ordering, not the torch calls.
  * TORCH (skipped without torch): the same comparison through ``torch`` tensors (numpy in / numpy out wrappers, float64), the
    tangent against ``torch.autograd`` of the force, the whole-system assembly (dense and CSR) against ``FESystem``'s, the refusals,
    and ``device="cuda"`` when a GPU is present.
The author could run only the first group (no torch in the development sandbox); see the module docstring of the backend.
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
from fea_engine import batched_kernels_torch as bt
from fea_engine.contact_batch import ContactSet2D
from fea_engine.elements import Quad4PlaneStressTL, Hex8SolidTL, Hex20SolidTL, Tet10SolidTL
from fea_engine.material import D_plane_stress, D_solid3d
from fea_engine.torch_sparse_solver import _HAS_TORCH

pytestmark = pytest.mark.filterwarnings("ignore:.*spurious zero-energy modes.*:UserWarning")
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch is not installed")
MAT = Material(E=gb.E_MOD, nu=gb.NU)
TH = 0.5


def _elements():
    """(name, element, coords (E, n, dim), mat, kwargs for thickness)"""
    out = []
    nodes, el, _, _ = gb.quad_plate(3, 2)
    out.append(("quad4", Quad4PlaneStressTL(), nodes[el][:5], D_plane_stress(MAT), dict(thickness=TH)))
    from hex_meshes import hex20_from_hex8
    from fea_engine import mesh as meshlib
    m8 = meshlib.box_mesh(2.0, 1.0, 1.0, 2, 1, 1)
    out.append(("hex8", Hex8SolidTL(), m8.nodes[m8.elements], MAT, {}))
    m20 = hex20_from_hex8(m8)
    out.append(("hex20", Hex20SolidTL(), m20.nodes[m20.elements], D_solid3d(MAT), {}))
    tn, te = gb.tet10_box(2, 1, 1)
    out.append(("tet10", Tet10SolidTL(), tn[te][:6], MAT, {}))
    return out


def _distorted(X, seed):
    return X + np.random.default_rng(seed).normal(size=X.shape) * 0.02


ELEMENTS = _elements()
IDS = [e[0] for e in ELEMENTS]


# ----------------------------------------------------------------------------- algebra with xp = numpy (always runs)
@pytest.mark.parametrize("case", ELEMENTS, ids=IDS)
@pytest.mark.parametrize("scale", [0.02, 0.3])
def test_kernels_with_numpy_equal_the_numpy_batch_kernels(case, scale):
    name, el, X, mat, kw = case
    X = _distorted(X, 1)
    u = np.random.default_rng(2).normal(size=(len(X), X.shape[1] * X.shape[2])) * scale
    dN, JxW, D = bt._inputs(el, X, mat, kw.get("thickness", 1.0), None)
    eye = np.eye(dN.shape[-1])
    bc.equal_to_loop(bt.tl_internal_force(np, dN, JxW, u, D, eye), el.internal_force_batch(X, u, mat, **kw), label=f"{name} force")
    bc.equal_to_loop(bt.tl_tangent(np, dN, JxW, u, D, eye), el.tangent_stiffness_batch(X, u, mat, **kw), label=f"{name} tangent")


@pytest.mark.parametrize("case", ELEMENTS, ids=IDS)
def test_kernel_tangent_is_symmetric_and_the_derivative_of_the_force(case):
    name, el, X, mat, kw = case
    X = _distorted(X[:1], 3)
    dN, JxW, D = bt._inputs(el, X, mat, kw.get("thickness", 1.0), None)
    eye = np.eye(dN.shape[-1])
    n = X.shape[1] * X.shape[2]
    f = lambda w: bt.tl_internal_force(np, dN, JxW, w.reshape(1, n), D, eye)[0]
    K = lambda w: bt.tl_tangent(np, dN, JxW, w.reshape(1, n), D, eye)[0]
    bc.fd_tangent(f, K, np.random.default_rng(4).normal(size=n) * 0.1)


def test_supported_classes_and_refusal_of_others():
    assert all(bt.supports_torch_kernel(e[1]) for e in ELEMENTS)
    from fea_engine.elements import Hex8Solid3D
    assert not bt.supports_torch_kernel(Hex8Solid3D())
    with pytest.raises(NotImplementedError, match="no torch kernel"):
        bt._inputs(Hex8Solid3D(), np.zeros((1, 8, 3)), MAT, 1.0, None)


def test_importing_the_module_does_not_need_torch_and_calling_without_torch_says_so():
    if _HAS_TORCH:
        pytest.skip("torch is installed")
    name, el, X, mat, kw = ELEMENTS[1]
    with pytest.raises(ImportError, match="torch"):
        bt.internal_force_batch_torch(el, X, np.zeros((len(X), 24)), mat)


def _plate_system():
    nodes, el, nx, ny = gb.quad_plate(6, 2)
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": el}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    return fes, {"q": D_plane_stress(MAT)}


def test_system_refusals_do_not_need_torch():
    """Contact sets, springs, constraints and element state are refused before torch is even needed."""
    for add in ("contact_set", "spring", "constraint", "state"):
        fes, mat = _plate_system()
        if add == "contact_set":
            fes.add_contact_set(ContactSet2D([[0, 1, 2]], 1.0))
        elif add == "spring":
            fes._extra_K = tuple(fes._extra_K) + ((np.array([0, 1]), np.eye(2)),)
        elif add == "constraint":
            fes.add_constraint([(3, "ux", 1.0), (4, "ux", -1.0)])
        else:
            fes.state = {"q": [None] * len(fes._blocks[0][2])}
        with pytest.raises(NotImplementedError):
            bt.assemble_internal_force_torch(fes, np.zeros(fes.n_dof), mat)
        with pytest.raises(NotImplementedError):
            bt.assemble_tangent_stiffness_torch(fes, np.zeros(fes.n_dof), mat, sparse=True)


# ----------------------------------------------------------------------------- torch (skipped without torch)
@needs_torch
@pytest.mark.parametrize("case", ELEMENTS, ids=IDS)
@pytest.mark.parametrize("scale", [0.02, 0.3])
def test_torch_kernels_equal_the_numpy_batch_kernels(case, scale):
    name, el, X, mat, kw = case
    X = _distorted(X, 1)
    u = np.random.default_rng(2).normal(size=(len(X), X.shape[1] * X.shape[2])) * scale
    bc.equal_to_loop(bt.internal_force_batch_torch(el, X, u, mat, **kw), el.internal_force_batch(X, u, mat, **kw), label=f"{name} force")
    bc.equal_to_loop(bt.tangent_stiffness_batch_torch(el, X, u, mat, **kw), el.tangent_stiffness_batch(X, u, mat, **kw),
                     label=f"{name} tangent")
    geo = el._reference_geometry_batch(X)
    bc.equal_to_loop(bt.internal_force_batch_torch(el, X, u, mat, geometry=geo, **kw), el.internal_force_batch(X, u, mat, **kw))


@needs_torch
@pytest.mark.parametrize("case", ELEMENTS, ids=IDS)
def test_autograd_of_the_torch_force_is_the_analytic_tangent(case):
    import torch
    name, el, X, mat, kw = case
    X = _distorted(X[:2], 5)
    dN, JxW, D = bt._inputs(el, X, mat, kw.get("thickness", 1.0), None)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    dNt, Jt, Dt, eye = t(dN), t(JxW), t(D), torch.eye(dN.shape[-1], dtype=torch.float64)
    n = X.shape[1] * X.shape[2]
    u = np.random.default_rng(6).normal(size=(2, n)) * 0.1
    ref = el.tangent_stiffness_batch(X, u, mat, **kw)
    for e in range(2):
        J = torch.autograd.functional.jacobian(lambda w: bt.tl_internal_force(torch, dNt[e:e + 1], Jt[e:e + 1], w.reshape(1, n), Dt, eye)[0],
                                               t(u[e]))
        bc.equal_to_loop(J.numpy(), ref[e], rtol=1e-9, label=f"{name} autograd tangent")


@needs_torch
def test_torch_system_assembly_equals_fesystem_dense_and_csr():
    fes, mat = _plate_system()
    u = np.random.default_rng(7).normal(size=fes.n_dof) * 0.05
    kw = dict(thickness=TH)
    bc.equal_to_loop(bt.assemble_internal_force_torch(fes, u, mat, **kw), fes.assemble_internal_force(u, mat, **kw), label="force")
    K = fes.assemble_tangent_stiffness(u, mat, **kw)
    bc.equal_to_loop(bt.assemble_tangent_stiffness_torch(fes, u, mat, **kw), K, label="dense tangent")
    bc.equal_to_loop(bt.assemble_tangent_stiffness_torch(fes, u, mat, sparse=True, **kw), K, label="CSR tangent")
    for c in (1, 5):
        bc.equal_to_loop(bt.assemble_internal_force_torch(fes, u, mat, chunk_size=c, **kw), fes.assemble_internal_force(u, mat, **kw))


@needs_torch
def test_torch_assembly_of_a_hexahedron_and_a_tet10_system():
    nodes, elems = gb.tet10_box(2, 1, 1)
    from fea_engine import Mesh
    fes = FESystem(Mesh(nodes, elems, 3), Tet10SolidTL(), sparse=False)
    u = np.random.default_rng(8).normal(size=fes.n_dof) * 0.05
    bc.equal_to_loop(bt.assemble_internal_force_torch(fes, u, MAT), fes.assemble_internal_force(u, MAT))
    bc.equal_to_loop(bt.assemble_tangent_stiffness_torch(fes, u, MAT, sparse=True), fes.assemble_tangent_stiffness(u, MAT))


@needs_torch
def test_cuda_if_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    name, el, X, mat, kw = ELEMENTS[1]
    u = np.random.default_rng(9).normal(size=(len(X), 24)) * 0.1
    bc.equal_to_loop(bt.internal_force_batch_torch(el, X, u, mat, device="cuda"), el.internal_force_batch(X, u, mat), rtol=1e-10)

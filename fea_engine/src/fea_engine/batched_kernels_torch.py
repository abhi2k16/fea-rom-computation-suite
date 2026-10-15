# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batched_kernels_torch.py -- a side-by-side PyTorch backend for the batched total-Lagrangian kernels (Quad4PlaneStressTL, Hex8SolidTL,
Hex20SolidTL, Tet10SolidTL).

Same convention as the other torch peer modules (``beam2d_reissner_vectorized_torch.py``, ``torch_sparse_solver.py``): a separate file, an
explicit ``device=`` choice ("cpu" or "cuda"), numpy in / numpy out for the drop-in functions, optional dependency (importing this module
works without torch; calling a torch function without it raises ImportError). Nothing in the NumPy path changes.

What is here
------------
``tl_internal_force`` / ``tl_tangent`` -- the kernels themselves, written ONCE against an array namespace ``xp`` (``numpy`` or ``torch``)
    and using only operations with the same meaning in both (``einsum``, ``stack``, ``reshape``, arithmetic). They take
    ``dN`` (E, Q, n, dim) = dN/dX, ``JxW`` (E, Q) = det(J0) * w * (thickness), ``u`` (E, n*dim), the elasticity matrix ``D`` and an identity
    ``eye`` (dim, dim) of the right array type. With ``xp=numpy`` they run anywhere, which is how the algebra is tested in the sandbox;
    with ``xp=torch`` they are differentiable (``torch.autograd`` gives the tangent from the force, tested against the analytic tangent).
``internal_force_batch_torch`` / ``tangent_stiffness_batch_torch`` -- element-level, numpy in / numpy out, like the element's own
    ``internal_force_batch`` / ``tangent_stiffness_batch`` (same arguments plus ``device=``, ``dtype=``).
``assemble_internal_force_torch`` / ``assemble_tangent_stiffness_torch`` -- whole ``FESystem`` assembly with the cached block geometry,
    chunking and the NumPy scatter (dense, or CSR with ``sparse=True``), result numpy.

Scope: structural blocks of the four element classes above. A system with contact elements, contact sets, springs, linear constraints or
path-dependent element state is refused (``NotImplementedError``) instead of silently dropping them. The Newton drivers are not changed:
they call the NumPy assembly; the torch assembly is for sweeps, GPU use and differentiation, as for the beam torch module.

STATUS (2026-10-15): the algebra of ``tl_internal_force`` / ``tl_tangent`` was executed by the author with ``xp=numpy`` (no torch in the
development sandbox). The torch tests (tests/test_batched_kernels_torch.py: torch kernels, autograd tangent, system assembly, CUDA force)
were run by the user on their machine: 29 passed, 1 skipped (the torch-absent test), 0 failed, including the CUDA test (not skipped).
No timing of the torch path has been made.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .torch_sparse_solver import _HAS_TORCH, _require_torch

if _HAS_TORCH:
    import torch

__all__ = ["tl_internal_force", "tl_tangent", "internal_force_batch_torch", "tangent_stiffness_batch_torch",
           "assemble_internal_force_torch", "assemble_tangent_stiffness_torch", "supports_torch_kernel"]


# ----------------------------------------------------------------------------- the kernels, once, for numpy and torch
def _shear_pairs(d):
    return [(0, 1)] if d == 2 else [(0, 1), (1, 2), (0, 2)]


def _kinematics(xp, dN, u, D, eye):
    E, Q, n, d = dN.shape
    ue = u.reshape(E, n, d)
    F = eye + xp.einsum('eai,eqaj->eqij', ue, dN)                              # F[i, j] = delta_ij + u_a,i dN_a,j
    C = xp.einsum('eqki,eqkj->eqij', F, F)                                    # F^T F
    pairs = _shear_pairs(d)
    strain = [0.5 * (C[..., i, i] - 1.0) for i in range(d)] + [C[..., i, j] for i, j in pairs]     # engineering shear
    Ev = xp.stack(strain, -1)                                                  # (E, Q, nv)
    S = xp.einsum('eqs,ts->eqt', Ev, D)                                        # = Ev @ D^T
    rows = []
    for k in range(d):                                                         # normal strain k: dE_kk = F_ik dN_a,k du_a,i
        rows.append(xp.stack([F[..., i, k][..., None] * dN[..., k] for i in range(d)], -1))
    for k, l in pairs:                                                         # shear strain (k, l)
        rows.append(xp.stack([F[..., i, k][..., None] * dN[..., l] + F[..., i, l][..., None] * dN[..., k]
                              for i in range(d)], -1))
    B = xp.stack(rows, 2).reshape(E, Q, len(rows), n * d)                      # (E, Q, nv, n*d), node-major DOFs
    return F, S, B


def tl_internal_force(xp, dN, JxW, u, D, eye):
    """Internal forces (E, n*dim): sum over Gauss points of JxW * B_nl^T S (St. Venant-Kirchhoff, total Lagrangian)."""
    _, S, B = _kinematics(xp, dN, u, D, eye)
    return xp.einsum('eqsd,eqs,eq->ed', B, S, JxW)


def tl_tangent(xp, dN, JxW, u, D, eye):
    """Tangent matrices (E, n*dim, n*dim): sum over Gauss points of JxW * (B_nl^T D B_nl + (dN^T S dN) (x) I)."""
    E, Q, n, d = dN.shape
    _, S, B = _kinematics(xp, dN, u, D, eye)
    K_mat = xp.einsum('eq,eqsa,st,eqtb->eab', JxW, B, D, B)
    pairs = _shear_pairs(d)
    def comp(i, j):
        return i if i == j else d + pairs.index((min(i, j), max(i, j)))
    Smat = xp.stack([xp.stack([S[..., comp(i, j)] for j in range(d)], -1) for i in range(d)], -2)     # (E, Q, d, d)
    Kg = xp.einsum('eq,eqak,eqkl,eqbl->eab', JxW, dN, Smat, dN)                                      # (E, n, n)
    K_geo = xp.einsum('eab,ij->eaibj', Kg, eye).reshape(E, n * d, n * d)
    return K_mat + K_geo


# ----------------------------------------------------------------------------- element level, numpy in / numpy out
def supports_torch_kernel(formulation):
    """True for the element classes this module has kernels for."""
    from .elements import Quad4PlaneStressTL, Hex8SolidTL, Hex20SolidTL, Tet10SolidTL
    return isinstance(formulation, (Quad4PlaneStressTL, Hex8SolidTL, Hex20SolidTL, Tet10SolidTL))


def _inputs(element, coords, mat, thickness, geometry):
    """(dN, JxW including thickness, D) as float64 numpy arrays for the element class."""
    from .elements import Quad4PlaneStressTL
    if not supports_torch_kernel(element):
        raise NotImplementedError(f"no torch kernel for {type(element).__name__}")
    dN, JxW = element._reference_geometry_batch(np.asarray(coords, float)) if geometry is None else geometry
    if isinstance(element, Quad4PlaneStressTL):
        D, JxW = np.asarray(mat, float), np.asarray(JxW, float) * thickness
    else:
        D = element._elasticity(mat)
    return np.asarray(dN, float), np.asarray(JxW, float), np.asarray(D, float)


def _run(kernel, element, coords, u, mat, thickness, geometry, device, dtype):
    _require_torch()
    dtype = torch.float64 if dtype is None else dtype
    dN, JxW, D = _inputs(element, coords, mat, thickness, geometry)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    dNt, Jt, Dt = t(dN), t(JxW), t(D)
    eye = torch.eye(dNt.shape[-1], dtype=dtype, device=device)
    out = kernel(torch, dNt, Jt, t(u), Dt, eye)
    return out.detach().cpu().numpy()


def internal_force_batch_torch(element, coords, u, mat, thickness=1.0, geometry=None, device="cpu", dtype=None):
    """Torch version of ``element.internal_force_batch``: numpy ``coords`` (E, n, dim), ``u`` (E, n*dim), numpy result (E, n*dim).
    ``geometry`` is the optional precomputed (dN/dX, det(J0) w) pair, as for the NumPy kernel. ``dtype`` defaults to float64."""
    return _run(tl_internal_force, element, coords, u, mat, thickness, geometry, device, dtype)


def tangent_stiffness_batch_torch(element, coords, u, mat, thickness=1.0, geometry=None, device="cpu", dtype=None):
    """Torch version of ``element.tangent_stiffness_batch``: numpy result (E, n*dim, n*dim)."""
    return _run(tl_tangent, element, coords, u, mat, thickness, geometry, device, dtype)


# ----------------------------------------------------------------------------- whole-system assembly
def _check_system(fes):
    if fes.contact_elements or fes.contact_sets or fes._extra_K or fes._constraints:
        raise NotImplementedError("torch assembly covers structural blocks only: contact elements, contact sets, springs and "
                                  "linear constraints are not included (use the NumPy assembly)")
    if fes.state is not None or fes.iter_state is not None:
        raise NotImplementedError("torch assembly does not support path-dependent element state")
    for name, formulation, _ in fes._blocks:
        if not supports_torch_kernel(formulation):
            raise NotImplementedError(f"block {name!r}: no torch kernel for {type(formulation).__name__}")


def _block_chunks(fes, name, formulation, connectivity, kind, chunk_size):
    from . import batched_kernels as bk
    geo = fes._geometry_cache.get(name, fes.mesh.nodes, formulation, connectivity)
    bpe = bk.bytes_per_element(formulation, kind, n_gauss=geo.JxW.shape[1])
    nodes = np.asarray(fes.mesh.nodes)
    for s in bk.chunk_slices(len(connectivity), chunk_size, bpe):
        yield geo, s, nodes[geo.connectivity[s]]


def assemble_internal_force_torch(fes, u_global, mat, device="cpu", chunk_size=None, dtype=None, **kwargs):
    """Global internal force of the structural blocks with the torch kernels (numpy result). ``mat`` as for
    ``FESystem.assemble_internal_force`` (a dict per block is allowed); ``thickness`` is passed as a keyword for Quad4PlaneStressTL."""
    from . import batched_kernels as bk
    _check_system(fes)          # refusals first: they need no torch
    _require_torch()
    u_global = np.asarray(u_global, float)
    F = np.zeros(fes.n_dof)
    for name, formulation, connectivity in fes._blocks:
        block_mat = fes._per_block_arg(mat, name)
        for geo, s, X in _block_chunks(fes, name, formulation, connectivity, "force", chunk_size):
            fe = internal_force_batch_torch(formulation, X, u_global[geo.dofs[s]], block_mat,
                                            thickness=kwargs.get("thickness", 1.0),
                                            geometry=(geo.shape_grad[s], geo.JxW[s]), device=device, dtype=dtype)
            F += bk.scatter_add_vector(fes.n_dof, geo.dofs[s], fe)
    return F


def assemble_tangent_stiffness_torch(fes, u_global, mat, device="cpu", sparse=False, chunk_size=None, dtype=None, **kwargs):
    """Global tangent of the structural blocks with the torch kernels: dense array, or a CSR matrix with ``sparse=True``
    (blocks scattered through the cached pattern, as in the NumPy path)."""
    import scipy.sparse as sps
    from . import batched_kernels as bk
    _check_system(fes)          # refusals first: they need no torch
    _require_torch()
    u_global = np.asarray(u_global, float)
    out = sps.csr_matrix((fes.n_dof, fes.n_dof)) if sparse else np.zeros((fes.n_dof, fes.n_dof))
    for name, formulation, connectivity in fes._blocks:
        block_mat = fes._per_block_arg(mat, name)
        if sparse:
            geo = fes._geometry_cache.get(name, fes.mesh.nodes, formulation, connectivity)
            pat = fes._pattern_cache.get(name, geo.dofs, fes.n_dof)
            data = np.zeros(pat.nnz)
        for geo, s, X in _block_chunks(fes, name, formulation, connectivity, "tangent", chunk_size):
            ke = tangent_stiffness_batch_torch(formulation, X, u_global[geo.dofs[s]], block_mat,
                                               thickness=kwargs.get("thickness", 1.0),
                                               geometry=(geo.shape_grad[s], geo.JxW[s]), device=device, dtype=dtype)
            if sparse:
                bk.scatter_add_csr_data(data, pat, s, ke)
            else:
                bk.scatter_add_matrix(out, geo.dofs[s], ke)
        if sparse:
            out = out + sps.csr_matrix((data, pat.indices, pat.indptr), shape=(fes.n_dof, fes.n_dof))
    return out

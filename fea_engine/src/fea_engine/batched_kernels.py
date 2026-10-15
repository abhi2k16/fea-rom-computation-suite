# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batched_kernels.py -- shared infrastructure for opt-in batched (all elements of a block at once) nonlinear element
kernels: the batch-method protocol, memory-bounded chunking, vectorised DOF maps and scatter, and the per-block
geometry cache.

Nothing here changes any existing behaviour. No element implements the batch methods yet and FESystem does not call
them yet; this module is the foundation that the batched kernels and the vectorised assembly build on.

Batch-method protocol (all optional; a formulation without them is simply assembled by the existing per-element loop)
---------------------------------------------------------------------------------------------------------------------
An Element subclass opts in by setting ``supports_batch = True`` and defining

    internal_force_batch(coords, u, mat, thickness=1.0, **kwargs)     -> (E, ndof)
    tangent_stiffness_batch(coords, u, mat, thickness=1.0, **kwargs)  -> (E, ndof, ndof)

with ``coords`` of shape (E, n_nodes, dim) and ``u`` of shape (E, ndof), ndof = n_nodes * dofs_per_node, in the same
local DOF order as the per-element methods. ``mat`` is the same object the per-element method takes (one per block,
constant over the block). A formulation may also provide ``batch_bytes_per_element(kind)`` (kind "force" or
"tangent") to give a more accurate working-memory estimate than the generic one below. Path-dependent elements (those
that need ``state=`` / ``iter_state=``) do not use this protocol.

Chunking
--------
A batch call allocates arrays of size (elements x Gauss points x strain components x element DOFs). For a 3-D
second-order element that is tens of kilobytes per element (measured for a Hex20 3x3x3 rule: 78 kB for the
strain-displacement array alone), so a 10^5-element block would need gigabytes. ``chunk_slices`` therefore splits a
block into slices whose estimated working memory stays under a budget (default 256 MB), or into slices of an explicit
``chunk_size``. Results of a chunked evaluation equal the unchunked ones up to floating-point summation order, not
bit for bit.

Sparse tangent
--------------
``SparsePatternCache`` stores, per block, the CSR sparsity pattern of the block's assembled tangent and, for every
entry of every element matrix, the position of that entry in the CSR data array (one ``np.unique`` over the flattened
(row, column) keys, done once). Each later assembly is then ``bincount(positions, weights=Ke)``; no index sorting,
no COO-to-CSR conversion and no n x n array. The pattern depends only on the element-to-DOF map, so it survives mesh
motion; it is rebuilt if the map changes.

Geometry cache
--------------
``BlockGeometryCache`` stores, per block, the reference-configuration shape-function gradients and ``JxW`` (from
``mesh_transform.MeshTransformation``) together with the element-to-DOF map. For a total-Lagrangian element these are
exactly dN/dX and det(J0)*w. An entry is rebuilt when the node coordinates, the connectivity or the Gauss order
differ from those it was built with (exact comparison against a stored copy), so moving the mesh can never leave a
stale cache.

References (methods in this module are adopted from):
  - O. C. Zienkiewicz, R. L. Taylor (2000), "The Finite Element Method", 5th ed., Butterworth-Heinemann (element-level
    isoparametric integration that the batch kernels reproduce)
  - T. Belytschko, W. K. Liu, B. Moran, K. Elkhodary (2014), "Nonlinear Finite Elements for Continua and Structures",
    2nd ed., Wiley (total-Lagrangian internal force and tangent)
  - R. Cuvelier, C. Japhet, G. Scarella (2014), "An efficient way to assemble finite element matrices in vector
    languages", arXiv:1401.3301 (assembly of all elements at once with array operations)
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass
import numpy as np

DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024      # working-array budget per chunk
BATCH_METHODS = ("internal_force_batch", "tangent_stiffness_batch")


# ----------------------------------------------------------------------------- protocol
def supports_batch(formulation):
    """True if ``formulation`` opts in (``supports_batch`` set) AND defines both batch methods."""
    return bool(getattr(formulation, "supports_batch", False)) and all(
        callable(getattr(formulation, m, None)) for m in BATCH_METHODS)


def batch_unavailable_reason(formulation):
    """None if the formulation can be batched, else a short reason string (used for per-block reports)."""
    if supports_batch(formulation):
        return None
    name = type(formulation).__name__
    if not getattr(formulation, "supports_batch", False):
        return f"{name} does not declare supports_batch"
    missing = [m for m in BATCH_METHODS if not callable(getattr(formulation, m, None))]
    return f"{name} declares supports_batch but lacks {', '.join(missing)}"


# ----------------------------------------------------------------------------- memory estimates
def n_strain_components(dim):
    """Independent symmetric strain components: 1, 3, 6 for dim 1, 2, 3."""
    return dim * (dim + 1) // 2


def default_bytes_per_element(n_gauss, ndof, dim, kind="tangent"):
    """Generic working-memory estimate in bytes per element (float64).

    force   : B (G*s*ndof) + the deformation-gradient/strain/stress work arrays (4 * G * dim**2) + the force (ndof)
    tangent : the force estimate + D@B (G*s*ndof) + B^T D B accumulation (G*s*ndof) + the element tangent (ndof**2)

    Check against the measured sizes for a Hex20 3x3x3 rule (G=27, ndof=60, dim=3): B is 27*6*60*8 = 77.8 kB per
    element and the element tangent 60**2*8 = 28.8 kB, matching the arrays measured in the planning study.
    It is an estimate of the dominant arrays, not a guarantee about peak resident memory.
    """
    s = n_strain_components(dim)
    b = n_gauss * s * ndof
    force = b + 4 * n_gauss * dim * dim + ndof
    if kind == "force":
        return 8 * force
    if kind == "tangent":
        return 8 * (force + 2 * b + ndof * ndof)
    raise ValueError(f"kind must be 'force' or 'tangent', got {kind!r}")


def bytes_per_element(formulation, kind="tangent", n_gauss=None):
    """Working bytes per element for ``formulation``: its own ``batch_bytes_per_element(kind)`` if it has one,
    otherwise the generic estimate from its node count, DOFs per node, dimension and Gauss rule."""
    own = getattr(formulation, "batch_bytes_per_element", None)
    if callable(own):
        return int(own(kind))
    if n_gauss is None:
        order = getattr(formulation, "gauss_order", 2)
        n_gauss = order ** formulation.dim
    ndof = formulation.n_nodes * formulation.dofs_per_node
    return default_bytes_per_element(n_gauss, ndof, formulation.dim, kind)


# ----------------------------------------------------------------------------- chunking
def chunk_size_for(bytes_per_elem, budget_bytes=DEFAULT_BUDGET_BYTES):
    """Largest number of elements whose estimated working memory fits the budget (at least 1)."""
    if bytes_per_elem <= 0:
        raise ValueError("bytes_per_elem must be positive")
    return max(1, int(budget_bytes // bytes_per_elem))


def chunk_slices(n_elements, chunk_size=None, bytes_per_elem=None, budget_bytes=DEFAULT_BUDGET_BYTES):
    """Split range(n_elements) into consecutive slices.

    ``chunk_size`` (elements per slice) wins if given; otherwise it is derived from ``bytes_per_elem`` and
    ``budget_bytes``; with neither, a single slice covers the whole block. An empty block gives no slices."""
    n_elements = int(n_elements)
    if n_elements <= 0:
        return []
    if chunk_size is None:
        chunk_size = n_elements if bytes_per_elem is None else chunk_size_for(bytes_per_elem, budget_bytes)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [slice(i, min(i + chunk_size, n_elements)) for i in range(0, n_elements, chunk_size)]


def map_chunks(fn, n_elements, chunk_size=None, bytes_per_elem=None, budget_bytes=DEFAULT_BUDGET_BYTES):
    """Call ``fn(slice)`` for every chunk and concatenate the per-chunk arrays along the element axis.

    ``fn`` must return an array whose first axis is the number of elements in the slice. Returns an array with
    ``n_elements`` rows (or None for an empty block)."""
    parts = [np.asarray(fn(s)) for s in chunk_slices(n_elements, chunk_size, bytes_per_elem, budget_bytes)]
    return np.concatenate(parts, axis=0) if parts else None


# ----------------------------------------------------------------------------- DOF map and scatter
def element_dof_map(connectivity, dofs_per_node):
    """(E, n_nodes*dofs_per_node) global DOF numbers, in the order FESystem._global_dofs uses
    (node-major: DOFs of the first node, then the second, ...)."""
    conn = np.asarray(connectivity, dtype=np.int64)
    k = np.arange(dofs_per_node, dtype=np.int64)
    return (dofs_per_node * conn[:, :, None] + k[None, None, :]).reshape(conn.shape[0], -1)


def scatter_add_vector(n_dof, dofs, fe):
    """Sum element vectors ``fe`` (E, ndof) into a global vector of length ``n_dof`` using ``dofs`` (E, ndof).
    bincount sums repeated indices correctly (unlike fancy-index ``+=``)."""
    return np.bincount(np.asarray(dofs).ravel(), weights=np.asarray(fe, float).ravel(), minlength=n_dof)


def scatter_add_matrix(K, dofs, Ke):
    """Sum element matrices ``Ke`` (E, ndof, ndof) into the dense global matrix ``K`` (n, n) in place, using ``dofs``
    (E, ndof). Repeated indices are summed correctly.

    Two equivalent routes, chosen by size: bincount over flattened indices is fastest but allocates a second n*n
    array, so it is used only when that array is at most 64 times the number of scattered entries; otherwise
    ``np.add.at`` adds in place without the extra allocation. (Measured on one machine: 324 Quad4 elements, n=722,
    bincount 0.7 ms against add.at 3.6 ms; 3600 elements, n=7442, bincount needs a 443 MB temporary.)"""
    dofs = np.asarray(dofs)
    n = K.shape[0]
    if n * n <= 64 * Ke.size:
        idx = (dofs[:, :, None] * n + dofs[:, None, :]).ravel()
        K += np.bincount(idx, weights=np.asarray(Ke, float).ravel(), minlength=n * n).reshape(n, n)
    else:
        np.add.at(K, (dofs[:, :, None], dofs[:, None, :]), Ke)
    return K


# ----------------------------------------------------------------------------- sparse pattern cache
@dataclass
class SparsePattern:
    """CSR pattern of one block's tangent: ``indices``/``indptr`` (sorted), ``pos`` (E, ndof, ndof) positions of every
    element-matrix entry in the CSR data array, ``dofs`` copy of the DOF map it was built from, ``n_dof``."""
    indices: np.ndarray
    indptr: np.ndarray
    pos: np.ndarray
    dofs: np.ndarray
    n_dof: int

    @property
    def nnz(self):
        return int(self.indices.size)

    @property
    def nbytes(self):
        return int(self.indices.nbytes + self.indptr.nbytes + self.pos.nbytes + self.dofs.nbytes)


def build_sparse_pattern(dofs, n_dof):
    """Pattern and position map for element matrices scattered through ``dofs`` (E, ndof) into an n_dof x n_dof matrix."""
    dofs = np.asarray(dofs, dtype=np.int64)
    key = (dofs[:, :, None] * n_dof + dofs[:, None, :]).ravel()
    uniq, inv = np.unique(key, return_inverse=True)
    rows, cols = uniq // n_dof, uniq % n_dof
    indptr = np.zeros(n_dof + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n_dof), out=indptr[1:])
    pos_dtype = np.int32 if uniq.size < 2 ** 31 - 1 else np.int64
    return SparsePattern(indices=cols.astype(np.int32 if n_dof < 2 ** 31 - 1 else np.int64), indptr=indptr,
                         pos=inv.reshape(dofs.shape[0], dofs.shape[1], dofs.shape[1]).astype(pos_dtype),
                         dofs=dofs.copy(), n_dof=int(n_dof))


class SparsePatternCache:
    """Per-block ``SparsePattern`` store; ``builds`` counts pattern constructions."""

    def __init__(self):
        self._entries = {}
        self.builds = 0

    def clear(self):
        self._entries.clear()

    @property
    def nbytes(self):
        return sum(e.nbytes for e in self._entries.values())

    def get(self, key, dofs, n_dof):
        entry = self._entries.get(key)
        if entry is not None and entry.n_dof == n_dof and entry.dofs.shape == np.shape(dofs) \
                and np.array_equal(entry.dofs, dofs):
            return entry
        entry = build_sparse_pattern(dofs, n_dof)
        self._entries[key] = entry
        self.builds += 1
        return entry


def scatter_add_csr_data(data, pattern, sl, Ke):
    """Add element matrices ``Ke`` (m, ndof, ndof) of the elements in slice ``sl`` into the CSR data array ``data`` of
    ``pattern`` in place (bincount over the cached positions; repeated entries are summed)."""
    data += np.bincount(pattern.pos[sl].ravel(), weights=np.asarray(Ke, float).ravel(), minlength=data.size)
    return data


# ----------------------------------------------------------------------------- geometry cache
@dataclass
class BlockGeometry:
    """Reference-configuration data of one block.

    shape_grad (E, Q, n_basis, dim): physical gradients dN/dX at every Gauss point; JxW (E, Q); dofs (E, ndof) global
    DOF map; connectivity (E, n_basis); gauss_order; plus the copies of the inputs it was built from."""
    shape_grad: np.ndarray
    JxW: np.ndarray
    dofs: np.ndarray
    connectivity: np.ndarray
    gauss_order: int
    nodes_copy: np.ndarray

    @property
    def n_elements(self):
        return self.connectivity.shape[0]

    @property
    def nbytes(self):
        return int(self.shape_grad.nbytes + self.JxW.nbytes + self.dofs.nbytes + self.connectivity.nbytes
                   + self.nodes_copy.nbytes)

    def is_current(self, nodes, connectivity, gauss_order):
        """True if built from exactly these nodes, connectivity and Gauss order."""
        return (self.gauss_order == gauss_order and self.connectivity.shape == np.shape(connectivity)
                and self.nodes_copy.shape == np.shape(nodes)
                and np.array_equal(self.nodes_copy, nodes) and np.array_equal(self.connectivity, connectivity))


class BlockGeometryCache:
    """Per-block ``BlockGeometry`` store with exact invalidation. Keys are block names (None for a single-block mesh).

    ``get(key, mesh_nodes, formulation, connectivity, gauss_order=None)`` returns a current entry, building it with
    ``MeshTransformation`` (or, for an element with ``own_batch_geometry = True``, with its ``_reference_geometry_batch``; the
    ``gauss_order`` argument is then ignored) if none exists or the inputs changed. ``builds`` counts rebuilds (for tests and reports)."""

    def __init__(self):
        self._entries = {}
        self.builds = 0

    def clear(self, key="__all__"):
        if key == "__all__":
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    def __len__(self):
        return len(self._entries)

    @property
    def nbytes(self):
        return sum(e.nbytes for e in self._entries.values())

    def get(self, key, mesh_nodes, formulation, connectivity, gauss_order=None):
        from .mesh_transform import MeshTransformation
        order = formulation.gauss_order if gauss_order is None else int(gauss_order)
        nodes = np.asarray(mesh_nodes)
        conn = np.asarray(connectivity)
        entry = self._entries.get(key)
        if entry is not None and entry.is_current(nodes, conn, order):
            return entry

        if getattr(formulation, "own_batch_geometry", False):
            # the element supplies its own reference geometry (its quadrature differs from MeshTransformation's; Tet10SolidTL)
            shape_grad, JxW = formulation._reference_geometry_batch(nodes[conn])
        else:
            class _Mesh:                                   # MeshTransformation only needs .nodes (+ .elements)
                pass
            m = _Mesh()
            m.nodes, m.elements = nodes, conn
            mt = MeshTransformation(m, formulation, gauss_order=order, connectivity=conn)
            shape_grad, JxW = mt.shape_grad, mt.JxW
        entry = BlockGeometry(
            shape_grad=np.ascontiguousarray(shape_grad), JxW=np.ascontiguousarray(JxW),
            dofs=element_dof_map(conn, formulation.dofs_per_node), connectivity=conn.copy(),
            gauss_order=order, nodes_copy=nodes.copy())
        self._entries[key] = entry
        self.builds += 1
        return entry

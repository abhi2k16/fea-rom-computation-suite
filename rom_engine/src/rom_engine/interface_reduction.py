# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
interface_reduction.py -- interface-reduction and dual-assembly variants of component mode synthesis, for
substructures that meet a joint, built on `cms.py`.

- `dual_craig_bampton` / `couple_dual`  Dual Craig-Bampton: free-interface modes plus residual attachment modes,
  so the interface FORCES are the coupling coordinates; substructures are joined by compatibility constraints on
  their interface displacements and the constrained system is reduced on the null space. After Rixen (2002), as
  used for joints by Yuan et al. 2019. The subspace of one substructure equals the Rubin basis of
  `cms.free_interface_cms`; what is new here is the assembly.
- `joint_interface_modes` / `jim_craig_bampton`  Craig-Bampton model whose JOINT DOFs are replaced by a few joint
  interface modes: the Craig-Bampton boundary block is statically condensed onto the joint DOFs, the eigenproblem
  of that condensed pair gives the modes, and the non-joint interface DOFs stay physical. This is our reading of
  "statically condense to the joint, then eigenanalysis" in the plan; it has NOT been checked against the source
  paper's full definition.
- `gram_schmidt_interface_modes` / `interface_reduced_craig_bampton`  Interface modes built by Gram-Schmidt
  orthonormalisation of the interface rows of several mode sets (for example free-interface and full-stick modes,
  after Battiato et al.), applied as the interface reduction of a Craig-Bampton model.
- `fourier_interface_basis` / `multistage_fourier_basis`  Fourier reduction for a circular interface: stage one
  expands each ring in cos/sin harmonics, stage two compresses the ring-wise harmonic coefficients across rings by a
  singular value decomposition of interface snapshots.
- `trial_vector_derivative_basis`  modal derivatives of the free-interface (or any) basis, filtered by the Gram
  eigenvalue cut-off of `augment_with_derivatives`: a thin wrapper over `modal_derivatives`.
- `natural_frequency_errors`, `mac_matrix`, `mac_diagonal`  the comparison metrics against the unreduced model.

Scope: dense linear algebra, linear substructures (the joint nonlinearity is added by the caller), and
substructures with non-singular K (restrain rigid-body motion first).

References (methods in this module are adopted from):
  - D. J. Rixen (2004), "A dual Craig-Bampton method for dynamic substructuring", J. Comput. Appl. Math. 168
  - J. Yuan, F. El-Haddad, L. Salles, C. Wong (2019), "Numerical assessment of reduced order modeling techniques for dynamic analysis of jointed structures with contact nonlinearities"
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, null_space

from .cms import craig_bampton, free_interface_cms
from .modal_derivatives import static_modal_derivatives, augment_with_derivatives

__all__ = ["DualCraigBamptonModel", "dual_craig_bampton", "couple_dual", "joint_interface_modes",
           "jim_craig_bampton", "gram_schmidt_interface_modes", "interface_reduced_craig_bampton",
           "fourier_interface_basis", "multistage_fourier_basis", "trial_vector_derivative_basis",
           "natural_frequency_errors", "mac_matrix", "mac_diagonal", "InterfaceReducedBasis"]


# ----------------------------------------------------------------------------------------- dual CB
@dataclass
class DualCraigBamptonModel:
    """T (n, n_modes + n_interface_force_modes), reduced K and M, free-interface frequencies and the interface DOFs."""
    T: np.ndarray
    K: np.ndarray
    M: np.ndarray
    omega_free: np.ndarray
    interface: np.ndarray
    n_modes: int

    def expand(self, coords):
        return self.T @ np.asarray(coords, float)


def dual_craig_bampton(K, M, interface, n_modes):
    """Dual Craig-Bampton reduction: `n_modes` free-interface modes plus the residual attachment modes of the
    interface forces. K must be non-singular."""
    fi = free_interface_cms(K, M, interface, n_modes, method="rubin")
    return DualCraigBamptonModel(fi.T, fi.K, fi.M, fi.omega_free, fi.interface, fi.n_modes)


def couple_dual(models, pairs):
    """Couple dual Craig-Bampton substructures by displacement compatibility.

    models : list of DualCraigBamptonModel. pairs : list of (a, dof_a, b, dof_b): the physical DOF `dof_a` of
    model a must equal DOF `dof_b` of model b. The constraints C q = 0 on the stacked reduced coordinates are
    eliminated on the null space of C, so the result is a standard symmetric eigenproblem.
    Returns dict(K, M, N (null-space basis of the stacked coordinates), omega (rad/s), offsets)."""
    sizes = [m.T.shape[1] for m in models]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    Kb = np.zeros((offs[-1],) * 2)
    Mb = np.zeros_like(Kb)
    for m, o in zip(models, offs[:-1]):
        s = slice(o, o + m.T.shape[1])
        Kb[s, s], Mb[s, s] = m.K, m.M
    C = np.zeros((len(pairs), offs[-1]))
    for r, (a, da, b, db) in enumerate(pairs):
        C[r, offs[a]:offs[a + 1]] += models[a].T[da]
        C[r, offs[b]:offs[b + 1]] -= models[b].T[db]
    N = null_space(C) if len(pairs) else np.eye(offs[-1])
    Kc, Mc = N.T @ Kb @ N, N.T @ Mb @ N
    w2 = eigh(0.5 * (Kc + Kc.T), 0.5 * (Mc + Mc.T), eigvals_only=True)
    return dict(K=Kc, M=Mc, N=N, omega=np.sqrt(np.clip(w2, 0.0, None)), offsets=offs)


# ----------------------------------------------------------------------------------------- JIM
def joint_interface_modes(cb, joint, n_modes):
    """Joint interface modes of a Craig-Bampton model `cb`: `joint` are positions within `cb.interface`. The
    boundary block (K_bb, M_bb) is statically condensed to the joint DOFs and its lowest `n_modes` eigenvectors
    are returned as (n_joint, n_modes), together with their frequencies [rad/s]."""
    nb = cb.n_interface
    joint = np.asarray(joint, int)
    rest = np.setdiff1d(np.arange(nb), joint)
    Kbb, Mbb = cb.K[:nb, :nb], cb.M[:nb, :nb]
    if len(rest):
        X = -np.linalg.solve(Kbb[np.ix_(rest, rest)], Kbb[np.ix_(rest, joint)])
        T = np.zeros((nb, len(joint)))
        T[joint] = np.eye(len(joint))
        T[rest] = X
    else:
        T = np.eye(nb)[:, joint]
    Kj, Mj = T.T @ Kbb @ T, T.T @ Mbb @ T
    lam, V = eigh(0.5 * (Kj + Kj.T), 0.5 * (Mj + Mj.T))
    k = int(min(n_modes, len(joint)))
    return V[:, :k], np.sqrt(np.clip(lam[:k], 0.0, None))


@dataclass
class InterfaceReducedBasis:
    """T (n, n_kept + n_modes), reduced K and M, the coordinate transformation S from the Craig-Bampton coordinates,
    and the interface modes used."""
    T: np.ndarray
    K: np.ndarray
    M: np.ndarray
    S: np.ndarray
    interface_modes: np.ndarray
    n_modes: int

    def expand(self, coords):
        return self.T @ np.asarray(coords, float)


def _apply_S(cb, S):
    T = cb.T @ S
    K, M = S.T @ cb.K @ S, S.T @ cb.M @ S
    return T, 0.5 * (K + K.T), 0.5 * (M + M.T)


def jim_craig_bampton(K, M, interface, joint, n_modes, n_joint_modes):
    """Craig-Bampton reduction with joint interface modes: fixed-interface modes (`n_modes`), physical non-joint
    interface DOFs, and `n_joint_modes` joint interface modes replacing the `joint` DOFs (positions within
    `interface`). With n_joint_modes = len(joint) the model equals plain Craig-Bampton."""
    cb = craig_bampton(K, M, interface, n_modes)
    nb, nm = cb.n_interface, cb.n_modes
    joint = np.asarray(joint, int)
    rest = np.setdiff1d(np.arange(nb), joint)
    Psi, _ = joint_interface_modes(cb, joint, n_joint_modes)
    S = np.zeros((nb + nm, len(rest) + Psi.shape[1] + nm))
    S[rest, np.arange(len(rest))] = 1.0
    S[np.ix_(joint, len(rest) + np.arange(Psi.shape[1]))] = Psi
    S[nb:, len(rest) + Psi.shape[1]:] = np.eye(nm)
    T, Kr, Mr = _apply_S(cb, S)
    return InterfaceReducedBasis(T, Kr, Mr, S, Psi, nm)


# ----------------------------------------------------------------------------------------- Gram-Schmidt
def gram_schmidt_interface_modes(mode_sets, interface, tol=1e-8, weights=None, n_keep=None):
    """Interface modes from several mode sets by Gram-Schmidt on their interface rows.

    mode_sets : list of (n, m_i) arrays (e.g. free-interface modes, full-stick modes), in priority order.
    interface : DOF indices. weights : optional positive vector (length n_interface) for the inner product.
    A candidate whose component orthogonal to the accepted vectors is below `tol` times its own norm is dropped
    (it adds no new interface motion). Returns the (n_interface, k) matrix with W-orthonormal columns."""
    b = np.asarray(interface, int)
    w = np.ones(len(b)) if weights is None else np.asarray(weights, float)
    Q = []
    for S in mode_sets:
        for c in np.asarray(S, float)[b].T:
            v = c.copy()
            nv0 = np.sqrt(v @ (w * v))
            if nv0 == 0.0:
                continue
            for _ in range(2):                          # re-orthogonalise once: classical GS loses orthogonality
                for q in Q:
                    v = v - (q @ (w * v)) * q
            nv = np.sqrt(v @ (w * v))
            if nv > tol * nv0:
                Q.append(v / nv)
            if n_keep is not None and len(Q) >= n_keep:
                return np.array(Q).T
    return np.array(Q).T if Q else np.zeros((len(b), 0))


def interface_reduced_craig_bampton(K, M, interface, n_modes, interface_modes):
    """Craig-Bampton model whose interface coordinates are the amplitudes of `interface_modes` (n_interface, k)."""
    cb = craig_bampton(K, M, interface, n_modes)
    Psi = np.asarray(interface_modes, float)
    nb, nm = cb.n_interface, cb.n_modes
    if Psi.shape[0] != nb:
        raise ValueError("interface_modes must have one row per interface DOF")
    S = np.zeros((nb + nm, Psi.shape[1] + nm))
    S[:nb, :Psi.shape[1]] = Psi
    S[nb:, Psi.shape[1]:] = np.eye(nm)
    T, Kr, Mr = _apply_S(cb, S)
    return InterfaceReducedBasis(T, Kr, Mr, S, Psi, nm)


# ----------------------------------------------------------------------------------------- Fourier
def fourier_interface_basis(angles, n_harmonics):
    """Cos/sin basis (n_nodes, 2H+1) of one scalar field on a ring: [1, cos t, sin t, ..., cos Ht, sin Ht],
    columns scaled to unit Euclidean norm for uniformly spaced angles."""
    th = np.asarray(angles, float)
    cols = [np.ones_like(th)]
    for h in range(1, n_harmonics + 1):
        cols += [np.cos(h * th), np.sin(h * th)]
    B = np.array(cols).T
    return B / np.linalg.norm(B, axis=0)


def multistage_fourier_basis(ring_index, angles, n_harmonics, snapshots=None, n_ring_modes=None, tol=1e-10):
    """Multi-stage Fourier reduction of a scalar field on a set of rings.

    ring_index : (n_nodes,) integer ring of each node; angles : (n_nodes,) angle of each node.
    Stage 1: per ring, project onto cos/sin harmonics 0..H. Stage 2: for each harmonic, the coefficients across
    rings, collected over the interface `snapshots` (n_nodes, n_snap), are compressed by an SVD and the leading
    `n_ring_modes` (default: all above `tol`) ring profiles are kept. The basis column for harmonic h and ring
    profile p is  F_h(angle) * u_hp(ring)  (zero if the node's ring is not in the profile). Without snapshots
    stage 2 is skipped and every ring keeps its own coefficient.
    Returns (basis (n_nodes, k), info) where basis columns are Euclidean-orthonormal when angles are uniform on
    every ring."""
    ring_index = np.asarray(ring_index, int)
    angles = np.asarray(angles, float)
    rings = np.unique(ring_index)
    nr = len(rings)
    nodes_of = [np.flatnonzero(ring_index == r) for r in rings]
    F = [fourier_interface_basis(angles[ix], n_harmonics) for ix in nodes_of]    # per ring (n_r_nodes, 2H+1)
    cols, kept = [], []
    if snapshots is not None:
        S = np.asarray(snapshots, float)
        coeff = [np.array([F[j][:, h] @ S[nodes_of[j]] for j in range(nr)]) for h in range(2 * n_harmonics + 1)]
        scale = max(np.linalg.norm(c, 2) for c in coeff)                       # one global scale: noise harmonics drop
    for h in range(2 * n_harmonics + 1):
        if snapshots is None:
            profiles = np.eye(nr)
        else:
            U, s, _ = np.linalg.svd(coeff[h], full_matrices=False)
            k = int((s > tol * max(scale, 1e-300)).sum())
            if n_ring_modes is not None:
                k = min(k, int(n_ring_modes))
            profiles = U[:, :k]
        for p in profiles.T:
            col = np.zeros(len(angles))
            for j in range(nr):
                col[nodes_of[j]] = F[j][:, h] * p[j]
            nrm = np.linalg.norm(col)
            if nrm > 0:
                cols.append(col / nrm)
        kept.append(profiles.shape[1])
    B = np.array(cols).T
    return B, dict(profiles_per_harmonic=kept, n_columns=B.shape[1])


# ----------------------------------------------------------------------------------------- derivatives
def trial_vector_derivative_basis(K, modes, tangent_fn=None, dK=None, M=None, tol=1e-8, max_vectors=None, u0=None):
    """Trial vector derivatives of a mode basis: static modal derivatives (`modal_derivatives`) of `modes`,
    orthogonalised against them and filtered by the Gram eigenvalue cut-off. Returns (basis, info) with
    info['n_added']; M-orthonormal when `M` is given."""
    md = static_modal_derivatives(K, modes, tangent_fn=tangent_fn, dK=dK, u0=u0)
    return augment_with_derivatives(modes, md, M=M, tol=tol, max_vectors=max_vectors)


# ----------------------------------------------------------------------------------------- metrics
def natural_frequency_errors(K, M, T, n=None):
    """Relative error of the lowest `n` natural frequencies of the model reduced by the basis T (n_dof, r)
    against the unreduced pair (K, M)."""
    wf = np.sqrt(np.clip(eigh(K, M, eigvals_only=True), 0, None))
    Kr, Mr = T.T @ K @ T, T.T @ M @ T
    wr = np.sqrt(np.clip(eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T), eigvals_only=True), 0, None))
    n = min(len(wr), len(wf)) if n is None else min(n, len(wr), len(wf))
    return (wr[:n] - wf[:n]) / wf[:n]


def mac_matrix(A, B):
    """Modal assurance criterion between the columns of A and B."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    num = (A.T @ B) ** 2
    return num / np.outer(np.sum(A * A, 0), np.sum(B * B, 0))


def mac_diagonal(K, M, T, n=None):
    """MAC between the lowest full-order modes and the reduced modes expanded back by T (diagonal entries)."""
    wf, Vf = eigh(K, M)
    Kr, Mr = T.T @ K @ T, T.T @ M @ T
    wr, Vr = eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    n = min(Vr.shape[1], Vf.shape[1]) if n is None else min(n, Vr.shape[1])
    return np.diag(mac_matrix(Vf[:, :n], T @ Vr[:, :n]))

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
cms.py -- component mode synthesis: Guyan (static) condensation and Craig-Bampton substructure
reduction, plus coupling of reduced substructures into one reduced model.

WHY. Classic structural-dynamics reduction for assemblies. Each substructure is reduced on its own
(offline, possibly by different teams or from different FE models) while its INTERFACE degrees of
freedom stay physical, so substructures can be joined, replaced or modified without re-reducing the
rest. The package's other methods reduce a whole model at once; this one is about reducing parts.

METHODS.
  * `guyan(K, M, master)`           static condensation: keep `master` DOFs, slave DOFs follow
                                     statically. Exact for static loads on the master DOFs; the mass
                                     is approximate, so frequencies are accurate only for modes much
                                     lower than the first slave-DOF mode.
  * `craig_bampton(K, M, interface, n_modes)`
                                     Craig-Bampton (Hurty/Craig-Bampton fixed-interface method): the
                                     basis is  [ constraint modes | fixed-interface normal modes ].
                                     Reduced coordinates are [interface displacements; modal
                                     amplitudes]. K_r is block-diagonal in the modal part (the
                                     eigenvalues) and M_r has identity there (mass-normalised modes).
  * `couple(subs, interface_maps)`  joins several Craig-Bampton substructures by identifying their
                                     interface DOFs (displacement compatibility), giving one reduced
                                     model with `solve_modal()`.

Everything works on plain arrays; the package's usual rule (never imports fea_engine). Interface DOFs
must be listed in the same physical order in every substructure that shares them (the
`interface_maps` give each substructure's interface DOF a global interface index).

CHECKED IN THE TESTS against full-order eigenfrequencies of a beam split into two substructures,
including the exactness limit (keeping all fixed-interface modes reproduces the full model) and the
expected monotone improvement with the number of retained modes.

References (methods in this module are adopted from):
  - W. C. Hurty (1965), "Dynamic analysis of structural systems using component modes", AIAA J. 3(4)
  - R. R. Craig Jr., M. C. C. Bampton (1968), "Coupling of substructures for dynamic analyses", AIAA J. 6(7)
  - R. H. MacNeal (1971), "A hybrid method of component mode synthesis", Comput. Struct. 1(4)
  - S. Rubin (1975), "Improved component-mode representation for structural dynamic analysis", AIAA J. 13(8)
  - M. P. Castanier, Y.-C. Tan, C. Pierre (2001), "Characteristic constraint modes for component mode synthesis", AIAA J. 39(6)
  - P. Tiso, M. Karamooz Mahdiabadi, J. Marconi (2021), "Modal methods for reduced order modeling", Chapter 4, ETH Research Collection, doi:10.3929/ethz-b-000526692
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


def _arr(x):
    return np.asarray(x, dtype=float)


def _split(n, keep):
    keep = np.asarray(keep, dtype=int)
    if len(set(keep.tolist())) != len(keep) or keep.min(initial=0) < 0 or keep.max(initial=-1) >= n:
        raise ValueError("DOF list must contain distinct indices inside 0..n-1")
    rest = np.setdiff1d(np.arange(n), keep)
    return keep, rest


# ----------------------------------------------------------------------------- Guyan
@dataclass
class GuyanModel:
    T: np.ndarray            # (n, n_master) static transformation  u = T u_master
    K: np.ndarray
    M: np.ndarray
    master: np.ndarray
    slave: np.ndarray

    def expand(self, u_master):
        return self.T @ _arr(u_master)


def guyan(K, M, master):
    """Static (Guyan-Irons) condensation onto the `master` DOFs.

    u = T u_m with T = [I; -K_ss^-1 K_sm] (rows ordered like the original DOFs)."""
    K, M = _arr(K), _arr(M)
    n = K.shape[0]
    master, slave = _split(n, master)
    Ksm = K[np.ix_(slave, master)]
    Psi = -np.linalg.solve(K[np.ix_(slave, slave)], Ksm)
    T = np.zeros((n, len(master)))
    T[master, :] = np.eye(len(master))
    T[slave, :] = Psi
    return GuyanModel(T, T.T @ K @ T, T.T @ M @ T, master, slave)


# ----------------------------------------------------------------------------- Craig-Bampton
@dataclass
class CraigBamptonModel:
    """Reduced substructure. Coordinates: [u_interface (n_b); modal amplitudes eta (n_m)]."""
    T: np.ndarray                 # (n, n_b + n_m) basis: constraint modes then fixed-interface modes
    K: np.ndarray                 # (n_b+n_m)^2
    M: np.ndarray
    interface: np.ndarray         # original DOF indices of the interface (n_b)
    internal: np.ndarray          # original DOF indices of the internal DOFs
    omega_fixed: np.ndarray       # fixed-interface natural frequencies [rad/s] kept
    n_interface: int
    n_modes: int

    def expand(self, coords):
        """Full-order displacement from reduced coordinates."""
        return self.T @ _arr(coords)

    @property
    def freq_fixed_hz(self):
        return self.omega_fixed / (2 * np.pi)


def craig_bampton(K, M, interface, n_modes=None):
    """Craig-Bampton reduction of one substructure.

    Parameters
    ----------
    K, M : (n, n) full-order stiffness and mass (symmetric; K may be singular only through the
        interface, i.e. K_ii must be positive definite).
    interface : indices of boundary (interface) DOFs kept physical.
    n_modes : number of fixed-interface normal modes to keep (default: all).
    """
    K, M = _arr(K), _arr(M)
    n = K.shape[0]
    b, i = _split(n, interface)
    Kii, Kib = K[np.ix_(i, i)], K[np.ix_(i, b)]
    Mii = M[np.ix_(i, i)]
    nm = len(i) if n_modes is None else int(min(n_modes, len(i)))
    w2, Phi = eigh(Kii, Mii)
    if w2[0] <= 0:
        raise ValueError("K_ii is not positive definite: the internal DOFs are not constrained by the "
                         "interface (rigid-body motion left over)")
    Phi = Phi[:, :nm]                         # mass-normalised: Phi^T M_ii Phi = I
    Psi = -np.linalg.solve(Kii, Kib)          # constraint modes
    T = np.zeros((n, len(b) + nm))
    T[b, :len(b)] = np.eye(len(b))
    T[i, :len(b)] = Psi
    T[i, len(b):] = Phi
    Kr = T.T @ K @ T
    Mr = T.T @ M @ T
    # these blocks are known analytically; set them exactly instead of keeping round-off noise
    nb = len(b)
    Kr[:nb, nb:] = 0.0
    Kr[nb:, :nb] = 0.0
    Kr[nb:, nb:] = np.diag(w2[:nm])
    Mr[nb:, nb:] = np.eye(nm)
    return CraigBamptonModel(T, Kr, Mr, b, i, np.sqrt(w2[:nm]), len(b), nm)


# ----------------------------------------------------------------------------- coupling
class CoupledModel:
    """Several Craig-Bampton substructures joined at shared interface DOFs.

    Global coordinates: [global interface DOFs (n_g); modes of substructure 0; modes of 1; ...].
    """

    def __init__(self, subs, interface_maps):
        if len(subs) != len(interface_maps):
            raise ValueError("one interface map per substructure")
        maps = [np.asarray(m, dtype=int) for m in interface_maps]
        for s, m in zip(subs, maps):
            if len(m) != s.n_interface:
                raise ValueError("interface map length must equal the substructure's n_interface")
        self.subs, self.maps = list(subs), maps
        self.n_g = int(max(m.max() for m in maps)) + 1
        self.offsets = np.cumsum([0] + [s.n_modes for s in subs]) + self.n_g
        n_tot = self.n_g + sum(s.n_modes for s in subs)
        K = np.zeros((n_tot, n_tot))
        M = np.zeros((n_tot, n_tot))
        for k, (s, m) in enumerate(zip(subs, maps)):
            # local -> global coordinate index of each reduced coordinate of this substructure
            gidx = np.concatenate([m, self.offsets[k] + np.arange(s.n_modes)])
            K[np.ix_(gidx, gidx)] += s.K
            M[np.ix_(gidx, gidx)] += s.M
        self.K, self.M = K, M

    @property
    def n(self):
        return self.K.shape[0]

    def solve_modal(self, n_modes=6):
        """Lowest natural frequencies [Hz] and shapes in reduced coordinates."""
        w2, X = eigh(self.K, self.M)
        w2 = np.clip(w2, 0.0, None)
        k = min(n_modes, len(w2))
        return np.sqrt(w2[:k]) / (2 * np.pi), X[:, :k]

    def local_coords(self, k, x):
        """Reduced coordinates of substructure k extracted from a global reduced vector x."""
        s = self.subs[k]
        gidx = np.concatenate([self.maps[k], self.offsets[k] + np.arange(s.n_modes)])
        return x[gidx]

    def expand(self, k, x):
        """Full-order displacement of substructure k for a global reduced vector x."""
        return self.subs[k].expand(self.local_coords(k, x))


def couple(subs, interface_maps):
    """Join Craig-Bampton substructures; see `CoupledModel`."""
    return CoupledModel(subs, interface_maps)


# ----------------------------------------------------------------------------- 
# Free-interface methods and interface reduction (Tiso, Karamooz Mahdiabadi and Marconi 2021, Section 4.4;
# Rubin 1975; MacNeal 1971; Castanier, Tan and Pierre 2001 for characteristic constraint modes).
@dataclass
class FreeInterfaceModel:
    """Free-interface reduction of one substructure. Coordinates: [modal amplitudes (n_modes); residual-
    flexibility amplitudes (n_interface)] for method 'rubin', modal amplitudes only for 'macneal' (residual
    inertia neglected and the residual coordinates condensed out)."""
    T: np.ndarray
    K: np.ndarray
    M: np.ndarray
    method: str
    omega_free: np.ndarray       # free-interface frequencies [rad/s] kept
    interface: np.ndarray
    n_modes: int

    def expand(self, coords):
        return self.T @ _arr(coords)


def free_interface_cms(K, M, interface, n_modes, method="rubin"):
    """Free-interface component mode synthesis with residual flexibility.

    The retained free-interface modes Phi_k (M-orthonormal) are enriched by the RESIDUAL ATTACHMENT modes of the
    interface: columns of  G_r = K^-1 - Phi_k diag(1/omega^2) Phi_k^T  at the interface DOFs, which carry the static
    flexibility of the truncated modes. 'rubin' (Rubin 1975) keeps the residual modes' inertia; 'macneal' (MacNeal
    1971) neglects it and condenses the residual coordinates statically.

    Requires a non-singular K (the substructure must be restrained; free rigid-body motion is not handled here).
    """
    K, M = _arr(K), _arr(M)
    n = K.shape[0]
    b, _ = _split(n, interface)
    w2, Phi = eigh(K, M)
    if w2[0] <= 1e-12 * w2[-1]:
        raise ValueError("K is singular (rigid-body modes): restrain the substructure first")
    nm = int(min(n_modes, n))
    Pk, wk2 = Phi[:, :nm], w2[:nm]
    G = np.linalg.inv(K) - (Pk / wk2) @ Pk.T
    R = G[:, b]
    # residual flexibility vanishes when all modes are kept: drop (numerically) null residual directions
    U, sv, _ = np.linalg.svd(R, full_matrices=False)
    keep = sv > 1e-9 * max(float(np.linalg.norm(np.linalg.inv(K), 2)), 1e-300)
    R = U[:, keep] * sv[keep]
    T = np.hstack([Pk, R])
    Kr, Mr = T.T @ K @ T, T.T @ M @ T
    omega = np.sqrt(wk2)
    if method == "rubin":
        return FreeInterfaceModel(T, 0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T), "rubin", omega, b, nm)
    if method == "macneal":
        k, r = slice(0, nm), slice(nm, None)
        Kc = Kr[k, k] - Kr[k, r] @ np.linalg.solve(Kr[r, r], Kr[r, k])
        X = np.vstack([np.eye(nm), -np.linalg.solve(Kr[r, r], Kr[r, k])])      # static elimination of residuals
        Mc = X.T @ np.block([[Mr[k, k], np.zeros((nm, R.shape[1]))],
                             [np.zeros((R.shape[1], nm)), np.zeros((R.shape[1],) * 2)]]) @ X
        return FreeInterfaceModel(T @ X, 0.5 * (Kc + Kc.T), 0.5 * (Mc + Mc.T), "macneal", omega, b, nm)
    raise ValueError("method must be 'rubin' or 'macneal'")


@dataclass
class InterfaceReducedModel:
    """Craig-Bampton model whose interface coordinates were replaced by characteristic constraint modes."""
    T: np.ndarray
    K: np.ndarray
    M: np.ndarray
    S: np.ndarray                # (n_b + n_m, n_keep + n_m) coordinate transformation
    omega_interface: np.ndarray  # frequencies of the interface (constraint-mode) eigenproblem [rad/s]
    n_interface_kept: int
    n_modes: int

    def expand(self, coords):
        return self.T @ _arr(coords)


def reduce_interface(cb, n_keep):
    """Interface reduction by characteristic constraint modes (Castanier, Tan and Pierre 2001): solve the
    eigenproblem of the constraint-mode block, K_bb x = lambda M_bb x, keep the `n_keep` lowest, and replace the
    physical interface coordinates by their amplitudes. With n_keep = n_interface the model is unchanged
    (exact change of variables). Returns an InterfaceReducedModel; because the coordinates are no longer physical
    interface displacements it is used standalone, not with `couple`."""
    nb, nm = cb.n_interface, cb.n_modes
    n_keep = int(min(n_keep, nb))
    lam, X = eigh(cb.K[:nb, :nb], cb.M[:nb, :nb])
    S = np.zeros((nb + nm, n_keep + nm))
    S[:nb, :n_keep] = X[:, :n_keep]
    S[nb:, n_keep:] = np.eye(nm)
    K = S.T @ cb.K @ S
    M = S.T @ cb.M @ S
    return InterfaceReducedModel(cb.T @ S, 0.5 * (K + K.T), 0.5 * (M + M.T), S, np.sqrt(np.clip(lam, 0, None)),
                                 n_keep, nm)

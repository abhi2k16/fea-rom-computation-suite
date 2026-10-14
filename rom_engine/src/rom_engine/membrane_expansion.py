# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
membrane_expansion.py -- membrane-basis estimation and expansion for implicit condensation and expansion (ICE) ROMs
(Hollkamp and Gordon 2008, Sec. 2.2, Eqs. 9-16, Table 1 steps 6-8), which `nonlinear_rom.py` does not provide (its
module docstring flags: "(e.g. membrane augmentation) this module does not yet implement").

Why it exists: a bending-only ICE-ROM predicts the physical transverse displacement via `w_b = Phi_b @ p` (Eq. 8)
using ONLY the retained bending modes. For a straight (flat) beam or plate, those bending mode shapes have ZERO
in-plane/axial component by construction (a linear bending mode is, to first order, a pure transverse shape), so
`w_b`'s in-plane prediction is identically zero. The expansion step supplies that missing membrane component, which
fiber-level stress recovery (`fea_engine.elements.beams.Beam2DCorotational.recover_stress`) needs: a membrane stress
cannot be recovered from a displacement field that has no membrane component.

Generalises an n_modes=1-only prototype to any number of retained modes, by reusing
`nonlinear_rom._monomial_indices(n_modes, 2)`, the same quadratic-monomial enumeration `PolynomialModalROM` uses for
its own `quad_idx`: Eq. 16's "quadratic combinations of modal bending amplitudes" (p1^2, p1*p2, p2^2, ...) is that
same combinatorial object, used here to predict a MEMBRANE SHAPE coefficient instead of a nonlinear force.

Orientation convention, matching every other array this package hands
around: `q_nl`/`p_hist` are (n_samples_or_steps, n_modes) -- SAMPLES-
major, matching `AppliedLoadStrategy.generate()`'s own q_nl_samples and
`nonlinear_dynamics.integrate_newmark_surrogate()`'s own q_nl_hist/
q_l_hist returns, so a caller can pass either straight through with no
transpose. `W`/`V` stay DOF-major ((n_free, n_samples) / (n_free,
n_modes)), matching every mode-shape and snapshot-matrix convention
already established elsewhere in this package (`pod.py`, `galerkin.py`,
`AppliedLoadStrategy`'s own `V` parameter).

References (methods in this module are adopted from):
  - S. A. Hollkamp, R. W. Gordon (2008), "Reduced-order models for nonlinear response prediction: Implicit condensation and expansion", J. Sound Vib. 318(4-5)
  - M. P. Mignolet, A. Przekop, S. A. Rizzi, S. M. Spottswood (2013), "A review of indirect/non-intrusive reduced order modeling of nonlinear geometric structures", J. Sound Vib. 332(10):2437-2460
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .nonlinear_rom import _monomial_indices


class MembraneBasis:
    """Eq. 13-16: estimates a membrane basis `T_m` from the SAME static
    training data already generated for a bending-only ICE-ROM fit
    (`AppliedLoadStrategy.generate(..., return_snapshots=True)`), then
    expands a modal bending-amplitude time history into the physical
    membrane displacement time history the bending-only prediction
    structurally cannot produce.

    Not a `ReducedForceModel` -- this is a POST-PROCESSING/expansion
    step layered on top of an already-fit `PolynomialModalROM`/
    `MultiFidelitySurrogate`'s own bending prediction, not an
    alternative regression of the nonlinear restoring force itself
    (Table 1 of Hollkamp & Gordon treats bending-force fitting, steps 2-4,
    and membrane-basis estimation, steps 6-8, as two SEPARATE stages
    for exactly this reason).
    """

    def __init__(self):
        self.Tm = None          # (n_quad_terms, n_free), set by fit()
        self.quad_idx = None    # list of (i, j) monomial index pairs
        self.n_modes = None
        self.n_free = None

    def fit(self, V, q_nl, W):
        """Eq. 15: `T_m ~= [W - Phi_b @ P] @ Q^+`, generalized to any
        `n_modes` via `_monomial_indices(n_modes, 2)` for Eq. 16's own
        quadratic-combination design matrix `Q`.

        Parameters
        ----------
        V : ndarray, shape (n_free, n_modes)
            Mass-normalized retained BENDING mode shapes (`Phi_b` in
            Hollkamp & Gordon's notation) -- the SAME `V` a bending-only
            `PolynomialModalROM`/`MultiFidelitySurrogate` was fit
            against.
        q_nl : ndarray, shape (n_samples, n_modes)
            The TRUE (projected, not linear-estimate) nonlinear modal
            bending amplitudes at each training sample -- `p` in the
            paper's Eq. 13-16, and exactly the same `q_nl_samples`
            array `PolynomialModalROM.fit()` itself consumes.
        W : ndarray, shape (n_free, n_samples)
            The FULL free-DOF displacement snapshots at those same
            training samples -- `AppliedLoadStrategy.generate(...,
            return_snapshots=True)`'s new fourth return value (item
            112's own upstream change to that method).

        Returns
        -------
        self (chainable, matching `PodBasis.fit()`/`GalerkinROM.
        reduce_system()`'s own convention).
        """
        V = np.asarray(V, dtype=float)
        q_nl = np.asarray(q_nl, dtype=float)
        W = np.asarray(W, dtype=float)
        n_free, n_modes = V.shape
        if q_nl.ndim != 2 or q_nl.shape[1] != n_modes:
            raise ValueError(
                f"q_nl must be (n_samples, n_modes={n_modes}), got shape {q_nl.shape}")
        if W.shape[0] != n_free:
            raise ValueError(f"W must have {n_free} rows (n_free, matching V), got {W.shape}")
        if W.shape[1] != q_nl.shape[0]:
            raise ValueError(
                f"W and q_nl must agree on n_samples, got W.shape[1]={W.shape[1]} "
                f"and q_nl.shape[0]={q_nl.shape[0]}")

        self.n_modes = n_modes
        self.n_free = n_free
        self.quad_idx = _monomial_indices(n_modes, 2)

        residual = W - V @ q_nl.T                        # (n_free, n_samples), Eq. 14 LHS target
        Q = np.array([q_nl[:, i] * q_nl[:, j]             # (n_quad_terms, n_samples), Eq. 16
                      for (i, j) in self.quad_idx])
        Tm, *_ = np.linalg.lstsq(Q.T, residual.T, rcond=None)   # (n_quad_terms, n_free)
        self.Tm = Tm
        return self

    def expand(self, p_hist):
        """Eq. 16 + Eq. 11's expansion term: given a time (or sample)
        history of modal bending amplitudes, returns the physical
        membrane displacement history `T_m.T @ q(t)`.

        Parameters
        ----------
        p_hist : ndarray, shape (n_steps, n_modes)
            Modal bending-amplitude history -- e.g. `q_nl_hist` (or,
            with the usual `q_l`-vs-`q_nl` caveat this whole package
            already documents elsewhere, `q_l_hist`) straight from
            `nonlinear_dynamics.integrate_newmark_surrogate()`.

        Returns
        -------
        w_m_hist : ndarray, shape (n_steps, n_free)
            Physical membrane displacement history, SAME (time-major)
            orientation as `p_hist` itself.
        """
        if self.Tm is None:
            raise RuntimeError("MembraneBasis.expand() called before fit()")
        p_hist = np.asarray(p_hist, dtype=float)
        if p_hist.ndim != 2 or p_hist.shape[1] != self.n_modes:
            raise ValueError(
                f"p_hist must be (n_steps, n_modes={self.n_modes}), got shape {p_hist.shape}")
        Q_hist = np.column_stack([p_hist[:, i] * p_hist[:, j]   # (n_steps, n_quad_terms)
                                   for (i, j) in self.quad_idx])
        return Q_hist @ self.Tm                                 # (n_steps, n_free)

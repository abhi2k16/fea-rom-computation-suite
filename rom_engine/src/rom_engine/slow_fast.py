# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
slow_fast.py -- slow-fast static condensation of the in-plane (membrane) coordinates of a
geometrically nonlinear thin structure against its bending coordinates. After Tiso, Karamooz Mahdiabadi and
Marconi (2021), Section 4.6 (membrane/bending separation of time scales, with the membrane DOFs treated as
quasi-static).

THE IDEA. Split the reduced state into slow coordinates q (bending, from vibration modes) and fast coordinates p
(membrane, from modal derivatives or dual modes):   u = Vs q + Vf p.  A thin structure's membrane response is much
stiffer than its bending response, so p follows q almost statically. Setting the fast equations to their static
balance,
        Vf^T ( f_int(Vs q + Vf p) - f_ext ) = 0     =>     p = p(q),
removes p from the dynamics, leaving only the slow coordinates:
        slow force   g(q) = Vs^T f_int(Vs q + Vf p(q)),
        slow tangent K~(q) = Vs^T K_T V_s - Vs^T K_T Vf (Vf^T K_T Vf)^-1 Vf^T K_T Vs,   (a Schur complement).
The condensation is EXACT with respect to the full reduced system [Vs Vf] when the fast equations are solved
exactly (the same algebra as the Galerkin solve, in a different order); what it buys is a smaller stiffness-
dominated problem for the slow coordinates and a larger stable time step when the fast dynamics are what limits it.
Nothing in this module claims a speed-up: that is measured on a given model.

WHAT IS NOT CLAIMED. The inertia of the fast coordinates is dropped (quasi-static assumption); it is valid when
the membrane frequencies are far above the slow ones. `fast_response` reports Newton convergence so a failed
condensation is visible, never silent.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.optimize import root

from .modal_derivatives import _as_array


class SlowFastROM:
    """Condensed slow-coordinate model. `f_int(u)` and `tangent_fn(u)` are the full-order callables; `Vs` and `Vf`
    the slow and fast bases (n, ms) and (n, mf)."""

    def __init__(self, Vs, Vf, f_int, tangent_fn):
        self.Vs, self.Vf = _as_array(Vs), _as_array(Vf)
        self.f_int, self.tangent_fn = f_int, tangent_fn
        self.n_fast_solves = 0

    def fast_response(self, q, f_ext=None, p0=None):
        """p(q): Newton solve of  Vf^T (f_int(Vs q + Vf p) - f_ext) = 0.  Returns p."""
        Vs, Vf = self.Vs, self.Vf
        f_ext = 0.0 if f_ext is None else _as_array(f_ext)
        p0 = np.zeros(Vf.shape[1]) if p0 is None else p0
        p = np.array(p0, dtype=float)
        # residual scale: size of the terms that cancel (internal force and load), so round-off is not mistaken
        # for non-convergence when the membrane load is nearly zero
        scale = max(np.linalg.norm(self.f_int(Vs @ q + Vf @ p)), np.linalg.norm(f_ext) if np.ndim(f_ext) else 0.0,
                    1e-300)
        self.n_fast_solves += 1
        for _ in range(50):
            r = Vf.T @ (self.f_int(Vs @ q + Vf @ p) - f_ext)
            if np.linalg.norm(r) <= 1e-9 * scale:
                return p
            J = Vf.T @ self.tangent_fn(Vs @ q + Vf @ p) @ Vf
            p = p - np.linalg.solve(J, r)
        raise RuntimeError("fast (membrane) Newton solve did not converge")

    def state(self, q, f_ext=None):
        q = _as_array(q)
        return self.Vs @ q + self.Vf @ self.fast_response(q, f_ext)

    def reduced_force(self, q, f_ext=None):
        q = _as_array(q)
        return self.Vs.T @ self.f_int(self.state(q, f_ext))

    def reduced_tangent(self, q, f_ext=None):
        q = _as_array(q)
        Vs, Vf = self.Vs, self.Vf
        K = self.tangent_fn(self.state(q, f_ext))
        Kss, Ksf, Kff = Vs.T @ K @ Vs, Vs.T @ K @ Vf, Vf.T @ K @ Vf
        return Kss - Ksf @ np.linalg.solve(Kff, Ksf.T)

    def static_solve(self, f_ext, q0=None):
        """Slow-coordinate static solve  Vs^T (f_int(u(q)) - f_ext) = 0  with p(q) condensed out."""
        f_ext = _as_array(f_ext)
        q0 = np.zeros(self.Vs.shape[1]) if q0 is None else q0
        sol = root(lambda q: self.Vs.T @ (self.f_int(self.state(q, f_ext)) - f_ext), q0,
                   jac=lambda q: self.reduced_tangent(q, f_ext), method="hybr", tol=1e-12)
        if not sol.success:
            raise RuntimeError(f"slow static solve did not converge: {sol.message}")
        return sol.x

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
forced_response_hbm.py -- forced, damped multi-harmonic response of a reduced model with a local nonlinearity,
traced over excitation frequency by pseudo-arclength continuation.

The existing `nnm.py` continuation is for UNFORCED, undamped orbits (its docstring says so). This module is the
different problem: M q'' + C q' + K q + f_nl(q) = F(t) with a given harmonic forcing, damping, and the response
followed along the frequency axis through folds (hardening/softening resonance peaks). It uses the same real
trigonometric representation as `nnm.HarmonicBalanceSystem`: Z has shape (1 + 2H, n), Z[0] the mean, Z[2k-1],
Z[2k] the cos/sin coefficients of harmonic k.

    R(Z, w) = A(w) Z + T^-1 f_nl(T Z) - F = 0,   A_k(w) = [[K - (kw)^2 M,  k w C], [-k w C, K - (kw)^2 M]].

`f_nl` is any object with `force(q_time) -> (NT, n)` and `jacobian(q_time) -> (NT*n, NT*n)` on the uniform grid of
one period (e.g. `friction_contact_models.JenkinsContact`, or `PolynomialForce` here). The Jacobian is exact
when the model supplies it. The continuation is a secant predictor with a Newton corrector on the hyperplane
orthogonal to the secant (Keller), step size adapted to the corrector's behaviour, on scaled variables.

Scope, stated honestly: dense linear algebra (reduced models of tens of DOFs, a few harmonics), full (non
diagonal) M, C, K allowed, one forcing frequency (no quasi-periodic response) and no subharmonics; stability
(Floquet) is not computed, so unstable branches are traced but not labelled.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["ForcedHBM", "PolynomialForce", "ForcedResponse", "forced_response_continuation",
           "equivalent_resonance", "equivalent_properties_vs_force"]


class PolynomialForce:
    """f_nl = sum over terms (dof, coefficient, power) of coefficient * q[dof]**power on that DOF (e.g. a cubic
    spring). The exact Jacobian is diagonal in time."""

    def __init__(self, n_dof, terms):
        self.n = int(n_dof)
        self.terms = [(int(d), float(c), int(p)) for d, c, p in terms]

    def force(self, q_time):
        q = np.asarray(q_time, float)
        out = np.zeros_like(q)
        for d, c, p in self.terms:
            out[:, d] += c * q[:, d] ** p
        return out

    def jacobian(self, q_time):
        q = np.asarray(q_time, float)
        NT, n = q.shape
        diag = np.zeros((NT, n))
        for d, c, p in self.terms:
            diag[:, d] += c * p * q[:, d] ** (p - 1)
        return np.diag(diag.reshape(-1))


class ForcedHBM:
    """Harmonic-balance operator of M q'' + C q' + K q = F with H harmonics (+ mean) and NT time samples."""

    def __init__(self, M, C, K, n_harmonics=3, n_time_samples=None):
        self.M, self.C, self.K = (np.asarray(x, float) for x in (M, C, K))
        self.n = self.M.shape[0]
        self.H = int(n_harmonics)
        if self.H < 1:
            raise ValueError("n_harmonics must be >= 1")
        self.NC = 1 + 2 * self.H
        NT = n_time_samples or max(8 * self.H + 8, 32)
        NT = max(NT, 2 * self.H + 1)
        NT += NT % 2
        self.NT = NT
        theta = np.arange(NT) * 2 * np.pi / NT
        T = np.zeros((NT, self.NC))
        Ti = np.zeros((self.NC, NT))
        T[:, 0] = 1.0
        Ti[0] = 1.0 / NT
        for k in range(1, self.H + 1):
            T[:, 2 * k - 1], T[:, 2 * k] = np.cos(k * theta), np.sin(k * theta)
            Ti[2 * k - 1], Ti[2 * k] = 2 / NT * np.cos(k * theta), 2 / NT * np.sin(k * theta)
        self.T, self.Ti = T, Ti
        self._Tk = np.kron(T, np.eye(self.n))          # coefficients -> time samples on flattened (c*n+m) -> (t*n+m)
        self._Tik = np.kron(Ti, np.eye(self.n))

    # ------------------------------------------------------------------ linear part
    def A(self, w):
        n, NC = self.n, self.NC
        A = np.zeros((NC * n, NC * n))
        A[:n, :n] = self.K
        for k in range(1, self.H + 1):
            ic, is_ = slice((2 * k - 1) * n, 2 * k * n), slice(2 * k * n, (2 * k + 1) * n)
            Kk = self.K - (k * w) ** 2 * self.M
            A[ic, ic] = Kk
            A[is_, is_] = Kk
            A[ic, is_] = k * w * self.C
            A[is_, ic] = -k * w * self.C
        return A

    def dA_dw(self, w):
        n, NC = self.n, self.NC
        dA = np.zeros((NC * n, NC * n))
        for k in range(1, self.H + 1):
            ic, is_ = slice((2 * k - 1) * n, 2 * k * n), slice(2 * k * n, (2 * k + 1) * n)
            d = -2 * k * k * w * self.M
            dA[ic, ic] = d
            dA[is_, is_] = d
            dA[ic, is_] = k * self.C
            dA[is_, ic] = -k * self.C
        return dA

    # ------------------------------------------------------------------ nonlinear part and residual
    def nl_coefficients(self, z, force_model):
        q_time = (self._Tk @ z).reshape(self.NT, self.n)
        f_time = force_model.force(q_time)
        return self._Tik @ f_time.reshape(-1), q_time

    def residual(self, z, w, F, force_model=None):
        r = self.A(w) @ z - F.reshape(-1)
        if force_model is not None:
            r = r + self.nl_coefficients(z, force_model)[0]
        return r

    def jacobian_z(self, z, w, force_model=None):
        J = self.A(w)
        if force_model is not None:
            q_time = (self._Tk @ z).reshape(self.NT, self.n)
            J = J + self._Tik @ force_model.jacobian(q_time) @ self._Tk
        return J

    def harmonic_amplitude(self, z, dof, k=1):
        Z = z.reshape(self.NC, self.n)
        return float(np.hypot(Z[2 * k - 1, dof], Z[2 * k, dof]))

    def forcing(self, f_cos, f_sin=None, harmonic=1):
        """Excitation coefficient array (NC, n): f_cos cos(k w t) + f_sin sin(k w t) at the given harmonic."""
        F = np.zeros((self.NC, self.n))
        F[2 * harmonic - 1] = np.asarray(f_cos, float)
        if f_sin is not None:
            F[2 * harmonic] = np.asarray(f_sin, float)
        return F


class ForcedResponse:
    """omega (n_pts,), Z (n_pts, NC, n), amplitude (n_pts,) of the monitored DOF's first harmonic, plus the number
    of corrector iterations per point and whether the run reached the end of the range."""

    def __init__(self, omega, Z, amplitude, iterations, completed, dof):
        self.omega, self.Z, self.amplitude = omega, Z, amplitude
        self.iterations, self.completed, self.dof = iterations, completed, dof


def _newton(f_res, f_jac, x0, tol, max_iter):
    """Newton with backtracking on the residual norm (stick/slip switching makes plain Newton cycle)."""
    x = x0.copy()
    r = f_res(x)
    for it in range(max_iter):
        if np.linalg.norm(r) < tol:
            return x, True, it
        dx = np.linalg.solve(f_jac(x), r)
        lam = 1.0
        while True:
            xn = x - lam * dx
            rn = f_res(xn)
            if np.linalg.norm(rn) < (1 - 1e-4 * lam) * np.linalg.norm(r) or lam < 1e-3:
                break
            lam *= 0.5
        x, r = xn, rn
    return x, np.linalg.norm(r) < tol, max_iter


def forced_response_continuation(hbm, force_model, F, w_start, w_end, dof=0, ds=None, ds_min=None, ds_max=None,
                                 tol=1e-9, max_points=2000, max_corrector=12):
    """Trace the forced response from w_start to w_end (either order) by pseudo-arclength continuation.

    F : (NC, n) excitation coefficients (see `ForcedHBM.forcing`). `dof` : DOF whose first-harmonic amplitude is
    recorded. ds : initial arclength in the scaled variables (default 0.02). The run stops at the first point
    that passes w_end (that point is clipped by one extra solve at w_end when possible) or after `max_points`.
    Folds are passed because the corrector is orthogonal to the secant, not constrained in w."""
    w_scale = float(w_start if w_start != 0 else 1.0)
    # start: Newton at fixed w, with a force-level homotopy (0.05 ... 1) so that stick-slip switching cannot derail it
    F = np.asarray(F, float)
    z0 = np.linalg.solve(hbm.A(w_start), 0.05 * F.reshape(-1))
    for s in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
        fixed = lambda z, s=s: hbm.residual(z, w_start, s * F, force_model)
        z0, ok, _ = _newton(fixed, lambda z, s=s: hbm.jacobian_z(z, w_start, force_model), z0,
                            tol * max(1, np.linalg.norm(F)), 40)
        if not ok:
            raise RuntimeError("Newton did not converge at the starting frequency")
    z_scale = max(np.linalg.norm(z0), 1e-12)
    direction = 1.0 if w_end >= w_start else -1.0
    ds = 0.02 if ds is None else ds
    ds_min = ds / 1e3 if ds_min is None else ds_min
    ds_max = ds * 50 if ds_max is None else ds_max
    nz = z0.size

    def to_y(z, w):
        return np.concatenate([z / z_scale, [w / w_scale]])

    def res_y(y):
        return hbm.residual(y[:nz] * z_scale, y[nz] * w_scale, F, force_model) / z_scale

    # tangent at the start: null vector of the augmented Jacobian, oriented along +direction in w
    def tangent(y, prev=None):
        t = np.linalg.svd(jac_y(y))[2][-1]
        ref = prev if prev is not None else np.concatenate([np.zeros(nz), [direction]])
        return t if t @ ref >= 0 else -t

    y = to_y(z0, w_start)

    def jac_y(y):                                                      # exact d(res_y)/dy
        z, w = y[:nz] * z_scale, y[nz] * w_scale
        Jz = hbm.jacobian_z(z, w, force_model)                        # d r / d z
        Jw = hbm.dA_dw(w) @ z                                          # d r / d w
        return np.hstack([Jz, (Jw * w_scale / z_scale)[:, None]])
    t = tangent(y)
    omegas, Zs, amps, its = [w_start], [z0.copy()], [hbm.harmonic_amplitude(z0, dof)], [0]
    completed = False
    for _ in range(max_points):
        accepted = False
        while not accepted:
            y_pred = y + ds * t
            def aug_res(yy):
                return np.concatenate([res_y(yy), [t @ (yy - y_pred)]])

            def aug_jac(yy):
                return np.vstack([jac_y(yy), t[None, :]])
            try:
                yn, ok, it = _newton(aug_res, aug_jac, y_pred, tol, max_corrector)
            except np.linalg.LinAlgError:
                ok, it = False, max_corrector
            if ok:
                accepted = True
            else:
                ds *= 0.5
                if ds < ds_min:
                    return ForcedResponse(np.array(omegas), np.array(Zs).reshape(len(Zs), hbm.NC, hbm.n),
                                          np.array(amps), np.array(its), False, dof)
        t_new = tangent(yn, t)
        y, t = yn, t_new
        z, w = y[:nz] * z_scale, y[nz] * w_scale
        omegas.append(w)
        Zs.append(z.copy())
        amps.append(hbm.harmonic_amplitude(z, dof))
        its.append(it)
        if it <= 3:
            ds = min(ds * 1.5, ds_max)
        elif it >= 7:
            ds *= 0.6
        if (w - w_end) * direction >= 0:
            completed = True
            break
    return ForcedResponse(np.array(omegas), np.array(Zs).reshape(len(Zs), hbm.NC, hbm.n), np.array(amps),
                          np.array(its), completed, dof)


def equivalent_resonance(omega, amplitude):
    """Equivalent resonance frequency, peak amplitude and half-power damping ratio of one response curve.

    The peak is the maximum sampled amplitude; the half-power points are the first crossings of peak/sqrt(2)
    on each side, found by linear interpolation along the traced curve (which may fold, so the crossings are
    searched outward from the peak along the point sequence, not in sorted frequency). zeta = (w2 - w1) / (2 w_peak).
    Returns None for the damping if the curve does not reach the half-power level on both sides."""
    omega, amplitude = np.asarray(omega, float), np.asarray(amplitude, float)
    ip = int(np.argmax(amplitude))
    level = amplitude[ip] / np.sqrt(2.0)

    def crossing(seq):
        for a, b in zip(seq[:-1], seq[1:]):
            if (amplitude[a] - level) * (amplitude[b] - level) <= 0 and amplitude[a] != amplitude[b]:
                s = (level - amplitude[a]) / (amplitude[b] - amplitude[a])
                return omega[a] + s * (omega[b] - omega[a])
        return None
    lo = crossing(list(range(ip, -1, -1)))
    hi = crossing(list(range(ip, len(omega))))
    zeta = None if lo is None or hi is None else float(abs(hi - lo) / (2 * omega[ip]))
    return dict(omega_peak=float(omega[ip]), amplitude_peak=float(amplitude[ip]), zeta=zeta, omega_low=lo,
                omega_high=hi)


def equivalent_properties_vs_force(hbm, force_model, F_unit, levels, w_start, w_end, dof=0, **kwargs):
    """Equivalent resonance frequency and damping versus response amplitude: run the continuation for each force
    level (F = level * F_unit) and apply `equivalent_resonance`. Returns a list of dicts (one per level) with the
    level added."""
    out = []
    for lev in levels:
        resp = forced_response_continuation(hbm, force_model, lev * np.asarray(F_unit, float), w_start, w_end, dof=dof,
                                            **kwargs)
        r = equivalent_resonance(resp.omega, resp.amplitude)
        r["level"] = float(lev)
        r["completed"] = resp.completed
        out.append(r)
    return out

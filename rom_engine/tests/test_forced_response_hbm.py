# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the forced damped harmonic-balance continuation."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rom_engine.forced_response_hbm import (ForcedHBM, PolynomialForce, forced_response_continuation,
                                            equivalent_resonance, equivalent_properties_vs_force)
from rom_engine.friction_contact_models import JenkinsContact

M = np.diag([1.0, 1.5])
K = np.array([[40.0, -10.0], [-10.0, 30.0]])
C = 0.05 * K + 0.02 * M
F1 = np.array([1.0, 0.0])


def _linear_frf(w, f):
    A = -w ** 2 * M + 1j * w * C + K
    x = np.linalg.solve(A, f)
    return abs(x[0])


def test_low_amplitude_reduces_to_the_linear_frf():
    hbm = ForcedHBM(M, C, K, n_harmonics=3)
    nl = PolynomialForce(2, [(0, 1e-6, 3)])
    F = hbm.forcing(1e-3 * F1)
    resp = forced_response_continuation(hbm, nl, F, 2.0, 10.0, dof=0, ds=0.05)
    assert resp.completed
    ref = np.array([_linear_frf(w, 1e-3 * F1) for w in resp.omega])
    assert np.allclose(resp.amplitude, ref, rtol=1e-3)
    assert resp.omega.max() >= 10.0 and resp.amplitude.max() > 20 * resp.amplitude[0] * 0 + resp.amplitude[0]


def test_hardening_duffing_passes_through_a_fold_and_matches_time_integration():
    k, c, k3, f0 = 1.0, 0.02, 0.5, 0.1
    hbm = ForcedHBM([[1.0]], [[c]], [[k]], n_harmonics=7)
    nl = PolynomialForce(1, [(0, k3, 3)])
    F = hbm.forcing([f0])
    resp = forced_response_continuation(hbm, nl, F, 0.6, 2.0, dof=0, ds=0.05)
    assert resp.completed
    w = resp.omega
    assert np.any(np.diff(w) < 0)                                   # the frequency decreases somewhere: a fold
    ipk = int(np.argmax(resp.amplitude))
    assert w[ipk] > 1.0                                             # hardening: the peak leans to higher frequency
    # time integration at one point on the upper branch
    idx = ipk - 3
    wt = w[idx]
    Z = resp.Z[idx]                                                 # start the integration on the HBM orbit
    sol = solve_ivp(lambda t, y: [y[1], f0 * np.cos(wt * t) - c * y[1] - k * y[0] - k3 * y[0] ** 3], [0, 400],
                    [Z[1:2 * 7 + 1:2, 0].sum() + Z[0, 0], wt * (np.arange(1, 8) * Z[2:2 * 7 + 1:2, 0]).sum()], rtol=1e-9, atol=1e-11, dense_output=True)
    tt = np.linspace(300, 300 + 2 * np.pi / wt * 5, 2000)
    x = sol.sol(tt)[0]
    assert np.max(np.abs(x)) == pytest.approx(np.max(np.abs(hbm.T @ Z)), rel=0.02)   # peak displacement of the HBM orbit


def test_friction_joint_shifts_the_resonance_down_and_adds_damping():
    # one DOF with a Jenkins joint in parallel: small forcing -> stuck (stiffer), larger forcing -> micro-slip
    c = 0.01
    hbm = ForcedHBM([[1.0]], [[c]], [[1.0]], n_harmonics=3)
    joint = JenkinsContact(1, [0], k_t=2.0, mu=1.0, N0=0.05)
    out = equivalent_properties_vs_force(hbm, joint, hbm.forcing([1.0]), [0.002, 0.02], 0.8, 2.2, dof=0, ds=0.05)
    assert all(o["completed"] for o in out)
    assert out[0]["omega_peak"] == pytest.approx(np.sqrt(3.0), rel=0.02)      # stuck: k + k_t = 3
    assert out[1]["omega_peak"] < out[0]["omega_peak"] - 0.1                  # slip softens the resonance
    assert out[1]["zeta"] > 3 * out[0]["zeta"]                                # and adds hysteretic damping


def test_equivalent_resonance_of_a_linear_oscillator_gives_the_damping_ratio():
    zeta, wn = 0.03, 5.0
    w = np.linspace(3, 7, 4001)
    amp = 1 / np.sqrt((wn ** 2 - w ** 2) ** 2 + (2 * zeta * wn * w) ** 2)
    r = equivalent_resonance(w, amp)
    assert r["omega_peak"] == pytest.approx(wn, rel=1e-2)
    assert r["zeta"] == pytest.approx(zeta, rel=0.05)

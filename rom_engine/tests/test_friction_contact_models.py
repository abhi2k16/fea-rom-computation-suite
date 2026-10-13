# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the Jenkins macro-slip contact element: closed-form loops and Jacobian against finite differences."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.friction_contact_models import JenkinsContact, jenkins_hysteresis_energy

NT = 64
TH = np.arange(NT) * 2 * np.pi / NT


def test_full_stick_is_a_linear_spring():
    j = JenkinsContact(1, [0], k_t=100.0, mu=1.0, N0=1e6)
    v = 0.01 * np.cos(TH)
    f = j.force(v[:, None])[:, 0]
    assert np.allclose(f, 100.0 * v, atol=1e-12)
    assert abs(jenkins_hysteresis_energy(v, f)) < 1e-12


def test_gross_slip_closed_form_loop():
    k, mu, N, A = 100.0, 0.5, 1.0, 0.1                # slip limit mu N = 0.5, stick amplitude would be k A = 10
    j = JenkinsContact(1, [0], k_t=k, mu=mu, N0=N)
    v = A * np.cos(TH)
    f = j.force(v[:, None])[:, 0]
    assert np.max(np.abs(f)) == pytest.approx(mu * N)
    # closed form for a sinusoid under gross slip: area = 4 mu N (A - mu N / k)
    expected = 4 * mu * N * (A - mu * N / k)
    assert jenkins_hysteresis_energy(v, f) == pytest.approx(expected, rel=0.05)


def test_lift_off_removes_the_force():
    j = JenkinsContact(2, [0], k_t=50.0, mu=0.4, N0=1.0, normal_dofs=[1], k_n=10.0)
    q = np.zeros((NT, 2))
    q[:, 0] = 0.05 * np.cos(TH)
    q[:, 1] = -0.3                                     # N0 + k_n v_n = -2 < 0: separated everywhere
    f = j.force(q)
    assert np.allclose(f[:, 0], 0.0)
    assert np.allclose(f[:, 1], -1.0)                  # N - N0 = -N0


@pytest.mark.parametrize("case", ["stick", "slip", "mixed_normal"])
def test_jacobian_matches_finite_differences(case):
    rng = np.random.default_rng(3)
    if case == "stick":
        j, q = JenkinsContact(2, [0], 100.0, 1.0, 1e3), np.zeros((NT, 2))
        q[:, 0] = 0.01 * np.cos(TH)
    elif case == "slip":
        j, q = JenkinsContact(2, [0], 100.0, 0.3, 1.0), np.zeros((NT, 2))
        q[:, 0] = 0.1 * np.cos(TH + 0.3)
    else:
        j = JenkinsContact(2, [0], 100.0, 0.4, 1.0, normal_dofs=[1], k_n=20.0)
        q = np.zeros((NT, 2))
        q[:, 0] = 0.05 * np.cos(TH)
        q[:, 1] = 0.04 * np.cos(2 * TH + 1.0) - 0.01
    J = j.jacobian(q)
    h = 1e-7
    Jfd = np.zeros_like(J)
    flat = q.reshape(-1)
    for c in range(flat.size):
        qp, qm = flat.copy(), flat.copy()
        qp[c] += h
        qm[c] -= h
        Jfd[:, c] = (j.force(qp.reshape(q.shape)) - j.force(qm.reshape(q.shape))).reshape(-1) / (2 * h)
    # one-sided derivatives at switching samples can differ; compare in the bulk
    bad = np.abs(J - Jfd) > 1e-4 * max(np.abs(J).max(), 1.0)
    assert bad.sum() <= 0.01 * J.size

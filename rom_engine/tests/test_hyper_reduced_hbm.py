# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Hyper-reduced harmonic balance for a chain with an array of friction dampers (the jointed-structure pattern)."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine.forced_response_hbm import ForcedHBM, forced_response_continuation, equivalent_resonance
from rom_engine.friction_contact_models import JenkinsContact
from rom_engine.hyper_reduced_hbm import (JointElementSet, harmonic_training_matrix, fit_harmonic_ecsw,
                                          HyperReducedJointForce, residual_error_indicator, _FullJointForce)

N = 24
JOINT_DOFS = np.arange(10, 20)
KT = 20.0
H = 3


def build():
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += 100.0
        if i:
            K[i, i - 1] = K[i - 1, i] = -100.0
            K[i - 1, i - 1] += 100.0
    M = np.eye(N)
    C = 0.02 * K
    # friction dampers with graded preloads, so some elements slip first
    N0 = np.linspace(0.02, 0.5, len(JOINT_DOFS))
    elements = [([d], JenkinsContact(1, [0], k_t=KT, mu=0.5, N0=n0)) for d, n0 in zip(JOINT_DOFS, N0)]
    joints = JointElementSet(N, elements)
    Kst = K.copy()
    Kst[JOINT_DOFS, JOINT_DOFS] += KT                           # all dampers stuck
    w2, V = eigh(Kst, M)
    return M, C, K, joints, V, np.sqrt(w2)


@pytest.fixture(scope="module")
def ctx():
    M, C, K, joints, V, w = build()
    F = M @ V[:, :3] @ np.array([1.0, 0.5, 0.3])               # a smooth load the low modes can represent
    F = F / np.abs(F).max()
    out = dict(M=M, C=C, K=K, joints=joints, V=V, w=w, F=F)
    for r in (8, 2):
        W = V[:, :r]
        red = ForcedHBM(W.T @ M @ W, W.T @ C @ W, W.T @ K @ W, n_harmonics=H)
        out[r] = dict(W=W, red=red, Fr=red.forcing(W.T @ F * 0.05))
    return out


def _sweep(ctx, r, force_model, w0=None, w1=None, ds=0.05):
    d = ctx[r]
    w0 = ctx["w"][0] * 0.8 if w0 is None else w0
    w1 = ctx["w"][0] * 1.2 if w1 is None else w1
    return forced_response_continuation(d["red"], force_model, d["Fr"], w0, w1, dof=0, ds=ds)


def test_training_columns_sum_to_the_target_and_weights_are_sparse(ctx):
    d = ctx[8]
    full = HyperReducedJointForce(ctx["joints"], d["W"])
    resp = _sweep(ctx, 8, full)
    assert resp.completed
    ipk = int(np.argmax(resp.amplitude))
    orbits = [resp.Z[ipk], resp.Z[max(ipk - 4, 0)]]
    G, b = harmonic_training_matrix(d["red"], ctx["joints"], d["W"], orbits)
    assert np.allclose(G.sum(axis=1), b)
    assert G.shape == (2 * d["red"].NC * 8, len(JOINT_DOFS))
    xi, info = fit_harmonic_ecsw(d["red"], ctx["joints"], d["W"], orbits, tau=0.01)
    assert np.all(xi >= 0) and info["residual"] <= 0.01 + 1e-12
    assert info["n_selected"] < len(JOINT_DOFS)                     # fewer elements than the full array


def test_hyper_reduced_response_tracks_the_unreduced_reduced_model(ctx):
    d = ctx[8]
    full = HyperReducedJointForce(ctx["joints"], d["W"])
    ref = _sweep(ctx, 8, full)
    ipk = int(np.argmax(ref.amplitude))
    xi, info = fit_harmonic_ecsw(d["red"], ctx["joints"], d["W"], [ref.Z[ipk], ref.Z[max(ipk - 4, 0)], ref.Z[min(ipk + 4, len(ref.Z) - 1)]], tau=0.01)
    hr = HyperReducedJointForce(ctx["joints"], d["W"], xi)
    got = _sweep(ctx, 8, hr)
    assert got.completed
    a, b = equivalent_resonance(ref.omega, ref.amplitude), equivalent_resonance(got.omega, got.amplitude)
    assert b["omega_peak"] == pytest.approx(a["omega_peak"], rel=0.01)
    assert b["amplitude_peak"] == pytest.approx(a["amplitude_peak"], rel=0.10)
    assert len(hr.active) < len(JOINT_DOFS)


def test_hyper_reduced_jacobian_matches_finite_differences(ctx):
    d = ctx[8]
    xi = np.zeros(len(JOINT_DOFS))
    xi[[1, 4, 7]] = [1.5, 2.0, 0.7]
    hr = HyperReducedJointForce(ctx["joints"], d["W"], xi)
    rng = np.random.default_rng(0)
    nt = d["red"].NT
    q = 0.05 * rng.standard_normal((nt, 8))
    J = hr.jacobian(q)
    h = 1e-7
    Jfd = np.zeros_like(J)
    flat = q.reshape(-1)
    for c in range(0, flat.size, 3):                                 # every third column keeps the test fast
        qp, qm = flat.copy(), flat.copy()
        qp[c] += h; qm[c] -= h
        Jfd[:, c] = (hr.force(qp.reshape(q.shape)) - hr.force(qm.reshape(q.shape))).reshape(-1) / (2 * h)
    cols = np.arange(0, flat.size, 3)
    bad = np.abs(J[:, cols] - Jfd[:, cols]) > 1e-4 * max(np.abs(J).max(), 1)
    assert bad.mean() < 0.01


def test_error_indicator_vanishes_for_an_exact_solution_and_ranks_hyper_meshes_without_a_full_order_solve(ctx):
    M, C, K = ctx["M"], ctx["C"], ctx["K"]
    full_hbm = ForcedHBM(M, C, K, n_harmonics=H)
    lev = 0.05
    Ffull = full_hbm.forcing(ctx["F"] * lev)
    # (1) the indicator is zero (round-off) for the full-order solution
    from rom_engine.forced_response_hbm import _newton
    w0 = ctx["w"][0] * 0.9
    full_force = _FullJointForce(ctx["joints"])
    z0 = np.linalg.solve(full_hbm.A(w0), Ffull.reshape(-1))
    z, ok, _ = _newton(lambda z: full_hbm.residual(z, w0, Ffull, full_force),
                       lambda z: full_hbm.jacobian_z(z, w0, full_force), z0, 1e-10, 30)
    assert ok
    assert residual_error_indicator(full_hbm, ctx["joints"], np.eye(N), z.reshape(full_hbm.NC, N), w0, Ffull, 22) < 1e-8
    # (2) for a reduced model, a well-trained hyper mesh scores lower (better) than a crude one, and the ordering
    # agrees with the true error against the unreduced reduced model
    d = ctx[8]
    d["Fr"] = d["red"].forcing(d["W"].T @ ctx["F"] * lev)
    ref = _sweep(ctx, 8, HyperReducedJointForce(ctx["joints"], d["W"]))
    ipk = int(np.argmax(ref.amplitude))
    xi_good, _ = fit_harmonic_ecsw(d["red"], ctx["joints"], d["W"], [ref.Z[ipk], ref.Z[max(ipk - 4, 0)]], tau=0.01)
    xi_bad = np.zeros(len(JOINT_DOFS))
    xi_bad[0] = float(len(JOINT_DOFS))                      # one element standing in for all ten
    scores, errs = {}, {}
    for name, xi in (("good", xi_good), ("bad", xi_bad)):
        resp = _sweep(ctx, 8, HyperReducedJointForce(ctx["joints"], d["W"], xi))
        i = int(np.argmax(resp.amplitude))
        scores[name] = residual_error_indicator(full_hbm, ctx["joints"], d["W"], resp.Z[i], resp.omega[i], Ffull, 22)
        errs[name] = abs(resp.amplitude.max() - ref.amplitude.max()) / ref.amplitude.max()
    scores["exact_rom"] = residual_error_indicator(full_hbm, ctx["joints"], d["W"], ref.Z[ipk], ref.omega[ipk], Ffull, 22)
    assert scores["good"] < scores["bad"] and errs["good"] < errs["bad"]
    assert scores["good"] < 1.5 * scores["exact_rom"] + 1e-3        # close to what the unreduced ROM scores

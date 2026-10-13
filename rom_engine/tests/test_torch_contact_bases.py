# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
Cross-checks of rom_engine.torch_contact_bases against the NumPy originals. Skips cleanly when torch is not
importable. Verified 2026-10-13: 5 passed on a PyTorch-equipped machine.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine import torch_contact_bases as tcb

pytestmark = pytest.mark.skipif(not tcb._HAS_TORCH, reason="PyTorch not installed in this environment")

from rom_engine.contact_dual_basis import nonneg_dual_basis
from rom_engine.interface_reduction import gram_schmidt_interface_modes, joint_interface_modes
from rom_engine.operator_inference import fit_spd_operators
from rom_engine.rubin_dual_basis import reidentify_polynomial_force


def test_nmf_matches_numpy():
    rng = np.random.default_rng(0)
    X = rng.random((40, 4)) @ rng.random((4, 30))
    a = nonneg_dual_basis(X, 4, n_iter=300)
    b = tcb.torch_nonneg_dual_basis(X, 4, n_iter=300)
    assert b["rel_error"] == pytest.approx(a.rel_error, rel=1e-6, abs=1e-9)
    assert np.allclose(b["U"].numpy(), a.U, atol=1e-8)
    assert float(b["U"].min()) >= 0
    with pytest.raises(ValueError):
        tcb.torch_nonneg_dual_basis(-X, 2)


def test_spd_fit_matches_numpy_when_unconstrained_fit_is_spd():
    rng = np.random.default_rng(1)
    n = 4
    G = rng.standard_normal((n, n)); M = G @ G.T + n * np.eye(n)
    G = rng.standard_normal((n, n)); K = G @ G.T + n * np.eye(n)
    X = rng.standard_normal((n, 80)); A = rng.standard_normal((n, 80))
    F = M @ A + K @ X
    Mn, Kn, info = fit_spd_operators(X, A, F)
    out = tcb.torch_fit_spd_operators(X, A, F)
    assert out["spd_unconstrained"] and info["spd_unconstrained"]
    assert np.allclose(out["M"].numpy(), Mn, rtol=1e-6) and np.allclose(out["K"].numpy(), Kn, rtol=1e-6)


def test_spd_fit_is_spd_for_nonsense_data():
    X = np.random.default_rng(2).standard_normal((3, 50))
    out = tcb.torch_fit_spd_operators(X, X[::-1] * 0.3, -X, max_iter=50)
    import torch
    assert float(torch.linalg.eigvalsh(out["M"]).min()) > 0 and float(torch.linalg.eigvalsh(out["K"]).min()) > 0


def test_joint_interface_modes_and_gram_schmidt_match_numpy():
    rng = np.random.default_rng(3)
    n = 10
    G = rng.standard_normal((n, n)); K = G @ G.T + n * np.eye(n)
    M = np.diag(rng.random(n) + 0.5)

    class CB:                                           # just the attributes joint_interface_modes reads
        pass
    cb = CB(); cb.n_interface = n; cb.K = K; cb.M = M
    joint = [1, 3, 4, 7, 8]
    Vn, wn = joint_interface_modes(cb, joint, 3)
    Vt, wt = tcb.torch_joint_interface_modes(K, M, joint, 3)
    assert np.allclose(wt.numpy(), wn, rtol=1e-8)
    assert np.allclose(np.abs(Vt.numpy().T @ M[np.ix_(joint, joint)] @ Vn), np.eye(3), atol=1e-6) or \
        np.allclose(np.abs(Vt.numpy()), np.abs(Vn), atol=1e-6)        # equal up to sign (condensed M differs in scaling)
    S1, S2 = rng.standard_normal((n, 4)), rng.standard_normal((n, 4))
    Qn = gram_schmidt_interface_modes([S1, S2], list(range(n)))
    Qt = tcb.torch_gram_schmidt_interface_modes([S1, S2], list(range(n)))
    assert np.allclose(Qt.numpy(), Qn, atol=1e-10)


def test_polynomial_reidentification_matches_numpy():
    rng = np.random.default_rng(4)
    nt, r = 200, 2
    q, qd, qdd = (rng.standard_normal((nt, r)) * 0.1 for _ in range(3))
    Mr, Cr, Kr = np.eye(r), 0.1 * np.eye(r), 20 * np.eye(r)
    g = np.column_stack([3 * q[:, 0] ** 3 + q[:, 0] * q[:, 1], 2 * q[:, 1] ** 2])
    f = qdd @ Mr.T + qd @ Cr.T + q @ Kr.T + g
    fit, info = reidentify_polynomial_force(Mr, Cr, Kr, q, qd, qdd, f)
    out = tcb.torch_reidentify_polynomial_force(Mr, Cr, Kr, q, qd, qdd, f)
    assert np.allclose(out["coeff"].numpy(), fit.coeff, atol=1e-7)
    assert out["relative_residual"] < 1e-8

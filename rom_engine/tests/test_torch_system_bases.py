# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_torch_system_bases.py -- cross-checks every torch_* function of rom_engine.torch_system_bases against its
NumPy original (modal derivatives, selection weights, augmentation, dual modes, quadratic manifold, slow-fast
condensation, free-interface CMS, interface reduction, mode sensitivities), plus autograd through the manifold.

Skips cleanly when torch is not importable (the module-level skip), like the other torch test files. Full-order
callables are the NumPy beam wrapped into tensors (not differentiable); the autograd test uses an explicit torch
cubic model instead.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine import torch_system_bases as tsb

pytestmark = pytest.mark.skipif(not tsb._HAS_TORCH, reason="PyTorch not installed in this environment")

if tsb._HAS_TORCH:
    import torch

from rom_engine.cms import craig_bampton, free_interface_cms, reduce_interface
from rom_engine.dual_modes import dual_modes
from rom_engine.modal_derivatives import (augment_with_derivatives, modal_interaction_weights,
                                          static_modal_derivatives)
from rom_engine.parametric_modes import mode_sensitivities
from rom_engine.quadratic_manifold import (QuadraticManifold, QuadraticManifoldROM, fit_quadratic_manifold,
                                           manifold_from_derivatives)
from rom_engine.slow_fast import SlowFastROM
from nonlinear_beam_fixtures import nonlinear_beam_model, synthetic_cubic_model


def tt(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def npy(x):
    return x.detach().cpu().numpy()


@pytest.fixture(scope="module")
def beam():
    b = nonlinear_beam_model(n_elem=10, n_modes=3)
    b["f_t"] = lambda u: tt(b["f_int"](npy(u)))
    b["K_t_t"] = lambda u: tt(b["K_t"](npy(u)))
    b["static_t"] = lambda F: tt(b["static"](npy(F)))
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    mid = int(np.argmax(np.abs(b["Phi"][:, 0])))
    b["F"] = b["K0"] @ (3 * h / b["Phi"][mid, 0] * b["Phi"][:, 0])
    b["h"] = h
    return b


def test_static_modal_derivatives_match_numpy(beam):
    ref = static_modal_derivatives(beam["K0"], beam["Phi"], tangent_fn=beam["K_t"], length_scale=beam["L"])
    th, info = tsb.torch_static_modal_derivatives(beam["K0"], beam["Phi"], tangent_fn=beam["K_t_t"],
                                                  length_scale=beam["L"])
    assert th.shape == tuple(ref.theta.shape)
    assert np.linalg.norm(npy(th) - ref.theta) < 1e-8 * np.linalg.norm(ref.theta)
    assert info["n_tangent_evals"] == ref.n_tangent_evals and abs(info["asymmetry"] - ref.asymmetry) < 1e-6
    # supplied derivative matrices give the same result as the finite-difference route on an exact cubic model
    s = synthetic_cubic_model(n=8)
    w2, P = eigh(s["K0"], s["M"])
    Phi = P[:, :3]
    dK = [np.einsum("abc,c->ab", s["T"], Phi[:, j]) for j in range(3)]
    a, _ = tsb.torch_static_modal_derivatives(s["K0"], Phi, dK=dK)
    b = static_modal_derivatives(s["K0"], Phi, dK=dK)
    assert np.allclose(npy(a), b.theta, atol=1e-9 * np.abs(b.theta).max())
    with pytest.raises(ValueError):
        tsb.torch_static_modal_derivatives(s["K0"], Phi)


def test_interaction_weights_and_augmentation_match_numpy(beam):
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((200, 3))
    assert np.allclose(npy(tsb.torch_modal_interaction_weights(Q, dt=0.01)), modal_interaction_weights(Q, 0.01))
    ref = static_modal_derivatives(beam["K0"], beam["Phi"], tangent_fn=beam["K_t"], length_scale=beam["L"])
    B_ref, i_ref = augment_with_derivatives(beam["Phi"], ref, M=beam["M"])
    B, info = tsb.torch_augment_with_derivatives(beam["Phi"], ref.theta, M=beam["M"])
    assert B.shape == B_ref.shape and info["n_added"] == i_ref["n_added"]
    # same span: projector difference
    assert np.linalg.norm(npy(B) @ npy(B).T - B_ref @ B_ref.T) < 1e-7
    Bn = npy(B)
    assert np.allclose(Bn.T @ beam["M"] @ Bn, np.eye(Bn.shape[1]), atol=1e-8)
    none, ni = tsb.torch_augment_with_derivatives(beam["Phi"], ref.theta, pairs=[], M=beam["M"])
    assert ni["n_added"] == 0 and none.shape[1] == 3


def test_dual_modes_match_numpy(beam):
    amps = [0.5 * beam["h"], 3 * beam["h"]]
    B_ref, i_ref = dual_modes(beam["K0"], beam["M"], beam["Phi"], beam["static"], amps, dominant=0)
    B, info = tsb.torch_dual_modes(beam["K0"], beam["M"], beam["Phi"], beam["static_t"], amps, dominant=0)
    assert info["n_solves"] == i_ref["n_solves"] and B.shape == B_ref.shape
    assert np.linalg.norm(npy(B) @ npy(B).T - B_ref @ B_ref.T) < 1e-6
    with pytest.raises(ValueError):
        tsb.torch_dual_modes(beam["K0"], beam["M"], beam["Phi"], beam["static_t"], amps)


def test_quadratic_manifold_matches_numpy(beam):
    ref = static_modal_derivatives(beam["K0"], beam["Phi"], tangent_fn=beam["K_t"], length_scale=beam["L"])
    man_np = manifold_from_derivatives(beam["Phi"], ref, beam["M"])
    man = tsb.torch_manifold_from_derivatives(beam["Phi"], ref.theta, beam["M"])
    rng = np.random.default_rng(1)
    q, qd = rng.standard_normal(3) * 1e-3, rng.standard_normal(3)
    assert np.allclose(npy(man.map(q)), man_np.map(q), atol=1e-12)
    assert np.allclose(npy(man.tangent(q)), man_np.tangent(q), atol=1e-12)
    assert np.allclose(npy(man.reduced_mass(q)), man_np.reduced_mass(q), atol=1e-10)
    assert np.allclose(npy(man.convective_force(q, qd)), man_np.convective_force(q, qd), atol=1e-10)
    Q = rng.standard_normal((3, 5)) * 1e-3
    assert np.allclose(npy(man.lift(Q)), man_np.lift(Q), atol=1e-12)
    with pytest.raises(ValueError):
        tsb.TorchQuadraticManifold(beam["Phi"], np.zeros((2, 2, 2)))


def test_manifold_fit_matches_numpy():
    s = synthetic_cubic_model(n=8)
    w2, P = eigh(s["K0"], s["M"])
    Phi = P[:, :3]
    rng = np.random.default_rng(2)
    X = Phi @ rng.uniform(-1, 1, (3, 60)) + 0.05 * rng.standard_normal((8, 60))
    ref, ri = fit_quadratic_manifold(Phi, X, M=s["M"], ridge=1e-10)
    man, info = tsb.torch_fit_quadratic_manifold(Phi, X, M=s["M"], ridge=1e-10)
    assert np.allclose(npy(man.Omega), ref.Omega, atol=1e-6 * np.abs(ref.Omega).max())
    assert abs(info["residual"] - ri["residual"]) < 1e-8


def test_manifold_static_solve_matches_numpy_and_autograd_jacobian_agrees(beam):
    ref = static_modal_derivatives(beam["K0"], beam["Phi"], tangent_fn=beam["K_t"], length_scale=beam["L"])
    q_ref = QuadraticManifoldROM(manifold_from_derivatives(beam["Phi"], ref, beam["M"]),
                                 f_int=beam["f_int"]).static_solve(beam["F"])
    man = tsb.torch_manifold_from_derivatives(beam["Phi"], ref.theta, beam["M"])
    rom = tsb.TorchQuadraticManifoldROM(man, beam["f_t"], tangent_fn=beam["K_t_t"])
    q = rom.static_solve(beam["F"])
    assert np.linalg.norm(npy(q) - q_ref) < 1e-6 * np.linalg.norm(q_ref)


def test_autograd_through_a_torch_native_manifold_model():
    s = synthetic_cubic_model(n=6)
    w2, P = eigh(s["K0"], s["M"])
    Phi = P[:, :2]
    T, K0, S = tt(s["T"]), tt(s["K0"]), tt(s["S"])
    f = lambda u: K0 @ u + 0.5 * torch.einsum("ijk,j,k->i", T, u, u)
    Om = torch.zeros(6, 2, 2, dtype=torch.float64, requires_grad=True)
    man = tsb.TorchQuadraticManifold(Phi, Om)
    q = torch.tensor([0.01, -0.005], dtype=torch.float64)
    loss = (man.map(q) ** 2).sum()
    loss.backward()
    assert Om.grad is not None and float(Om.grad.abs().sum()) > 0
    # the autograd Jacobian path of the Newton solve (no tangent_fn) agrees with the analytic path
    man2 = tsb.TorchQuadraticManifold(Phi, 0.1 * torch.randn(6, 2, 2, dtype=torch.float64))
    F = K0 @ tt(Phi[:, 0]) * 1e-3
    q_auto = tsb.TorchQuadraticManifoldROM(man2, f).static_solve(F)
    Kt = lambda u: K0 + torch.einsum("ijk,k->ij", T, u)
    q_ana = tsb.TorchQuadraticManifoldROM(man2, f, tangent_fn=Kt).static_solve(F)
    assert np.allclose(npy(q_auto), npy(q_ana), atol=1e-9)


def test_slow_fast_matches_numpy(beam):
    md = static_modal_derivatives(beam["K0"], beam["Phi"], tangent_fn=beam["K_t"], length_scale=beam["L"])
    B, _ = augment_with_derivatives(beam["Phi"], md, M=beam["M"])
    Vf = B[:, 3:]
    ref = SlowFastROM(beam["Phi"], Vf, beam["f_int"], beam["K_t"])
    tor = tsb.TorchSlowFastROM(beam["Phi"], Vf, beam["f_t"], beam["K_t_t"])
    q_ref = ref.static_solve(beam["F"])
    q = tor.static_solve(beam["F"])
    assert np.linalg.norm(npy(q) - q_ref) < 1e-6 * np.linalg.norm(q_ref)
    assert np.linalg.norm(npy(tor.state(q, beam["F"])) - ref.state(q_ref, beam["F"])) < 1e-6 * np.linalg.norm(
        ref.state(q_ref, beam["F"]))
    assert np.allclose(npy(tor.reduced_tangent(q, beam["F"])), ref.reduced_tangent(q_ref, beam["F"]),
                       rtol=1e-5, atol=1e-6 * np.abs(ref.reduced_tangent(q_ref, beam["F"])).max())


def _chain(n=30, k=1.0e4):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k
        if i + 1 < n:
            K[i + 1, i + 1] += k
            K[i, i + 1] = K[i + 1, i] = -k
    return K, np.eye(n)


def test_free_interface_cms_and_interface_reduction_match_numpy():
    K, M = _chain()
    for method in ("rubin", "macneal"):
        ref = free_interface_cms(K, M, [29, 15], 5, method)
        out = tsb.torch_free_interface_cms(K, M, [29, 15], 5, method)
        f_ref = np.sqrt(eigh(ref.K, ref.M, eigvals_only=True)[:3])
        f = np.sqrt(eigh(npy(out["K"]), npy(out["M"]), eigvals_only=True)[:3])
        assert np.allclose(f, f_ref, rtol=1e-8)
    with pytest.raises(ValueError):
        tsb.torch_free_interface_cms(K, M, [29], 5, "bogus")
    cb = craig_bampton(K, M, [29, 20, 10], n_modes=6)
    r_ref = reduce_interface(cb, 2)
    r = tsb.torch_reduce_interface(cb.K, cb.M, cb.T, cb.n_interface, cb.n_modes, 2)
    assert np.allclose(np.sqrt(eigh(npy(r["K"]), npy(r["M"]), eigvals_only=True)[:4]),
                       np.sqrt(eigh(r_ref.K, r_ref.M, eigvals_only=True)[:4]), rtol=1e-8)
    assert np.allclose(npy(r["omega_interface"]), r_ref.omega_interface, rtol=1e-8)


def test_mode_sensitivities_match_numpy():
    rng = np.random.default_rng(4)
    n = 9
    A = rng.standard_normal((n, n)); B = rng.standard_normal((n, n)); D = rng.standard_normal((n, n))
    K0, M0, dK, dM = A @ A.T + n * np.eye(n), B @ B.T + n * np.eye(n), 0.5 * (D + D.T), 0.1 * (B @ B.T)
    dw, dP, _ = mode_sensitivities(K0, M0, dK, dM, n_modes=3)
    tw, tP, info = tsb.torch_mode_sensitivities(K0, M0, dK, dM, n_modes=3)
    assert np.allclose(npy(tw), dw, rtol=1e-8)
    # eigenvector sign is arbitrary: compare with each column's sign fixed by the numpy mode
    P0 = eigh(K0, M0)[1][:, :3]
    sgn = np.sign(np.sum(P0 * npy(tsb._gen_eigh(tsb._t(K0), tsb._t(M0))[1])[:, :3], axis=0))
    assert np.allclose(npy(tP) * sgn, dP, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError):
        tsb.torch_mode_sensitivities(np.eye(4), np.eye(4), np.eye(4), n_modes=2)


def test_helpers_reject_missing_torch_gracefully(monkeypatch):
    monkeypatch.setattr(tsb, "_HAS_TORCH", False)
    with pytest.raises(ImportError):
        tsb._require_torch()

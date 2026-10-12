# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
torch_system_bases.py -- opt-in PyTorch counterparts of the system-driven-basis and quadratic-manifold modules
(`modal_derivatives`, `dual_modes`, `quadratic_manifold`, `slow_fast`, the free-interface part of `cms`, and
`parametric_modes`), placed SIDE BY SIDE with the NumPy/SciPy originals rather than replacing them.

CONVENTION (the same as `torch_linalg.py` and the `*_torch` methods of `IntrusiveNonlinearROM`): the NumPy path
stays the default and needs no torch install; every function here is named `torch_*` (or `Torch*`), is an explicit
opt-in, accepts NumPy arrays or tensors, and returns tensors (float64 by default, on the requested `device`). Full-
order callables (`f_int`, `tangent_fn`, `fom_solver`) passed to these functions must be torch-native: they receive
and return tensors. Nothing here changes a NumPy result.

WHAT TORCH ADDS. (1) Batched dense algebra on a GPU: all m^2 modal-derivative right-hand sides are solved against
ONE LU factorisation in a single call. (2) Differentiability: the quadratic-manifold map, the reduced forces and the
Newton solves are built from torch ops, so `torch.autograd` can differentiate a reduced response with respect to
the manifold coefficients, a load or a parameter, and the reduced Jacobian of the manifold static solve is taken by
autograd rather than assembled by hand.

WHAT IS NOT CLAIMED. No speed-up is claimed: for the small dense matrices of a reduced model the CPU NumPy path is
often faster, and a GPU win appears only for large dense full-order matrices or batched sweeps. Sparse SciPy inputs
are densified (`.toarray()`), so these functions suit moderate-size models. Not ported, deliberately: the dynamic
modal derivatives (bordered sparse system), the enhanced enforced-displacement identification, ECSW weight selection
(sequential NNLS, no torch equivalent) and the NNM glue, which call SciPy-only machinery with nothing to gain here.
Every function is cross-checked against its NumPy original in tests/test_torch_system_bases.py, which skips cleanly
when torch is not importable.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:       # bare Exception: a CUDA wheel without its runtime can fail at import with OSError/ValueError
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError("torch_system_bases.py: PyTorch is not importable in this environment -- install it "
                          "(`pip install torch`) or use the NumPy functions of the same name without the torch_ "
                          "prefix.")


def _t(x, device="cpu", dtype=None):
    """Tensor from a tensor, ndarray, list or scipy sparse matrix."""
    _require_torch()
    dtype = torch.float64 if dtype is None else dtype
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=device)


def _gen_eigh(K, M):
    """Generalised symmetric eigenproblem K x = lambda M x via the Cholesky factor of M. Returns ascending
    eigenvalues and M-orthonormal eigenvectors (torch has no direct generalised eigh)."""
    L = torch.linalg.cholesky(0.5 * (M + M.T))
    Li = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], dtype=L.dtype, device=L.device), upper=False)
    A = Li @ K @ Li.T
    w, Y = torch.linalg.eigh(0.5 * (A + A.T))
    return w, Li.T @ Y


# --------------------------------------------------------------------------------------- modal derivatives
def torch_directional_stiffness_derivative(tangent_fn, direction, u0=None, step=None, length_scale=1.0,
                                           rel_step=1e-6, device="cpu"):
    """Torch counterpart of `modal_derivatives.directional_stiffness_derivative` (central difference); `tangent_fn`
    takes and returns tensors."""
    d = _t(direction, device)
    u0 = torch.zeros_like(d) if u0 is None else _t(u0, device)
    if step is None:
        m = float(d.abs().max())
        if m == 0.0:
            raise ValueError("direction is zero")
        step = rel_step * length_scale / m
    return (_t(tangent_fn(u0 + step * d), device) - _t(tangent_fn(u0 - step * d), device)) / (2.0 * step)


def torch_static_modal_derivatives(K, directions, tangent_fn=None, dK=None, u0=None, step=None, length_scale=1.0,
                                   rel_step=1e-6, symmetrize=True, device="cpu"):
    """Torch counterpart of `static_modal_derivatives`. K is factorised ONCE and all m^2 right-hand sides
    -(dK_j phi_i) are solved in a single batched back-substitution. Returns (theta, info): theta (n, m, m) tensor,
    info with `asymmetry` (measured before symmetrisation) and `n_tangent_evals`."""
    Phi = _t(directions, device)
    n, m = Phi.shape
    if (tangent_fn is None) == (dK is None):
        raise ValueError("give exactly one of tangent_fn or dK")
    evals = 0
    if dK is not None:
        dKs = torch.stack([_t(d, device) for d in dK])
    else:
        dKs = torch.stack([torch_directional_stiffness_derivative(tangent_fn, Phi[:, j], u0, step, length_scale,
                                                                  rel_step, device) for j in range(m)])
        evals = 2 * m
    rhs = -torch.einsum("jab,bi->aij", dKs, Phi).reshape(n, m * m)
    LU, piv = torch.linalg.lu_factor(_t(K, device))
    theta = torch.linalg.lu_solve(LU, piv, rhs).reshape(n, m, m)         # theta[:, i, j] = d phi_i / d q_j
    scale = float(theta.abs().max())
    asym = 0.0 if scale == 0.0 else float((theta - theta.transpose(1, 2)).abs().max()) / scale
    if symmetrize:
        theta = 0.5 * (theta + theta.transpose(1, 2))
    return theta, dict(asymmetry=asym, n_tangent_evals=evals, kind="static")


def torch_modal_interaction_weights(q_history, dt=1.0, device="cpu"):
    """Torch counterpart of `modal_interaction_weights`: w_ij = integral |q_i q_j| dt, all pairs at once."""
    Q = _t(q_history, device)
    prod = (Q[:, :, None] * Q[:, None, :]).abs()
    return torch.trapezoid(prod, dx=dt, dim=0)


def torch_augment_with_derivatives(modes, theta, pairs=None, M=None, tol=1e-8, max_vectors=None, device="cpu"):
    """Torch counterpart of `augment_with_derivatives`. Returns (basis, info)."""
    Phi = _t(modes, device)
    th = _t(theta, device)
    n, m = Phi.shape
    Mm = (lambda B: B) if M is None else (lambda B, _M=_t(M, device): _M @ B)
    if pairs is None:
        pairs = [(i, j) for i in range(m) for j in range(i, m)]
    if not pairs:
        return Phi.clone(), dict(n_candidates=0, n_added=0, gram_eigenvalues=torch.zeros(0), retained=1.0)
    X = torch.stack([th[:, i, j] for i, j in pairs], dim=1)
    X = X - Phi @ (Phi.T @ Mm(X))
    G = X.T @ Mm(X)
    lam, U = torch.linalg.eigh(0.5 * (G + G.T))
    lam, U = lam.flip(0), U.flip(1)
    keep = lam > tol * max(float(lam[0]), 1e-300)
    if max_vectors is not None:
        keep = keep & (torch.arange(len(lam), device=lam.device) < int(max_vectors))
    Psi = X @ (U[:, keep] / torch.sqrt(lam[keep]))
    retained = float(lam[keep].sum() / max(float(lam.sum()), 1e-300))
    return torch.cat([Phi, Psi], dim=1), dict(n_candidates=X.shape[1], n_added=int(keep.sum()),
                                              gram_eigenvalues=lam, retained=retained)


# --------------------------------------------------------------------------------------- dual modes
def torch_dual_modes(K, M, modes, fom_solver, amplitudes, p=None, dominant=None, n_dual=None,
                     n_keep_per_set=None, tol=1e-10, device="cpu"):
    """Torch counterpart of `dual_modes.dual_modes`. `fom_solver(load_tensor) -> displacement tensor` is the
    torch-native nonlinear static solver of the full model. Same procedure, scoring and caveats as the NumPy
    version (see its docstring, including the reconstructed scoring rule). Returns (basis, info)."""
    Phi, Kt, Mt = _t(modes, device), _t(K, device), _t(M, device)
    n, m = Phi.shape
    if dominant is None:
        if p is None:
            raise ValueError("give the load pattern p or the dominant mode index")
        dominant = int(torch.argmax((Phi.T @ _t(p, device)).abs()))
    amplitudes = [float(a) for a in amplitudes]
    peak = float(Phi[:, dominant].abs().max())
    if peak == 0.0:
        raise ValueError("mode is zero")
    alphas = [a / peak for a in amplitudes]
    keep_k = int(min(len(alphas), 2)) if n_keep_per_set is None else int(n_keep_per_set)
    cands, scores, n_solves = [], [], 0
    for j in range(m):
        Kshape = Kt @ (0.5 * (Phi[:, dominant] + Phi[:, j]))
        X = torch.stack([_t(fom_solver(a * Kshape), device) for a in alphas], dim=1)
        n_solves += len(alphas)
        X = X - Phi @ (Phi.T @ (Mt @ X))
        U, s, _ = torch.linalg.svd(X, full_matrices=False)
        ok = s > tol * max(float(s[0]), 1e-300)
        for k in range(min(keep_k, int(ok.sum()))):
            psi = U[:, k]
            beta = (psi @ X) / (psi @ psi)
            scores.append(float(psi @ (Kt @ psi)) * float((beta ** 2).sum()))
            cands.append(psi)
    if not cands:
        return Phi.clone(), dict(dominant=dominant, n_solves=n_solves, scores=torch.zeros(0), n_candidates=0,
                                 n_dual=0, duals=torch.zeros(n, 0, dtype=Phi.dtype, device=Phi.device))
    order = np.argsort(-np.array(scores), kind="stable")
    n_dual = m if n_dual is None else int(n_dual)
    chosen = torch.stack([cands[k] for k in order[:n_dual]], dim=1)
    chosen = chosen - Phi @ (Phi.T @ (Mt @ chosen))
    G = chosen.T @ (Mt @ chosen)
    lam, W = torch.linalg.eigh(0.5 * (G + G.T))
    lam, W = lam.flip(0), W.flip(1)
    keep = lam > 1e-10 * max(float(lam[0]), 1e-300)
    duals = chosen @ (W[:, keep] / torch.sqrt(lam[keep]))
    return torch.cat([Phi, duals], dim=1), dict(dominant=dominant, n_solves=n_solves,
                                                scores=torch.as_tensor(np.array(scores)[order]),
                                                n_candidates=len(cands), n_dual=int(duals.shape[1]), duals=duals)


# --------------------------------------------------------------------------------------- quadratic manifold
class TorchQuadraticManifold:
    """Torch counterpart of `QuadraticManifold`: u = Phi q + 1/2 Omega(q, q). Everything is built from torch ops, so
    `torch.autograd` differentiates through `map`, `tangent`, `reduced_mass` and `convective_force`. `Omega` may
    require grad (e.g. to tune the coefficients against a loss)."""

    def __init__(self, Phi, Omega, M=None, device="cpu"):
        self.Phi = _t(Phi, device)
        Om = _t(Omega, device)
        n, m = self.Phi.shape
        if tuple(Om.shape) != (n, m, m):
            raise ValueError(f"Omega must have shape {(n, m, m)}, got {tuple(Om.shape)}")
        self.Omega = 0.5 * (Om + Om.transpose(1, 2))
        self.M = None if M is None else _t(M, device)
        self.device, self.n, self.m = device, n, m

    def map(self, q):
        q = _t(q, self.device)
        return self.Phi @ q + 0.5 * torch.einsum("ajk,j,k->a", self.Omega, q, q)

    def tangent(self, q):
        return self.Phi + torch.einsum("ajk,k->aj", self.Omega, _t(q, self.device))

    def second_derivative(self, qd):
        qd = _t(qd, self.device)
        return torch.einsum("ajk,j,k->a", self.Omega, qd, qd)

    def reduced_mass(self, q):
        P = self.tangent(q)
        return P.T @ (P if self.M is None else self.M @ P)

    def convective_force(self, q, qd):
        P = self.tangent(q)
        w = self.second_derivative(qd)
        return P.T @ (w if self.M is None else self.M @ w)

    def lift(self, Q):
        """Map columns of Q (m, k) to states (n, k) in one batched call."""
        Q = _t(Q, self.device)
        return self.Phi @ Q + 0.5 * torch.einsum("ajk,jb,kb->ab", self.Omega, Q, Q)


def torch_fit_quadratic_manifold(Phi, snapshots, M=None, ridge=1e-8, device="cpu"):
    """Torch counterpart of `fit_quadratic_manifold` (ridge least squares for Omega). Returns
    (TorchQuadraticManifold, info)."""
    Phi, X = _t(Phi, device), _t(snapshots, device)
    n, m = Phi.shape
    Mt = None if M is None else _t(M, device)
    MX = X if Mt is None else Mt @ X
    Q = Phi.T @ MX
    R = X - Phi @ Q
    pairs = [(i, j) for i in range(m) for j in range(i, m)]
    F = torch.stack([Q[i] * Q[j] for i, j in pairs], dim=1)
    scale = float(torch.linalg.matrix_norm(F, 2)) ** 2
    A = F.T @ F + ridge * scale * torch.eye(len(pairs), dtype=F.dtype, device=F.device)
    C = torch.linalg.solve(A, F.T @ R.T).T
    Om = torch.zeros(n, m, m, dtype=Phi.dtype, device=Phi.device)
    for c, (i, j) in enumerate(pairs):
        if i == j:
            Om[:, i, i] = 2.0 * C[:, c]
        else:
            Om[:, i, j] = C[:, c]
            Om[:, j, i] = C[:, c]
    rn = float(torch.linalg.norm(R))
    res = float(torch.linalg.norm(R - C @ F.T)) / rn if rn > 0 else 0.0
    return TorchQuadraticManifold(Phi, Om, Mt, device), dict(residual=res, n_pairs=len(pairs),
                                                              n_snapshots=X.shape[1])


def torch_manifold_from_derivatives(Phi, theta, M=None, device="cpu"):
    """Omega = modal derivatives (tensor or array of shape (n, m, m))."""
    return TorchQuadraticManifold(Phi, theta, M, device)


class TorchQuadraticManifoldROM:
    """Torch counterpart of `QuadraticManifoldROM` (full internal force only; no ECSW). `f_int(u_tensor)` is a
    torch-native internal-force callable. The static Newton Jacobian is analytic,
    P^T K_T P + sum_a Omega[a] (f_int - f_ext)_a, when `tangent_fn(u_tensor)` is given, and is otherwise taken by
    autograd (which then requires `f_int` to be differentiable torch code)."""

    def __init__(self, manifold, f_int, tangent_fn=None):
        self.man, self.f_int, self.tangent_fn = manifold, f_int, tangent_fn

    def reduced_force(self, q):
        return self.man.tangent(q).T @ _t(self.f_int(self.man.map(q)), self.man.device)

    def static_solve(self, f_ext, q0=None, tol=1e-9, max_iter=50):
        f_ext = _t(f_ext, self.man.device)
        q = torch.zeros(self.man.m, dtype=f_ext.dtype, device=f_ext.device) if q0 is None else _t(q0, self.man.device)
        scale = max(float(torch.linalg.norm(self.man.Phi.T @ f_ext)), 1e-300)

        def res(x):
            return self.reduced_force(x) - self.man.tangent(x).T @ f_ext

        for _ in range(max_iter):
            r = res(q)
            if float(torch.linalg.norm(r)) <= tol * scale:
                return q.detach()
            if self.tangent_fn is not None:
                qd = q.detach()
                P = self.man.tangent(qd)
                gap = _t(self.f_int(self.man.map(qd)), self.man.device) - f_ext
                J = P.T @ _t(self.tangent_fn(self.man.map(qd)), self.man.device) @ P \
                    + torch.einsum("a,ajk->jk", gap, self.man.Omega)
            else:
                J = torch.autograd.functional.jacobian(res, q.detach())
            q = q.detach() - torch.linalg.solve(J, r.detach())
        raise RuntimeError("manifold static Newton solve did not converge")


# --------------------------------------------------------------------------------------- slow-fast condensation
class TorchSlowFastROM:
    """Torch counterpart of `SlowFastROM`. `f_int(u)` and `tangent_fn(u)` are torch-native."""

    def __init__(self, Vs, Vf, f_int, tangent_fn, device="cpu"):
        self.Vs, self.Vf = _t(Vs, device), _t(Vf, device)
        self.f_int, self.tangent_fn, self.device = f_int, tangent_fn, device
        self.n_fast_solves = 0

    def fast_response(self, q, f_ext=None, p0=None):
        Vs, Vf = self.Vs, self.Vf
        q = _t(q, self.device)
        f_ext = torch.zeros(Vs.shape[0], dtype=Vs.dtype, device=Vs.device) if f_ext is None else _t(f_ext, self.device)
        p = torch.zeros(Vf.shape[1], dtype=Vs.dtype, device=Vs.device) if p0 is None else _t(p0, self.device)
        scale = max(float(torch.linalg.norm(_t(self.f_int(Vs @ q + Vf @ p), self.device))),
                    float(torch.linalg.norm(f_ext)), 1e-300)
        self.n_fast_solves += 1
        for _ in range(50):
            r = Vf.T @ (_t(self.f_int(Vs @ q + Vf @ p), self.device) - f_ext)
            if float(torch.linalg.norm(r)) <= 1e-9 * scale:
                return p
            J = Vf.T @ _t(self.tangent_fn(Vs @ q + Vf @ p), self.device) @ Vf
            p = p - torch.linalg.solve(J, r)
        raise RuntimeError("fast (membrane) Newton solve did not converge")

    def state(self, q, f_ext=None):
        q = _t(q, self.device)
        return self.Vs @ q + self.Vf @ self.fast_response(q, f_ext)

    def reduced_force(self, q, f_ext=None):
        return self.Vs.T @ _t(self.f_int(self.state(q, f_ext)), self.device)

    def reduced_tangent(self, q, f_ext=None):
        Vs, Vf = self.Vs, self.Vf
        K = _t(self.tangent_fn(self.state(q, f_ext)), self.device)
        Kss, Ksf, Kff = Vs.T @ K @ Vs, Vs.T @ K @ Vf, Vf.T @ K @ Vf
        return Kss - Ksf @ torch.linalg.solve(Kff, Ksf.T)

    def static_solve(self, f_ext, q0=None, tol=1e-9, max_iter=50):
        f_ext = _t(f_ext, self.device)
        q = torch.zeros(self.Vs.shape[1], dtype=f_ext.dtype, device=f_ext.device) if q0 is None else _t(q0, self.device)
        scale = max(float(torch.linalg.norm(self.Vs.T @ f_ext)), 1e-300)
        for _ in range(max_iter):
            r = self.Vs.T @ (_t(self.f_int(self.state(q, f_ext)), self.device) - f_ext)
            if float(torch.linalg.norm(r)) <= tol * scale:
                return q
            q = q - torch.linalg.solve(self.reduced_tangent(q, f_ext), r)
        raise RuntimeError("slow static Newton solve did not converge")


# --------------------------------------------------------------------------------------- CMS extensions
def torch_free_interface_cms(K, M, interface, n_modes, method="rubin", device="cpu"):
    """Torch counterpart of `cms.free_interface_cms`. Returns a dict with T, K, M (tensors), omega_free, method."""
    Kt, Mt = _t(K, device), _t(M, device)
    n = Kt.shape[0]
    b = torch.as_tensor(np.atleast_1d(np.asarray(interface, dtype=int)), device=Kt.device)
    w2, Phi = _gen_eigh(Kt, Mt)
    if float(w2[0]) <= 1e-12 * float(w2[-1]):
        raise ValueError("K is singular (rigid-body modes): restrain the substructure first")
    nm = int(min(n_modes, n))
    Pk, wk2 = Phi[:, :nm], w2[:nm]
    Kinv = torch.linalg.inv(Kt)
    G = Kinv - (Pk / wk2) @ Pk.T
    U, sv, _ = torch.linalg.svd(G[:, b], full_matrices=False)
    keep = sv > 1e-9 * max(float(torch.linalg.matrix_norm(Kinv, 2)), 1e-300)
    R = U[:, keep] * sv[keep]
    T = torch.cat([Pk, R], dim=1)
    Kr, Mr = T.T @ Kt @ T, T.T @ Mt @ T
    omega = torch.sqrt(wk2)
    if method == "rubin":
        return dict(T=T, K=0.5 * (Kr + Kr.T), M=0.5 * (Mr + Mr.T), method="rubin", omega_free=omega)
    if method == "macneal":
        nr = R.shape[1]
        Krr_inv_Krk = torch.linalg.solve(Kr[nm:, nm:], Kr[nm:, :nm])
        X = torch.cat([torch.eye(nm, dtype=Kr.dtype, device=Kr.device), -Krr_inv_Krk], dim=0)
        Mfull = torch.zeros_like(Mr)
        Mfull[:nm, :nm] = Mr[:nm, :nm]
        Kc, Mc = X.T @ Kr @ X, X.T @ Mfull @ X
        return dict(T=T @ X, K=0.5 * (Kc + Kc.T), M=0.5 * (Mc + Mc.T), method="macneal", omega_free=omega)
    raise ValueError("method must be 'rubin' or 'macneal'")


def torch_reduce_interface(cb_K, cb_M, cb_T, n_interface, n_modes, n_keep, device="cpu"):
    """Torch counterpart of `cms.reduce_interface` on the arrays of a Craig-Bampton model (K, M, T, n_interface,
    n_modes). Returns a dict with T, K, M, S, omega_interface."""
    Kc, Mc, Tc = _t(cb_K, device), _t(cb_M, device), _t(cb_T, device)
    nb, nm = int(n_interface), int(n_modes)
    n_keep = int(min(n_keep, nb))
    lam, X = _gen_eigh(Kc[:nb, :nb], Mc[:nb, :nb])
    S = torch.zeros(nb + nm, n_keep + nm, dtype=Kc.dtype, device=Kc.device)
    S[:nb, :n_keep] = X[:, :n_keep]
    S[nb:, n_keep:] = torch.eye(nm, dtype=Kc.dtype, device=Kc.device)
    K, M = S.T @ Kc @ S, S.T @ Mc @ S
    return dict(T=Tc @ S, K=0.5 * (K + K.T), M=0.5 * (M + M.T), S=S,
                omega_interface=torch.sqrt(torch.clamp(lam, min=0.0)))


# --------------------------------------------------------------------------------------- mode sensitivities
def torch_mode_sensitivities(K, M, dK, dM=None, n_modes=None, gap_tol=1e-8, device="cpu"):
    """Torch counterpart of `parametric_modes.mode_sensitivities` (exact, full eigen-decomposition). Returns
    (d_omega2, dPhi, info)."""
    Kt, Mt, dKt = _t(K, device), _t(M, device), _t(dK, device)
    dMt = torch.zeros_like(Mt) if dM is None else _t(dM, device)
    w2, P = _gen_eigh(Kt, Mt)
    N = P.shape[1]
    k = N if n_modes is None else int(n_modes)
    d_w2 = torch.zeros(k, dtype=Kt.dtype, device=Kt.device)
    dPhi = torch.zeros(Kt.shape[0], k, dtype=Kt.dtype, device=Kt.device)
    worst = float("inf")
    for i in range(k):
        phi = P[:, i]
        A = dKt - w2[i] * dMt
        d_w2[i] = phi @ A @ phi
        gap = w2[i] - w2
        rel = gap.abs() / max(abs(float(w2[i])), 1e-300)
        mask = torch.ones(N, dtype=torch.bool, device=gap.device)
        mask[i] = False
        if bool((rel[mask] < gap_tol).any()):
            raise ValueError("(nearly) repeated eigenvalues; sensitivities are not defined by Nelson's formula")
        worst = min(worst, float(rel[mask].min()) if N > 1 else float("inf"))
        coeff = torch.zeros(N, dtype=Kt.dtype, device=Kt.device)
        coeff[mask] = (P.T @ (A @ phi))[mask] / gap[mask]
        coeff[i] = -0.5 * (phi @ dMt @ phi)
        dPhi[:, i] = P @ coeff
    return d_w2, dPhi, dict(min_relative_gap=worst, n_modes_in_sum=N, exact=True)

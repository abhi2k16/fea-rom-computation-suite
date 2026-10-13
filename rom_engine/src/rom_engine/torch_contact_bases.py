# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
torch_contact_bases.py -- opt-in PyTorch counterparts of the dense, differentiable parts of the contact modules:
the non-negative dual basis (`contact_dual_basis.nonneg_dual_basis`), the SPD operator fit of
`operator_inference.fit_spd_operators`, the joint interface modes and Gram-Schmidt interface modes of
`interface_reduction`, and `rubin_dual_basis.reidentify_polynomial_force`.

Same convention as `torch_system_bases.py`: the NumPy path stays the default and needs no torch install; every
function is named `torch_*`, accepts NumPy arrays or tensors, returns tensors (float64, requested device), and is
NOT exported from the package (import this module explicitly). The LCP and active-set solvers are deliberately not
ported: they are not differentiable and not the bottleneck.

WHAT TORCH ADDS: autograd through the dense algebra (for example the sensitivity of interface modes to the joint
stiffness) and batched GPU linear algebra for large dense matrices. WHAT IS NOT CLAIMED: no speed-up; for the small
matrices of a reduced model the NumPy path is often faster. Every function is cross-checked against its NumPy
original in tests/test_torch_contact_bases.py, which skips cleanly when torch is not importable. Verified 2026-10-13: all 5 tests pass on a PyTorch-equipped
machine.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools
import numpy as np

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:       # a CUDA wheel without its runtime can fail at import with OSError/ValueError
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError("torch_contact_bases.py: PyTorch is not importable in this environment -- install it "
                          "(`pip install torch`) or use the NumPy functions of the same name without the torch_ "
                          "prefix.")


def _t(x, device="cpu"):
    _require_torch()
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float64)
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=device)


def _gen_eigh(K, M):
    L = torch.linalg.cholesky(0.5 * (M + M.T))
    Li = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], dtype=L.dtype, device=L.device), upper=False)
    w, Y = torch.linalg.eigh(Li @ (0.5 * (K + K.T)) @ Li.T)
    return w, Li.T @ Y


# ----------------------------------------------------------------------------------------- NMF dual basis
def _nndsvd(X, r):
    Uu, S, Vt = torch.linalg.svd(X, full_matrices=False)
    n, m = X.shape
    W = torch.zeros(n, r, dtype=X.dtype, device=X.device)
    H = torch.zeros(r, m, dtype=X.dtype, device=X.device)
    W[:, 0] = torch.sqrt(S[0]) * Uu[:, 0].abs()
    H[0] = torch.sqrt(S[0]) * Vt[0].abs()
    for j in range(1, min(r, len(S))):
        x, y = Uu[:, j], Vt[j]
        xp, xn = x.clamp(min=0), (-x).clamp(min=0)
        yp, yn = y.clamp(min=0), (-y).clamp(min=0)
        mp, mn = xp.norm() * yp.norm(), xn.norm() * yn.norm()
        if mp >= mn:
            u, v, sig = xp / (xp.norm() + 1e-300), yp / (yp.norm() + 1e-300), mp
        else:
            u, v, sig = xn / (xn.norm() + 1e-300), yn / (yn.norm() + 1e-300), mn
        W[:, j] = torch.sqrt(S[j] * sig) * u
        H[j] = torch.sqrt(S[j] * sig) * v
    fill = X.mean() / 100.0
    W = torch.where(W < 1e-12 * max(1.0, float(W.max())), fill, W)
    H = torch.where(H < 1e-12 * max(1.0, float(H.max())), fill, H)
    return W, H


def torch_nonneg_dual_basis(snapshots, rank, n_iter=500, tol=1e-9, device="cpu"):
    """Torch HALS NMF with the NNDSVDa start. Returns dict(U (unit columns), V, rel_error, n_iter)."""
    X = _t(snapshots, device)
    if bool((X < -1e-12).any()):
        raise ValueError("multiplier snapshots must be non-negative")
    X = X.clamp(min=0.0)
    n, m = X.shape
    r = int(min(rank, n, m))
    W, H = _nndsvd(X, r)
    nx = X.norm()
    prev, eps = float("inf"), 1e-14
    for it in range(1, n_iter + 1):
        HHt, XHt = H @ H.T, X @ H.T
        for k in range(r):
            W[:, k] = ((XHt[:, k] - W @ HHt[:, k] + W[:, k] * HHt[k, k]) / HHt[k, k].clamp(min=1e-300)).clamp(min=eps)
        WtW, WtX = W.T @ W, W.T @ X
        for k in range(r):
            H[k] = ((WtX[k] - WtW[k] @ H + WtW[k, k] * H[k]) / WtW[k, k].clamp(min=1e-300)).clamp(min=eps)
        err = float((X - W @ H).norm() / (nx + 1e-300))
        if abs(prev - err) < tol * max(err, 1e-300) or err < 1e-13:
            break
        prev = err
    norms = W.norm(dim=0)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return dict(U=W / norms, V=H * norms[:, None], rel_error=err, n_iter=it)


# ----------------------------------------------------------------------------------------- SPD operator fit
def torch_fit_spd_operators(X, A, F, max_iter=200, floor=1e-8, device="cpu"):
    """Torch counterpart of `operator_inference.fit_spd_operators`: unconstrained least squares first; if that is not
    SPD, Cholesky-parametrised refinement by L-BFGS on the data residual. Returns dict(M, K, spd_unconstrained,
    residual). Differentiable with respect to the data only when no refinement was needed."""
    X, A, F = _t(X, device), _t(A, device), _t(F, device)
    n = X.shape[0]
    D = torch.cat([A, X], dim=0)
    Wls = torch.linalg.lstsq(D.T, F.T).solution.T
    M0, K0 = Wls[:, :n], Wls[:, n:]
    M0, K0 = 0.5 * (M0 + M0.T), 0.5 * (K0 + K0.T)
    spd0 = bool(torch.linalg.eigvalsh(M0).min() > 0 and torch.linalg.eigvalsh(K0).min() > 0)
    fnorm = F.norm().clamp(min=1e-300)

    def chol(S):
        w, V = torch.linalg.eigh(S)
        w = w.clamp(min=1e-8 * float(w.abs().max()))
        return torch.linalg.cholesky(0.5 * ((V * w) @ V.T + ((V * w) @ V.T).T))
    if spd0:
        M, K = M0, K0
    else:
        idx = torch.tril_indices(n, n)
        LM = chol(M0)[idx[0], idx[1]].clone().requires_grad_(True)
        LK = chol(K0)[idx[0], idx[1]].clone().requires_grad_(True)

        def mats():
            Lm = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            Lk = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            Lm = Lm.index_put((idx[0], idx[1]), LM)
            Lk = Lk.index_put((idx[0], idx[1]), LK)
            return Lm @ Lm.T, Lk @ Lk.T
        opt = torch.optim.LBFGS([LM, LK], max_iter=max_iter, tolerance_grad=1e-14, tolerance_change=1e-16,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Mm, Kk = mats()
            loss = ((Mm @ A + Kk @ X - F).norm() / fnorm) ** 2
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            M, K = mats()
    eye = torch.eye(n, dtype=X.dtype, device=X.device)
    M = M + floor * torch.trace(M) / n * eye
    K = K + floor * torch.trace(K) / n * eye
    return dict(M=M, K=K, spd_unconstrained=spd0, residual=float((M @ A + K @ X - F).norm() / fnorm))


# ----------------------------------------------------------------------------------------- interface modes
def torch_joint_interface_modes(Kbb, Mbb, joint, n_modes, device="cpu"):
    """Torch counterpart of `interface_reduction.joint_interface_modes` on the boundary block of a Craig-Bampton
    model. Returns (modes (n_joint, k), omega)."""
    Kb, Mb = _t(Kbb, device), _t(Mbb, device)
    nb = Kb.shape[0]
    joint = torch.as_tensor(np.asarray(joint, int), device=Kb.device)
    rest = torch.as_tensor(np.setdiff1d(np.arange(nb), joint.cpu().numpy()), device=Kb.device)
    T = torch.zeros(nb, len(joint), dtype=Kb.dtype, device=Kb.device)
    T[joint, torch.arange(len(joint))] = 1.0
    if len(rest):
        T[rest] = -torch.linalg.solve(Kb[rest][:, rest], Kb[rest][:, joint])
    w2, V = _gen_eigh(T.T @ Kb @ T, T.T @ Mb @ T)
    k = int(min(n_modes, len(joint)))
    return V[:, :k], torch.sqrt(w2[:k].clamp(min=0.0))


def torch_gram_schmidt_interface_modes(mode_sets, interface, tol=1e-8, weights=None, n_keep=None, device="cpu"):
    """Torch counterpart of `interface_reduction.gram_schmidt_interface_modes`."""
    b = torch.as_tensor(np.asarray(interface, int), device=device)
    w = torch.ones(len(b), dtype=torch.float64, device=device) if weights is None else _t(weights, device)
    Q = []
    for S in mode_sets:
        S = _t(S, device)[b]
        for c in S.T:
            v = c.clone()
            n0 = torch.sqrt(v @ (w * v))
            if float(n0) == 0.0:
                continue
            for _ in range(2):
                for q in Q:
                    v = v - (q @ (w * v)) * q
            nv = torch.sqrt(v @ (w * v))
            if float(nv) > tol * float(n0):
                Q.append(v / nv)
            if n_keep is not None and len(Q) >= n_keep:
                return torch.stack(Q, dim=1)
    return torch.stack(Q, dim=1) if Q else torch.zeros(len(b), 0, dtype=torch.float64, device=device)


# ----------------------------------------------------------------------------------------- polynomial re-fit
def torch_reidentify_polynomial_force(M_r, C_r, K_r, q, qd, qdd, f_ext, orders=(2, 3), device="cpu"):
    """Torch counterpart of `rubin_dual_basis.reidentify_polynomial_force` (least squares; differentiable).
    Returns dict(monomials, coeff (n_terms, r), relative_residual)."""
    q, qd, qdd, f = (_t(v, device) for v in (q, qd, qdd, f_ext))
    Mr, Cr, Kr = _t(M_r, device), _t(C_r, device), _t(K_r, device)
    nt, r = q.shape
    g = f - qdd @ Mr.T - qd @ Cr.T - q @ Kr.T
    monomials = []
    for o in orders:
        monomials += list(itertools.combinations_with_replacement(range(r), o))
    X = torch.stack([torch.prod(q[:, list(m)], dim=1) for m in monomials], dim=1)
    coeff = torch.linalg.lstsq(X, g, driver="gelsd").solution
    res = float((X @ coeff - g).norm() / g.norm().clamp(min=1e-300))
    return dict(monomials=monomials, coeff=coeff, relative_residual=res)

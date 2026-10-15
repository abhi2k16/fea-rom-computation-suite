# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batch_conformance.py -- reusable checks for any batched / sparse nonlinear element kernel.

A batched kernel is accepted only if it passes ALL of these against the independent per-element loop code:

  equal_to_loop        batched force / tangent equal the per-element loop results (default rtol 1e-12 relative to the
                       largest entry of the loop result);
  objective            a rigid rotation plus translation of the current configuration gives zero internal force;
  fd_tangent           the tangent equals the central finite difference of the internal force, and is symmetric;
  chunk_invariant      results do not depend on the chunk size (sizes 1, a small value and larger than the block);
  same_newton_path     a Newton solve with the batched kernel reproduces the loop's converged displacements.

The helpers take plain callables so they can be used with an element class (``internal_force`` / ``tangent_stiffness``
versus ``internal_force_batch`` / ``tangent_stiffness_batch``), with FESystem assembly, or with a sparse variant.
Each helper returns a dict of measured errors and raises AssertionError when a tolerance is exceeded, so the measured
numbers can be printed by a benchmark script and asserted by a test with the same code.

Nothing here asserts speed: timing is reported by tests/batch_timing.py and is never a pass/fail criterion.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

DEFAULT_RTOL = 1.0e-12


def rel_err(a, b):
    """max |a - b| / max(max |b|, tiny). Works for dense arrays and scipy sparse matrices."""
    if hasattr(a, "toarray"):
        a = a.toarray()
    if hasattr(b, "toarray"):
        b = b.toarray()
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.max(np.abs(b))) if b.size else 0.0, 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def equal_to_loop(batched, loop, rtol=DEFAULT_RTOL, label="result"):
    """Compare a batched result with the loop result; return the relative error."""
    err = rel_err(batched, loop)
    assert err <= rtol, f"{label}: batched differs from loop, relative error {err:.3e} > {rtol:.1e}"
    return {label: err}


def rigid_motion(X, theta=0.7, shift=None):
    """Displacement field u = (R - I) X + shift (2-D: rotation about z; 3-D: about the axis (1,2,3)/sqrt(14))
    for nodal coordinates X of shape (n, dim). Returns the flattened (n*dim,) vector."""
    n, dim = X.shape
    if dim == 2:
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    else:
        k = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    shift = np.full(dim, 0.3) if shift is None else np.asarray(shift, float)
    return (((R - np.eye(dim)) @ X.T).T + shift).ravel()


def objective(force_fn, X, reference_force_scale, theta=0.7, tol=1e-9):
    """Internal force at a rigid motion of nodal coordinates X (n, dim) must vanish.
    ``force_fn(u)`` returns the (n*dim,) element/system force. The error is reported relative to
    ``reference_force_scale`` (the force of a typical 1 % stretch), as the project's tests do."""
    f = np.asarray(force_fn(rigid_motion(X, theta)))
    err = float(np.max(np.abs(f))) / reference_force_scale
    assert err <= tol, f"objectivity: force at rigid motion is {err:.3e} of the reference scale (> {tol:.1e})"
    return {"objectivity": err}


def fd_tangent(force_fn, tangent_fn, u, h=1e-6, rtol=1e-6, sym_tol=1e-10):
    """Tangent versus central differences of the force; also checks symmetry. Dense, so for small systems only."""
    u = np.asarray(u, float)
    K = tangent_fn(u)
    K = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
    Kfd = np.column_stack([(force_fn(u + h * e) - force_fn(u - h * e)) / (2 * h) for e in np.eye(u.size)])
    scale = max(1.0, float(np.max(np.abs(K))))
    err_fd = float(np.max(np.abs(K - Kfd))) / scale
    err_sym = float(np.max(np.abs(K - K.T))) / scale
    assert err_fd <= rtol, f"tangent vs finite difference: {err_fd:.3e} > {rtol:.1e}"
    assert err_sym <= sym_tol, f"tangent asymmetry: {err_sym:.3e} > {sym_tol:.1e}"
    return {"fd_tangent": err_fd, "symmetry": err_sym}


def chunk_invariant(fn, n_elements, chunk_sizes=None, rtol=DEFAULT_RTOL):
    """``fn(chunk_size)`` returns an array (or sparse matrix); results for different chunk sizes must agree to
    rtol (floating-point summation order differs, so exact equality is not expected)."""
    if chunk_sizes is None:
        chunk_sizes = [1, max(2, n_elements // 3), n_elements + 7]
    ref = fn(None)
    out = {}
    for c in chunk_sizes:
        out[f"chunk_{c}"] = rel_err(fn(c), ref)
        assert out[f"chunk_{c}"] <= rtol, f"chunk size {c}: relative difference {out[f'chunk_{c}']:.3e} > {rtol:.1e}"
    return out


def same_newton_path(U_batched, U_loop, rtol=1e-9):
    """Converged displacements at every load step must agree (for smooth, non-contact cases)."""
    err = rel_err(U_batched, U_loop)
    assert err <= rtol, f"Newton path: batched and loop displacements differ by {err:.3e} > {rtol:.1e}"
    return {"newton_path": err}

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_greedy.py -- weak-greedy snapshot selection for parametric contact ROMs, driven by the violation of the
contact conditions (Balajewicz, Amsallem and Farhat 2015, Section 4.5).

A greedy loop needs an error indicator that does not require the full-order solution. For contact the reduced
solution u_r can be checked against the three Hertz-Signorini-Moreau conditions directly:

    r1 = B u - c                                     non-penetration (must be >= 0),
    r2 = lam_hat * (B u - c)                         complementary slackness (must be 0),
    r3 = lam_hat = (B^T)^+ (K u - f)                 contact-force positivity (must be >= 0),

and the indicator is  I = a1 |[r1]_-|^2 + a2 |r2|^2 + a3 |[r3]_-|^2  with [x]_- = min(x, 0). Evaluating it needs
matrix-vector products only, no solve; a1 = 1, a2 = a3 = 0 is the cheapest and, in the source's static test, the best.

`greedy_contact_sampling` runs the loop for any parametrisation: you supply the full-order solver, a function that
builds a ROM from the snapshots collected so far, and the indicator. The loop adds the candidate with the largest
indicator and stops at `tol` or `max_snapshots`.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["contact_condition_indicator", "GreedyContactHistory", "greedy_contact_sampling"]


def contact_condition_indicator(K, B, f, c, u, weights=(1.0, 0.0, 0.0)):
    """Violation of the contact conditions by a (reduced) displacement u, per the formula above. `weights` =
    (a1, a2, a3). B may be dense or sparse. Returns a non-negative float (0 = the conditions all hold)."""
    u = np.asarray(u, float)
    gap = B @ u - np.asarray(c, float)
    a1, a2, a3 = weights
    val = 0.0
    if a1:
        val += a1 * np.sum(np.minimum(gap, 0.0) ** 2)
    if a2 or a3:
        Bd = B.toarray() if hasattr(B, "toarray") else np.asarray(B, float)
        lam_hat = np.linalg.lstsq(Bd.T, K @ u - f, rcond=None)[0]
        if a2:
            val += a2 * np.sum((lam_hat * gap) ** 2)
        if a3:
            val += a3 * np.sum(np.minimum(lam_hat, 0.0) ** 2)
    return float(val)


class GreedyContactHistory:
    """selected (indices into the candidate list, in order), indicator_max (maximum indicator over the remaining
    candidates after each round), n_snapshots, converged."""

    def __init__(self):
        self.selected, self.indicator_max, self.converged = [], [], False

    @property
    def n_snapshots(self):
        return len(self.selected)


def greedy_contact_sampling(candidates, fom_solve, make_rom, indicator, max_snapshots, tol=0.0, first=0):
    """Weak greedy loop.

    candidates       list of parameter values;
    fom_solve(p)     returns (u, lam), the full-order displacement and multiplier snapshots at p;
    make_rom(U, L)   builds a reduced model from the displacement snapshots U (n x k) and multiplier snapshots L
                     (m x k) collected so far and returns a callable rom(p) -> u_reduced_lifted (n,);
    indicator(p, u) the error indicator for a ROM displacement u at parameter p (see contact_condition_indicator);
    first            index of the initial candidate.

    Returns (rom, GreedyContactHistory). Every round costs one full-order solve plus one cheap ROM solve and one
    indicator evaluation per remaining candidate."""
    hist = GreedyContactHistory()
    U, L = [], []
    chosen = int(first)
    rom = None
    for _ in range(int(max_snapshots)):
        u, lam = fom_solve(candidates[chosen])
        U.append(u)
        L.append(lam)
        hist.selected.append(chosen)
        rom = make_rom(np.array(U).T, np.array(L).T)
        rest = [i for i in range(len(candidates)) if i not in hist.selected]
        if not rest:
            hist.converged = True
            break
        vals = np.array([indicator(candidates[i], rom(candidates[i])) for i in rest])
        hist.indicator_max.append(float(vals.max()))
        if vals.max() <= tol:
            hist.converged = True
            break
        chosen = rest[int(np.argmax(vals))]
    return rom, hist

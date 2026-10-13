# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_validation.py -- error metrics for reduced contact models (plain arrays).

All errors are normalised by the maximum norm of the reference (Manvelyan-Stroot et al. 2025), so a small
reference value cannot inflate the ratio. Inputs are arrays of shape (n,) or (n_steps, n) / (n, n_cases).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["max_normalised_error", "contact_errors", "ContactErrors", "contact_set_agreement"]


def max_normalised_error(approx, reference):
    """max |approx - reference| / max |reference| (0 if both vanish)."""
    a, r = np.asarray(approx, float), np.asarray(reference, float)
    scale = np.max(np.abs(r))
    err = np.max(np.abs(a - r)) if a.size else 0.0
    return 0.0 if scale == 0 and err == 0 else err / max(scale, np.finfo(float).tiny)


class ContactErrors:
    """displacement, gap, force: max-normalised errors; complementarity: max violation of gap>=0, lam>=0,
    lam*gap=0 of the approximation; active_agreement: Jaccard overlap of the active contact sets."""

    def __init__(self, displacement, gap, force, complementarity, active_agreement):
        self.displacement, self.gap, self.force = displacement, gap, force
        self.complementarity, self.active_agreement = complementarity, active_agreement

    def as_dict(self):
        return dict(self.__dict__)

    def __repr__(self):
        return ("ContactErrors(" + ", ".join(f"{k}={v:.3e}" for k, v in self.__dict__.items()) + ")")


def contact_set_agreement(lam_a, lam_b, tol=0.0):
    """Jaccard index of {lam_a > tol} and {lam_b > tol}; 1 if both are empty."""
    a, b = np.asarray(lam_a) > tol, np.asarray(lam_b) > tol
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def contact_errors(u, lam, gap, u_ref, lam_ref, gap_ref):
    """Compare a reduced solution (u, lam, gap) to a full-order reference."""
    lam, gap = np.asarray(lam, float), np.asarray(gap, float)
    comp = 0.0 if lam.size == 0 else float(max(np.max(np.maximum(-gap, 0)), np.max(np.maximum(-lam, 0)),
                                               np.max(np.abs(lam * gap))))
    return ContactErrors(max_normalised_error(u, u_ref), max_normalised_error(gap, gap_ref),
                         max_normalised_error(lam, lam_ref), comp, contact_set_agreement(lam, lam_ref))

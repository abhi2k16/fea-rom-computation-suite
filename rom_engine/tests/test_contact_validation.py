# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the contact error metrics, including a check on the obstacle-membrane fixture."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.contact_validation import contact_errors, max_normalised_error, contact_set_agreement


def test_max_normalised_error_and_zero_cases():
    assert max_normalised_error([1.0, 2.2], [1.0, 2.0]) == pytest.approx(0.1)
    assert max_normalised_error([0.0], [0.0]) == 0.0
    assert max_normalised_error([[1, 2], [3, 4]], [[1, 2], [3, 4]]) == 0.0


def test_contact_set_agreement():
    assert contact_set_agreement([0, 1, 1], [0, 1, 0]) == pytest.approx(0.5)
    assert contact_set_agreement([0, 0], [0, 0]) == 1.0


def test_contact_errors_exact_and_violating():
    u, lam, gap = np.array([1.0, 2.0]), np.array([0.0, 3.0]), np.array([0.5, 0.0])
    e = contact_errors(u, lam, gap, u, lam, gap)
    assert e.displacement == e.gap == e.force == e.complementarity == 0.0 and e.active_agreement == 1.0
    bad = contact_errors(u, np.array([-0.2, 3.0]), np.array([0.5, 0.1]), u, lam, gap)
    assert bad.complementarity == pytest.approx(0.3) and bad.force > 0

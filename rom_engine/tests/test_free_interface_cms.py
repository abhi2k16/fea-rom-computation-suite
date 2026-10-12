# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""test_free_interface_cms.py -- free-interface (Rubin, MacNeal) CMS and characteristic constraint modes."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh

from rom_engine.cms import craig_bampton, free_interface_cms, reduce_interface


def chain(n=40, k=1.0e4, m=1.0):
    """Fixed-free spring-mass chain (non-singular K); interface = the free end and the middle node."""
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k
        if i + 1 < n:
            K[i + 1, i + 1] += k
            K[i, i + 1] = K[i + 1, i] = -k
    return K, m * np.eye(n)


def freqs(K, M, k):
    return np.sqrt(eigh(K, M, eigvals_only=True)[:k])


def test_free_interface_models_converge_and_rubin_is_at_least_as_good():
    K, M = chain()
    ref = freqs(K, M, 6)
    errs = {}
    for method in ("rubin", "macneal"):
        e = []
        for nm in (3, 6, 10):
            m = free_interface_cms(K, M, [39, 20], nm, method)
            f = freqs(m.K, m.M, 3)
            e.append(np.max(np.abs(f - ref[:3]) / ref[:3]))
        errs[method] = e
        assert e[-1] <= e[0] + 1e-12
    assert errs["rubin"][0] < 1e-2
    assert all(r <= mc * 1.0001 + 1e-12 for r, mc in zip(errs["rubin"], errs["macneal"]))


def test_all_free_interface_modes_reproduce_the_full_model():
    K, M = chain(12)
    m = free_interface_cms(K, M, [11], 12, "rubin")
    assert np.allclose(freqs(m.K, m.M, 5), freqs(K, M, 5), rtol=1e-8)


def test_free_interface_rejects_singular_k_and_unknown_method():
    K, M = chain(6)
    K[0, 0] -= 1.0e4                       # free-free chain: singular
    with pytest.raises(ValueError):
        free_interface_cms(K, M, [5], 3)
    K, M = chain(6)
    with pytest.raises(ValueError):
        free_interface_cms(K, M, [5], 3, "bogus")


def test_characteristic_constraint_modes_are_exact_when_all_kept_and_degrade_gracefully():
    K, M = chain(40)
    cb = craig_bampton(K, M, [39, 30, 20, 10], n_modes=8)
    ref = freqs(K, M, 4)
    full = reduce_interface(cb, 4)
    assert np.allclose(freqs(full.K, full.M, 4), freqs(cb.K, cb.M, 4), rtol=1e-9)
    errs = []
    for keep in (1, 2, 4):
        r = reduce_interface(cb, keep)
        errs.append(np.max(np.abs(freqs(r.K, r.M, 4) - ref) / ref))
    assert errs[2] <= errs[0] + 1e-12 and r.n_interface_kept == 4
    assert reduce_interface(cb, 2).K.shape == (2 + 8, 2 + 8)

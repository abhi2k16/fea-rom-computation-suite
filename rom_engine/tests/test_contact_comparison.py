# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""The contact comparison harness on the obstacle membrane."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from contact_fixtures import ObstacleMembrane
from rom_engine.contact_comparison import compare_contact_models, format_contact_comparison
from rom_engine.contact_dual_basis import nonneg_dual_basis, svd_dual_basis, ContactROM


def test_harness_reports_the_known_advantage_of_the_nmf_dual_basis():
    m = ObstacleMembrane(n=14)
    train = [(a, b) for a in np.linspace(0.3, 0.6, 5) for b in np.linspace(0.0, 0.1, 4)]
    Xu, Xl = m.training_set(train)
    U = np.linalg.svd(Xu, full_matrices=False)[0][:, :12]
    roms = {"svd dual": ContactROM(m.K, m.B, U, svd_dual_basis(Xl, 12)),
            "nmf dual": ContactROM(m.K, m.B, U, nonneg_dual_basis(Xl, 12).U)}

    def fom(g):
        u, lam, c = m.solve(g)
        return u, lam, m.B @ u - c

    def make(rom):
        def predict(g):
            c = m.obstacle(g)
            r = rom.solve(m.f, c)
            return r.u, r.lam, r.gap
        return predict
    variants = {k: dict(predict=make(r), offline_time=0.5, size=r.sizes) for k, r in roms.items()}
    tests = [(0.37, 0.03), (0.52, 0.08), (0.44, 0.05)]
    table = compare_contact_models(fom, variants, tests, n_queries=100)
    nmf, svd = table["nmf dual"], table["svd dual"]
    assert nmf["displacement"] < 3e-2 and nmf["displacement"] < svd["displacement"]
    assert nmf["complementarity"] < svd["complementarity"] or nmf["complementarity"] < 1e-6
    assert nmf["size"] == (12, 12) and nmf["t_online"] > 0 and nmf["t_fom"] > 0
    assert nmf["effective_speedup"] == pytest.approx(100 * nmf["t_fom"] / (100 * nmf["t_online"] + 0.5))
    text = format_contact_comparison(table)
    assert "nmf dual" in text and "svd dual" in text

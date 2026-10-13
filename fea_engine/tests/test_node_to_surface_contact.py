# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the 3-D node-to-surface penalty contact element and its geometry helpers."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine.elements.contact import (NodeToSurfaceContact3D, closest_point_on_triangle_3d,
                                         find_contact_pairs_3d)

A, B, C = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])   # outward normal +z


def _coords(p):
    return np.vstack([p, A, B, C])


def test_closest_point_regions():
    cases = {
        "interior": ([0.2, 0.3, 1.0], [0.2, 0.3, 0.0]),
        "corner a": ([-1.0, -1.0, 0.5], A),
        "corner b": ([2.0, -1.0, 0.0], B),
        "corner c": ([-1.0, 2.0, 0.0], C),
        "edge ab": ([0.5, -1.0, 0.2], [0.5, 0.0, 0.0]),
        "edge ac": ([-1.0, 0.5, 0.2], [0.0, 0.5, 0.0]),
        "edge bc": ([1.0, 1.0, 0.0], [0.5, 0.5, 0.0]),
    }
    for name, (p, expected) in cases.items():
        w, proj = closest_point_on_triangle_3d(p, A, B, C)
        assert np.allclose(proj, expected), name
        assert w.min() >= -1e-14 and w.sum() == pytest.approx(1.0), name
        assert np.allclose(w[0] * A + w[1] * B + w[2] * C, proj), name


def test_closest_point_is_the_minimum_distance_over_random_points():
    rng = np.random.default_rng(1)
    tri = [rng.standard_normal(3) for _ in range(3)]
    for _ in range(50):
        p = rng.standard_normal(3) * 2
        _, proj = closest_point_on_triangle_3d(p, *tri)
        w = rng.dirichlet(np.ones(3), size=400)
        samples = w @ np.array(tri)
        assert np.linalg.norm(p - proj) <= np.min(np.linalg.norm(samples - p, axis=1)) + 1e-12


def test_force_analytic_and_equilibrated():
    el = NodeToSurfaceContact3D()
    k = 1e4
    X = _coords(np.array([0.2, 0.3, 0.0]))
    assert np.allclose(el.internal_force(X, np.zeros(12), (k,)), 0)          # touching, no penetration
    u = np.zeros(12)
    u[2] = -0.01                                                              # slave 0.01 below the surface
    f = el.internal_force(X, u, (k,))
    assert np.allclose(f[0:3], [0, 0, -k * 0.01])                             # -k delta n
    assert np.allclose(f[0:3] + f[3:6] + f[6:9] + f[9:12], 0, atol=1e-12)     # Newton's third law
    w = np.array([1 - 0.2 - 0.3, 0.2, 0.3])
    assert np.allclose(f[3:6], [0, 0, k * 0.01 * w[0]])
    u[2] = +0.01                                                              # separated
    assert np.allclose(el.internal_force(X, u, (k,)), 0)


def test_force_is_gradient_of_the_penalty_energy_for_slave_and_master_translations():
    el = NodeToSurfaceContact3D()
    k = 5e3
    X = _coords(np.array([0.3, 0.2, 0.0]))
    u = np.zeros(12)
    u[0:3] = [0.01, -0.02, -0.03]

    def energy(uu):
        g = el._geometry(X, uu)
        return 0.5 * k * max(g["delta"], 0.0) ** 2

    f = el.internal_force(X, u, (k,))
    h = 1e-7
    for dof in range(3):                                                       # slave
        e = np.zeros(12); e[dof] = h
        assert (energy(u + e) - energy(u - e)) / (2 * h) == pytest.approx(f[dof], rel=1e-5, abs=1e-8)
    for dof in range(3):                                                       # translate the whole master
        e = np.zeros(12); e[3 + dof::3][:3] = h
        assert (energy(u + e) - energy(u - e)) / (2 * h) == pytest.approx(f[3:].reshape(3, 3).sum(0)[dof],
                                                                          rel=1e-5, abs=1e-8)


def test_consistent_tangent_properties_and_agreement_with_finite_difference_for_small_force():
    el = NodeToSurfaceContact3D()
    k = 1e4
    X = _coords(np.array([0.3, 0.2, 0.0]))
    u = np.zeros(12)
    u[2] = -1e-4
    Kc = el.tangent_stiffness(X, u, (k,))
    assert np.allclose(Kc, Kc.T)
    assert np.linalg.matrix_rank(Kc) == 1 and np.linalg.eigvalsh(Kc).min() > -1e-9
    rigid = np.tile([0.3, -0.2, 0.5], 4)
    assert np.allclose(Kc @ rigid, 0, atol=1e-9)                              # rigid translation: no force change
    Kfd = el.tangent_stiffness(X, u, (k,), consistent=False)
    # normal-direction stiffness of the slave agrees; the rest differs only by terms proportional to f_c
    assert Kc[2, 2] == pytest.approx(Kfd[2, 2], rel=1e-4)
    assert np.max(np.abs(Kc - Kfd)) <= 2 * k * 1e-4 * 4 + 1e-8


def test_penalty_equilibrium_under_applied_load():
    el = NodeToSurfaceContact3D()
    k, P = 2e4, 50.0
    X = _coords(np.array([0.25, 0.25, 0.0]))
    u = np.zeros(12)
    for _ in range(20):                                                        # Newton on the 3 slave DOFs
        R = el.internal_force(X, u, (k,))[0:3] - np.array([0, 0, -P])          # external load pushes down
        Kt = el.tangent_stiffness(X, u, (k,))[0:3, 0:3] + 1e-6 * np.eye(3)
        u[0:3] -= np.linalg.solve(Kt, R)
        if np.linalg.norm(R) < 1e-9:
            break
    assert u[2] == pytest.approx(-P / k, rel=1e-6)
    assert np.allclose(u[0:2], 0, atol=1e-9)


def test_find_contact_pairs_3d():
    X = np.array([[0.2, 0.2, 0.05], [5.0, 5.0, 0.0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    pairs = find_contact_pairs_3d([0, 1], [(2, 3, 4)], X, 0.1)
    assert pairs == [(0, (2, 3, 4))]
    assert find_contact_pairs_3d([0, 1], [(2, 3, 4)], X, 0.01) == []

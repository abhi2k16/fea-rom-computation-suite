# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for `solve_contact_static_sparse` (fea_engine.contact_solvers): the routes "reduced", "saddle" and "condensed",
the failure flag with its reason and suggestion, the opt-in fallback, and the warm start.

Reference everywhere: the dense routes of `solve_contact_static` (active set, Lemke), which were validated earlier.
The membrane is the Q4 Laplace stiffness on a uniform grid (Kronecker form, built here because the package has no
scalar element yet)."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp

from fea_engine import FESystem, Material, solve_contact_static, solve_contact_static_sparse, ContactResult
from fea_engine import mesh as meshlib
from fea_engine.elements import Quad4PlaneStress
from fea_engine.material import D_plane_stress


def membrane(ne, load=-10.0):
    n = ne - 1
    h = 1.0 / ne
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    Mm = sp.diags([1, 4, 1], [-1, 0, 1], shape=(n, n)) / 6.0
    K = (sp.kron(T / h ** 2, Mm) + sp.kron(Mm, T / h ** 2)).tocsr() * h ** 2
    x = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return K, load * h ** 2 * np.ones(n * n), X.ravel(), Y.ravel()


def obstacles(X, Y):
    rng = np.random.default_rng(0)
    bump = (-1 + 0.4 * np.exp(-200 * ((X - 0.45) ** 2 + (Y - 0.5) ** 2)) + 0.4 * np.exp(-355.56 * ((X - 0.7) ** 2 + (Y - 0.5) ** 2)))
    return {"bump": bump, "plateau -0.5": np.full(X.size, -0.5), "plateau -0.3": np.full(X.size, -0.3),
            "ridge": -0.2 - 0.8 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2),
            "rough": -0.45 + 0.1 * rng.standard_normal(X.size), "tilted": -0.2 - 0.6 * X}


K_M, F_M, X_M, Y_M = membrane(23)           # 484 unknowns
OBST = obstacles(X_M, Y_M)
EYE = sp.identity(K_M.shape[0], format="csr")


def dense(K, f, B, c, method="active_set"):
    return solve_contact_static(K.toarray() if sp.issparse(K) else K, f, B.toarray() if sp.issparse(B) else B, c, method=method)


def assert_kkt(r, tol=1e-9):
    assert r.converged
    assert r.gap.min() >= -tol and r.lam.min() >= -tol and np.max(np.abs(r.lam * r.gap)) <= tol


# ------------------------------------------------------------------------------------------ reduced route
@pytest.mark.parametrize("name", list(OBST))
def test_reduced_equals_dense_routes_on_the_membrane(name):
    g = OBST[name]
    r = solve_contact_static_sparse(K_M, F_M, EYE, g, route="reduced")
    d = dense(K_M, F_M, EYE, g)
    assert_kkt(r)
    assert r.route == "reduced"
    assert np.abs(r.u - d.u).max() < 1e-12 and np.abs(r.lam - d.lam).max() < 1e-12
    assert np.array_equal(r.active_mask, d.lam > 0) or np.abs(r.lam[r.active_mask != (d.lam > 0)]).max() < 1e-10


def test_the_membrane_cases_include_large_active_sets():
    counts = [int((solve_contact_static_sparse(K_M, F_M, EYE, g, route="reduced").lam > 1e-12).sum()) for g in OBST.values()]
    assert min(counts) >= 1 and max(counts) > 100, counts


def test_reduced_with_scaled_selection_rows():
    """B = diag(s) with s > 0 is still a selection: s u >= c."""
    n = K_M.shape[0]
    s = 0.5 + np.linspace(0, 1, n)
    B = sp.diags(s).tocsr()
    c = s * OBST["plateau -0.5"]
    r = solve_contact_static_sparse(K_M, F_M, B, c, route="reduced")
    d = dense(K_M, F_M, B, c)
    assert_kkt(r)
    assert np.abs(r.u - d.u).max() < 1e-11 and np.abs(r.lam - d.lam).max() < 1e-10


def test_reduced_is_not_applicable_to_a_general_B_and_says_so():
    B = sp.csr_matrix(np.array([[1.0, 1.0, 0, 0, 0, 0]]))
    K = np.eye(6) * 2
    r = solve_contact_static_sparse(K, np.ones(6), B, np.array([0.5]), route="reduced")
    assert not r.converged and r.reason == "not_applicable" and r.suggestion == "saddle" and "saddle" in r.message
    r2 = solve_contact_static_sparse(K, np.ones(6), B, np.array([0.5]), route="reduced", fallback=True)
    assert r2.converged and r2.route == "saddle" and r2.fallbacks == ["reduced"]


# ------------------------------------------------------------------------------------------- saddle route
@pytest.mark.parametrize("name", ["bump", "plateau -0.3", "ridge"])
def test_saddle_equals_dense_routes_on_the_membrane(name):
    g = OBST[name]
    r = solve_contact_static_sparse(K_M, F_M, EYE, g, route="saddle")
    d = dense(K_M, F_M, EYE, g)
    assert_kkt(r)
    assert np.abs(r.u - d.u).max() < 1e-11 and np.abs(r.lam - d.lam).max() < 1e-10


def test_two_plate_gap_matrix():
    """Two membranes, one loaded upward: B = [-I, I], w2 - w1 >= -gap (the structure of the two-plate problem)."""
    K1, f1, _, _ = membrane(11, load=+60.0)
    K2, _, _, _ = membrane(11)
    n = K1.shape[0]
    K = sp.block_diag([K1, K2]).tocsr()
    f = np.concatenate([f1, np.zeros(n)])
    B = sp.hstack([-sp.identity(n), sp.identity(n)]).tocsr()
    c = np.full(n, -0.15)
    r = solve_contact_static_sparse(K, f, B, c)          # auto -> saddle (B is not a selection)
    d = dense(K, f, B, c, method="lemke")
    assert r.route == "saddle"
    assert_kkt(r)
    assert (r.lam > 1e-12).sum() > 3
    assert np.abs(r.u - d.u).max() < 1e-9 and np.abs(r.lam - d.lam).max() < 1e-8


def test_auto_selects_the_route_from_the_structure_of_B():
    assert solve_contact_static_sparse(K_M, F_M, EYE, OBST["bump"]).route == "reduced"
    K = np.eye(4) * 3
    B = sp.csr_matrix(np.array([[1.0, -1.0, 0, 0], [0, 1.0, 1.0, 0]]))
    assert solve_contact_static_sparse(K, np.array([1.0, -2.0, 1.0, 0]), B, np.array([-0.1, -0.1])).route == "saddle"


def test_random_spd_with_general_sparse_B_agrees_with_dense_or_reports_failure():
    rng = np.random.default_rng(3)
    agree = failed = 0
    for _ in range(120):
        n = int(rng.integers(5, 10))
        Q = rng.standard_normal((n, n))
        K = Q @ Q.T + 0.1 * np.eye(n)
        m = int(rng.integers(2, n))
        B = sp.csr_matrix(rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.4) + 1e-3 * rng.standard_normal((m, n)))
        f, c = rng.standard_normal(n) * 3, rng.standard_normal(m) * 0.5
        r = solve_contact_static_sparse(K, f, B, c, route="saddle")
        if r.converged:
            d = dense(K, f, B, c, method="lemke")
            assert np.abs(r.u - d.u).max() < 1e-8
            agree += 1
        else:
            assert r.reason in ("cycled", "max_iter", "singular") and r.suggestion == "condensed"
            failed += 1
    assert agree > 60, (agree, failed)


def test_elasticity_with_inclined_constraints_agrees_or_reports_failure():
    """A plane-stress stiffness (not an M-matrix) with inclined unilateral constraints n.u >= c on the bottom edge."""
    m = meshlib.rectangle_mesh(2.0, 1.0, 12, 6)
    fes = FESystem(m, Quad4PlaneStress(), sparse=True)
    fes.assemble_stiffness(D_plane_stress(Material(E=1000.0, nu=0.3)), thickness=1.0)
    m.select_nodes(x=0.0, name="left")
    fes.fix_dofs("left", ["ux", "uy"])
    free = fes.free_dofs
    K = sp.csr_matrix(fes.K)[free][:, free]
    pos = {int(d): i for i, d in enumerate(free)}
    bottom = [i for i in range(len(m.nodes)) if abs(m.nodes[i, 1]) < 1e-12 and m.nodes[i, 0] > 1e-9]
    rows, cols, vals = [], [], []
    for k, node in enumerate(bottom):
        a = 0.3 * np.sin(2.0 * m.nodes[node, 0])
        for comp, v in ((0, -np.sin(a)), (1, np.cos(a))):
            rows.append(k); cols.append(pos[2 * node + comp]); vals.append(v)
    B = sp.csr_matrix((vals, (rows, cols)), shape=(len(bottom), K.shape[0]))
    f = np.zeros(K.shape[0])
    for node in range(len(m.nodes)):
        if 2 * node + 1 in pos:
            f[pos[2 * node + 1]] = -2.0
    c = np.full(len(bottom), -0.004)
    r = solve_contact_static_sparse(K, f, B, c, route="saddle")
    d = dense(K, f, B, c)
    if r.converged:
        assert_kkt(r)
        assert np.abs(r.u - d.u).max() < 1e-9 * max(1.0, np.abs(d.u).max())
        assert (r.lam > 1e-12).sum() >= 1
    else:                                              # allowed: failure must come with a way out
        assert r.suggestion == "condensed"
        r2 = solve_contact_static_sparse(K, f, B, c, route="saddle", fallback=True)
        assert r2.converged and np.abs(r2.u - d.u).max() < 1e-8


# ------------------------------------------------------------------------------- failures and fallbacks
# a 6 x 6 problem on which the active-set iteration of the reduced route cycles (found by a seeded search over
# random SPD matrices with B = I; entries rounded to 6 digits so that the instance is stable)
_Q = [[-0.655408, -0.846941, -0.708724, -1.912458, -0.443349, -0.276862], [1.107316, 0.070326, -0.264244, 1.89554, 0.662529, -1.471969],
      [-0.448556, -0.194671, 0.630708, 0.547168, 0.140832, 1.606428], [0.804694, 0.120947, 1.20148, -0.417223, -1.311414, -0.806885],
      [0.765977, 1.615742, 2.02264, 0.561309, 0.253741, 0.38904], [-1.026379, -1.541998, -1.753003, 0.595986, 0.690404, 0.622938]]
_F = [-1.060489, 5.313366, -2.616003, 0.957772, 0.111988, 0.619804]
_C = [0.632169, -0.39329, -0.280382, 0.585646, 0.634009, 0.180328]
CYC_K = np.array(_Q) @ np.array(_Q).T + 0.05 * np.eye(6)
CYC_F, CYC_C = np.array(_F), np.array(_C)


def test_cycling_is_detected_reported_and_never_returned_as_a_solution():
    for route, nxt in (("reduced", "saddle"), ("saddle", "condensed")):
        r = solve_contact_static_sparse(CYC_K, CYC_F, sp.identity(6), CYC_C, route=route)
        assert not r.converged and not r.ok
        assert r.reason == "cycled" and r.suggestion == nxt and nxt in r.message and r.route == route
        assert r.active_mask is not None


def test_message_for_the_dense_route_states_its_memory():
    r = solve_contact_static_sparse(CYC_K, CYC_F, sp.identity(6), CYC_C, route="saddle")
    assert "GB" in r.message and "n = 6" in r.message


def test_fallback_follows_the_suggestions_to_a_route_that_works():
    r = solve_contact_static_sparse(CYC_K, CYC_F, sp.identity(6), CYC_C, fallback=True)
    d = solve_contact_static(CYC_K, CYC_F, np.eye(6), CYC_C, method="lemke")
    assert r.converged and r.route == "condensed" and r.fallbacks == ["reduced", "saddle"]
    assert np.abs(r.u - d.u).max() < 1e-10


def test_fallback_does_not_switch_to_the_dense_route_when_it_would_not_fit():
    r = solve_contact_static_sparse(CYC_K, CYC_F, sp.identity(6), CYC_C, fallback=True, max_dense_bytes=1.0)
    assert not r.converged and r.reason == "memory" and r.suggestion is None and r.fallbacks == ["reduced"]


def test_iteration_limit_is_reported():
    g = OBST["plateau -0.3"]
    r = solve_contact_static_sparse(K_M, F_M, EYE, g, route="reduced", max_iter=1)
    assert not r.converged and r.reason == "max_iter" and r.suggestion == "saddle"


def test_dependent_active_constraints_make_the_saddle_system_singular_and_the_dense_route_recovers():
    n = 6
    K = np.diag(np.arange(2.0, 2.0 + n)) + 0.1
    B = sp.csr_matrix(np.array([[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]]))     # the same gap twice
    f = np.array([-3.0, 0, 0, 0, 0, 0])
    c = np.array([0.0, 0.0])
    r = solve_contact_static_sparse(K, f, B, c)
    assert not r.converged and r.reason == "singular" and r.suggestion == "condensed"
    r2 = solve_contact_static_sparse(K, f, B, c, fallback=True)
    d = dense(K, f, B, c, method="lemke")
    assert r2.converged and r2.route == "condensed" and np.abs(r2.u - d.u).max() < 1e-10
    assert abs(r2.u[0]) < 1e-10                         # held at the constraint


# ------------------------------------------------------------------------------------------- the rest
def test_warm_start_from_the_previous_active_set_needs_fewer_iterations():
    g = OBST["plateau -0.3"]
    cold = solve_contact_static_sparse(K_M, F_M, EYE, g)
    warm = solve_contact_static_sparse(K_M, F_M, EYE, g, active0=cold.active_mask)
    assert cold.iterations >= 3 and warm.iterations == 1
    assert np.abs(warm.u - cold.u).max() < 1e-13
    idx = solve_contact_static_sparse(K_M, F_M, EYE, g, active0=np.flatnonzero(cold.active_mask))     # index array form
    assert idx.iterations == 1


def test_warm_start_with_a_slightly_changed_problem():
    g = OBST["ridge"]
    a = solve_contact_static_sparse(K_M, F_M, EYE, g)
    b = solve_contact_static_sparse(K_M, 1.02 * F_M, EYE, g, active0=a.active_mask)
    cold = solve_contact_static_sparse(K_M, 1.02 * F_M, EYE, g)
    assert b.iterations <= cold.iterations and np.abs(b.u - cold.u).max() < 1e-12


def test_method_pdas_of_solve_contact_static_uses_the_sparse_route():
    g = OBST["bump"]
    r = solve_contact_static(K_M, F_M, EYE, g, method="pdas")
    assert r.route == "reduced"
    assert np.abs(r.u - dense(K_M, F_M, EYE, g).u).max() < 1e-12


def test_existing_results_and_signature_are_unchanged():
    r = ContactResult(np.zeros(2), np.array([0.0, 1.0]), np.zeros(2), 3, True)
    assert r.route is None and r.reason is None and r.fallbacks == [] and r.ok and list(r.active) == [1]
    d = solve_contact_static(K_M.toarray()[:30, :30] + 0 * np.eye(30), F_M[:30], np.eye(30), -0.001 * np.ones(30))
    assert d.route is None and d.converged


def test_no_constraints_and_input_checks():
    r = solve_contact_static_sparse(K_M, F_M, sp.csr_matrix((0, K_M.shape[0])), np.zeros(0))
    assert r.converged and np.abs(K_M @ r.u - F_M).max() < 1e-10
    with pytest.raises(ValueError, match="route must be"):
        solve_contact_static_sparse(K_M, F_M, EYE, OBST["bump"], route="lemke")
    with pytest.raises(ValueError, match="shapes"):
        solve_contact_static_sparse(K_M, F_M[:-1], EYE, OBST["bump"])
    with pytest.raises(ValueError, match="active0"):
        solve_contact_static_sparse(K_M, F_M, EYE, OBST["bump"], active0=np.zeros(3, bool))


def test_condensed_route_wraps_the_dense_solver_unchanged():
    g = OBST["ridge"]
    r = solve_contact_static_sparse(K_M, F_M, EYE, g, route="condensed")
    d = dense(K_M, F_M, EYE, g)
    assert r.route == "condensed" and r.converged
    assert np.abs(r.u - d.u).max() < 1e-13 and np.abs(r.lam - d.lam).max() < 1e-13      # same algorithm; sparse K factors by LU

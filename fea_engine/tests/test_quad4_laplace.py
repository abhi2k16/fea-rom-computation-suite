# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""Tests for the scalar-field element `Quad4Laplace` (elements/scalar.py): element matrices against closed forms, the patch test on
distorted meshes, the assembled stiffness against the Kronecker form of the uniform-grid Laplacian, Poisson convergence on a manufactured
and on a known solution, membrane eigenvalues with the consistent and the lumped mass, and the input conventions."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp

from fea_engine import FESystem, Quad4Laplace, Quad4PlaneStress
from fea_engine import mesh as meshlib
from fea_engine.elements import ELEMENT_REGISTRY

EL = Quad4Laplace()
UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
K_UNIT = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6.0
M_UNIT = np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]]) / 36.0


def square_system(ne, distort=0.0, seed=0, sparse=True):
    m = meshlib.rectangle_mesh(1.0, 1.0, ne, ne)
    if distort:
        rng = np.random.default_rng(seed)
        X = m.nodes
        inner = (X[:, 0] > 1e-9) & (X[:, 0] < 1 - 1e-9) & (X[:, 1] > 1e-9) & (X[:, 1] < 1 - 1e-9)
        X[inner] += distort / ne * rng.standard_normal((inner.sum(), 2))
    return m, FESystem(m, Quad4Laplace(), sparse=sparse)


def boundary(m):
    X = m.nodes
    return np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > 1 - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > 1 - 1e-9))


def dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A)


# --------------------------------------------------------------------------------------------- element matrices
def test_unit_square_stiffness_and_mass_equal_the_closed_forms():
    assert np.abs(EL.stiffness(UNIT, 1.0) - K_UNIT).max() < 1e-15
    assert np.abs(EL.mass(UNIT, 1.0) - M_UNIT).max() < 1e-15
    assert np.abs(EL.stiffness(UNIT, 3.0, thickness=2.0) - 6.0 * K_UNIT).max() < 1e-14
    assert np.abs(EL.mass(UNIT, 2.0, thickness=0.5) - M_UNIT).max() < 1e-15


def test_stiffness_is_symmetric_psd_with_the_constants_as_null_space_on_a_distorted_quad():
    X = np.array([[0.0, 0.0], [2.0, 0.3], [1.7, 1.4], [-0.2, 1.1]])
    K = EL.stiffness(X, 1.0)
    w = np.linalg.eigvalsh(K)
    assert np.abs(K - K.T).max() < 1e-14 and abs(w[0]) < 1e-14 and w[1] > 1e-3
    assert np.abs(K @ np.ones(4)).max() < 1e-14


def test_anisotropic_conductivity_matrix():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    D = np.diag([3.0, 0.5])
    K = EL.stiffness(X, D)
    ux = X[:, 0].copy()                                        # u = x: energy = kx * area = 3 * 2
    uy = X[:, 1].copy()                                        # u = y: energy = ky * area = 0.5 * 2
    assert abs(ux @ K @ ux - 3.0 * 2.0) < 1e-13 and abs(uy @ K @ uy - 0.5 * 2.0) < 1e-13 and abs(ux @ K @ uy) < 1e-13


def test_mass_totals_and_lumped_mass_on_a_distorted_quad():
    X = np.array([[0.0, 0.0], [2.0, 0.3], [1.7, 1.4], [-0.2, 1.1]])
    d1, d2 = X[2] - X[0], X[3] - X[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])                     # quadrilateral area from its diagonals
    M = EL.mass(X, 2.5, thickness=0.2)
    L = EL.lumped_mass(X, 2.5, thickness=0.2)
    assert abs(M.sum() - 2.5 * 0.2 * area) < 1e-13
    assert np.abs(L - np.diag(np.diag(L))).max() == 0 and abs(L.sum() - M.sum()) < 1e-13 and np.diag(L).min() > 0


def test_source_vector_is_the_consistent_load():
    X = np.array([[0.0, 0.0], [2.0, 0.3], [1.7, 1.4], [-0.2, 1.1]])
    q = EL.source_vector(X, 4.0, thickness=0.5)
    assert np.abs(q - EL.mass(X, 1.0, thickness=0.5) @ np.full(4, 4.0)).max() < 1e-14       # uniform: M q
    qn = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.abs(EL.source_vector(X, qn) - EL.mass(X, 1.0) @ qn).max() < 1e-14              # nodal values: M q_nodal
    with pytest.raises(ValueError, match="nodal values"):
        EL.source_vector(X, np.ones(3))


def test_scalar_or_matrix_inputs_and_clear_errors():
    assert np.array_equal(EL.stiffness(UNIT, 2.0), EL.stiffness(UNIT, 2.0 * np.eye(2)))
    assert np.array_equal(EL.mass(UNIT, 2.0), EL.mass(UNIT, np.array([[2.0]])))
    assert np.array_equal(EL.lumped_mass(UNIT, 2.0), EL.lumped_mass(UNIT, np.array([[2.0]])))
    with pytest.raises(ValueError, match="conductivity must be"):
        EL.stiffness(UNIT, np.eye(3))
    with pytest.raises(ValueError, match="density must be"):
        EL.mass(UNIT, np.eye(2))


def test_is_registered_and_is_not_a_plane_stress_element():
    assert ELEMENT_REGISTRY["quad4_laplace"] is Quad4Laplace
    assert not isinstance(EL, Quad4PlaneStress)
    assert (Quad4Laplace.n_nodes, Quad4Laplace.dofs_per_node, Quad4Laplace.dim) == (4, 1, 2)


# ------------------------------------------------------------------------------------------------ assembled system
def kron_laplacian(ne):
    n = ne - 1
    h = 1.0 / ne
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    Mm = sp.diags([1, 4, 1], [-1, 0, 1], shape=(n, n)) / 6.0
    return (sp.kron(T / h ** 2, Mm) + sp.kron(Mm, T / h ** 2)).toarray() * h ** 2


@pytest.mark.parametrize("sparse", [True, False])
def test_assembled_stiffness_equals_the_kronecker_form_on_the_uniform_grid(sparse):
    ne = 8
    m, fes = square_system(ne, sparse=sparse)
    fes.assemble_stiffness(1.0)
    K = dense(fes.K)
    X = m.nodes
    inner = np.flatnonzero((X[:, 0] > 1e-9) & (X[:, 0] < 1 - 1e-9) & (X[:, 1] > 1e-9) & (X[:, 1] < 1 - 1e-9))
    inner = inner[np.lexsort((X[inner, 1], X[inner, 0]))]                     # x-major, as the Kronecker form
    assert np.abs(K[np.ix_(inner, inner)] - kron_laplacian(ne)).max() < 1e-13
    assert np.abs(K @ np.ones(K.shape[0])).max() < 1e-13


def test_patch_test_linear_field_on_a_distorted_mesh():
    m, fes = square_system(6, distort=0.6, seed=1)
    fes.assemble_stiffness(1.0)
    K = dense(fes.K)
    X = m.nodes
    u = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
    bnd = boundary(m)
    inner = np.setdiff1d(np.arange(len(X)), bnd)
    assert np.abs((K @ u)[inner]).max() < 1e-13                                # equilibrium of the linear field
    # a Dirichlet solve with the exact boundary values recovers the field: K_ii u_i = - K_ib u_b
    ub = u[bnd]
    ui = np.linalg.solve(K[np.ix_(inner, inner)], -K[np.ix_(inner, bnd)] @ ub)
    assert np.abs(ui - u[inner]).max() < 1e-12


def solve_poisson(ne, source, exact=None):
    """-laplace(u) = source on the unit square, u = 0 on the boundary; consistent load M @ q_nodal."""
    m, fes = square_system(ne)
    fes.assemble_stiffness(1.0)
    fes.assemble_mass(1.0)
    X = m.nodes
    F = dense(fes.M) @ source(X[:, 0], X[:, 1])
    bnd = boundary(m)
    inner = np.setdiff1d(np.arange(len(X)), bnd)
    K = dense(fes.K)
    u = np.zeros(len(X))
    u[inner] = np.linalg.solve(K[np.ix_(inner, inner)], F[inner])
    return m, u


def test_manufactured_poisson_solution_converges_at_second_order():
    f = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    errs = []
    for ne in (8, 16, 32):
        m, u = solve_poisson(ne, f)
        ex = np.sin(np.pi * m.nodes[:, 0]) * np.sin(np.pi * m.nodes[:, 1])
        errs.append(np.abs(u - ex).max())
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 2e-3 and np.all(rates > 1.8), (errs, rates)


def test_uniform_source_membrane_centre_value():
    """-laplace(u) = 1 on the unit square: centre value 0.0736713532 (series solution)."""
    vals = []
    for ne in (8, 16, 32):
        m, u = solve_poisson(ne, lambda x, y: np.ones_like(x))
        centre = np.argmin(np.abs(m.nodes[:, 0] - 0.5) + np.abs(m.nodes[:, 1] - 0.5))
        vals.append(u[centre])
    assert abs(vals[-1] - 0.0736713532) < 2e-4 and abs(vals[-1] - 0.0736713532) < abs(vals[0] - 0.0736713532)


def test_element_source_vector_assembles_to_the_same_load_as_the_mass_product():
    m, fes = square_system(6, distort=0.4, seed=2)
    fes.assemble_mass(1.0)
    F = np.zeros(fes.n_dof)
    for conn in m.elements:
        F[np.asarray(conn)] += EL.source_vector(m.nodes[conn], 3.0)
    assert np.abs(F - dense(fes.M) @ np.full(fes.n_dof, 3.0)).max() < 1e-13


def test_membrane_eigenvalues_consistent_mass_above_and_lumped_mass_below_the_exact_value():
    exact = 2 * np.pi ** 2
    lam_c, lam_l = [], []
    for ne in (8, 16):
        m, fes = square_system(ne)
        fes.assemble_stiffness(1.0)
        fes.assemble_mass(1.0)
        fes.assemble_lumped_mass(1.0)
        bnd = boundary(m)
        inner = np.setdiff1d(np.arange(len(m.nodes)), bnd)
        K = dense(fes.K)[np.ix_(inner, inner)]
        lam_c.append(sl.eigh(K, dense(fes.M)[np.ix_(inner, inner)], eigvals_only=True)[0])
        lam_l.append(sl.eigh(K, dense(fes.M_lumped)[np.ix_(inner, inner)], eigvals_only=True)[0])
    assert all(l > exact for l in lam_c) and all(l < exact for l in lam_l), (lam_c, lam_l)
    assert abs(lam_c[1] - exact) < 0.3 * abs(lam_c[0] - exact) and abs(lam_c[1] - exact) / exact < 5e-3
    assert abs(lam_l[1] - exact) < 0.3 * abs(lam_l[0] - exact)


def test_package_modal_solver_on_the_membrane():
    m, fes = square_system(16)
    fes.assemble_stiffness(1.0)
    fes.assemble_mass(1.0)
    fes.fix_dofs(boundary(m), ["u"])
    freq, shapes = fes.solve_modal(n_modes=3)
    lam = (2 * np.pi * np.asarray(freq)) ** 2
    exact = np.pi ** 2 * np.array([2.0, 5.0, 5.0])
    assert np.abs(lam / exact - 1).max() < 0.015            # measured 0.3%, 1.1%, 1.1% (consistent mass is an upper bound)
    assert np.all(lam > exact)


def test_lumped_mass_of_the_uniform_grid_is_h_squared_per_interior_node():
    ne = 8
    m, fes = square_system(ne)
    fes.assemble_lumped_mass(1.0)
    L = np.diag(dense(fes.M_lumped))
    inner = np.setdiff1d(np.arange(len(m.nodes)), boundary(m))
    assert np.abs(L[inner] - (1.0 / ne) ** 2).max() < 1e-14


def test_named_dof_u_and_a_wrong_name_is_refused():
    m, fes = square_system(4)
    fes.fix_dofs(boundary(m), ["u"])
    with pytest.raises((ValueError, KeyError)):
        fes.fix_dofs(boundary(m), ["ux"])

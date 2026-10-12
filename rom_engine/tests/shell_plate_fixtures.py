# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
shell_plate_fixtures.py -- a geometrically nonlinear clamped square plate built from fea_engine's
Shell4MITCCorotational shell, returned as plain arrays and callables with the same keys as
`nonlinear_beam_fixtures.nonlinear_beam_model` (free-DOF K0, M, vibration modes Phi, f_int, K_t, element data for
ECSW), so every system-driven-basis routine can be run on a shell without change.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh

from fea_engine import Material, D_shell, Shell4MITCCorotational, rectangle_mesh, FESystem
from fea_engine.mesh import Mesh


def nonlinear_shell_plate_model(n=6, a=1.0, h=0.005, E=210e9, nu=0.3, rho=7800.0, n_modes=3):
    mat = Material(E=E, nu=nu, rho=rho)
    D = D_shell(mat, h)
    mesh2d = rectangle_mesh(a, a, n, n)
    nodes3d = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])
    mesh = Mesh(nodes=nodes3d, elements=mesh2d.elements, dim=2)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    edge = np.where((x < 1e-9) | (x > a - 1e-9) | (y < 1e-9) | (y > a - 1e-9))[0]
    elem = Shell4MITCCorotational()
    fes = FESystem(mesh, elem)
    fes.fix_dofs(edge, [0, 1, 2, 3, 4, 5])
    # translational density rho; rotary inertia rho h^2/12 (a bare rho*I would weigh a rotation of 1 rad like a 1 m
    # translation and make the lowest 'modes' rotational)
    rho_matrix = np.diag([rho] * 3 + [rho * h ** 2 / 12.0] * 3)
    fes.assemble_mass(rho_matrix, thickness=h)
    fes.assemble_stiffness(D, thickness=h)
    free = np.asarray(fes.free_dofs)
    N = fes.n_dof

    def embed(u):
        full = np.zeros(N)
        full[free] = u
        return full

    def f_int(u):
        return np.asarray(fes.assemble_internal_force(embed(u), D, thickness=h))[free]

    def K_t(u):
        K = fes.assemble_tangent_stiffness(embed(u), D, thickness=h)
        K = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
        return K[np.ix_(free, free)]

    M = fes.M[np.ix_(free, free)]
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    K0 = K_t(np.zeros(len(free)))
    w2, Phi = eigh(K0, M)
    conn = mesh.elements
    elem_dofs = [np.asarray(fes._global_dofs(c)) for c in conn]
    elem_force = lambda e, ul: elem.internal_force(mesh.nodes[conn[e]], ul, D, thickness=h)
    translational = (free % 6) < 3          # displacement DOFs (the rotation DOFs would dominate a naive max-norm)
    pressure = np.zeros(len(free)); pressure[(free % 6) == 2] = 1.0       # uniform transverse load pattern
    return dict(fes=fes, free=free, translational=translational, load_pattern=pressure, n=N, n_free=len(free), K0=K0, M=M, Phi=Phi[:, :n_modes], omega2=w2[:n_modes],
                f_int=f_int, K_t=K_t, embed=embed, elem_dofs=elem_dofs, elem_force=elem_force, h=h, a=a, mat=mat,
                D=D, mesh=mesh)

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
hyper_reduction_timing.py -- timing report for the batched ECSW / nonlinearity-indicator routes (a script, not a test).

Model: distorted Quad4PlaneStressTL cantilever plate (nx x ny elements), large deflection, snapshots from a nonlinear
static solve, 4-mode POD basis. Per-element route = the element's own internal_force called element by element;
batched route = FESystem.element_forces_local(vectorized=True). Reports training-matrix time, the equality of G, the
selected-element counts, and reduced force / tangent evaluation time for the selected elements (best of N).

    python tests/hyper_reduction_timing.py [--nx 40 --ny 10] [--repeats 3]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse, os, platform, time
import numpy as np
from fea_engine import FESystem, Material, MultiBlockMesh, nonlinear_solver as nls
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL
from rom_engine.hyper_reduction import ECSW, ecsw_training_matrix, ecsw_training_matrix_batched
from rom_engine.domain_partition import nonlinearity_indicator, nonlinearity_indicator_batched

TH = 0.5


def best_of(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=40); ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    nx, ny = a.nx, a.ny
    xs, ys = np.linspace(0, nx / 2.0, nx + 1), np.linspace(0, ny / 4.0, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs]); nodes[:, 0] += 0.04 * np.sin(2.0 * nodes[:, 1])
    el = np.array([[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
                   for j in range(ny) for i in range(nx)])
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": el}, dim=2), {"q": Quad4PlaneStressTL()}, sparse=False)
    fes.fix_dofs([j * (nx + 1) for j in range(ny + 1)], [0, 1])
    fes.add_nodal_force([(j + 1) * (nx + 1) - 1 for j in range(ny + 1)], 1, -0.5 * ny)
    mat = {"q": D_plane_stress(Material(E=1.0e3, nu=0.3))}
    _, U = nls.solve_nonlinear_static(fes, mat, n_steps=a.steps, tol=1e-7, max_iter=40, vectorized=True, sparse_tangent=True, thickness=TH)
    S = np.asarray(U).T[:, 1:]
    free = fes.free_dofs
    Ub = np.linalg.svd(S[free, :], full_matrices=False)[0][:, :4]
    V = np.zeros((fes.n_dof, 4)); V[free] = Ub
    dofs = [np.asarray(g) for g in fes.element_dofs()]
    form, D = fes._blocks[0][1], mat["q"]
    fl = lambda e, u: form.internal_force(nodes[el[e]], u, D, thickness=TH)
    kl = lambda e, u: form.tangent_stiffness(nodes[el[e]], u, D, thickness=TH)
    fb = lambda e, U_: fes.element_forces_local(e, U_, mat, vectorized=True, thickness=TH)
    kb = lambda e, U_: fes.element_tangents_local(e, U_, mat, vectorized=True, thickness=TH)
    print(f"machine: {platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, cores {os.cpu_count()}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"Quad4PlaneStressTL plate {nx}x{ny}: {len(el)} elements, {fes.n_dof} DOFs, {S.shape[1]} snapshots, max |u| {np.abs(S).max():.3f}, best of {a.repeats}")
    fb(np.arange(len(el)), S[np.array(dofs), 0])     # geometry cache built, not timed
    t0 = best_of(lambda: ecsw_training_matrix(V, S, dofs, fl), 1)
    t1 = best_of(lambda: ecsw_training_matrix_batched(V, S, dofs, fb), a.repeats)
    G0, _ = ecsw_training_matrix(V, S, dofs, fl); G1, _ = ecsw_training_matrix_batched(V, S, dofs, fb)
    print(f"  training matrix   per-element {1e3*t0:9.1f} ms (1 run)   batched {1e3*t1:8.2f} ms   x{t0/t1:6.1f}   max rel diff {np.abs(G1-G0).max()/np.abs(G0).max():.1e}")
    r0 = ECSW.fit(V, S, dofs, fl, kl, tol=1e-4); r1 = ECSW.fit_batched(V, S, dofs, fb, kb, tol=1e-4)
    print(f"  ECSW tol 1e-4: selected {r0.n_selected} (per-element) / {r1.n_selected} (batched) of {len(el)}; same set: {np.array_equal(r0.elements, r1.elements)}")
    q = np.full(4, 0.2)
    t0 = best_of(lambda: r0.reduced_force(q), a.repeats); t1 = best_of(lambda: r1.reduced_force(q), a.repeats)
    print(f"  reduced force ({r0.n_selected} elements)   per-element {1e3*t0:8.3f} ms   batched {1e3*t1:8.3f} ms   x{t0/t1:5.1f}")
    t0 = best_of(lambda: r0.reduced_tangent(q), a.repeats); t1 = best_of(lambda: r1.reduced_tangent(q), a.repeats)
    print(f"  reduced tangent                      per-element {1e3*t0:8.3f} ms   batched {1e3*t1:8.3f} ms   x{t0/t1:5.1f}")
    St = S[:, [1, 4, 7]]
    t0 = best_of(lambda: nonlinearity_indicator(dofs, fl, kl, St), 1); t1 = best_of(lambda: nonlinearity_indicator_batched(dofs, fb, kb, St), a.repeats)
    k0 = nonlinearity_indicator(dofs, fl, kl, St); k1 = nonlinearity_indicator_batched(dofs, fb, kb, St)
    print(f"  nonlinearity indicator (3 states)  per-element {1e3*t0:8.1f} ms (1 run)   batched {1e3*t1:8.2f} ms   x{t0/t1:6.1f}   max rel diff {np.abs(k1-k0).max()/np.abs(k0).max():.1e}")


if __name__ == "__main__":
    main()

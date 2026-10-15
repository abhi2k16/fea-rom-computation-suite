# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batch_timing_contact_3d.py -- timing report for batched 3-D node-to-triangle contact (a script, not a test; speed is never asserted).

Two Hex8SolidTL blocks facing each other (nx x ny elements, one layer each); slave nodes = bottom face of the upper block, master
triangles = the triangulated top face of the lower block. Compares on identical pairs: broad-phase search (loop vs KD-tree), contact force
and tangent of one NodeToSurfaceContact3D element per pair vs one ContactSet3D, and whole-system force and CSR tangent (batched hexahedra).

    python tests/batch_timing_contact_3d.py [--nx 6 --ny 6] [--radius 0.45] [--repeats 5]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse, os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
from fea_engine.contact_batch import ContactSet3D, candidate_pairs_3d
from fea_engine.elements import NodeToSurfaceContact3D
from fea_engine.elements.contact import find_contact_pairs_3d
import test_contact_set_3d as T


def best_of(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=6); ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--radius", type=float, default=0.45); ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    print(f"machine: {platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, cores {os.cpu_count()}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    fes_el, mat, nodes, tris, slaves, off, top = T.two_bodies(a.nx, a.ny)
    fes_set = T.two_bodies(a.nx, a.ny)[0]
    t_loop = best_of(lambda: find_contact_pairs_3d(slaves, tris, nodes, a.radius), min(a.repeats, 2))
    t_kd = best_of(lambda: candidate_pairs_3d(nodes, slaves, tris, a.radius), a.repeats)
    pairs = find_contact_pairs_3d(slaves, tris, nodes, a.radius)
    same = candidate_pairs_3d(nodes, slaves, tris, a.radius) == pairs
    for s, t in pairs:
        fes_el.add_contact_element(NodeToSurfaceContact3D(), [s, *t], (T.KP,))
    cs = fes_set.add_contact_set(ContactSet3D(T._rows(pairs), T.KP))
    u = T._press(fes_el, nodes, off)
    kw = dict(vectorized=True)
    print(f"two Hex8SolidTL blocks {a.nx}x{a.ny}x1 each: {fes_el.n_elements} elements, {fes_el.n_dof} DOFs, {len(slaves)} slave nodes, "
          f"{len(tris)} triangles, {len(pairs)} pairs ({int(cs.active(nodes, u).sum())} penetrating), best of {a.repeats}")
    print(f"  search      loop {1e3*t_loop:9.2f} ms (best of {min(a.repeats, 2)})   KD-tree {1e3*t_kd:8.2f} ms   x{t_loop/t_kd:6.1f}   same pairs and order: {same}")
    ids, el = T._rows(pairs), NodeToSurfaceContact3D()
    def per_pair_force():
        f = np.zeros(fes_el.n_dof)
        for k, r in enumerate(ids):
            f[cs.dofs[k]] += el.internal_force(nodes[r], u[cs.dofs[k]], (T.KP,))
        return f
    per_pair_tangent = lambda: [el.tangent_stiffness(nodes[r], u[cs.dofs[k]], (T.KP,)) for k, r in enumerate(ids)]
    t0 = best_of(per_pair_force, a.repeats); t1 = best_of(lambda: cs.internal_force(nodes, u, fes_el.n_dof), a.repeats)
    d = np.abs(per_pair_force() - cs.internal_force(nodes, u, fes_el.n_dof)).max()
    print(f"  contact force only    per-pair elements {1e3*t0:9.2f} ms   set {1e3*t1:8.3f} ms   x{t0/t1:6.1f}   max diff {d:.1e}")
    t0 = best_of(per_pair_tangent, a.repeats); t1 = best_of(lambda: cs.pair_tangents(nodes, u), a.repeats)
    d = np.abs(np.array(per_pair_tangent()) - cs.pair_tangents(nodes, u)).max()
    print(f"  contact tangent only  per-pair elements {1e3*t0:9.2f} ms   set {1e3*t1:8.3f} ms   x{t0/t1:6.1f}   max diff {d:.1e}")
    fes_el.assemble_internal_force(u, mat, **kw); fes_set.assemble_internal_force(u, mat, **kw)
    fes_el.assemble_tangent_stiffness(u, mat, sparse=True, **kw); fes_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw)
    t0 = best_of(lambda: fes_el.assemble_internal_force(u, mat, **kw), a.repeats)
    t1 = best_of(lambda: fes_set.assemble_internal_force(u, mat, **kw), a.repeats)
    d = np.abs(fes_el.assemble_internal_force(u, mat, **kw) - fes_set.assemble_internal_force(u, mat, **kw)).max()
    print(f"  system force (batched hexahedra)  add_contact_element {1e3*t0:8.2f} ms   add_contact_set {1e3*t1:8.2f} ms   x{t0/t1:5.1f}   max diff {d:.1e}")
    t0 = best_of(lambda: fes_el.assemble_tangent_stiffness(u, mat, sparse=True, **kw), a.repeats)
    t1 = best_of(lambda: fes_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw), a.repeats)
    d = abs(fes_el.assemble_tangent_stiffness(u, mat, sparse=True, **kw) - fes_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw)).max()
    print(f"  system CSR tangent                add_contact_element {1e3*t0:8.2f} ms   add_contact_set {1e3*t1:8.2f} ms   x{t0/t1:5.1f}   max diff {d:.1e}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()

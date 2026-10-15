# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
batch_timing_contact.py -- timing report for batched 2-D node-to-segment contact (a script, not a test; speed is never asserted).

Two Quad4PlaneStressTL plates facing each other (nx x ny elements each); every slave node of the upper body is paired with the
master segments within the search radius. Compares, on identical pairs: the broad-phase search (loop vs KD-tree); contact force and
tangent of one NodeToSegmentContact2D element per pair (FESystem.add_contact_element) vs one ContactSet2D (add_contact_set);
whole-system force and CSR tangent assembly with the batched structural kernels in both cases. Prints the maximum difference.

    python tests/batch_timing_contact.py [--nx 40 --ny 4] [--radius 0.2] [--repeats 5]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse, os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine.contact_batch import ContactSet2D, candidate_pairs_2d
from fea_engine.elements import Quad4PlaneStressTL, NodeToSegmentContact2D
from fea_engine.elements.contact import find_contact_pairs_2d
from fea_engine.material import D_plane_stress
import test_contact_set_2d as T


def best_of(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=40); ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--radius", type=float, default=0.2); ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    print(f"machine: {platform.platform()}, python {platform.python_version()}, numpy {np.__version__}, cores {os.cpu_count()}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    fes_el, mat, nodes, segs, slaves, off = T.two_bodies(a.nx, a.ny)
    t_loop = best_of(lambda: find_contact_pairs_2d(slaves, segs, nodes, a.radius), a.repeats)
    t_kd = best_of(lambda: candidate_pairs_2d(nodes, slaves, segs, a.radius), a.repeats)
    pairs = find_contact_pairs_2d(slaves, segs, nodes, a.radius)
    same = candidate_pairs_2d(nodes, slaves, segs, a.radius) == pairs
    fes_set = T.two_bodies(a.nx, a.ny)[0]
    for s, (p, q) in pairs:
        fes_el.add_contact_element(NodeToSegmentContact2D(), [s, p, q], (T.KP,))
    cs = fes_set.add_contact_set(ContactSet2D(T._rows(pairs), T.KP))
    u = T._press(fes_el, nodes, off)
    kw = dict(thickness=T.TH, vectorized=True)
    print(f"two Quad4PlaneStressTL plates {a.nx}x{a.ny} each: {fes_el.n_elements} elements, {fes_el.n_dof} DOFs, {len(slaves)} slave nodes, "
          f"{len(segs)} segments, {len(pairs)} pairs ({int(cs.active(nodes, u).sum())} penetrating), best of {a.repeats}")
    print(f"  search      loop {1e3*t_loop:9.2f} ms   KD-tree {1e3*t_kd:8.2f} ms   x{t_loop/t_kd:6.1f}   same pairs and order: {same}")
    ids = T._rows(pairs)
    el = NodeToSegmentContact2D()
    def per_pair_force():
        f = np.zeros(fes_el.n_dof)
        for k, r in enumerate(ids):
            f[cs.dofs[k]] += el.internal_force(nodes[r], u[cs.dofs[k]], (T.KP,))
        return f
    def per_pair_tangent():
        return [el.tangent_stiffness(nodes[r], u[cs.dofs[k]], (T.KP,)) for k, r in enumerate(ids)]
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
    print(f"  system force (batched structure)  add_contact_element {1e3*t0:8.2f} ms   add_contact_set {1e3*t1:8.2f} ms   x{t0/t1:5.1f}   max diff {d:.1e}")
    t0 = best_of(lambda: fes_el.assemble_tangent_stiffness(u, mat, sparse=True, **kw), a.repeats)
    t1 = best_of(lambda: fes_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw), a.repeats)
    d = abs(fes_el.assemble_tangent_stiffness(u, mat, sparse=True, **kw) - fes_set.assemble_tangent_stiffness(u, mat, sparse=True, **kw)).max()
    print(f"  system CSR tangent                add_contact_element {1e3*t0:8.2f} ms   add_contact_set {1e3*t1:8.2f} ms   x{t0/t1:5.1f}   max diff {d:.1e}")


if __name__ == "__main__":
    main()

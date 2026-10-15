# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_batch.py -- node-to-segment penalty contact evaluated for all pairs at once.

``ContactSet2D`` holds many (slave node, master segment) pairs and evaluates their force, gaps and tangent in array form. With
``smoothing_length=0`` (the default) it is the same law as one ``NodeToSegmentContact2D`` element per pair
(``elements/contact.py``): penalty force ``k_p * delta`` for penetration ``delta > 0``, applied to the slave node along ``-n_hat`` and
split over the two master nodes by the projection weights ``(1-t, t)``; the tangent is the same central difference of the force
(step ``1e-7 * max(1, |u_pair|)``). Register a set with ``FESystem.add_contact_set``; the assembly routines then add it in one
call instead of looping over ``add_contact_element`` pairs. Pairs may repeat slave or master nodes (the scatter sums).

Smoothing (off by default): with ``smoothing_length = eps > 0`` the force magnitude is
``0`` for ``delta <= 0``, ``k_p delta^2 / (2 eps)`` for ``0 < delta < eps`` and ``k_p (delta - eps/2)`` for ``delta >= eps``
(continuous with a continuous slope). It is a different constitutive law from the package element, not a numerical detail: anything
that uses it must say so (``ContactSet2D.describe()`` returns the law in words).

Scope (M5.2): 2-D only, frictionless, no state. The single-segment limitation of the element holds unchanged: a slave node sliding
past a segment end does not hand off smoothly to the next segment.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["ContactSet2D", "candidate_pairs_2d"]


def candidate_pairs_2d(node_coords, slave_node_ids, master_segments, search_radius):
    """Broad-phase search with a KD-tree. Returns the same list of ``(slave, (master_a, master_b))`` as
    ``elements.contact.find_contact_pairs_2d`` (same members, same order: by slave, then by segment), but without the
    slaves x segments double loop. A pair qualifies when the slave node is within ``search_radius`` of the closest point of the
    segment (the exact test of the loop version; the tree only discards pairs that cannot qualify)."""
    from scipy.spatial import cKDTree
    X = np.asarray(node_coords, float)
    slaves = np.asarray(list(slave_node_ids), dtype=np.int64)
    segs = np.asarray(list(master_segments), dtype=np.int64).reshape(-1, 2)
    if slaves.size == 0 or segs.size == 0:
        return []
    A, B = X[segs[:, 0]], X[segs[:, 1]]
    half = 0.5 * np.linalg.norm(B - A, axis=1)
    tree = cKDTree(X[slaves])
    out = []
    for j in range(len(segs)):                                       # one tree query per segment (not per pair)
        for i in tree.query_ball_point(0.5 * (A[j] + B[j]), search_radius + half[j] + 1e-12):
            out.append((int(slaves[i]), j, i))
    if not out:
        return []
    s_idx = np.array([o[1] for o in out]); sl = np.array([o[2] for o in out])
    P, a, b = X[slaves[sl]], A[s_idx], B[s_idx]
    ab = b - a
    L2 = np.einsum("ni,ni->n", ab, ab)
    ok = L2 >= 1e-30
    t = np.where(ok, np.einsum("ni,ni->n", P - a, ab) / np.where(ok, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = np.where(ok[:, None], a + t[:, None] * ab, a)
    keep = np.linalg.norm(P - proj, axis=1) <= search_radius
    res = sorted((o[0], o[1]) for o, k in zip(out, keep) if k)
    return [(s, (int(segs[j, 0]), int(segs[j, 1]))) for s, j in res]


class ContactSet2D:
    """Many 2-D node-to-segment penalty pairs, evaluated in array form.

    pairs : (n, 3) integer node numbers ``[slave, master_a, master_b]`` of the main mesh (the node order of
        ``NodeToSegmentContact2D``; the segment normal is the right-hand rotation of ``b - a`` and must point away from the
        master body, as for the element).
    k_p : penalty stiffness, a scalar or one value per pair.
    smoothing_length : 0 (default) = the package law; > 0 = the smoothed law in the module docstring.
    """
    npn = 2

    def __init__(self, pairs, k_p, smoothing_length=0.0):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 3)
        if pairs.size and pairs.min() < 0:
            raise ValueError("node numbers must be >= 0")
        self.pairs = pairs
        kp = np.asarray(k_p, float)
        if kp.ndim == 0:
            kp = np.full(len(pairs), float(kp))
        if kp.shape != (len(pairs),):
            raise ValueError(f"k_p must be a scalar or have one value per pair ({len(pairs)}); got shape {kp.shape}")
        if np.any(kp < 0):
            raise ValueError("k_p must be >= 0")
        self.k_p = kp
        eps = float(smoothing_length)
        if eps < 0:
            raise ValueError("smoothing_length must be >= 0")
        self.smoothing_length = eps
        self.dofs = (2 * pairs[:, :, None] + np.arange(2)).reshape(-1, 6)

    @classmethod
    def from_search(cls, node_coords, slave_node_ids, master_segments, search_radius, k_p, smoothing_length=0.0):
        """Build the set from a KD-tree broad-phase search (``candidate_pairs_2d``)."""
        found = candidate_pairs_2d(node_coords, slave_node_ids, master_segments, search_radius)
        pairs = np.array([[s, a, b] for s, (a, b) in found], dtype=np.int64).reshape(-1, 3)
        return cls(pairs, k_p, smoothing_length)

    @property
    def n_pairs(self):
        return len(self.pairs)

    def describe(self):
        if self.smoothing_length == 0.0:
            law = "linear penalty k_p*delta for delta > 0 (same law as NodeToSegmentContact2D)"
        else:
            law = (f"SMOOTHED penalty, smoothing_length={self.smoothing_length:g}: quadratic ramp on 0 < delta < eps, then "
                   f"linear (not the package element's law)")
        return f"ContactSet2D with {self.n_pairs} node-to-segment pairs; {law}"

    def __repr__(self):
        return f"<{self.describe()}>"

    # ------------------------------------------------------------------ kernels
    def _geometry(self, X0, ue):
        p = X0[:, 0] + ue[:, 0:2]
        a = X0[:, 1] + ue[:, 2:4]
        b = X0[:, 2] + ue[:, 4:6]
        ab = b - a
        L2 = np.einsum("ni,ni->n", ab, ab)
        ok = L2 >= 1e-30
        t = np.where(ok, np.einsum("ni,ni->n", p - a, ab) / np.where(ok, L2, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        proj = np.where(ok[:, None], a + t[:, None] * ab, a)
        L = np.sqrt(L2)
        okn = L >= 1e-30                                              # the element tests the length, not its square, here
        n_hat = np.where(okn[:, None], np.stack([ab[:, 1], -ab[:, 0]], axis=1) / np.where(okn, L, 1.0)[:, None],
                         np.array([0.0, 1.0]))
        delta = -np.einsum("ni,ni->n", p - proj, n_hat)
        return t, n_hat, delta

    def _magnitude(self, delta, kp):
        eps = self.smoothing_length
        if eps == 0.0:
            return np.where(delta > 0.0, kp * delta, 0.0)
        return np.where(delta <= 0.0, 0.0, np.where(delta < eps, kp * delta ** 2 / (2.0 * eps), kp * (delta - 0.5 * eps)))

    def _pair_forces(self, X0, ue, kp):
        t, n_hat, delta = self._geometry(X0, ue)
        fc = self._magnitude(delta, kp)
        f = np.zeros((len(kp), 6))
        f[:, 0:2] = -fc[:, None] * n_hat
        f[:, 2:4] = fc[:, None] * n_hat * (1.0 - t)[:, None]
        f[:, 4:6] = fc[:, None] * n_hat * t[:, None]
        return f

    def _gather(self, nodes, u):
        nodes = np.asarray(nodes, float)
        return nodes[self.pairs], np.asarray(u, float)[self.dofs]

    # ------------------------------------------------------------------ public evaluation
    def gaps(self, nodes, u):
        """Penetration ``delta`` of every pair, (n,): positive = the slave node is on the wrong side of the segment."""
        if self.n_pairs == 0:
            return np.zeros(0)
        X0, ue = self._gather(nodes, u)
        return self._geometry(X0, ue)[2]

    def active(self, nodes, u):
        """Boolean (n,): pairs carrying force (``delta > 0``)."""
        return self.gaps(nodes, u) > 0.0

    def pair_forces(self, nodes, u):
        """Element-like force of each pair, (n, 6), in the DOF order ``[slave_x, slave_y, a_x, a_y, b_x, b_y]``."""
        if self.n_pairs == 0:
            return np.zeros((0, 6))
        X0, ue = self._gather(nodes, u)
        return self._pair_forces(X0, ue, self.k_p)

    def internal_force(self, nodes, u, n_dof):
        """Global contact force vector of length ``n_dof``."""
        if self.n_pairs == 0:
            return np.zeros(n_dof)
        return np.bincount(self.dofs.ravel(), weights=self.pair_forces(nodes, u).ravel(), minlength=n_dof)

    def pair_tangents(self, nodes, u):
        """Central-difference tangent of each pair, (n, 6, 6) (same step rule as ``NodeToSegmentContact2D``)."""
        if self.n_pairs == 0:
            return np.zeros((0, 6, 6))
        X0, ue = self._gather(nodes, u)
        h = 1e-7 * np.maximum(1.0, np.linalg.norm(ue, axis=1))
        K = np.zeros((self.n_pairs, 6, 6))
        for j in range(6):
            d = np.zeros_like(ue)
            d[:, j] = h
            K[:, :, j] = (self._pair_forces(X0, ue + d, self.k_p) - self._pair_forces(X0, ue - d, self.k_p)) / (2 * h)[:, None]
        return K

    def tangent_entries(self, nodes, u):
        """COO triplets ``(rows, cols, vals)`` of the global contact tangent (duplicates are summed by the consumer)."""
        K = self.pair_tangents(nodes, u)
        return np.repeat(self.dofs, 6, axis=1).ravel(), np.tile(self.dofs, (1, 6)).ravel(), K.ravel()

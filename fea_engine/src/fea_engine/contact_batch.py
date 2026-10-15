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

``ContactSet3D`` is the 3-D counterpart for ``NodeToSurfaceContact3D`` (slave node against a master triangle; consistent Gauss-Newton
tangent by default, central difference on request) with ``candidate_pairs_3d`` for the broad phase.

Scope: frictionless, no state. ``ContactSet2D`` (M5.2) and ``ContactSet3D`` (M5.3) take a fixed list of pairs. The single-segment limitation of the element holds unchanged: a slave node sliding
past a segment end does not hand off smoothly to the next segment.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["ContactSet2D", "candidate_pairs_2d", "ContactSet3D", "candidate_pairs_3d", "closest_point_on_triangles"]


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


# ====================================================================================================================
# 3-D: node-to-triangle
# ====================================================================================================================
def closest_point_on_triangles(p, a, b, c):
    """Closest point of each triangle ``(a, b, c)`` (rows of (N, 3) arrays) to the point ``p`` (N, 3). Returns barycentric weights
    ``(N, 3)`` and the points ``(N, 3)``. Same region test and the same precedence between regions as the scalar
    ``elements.contact.closest_point_on_triangle_3d`` (vertex a, vertex b, edge ab, vertex c, edge ac, edge bc, interior; a
    zero-area triangle gives corner a), so the two agree also on region boundaries."""
    p, a, b, c = (np.asarray(x, float) for x in (p, a, b, c))
    ab, ac, ap = b - a, c - a, p - a
    dot = lambda x, y: np.einsum("ni,ni->n", x, y)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp, cp = p - b, p - c
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    n = len(p)
    den = va + vb + vc
    degenerate = np.abs(den) < 1e-300
    safe = np.where(degenerate, 1.0, den)
    v, w = vb / safe, vc / safe
    W = np.stack([1.0 - v - w, v, w], axis=1)
    W[degenerate] = (1.0, 0.0, 0.0)
    zero, one = np.zeros(n), np.ones(n)
    # lowest precedence first; later assignments override, so the final precedence is a, b, ab, c, ac, bc, interior
    m = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)                      # edge bc
    t = (d4 - d3) / np.where(m, (d4 - d3) + (d5 - d6), 1.0)
    W[m] = np.stack([zero, 1.0 - t, t], axis=1)[m]
    m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)                                    # edge ac
    t = d2 / np.where(m, d2 - d6, 1.0)
    W[m] = np.stack([1.0 - t, zero, t], axis=1)[m]
    m = (d6 >= 0.0) & (d5 <= d6)                                                   # vertex c
    W[m] = (0.0, 0.0, 1.0)
    m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)                                    # edge ab
    t = d1 / np.where(m, d1 - d3, 1.0)
    W[m] = np.stack([1.0 - t, t, zero], axis=1)[m]
    m = (d3 >= 0.0) & (d4 <= d3)                                                   # vertex b
    W[m] = (0.0, 1.0, 0.0)
    m = (d1 <= 0.0) & (d2 <= 0.0)                                                  # vertex a
    W[m] = (1.0, 0.0, 0.0)
    proj = W[:, 0:1] * a + W[:, 1:2] * b + W[:, 2:3] * c
    return W, proj


def candidate_pairs_3d(node_coords, slave_node_ids, master_triangles, search_radius):
    """KD-tree broad phase. Returns the same list of ``(slave, (a, b, c))`` as ``elements.contact.find_contact_pairs_3d``, same members
    and order (by slave, then by triangle), without the slaves x triangles double loop."""
    from scipy.spatial import cKDTree
    X = np.asarray(node_coords, float)
    slaves = np.asarray(list(slave_node_ids), dtype=np.int64)
    tris = np.asarray(list(master_triangles), dtype=np.int64).reshape(-1, 3)
    if slaves.size == 0 or tris.size == 0:
        return []
    cen = X[tris].mean(axis=1)
    rad = np.linalg.norm(X[tris] - cen[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(X[slaves])
    cand = []
    for j in range(len(tris)):
        for i in tree.query_ball_point(cen[j], search_radius + rad[j] + 1e-12):
            cand.append((i, j))
    if not cand:
        return []
    si = np.array([c[0] for c in cand]); tj = np.array([c[1] for c in cand])
    P = X[slaves[si]]
    _, proj = closest_point_on_triangles(P, X[tris[tj, 0]], X[tris[tj, 1]], X[tris[tj, 2]])
    keep = np.linalg.norm(P - proj, axis=1) <= search_radius
    res = sorted((int(slaves[i]), int(j)) for (i, j), k in zip(cand, keep) if k)
    return [(s, (int(tris[j, 0]), int(tris[j, 1]), int(tris[j, 2]))) for s, j in res]


class ContactSet3D:
    """Many 3-D node-to-triangle penalty pairs, evaluated in array form (``NodeToSurfaceContact3D`` for all pairs at once).

    pairs : (n, 4) integer node numbers ``[slave, a, b, c]``; the normal ``(b-a) x (c-a)`` of the current triangle must point away from
        the master body, as for the element.
    k_p : scalar or one value per pair. smoothing_length : 0 (default) = the package law; > 0 = the smoothed law of the module docstring.
    consistent_tangent : True (default) = the element's Gauss-Newton tangent ``dF/d(delta) C^T C`` with the projection frozen;
        False = central difference of the pair force (the element's ``consistent=False``).
    nearest_only : False (default) = every listed pair acts, exactly like one element per pair. True = of all rows that share a slave
        node only the one whose closest point is nearest to the slave acts at each evaluation (a sliding slave then hands over to the next
        triangle). This is a DIFFERENT model from registering all pairs; ``describe()`` says so. Ties go to the lower row number.

    Convergence note (measured, see docs/batched_nonlinear_kernels_plan.md, M5.4): with the plain ramp and the consistent tangent, Newton
    can stall for stiff penalties (k_p of about 0.7 E and above on the test model) because the set of pairs in contact changes from
    iteration to iteration; the per-element registration behaves the same. ``consistent_tangent=False`` or a small ``smoothing_length``
    converged in every case tried. Use either only knowingly: both change what is solved or how it is solved, and ``describe()`` names them.
    """
    npn = 3

    def __init__(self, pairs, k_p, smoothing_length=0.0, consistent_tangent=True, nearest_only=False):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
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
        self.consistent_tangent = bool(consistent_tangent)
        self.nearest_only = bool(nearest_only)
        self.dofs = (3 * pairs[:, :, None] + np.arange(3)).reshape(-1, 12)

    @classmethod
    def from_search(cls, node_coords, slave_node_ids, master_triangles, search_radius, k_p, **kwargs):
        """Build the set from the KD-tree broad phase (``candidate_pairs_3d``); keyword arguments go to the constructor."""
        found = candidate_pairs_3d(node_coords, slave_node_ids, master_triangles, search_radius)
        pairs = np.array([[s, *t] for s, t in found], dtype=np.int64).reshape(-1, 4)
        return cls(pairs, k_p, **kwargs)

    @property
    def n_pairs(self):
        return len(self.pairs)

    def describe(self):
        law = ("linear penalty k_p*delta for delta > 0 (same law as NodeToSurfaceContact3D)" if self.smoothing_length == 0.0 else
               f"SMOOTHED penalty, smoothing_length={self.smoothing_length:g} (not the package element's law)")
        tan = "consistent Gauss-Newton tangent" if self.consistent_tangent else "central-difference tangent"
        sel = ("; NEAREST-CANDIDATE selection per slave node (not the same model as one element per pair)"
               if self.nearest_only else "")
        return f"ContactSet3D with {self.n_pairs} node-to-triangle pairs; {law}; {tan}{sel}"

    def __repr__(self):
        return f"<{self.describe()}>"

    # ------------------------------------------------------------------ kernels
    def _law(self, delta, kp):
        """Force magnitude and its derivative with respect to delta."""
        eps = self.smoothing_length
        if eps == 0.0:
            pos = delta > 0.0
            return np.where(pos, kp * delta, 0.0), np.where(pos, kp, 0.0)
        d = np.maximum(delta, 0.0)
        inside = d < eps
        f = np.where(inside, kp * d * d / (2.0 * eps), kp * (d - 0.5 * eps))
        df = np.where(inside, kp * d / eps, kp)
        return np.where(delta <= 0.0, 0.0, f), np.where(delta <= 0.0, 0.0, df)

    def _state(self, X0, ue):
        """Weights, unit normals, penetration and the acting-pair mask of every row."""
        Xc = X0 + ue.reshape(len(X0), 4, 3)
        p, a, b, c = Xc[:, 0], Xc[:, 1], Xc[:, 2], Xc[:, 3]
        W, proj = closest_point_on_triangles(p, a, b, c)
        nrm = np.cross(b - a, c - a)
        L = np.linalg.norm(nrm, axis=1)
        okn = L > 1e-30
        n_hat = np.where(okn[:, None], nrm / np.where(okn, L, 1.0)[:, None], np.array([0.0, 0.0, 1.0]))
        delta = -np.einsum("ni,ni->n", p - proj, n_hat)
        sel = np.ones(len(p), bool)
        if self.nearest_only and len(p):
            d2 = np.einsum("ni,ni->n", p - proj, p - proj)
            order = np.lexsort((np.arange(len(p)), d2, self.pairs[:, 0]))      # by slave, then distance, then row number
            first = np.ones(len(p), bool)
            first[1:] = self.pairs[order[1:], 0] != self.pairs[order[:-1], 0]
            sel = np.zeros(len(p), bool)
            sel[order[first]] = True
        return W, n_hat, delta, sel

    def _gather(self, nodes, u):
        return np.asarray(nodes, float)[self.pairs], np.asarray(u, float)[self.dofs]

    def _pair_forces(self, X0, ue, kp):
        W, n_hat, delta, sel = self._state(X0, ue)
        fc = np.where(sel, self._law(delta, kp)[0], 0.0)
        f = np.zeros((len(kp), 12))
        f[:, 0:3] = -fc[:, None] * n_hat
        for i in range(3):
            f[:, 3 + 3 * i:6 + 3 * i] = fc[:, None] * W[:, i:i + 1] * n_hat
        return f

    # ------------------------------------------------------------------ public evaluation
    def gaps(self, nodes, u):
        """Penetration ``delta`` of every row, (n,): positive = the slave node is on the wrong side of its triangle."""
        if self.n_pairs == 0:
            return np.zeros(0)
        return self._state(*self._gather(nodes, u))[2]

    def active(self, nodes, u):
        """Boolean (n,): rows carrying force (``delta > 0`` and, with ``nearest_only``, nearest for their slave)."""
        if self.n_pairs == 0:
            return np.zeros(0, bool)
        _, _, delta, sel = self._state(*self._gather(nodes, u))
        return (delta > 0.0) & sel

    def pair_forces(self, nodes, u):
        """Element-like force of each row, (n, 12), DOF order ``[slave, a, b, c]`` x (x, y, z)."""
        if self.n_pairs == 0:
            return np.zeros((0, 12))
        X0, ue = self._gather(nodes, u)
        return self._pair_forces(X0, ue, self.k_p)

    def internal_force(self, nodes, u, n_dof):
        """Global contact force vector of length ``n_dof``."""
        if self.n_pairs == 0:
            return np.zeros(n_dof)
        return np.bincount(self.dofs.ravel(), weights=self.pair_forces(nodes, u).ravel(), minlength=n_dof)

    def pair_tangents(self, nodes, u):
        """Tangent of each row, (n, 12, 12)."""
        if self.n_pairs == 0:
            return np.zeros((0, 12, 12))
        X0, ue = self._gather(nodes, u)
        if self.consistent_tangent:
            W, n_hat, delta, sel = self._state(X0, ue)
            dfc = np.where(sel, self._law(delta, self.k_p)[1], 0.0)
            C = np.zeros((self.n_pairs, 12))
            C[:, 0:3] = -n_hat
            for i in range(3):
                C[:, 3 + 3 * i:6 + 3 * i] = W[:, i:i + 1] * n_hat
            return dfc[:, None, None] * C[:, :, None] * C[:, None, :]
        h = 1e-7 * np.maximum(1.0, np.linalg.norm(ue, axis=1))
        K = np.zeros((self.n_pairs, 12, 12))
        for j in range(12):
            d = np.zeros_like(ue)
            d[:, j] = h
            K[:, :, j] = (self._pair_forces(X0, ue + d, self.k_p) - self._pair_forces(X0, ue - d, self.k_p)) / (2 * h)[:, None]
        return K

    def tangent_entries(self, nodes, u):
        """COO triplets ``(rows, cols, vals)`` of the global contact tangent (duplicates are summed by the consumer)."""
        K = self.pair_tangents(nodes, u)
        return np.repeat(self.dofs, 12, axis=1).ravel(), np.tile(self.dofs, (1, 12)).ravel(), K.ravel()

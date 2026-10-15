# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""hex_meshes.py -- small mesh builders shared by the hexahedron tests."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import Mesh

# mid-edge nodes of Hex20Solid3D: 8-11 bottom edges, 12-15 top edges, 16-19 vertical edges
HEX20_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def hex20_from_hex8(mesh8):
    """Hex20 mesh with straight edges from a Hex8 mesh: mid-edge nodes at edge midpoints, shared between elements."""
    nodes = [p for p in mesh8.nodes]
    mid = {}
    elems = []
    for conn in mesh8.elements:
        row = list(conn)
        for a, b in HEX20_EDGES:
            key = tuple(sorted((int(conn[a]), int(conn[b]))))
            if key not in mid:
                mid[key] = len(nodes)
                nodes.append(0.5 * (mesh8.nodes[key[0]] + mesh8.nodes[key[1]]))
            row.append(mid[key])
        elems.append(row)
    return Mesh(np.array(nodes), np.array(elems, dtype=int), 3)

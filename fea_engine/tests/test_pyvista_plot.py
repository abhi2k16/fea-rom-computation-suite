# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_pyvista_plot.py -- validation of fea_engine.visualization.pyvista_plot.

All tests skip when the optional ``pyvista`` package is missing. Tests that only build and inspect
``pyvista.UnstructuredGrid`` objects need no display. Tests that render a picture additionally skip unless an
off-screen/virtual display works (checked once, in a subprocess, because VTK can crash the interpreter without one;
run with ``xvfb-run -a pytest tests/test_pyvista_plot.py`` on a headless Linux machine).

Lines of evidence:

1. Conversion: node/cell counts and VTK cell types for all nine supported cell types, for Mesh and MultiBlockMesh;
   registered point/cell data carried over; wrong-size data rejected.
2. Node ordering of the quadratic cells (tri6, quad8, tet10, hex20): the element nodes built from each element
   class's own shape functions must put every mid-edge node at the midpoint of the edge pair that the VTK ordering
   assigns to it (the VTK edge tables are written out explicitly below, independent of meshio).
3. Displacement handling: flat / (n, k) / rotation-carrying input, error cases, and that the deformed grid's points
   equal nodes + scale * displacement.
4. Fields: von Mises stress of a solved cantilever reaches the grid unchanged; element data and components resolve.
5. Output: .vtu round trip and (display permitting) PNG screenshots that are non-empty and not a single colour.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from fea_engine import (FESystem, Material, D_plane_stress, Quad4PlaneStress, Hex8Solid3D, mesh)
from fea_engine.mesh import MultiBlockMesh, box_mesh, rectangle_mesh
from fea_engine.elements.solids import Tet10Solid3D, Hex20Solid3D, Tri6PlaneStress, Quad8PlaneStress
from fea_engine.visualization import pyvista_plot as vp


# ------------------------------------------------------------------ helpers
def _natural_nodes(f):
    grid = [-1.0, 0.0, 1.0] if f.quadrature_family == "tensor" else [0.0, 0.5, 1.0]
    out = []
    for a in range(f.n_nodes):
        for p in itertools.product(grid, repeat=f.dim):
            N = f.shape_and_derivs(p)[0]
            if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                out.append(p)
                break
        else:
            raise AssertionError(f"node {a} of {type(f).__name__} not found")
    return np.array(out)


# VTK node ordering of the quadratic cells: mid-edge node index -> the two corner nodes it lies between.
VTK_MID_EDGES = {
    "tri6": {3: (0, 1), 4: (1, 2), 5: (2, 0)},
    "quad8": {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)},
    "tet10": {4: (0, 1), 5: (1, 2), 6: (2, 0), 7: (0, 3), 8: (1, 3), 9: (2, 3)},
    "hex20": {8: (0, 1), 9: (1, 2), 10: (2, 3), 11: (3, 0), 12: (4, 5), 13: (5, 6), 14: (6, 7), 15: (7, 4),
              16: (0, 4), 17: (1, 5), 18: (2, 6), 19: (3, 7)},
}
ELEMENTS = {"tri6": Tri6PlaneStress, "quad8": Quad8PlaneStress, "tet10": Tet10Solid3D, "hex20": Hex20Solid3D}


def _cantilever(nx=20, ny=4):
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    m = rectangle_mesh(4.0, 1.0, nx, ny)
    s = FESystem(m, Quad4PlaneStress(), thickness=0.1)
    s.assemble_stiffness(D_plane_stress(mat))
    for n in m.nodes_on_line(0, 0.0):
        s.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - 4.0) < 1e-9)[0]
    s.F[2 * tip + 1] = -1.0e4 / len(tip)
    return m, s, s.solve_static()


# ====================================================================== 1. conversion
class TestConversion:
    def test_quad4_mesh(self):
        m = rectangle_mesh(2.0, 1.0, 4, 3)
        g = vp.to_pyvista(m)
        assert g.n_points == len(m.nodes) and g.n_cells == len(m.elements)
        assert set(g.celltypes) == {9}
        assert np.allclose(g.points[:, :2], m.nodes) and np.all(g.points[:, 2] == 0.0)

    def test_hex8_mesh(self):
        m = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
        g = vp.to_pyvista(m)
        assert set(g.celltypes) == {12} and g.n_cells == 8 and g.n_points == 27
        assert np.isclose(g.compute_cell_sizes()["Volume"].sum(), 1.0)

    @pytest.mark.parametrize("name", sorted(ELEMENTS))
    def test_quadratic_cell_types(self, name):
        f = ELEMENTS[name]()
        xi = _natural_nodes(f)
        m = mesh.Mesh(nodes=xi.copy(), elements=np.arange(f.n_nodes)[None, :], dim=f.dim)
        g = vp.to_pyvista(m)
        assert g.n_cells == 1 and int(g.celltypes[0]) == {"tri6": 22, "quad8": 23, "tet10": 24, "hex20": 25}[name]

    def test_multiblock(self):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1.0]])
        m = MultiBlockMesh(nodes=nodes, blocks={"quad4": np.array([[0, 1, 2, 3]]), "tri3": np.array([[1, 4, 5], [1, 5, 2]])},
                           dim=2)
        g = vp.to_pyvista(m)
        assert g.n_cells == 3 and list(g.celltypes) == [9, 5, 5]
        assert np.isclose(g.compute_cell_sizes()["Area"].sum(), 2.0)

    def test_registered_data_carried_over(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        m.register_point_data("T", np.arange(len(m.nodes), dtype=float))
        m.register_element_data("mat", np.arange(len(m.elements), dtype=float))
        g = vp.to_pyvista(m, point_data={"extra": np.ones(len(m.nodes))})
        assert np.allclose(g.point_data["T"], np.arange(len(m.nodes)))
        assert np.allclose(g.cell_data["mat"], np.arange(len(m.elements)))
        assert np.allclose(g.point_data["extra"], 1.0)

    def test_wrong_size_data_rejected(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(ValueError, match="one per node"):
            vp.to_pyvista(m, point_data={"bad": np.ones(3)})
        with pytest.raises(ValueError, match="one per cell"):
            vp.to_pyvista(m, cell_data={"bad": np.ones(3)})

    def test_unsupported_topology(self):
        m = mesh.Mesh(nodes=np.zeros((5, 3)), elements=np.arange(5)[None, :], dim=3)
        with pytest.raises(ValueError, match="no VTK cell type"):
            vp.to_pyvista(m)


# ====================================================================== 2. quadratic node order
class TestQuadraticNodeOrderIsVTKOrder:
    @pytest.mark.parametrize("name", sorted(ELEMENTS))
    def test_mid_edge_nodes_sit_at_edge_midpoints(self, name):
        f = ELEMENTS[name]()
        xi = _natural_nodes(f)
        A = np.array([[2.0, 0.3, 0.1], [0.1, 1.5, 0.2], [0.05, 0.2, 1.2]])[:f.dim, :f.dim]
        x = xi @ A.T + np.array([0.3, -0.2, 0.5])[:f.dim]            # distorted but straight-sided
        m = mesh.Mesh(nodes=x, elements=np.arange(f.n_nodes)[None, :], dim=f.dim)
        g = vp.to_pyvista(m)
        pts = g.points[:, :f.dim]
        for k, (a, b) in VTK_MID_EDGES[name].items():
            assert np.allclose(pts[k], 0.5 * (pts[a] + pts[b]), atol=1e-12), (name, k, a, b)

    @pytest.mark.parametrize("name,ref_size", [("tet10", 1.0 / 6.0), ("hex20", 8.0), ("tri6", 0.5), ("quad8", 4.0)])
    def test_cell_size_matches_analytic(self, name, ref_size):
        f = ELEMENTS[name]()
        xi = _natural_nodes(f)
        A = np.array([[2.0, 0.3, 0.1], [0.1, 1.5, 0.2], [0.05, 0.2, 1.2]])[:f.dim, :f.dim]
        x = xi @ A.T
        g = vp.to_pyvista(mesh.Mesh(nodes=x, elements=np.arange(f.n_nodes)[None, :], dim=f.dim))
        key = "Volume" if f.dim == 3 else "Area"
        assert np.isclose(g.compute_cell_sizes()[key][0], abs(np.linalg.det(A)) * ref_size, rtol=1e-9)


# ====================================================================== 3. displacement
class TestDisplacement:
    def test_flat_and_2d_input_agree(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n = len(m.nodes)
        U = np.random.RandomState(1).rand(2 * n)
        v1 = vp.displacement_vectors(m, U)
        v2 = vp.displacement_vectors(m, U.reshape(n, 2))
        assert v1.shape == (n, 3) and np.allclose(v1, v2) and np.all(v1[:, 2] == 0.0)

    def test_rotation_dofs_are_ignored(self):
        m = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        n = len(m.nodes)
        U = np.random.RandomState(2).rand(n, 6)
        assert np.allclose(vp.displacement_vectors(m, U.ravel()), U[:, :3])

    def test_errors(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n = len(m.nodes)
        with pytest.raises(ValueError, match="not a multiple"):
            vp.displacement_vectors(m, np.ones(2 * n + 1))
        with pytest.raises(ValueError, match="fewer than the mesh dimension"):
            vp.displacement_vectors(m, np.ones(n))
        with pytest.raises(ValueError, match="rows"):
            vp.displacement_vectors(m, np.ones((n + 1, 2)))

    def test_deformed_grid_points(self):
        m = rectangle_mesh(1.0, 1.0, 3, 3)
        U = np.random.RandomState(3).rand(2 * len(m.nodes))
        g = vp.deformed_grid(m, U, scale=7.0)
        expect = vp.to_pyvista(m).points + 7.0 * vp.displacement_vectors(m, U)
        assert np.allclose(g.points, expect)

    def test_add_displacement(self):
        m = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        g = vp.to_pyvista(m)
        U = np.random.RandomState(4).rand(3 * len(m.nodes))
        vp.add_displacement(g, U)
        assert np.allclose(g.point_data["displacement_magnitude"], np.linalg.norm(U.reshape(-1, 3), axis=1))


# ====================================================================== 4. fields
class TestFields:
    def test_von_mises_reaches_the_grid_unchanged(self):
        m, s, U = _cantilever()
        vm = s.von_mises(U)
        g = vp.to_pyvista(m)
        assoc, vals, title = vp._resolve_field(g, m, vm, U, "von_mises")
        assert assoc == "point" and np.allclose(vals, vm.component("von_mises"))
        assert vals.max() > 0

    def test_element_field_and_named_field(self):
        m = rectangle_mesh(1.0, 1.0, 3, 3)
        m.register_element_data("e", np.arange(len(m.elements), dtype=float))
        g = vp.to_pyvista(m)
        assert vp._resolve_field(g, m, "e", None, None)[0] == "cell"
        assert vp._resolve_field(g, m, np.arange(len(m.elements), dtype=float), None, None)[0] == "cell"
        with pytest.raises(KeyError):
            vp._resolve_field(g, m, "nope", None, None)
        with pytest.raises(ValueError, match="needs U"):
            vp._resolve_field(g, m, "displacement", None, None)

    def test_displacement_components(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n = len(m.nodes)
        U = np.random.RandomState(5).rand(2 * n)
        g = vp.to_pyvista(m)
        _, ux, _ = vp._resolve_field(g, m, "ux", U, None)
        _, mag, _ = vp._resolve_field(g, m, "displacement", U, None)
        assert np.allclose(ux, U[0::2]) and np.allclose(mag, np.hypot(U[0::2], U[1::2]))

    def test_multicomponent_array_is_norm_or_component(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n = len(m.nodes)
        A = np.random.RandomState(6).rand(n, 3)
        g = vp.to_pyvista(m)
        assert np.allclose(vp._resolve_field(g, m, A, None, None)[1], np.linalg.norm(A, axis=1))
        assert np.allclose(vp._resolve_field(g, m, A, None, 1)[1], A[:, 1])


# ====================================================================== 5. output
class TestOutput:
    def test_vtu_round_trip(self, tmp_path):
        m, s, U = _cantilever(8, 2)
        path = tmp_path / "out.vtu"
        vp.save_vtk(m, path, U=U, point_data={"vm": s.von_mises(U).component("von_mises")})
        back = pv.read(str(path))
        assert back.n_cells == len(m.elements) and np.allclose(back.points[:, :2], m.nodes)
        assert np.allclose(back.point_data["displacement"], vp.displacement_vectors(m, U))
        assert np.allclose(back.point_data["vm"], s.von_mises(U).component("von_mises"))

    @staticmethod
    def _assert_picture(path):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"))
        assert img.size > 0
        assert len(np.unique(img.reshape(-1, 3), axis=0)) > 20

    def test_plot_field_screenshot(self, tmp_path, display):
        m, s, U = _cantilever()
        out = tmp_path / "vm.png"
        vp.plot_field(m, s.von_mises(U), component="von_mises", U=U, scale="auto", screenshot=out, show=False)
        self._assert_picture(out)

    def test_plot_deformed_screenshot(self, tmp_path, display):
        m, s, U = _cantilever()
        out = tmp_path / "def.png"
        vp.plot_deformed(m, U, scale="auto", screenshot=out, show=False)
        self._assert_picture(out)

    def test_3d_screenshot(self, tmp_path, display):
        m = box_mesh(2.0, 1.0, 1.0, 4, 2, 2)
        U = np.zeros((len(m.nodes), 3))
        U[:, 2] = -1e-3 * m.nodes[:, 0] ** 2
        out = tmp_path / "box.png"
        vp.plot_deformed(m, U.ravel(), scale="auto", screenshot=out, show=False)
        self._assert_picture(out)

# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_pyvista_sections.py -- validation of fea_engine.visualization.pyvista_sections (slices, clips, thresholds).

Skips without ``pyvista``. The geometry is checked against analytic values: slice area = cross-section area; a clip
along cell faces keeps exactly half the volume; slicing a field linear in x returns that x on every slice node; a
quadratic hex20 cell slices to the right area; a threshold keeps exactly the cells whose value is in range.
Screenshots need an off-screen display (conftest.py ``display``).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools

import numpy as np
import pytest

pytest.importorskip("pyvista")
from fea_engine import mesh
from fea_engine.mesh import box_mesh, rectangle_mesh
from fea_engine.elements.solids import Hex20Solid3D
from fea_engine.visualization import pyvista_sections as ps

L, H, B = 2.0, 1.0, 1.0


def _box():
    return box_mesh(L, H, B, 8, 4, 4)


def _x(m):
    return m.nodes[:, 0].copy()


class TestGeometry:
    def test_slice_area_equals_cross_section(self):
        m = _box()
        sl = ps.slice_field(m, _x(m), normal="x", origin=(1.3, 0.5, 0.5))
        assert sl.n_cells > 0
        assert np.isclose(sl.compute_cell_sizes()["Area"].sum(), H * B)

    def test_slice_of_linear_field_returns_slice_position(self):
        m = _box()
        sl = ps.slice_field(m, _x(m), normal="x", origin=(1.3, 0.5, 0.5))
        name = [k for k in sl.point_data.keys() if "field" in k or "x" in k.lower()][0]
        assert np.allclose(sl.point_data[name], 1.3)

    def test_oblique_slice_normal(self):
        m = _box()
        sl = ps.slice_field(m, _x(m), normal=(1.0, 1.0, 0.0), origin=(1.0, 0.5, 0.5))
        assert sl.n_cells > 0
        assert np.allclose((sl.points - np.array([1.0, 0.5, 0.5])) @ np.array([1, 1, 0]) / np.sqrt(2), 0.0, atol=1e-12)

    def test_clip_along_cell_faces_keeps_half_the_volume(self):
        m = _box()
        half = ps.clip_field(m, _x(m), normal="x", origin=(1.0, 0.5, 0.5), invert=True)
        other = ps.clip_field(m, _x(m), normal="x", origin=(1.0, 0.5, 0.5), invert=False)
        v = lambda g: g.compute_cell_sizes()["Volume"].sum()
        assert np.isclose(v(half), 0.5 * L * H * B) and np.isclose(v(other), 0.5 * L * H * B)

    def test_clip_through_cell_interior_is_exact(self):
        m = _box()
        part = ps.clip_field(m, _x(m), normal="x", origin=(0.3, 0.5, 0.5), invert=False)   # keeps x >= 0.3
        assert np.isclose(part.compute_cell_sizes()["Volume"].sum(), (L - 0.3) * H * B)

    def test_quadratic_hex20_slice_area(self):
        f = Hex20Solid3D()
        grid = [-1.0, 0.0, 1.0]
        xi = []
        for a in range(f.n_nodes):
            for p in itertools.product(grid, repeat=3):
                N = f.shape_and_derivs(p)[0]
                if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                    xi.append(p)
                    break
        x = 0.5 * np.array(xi) + 0.5                                      # unit cube [0, 1]^3, straight-sided
        m = mesh.Mesh(nodes=x, elements=np.arange(f.n_nodes)[None, :], dim=3)
        sl = ps.slice_field(m, x[:, 0], normal="z", origin=(0.5, 0.5, 0.5))
        assert np.isclose(sl.compute_cell_sizes()["Area"].sum(), 1.0)

    def test_threshold_keeps_exactly_the_cells_in_range(self):
        m = _box()
        cell_x = m.nodes[m.elements].mean(axis=1)[:, 0]
        m.register_element_data("cx", cell_x)
        kept = ps.threshold_field(m, "cx", (0.5, 1.0))
        assert kept.n_cells == int(np.sum((cell_x >= 0.5) & (cell_x <= 1.0)))
        open_hi = ps.threshold_field(m, "cx", (1.5, None))
        assert open_hi.n_cells == int(np.sum(cell_x >= 1.5))

    def test_requires_3d(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(ValueError, match="3-D"):
            ps.slice_field(m, m.nodes[:, 0])

    def test_bad_normal(self):
        m = _box()
        with pytest.raises(ValueError, match="normal"):
            ps.slice_field(m, _x(m), normal=(0, 0, 0))

    def test_deformed_slice_moves_with_displacement(self):
        m = _box()
        U = np.zeros((len(m.nodes), 3))
        U[:, 2] = 0.1
        sl = ps.slice_field(m, _x(m), normal="x", origin=(1.0, 0.5, 0.5), U=U.ravel(), scale=1.0)
        assert np.isclose(sl.points[:, 2].min(), 0.1) and np.isclose(sl.points[:, 2].max(), B + 0.1)


class TestPlots:
    @staticmethod
    def _picture(path):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"))
        assert len(np.unique(img.reshape(-1, 3), axis=0)) > 20

    def test_plot_slice_many(self, tmp_path, display):
        m = _box()
        out = tmp_path / "s.png"
        ps.plot_slice(m, _x(m), normal="x", n_slices=3, screenshot=out, show=False)
        self._picture(out)

    def test_plot_clipped(self, tmp_path, display):
        m = _box()
        out = tmp_path / "c.png"
        ps.plot_clipped(m, _x(m), normal="y", screenshot=out, show=False)
        self._picture(out)

    def test_plot_threshold(self, tmp_path, display):
        m = _box()
        out = tmp_path / "t.png"
        ps.plot_threshold(m, _x(m), (1.0, None), screenshot=out, show=False)
        self._picture(out)

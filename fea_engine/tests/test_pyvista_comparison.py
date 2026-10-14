# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_pyvista_comparison.py -- validation of fea_engine.visualization.pyvista_comparison (full-order vs reduced-order).

Skips without ``pyvista``. The error metrics are checked against values known by construction (a field scaled by
1 - eps has relative L2 and max error eps; a rotated displacement has a nonzero vector error although the magnitudes
agree) and, end to end, against a real POD reduced-order model from rom_engine (skipped if rom_engine is absent): the
reported error equals the directly computed projection error and falls as the rank grows. Screenshots need an
off-screen display (conftest.py ``display``).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

pytest.importorskip("pyvista")
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress
from fea_engine.mesh import rectangle_mesh
from fea_engine.visualization import pyvista_comparison as pc


def _field(m):
    U = np.zeros((len(m.nodes), 2))
    U[:, 1] = -1e-3 * m.nodes[:, 0] ** 2
    return U.ravel()


class TestMetrics:
    def test_scaled_field_has_known_error(self):
        m = rectangle_mesh(2.0, 1.0, 8, 4)
        U = _field(m)
        for eps in (0.0, 0.05, 0.3):
            _, f, r, err, met = pc.field_error(m, U, (1.0 - eps) * U)
            assert np.isclose(met["rel_l2"], eps) and np.isclose(met["rel_max"], eps)
            assert np.isclose(met["rel_l2_displacement"], eps)
            assert np.isclose(err.max(), met["max_abs_error"])

    def test_vector_error_is_not_difference_of_magnitudes(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n = len(m.nodes)
        U = np.zeros((n, 2)); U[:, 0] = 1.0
        V = np.zeros((n, 2)); V[:, 1] = 1.0                        # same magnitude, orthogonal direction
        _, f, r, err, met = pc.field_error(m, U.ravel(), V.ravel())
        assert np.allclose(f, r) and np.allclose(err, np.sqrt(2.0))
        assert np.isclose(met["rel_l2"], np.sqrt(2.0))

    def test_component_field(self):
        m = rectangle_mesh(2.0, 1.0, 4, 2)
        U = _field(m)
        _, f, r, err, met = pc.field_error(m, U, 0.5 * U, field="uy")
        assert np.allclose(f, U[1::2]) and np.isclose(met["rel_l2"], 0.5)

    def test_callable_field(self):
        m = rectangle_mesh(2.0, 1.0, 4, 2)
        U = _field(m)
        _, f, r, err, met = pc.field_error(m, U, 0.8 * U, field=lambda W: np.abs(W[0::2]) + np.abs(W[1::2]))
        assert np.isclose(met["rel_l2"], 0.2)

    def test_zero_reference_gives_nan_not_an_error(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        z = np.zeros(2 * len(m.nodes))
        met = pc.field_error(m, z, z)[-1]
        assert np.isnan(met["rel_l2"]) and met["max_abs_error"] == 0.0

    def test_shape_mismatch(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(ValueError):
            pc.field_error(m, np.zeros(2 * len(m.nodes)), np.zeros(2 * len(m.nodes) + 1))


class TestAgainstRealPOD:
    def _model(self):
        mat = Material(E=210e9, nu=0.3, rho=7800.0)
        m = rectangle_mesh(4.0, 1.0, 16, 4)
        s = FESystem(m, Quad4PlaneStress(), thickness=0.1)
        s.assemble_stiffness(D_plane_stress(mat), thickness=0.1)
        for n in m.nodes_on_line(0, 0.0):
            s.fix_dofs([n], [0, 1])
        return m, s

    def test_reported_error_equals_projection_error_and_falls_with_rank(self):
        pod = pytest.importorskip("rom_engine.pod")
        m, s = self._model()
        rng = np.random.default_rng(0)
        nodes = np.arange(len(m.nodes))
        snaps = []
        for _ in range(6):
            s.F[:] = 0.0
            w = rng.standard_normal(len(m.nodes))
            s.F[1::2] = 1e3 * w
            snaps.append(np.asarray(s.solve_static()))
        snaps = np.array(snaps).T
        basis = pod.PodBasis().fit(snaps, n_modes=6)
        s.F[:] = 0.0
        s.F[1::2] = 1e3 * rng.standard_normal(len(m.nodes))
        U = np.asarray(s.solve_static())
        errs = []
        for r in (1, 3, 6):
            V = basis.V[:, :r]
            Ur = V @ (V.T @ U)
            met = pc.field_error(m, U, Ur)[-1]
            assert np.isclose(met["rel_l2_displacement"], np.linalg.norm(U - Ur) / np.linalg.norm(U))
            errs.append(met["rel_l2_displacement"])
        assert errs[0] >= errs[1] >= errs[2]


class TestPlot:
    def test_three_panel_screenshot_and_metrics(self, tmp_path, display):
        from PIL import Image
        m = rectangle_mesh(2.0, 1.0, 8, 4)
        U = _field(m)
        out = tmp_path / "cmp.png"
        pl, met = pc.compare_fields(m, U, 0.9 * U, screenshot=out, show=False)
        assert np.isclose(met["rel_l2"], 0.1)
        img = np.asarray(Image.open(out).convert("RGB"))
        assert img.shape[1] > img.shape[0] and len(np.unique(img.reshape(-1, 3), axis=0)) > 20

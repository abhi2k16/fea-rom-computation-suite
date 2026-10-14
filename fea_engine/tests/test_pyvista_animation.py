# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_pyvista_animation.py -- validation of fea_engine.visualization.pyvista_animation.

Skips without the optional ``pyvista`` (and, for GIF output, ``imageio``). Frame selection, colour limits, scale and
mode histories are checked without rendering; GIF writing needs an off-screen display (see conftest.py ``display``).

Lines of evidence: (1) frame selection (explicit, evenly spaced, errors); (2) the global colour limits and auto scale
are computed over ALL frames; (3) a mode history is one exact period per ``n_frames`` rows; (4) a written GIF has the
requested number of frames and consecutive frames differ; (5) a callable field (von Mises per frame) is accepted.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

pytest.importorskip("pyvista")
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress
from fea_engine.mesh import rectangle_mesh, box_mesh
from fea_engine.visualization import pyvista_animation as pa


def _hist(m, n_steps=6):
    n = len(m.nodes)
    U = np.zeros((n, m.dim))
    U[:, -1] = -1e-3 * m.nodes[:, 0] ** 2
    return np.sin(np.linspace(0, np.pi, n_steps))[:, None] * U.ravel()[None, :], U.ravel()


class TestFrameSelection:
    def test_all_rows_by_default(self):
        assert list(pa._frame_indices(5, None, None)) == [0, 1, 2, 3, 4]

    def test_evenly_spaced(self):
        idx = pa._frame_indices(101, None, 5)
        assert list(idx) == [0, 25, 50, 75, 100]

    def test_explicit_and_errors(self):
        assert list(pa._frame_indices(10, [2, 4], None)) == [2, 4]
        with pytest.raises(ValueError):
            pa._frame_indices(10, [10], None)
        with pytest.raises(ValueError):
            pa._frame_indices(10, [], None)
        with pytest.raises(ValueError):
            pa._frame_indices(10, None, 0)


class TestPrepare:
    def test_global_clim_and_scale_cover_all_frames(self):
        m = box_mesh(2.0, 1.0, 1.0, 4, 2, 2)
        hist, U = _hist(m, 6)
        p = pa._prepare(m, hist, "displacement", None, "auto", "global", None, None)
        mags = np.linalg.norm(hist.reshape(6, -1, 3), axis=2)
        assert np.isclose(p["clim"][0], mags.min()) and np.isclose(p["clim"][1], mags.max())
        diag = np.linalg.norm(m.nodes.max(0) - m.nodes.min(0))
        assert np.isclose(p["scale"], 0.1 * diag / np.abs(hist).max())

    def test_fixed_clim_and_scale_pass_through(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        hist, _ = _hist(m, 3)
        p = pa._prepare(m, hist, "ux", None, 5.0, (0.0, 1.0), None, None)
        assert p["clim"] == (0.0, 1.0) and p["scale"] == 5.0

    def test_callable_field_is_evaluated_per_frame(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        hist, _ = _hist(m, 4)
        p = pa._prepare(m, hist, lambda U: np.abs(U[0::2]) + 1.0, None, "auto", "global", None, None)
        assert p["vals"].shape == (4, len(m.nodes)) and p["vals"].min() >= 1.0

    def test_bad_inputs(self):
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(ValueError, match="2-D"):
            pa._prepare(m, np.zeros(10), "ux", None, 1.0, "global", None, None)
        hist, _ = _hist(m, 3)
        with pytest.raises(ValueError, match="clim"):
            pa._prepare(m, hist, "ux", None, 1.0, "bogus", None, None)


class TestModeHistory:
    def test_periodic_and_exact(self):
        phi = np.arange(1.0, 7.0)
        h = pa.mode_history(phi, n_frames=8, n_cycles=2)
        assert h.shape == (16, 6)
        assert np.allclose(h[0], 0.0) and np.allclose(h[8], 0.0)
        assert np.allclose(h[:8], h[8:])
        assert np.allclose(h[2], phi) and np.allclose(h[6], -phi)     # quarter and three-quarter period


class TestGif:
    imageio = pytest.importorskip("imageio")

    def test_gif_frame_count_and_motion(self, tmp_path, display):
        m = box_mesh(2.0, 1.0, 1.0, 4, 2, 2)
        hist, _ = _hist(m, 8)
        out = pa.animate_history(m, hist, tmp_path / "h.gif", times=np.arange(8) * 0.1, fps=5)
        assert out["n_frames"] == 8
        frames = self.imageio.mimread(str(tmp_path / "h.gif"))
        assert len(frames) == 8
        assert np.abs(frames[1].astype(int) - frames[3].astype(int)).mean() > 0.5

    def test_frame_subset(self, tmp_path, display):
        m = rectangle_mesh(2.0, 1.0, 4, 2)
        hist, _ = _hist(m, 20)
        out = pa.animate_history(m, hist, tmp_path / "s.gif", n_frames=5)
        assert out["n_frames"] == 5 and len(self.imageio.mimread(str(tmp_path / "s.gif"))) == 5

    def test_mode_animation_loops_seamlessly(self, tmp_path, display):
        m = box_mesh(2.0, 1.0, 1.0, 4, 2, 2)
        _, U = _hist(m)
        pa.animate_mode(m, U, tmp_path / "m.gif", n_frames=8, freq=12.5)
        frames = self.imageio.mimread(str(tmp_path / "m.gif"))
        assert len(frames) == 8
        # frame 0 (u = 0) differs from frame 2 (peak) but frame 4 (u = 0 again) equals frame 0
        assert np.abs(frames[0].astype(int) - frames[2].astype(int)).mean() > 0.5
        assert np.abs(frames[0].astype(int) - frames[4].astype(int)).mean() < 0.5

    def test_stress_by_callable_on_a_solved_transient(self, tmp_path, display):
        mat = Material(E=210e9, nu=0.3, rho=7800.0)
        m = rectangle_mesh(4.0, 1.0, 16, 4)
        s = FESystem(m, Quad4PlaneStress(), thickness=0.1)
        s.assemble_stiffness(D_plane_stress(mat), thickness=0.1)
        for n in m.nodes_on_line(0, 0.0):
            s.fix_dofs([n], [0, 1])
        tip = np.where(np.abs(m.nodes[:, 0] - 4.0) < 1e-9)[0]
        hist = []
        for f in (0.2, 0.6, 1.0):
            s.F[:] = 0.0
            s.F[2 * tip + 1] = -1e4 * f / len(tip)
            hist.append(np.asarray(s.solve_static()))
        out = pa.animate_history(m, np.array(hist), tmp_path / "vm.gif",
                                 field=lambda U: s.von_mises(U), component="von_mises")
        vm_last = s.von_mises(hist[-1]).component("von_mises")
        assert np.isclose(out["clim"][1], vm_last.max()) and out["n_frames"] == 3

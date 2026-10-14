# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
Shared pytest configuration for the fea_engine test suite.

Forces matplotlib's non-interactive Agg backend before any test module
imports pyplot, since several tests (test_nonlinear_beam,
test_nonlinear_cantilever) save validation plots as a side effect and
must not require a display.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import matplotlib
matplotlib.use("Agg")


# ---- PyVista rendering tests: skip unless an off-screen / virtual display works (checked once, in a subprocess,
# because VTK can crash the interpreter when no display exists). Use as ``def test_x(display): ...``.
import subprocess
import sys

import pytest

_RENDER_OK = None


def _can_render():
    code = ("import pyvista as pv\n"
            "pl = pv.Plotter(off_screen=True)\n"
            "pl.add_mesh(pv.Sphere())\n"
            "assert pl.screenshot(return_img=True).size > 0\n")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


@pytest.fixture
def display():
    global _RENDER_OK
    if _RENDER_OK is None:
        _RENDER_OK = _can_render()
    if not _RENDER_OK:
        pytest.skip("no off-screen/virtual display for VTK (try xvfb-run -a pytest ...)")

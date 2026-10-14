# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
visualization -- optional plotting helpers.

pyvista_plot   3D mesh and result visualization with PyVista (``pip install fea_engine[viz]``): Mesh/MultiBlockMesh to
               ``pyvista.UnstructuredGrid``, field contours, deformed shapes, VTK export. See that module's docstring.

pyvista_animation   GIF/MP4 animations of time histories, load paths and mode shapes.
pyvista_sections    slices, clips and thresholds of 3-D meshes.
pyvista_comparison  full-order versus reduced-order side-by-side plots with error metrics.

The functions are re-exported here and import PyVista lazily, so ``import fea_engine`` works without it.
Plain-matplotlib mesh plots (``fea_engine.mesh.plot_mesh_2d/plot_mesh_3d/plot_mesh_annotated``) and the matplotlib
field/series plots on ``FEField`` / ``FieldSeries`` are independent of this sub-package and need only matplotlib.

A Gmsh-native screenshot renderer (formerly ``gmsh_plot.py``) has been removed: it needed ``pip install gmsh`` plus the
system libGLU library, which could not be reliably provided.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from .pyvista_plot import (to_pyvista, add_displacement, displacement_vectors, deformed_grid, plot_field,
                           plot_deformed, save_vtk, set_off_screen, VTK_CELL_TYPES)

from .pyvista_animation import animate_history, animate_mode, mode_history
from .pyvista_sections import slice_field, clip_field, threshold_field, plot_slice, plot_clipped, plot_threshold
from .pyvista_comparison import compare_fields, field_error

__all__ = ["animate_history", "animate_mode", "mode_history", "slice_field", "clip_field", "threshold_field",
           "plot_slice", "plot_clipped", "plot_threshold", "compare_fields", "field_error", "to_pyvista", "add_displacement", "displacement_vectors", "deformed_grid", "plot_field", "plot_deformed",
           "save_vtk", "set_off_screen", "VTK_CELL_TYPES"]

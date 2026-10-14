# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
pyvista_sections.py -- slices, clips and thresholds of 3-D meshes with PyVista (optional dependency,
``pip install fea_engine[viz]``), for looking inside a solid where the outer surface hides the interesting stress.

- ``slice_field(mesh, field, normal, origin)``       the cut surface where a plane meets the mesh, carrying the field
- ``clip_field(mesh, field, normal, origin)``        the part of the mesh on one side of a plane, as volume cells
- ``threshold_field(mesh, field, value)``            the cells (or cells touching nodes) whose field lies in a range
- ``plot_slice`` / ``plot_clipped`` / ``plot_threshold``   the corresponding coloured plots

``normal`` is ``"x"``, ``"y"``, ``"z"`` or a 3-vector; ``origin`` is a point on the plane (default: the mesh centre).
``plot_slice(..., n_slices=k)`` draws k parallel slices evenly spaced across the mesh along the normal. ``field`` and
``U``/``scale`` are as in ``pyvista_plot.plot_field`` (any field form, optional deformed geometry). All of these need a
3-D mesh (``mesh.dim == 3``). Slices and clips of the quadratic cells (tet10, hex20) are computed by VTK on the
cells' linearised geometry.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .pyvista_plot import (_bar, _finish, _make_plotter, _resolve_field, _auto_scale, displacement_vectors,
                           to_pyvista)

__all__ = ["slice_field", "clip_field", "threshold_field", "plot_slice", "plot_clipped", "plot_threshold"]

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _need_3d(mesh):
    if int(mesh.dim) != 3:
        raise ValueError("slices, clips and thresholds need a 3-D mesh (mesh.dim == 3)")


def _normal(normal):
    n = np.asarray(_AXES[normal] if isinstance(normal, str) else normal, dtype=float)
    if n.shape != (3,) or np.linalg.norm(n) == 0:
        raise ValueError("normal must be 'x', 'y', 'z' or a nonzero 3-vector")
    return n / np.linalg.norm(n)


def _prepared_grid(mesh, field, U, scale, component, name):
    """Grid (deformed if U and scale) with the resolved field attached as ``name``; returns (grid, assoc, name)."""
    _need_3d(mesh)
    grid = to_pyvista(mesh)
    assoc, values, auto = _resolve_field(grid, mesh, field, U, component)
    name = name or auto
    heading = auto
    if U is not None and scale:
        s = _auto_scale(mesh, U) if scale == "auto" else float(scale)
        grid.point_data["_u"] = displacement_vectors(mesh, U)
        grid = grid.warp_by_vector("_u", factor=s)
        heading += f"  (deformation x{s:.3g})"
    (grid.point_data if assoc == "point" else grid.cell_data)[name] = np.asarray(values, dtype=float)
    return grid, assoc, name, heading


def _origin(grid, origin):
    return np.asarray(grid.center if origin is None else origin, dtype=float)


def slice_field(mesh, field, normal="x", origin=None, U=None, scale=0.0, component=None):
    """The cut surface (``pyvista.PolyData``) where the plane through ``origin`` with ``normal`` meets the mesh, with
    the field attached under its title (e.g. ``"von Mises stress (von_mises)"``)."""
    grid, _, name, _ = _prepared_grid(mesh, field, U, scale, component, None)
    return grid.slice(normal=tuple(_normal(normal)), origin=tuple(_origin(grid, origin)))


def clip_field(mesh, field, normal="x", origin=None, invert=True, U=None, scale=0.0, component=None):
    """The part of the mesh on one side of the plane, as an ``UnstructuredGrid``. ``invert=True`` keeps the side the
    normal points away from, ``False`` the side it points into. Cells cut by the plane are cut exactly (no whole-cell
    rounding)."""
    grid, _, name, _ = _prepared_grid(mesh, field, U, scale, component, None)
    return grid.clip(normal=tuple(_normal(normal)), origin=tuple(_origin(grid, origin)), invert=invert,
                     crinkle=False)


def threshold_field(mesh, field, value, U=None, scale=0.0, component=None, all_scalars=True):
    """The cells whose field lies in ``value = (lo, hi)`` (either bound may be ``None``). For nodal fields a cell
    is kept if ALL its nodes are in range (``all_scalars=True``) or ANY node is (``False``); for element fields the
    cell's own value decides."""
    grid, assoc, name, _ = _prepared_grid(mesh, field, U, scale, component, None)
    lo, hi = value
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return grid.threshold(value=(lo, hi), scalars=name, preference=assoc if assoc == "cell" else "point",
                          all_scalars=all_scalars)


def _plot(pieces, grid_outline, name, heading, cmap, clim, show_edges, title, window_size, screenshot, show,
          off_screen, outline_color="gray"):
    pl = _make_plotter(window_size, screenshot, off_screen)
    for piece in pieces:
        if piece.n_cells:
            pl.add_mesh(piece, scalars=name, cmap=cmap, clim=clim, show_edges=show_edges,
                        scalar_bar_args=_bar(name))
    pl.add_mesh(grid_outline.outline(), color=outline_color, line_width=1)
    pl.add_text(title or heading, font_size=10)
    return _finish(pl, screenshot, show)


def _global_clim(grid, name, assoc, clim):
    if clim is not None:
        return clim
    arr = np.asarray((grid.point_data if assoc == "point" else grid.cell_data)[name])
    return (float(arr.min()), float(arr.max()))


def plot_slice(mesh, field, normal="x", origin=None, n_slices=1, U=None, scale=0.0, component=None, cmap="turbo",
               clim=None, show_edges=False, show_mesh=True, title=None, window_size=(900, 650), screenshot=None,
               show=True, off_screen=None):
    """Plot one slice (``origin``) or ``n_slices`` parallel slices evenly spaced across the mesh along ``normal``,
    coloured by ``field`` on shared colour limits (default: the field's range over the whole mesh, so a slice shows
    where it sits in the global range). ``show_mesh``: draw the mesh outline faintly. Returns the Plotter."""
    grid, assoc, name, heading = _prepared_grid(mesh, field, U, scale, component, None)
    n = _normal(normal)
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    if n_slices == 1:
        planes = [_origin(grid, origin)]
    else:
        proj = grid.points @ n
        c0 = _origin(grid, origin)
        ts = np.linspace(proj.min(), proj.max(), n_slices + 2)[1:-1]
        planes = [c0 + (t - c0 @ n) * n for t in ts]
    slices = [grid.slice(normal=tuple(n), origin=tuple(p)) for p in planes]
    lim = _global_clim(grid, name, assoc, clim)
    pl = _make_plotter(window_size, screenshot, off_screen)
    if show_mesh:
        pl.add_mesh(grid, style="wireframe", color="lightgray", opacity=0.25, line_width=1)
    for sl in slices:
        if sl.n_cells:
            pl.add_mesh(sl, scalars=name, cmap=cmap, clim=lim, show_edges=show_edges, lighting=False,
                        scalar_bar_args=_bar(name))
    pl.add_mesh(grid.outline(), color="gray", line_width=1)
    pl.add_text(title or f"{heading}: {n_slices} slice(s), normal {np.round(n, 3).tolist()}", font_size=10)
    return _finish(pl, screenshot, show)


def plot_clipped(mesh, field, normal="x", origin=None, invert=True, U=None, scale=0.0, component=None, cmap="turbo",
                 clim=None, show_edges=True, title=None, window_size=(900, 650), screenshot=None, show=True,
                 off_screen=None):
    """Plot the part of the mesh on one side of a plane (the cut face shows the interior field)."""
    grid, assoc, name, heading = _prepared_grid(mesh, field, U, scale, component, None)
    part = grid.clip(normal=tuple(_normal(normal)), origin=tuple(_origin(grid, origin)), invert=invert,
                     crinkle=False)
    lim = _global_clim(grid, name, assoc, clim)
    return _plot([part], grid, name, heading + ": clipped", cmap, lim, show_edges, title, window_size, screenshot,
                 show, off_screen)


def plot_threshold(mesh, field, value, U=None, scale=0.0, component=None, cmap="turbo", clim=None, show_edges=True,
                   title=None, window_size=(900, 650), screenshot=None, show=True, off_screen=None,
                   all_scalars=True):
    """Plot only the cells whose field lies in ``value = (lo, hi)`` (for example the highly stressed region), with the
    whole mesh as an outline for context."""
    grid, assoc, name, heading = _prepared_grid(mesh, field, U, scale, component, None)
    lo, hi = value
    part = grid.threshold(value=(-np.inf if lo is None else lo, np.inf if hi is None else hi), scalars=name,
                          preference="cell" if assoc == "cell" else "point", all_scalars=all_scalars)
    lim = _global_clim(grid, name, assoc, clim)
    return _plot([part], grid, name, heading + f": {lo} <= field <= {hi}", cmap, lim, show_edges, title,
                 window_size, screenshot, show, off_screen)

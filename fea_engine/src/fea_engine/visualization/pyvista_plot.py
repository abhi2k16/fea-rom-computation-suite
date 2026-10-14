# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
pyvista_plot.py -- 3D visualization of meshes and FE results with PyVista (the Python interface to VTK).

PyVista is an OPTIONAL dependency (``pip install fea_engine[viz]``); importing this module never imports it, and
the rest of fea_engine stays numpy/scipy only. Every function raises a clear ImportError if PyVista is missing.

Pipeline
--------
1. ``to_pyvista(mesh)``                 Mesh / MultiBlockMesh -> ``pyvista.UnstructuredGrid`` (nodes, connectivity, cell
                                        types; registered point_data / cell_data are carried over).
2. ``add_displacement(grid, U)``        attach a nodal displacement vector and its magnitude.
3. ``plot_field(mesh, field, ...)``     colour-contour plot of a nodal or element field (displacement component or
                                        magnitude, von Mises stress, any registered field, any array), with a colorbar.
4. ``plot_deformed(mesh, U, ...)``      the deformed shape (displacement scaled by ``scale``), optionally coloured by a
                                        field and overlaid with the undeformed outline.
5. ``save_vtk(mesh, path, ...)``        write a ``.vtu``/``.vtk`` file that ParaView opens.

Supported cells and node ordering: line2, tri3, quad4, tet4, hex8 and the quadratic tri6, quad8, tet10, hex20. These
are the element types of ``Mesh`` / ``MultiBlockMesh``; fea_engine's node ordering for them is already the VTK
ordering (checked in tests/test_pyvista_plot.py by comparing the VTK cell volume of straight-sided quadratic cells
with the analytic volume). 2-D meshes are padded with z = 0. Beam, truss and shell elements are not ``Mesh`` objects
and are not handled here.

Displacement input ``U`` may be a flat vector ``(n_nodes * dofs_per_node,)`` (e.g. ``FESystem.solve_static()`` output
or an ``FEField``) or an array ``(n_nodes, k)``. The first ``mesh.dim`` DOFs of each node are taken as the
translations; any further DOFs (rotations) are ignored.

Field input ``field`` is one of: a string naming a registered point/cell array; ``"displacement"`` (magnitude, needs
``U``); ``"ux"``/``"uy"``/``"uz"`` (needs ``U``); an ``FEField`` (single value per node, or several values with
``component=``); a plain array of length n_nodes (point data) or n_elements (cell data); or a callable ``field(U)``
returning any of these for the given ``U`` (for example ``lambda U: sys.von_mises(U)``). Multi-component arrays
without ``component=`` are shown as the Euclidean norm.

Related modules: ``pyvista_animation`` (time histories and mode shapes to GIF/MP4), ``pyvista_sections`` (slices,
clips, thresholds of 3-D meshes) and ``pyvista_comparison`` (full-order versus reduced-order side-by-side plots).

Headless use: pass ``screenshot="out.png"`` (rendered off-screen, no window) or call ``set_off_screen(True)``. On a
machine with no display and no OpenGL, VTK needs an off-screen build (e.g. ``vtk-osmesa``) or xvfb.

References (methods in this module are adopted from):
  - W. Schroeder, K. Martin, B. Lorensen (2006), "The Visualization Toolkit: An Object-Oriented Approach to 3D
    Graphics", 4th ed., Kitware (cell types, warping, scalar mapping).
  - C. B. Sullivan, A. Kaszynski (2019), "PyVista: 3D plotting and mesh analysis through a streamlined interface for
    the Visualization Toolkit (VTK)", Journal of Open Source Software 4(37):1450.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

__all__ = ["to_pyvista", "add_displacement", "displacement_vectors", "deformed_grid", "plot_field", "plot_deformed",
           "save_vtk", "set_off_screen", "VTK_CELL_TYPES"]

# (nodes per element, topological dimension) -> VTK cell type id; and short block name -> VTK id.
VTK_CELL_TYPES = {(2, 1): 3, (3, 2): 5, (4, 2): 9, (4, 3): 10, (8, 3): 12, (6, 2): 22, (8, 2): 23, (10, 3): 24,
                  (20, 3): 25}
_BLOCK_TO_VTK = {"line2": 3, "tri3": 5, "quad4": 9, "tet4": 10, "hex8": 12, "tri6": 22, "quad8": 23, "tet10": 24,
                 "hex20": 25}


def _pv():
    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - exercised only without pyvista
        raise ImportError("fea_engine.visualization.pyvista_plot needs the optional 'pyvista' package: "
                          "pip install pyvista (or `pip install fea_engine[viz]`).") from exc
    return pyvista


def set_off_screen(flag=True):
    """Render without opening a window for every later plot (``pyvista.OFF_SCREEN``)."""
    _pv().OFF_SCREEN = bool(flag)


def _points3(nodes):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    pts = np.zeros((nodes.shape[0], 3))
    pts[:, :nodes.shape[1]] = nodes
    return pts


def _cells(mesh):
    """(cell_array, cell_types, cell_slices) for a Mesh or MultiBlockMesh; slices give each block's cell range."""
    from ..mesh import MultiBlockMesh
    if isinstance(mesh, MultiBlockMesh):
        blocks = [(name, np.asarray(conn, dtype=np.int64)) for name, conn in mesh.blocks.items()]
        types = []
        for name, conn in blocks:
            if name not in _BLOCK_TO_VTK:
                raise ValueError(f"unknown block name {name!r}; supported: {sorted(_BLOCK_TO_VTK)}")
            types.append(_BLOCK_TO_VTK[name])
    else:
        conn = np.asarray(mesh.elements, dtype=np.int64)
        key = (conn.shape[1], int(mesh.dim))
        if key not in VTK_CELL_TYPES:
            raise ValueError(f"no VTK cell type for {conn.shape[1]} nodes/element at dim={mesh.dim}; "
                             f"supported: {sorted(VTK_CELL_TYPES)}")
        blocks, types = [(None, conn)], [VTK_CELL_TYPES[key]]
    parts, ctypes, slices, start = [], [], {}, 0
    for (name, conn), t in zip(blocks, types):
        ne, npe = conn.shape
        parts.append(np.hstack([np.full((ne, 1), npe, dtype=np.int64), conn]).ravel())
        ctypes.append(np.full(ne, t, dtype=np.uint8))
        slices[name] = (start, start + ne)
        start += ne
    return np.concatenate(parts), np.concatenate(ctypes), slices


def to_pyvista(mesh, point_data=None, cell_data=None):
    """Convert a ``Mesh`` / ``MultiBlockMesh`` to a ``pyvista.UnstructuredGrid``.

    Arrays registered on the mesh (``register_point_data`` / ``register_element_data``) are attached; ``point_data``
    and ``cell_data`` (dicts name -> array, first axis n_nodes / n_cells) are added on top. For a MultiBlockMesh the
    cells are the blocks' cells concatenated in block order, and per-block cell data is concatenated (a name must be
    present in every block)."""
    pv = _pv()
    cells, ctypes, slices = _cells(mesh)
    grid = pv.UnstructuredGrid(cells, ctypes, _points3(mesh.nodes))
    n_nodes, n_cells = grid.n_points, grid.n_cells
    for name, arr in (getattr(mesh, "point_data", None) or {}).items():
        arr = np.asarray(arr)
        if arr.shape[0] == n_nodes:
            grid.point_data[name] = arr
    reg = getattr(mesh, "cell_data", None) or {}
    if None not in slices:                          # MultiBlockMesh: {block: {name: array}}
        names = set.intersection(*[set(reg.get(b, {})) for b in slices]) if reg else set()
        for name in names:
            grid.cell_data[name] = np.concatenate([np.asarray(reg[b][name]) for b in slices])
    else:
        for name, arr in reg.items():
            arr = np.asarray(arr)
            if arr.shape[0] == n_cells:
                grid.cell_data[name] = arr
    for name, arr in (point_data or {}).items():
        arr = np.asarray(arr)
        if arr.shape[0] != n_nodes:
            raise ValueError(f"point_data[{name!r}] has {arr.shape[0]} rows, expected {n_nodes} (one per node)")
        grid.point_data[name] = arr
    for name, arr in (cell_data or {}).items():
        arr = np.asarray(arr)
        if arr.shape[0] != n_cells:
            raise ValueError(f"cell_data[{name!r}] has {arr.shape[0]} rows, expected {n_cells} (one per cell)")
        grid.cell_data[name] = arr
    return grid


def displacement_vectors(mesh, U):
    """Nodal translations as an ``(n_nodes, 3)`` array (z padded with 0 for 2-D meshes); see the module docstring."""
    n, dim = len(mesh.nodes), int(mesh.dim)
    U = np.asarray(U, dtype=float)
    if U.ndim == 2:
        if U.shape[0] != n:
            raise ValueError(f"U has {U.shape[0]} rows, expected {n} (one per node)")
        k = U.shape[1]
        vec = U
    else:
        if U.size % n != 0:
            raise ValueError(f"U has length {U.size}, not a multiple of the node count {n}")
        k = U.size // n
        vec = U.reshape(n, k)
    if k < dim:
        raise ValueError(f"U has {k} DOFs per node, fewer than the mesh dimension {dim}")
    out = np.zeros((n, 3))
    out[:, :dim] = vec[:, :dim]
    return out


def add_displacement(grid, U, dim=None, name="displacement"):
    """Attach the nodal displacement vector (``name``) and its norm (``name + "_magnitude"``) to ``grid`` in place.
    ``dim``: number of translational DOFs per node; default = the DOFs per node of ``U`` if that is at most 3,
    otherwise 3 if the grid is not planar and 2 if it is."""
    n = grid.n_points
    Uarr = np.asarray(U, dtype=float)
    k = Uarr.shape[1] if Uarr.ndim == 2 else Uarr.size // n
    if dim is None:
        dim = k if k <= 3 else (3 if np.ptp(grid.points[:, 2]) > 0 else 2)

    class _M:
        nodes = grid.points

    _M.dim = dim
    vec = displacement_vectors(_M, U)
    grid.point_data[name] = vec
    grid.point_data[name + "_magnitude"] = np.linalg.norm(vec, axis=1)
    return grid


def deformed_grid(mesh, U, scale=1.0):
    """The grid with node positions moved by ``scale * displacement`` (``to_pyvista`` + warp)."""
    grid = to_pyvista(mesh)
    grid.point_data["displacement"] = displacement_vectors(mesh, U)
    grid.point_data["displacement_magnitude"] = np.linalg.norm(grid.point_data["displacement"], axis=1)
    return grid.warp_by_vector("displacement", factor=float(scale))


def _auto_scale(mesh, U, fraction=0.1):
    vec = displacement_vectors(mesh, U)
    umax = np.abs(vec).max()
    if umax == 0.0:
        return 1.0
    pts = np.asarray(mesh.nodes, dtype=float)
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return fraction * diag / umax


def _resolve_field(grid, mesh, field, U, component):
    """-> (association 'point'|'cell', 1-D array, title)."""
    n, nc = grid.n_points, grid.n_cells
    if callable(field) and not isinstance(field, (str, np.ndarray)):
        if U is None:
            raise ValueError("a callable field(U) needs U=")
        field = field(U)
    if isinstance(field, str):
        if field in ("displacement", "ux", "uy", "uz"):
            if U is None:
                raise ValueError(f"field={field!r} needs U=")
            vec = displacement_vectors(mesh, U)
            if field == "displacement":
                return "point", np.linalg.norm(vec, axis=1), "displacement magnitude"
            return "point", vec[:, "xyz".index(field[1])], f"displacement {field[1]}"
        if field in grid.point_data:
            arr, assoc = np.asarray(grid.point_data[field]), "point"
        elif field in grid.cell_data:
            arr, assoc = np.asarray(grid.cell_data[field]), "cell"
        else:
            raise KeyError(f"field {field!r} is not a registered point/cell array "
                           f"(point: {list(grid.point_data)}, cell: {list(grid.cell_data)})")
        title = field
    else:
        label = getattr(field, "label", None)
        arr = np.asarray(field, dtype=float)
        if hasattr(field, "component") and component is not None and getattr(field, "_dof_names", None):
            arr = np.asarray(field.component(component), dtype=float)
            return "point", arr, f"{label or 'field'} ({component})"
        if arr.ndim == 1 and arr.size not in (n, nc):
            k = getattr(field, "_npn", None) or (arr.size // n if arr.size % n == 0 else None)
            if k is None or arr.size != n * k:
                raise ValueError(f"field has length {arr.size}; expected n_nodes={n} or n_cells={nc} "
                                 f"(or n_nodes * k)")
            arr = arr.reshape(n, k)
        assoc = "cell" if (arr.shape[0] == nc and arr.shape[0] != n) else "point"
        title = label or "field"
    if arr.ndim == 2:
        if component is not None:
            arr = arr[:, int(component)]
            title += f" [{component}]"
        else:
            arr = np.linalg.norm(arr, axis=1)
            title += " (norm)"
    return assoc, arr, title


def _finish(pl, screenshot, show):
    if screenshot is not None:
        pl.screenshot(str(screenshot))
        if not show:
            pl.close()
            return pl
    if show and not pl.off_screen:
        pl.show()
    return pl


def _bar(title):
    """Vertical colorbar on the right, so a long title never overlaps the tick labels."""
    return {"title": title, "vertical": True, "position_x": 0.86, "position_y": 0.15, "width": 0.05, "height": 0.7,
            "title_font_size": 12, "label_font_size": 10, "fmt": "%.3g"}


def _make_plotter(window_size, screenshot, off_screen):
    pv = _pv()
    if off_screen is None:
        off_screen = screenshot is not None or bool(pv.OFF_SCREEN)
    return pv.Plotter(off_screen=bool(off_screen), window_size=list(window_size))


def plot_field(mesh, field, U=None, component=None, scale=0.0, cmap="turbo", show_edges=True, edge_color="black",
               clim=None, title=None, scalar_bar_title=None, window_size=(900, 650), screenshot=None, show=True,
               off_screen=None):
    """Colour-contour plot of ``field`` on ``mesh`` (see the module docstring for accepted ``field`` forms).

    ``U`` / ``scale``: if ``U`` is given and ``scale`` is nonzero the mesh is drawn deformed by ``scale * U``
    (``scale="auto"`` chooses 10 % of the model size for the largest displacement). ``component``: index or name
    selecting one component of a multi-component field. ``clim``: ``(min, max)`` colour limits. ``screenshot``: PNG
    path; the scene is rendered off-screen and saved. Returns the ``pyvista.Plotter``."""
    grid = to_pyvista(mesh)
    assoc, values, auto_title = _resolve_field(grid, mesh, field, U, component)
    heading = auto_title
    if U is not None and scale:
        s = _auto_scale(mesh, U) if scale == "auto" else float(scale)
        grid.point_data["_u"] = displacement_vectors(mesh, U)
        grid = grid.warp_by_vector("_u", factor=s)
        heading += f"  (deformation x{s:.3g})"
    name = scalar_bar_title or auto_title
    (grid.point_data if assoc == "point" else grid.cell_data)[name] = np.asarray(values, dtype=float)
    pl = _make_plotter(window_size, screenshot, off_screen)
    pl.add_mesh(grid, scalars=name, cmap=cmap, show_edges=show_edges, edge_color=edge_color, clim=clim,
                scalar_bar_args=_bar(name))
    pl.add_text(title or heading, font_size=10)
    if int(mesh.dim) == 2:
        pl.view_xy()
    return _finish(pl, screenshot, show)


def plot_deformed(mesh, U, scale="auto", field="displacement", component=None, show_undeformed=True, cmap="turbo",
                  show_edges=True, clim=None, title=None, window_size=(900, 650), screenshot=None, show=True,
                  off_screen=None):
    """Deformed shape: node positions moved by ``scale * U`` (``"auto"`` = largest displacement drawn as 10 % of the
    model size), coloured by ``field`` (default displacement magnitude; ``None`` for a plain grey mesh), with the
    undeformed mesh as a translucent wire outline if ``show_undeformed``. Returns the ``pyvista.Plotter``."""
    s = _auto_scale(mesh, U) if scale == "auto" else float(scale)
    grid = to_pyvista(mesh)
    grid.point_data["_u"] = displacement_vectors(mesh, U)
    warped = grid.warp_by_vector("_u", factor=s)
    pl = _make_plotter(window_size, screenshot, off_screen)
    label = None
    if field is not None:
        assoc, values, label = _resolve_field(warped, mesh, field, U, component)
        (warped.point_data if assoc == "point" else warped.cell_data)[label] = np.asarray(values, dtype=float)
        pl.add_mesh(warped, scalars=label, cmap=cmap, show_edges=show_edges, clim=clim,
                    scalar_bar_args=_bar(label))
    else:
        pl.add_mesh(warped, color="lightsteelblue", show_edges=show_edges)
    if show_undeformed:
        pl.add_mesh(grid, style="wireframe", color="gray", opacity=0.5, line_width=1)
    pl.add_text(title or f"deformed shape (displacement x{s:.3g})" + (f", {label}" if label else ""), font_size=10)
    if int(mesh.dim) == 2:
        pl.view_xy()
    return _finish(pl, screenshot, show)


def save_vtk(mesh, path, U=None, point_data=None, cell_data=None):
    """Write ``mesh`` (+ optional displacement ``U`` and extra arrays) to a VTK file; the extension picks the format
    (``.vtu`` XML unstructured grid, ``.vtk`` legacy). Opens directly in ParaView. Returns the grid that was saved."""
    grid = to_pyvista(mesh, point_data=point_data, cell_data=cell_data)
    if U is not None:
        grid.point_data["displacement"] = displacement_vectors(mesh, U)
        grid.point_data["displacement_magnitude"] = np.linalg.norm(grid.point_data["displacement"], axis=1)
    grid.save(str(path))
    return grid

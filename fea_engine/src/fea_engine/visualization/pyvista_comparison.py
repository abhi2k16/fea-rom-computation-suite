# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
pyvista_comparison.py -- full-order versus reduced-order side-by-side plots with PyVista (optional dependency,
``pip install fea_engine[viz]``).

``compare_fields(mesh, U_full, U_rom, ...)`` draws three linked panels on the same mesh: the full-order field, the
reduced-order field, and the pointwise error, with shared colour limits for the first two and its own colorbar for
the error. It returns ``(plotter, metrics)``; the metrics are computed on the plotted quantity.

The error of a vector displacement is the norm of the vector difference ``|u_full - u_rom|`` (not the difference of
magnitudes, which can vanish for a wrong direction); for a scalar or component field it is ``|f_full - f_rom|``;
for a callable or multi-component field it is the norm of the difference of the underlying arrays.

Metrics (dict): ``rel_l2`` = ||f_full - f_rom||_2 / ||f_full||_2 over the nodes (or cells), ``rel_max`` =
max|f_full - f_rom| / max|f_full|, ``max_abs_error``, ``rel_l2_displacement`` (always, from the displacement vectors).

Typical use with rom_engine: ``U_rom = basis.expand(basis.project(U_free))`` embedded back into the full DOF vector,
or the reconstruction ``V @ q`` of a reduced solve; this module only needs the two full-size vectors.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .pyvista_plot import (_auto_scale, _bar, _pv, _resolve_field, displacement_vectors, to_pyvista)

__all__ = ["compare_fields", "field_error"]


def _values(grid, mesh, field, U, component):
    """(assoc, 2-D array (n, k), title); k = 1 for scalars, 3 for displacement magnitude (so errors use vectors)."""
    if isinstance(field, str) and field == "displacement":
        return "point", displacement_vectors(mesh, U), "displacement magnitude"
    assoc, v, title = _resolve_field(grid, mesh, field, U, component)
    return assoc, np.asarray(v, dtype=float).reshape(len(v), -1), title


def field_error(mesh, U_full, U_rom, field="displacement", component=None):
    """The three arrays plotted by :func:`compare_fields` and the metrics dict, without any plotting.

    Returns ``(assoc, full, rom, err, metrics)`` where ``full``, ``rom`` and ``err`` are 1-D (norms for vector
    quantities)."""
    grid = to_pyvista(mesh)
    a1, f, title = _values(grid, mesh, field, U_full, component)
    a2, r, _ = _values(grid, mesh, field, U_rom, component)
    if a1 != a2 or f.shape != r.shape:
        raise ValueError("full-order and reduced-order fields must have the same association and shape")
    diff = f - r
    err = np.linalg.norm(diff, axis=1)
    fv, rv = np.linalg.norm(f, axis=1), np.linalg.norm(r, axis=1)
    if f.shape[1] == 1:
        fv, rv = f[:, 0], r[:, 0]
    ud = displacement_vectors(mesh, U_full) - displacement_vectors(mesh, U_rom)
    uref = np.linalg.norm(displacement_vectors(mesh, U_full))
    fn = np.linalg.norm(fv)
    metrics = {"title": title,
               "rel_l2": float(np.linalg.norm(err) / fn) if fn > 0 else float("nan"),
               "rel_max": float(err.max() / np.abs(fv).max()) if np.abs(fv).max() > 0 else float("nan"),
               "max_abs_error": float(err.max()),
               "rel_l2_displacement": float(np.linalg.norm(ud) / uref) if uref > 0 else float("nan")}
    return a1, fv, rv, err, metrics


def compare_fields(mesh, U_full, U_rom, field="displacement", component=None, scale="auto", cmap="turbo",
                   err_cmap="magma", clim=None, show_edges=False, labels=("full-order", "reduced-order", "error"),
                   title=None, window_size=(1500, 520), screenshot=None, show=True, off_screen=None):
    """Three linked panels: full-order field, reduced-order field, pointwise error. Both solutions are drawn on
    their own deformed mesh (``scale``: a number, ``"auto"`` = 10 % of the model size for the full-order
    displacement, or 0 for the undeformed mesh). Returns ``(plotter, metrics)``."""
    pv = _pv()
    assoc, fv, rv, err, metrics = field_error(mesh, U_full, U_rom, field, component)
    s = (_auto_scale(mesh, U_full) if scale == "auto" else float(scale)) if scale else 0.0
    lim = clim if clim is not None else (float(min(fv.min(), rv.min())), float(max(fv.max(), rv.max())))
    if off_screen is None:
        off_screen = screenshot is not None or bool(pv.OFF_SCREEN)
    pl = pv.Plotter(shape=(1, 3), off_screen=bool(off_screen), window_size=list(window_size))
    name = metrics["title"]
    short = ("full", "reduced", "abs. error")
    panels = [(0, U_full, fv, lim, cmap, labels[0]), (1, U_rom, rv, lim, cmap, labels[1]),
              (2, U_full, err, None, err_cmap, labels[2])]
    for col, U, vals, cl, cm, label in panels:
        pl.subplot(0, col)
        g = to_pyvista(mesh)
        if s:
            g.point_data["_u"] = displacement_vectors(mesh, U)
            g = g.warp_by_vector("_u", factor=s)
        key = f"{short[col]}: {name}"                    # distinct per panel: pyvista keys colorbars by title
        (g.point_data if assoc == "point" else g.cell_data)[key] = vals
        bar = {"title": short[col], "vertical": False, "position_x": 0.12, "position_y": 0.04, "width": 0.76,
               "height": 0.07, "title_font_size": 11, "label_font_size": 9, "fmt": "%.3g"}
        pl.add_mesh(g, scalars=key, cmap=cm, clim=cl, show_edges=show_edges, scalar_bar_args=bar)
        head = f"{label}\n{name}" if col < 2 else (f"{label}: rel L2 {metrics['rel_l2']:.3g}, "
                                                    f"max {metrics['rel_max']:.3g}")
        if col == 1 and title:
            head = f"{title}\n{head}"
        pl.add_text(head, font_size=9)
        if int(mesh.dim) == 2:
            pl.view_xy()
    pl.link_views()
    if screenshot is not None:
        pl.screenshot(str(screenshot))
        if not show:
            pl.close()
            return pl, metrics
    if show and not pl.off_screen:
        pl.show()
    return pl, metrics

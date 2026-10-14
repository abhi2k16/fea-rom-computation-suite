# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
pyvista_animation.py -- animations of time histories, load paths and mode shapes with PyVista (optional dependency,
``pip install fea_engine[viz]``, which includes ``imageio`` for GIF output; MP4 needs ``imageio-ffmpeg`` in addition).

- ``animate_history(mesh, U_hist, path, ...)``  one frame per (selected) row of ``U_hist`` of shape (n_steps, n_dof):
  a transient response (``solve_transient_implicit`` output), a load-step or arc-length path, or a ROM reconstruction.
- ``animate_mode(mesh, mode_shape, path, ...)``  a mode shape oscillating as ``sin(2 pi k / n_frames)`` over
  ``n_cycles`` periods.

Choices that make the animation readable rather than just correct:
- the colour limits are computed once over ALL frames (``clim="global"``, the default), so colours are comparable from
  frame to frame; pass ``clim=(lo, hi)`` to fix them;
- the displacement magnification ``scale="auto"`` is computed once from the largest displacement over all frames, so
  the picture does not rescale between frames;
- the field is evaluated per frame, so any ``field`` accepted by ``plot_field`` works, including a callable such as
  ``lambda U: sys.von_mises(U)`` for stress.

``path`` ending in ``.gif`` writes an animated GIF, ``.mp4`` a video (needs imageio-ffmpeg). The function returns a
dict with ``n_frames``, ``frame_indices``, ``clim``, ``scale`` and ``path``. Frame selection: ``frames=`` (explicit
indices) or ``n_frames=`` (evenly spaced rows) or every row by default.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .pyvista_plot import _pv, _bar, _resolve_field, displacement_vectors, to_pyvista

__all__ = ["animate_history", "animate_mode", "mode_history"]


def _frame_indices(n_steps, frames, n_frames):
    if frames is not None:
        idx = np.asarray(frames, dtype=int)
        if idx.size == 0 or idx.min() < 0 or idx.max() >= n_steps:
            raise ValueError(f"frames must be non-empty indices in [0, {n_steps - 1}]")
        return idx
    if n_frames is None or n_frames >= n_steps:
        return np.arange(n_steps)
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    return np.unique(np.round(np.linspace(0, n_steps - 1, n_frames)).astype(int))


def _prepare(mesh, U_hist, field, component, scale, clim, frames, n_frames):
    """Per-frame scalars, displacement vectors, auto scale and global colour limits (no rendering)."""
    U_hist = np.asarray(U_hist, dtype=float)
    if U_hist.ndim != 2:
        raise ValueError("U_hist must be 2-D: (n_steps, n_dof)")
    idx = _frame_indices(U_hist.shape[0], frames, n_frames)
    grid = to_pyvista(mesh)
    vecs, vals, assoc, title = [], [], None, None
    for i in idx:
        U = U_hist[i]
        vecs.append(displacement_vectors(mesh, U))
        a, v, t = _resolve_field(grid, mesh, field, U, component)
        assoc, title = a, t
        vals.append(np.asarray(v, dtype=float))
    vecs, vals = np.array(vecs), np.array(vals)
    umax = np.abs(vecs).max()
    if scale == "auto":
        pts = np.asarray(mesh.nodes, dtype=float)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        s = 1.0 if umax == 0.0 else 0.1 * diag / umax
    else:
        s = float(scale)
    if isinstance(clim, str):
        if clim != "global":
            raise ValueError("clim must be 'global' or (lo, hi)")
        lim = (float(vals.min()), float(vals.max()))
        if lim[0] == lim[1]:
            lim = (lim[0] - 0.5, lim[1] + 0.5)
    else:
        lim = (float(clim[0]), float(clim[1]))
    return {"idx": idx, "vecs": vecs, "vals": vals, "assoc": assoc, "title": title, "scale": s, "clim": lim}


def animate_history(mesh, U_hist, path, times=None, field="displacement", component=None, scale="auto", fps=15,
                    frames=None, n_frames=None, clim="global", cmap="turbo", show_edges=True, show_undeformed=True,
                    title=None, window_size=(900, 600), loop=0):
    """Animate the rows of ``U_hist`` (n_steps, n_dof) to ``path`` (.gif or .mp4). ``times``: optional (n_steps,)
    time stamps shown in each frame's caption. See the module docstring for the other options."""
    pv = _pv()
    path = str(path)
    prep = _prepare(mesh, U_hist, field, component, scale, clim, frames, n_frames)
    idx, vecs, vals, assoc, s = prep["idx"], prep["vecs"], prep["vals"], prep["assoc"], prep["scale"]
    base = to_pyvista(mesh)
    pts0 = base.points.copy()
    work = base.copy()
    name = prep["title"]
    (work.point_data if assoc == "point" else work.cell_data)[name] = vals[0]
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    work.points = pts0 + s * vecs[0]
    pl.add_mesh(work, scalars=name, cmap=cmap, show_edges=show_edges, clim=prep["clim"], scalar_bar_args=_bar(name))
    if show_undeformed:
        pl.add_mesh(base, style="wireframe", color="gray", opacity=0.4, line_width=1)
    if int(mesh.dim) == 2:
        pl.view_xy()
    caption = pl.add_text("", font_size=10)
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "gif":
        pl.open_gif(path, fps=fps, loop=loop)
    elif ext == "mp4":
        pl.open_movie(path, framerate=fps)
    else:
        raise ValueError("path must end in .gif or .mp4")
    times = None if times is None else np.asarray(times, dtype=float)
    for k, i in enumerate(idx):
        work.points = pts0 + s * vecs[k]
        (work.point_data if assoc == "point" else work.cell_data)[name][:] = vals[k]
        head = title or f"{name}  (deformation x{s:.3g})"
        caption.SetText(2, head + (f"\nt = {times[i]:.4g}" if times is not None else f"\nstep {i}"))
        pl.write_frame()
    pl.close()
    return {"n_frames": int(len(idx)), "frame_indices": idx, "clim": prep["clim"], "scale": s, "path": path}


def mode_history(mode_shape, n_frames=24, n_cycles=1):
    """``(n_frames * n_cycles, n_dof)`` array ``phi * sin(2 pi k / n_frames)``: one full period every ``n_frames``
    rows, last row one step short of a repeat of the first, so the loop is seamless."""
    phi = np.asarray(mode_shape, dtype=float).ravel()
    k = np.arange(int(n_frames) * int(n_cycles))
    return np.sin(2.0 * np.pi * k / n_frames)[:, None] * phi[None, :]


def animate_mode(mesh, mode_shape, path, n_frames=24, n_cycles=1, field="displacement", component=None,
                 scale="auto", fps=15, freq=None, clim="global", cmap="turbo", show_edges=True, show_undeformed=True,
                 window_size=(900, 600), loop=0):
    """Animate a mode shape (n_dof,) oscillating through ``n_cycles`` periods of ``n_frames`` frames each (see
    :func:`mode_history`). ``freq``: natural frequency in Hz, shown in the caption. ``field`` is evaluated on the
    oscillating displacement, so the default colours the displacement magnitude."""
    hist = mode_history(mode_shape, n_frames, n_cycles)
    title = "mode shape" + (f"  f = {freq:.4g} Hz" if freq is not None else "")
    out = animate_history(mesh, hist, path, field=field, component=component, scale=scale, fps=fps, clim=clim,
                          cmap=cmap, show_edges=show_edges, show_undeformed=show_undeformed, title=title,
                          window_size=window_size, loop=loop)
    out["n_frames_per_cycle"] = int(n_frames)
    return out

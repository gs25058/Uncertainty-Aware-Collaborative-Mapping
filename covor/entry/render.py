"""The drawing layer (DESIGN_entry_map.md §5).

    The grid decides. This file only draws what it already decided.

Nothing here reads a log-odds, applies a threshold, or changes a label. Region
fills come straight from entry_grid.npz at one pixel per cell, so a fill can
never disagree with the judgement. The only thing that is vectorised is the WALL
OUTLINE -- marching squares over the occupied mask, then Douglas-Peucker at half
a voxel -- because that is what makes the picture read as a floor plan instead of
a screenshot, and because it is checkable: tests/test_entry_render.py asserts
that the set of cells inside the outline IS the occupied mask.

Everything else is annotation: the unknown hatch, the dotted low-confidence
outline, the routes with their bottleneck width, a scale bar, a legend, and the
conditions the map was made under. A rescuer should not have to ask what they
are looking at, and a reader of the paper should not have to ask which run it is.
"""
import json

import numpy as np

from .config import (UNKNOWN, FREE, OCCUPIED, BLOCKED, NARROW, WALK,
                     CLASS_NAMES)

# DESIGN §5's fills. The WALL gets its own tone, separate from "free but too
# narrow": both are impassable, but one is structure and the other is floor a
# body cannot fit through, and a rescuer reads them differently. Merging them --
# the first version did -- turns the room into one grey mass and throws away the
# most useful thing on the page. Ordered light to dark by how walkable the cell
# is, so the drawing also survives being printed in greyscale.
COL = dict(walk="#ffffff", narrow="#f2bd3b", blocked="#c6c6c6",
           unreached="#dfe7ee", unknown="#ededed", occupied="#6d6d6d",
           hatch="#9a9a9a", wall="#111111",
           route="#c1272d", entry="#1f6fb4", goal="#6a1b9a")
FILL_ORDER = ("walk", "narrow", "blocked", "unknown", "unreached", "occupied")


# --- vector wall outline ---------------------------------------------------
def douglas_peucker(pts, eps):
    """Classic DP simplification of a polyline, in the units of ``pts``."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab)
    if n < 1e-12:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(np.cross(ab, pts - a)) / n
    k = int(np.argmax(d))
    if d[k] <= eps:
        return np.array([a, b])
    left = douglas_peucker(pts[:k + 1], eps)
    right = douglas_peucker(pts[k:], eps)
    return np.vstack([left[:-1], right])


def wall_outline(occ_mask, res, ijk_min, eps_voxels=0.5):
    """Marching-squares contours of the occupied mask, simplified, in metres.

    skimage's find_contours traces the 0.5 level of the mask, i.e. the boundary
    BETWEEN an occupied and a non-occupied cell -- exactly the cell edge. Index
    (r, c) there is offset half a cell from the cell centre convention, which is
    why the +0.5 that appears everywhere else in this package is absent here.
    """
    from skimage import measure
    m = np.asarray(occ_mask, bool)
    if not m.any():
        return []
    # pad so a wall touching the grid edge still closes into a loop
    pm = np.pad(m.astype(float), 1)
    out = []
    for c in measure.find_contours(pm, 0.5):
        c = douglas_peucker(c - 1.0, eps_voxels)
        out.append((c + np.asarray(ijk_min[:2], float) + 0.5) * res)
    return out


def outline_path(contours, res, ijk_min):
    """One matplotlib Path over every contour, so even-odd fills holes right."""
    from matplotlib.path import Path
    verts, codes = [], []
    for c in contours:
        verts.extend(c.tolist() + [c[0].tolist()])
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(c) - 1) + [Path.CLOSEPOLY])
    return Path(np.array(verts), codes) if verts else None


def field_contours(field, level, res, ijk_min, pad_value=-1.0):
    """Iso-contours of a CONTINUOUS field, in metres.

    Used for the width-grade boundaries, which come out of the clearance field
    at exactly the thresholds config.py names. That is the difference between
    this and smoothing: a level set of a continuous field is sub-voxel accurate,
    so the curve is smooth because the quantity is, not because it was filtered.
    The staircase in a binary mask is real information and is left alone (the
    wall outline is still the mask's own contour).

    Padded below the level so a region touching the grid edge still closes.
    """
    from skimage import measure
    f = np.pad(np.asarray(field, float), 1, constant_values=pad_value)
    return [(c - 1.0 + np.asarray(ijk_min[:2], float) + 0.5) * res
            for c in measure.find_contours(f, level)]


def _fill(ax, contours, res, ijk_min, color, z, **kw):
    """Fill every contour as ONE even-odd path, so holes stay holes."""
    from matplotlib.patches import PathPatch
    path = outline_path(contours, res, ijk_min)
    if path is None:
        return None
    pp = PathPatch(path, fc=color, ec=kw.pop("ec", color), lw=kw.pop("lw", 0.35),
                   zorder=z, **kw)
    ax.add_patch(pp)
    return pp


def vector_fills(ax, grid, band, cfg, ext):
    """DESIGN §5's region fills as vectors instead of pixels.

    Painted back to front in NESTED order, which is what keeps the seams shut:

        unknown (the whole window)
          > not-obstacle           binary mask, the same staircase as the wall
            > clearance >= narrow  level set of the clearance field
              > clearance >= walk  level set of the clearance field

    Each region is strictly inside the one before it -- clearance is zero on
    every obstacle cell, so a level set at a positive threshold cannot leave the
    free area -- so nothing has to line up with anything and no hairline gaps
    open between two independently simplified boundaries. That was the reason
    not to vectorise each class on its own.

    Not one label is recomputed: the thresholds are cfg's, the free mask and the
    reachability mask are the grid's.
    """
    from matplotlib.patches import Rectangle
    res = cfg.res
    ijk = np.asarray(grid["ijk_min"])
    lab = np.asarray(grid["%s_label" % band])
    clear = np.asarray(grid["%s_clearance" % band], float)
    reach = np.asarray(grid["%s_reachable" % band], bool)
    free = lab == FREE

    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                           fc=COL["unknown"], ec="none", zorder=0.5))
    # the free area: a mask boundary, simplified exactly like the wall it abuts
    _fill(ax, wall_outline(free, res, ijk, eps_voxels=0.5), res, ijk,
          COL["blocked"], 1.0)
    # the two width grades: level sets of the clearance field
    for lv, col, z in ((cfg.clear_narrow, COL["narrow"], 1.1),
                       (cfg.clear_walk, COL["walk"], 1.2)):
        _fill(ax, field_contours(clear, lv, res, ijk), res, ijk, col, z)
    # free, wide enough, but cut off from the entry -- a binary fact, so a mask
    lost = free & (clear >= cfg.clear_narrow) & ~reach
    if lost.any():
        _fill(ax, wall_outline(lost, res, ijk, eps_voxels=0.5), res, ijk,
              COL["unreached"], 1.3)
    _fill(ax, wall_outline(lab == OCCUPIED, res, ijk, eps_voxels=0.5), res, ijk,
          COL["occupied"], 1.4)


# --- raster layers ---------------------------------------------------------
def class_image(grid, band):
    """Per-cell fill index into FILL_ORDER, straight from the labels.

    No threshold is applied and no value is recomputed: every index here is a
    function of the label, the width class and the reachability mask that
    entry_grid.npz already holds.
    """
    lab = np.asarray(grid["%s_label" % band])
    wc = np.asarray(grid["%s_width_class" % band])
    reach = np.asarray(grid["%s_reachable" % band], bool)
    img = np.full(lab.shape, FILL_ORDER.index("unknown"), np.uint8)
    free = lab == FREE
    img[lab == OCCUPIED] = FILL_ORDER.index("occupied")
    img[free & (wc == BLOCKED)] = FILL_ORDER.index("blocked")
    img[free & (wc == NARROW)] = FILL_ORDER.index("narrow")
    img[free & (wc == WALK)] = FILL_ORDER.index("walk")
    img[free & ~reach & (wc >= NARROW)] = FILL_ORDER.index("unreached")
    return img


def _hatch_mask(shape, period=7, width=2):
    """Diagonal stripes, as a boolean raster.

    A raster hatch rather than a hatched polygon on purpose: the unknown region
    is riddled with holes and multiply connected, and a polygon hatch would need
    every hole traced correctly to avoid painting over the room. A stripe
    texture masked to the region cannot get that wrong.
    """
    i, j = np.indices(shape)
    return ((i + j) % period) < width


def extent_of(grid, res):
    ijk = np.asarray(grid["ijk_min"])
    nx, ny = np.asarray(grid["walk_label"]).shape
    return [ijk[0] * res, (ijk[0] + nx) * res, ijk[1] * res, (ijk[1] + ny) * res]


# --- the figure ------------------------------------------------------------
def draw_band(ax, grid, band, cfg, routes=None, title=None, upsample=6,
              fills="vector"):
    """One band onto one axis. Returns the wall contours it drew.

    fills="vector" draws the regions as polygons (see vector_fills);
    fills="raster" draws them at one pixel per cell. The raster form is the
    unretouched grid and is what scripts/entry/plot_entry_grid.py uses; the
    vector form is for the figures, and neither changes a label.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import PathPatch

    res = float(np.asarray(grid["res"]).ravel()[0])
    ijk = np.asarray(grid["ijk_min"])
    ext = extent_of(grid, res)
    if fills == "vector":
        vector_fills(ax, grid, band, cfg, ext)
    else:
        img = class_image(grid, band)
        cmap = ListedColormap([COL[k] for k in FILL_ORDER])
        ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap,
                  norm=BoundaryNorm(np.arange(-0.5, len(FILL_ORDER) + 0.5),
                                    len(FILL_ORDER)),
                  interpolation="nearest", zorder=1)

    # unknown hatch, drawn at a finer raster than the grid so the stripes are
    # thin lines rather than blocky steps. The MASK is still the grid's.
    lab = np.asarray(grid["%s_label" % band])
    unk = np.kron(lab == UNKNOWN, np.ones((upsample, upsample), bool))
    stripes = _hatch_mask(unk.shape, period=upsample + 1, width=1) & unk
    ax.imshow(np.where(stripes.T, 1.0, np.nan), origin="lower", extent=ext,
              cmap=ListedColormap([COL["hatch"]]), interpolation="nearest", zorder=2)

    # vector walls
    occ = lab == OCCUPIED
    contours = wall_outline(occ, res, ijk, eps_voxels=0.5)
    for c in contours:
        ax.plot(c[:, 0], c[:, 1], "-", color=COL["wall"], lw=1.25, zorder=5,
                solid_joinstyle="round")

    # low-confidence outline: columns the map barely saw
    if bool(np.asarray(grid["has_n_obs"]).ravel()[0]):
        thin = (np.asarray(grid["n_obs"]) < cfg.n_min_obs) & (lab != UNKNOWN)
        if thin.any():
            for c in wall_outline(thin, res, ijk, eps_voxels=0.5):
                ax.plot(c[:, 0], c[:, 1], ":", color="#2b6b2b", lw=1.0, zorder=6)

    # routes
    if routes:
        styles = ["-", (0, (5, 2))]
        for k, r in enumerate(routes.get("routes", [])):
            if not r.get("reachable"):
                continue
            p = (np.asarray(r["cells"]) + ijk[:2] + 0.5) * res
            ax.plot(p[:, 0], p[:, 1], color="white", lw=3.4, zorder=6.5,
                    alpha=.85, solid_capstyle="round")      # halo, for legibility
            ax.plot(p[:, 0], p[:, 1], ls=styles[k % len(styles)],
                    color=COL["route"], lw=1.9, zorder=7, solid_capstyle="round")
            g = np.asarray(r["goal_xy"], float)
            ax.plot(g[0], g[1], "*", ms=15, mfc=COL["goal"], mec="k", mew=.8,
                    zorder=9)
            gdx = -11 if g[0] > 0.5 * (ext[0] + ext[1]) else 11
            ax.annotate("G%d" % (k + 1), xy=g, xytext=(gdx, 7),
                        textcoords="offset points", fontsize=9, weight="bold",
                        color=COL["goal"], zorder=9, clip_on=False,
                        ha="right" if gdx < 0 else "left",
                        bbox=dict(fc="white", ec="none", pad=1.0, alpha=.8))
            # the bottleneck: ringed, and labelled with the gap a body must fit
            m = np.asarray(r["min_width_at"], float)
            ax.plot([m[0]], [m[1]], "o", ms=13, mfc="none", mec="white", mew=3.2,
                    zorder=8)
            ax.plot([m[0]], [m[1]], "o", ms=13, mfc="none", mec=COL["route"],
                    mew=1.8, zorder=8.1)
            # push the label away from whichever edge it is near, so it is never
            # clipped -- these figures are cropped with bbox_inches="tight"
            mid_x = 0.5 * (ext[0] + ext[1])
            mid_y = 0.5 * (ext[2] + ext[3])
            dx = -12 if m[0] > mid_x else 12
            dy = -16 if m[1] > mid_y else 12
            ax.annotate("%.2f m" % r["min_width_m"], xy=m, xytext=(dx, dy),
                        textcoords="offset points", fontsize=8,
                        color=COL["route"], weight="bold", zorder=9,
                        ha="right" if dx < 0 else "left", clip_on=False,
                        bbox=dict(fc="white", ec=COL["route"], lw=.7, pad=1.6,
                                  alpha=.95))
    e = np.asarray(grid["entry_xy"], float)
    ax.plot(e[0], e[1], "o", ms=11, mfc=COL["entry"], mec="k", mew=1.3, zorder=10,
            label="entry")

    rows = np.asarray(grid["%s_rows" % band])
    zlo, zhi = (rows[0] + ijk[2]) * res, (rows[1] + ijk[2]) * res
    ax.set_title(title or ("%s band   z = %.2f - %.2f m" % (band, zlo, zhi)),
                 fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    for sp in ax.spines.values():
        sp.set_linewidth(.6)
    _scale_bar(ax, ext, res)
    return contours


def _scale_bar(ax, ext, res, metres=1.0):
    from matplotlib.patches import Rectangle
    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.05 * (ext[3] - ext[2])
    h = 0.012 * (ext[3] - ext[2])
    for k in range(2):                       # a two-tone bar reads at any size
        ax.add_patch(Rectangle((x0 + k * metres / 2, y0), metres / 2, h,
                               fc="k" if k == 0 else "w", ec="k", lw=.7, zorder=11))
    ax.text(x0 + metres / 2, y0 + 1.6 * h, "%.0f m" % metres, ha="center",
            fontsize=8, zorder=11)


def legend_handles(cfg):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        Patch(fc=COL["walk"], ec="0.35", lw=.5,
              label="walk  (clearance $\\geq$ %.2f m)" % cfg.clear_walk),
        Patch(fc=COL["narrow"], ec="0.35", lw=.5,
              label="narrow  (%.2f - %.2f m, sideways)" % (cfg.clear_narrow,
                                                           cfg.clear_walk)),
        Patch(fc=COL["blocked"], ec="0.35", lw=.5,
              label="free but < %.2f m (body will not fit)" % cfg.clear_narrow),
        Patch(fc=COL["occupied"], ec="0.35", lw=.5, label="obstacle in the band"),
        Patch(fc=COL["unreached"], ec="0.35", lw=.5,
              label="free, wide enough, unreachable"),
        Patch(fc=COL["unknown"], ec="0.35", lw=.5, hatch="///",
              label="unknown  (not observed - treated as obstacle)"),
        Line2D([], [], color=COL["wall"], lw=1.25, label="wall outline (vector)"),
        Line2D([], [], color="#2b6b2b", lw=1.0, ls=":",
               label="observed by < %d frames" % cfg.n_min_obs),
        Line2D([], [], color=COL["route"], lw=2.0, label="route (A*), bottleneck ringed"),
        Line2D([], [], color="none", marker="o", mfc=COL["entry"], mec="k", ms=9,
               label="entry point"),
        Line2D([], [], color="none", marker="*", mfc=COL["goal"], mec="k", ms=13,
               label="goal"),
    ]


def caption(grid, cfg, meta=None):
    """The text DESIGN §5 requires: origin, floor height, generation conditions."""
    res = float(np.asarray(grid["res"]).ravel()[0])
    ijk = np.asarray(grid["ijk_min"])
    floor_z = (int(np.asarray(grid["floor_row"]).ravel()[0]) + ijk[2]) * res
    m = meta or {}
    bits = ["grid %.2f m" % res,
            "floor z = %.2f m" % floor_z,
            "body width w = %.2f m" % cfg.w,
            "k$_\\sigma$ = %g" % cfg.k_sigma,
            "origin = world (0, 0)"]
    if m:
        bits = ["%s / %s / %s" % (m.get("cond", "?"), m.get("coverage", "?"),
                                  m.get("arm", "?"))] + bits
        if m.get("corrupt", "none") not in ("none", ""):
            bits.append("CORRUPTED: %s" % m["corrupt"])
        if m.get("sigma_xy_median") is not None:
            bits.append("$\\sigma_{xy}$ median %.3f m" % m["sigma_xy_median"])
    # two lines: one long caption runs past the axes and gets cropped
    half = (len(bits) + 1) // 2
    return "   |   ".join(bits[:half]) + "\n" + "   |   ".join(bits[half:])


def draw_sigma(ax, grid, cfg, title=None, vmax=None):
    """The per-column sigma_xy field. MEASURED, and not used by any judgement.

    This is the quantity DESIGN §3-3 would turn into a geometric margin
    (r = w/2 + k*sigma_xy). At k_sigma = 0 it enters nothing, so a figure that
    shows it must say so -- which is why the title carries the k it was drawn
    at. It is here because it is the only thing that actually differs between the
    anchor-free conditions, and a comparison figure that shows two identical maps
    without it is a figure that says nothing.
    """
    res = float(np.asarray(grid["res"]).ravel()[0])
    ext = extent_of(grid, res)
    s = np.asarray(grid["sigma_xy"], float)
    m = np.where(s > 0, s, np.nan)
    im = ax.imshow(m.T, origin="lower", extent=ext, cmap="viridis",
                   vmin=0.0, vmax=vmax if vmax else np.nanpercentile(m, 98),
                   interpolation="nearest")
    lab = np.asarray(grid["walk_label"])
    for c in wall_outline(lab == OCCUPIED, res, np.asarray(grid["ijk_min"]),
                          eps_voxels=0.5):
        ax.plot(c[:, 0], c[:, 1], "-", color="w", lw=.55, alpha=.6, zorder=5)
    e = np.asarray(grid["entry_xy"], float)
    ax.plot(e[0], e[1], "o", ms=9, mfc=COL["entry"], mec="w", mew=1.1, zorder=6)
    ax.set_title(title or ("$\\sigma_{xy}$ per column  -  measured, and at "
                           "k$_\\sigma$ = %g it enters no decision" % cfg.k_sigma),
                 fontsize=9.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    return im

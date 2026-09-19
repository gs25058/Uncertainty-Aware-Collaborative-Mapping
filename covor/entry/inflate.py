"""Clearance, width grade and passability (DESIGN §2-2, §3-3, §3-4).

    obstacle  = occupied OR unknown          (DESIGN §3-2: unknown is not free)
    clearance = (EDT(not obstacle) - 0.5) * res
    r         = w/2 + k * sigma_xy
    passable  = clearance >= r

Two separate questions live here and are reported separately. The WIDTH GRADE
(walk / narrow / blocked) is anthropometry and does not move with w. PASSABILITY
is "does a body of width w, positioned to an accuracy of sigma_xy, fit", and is
what the w sweep and the k_sigma comparison act on.

The half-voxel term in the clearance is fixed in config.py, with the table that
shows why: without it the design's own 0.6 m door comes out narrow instead of
blocked.
"""
import numpy as np
from scipy import ndimage

from .config import UNKNOWN, FREE, OCCUPIED, BLOCKED, NARROW, WALK


def obstacle_mask(band):
    """DESIGN §3-2. Unknown is an obstacle -- it is drawn differently and
    penalised differently in routing, but it is never passed through."""
    band = np.asarray(band)
    return (band == OCCUPIED) | (band == UNKNOWN)


def clearance_map(band, cfg, return_indices=False):
    """Distance from each cell centre to the nearest obstacle SURFACE, in metres.

    Obstacle cells get 0. See config.py's CLEARANCE CONVENTION block for why the
    half voxel is subtracted and what it decides.
    """
    obst = obstacle_mask(band)
    if return_indices:
        d, idx = ndimage.distance_transform_edt(~obst, return_indices=True)
    else:
        d = ndimage.distance_transform_edt(~obst)
    clear = np.maximum(d - 0.5, 0.0) * cfg.res
    return (clear, idx) if return_indices else clear


def width_class(clear, cfg):
    """walk / narrow / blocked from clearance alone (DESIGN §4)."""
    clear = np.asarray(clear, float)
    wc = np.full(clear.shape, BLOCKED, np.uint8)
    wc[clear >= cfg.clear_narrow] = NARROW
    wc[clear >= cfg.clear_walk] = WALK
    return wc


def radius_map(shape, cfg, sigma_xy=None, k_sigma=None):
    """r = w/2 + k * sigma_xy, per cell (DESIGN §3-3).

    sigma_xy=None or k=0 gives a constant w/2 everywhere -- the Part 2 setting,
    and the control arm of PREREG_entry_sigma.md. The sigma field itself is built
    in sigma_map.py; this function only combines it, so the combination is
    testable without a map.
    """
    k = cfg.k_sigma if k_sigma is None else float(k_sigma)
    r = np.full(tuple(shape), cfg.half_width(), float)
    if sigma_xy is not None and k != 0.0:
        r = r + k * np.asarray(sigma_xy, float)
    return r


def passable_mask(band, clear, r):
    """clearance >= r, and only where the band itself is free.

    The free test is redundant as long as r > 0: occupied and unknown cells are
    obstacles, so their clearance is 0 and they fail the distance test already.
    It is written out anyway so that passability does not depend on that
    coincidence -- a cell is passable because the band says it is free AND it is
    wide enough, not because a distance field happened to come out small.
    """
    return (np.asarray(band) == FREE) & (np.asarray(clear) >= np.asarray(r))


def body_disk(res, radius):
    """A disk structuring element of ``radius`` metres on a ``res`` grid."""
    rc = int(np.ceil(radius / res))
    yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1]
    return (np.hypot(xx, yy) * res) <= radius + 1e-9


def swept_workspace(mask, cfg, radius=None):
    """The floor a body actually covers if its CENTRE may be anywhere in ``mask``.

    Ported from uacm/entry_map (the pipeline supplied in uacm.zip), which is
    where the distinction was first drawn, and it is a real one:

      * ``passable`` / ``reachable`` are CONFIGURATION SPACE -- the set of places
        the body's centre may legally be. That is the right space to plan in.
      * what a person looking at a floor plan means by "walkable floor" is
        WORKSPACE -- the floor the body sweeps out, which is the centre set
        dilated back by the body radius.

    Painting only the centre set makes an open room look half blocked, because
    every wall is surrounded by a band of floor no centre may occupy but a
    shoulder happily passes over. uacm measured 41.7 of 58.3 m^2 of apparent
    "missed" floor being exactly that band.

    This function does NOT enter any safety decision. Passability and
    reachability stay in configuration space, which is the conservative space;
    the swept set is reported and drawn beside them, never instead of them.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros_like(mask)
    r = cfg.half_width() if radius is None else float(radius)
    return ndimage.binary_dilation(mask, structure=body_disk(cfg.res, r))

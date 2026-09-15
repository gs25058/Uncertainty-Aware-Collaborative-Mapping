"""Reachability from the entry point (DESIGN_entry_map.md §2-3, §8).

Only the connected component of PASSABLE cells that contains the entry point is
reachable. Everything else is free space a rescuer cannot get to from the door,
which the map has to say out loud: an isolated clear pocket behind an unobserved
band is exactly the thing that makes a map dangerous rather than useless.

4-connectivity, not 8: a diagonal step between two cells whose shared corner is
pinched by obstacles is not a step a body takes, and passability was computed
per cell, not per move.
"""
import numpy as np
from scipy import ndimage

from .config import UNKNOWN, FREE, OCCUPIED

_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def world_to_cell(xy, ijk_min, res):
    """(x, y) in metres -> (i, j) array index, the mapper's floor() convention."""
    v = np.floor(np.asarray(xy, float)[:2] / res).astype(np.int64)
    return int(v[0] - ijk_min[0]), int(v[1] - ijk_min[1])


def cell_to_world(ij, ijk_min, res):
    """(i, j) array index -> the cell centre in metres."""
    return ((np.asarray(ij, float) + np.asarray(ijk_min[:2], float) + 0.5) * res)


def default_entry(passable):
    """Smallest-x passable cell OF THE LARGEST passable component (DESIGN §8
    leaves the entry point to the user; this is the stand-in when none is given).

    Returned as an (i, j) array index; the i axis is monotone in x, so smallest
    index is smallest x. Ties break on smallest y, so the result is a function of
    the grid alone.

    room909's scan has no door to read a GT entry point from -- it is open around
    x < -4 m rather than doored (covor/synth/mesh_gt.build_gt_grid) -- and no
    drone start position is passable in the walk band at w = 0.70. Smallest x is
    the reproducible stand-in for "nearest the real opening".

    The component restriction is not decoration. MEASURED on room909's GT walk
    band with closing off: the plain smallest-x passable cell is an isolated
    one-cell island at (-2.95, -2.75) m, and the map that comes out reports 1
    reachable cell out of 756 passable. A default that can silently answer "the
    building is one cell" is not a default -- and a one-cell island is not a
    door, which is why this is the right rule rather than a patch over a symptom.
    """
    passable = np.asarray(passable, bool)
    if not passable.any():
        raise ValueError("no passable cell: there is no entry point to choose")
    cc, _ = ndimage.label(passable, structure=_CONN4)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    idx = np.argwhere(cc == int(sizes.argmax()))
    order = np.lexsort((idx[:, 1], idx[:, 0]))
    return tuple(int(v) for v in idx[order[0]])


def reachable(passable, entry_ij, band=None):
    """Boolean mask of the passable component containing ``entry_ij``.

    Raises if the entry point is off the grid, or lands on a cell that is not
    passable. The message separates the three reasons, because they call for
    different actions: an unknown entry means go and observe it, an occupied
    entry means the door is blocked or the coordinate is wrong, and a free but
    too-narrow entry means the opening is real but not wide enough for w.
    """
    passable = np.asarray(passable, bool)
    i, j = int(entry_ij[0]), int(entry_ij[1])
    if not (0 <= i < passable.shape[0] and 0 <= j < passable.shape[1]):
        raise ValueError("entry point (%d, %d) is outside the %dx%d grid"
                         % (i, j, passable.shape[0], passable.shape[1]))
    if not passable[i, j]:
        why = "not passable"
        if band is not None:
            v = int(np.asarray(band)[i, j])
            why = {UNKNOWN: "unobserved (unknown)", OCCUPIED: "occupied",
                   FREE: "free but narrower than the required clearance"}.get(v, why)
        raise ValueError("entry point (%d, %d) is %s" % (i, j, why))
    cc, _ = ndimage.label(passable, structure=_CONN4)
    return cc == cc[i, j]

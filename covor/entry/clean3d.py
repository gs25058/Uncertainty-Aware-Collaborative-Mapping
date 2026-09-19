"""3D cleanup before the band projection (DESIGN_entry_map.md §2-0, §3-5, §3-6).

Three operations, all of them directional: floor/ceiling estimation, demoting
floating occupied blobs to UNKNOWN, and closing pinholes in walls. The
directionality is the point -- DESIGN §3-5:

    closing only ever grows the occupied set, floating blobs go to UNKNOWN and
    never to FREE, and nothing anywhere in this file grows the free set.

That is the 3D half of the rule RESULTS_SUMMARY §4.8 states for the map: a cell
we are not sure about must not end up called free, because in entry guidance
"unknown" and "clear" are different sentences.

The label space is covor.synth.mesh_gt's: UNKNOWN / FREE / OCC = 0 / 1 / 2, for
GT grids and for maps alike. ``labels_from_masks`` is the adapter that puts a
built map into it.
"""
import numpy as np
from scipy import ndimage

from .config import UNKNOWN, FREE, OCCUPIED


def labels_from_masks(M_occ, M_free):
    """Dense 3D labels from a built map's (occupied, free) boolean grids.

    M_occ / M_free come from covor.synth.metrics.map_masks, i.e. from
    OccupancyBuilder.classify_points: occupied is l > tau_occ, free is
    l < -tau_free, and EVERYTHING ELSE -- never observed, or observed but
    undecided -- is unknown. Cells the two masks both claim would be a builder
    bug, so this refuses rather than picking a winner.
    """
    M_occ = np.asarray(M_occ, bool)
    M_free = np.asarray(M_free, bool)
    if M_occ.shape != M_free.shape:
        raise ValueError("mask shapes differ: %s vs %s" % (M_occ.shape, M_free.shape))
    both = M_occ & M_free
    if both.any():
        raise ValueError("%d cells are both occupied and free -- classify_points "
                         "cannot produce that" % int(both.sum()))
    lab = np.full(M_occ.shape, UNKNOWN, np.uint8)
    lab[M_free] = FREE
    lab[M_occ] = OCCUPIED
    return lab


def occupied_z_histogram(lab):
    """Occupied cells per z row, as an array indexed like lab's third axis."""
    return (np.asarray(lab) == OCCUPIED).sum(axis=(0, 1))


def floor_ceiling_rows(lab):
    """(floor_row, ceiling_row, info) as ARRAY indices into lab's z axis.

    DESIGN §3-6: the floor is the lowest peak of the occupied z histogram. It has
    to be a PEAK and not simply the lowest occupied row -- a scan carries stray
    geometry below the floor plane (room909 has 93-893 cells in the four rows
    under it) and an argmin-style rule would sit on that. A strict local maximum
    is enough to exclude those: below the floor the histogram climbs
    monotonically into it.

    NO HEIGHT THRESHOLD. An earlier version also required a peak to hold 25 % of
    the tallest row, to reject noise. On room909 that constant landed on top of
    the floor peak itself: the C map's floor row held 1370 cells against a
    threshold of 1336 and passed by 34, the A map's held 1210 against 1360 and
    FAILED -- so the only surviving peak was the ceiling, the ceiling became the
    floor, the walk band was placed above the roof, and the map came out with
    zero passable cells. It raised nothing; it just answered wrongly. A constant
    that decides between "floor" and "ceiling" by 150 cells is not measuring
    anything, so it is gone.

    ``peaks`` is returned in full because the lowest peak is a candidate, not a
    verdict: build_entry_map validates it against the band above it and moves up
    the list if that band holds no free space. ``n_peaks`` also says whether
    there are several storeys, which DESIGN §3-6 sends round §2 once per storey
    -- not implemented, so it must at least be visible.
    """
    h = occupied_z_histogram(lab).astype(np.int64)
    if h.max() == 0:
        raise ValueError("no occupied cells: cannot estimate a floor")
    peaks = [k for k in range(len(h))
             if h[k] > 0
             and h[k] >= (h[k - 1] if k > 0 else -1)
             and h[k] > (h[k + 1] if k + 1 < len(h) else -1)]
    if not peaks:
        raise ValueError("the occupied z histogram has no local maximum")
    info = dict(n_peaks=len(peaks), peaks=[int(p) for p in peaks],
                peak_counts=[int(h[p]) for p in peaks], hist=h.tolist())
    return int(peaks[0]), int(peaks[-1]), info


def demote_floating(lab, min_voxels, connectivity=3):
    """Occupied components of fewer than ``min_voxels`` cells -> UNKNOWN.

    DESIGN §3-5: to UNKNOWN, never to FREE. A stereo mismatch that produced a
    floating cube is evidence we no longer trust, which is not the same as
    evidence that the space is clear.

    min_voxels <= 1 is an exact no-op and returns the input unchanged.
    """
    lab = np.asarray(lab, np.uint8)
    if min_voxels <= 1:
        return lab.copy(), dict(n_components=0, n_demoted_components=0,
                                n_demoted_cells=0)
    occ = lab == OCCUPIED
    st = ndimage.generate_binary_structure(3, connectivity)
    cc, n = ndimage.label(occ, structure=st)
    if n == 0:
        return lab.copy(), dict(n_components=0, n_demoted_components=0,
                                n_demoted_cells=0)
    sizes = np.bincount(cc.ravel())
    small = np.zeros(len(sizes), bool)
    small[1:] = sizes[1:] < min_voxels
    drop = small[cc]
    out = lab.copy()
    out[drop] = UNKNOWN
    return out, dict(n_components=int(n),
                     n_demoted_components=int(small[1:].sum()),
                     n_demoted_cells=int(drop.sum()))


def _close_structure(kind):
    """The structuring element for close_occupied.

    "3d" is a 6-neighbourhood; "2d" is a 4-neighbourhood inside each z slice,
    expressed as a (3, 3, 1) element so one binary_closing call does every slice.
    See EntryCfg.close_structure for the measured difference between them.
    """
    if kind == "3d":
        return ndimage.generate_binary_structure(3, 1)
    if kind == "2d":
        st = np.zeros((3, 3, 1), bool)
        st[1, :, 0] = True
        st[:, 1, 0] = True
        return st
    raise ValueError("close_structure must be '3d' or '2d', got %r" % (kind,))


def close_occupied(lab, iters, structure="3d"):
    """Binary closing of the OCCUPIED set; the cells it gains become OCCUPIED.

    Closing with a structuring element containing the origin is extensive, so the
    occupied set can only grow, and the cells it takes come out of FREE or
    UNKNOWN. That is exactly DESIGN §3-5's "occupied direction only", and
    tests/test_entry_map.py asserts it cell by cell rather than by count.

    What closing also does, by construction, is fill concave corners narrower
    than the element -- including the floor-wall junction under "3d". That is not
    a bug here; it is why EntryCfg.close_structure exists and is reported.
    """
    lab = np.asarray(lab, np.uint8)
    if iters <= 0:
        return lab.copy(), dict(n_added=0, n_added_from_free=0)
    occ = lab == OCCUPIED
    st = _close_structure(structure)
    closed = ndimage.binary_closing(occ, structure=st, iterations=int(iters))
    gained = closed & ~occ
    out = lab.copy()
    out[gained] = OCCUPIED
    return out, dict(n_added=int(gained.sum()),
                     n_added_from_free=int((gained & (lab == FREE)).sum()))


def clean(lab, cfg):
    """Run §2-0 in the order the design gives: floating blobs, then closing.

    Returns (labels, info). The floor/ceiling estimate is taken on the CLEANED
    grid, so a demoted blob cannot invent a storey.
    """
    lab1, i_cube = demote_floating(lab, cfg.cube_min_voxels, cfg.cube_connectivity)
    lab2, i_close = close_occupied(lab1, cfg.close_iter, cfg.close_structure)
    floor_row, ceil_row, i_fc = floor_ceiling_rows(lab2)
    info = dict(floating=i_cube, closing=i_close, floor=i_fc,
                floor_row=floor_row, ceiling_row=ceil_row)
    return lab2, info

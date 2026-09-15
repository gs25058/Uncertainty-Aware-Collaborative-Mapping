"""Band projection: 3D labels -> one 2D label per column (DESIGN §2-1, §3-1).

    occupied  <- ANY occupied cell in the band
    free      <- EVERY cell in the band is free     (strict)
    unknown   <- everything else

The asymmetry is the 2D restatement of RESULTS_SUMMARY §4.8. A single slice at
head height calls the space under a desk free and misses a knee-high obstacle
entirely; a band asks about the volume a body actually sweeps. Making FREE
require the whole band, while OCCUPIED needs one cell, is what stops an
unobserved slab in the middle of the band from reading as clear.

BAND ARITHMETIC. Rows are computed in integers from the floor row, never by
dividing metres. int(1.9 / 0.10) is 18, not 19, because 1.9/0.1 == 18.999999...;
that silently drops the top 10 cm of the walk band. round() on the ratio is used
once, at the point where a design constant in metres becomes a row count.
"""
import numpy as np

from .config import UNKNOWN, FREE, OCCUPIED


def rows_for_band(floor_row, height, cfg, nz):
    """[lo, hi) array rows covering z in [floor + z_min, floor + height).

    ``floor_row`` is an array index into the z axis; the band is expressed
    relative to the floor cell's LOWER face, which is the plane the design means
    by "floor". Clipped to the grid, and empty bands are refused rather than
    quietly returning nothing.
    """
    res = cfg.res
    lo = floor_row + int(round(cfg.z_min / res))
    hi = floor_row + int(round(height / res))
    lo_c, hi_c = max(lo, 0), min(hi, int(nz))
    if hi_c <= lo_c:
        raise ValueError("band z in [%.2f, %.2f) from floor row %d is empty in a "
                         "grid of %d rows" % (cfg.z_min, height, floor_row, nz))
    return lo_c, hi_c


def project_band(lab, rows):
    """(nx, ny) uint8 band label from (nx, ny, nz) 3D labels and a row range."""
    lo, hi = rows
    s = np.asarray(lab)[:, :, lo:hi]
    occ = (s == OCCUPIED).any(axis=2)
    fre = (s == FREE).all(axis=2)
    out = np.full(s.shape[:2], UNKNOWN, np.uint8)
    out[fre] = FREE
    out[occ] = OCCUPIED           # occupied wins: one cell is enough
    return out


def project(lab, floor_row, cfg):
    """Both bands at once: {"walk": 2D, "crawl": 2D}, plus the rows used."""
    nz = np.asarray(lab).shape[2]
    rows = {"walk": rows_for_band(floor_row, cfg.H_walk, cfg, nz),
            "crawl": rows_for_band(floor_row, cfg.H_crawl, cfg, nz)}
    return {k: project_band(lab, v) for k, v in rows.items()}, rows

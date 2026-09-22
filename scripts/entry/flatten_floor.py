#!/usr/bin/env python3
"""Flatten a scanned floor so the entry pipeline's one-floor assumption holds.

    python scripts/entry/flatten_floor.py --in results/synth_corridor915/gt_voxel.npz \
        --out results/synth_corridor915/gt_voxel_flat.npz
    python scripts/entry/flatten_floor.py --in <map3d_*.npz> --out <map3d_*_flat.npz>

WHY. DESIGN §3-6 defines the body band as z in [floor + 0.10, floor + H] with
ONE floor row per storey -- "the lowest peak of the occupied z histogram". The
2026-09-15 corridor's floor is not one row: the per-slab peak sits at -0.10 m
over most of it, 0.00 at y in [-15, -10] and -0.20 in the side-room stretch. A
single row then puts the band's bottom inside the floor slab wherever the floor
is high, and 45 % of the GT's walk-band occupied columns were occupied ONLY in
that bottom row (RESULTS_sgbm.md §3). That is the floor read as furniture.

WHAT. Estimate a per-column floor row from the grid itself, then shift every
column vertically by an integer number of rows so its floor lands on one
reference row. The band the unmodified covor/entry/ pipeline then cuts is
"0.10-1.90 m above the LOCAL floor", which is what §3-6 means. covor/entry/ is
not touched; this is a change of input, and it is applied to the GT and to the
map by the same function -- each estimating its OWN floor, as a deployed map
would have to. A map that gets its floor wrong pays for it in the metrics.

HOW the floor is found. To a walker the floor is the surface UNDER the free
volume, so per column the floor candidate is (lowest FREE row in a window
around the global floor peak) - 1. On the GT that is the floor slab by
definition -- the flood fill stops on it. On a map it holds even where the
floor voxel itself was never returned: a camera at 1.2 m looking ahead sees
the free air over the floor along a wall far more often than the floor cell
under it (measured on the corridor: in 2,081 wall-side columns the map held 42
occupied floor cells against 695 free cells one row up, and a rule built on
occupied cells alone put the floor 2-4 rows too high there). The grid is tiled
(tile_m square) and each tile takes the MODE over its columns. A tile with no
free evidence falls back to the mode of occupied candidate rows; a tile with
neither takes the nearest evidenced tile -- such columns are unknown in the
band anyway, so the choice cannot create passable floor.

The shift is integer and per column, so nothing is interpolated and every
voxel keeps its label. Rows pushed off the top or bottom of the grid are lost
(at most `window` rows at the extremes); cells shifted in from outside are
UNKNOWN / False / NaN.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.entry.config import UNKNOWN, FREE, OCCUPIED   # label space shared with mesh_gt


def floor_height_map(lab, tile_cells, window=3, min_cells=10, min_cols=5):
    """(floor_row per column (nx,ny) int, reference row, info).

    Rows are ARRAY indices along z. The reference row is the global lowest
    strict local maximum, i.e. exactly what clean3d.floor_ceiling_rows would
    pick, so a grid whose floor is already flat gets shift 0 everywhere and comes
    back bit-identical (tests/test_entry_flatten.py pins that).
    """
    lab = np.asarray(lab)
    nx, ny, nz = lab.shape
    occ = lab == OCCUPIED
    h = occ.sum(axis=(0, 1))
    if h.max() == 0:
        raise ValueError("no occupied cells: no floor to flatten")
    peaks = [k for k in range(nz) if h[k] > 0
             and h[k] >= (h[k - 1] if k > 0 else -1)
             and h[k] > (h[k + 1] if k + 1 < nz else -1)]
    ref = int(peaks[0])
    lo, hi = max(ref - window, 0), min(ref + window + 1, nz)
    # per-column candidate from the free volume: the row under its lowest free
    # cell inside [lo, hi+1). -1 where the column has no free cell there.
    fr = lab[:, :, lo:hi + 1] == FREE
    has = fr.any(axis=2)
    low_free = np.where(has, fr.argmax(axis=2) + lo, -1)
    cand = np.where(has & (low_free - 1 >= 0), low_free - 1, -1)
    t = max(1, int(tile_cells))
    tx, ty = -(-nx // t), -(-ny // t)
    fmap = np.full((tx, ty), -1, np.int64)
    src = np.full((tx, ty), "", object)
    for i in range(tx):
        for j in range(ty):
            c = cand[i * t:(i + 1) * t, j * t:(j + 1) * t]
            c = c[c >= 0]
            if c.size >= min_cols:
                vals, cnts = np.unique(c, return_counts=True)
                fmap[i, j] = int(vals[int(np.argmax(cnts))]); src[i, j] = "free"
                continue
            blk = occ[i * t:(i + 1) * t, j * t:(j + 1) * t, lo:hi]
            cnt = blk.sum(axis=(0, 1))
            if cnt.sum() >= min_cells:
                fmap[i, j] = lo + int(np.argmax(cnt)); src[i, j] = "occ"
    have = fmap >= 0
    n_free_tiles, n_occ_tiles = int((src == "free").sum()), int((src == "occ").sum())
    if not have.any():
        raise ValueError("no tile has %d floor cells within %d rows of row %d"
                         % (min_cells, window, ref))
    if not have.all():
        # nearest evidenced tile, by index distance
        _, (ii, jj) = ndimage.distance_transform_edt(~have, return_indices=True)
        fmap = fmap[ii, jj]
    per_col = np.repeat(np.repeat(fmap, t, axis=0), t, axis=1)[:nx, :ny]
    info = dict(reference_row=ref, window_rows=int(window), tile_cells=t,
                min_cells=int(min_cells), min_cols=int(min_cols),
                n_tiles=int(tx * ty), n_tiles_with_evidence=int(have.sum()),
                n_tiles_from_free=n_free_tiles, n_tiles_from_occupied=n_occ_tiles,
                floor_rows_present={int(k): int(v) for k, v in
                                    zip(*np.unique(per_col, return_counts=True))},
                global_hist_peaks=[int(p) for p in peaks])
    return per_col, ref, info


def shift_columns(A, shift, fill):
    """Move column (i, j) up by shift[i, j] rows (down if negative). Exact."""
    A = np.asarray(A)
    out = np.full(A.shape, fill, A.dtype)
    nz = A.shape[2]
    for s in np.unique(shift):
        m = shift == s
        s = int(s)
        if s == 0:
            out[m] = A[m]
        elif s > 0:
            out[m, s:] = A[m, :nz - s]
        else:
            out[m, :nz + s] = A[m, -s:]
    return out


def flatten_npz(path_in, path_out, tile_m=1.0, window=3, min_cells=10, min_cols=5):
    z = np.load(path_in, allow_pickle=False)
    res = float(np.asarray(z["res"]).ravel()[0])
    lab = z["labels"]
    fcol, ref, info = floor_height_map(lab, int(round(tile_m / res)), window,
                                       min_cells, min_cols)
    shift = ref - fcol                              # up if the local floor is low
    out = {k: z[k] for k in z.files}
    out["labels"] = shift_columns(lab, shift, UNKNOWN)
    for k, fill in (("M_occ", False), ("M_free", False)):
        if k in z.files:
            out[k] = shift_columns(z[k].astype(bool), shift, fill)
    for k, fill in (("logodds", np.nan), ("enclosure", -1.0)):
        if k in z.files:
            out[k] = shift_columns(z[k], shift, fill).astype(z[k].dtype)
    out["floor_row_map"] = fcol.astype(np.int32)     # before flattening
    out["flatten_shift"] = shift.astype(np.int32)
    info.update(tile_m=tile_m, source=os.path.basename(path_in),
                shift_hist={int(k): int(v) for k, v in
                            zip(*np.unique(shift, return_counts=True))})
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    meta["flatten"] = info
    out["meta"] = json.dumps(meta)
    if "info" in z.files:                             # gt_voxel.npz carries its own
        gi = json.loads(str(z["info"])); gi["flatten"] = info
        out["info"] = json.dumps(gi)
    np.savez_compressed(path_out, **out)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-m", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=3,
                    help="rows either side of the global floor peak that count "
                         "as floor candidates (3 = +-0.30 m at 0.10 m)")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="occupied cells a tile needs for the fallback rule")
    ap.add_argument("--min-cols", type=int, default=5,
                    help="columns with a free-volume bottom a tile needs")
    a = ap.parse_args()
    info = flatten_npz(a.inp, a.out, a.tile_m, a.window, a.min_cells, a.min_cols)
    print("floor reference row %d | tiles %d: from free %d, from occupied %d, filled %d "
          "| floor rows found %s"
          % (info["reference_row"], info["n_tiles"], info["n_tiles_from_free"],
             info["n_tiles_from_occupied"],
             info["n_tiles"] - info["n_tiles_with_evidence"], info["floor_rows_present"]))
    print("column shifts %s  -> %s" % (info["shift_hist"], a.out))


if __name__ == "__main__":
    main()

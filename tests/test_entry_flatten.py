#!/usr/bin/env python3
"""scripts/entry/flatten_floor.py: exact, label-preserving, and a no-op on a
flat floor. Every assertion is by value.

Run: python tests/test_entry_flatten.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
from covor.entry.config import UNKNOWN, FREE, OCCUPIED               # noqa: E402
import flatten_floor as FF                                          # noqa: E402


def _scene(step_at=None, step_rows=1, nx=40, ny=60, nz=30, floor=4):
    lab = np.zeros((nx, ny, nz), np.uint8)
    for j in range(ny):
        f = floor + (step_rows if (step_at is not None and j >= step_at) else 0)
        lab[:, j, f] = OCCUPIED                     # floor slab
        lab[:, j, f + 1:f + 20] = FREE              # room above it
        lab[0, j, f + 1:f + 20] = OCCUPIED          # a wall
        lab[-1, j, f + 1:f + 20] = OCCUPIED
    return lab


def test_flat_floor_is_a_bit_identical_no_op():
    lab = _scene()
    fcol, ref, info = FF.floor_height_map(lab, tile_cells=10)
    assert ref == 4 and (fcol == 4).all()
    out = FF.shift_columns(lab, ref - fcol, UNKNOWN)
    assert np.array_equal(out, lab)


def test_a_one_row_step_is_undone_exactly():
    lab = _scene(step_at=30, step_rows=1)
    lab[20, 45, 4 + 1 + 7] = OCCUPIED               # a marker 0.7 m over the HIGH floor
    lab[20, 10, 4 + 7] = OCCUPIED                   # and one 0.7 m over the LOW floor
    fcol, ref, info = FF.floor_height_map(lab, tile_cells=10)
    # the two halves have equal floor area, so the global reference may be
    # either row -- it only has to be ONE row. The local estimate must be exact.
    assert ref in (4, 5), ref
    assert (fcol[:, :30] == 4).all() and (fcol[:, 30:] == 5).all(), np.unique(fcol)
    shift = ref - fcol
    out = FF.shift_columns(lab, shift, UNKNOWN)
    # the whole floor now sits on the reference row, both markers 7 rows above it
    assert (out[:, :, ref] == OCCUPIED).all()
    assert (out[1:-1, :, ref + 1] == FREE).all()
    assert out[20, 45, ref + 7] == OCCUPIED and out[20, 10, ref + 7] == OCCUPIED
    # rows that slid in from outside the grid are UNKNOWN, on exactly the
    # columns that moved, and nothing else changed label
    moved_up = shift > 0
    if moved_up.any():
        assert (out[moved_up][:, -1] == UNKNOWN).all()
    moved_dn = shift < 0
    if moved_dn.any():
        assert (out[moved_dn][:, 0] == UNKNOWN).all()
    n_lab = {v: int((lab == v).sum()) for v in (FREE, OCCUPIED)}
    n_out = {v: int((out == v).sum()) for v in (FREE, OCCUPIED)}
    assert n_out == n_lab, (n_lab, n_out)          # nothing created or destroyed


def test_unobserved_tiles_borrow_the_nearest_floor():
    lab = _scene(step_at=30, step_rows=1)
    lab[:, 40:50, :] = UNKNOWN                      # a stretch nobody scanned
    fcol, ref, info = FF.floor_height_map(lab, tile_cells=10)
    assert (fcol[:, 40:50] == 5).all(), "must inherit the high floor next door"
    assert info["n_tiles_with_evidence"] == info["n_tiles"] - 4


def test_corridor_gt_spill_drops_after_flattening():
    """On the real scan the floor-in-band columns must fall, by value."""
    gt = os.path.join(ROOT, "results", "synth_corridor915", "gt_voxel.npz")
    if not os.path.exists(gt):
        return print("SKIP: corridor915 GT not built")
    from covor.entry.config import EntryCfg
    from build_entry_map import build_entry_grid
    z = np.load(gt, allow_pickle=False)
    lab, ijk = z["labels"], z["ijk_min"]
    res = float(np.asarray(z["res"]).ravel()[0])
    fcol, ref, info = FF.floor_height_map(lab, int(round(1.0 / res)))
    flat = FF.shift_columns(lab, ref - fcol, UNKNOWN)

    def spill(l):
        g, gi = build_entry_grid(l, EntryCfg(res=res), ijk_min=ijk, k_sigma=0.0)
        lo, hi = gi["rows"]["walk"]
        s = l[:, :, lo:hi] == OCCUPIED
        col = s.any(2)
        return int((s[:, :, 0] & ~s[:, :, 1:].any(2)).sum()), int(col.sum())
    b_only, b_all = spill(lab)
    a_only, a_all = spill(flat)
    print("      bottom-row-only occupied columns: %d/%d -> %d/%d" % (b_only, b_all, a_only, a_all))
    assert a_only < b_only, "flattening did not reduce floor spill"


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

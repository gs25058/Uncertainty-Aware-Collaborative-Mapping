#!/usr/bin/env python3
"""covor/entry/traverse.py pinned by value on grids with a known answer.

Each scene is a 0.10 m label grid: FREE air, OCCUPIED floor/walls/ceiling,
UNKNOWN outside. The entry is at the west end; the question is always which
cells are reached and in which posture.

Run: python tests/test_entry_traverse.py   (or: pytest tests/test_entry_traverse.py)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from covor.entry.config import UNKNOWN, FREE, OCCUPIED            # noqa: E402
from covor.entry import traverse as TV                            # noqa: E402

CFG = TV.TravCfg()
FLOOR = 2          # floor voxel row
CEIL = 28          # ceiling voxel row (2.5 m above the floor)


def corridor(nx=40, ny=11, nz=32):
    """x-long corridor, 0.9 m of free width (y cells 1..9), walls at y 0 and 10."""
    lab = np.full((nx, ny, nz), UNKNOWN, np.uint8)
    lab[:, :, FLOOR] = OCCUPIED
    lab[:, :, CEIL] = OCCUPIED
    lab[:, :, FLOOR + 1:CEIL] = FREE
    lab[:, 0, FLOOR:CEIL + 1] = OCCUPIED
    lab[:, -1, FLOOR:CEIL + 1] = OCCUPIED
    lab[0, :, FLOOR:CEIL + 1] = OCCUPIED
    lab[-1, :, FLOOR:CEIL + 1] = OCCUPIED
    return lab


def run(lab, entry=(0.45, 0.55)):
    return TV.traverse(lab, CFG, entry, (0, 0, 0), snap_m=1.0)


def mid(t, i):
    return t["reach"][i, 5], t["posture"][i, 5], t["width"][i, 5]


def test_flat_corridor_is_walked_standing_end_to_end():
    t = run(corridor())
    assert t["entry"] is not None
    assert mid(t, 35) == (True, TV.STAND, TV.NORMAL)
    assert (t["posture"][t["reach"]] == TV.STAND).all()


def test_floor_that_wanders_one_row_is_still_one_floor():
    lab = corridor()
    rng = np.random.default_rng(0)
    up = rng.random(lab.shape[:2]) < 0.4          # 40 % of columns one row higher
    up[:, 0] = up[:, -1] = False
    ii, jj = np.nonzero(up)
    lab[ii, jj, FLOOR + 1] = OCCUPIED
    t = run(lab)
    assert mid(t, 35)[0], "a +-1 row floor must not cut the corridor"


def stairs(riser_rows, n_steps=5, tread=3):
    lab = corridor(nx=60, nz=45)
    lab[:, :, 28] = FREE                          # lift the ceiling out of the way
    lab[:, :, 42] = OCCUPIED
    lab[:, 1:-1, 29:42] = FREE
    x = 10
    h = FLOOR
    for s in range(n_steps):
        h += riser_rows
        lab[x:, 1:-1, FLOOR + 1:h + 1] = OCCUPIED  # solid stair block, top at h
        x += tread
    return lab, h


def test_stairs_with_code_risers_are_climbed():
    lab, top = stairs(riser_rows=2)                # 0.20 m nominal risers
    t = run(lab)
    assert mid(t, 50)[0], "the landing above a 2-row stair must be reached"
    assert t["support_row"][50, 5] == top


def test_a_three_row_ledge_is_not_a_step():
    lab, _ = stairs(riser_rows=3, n_steps=1)
    t = run(lab)
    assert not mid(t, 30)[0], "a 0.3 m ledge must not be walked up"
    assert mid(t, 5)[0]


def test_low_beam_is_ducked_under():
    lab = corridor()
    lab[18:21, 1:-1, FLOOR + 17:FLOOR + 19] = OCCUPIED   # beam 1.6-1.8 m up
    t = run(lab)
    r, p, _ = mid(t, 19)
    assert r and p == TV.STOOP
    assert mid(t, 35)[:2] == (True, TV.STAND)


def test_low_slab_is_crawled_under():
    lab = corridor()
    lab[18:21, 1:-1, FLOOR + 11:FLOOR + 13] = OCCUPIED   # slab 1.0-1.2 m up
    t = run(lab)
    r, p, _ = mid(t, 19)
    assert r and p == TV.CRAWL
    assert mid(t, 35)[0]


def test_narrow_door_is_passed_sideways_and_a_slot_is_not():
    for free_cells, expect in ((5, True), (3, False)):
        lab = corridor()
        lab[20, 1:-1, FLOOR + 1:CEIL] = OCCUPIED
        a = 5 - free_cells // 2
        lab[20, a:a + free_cells, FLOOR + 1:CEIL] = FREE
        t = run(lab)
        assert mid(t, 35)[0] == expect, (free_cells, expect)
        if expect:
            assert t["width"][20, 5] == TV.SIDEWAYS


def test_ten_centimetre_bump_is_stepped_over():
    lab = corridor()
    lab[20, 1:-1, FLOOR + 1] = OCCUPIED               # 0.1 m hose across the corridor
    t = run(lab)
    assert mid(t, 35)[0]


def test_unknown_in_the_body_blocks():
    lab = corridor()
    lab[20, 1:-1, FLOOR + 5:FLOOR + 15] = UNKNOWN
    t = run(lab)
    assert not mid(t, 35)[0]


def test_unseen_floor_is_stepped_across_but_a_void_is_not():
    """DESIGN §7 amendment: <= 0.5 m of floor with no observed support is
    crossed at the last support's level; a 1.0 m void blocks."""
    for width, expect in ((4, True), (10, False)):
        lab = corridor()
        lab[20:20 + width, 1:-1, FLOOR] = UNKNOWN       # floor never observed
        lab[20:20 + width, 1:-1, :FLOOR] = UNKNOWN
        t = run(lab)
        assert mid(t, 35)[0] == expect, (width, expect)
        if expect:
            assert t["gap_only"][21, 5]
    lab = corridor()
    lab[20:24, 1:-1, FLOOR] = FREE                      # grazing rays carved it free
    assert mid(run(lab), 35)[0]


def test_score_counts_by_value():
    t = run(corridor())
    s = TV.score(t, t)
    assert s["reach_recall"] == 1.0 and s["n_false_reach"] == 0


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

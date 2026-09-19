#!/usr/bin/env python3
"""Regression tests for the entry-map stage (DESIGN_entry_map.md §6).

Every assertion here compares a VALUE -- a clearance in metres, a grade at a
named cell, a cell-by-cell set relation -- and never a cell count. That rule is
RESULTS_SUMMARY.md §9-2 / appendix A-4, and it exists because a count-only test
already let a free-evidence double-counting bug through once.

Each test isolates ONE rule. In particular the door test runs with close_iter=0:
closing has its own test, and letting the cleanup stage run inside the door test
would mean a failure there could come from either rule.

Run: python tests/test_entry_map.py   (or: pytest tests/test_entry_map.py)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
from covor.entry import clean3d, project as PJ, inflate as IN, reach as RE  # noqa: E402
from covor.entry.config import (EntryCfg, UNKNOWN, FREE, OCCUPIED,          # noqa: E402
                                BLOCKED, NARROW, WALK)
from build_entry_map import build_entry_grid                                # noqa: E402

RES = 0.10
NZ = 24          # 24 rows = 2.4 m: the walk band (rows 1..18) fits with room over


def _extrude(plan, nz=NZ):
    """A 3D grid whose every z row above the floor is ``plan``.

    Row 0 is a solid floor, which is what makes the floor estimator's lowest peak
    unambiguous (and is also what the real scan looks like).
    """
    plan = np.asarray(plan, np.uint8)
    lab = np.zeros(plan.shape + (nz,), np.uint8)
    lab[:, :, 0] = OCCUPIED
    for k in range(1, nz):
        lab[:, :, k] = plan
    return lab


def _room_with_door(gap_cells, nx=44, ny=32, wall_j=16):
    """A rectangular room split by a wall at ``wall_j`` with one door in it.

    The door is ``gap_cells`` wide and centred on the x axis, so its clearance is
    a function of the gap alone and not of where it sits.
    """
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    plan[:, wall_j] = OCCUPIED
    i0 = (nx - gap_cells) // 2
    plan[i0:i0 + gap_cells, wall_j] = FREE
    return plan, (i0, i0 + gap_cells)


# ---------------------------------------------------------------------------
# (a) door width -> grade, by value
# ---------------------------------------------------------------------------
def test_door_width_grades_are_blocked_narrow_walk():
    """DESIGN §6: at w = 0.70, a 0.6 m door is blocked, 0.8 m narrow, 1.0 m walk.

    The clearance VALUE is asserted too, not just the grade, because the 1.0 m
    door lands exactly on clear_walk and a grade-only check would not notice the
    convention drifting by half a voxel.
    """
    cfg = EntryCfg(close_iter=0)          # isolate the width rule; see module docstring
    # (grade, clearance in m, passable at w = 0.70). The grade and passability
    # answer different questions and deliberately disagree on the 0.8 m door: a
    # 0.70 m body fits through 0.80 m of opening (0.35 >= w/2) while still having
    # to turn sideways, which is what NARROW means.
    want = {6: (BLOCKED, 0.25, False), 8: (NARROW, 0.35, True),
            10: (WALK, 0.45, True)}
    for gap, (grade, clearance, can_pass) in want.items():
        plan, (i0, i1) = _room_with_door(gap)
        lab = _extrude(plan)
        floor_row, _, _ = clean3d.floor_ceiling_rows(lab)
        assert floor_row == 0, floor_row
        bands, _ = PJ.project(lab, floor_row, cfg)
        band = bands["walk"]
        clear = IN.clearance_map(band, cfg)
        wc = IN.width_class(clear, cfg)
        door = (slice(i0, i1), 16)
        got_c = float(clear[door].max())
        assert abs(got_c - clearance) < 1e-12, (
            "%.1f m door: widest door cell has clearance %.4f m, expected %.4f"
            % (gap * RES, got_c, clearance))
        got_g = int(wc[door].max())
        assert got_g == grade, (
            "%.1f m door: grade %d, expected %d" % (gap * RES, got_g, grade))
        r = IN.radius_map(band.shape, cfg, None, 0.0)
        assert float(r[door].max()) == cfg.half_width()     # k_sigma = 0 here
        p = IN.passable_mask(band, clear, r)
        assert bool(p[door].any()) is can_pass, (
            "%.1f m door: passable=%s at w=%.2f, expected %s"
            % (gap * RES, bool(p[door].any()), cfg.w, can_pass))


# ---------------------------------------------------------------------------
# (b) desk: band membership decides the two bands
# ---------------------------------------------------------------------------
def test_desk_band_membership_by_height():
    """A desktop with clear space under it, at two crawl-band heights.

    DESIGN §6 asks for "0.72 m desk, empty underneath -> occupied in walk,
    passable in crawl". That CANNOT hold at the H_crawl = 0.90 the design's own
    §4 fixes, because z = 0.72 lies inside [0.10, 0.90] and the band rule makes
    one occupied cell enough. Both readings are pinned here by value so the
    contradiction is recorded rather than resolved by whoever edits next:

        H_crawl = 0.90 -> the desk column is OCCUPIED in the crawl band
        H_crawl = 0.70 -> it is FREE and passable, which is what §6 describes

    The walk band is occupied either way, which is the half both agree on.
    """
    nx = ny = 30
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    lab = _extrude(plan)
    # a desktop: one slab of occupied cells at the row holding z = 0.72,
    # free below (the knee hole) and free above.
    desk_row = int(0.72 / RES)            # row 7 spans z in [0.70, 0.80)
    assert desk_row == 7
    di, dj = slice(12, 18), slice(12, 18)
    lab[di, dj, desk_row] = OCCUPIED
    cell = (14, 14)

    for h_crawl, want_crawl in ((0.90, OCCUPIED), (0.70, FREE)):
        cfg = EntryCfg(close_iter=0, H_crawl=h_crawl)
        floor_row, _, _ = clean3d.floor_ceiling_rows(lab)
        bands, rows = PJ.project(lab, floor_row, cfg)
        assert bands["walk"][cell] == OCCUPIED, (
            "H_crawl=%.2f: walk band rows %s must contain the desktop row %d"
            % (h_crawl, rows["walk"], desk_row))
        assert bands["crawl"][cell] == want_crawl, (
            "H_crawl=%.2f: crawl band rows %s over desktop row %d gave label %d, "
            "expected %d" % (h_crawl, rows["crawl"], desk_row,
                             int(bands["crawl"][cell]), want_crawl))
        if want_crawl == FREE:
            clear = IN.clearance_map(bands["crawl"], cfg)
            r = IN.radius_map(bands["crawl"].shape, cfg, None, 0.0)
            assert IN.passable_mask(bands["crawl"], clear, r)[cell], (
                "H_crawl=%.2f: the knee hole is free but not passable" % h_crawl)


# ---------------------------------------------------------------------------
# (c) an unknown band stops reachability
# ---------------------------------------------------------------------------
def test_unknown_strip_cuts_reachability_before_it():
    """DESIGN §3-2: unknown is an obstacle, so a strip of it severs the room.

    Asserted at named cells on both sides, not by counting the reachable set: the
    cell one step in front of the strip must be reachable and every cell of and
    beyond the strip must not be.
    """
    nx, ny, strip = 44, 32, 16
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    plan[:, strip] = UNKNOWN                     # an unobserved band, wall to wall
    cfg = EntryCfg(close_iter=0)
    lab = _extrude(plan)
    g, _ = build_entry_grid(lab, cfg, entry_xy=None, k_sigma=0.0)
    R, P = g["walk_reachable"], g["walk_passable"]
    entry = tuple(int(v) for v in g["entry_ij"])
    assert entry[1] < strip, "the default entry landed on the far side: %s" % (entry,)
    # r = w/2 = 0.35 m, so passability itself already stops 4 cells short of the
    # strip; the pair compared here is the mirror image of that margin on each
    # side, which isolates reachability from the inflation.
    near, far = (22, strip - 5), (22, strip + 5)
    assert P[near] and P[far], "both mirror cells must be passable to start with"
    assert R[near], "the near side of the strip must be reachable"
    assert not P[(22, strip)] and not R[(22, strip)], (
        "the unknown strip is an obstacle and cannot be passable or reachable")
    assert not P[(22, strip - 1)], "the cell touching the strip must not be passable"
    assert not R[far], (
        "the far side is passable but must NOT be reachable through unknown")


# ---------------------------------------------------------------------------
# (d) a bad entry point is an error, not a silent empty map
# ---------------------------------------------------------------------------
def test_entry_point_on_unknown_or_occupied_raises():
    nx, ny = 30, 30
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    plan[5:9, 5:9] = UNKNOWN
    cfg = EntryCfg(close_iter=0)
    lab = _extrude(plan)
    ijk = np.zeros(3, np.int64)
    for xy, word in (((0.65, 0.65), "unobserved"),     # inside the unknown patch
                     ((0.05, 1.55), "occupied")):      # on the outer wall
        try:
            build_entry_grid(lab, cfg, entry_xy=xy, ijk_min=ijk, k_sigma=0.0)
        except ValueError as e:
            assert word in str(e), "entry at %s: wrong reason %r" % (xy, str(e))
        else:
            raise AssertionError("entry at %s was accepted" % (xy,))


# ---------------------------------------------------------------------------
# (e) closing is one-directional
# ---------------------------------------------------------------------------
def _wall_with_hole(thick, hole_w, nx=40, ny=32, j0=15, i0=20):
    """A plan whose dividing wall is ``thick`` cells deep with a hole in it."""
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    plan[:, j0:j0 + thick] = OCCUPIED
    plan[i0:i0 + hole_w, j0:j0 + thick] = FREE
    return plan, (slice(i0, i0 + hole_w), slice(j0, j0 + thick))


def test_closing_only_grows_occupied_and_never_grows_free():
    """DESIGN §3-5, asserted as a cell-by-cell set relation.

    A count-based version of this test passes even if closing swaps one cell for
    another, which is exactly the class of bug appendix A-4 is about.
    """
    plan, hole = _wall_with_hole(thick=3, hole_w=1)
    lab = _extrude(plan)
    for structure in ("3d", "2d"):
        out, info = clean3d.close_occupied(lab, 1, structure)
        occ0, occ1 = lab == OCCUPIED, out == OCCUPIED
        free0, free1 = lab == FREE, out == FREE
        assert (occ1 | occ0 == occ1).all(), (
            "%s: closing dropped a cell that was occupied" % structure)
        assert not (free1 & ~free0).any(), (
            "%s: closing created %d free cells"
            % (structure, int((free1 & ~free0).sum())))
        assert (occ1 & ~occ0).any(), "%s: closing did nothing at all" % structure
        assert (out[hole][:, :, 10] == OCCUPIED).any(), (
            "%s: a 1-cell hole in a 3-cell wall was not sealed" % structure)
        assert info["n_added_from_free"] == int((occ1 & free0).sum())


def test_closing_cannot_seal_a_hole_in_a_thin_wall():
    """MEASURED, and it undercuts what DESIGN §2-0 says closing is for.

    Closing fills a gap only where the structuring element cannot be placed
    inside the non-occupied set. A hole through a wall has free space at BOTH
    mouths, so for any wall thinner than three voxels the element always fits and
    nothing is sealed -- at any iteration count, with either structuring element:

        wall thickness   1        2        3
        hole sealed      never    never    yes (the middle layer)

    A surface-voxelised scan produces walls one to two voxels thick, so on this
    data the closing step does not do the job its design rationale gives it. What
    it does do is fill concave corners -- see the next test.
    """
    for structure in ("3d", "2d"):
        for thick in (1, 2):
            for iters in (1, 2):
                plan, hole = _wall_with_hole(thick=thick, hole_w=1)
                lab = _extrude(plan)
                out, _ = clean3d.close_occupied(lab, iters, structure)
                sealed = out[hole][:, :, 10] == OCCUPIED
                assert not sealed.any(), (
                    "%s wall=%d iters=%d: %d hole cells sealed -- the geometry of "
                    "closing changed" % (structure, thick, iters, int(sealed.sum())))


def test_closing_3d_thickens_walls_at_the_floor_but_2d_does_not():
    """The measured side effect of the "3d" reading, pinned so it stays visible.

    The floor-wall junction is a 90 deg concave corner, so a 3D closing fills the
    ring of free cells standing on the floor against every wall. The band rule
    then reads that ring as an occupied column, thickening every wall by one cell
    in plan view -- 0.10 m of clearance lost along all of them. The "2d" element
    closes the same pinhole without touching the junction.
    """
    nx, ny = 40, 30
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    lab = _extrude(plan)
    probe = (1, 15, 1)                        # free, on the floor, against a wall
    assert lab[probe] == FREE
    out3, _ = clean3d.close_occupied(lab, 1, "3d")
    out2, _ = clean3d.close_occupied(lab, 1, "2d")
    assert out3[probe] == OCCUPIED, "3d closing no longer fills the floor junction"
    assert out2[probe] == FREE, "2d closing must leave the floor junction alone"
    cfg3, cfg2 = EntryCfg(close_structure="3d"), EntryCfg(close_structure="2d")
    b3, _ = PJ.project(clean3d.clean(lab, cfg3)[0], 0, cfg3)
    b2, _ = PJ.project(clean3d.clean(lab, cfg2)[0], 0, cfg2)
    c3 = IN.clearance_map(b3["walk"], cfg3)
    c2 = IN.clearance_map(b2["walk"], cfg2)
    mid = (20, 15)
    assert abs(float(c2[mid]) - float(c3[mid]) - RES) < 1e-12, (
        "room half-width: 2d %.3f m vs 3d %.3f m -- expected exactly one voxel "
        "of difference" % (float(c2[mid]), float(c3[mid])))


# ---------------------------------------------------------------------------
# (f) the distance transform, against a reference implementation
# ---------------------------------------------------------------------------
def _reference_clearance(band, res):
    """Brute force: for every cell, the distance to the nearest obstacle centre.

    O(n_free * n_obstacle) and deliberately written as loops -- it shares no code
    with the implementation, which is the whole point of a reference.
    """
    obst = np.argwhere(IN.obstacle_mask(band))
    out = np.zeros(band.shape, float)
    for i in range(band.shape[0]):
        for j in range(band.shape[1]):
            if IN.obstacle_mask(band)[i, j]:
                continue
            d = np.sqrt(((obst - np.array([i, j])) ** 2).sum(axis=1)).min()
            out[i, j] = max(d - 0.5, 0.0) * res
    return out


def test_clearance_matches_a_reference_implementation_cell_by_cell():
    cfg = EntryCfg(close_iter=0)
    plan, _ = _room_with_door(8, nx=34, ny=26, wall_j=13)
    plan[6:9, 6:9] = OCCUPIED                 # a free-standing obstacle
    plan[20:24, 20:23] = UNKNOWN              # and an unobserved patch
    band = PJ.project_band(_extrude(plan), (1, 19))
    got = IN.clearance_map(band, cfg)
    want = _reference_clearance(band, cfg.res)
    d = np.abs(got - want).max()
    assert d < 1e-12, "clearance differs from the reference by %.3e m" % d


# ---------------------------------------------------------------------------
# positive ceiling: the map path and the GT path agree cell for cell
# ---------------------------------------------------------------------------
GT = os.path.join(ROOT, "results", "synth_room909", "gt_voxel.npz")


def test_gt_ceiling_map_adapter_matches_direct_labels():
    """DESIGN §6's positive ceiling, in the only form in which it says anything.

    Taken literally ("gt_voxel.npz through the pipeline == the GT entry map") the
    check is an identity: same input, same code. What it can test is the ADAPTER
    -- that feeding the GT as a built map's (occupied, free) masks, the way a
    real map arrives, lands on exactly the same entry grid as feeding the labels
    directly. Everything downstream of the adapter is shared by construction.

    So this is a narrow check, and saying so is the point: it does NOT bound the
    discretisation or coverage error between a map and the truth.
    """
    if not os.path.exists(GT):
        print("SKIP: %s not built" % GT)
        return
    z = np.load(GT, allow_pickle=False)
    lab, ijk = z["labels"], z["ijk_min"]
    cfg = EntryCfg()
    direct, _ = build_entry_grid(lab, cfg, ijk_min=ijk, k_sigma=0.0)
    via = clean3d.labels_from_masks(lab == OCCUPIED, lab == FREE)
    through, _ = build_entry_grid(via, cfg, ijk_min=ijk, k_sigma=0.0)
    for k in sorted(direct):
        a, b = np.asarray(direct[k]), np.asarray(through[k])
        if a.dtype.kind == "f":
            d = np.abs(a - b).max() if a.size else 0.0
            assert d == 0.0, "%s differs by %.3e" % (k, d)
        else:
            assert (a == b).all(), "%s differs in %d cells" % (k, int((a != b).sum()))


def test_default_entry_ignores_isolated_passable_islands():
    """The default entry point must not land on a one-cell island.

    Built to reproduce what room909's GT actually does: a lone passable cell at
    smaller x than the room. A plain smallest-x rule picks it and the map then
    reports one reachable cell.
    """
    nx, ny = 44, 32
    plan = np.full((nx, ny), UNKNOWN, np.uint8)
    plan[12:40, 4:28] = FREE                    # the room
    plan[1:10, 12:21] = FREE                    # a detached pocket at smaller x
    cfg = EntryCfg(close_iter=0)
    g, _ = build_entry_grid(_extrude(plan), cfg, k_sigma=0.0)
    entry = tuple(int(v) for v in g["entry_ij"])
    P, R = g["walk_passable"], g["walk_reachable"]
    assert P[1:10, 12:21].any(), "the fixture's pocket must itself be passable"
    assert entry[0] >= 12, (
        "the default entry landed on the detached pocket at %s" % (entry,))
    assert int(R.sum()) == int(P[12:40, 4:28].sum()), (
        "reachable (%d) must be exactly the room's passable set (%d)"
        % (int(R.sum()), int(P[12:40, 4:28].sum())))


def test_floor_is_not_the_ceiling_even_when_the_ceiling_is_the_bigger_peak():
    """The failure that produced an empty map without raising anything.

    A scanned ceiling is a bigger, cleaner slab than a scanned floor -- on
    room909 it is 5344 occupied cells against 1370. A height threshold on the
    peak therefore sits exactly where it can drop the floor and keep the
    ceiling, which is what happened: the A-condition map's floor peak held 1210
    cells against a 0.25*max threshold of 1360, the walk band was placed above
    the roof, and every metric came back zero.

    Here the ceiling is deliberately four times the floor.
    """
    from build_entry_map import select_floor
    nx, ny, nz = 44, 36, 28
    lab = np.zeros((nx, ny, nz), np.uint8)
    lab[8:30, 6:24, 2] = OCCUPIED                  # a modest floor patch
    lab[:, :, 24] = OCCUPIED                       # a ceiling slab over everything
    lab[8:30, 6:24, 3:24] = FREE                   # the room between them, clear
                                                   # through the whole walk band
    h = (lab == OCCUPIED).sum(axis=(0, 1))
    assert h[24] > 3 * h[2], "fixture must make the ceiling the taller peak"
    floor_row, ceil_row, info = clean3d.floor_ceiling_rows(lab)
    assert floor_row == 2, "lowest peak is the floor, got row %d" % floor_row
    assert ceil_row == 24
    cfg = EntryCfg(close_iter=0)
    assert select_floor(lab, info["peaks"], cfg) == 2
    g, gi = build_entry_grid(lab, cfg, k_sigma=0.0)
    assert gi["floor_row"] == 2
    assert int((g["walk_label"] == FREE).sum()) > 0, (
        "the walk band above the chosen floor must contain free space")


def test_select_floor_raises_when_no_peak_has_space_above_it():
    """A solid slab with nothing over it is not a floor, and saying so beats
    returning a map of zero passable cells."""
    from build_entry_map import select_floor
    lab = np.zeros((20, 20, 24), np.uint8)
    lab[:, :, 22] = OCCUPIED                       # only a roof, no room
    cfg = EntryCfg(close_iter=0)
    _, _, info = clean3d.floor_ceiling_rows(lab)
    try:
        select_floor(lab, info["peaks"], cfg)
    except ValueError as e:
        assert "no floor to stand on" in str(e), str(e)
    else:
        raise AssertionError("a roof-only grid was accepted as having a floor")


def _storeys(lab, cfg):
    from build_entry_map import select_floors
    return select_floors(lab, clean3d.floor_ceiling_rows(lab)[2]["peaks"], cfg)


def test_two_storey_building_reports_both_floors():
    """DESIGN §3-6: two floor peaks -> run §2 once per storey."""
    nx, ny, nz = 44, 36, 64
    lab = np.zeros((nx, ny, nz), np.uint8)
    for base in (2, 32):
        lab[6:38, 6:30, base] = OCCUPIED            # the slab
        lab[6:38, 6:30, base + 1:base + 28] = FREE  # the storey standing on it
    lab[:, :, 60] = OCCUPIED                        # the roof
    cfg = EntryCfg(close_iter=0)
    assert _storeys(lab, cfg) == [2, 32]
    from build_entry_map import build_entry_grids
    gs = build_entry_grids(lab, cfg, k_sigma=0.0)
    assert len(gs) == 2
    for (g, info), want in zip(gs, (2, 32)):
        assert info["floor_row"] == want
        assert info["floors"] == [2, 32]
        assert int((g["walk_label"] == FREE).sum()) > 0, (
            "storey at row %d has no free space in its band" % want)


def test_a_wall_course_above_the_band_is_not_a_storey():
    """The bug this rule was rewritten for.

    A room 2.6 m tall has 0.7 m of air above the 1.9 m walk band. Any occupied
    peak up there -- a wall course, a light fitting, the top of a shelf -- has
    free space over it and sits outside the lower floor's band, so a
    "not inside a lower band" rule calls it a second storey. room909 did exactly
    that. Connectivity is what rejects it: the air over the course is the same
    3D component as the room under it.
    """
    nx, ny, nz = 40, 30, 40
    lab = np.zeros((nx, ny, nz), np.uint8)
    lab[4:36, 4:26, 2] = OCCUPIED                   # floor
    lab[4:36, 4:26, 3:28] = FREE                    # a 2.5 m room
    lab[4:36, 4:26, 28] = OCCUPIED                  # ceiling
    lab[4:36, 4:5, 24] = OCCUPIED                   # a wall course at ~2.2 m,
    lab[4:36, 25:26, 24] = OCCUPIED                 # above the 1.9 m band
    cfg = EntryCfg(close_iter=0)
    peaks = clean3d.floor_ceiling_rows(lab)[2]["peaks"]
    assert 24 in peaks, "fixture must make the wall course a histogram peak"
    assert _storeys(lab, cfg) == [2], (
        "the wall course at row 24 was taken for a storey: %s"
        % _storeys(lab, cfg))


def test_room909_is_one_storey():
    """The real map, which has six occupied peaks and one floor."""
    if not os.path.exists(GT):
        print("SKIP: %s not built" % GT)
        return
    lab = np.load(GT, allow_pickle=False)["labels"]
    cfg = EntryCfg()
    cleaned, info = clean3d.clean(lab, cfg)
    peaks = info["floor"]["peaks"]
    assert len(peaks) > 3, "fixture assumption: this map has several peaks"
    from build_entry_map import select_floors
    assert select_floors(cleaned, peaks, cfg) == [6], (
        "room909 is a single storey; got %s from peaks %s"
        % (select_floors(cleaned, peaks, cfg), peaks))


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

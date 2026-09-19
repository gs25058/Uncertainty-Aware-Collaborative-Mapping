#!/usr/bin/env python3
"""The render layer must not change the judgement (DESIGN_entry_map.md §5).

"격자 계단이 사라지고 도면처럼 보인다. 라벨은 손대지 않는다." A picture that
disagrees with the grid it was made from is worse than no picture, because it is
the picture that gets read. So:

  * the cells INSIDE the vector wall outline are the occupied mask, to within the
    half-voxel the Douglas-Peucker tolerance is allowed to move a boundary;
  * drawing mutates nothing;
  * the route cost actually prefers clearance, checked on a case where the short
    way is the narrow way.

Run: python tests/test_entry_render.py   (or: pytest tests/test_entry_render.py)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np                                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
from covor.entry import render as RD, route as RT                   # noqa: E402
from covor.entry.config import EntryCfg, UNKNOWN, FREE, OCCUPIED    # noqa: E402
from build_entry_map import build_entry_grid                        # noqa: E402

RES = 0.10


def _extrude(plan, nz=24):
    plan = np.asarray(plan, np.uint8)
    lab = np.zeros(plan.shape + (nz,), np.uint8)
    lab[:, :, 0] = OCCUPIED
    for k in range(1, nz):
        lab[:, :, k] = plan
    return lab


def _room(nx=48, ny=36):
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    plan[10:14, 10:26] = OCCUPIED            # a pillar
    plan[30:40, 6:9] = OCCUPIED              # a bench
    plan[20:26, 28:36] = UNKNOWN             # an unobserved corner
    return plan


def test_vector_outline_encloses_exactly_the_occupied_mask():
    """Cells inside the drawn outline == occupied cells, up to the DP tolerance.

    Douglas-Peucker at eps = 0.5 voxel is allowed to move a boundary by half a
    cell, so a cell ON the boundary may flip. Nothing else may: every
    disagreement has to touch the mask's edge.
    """
    from scipy import ndimage
    cfg = EntryCfg(close_iter=0)
    g, _ = build_entry_grid(_extrude(_room()), cfg, k_sigma=0.0)
    lab = g["walk_label"]
    occ = lab == OCCUPIED
    ijk = np.asarray(g["ijk_min"])
    contours = RD.wall_outline(occ, cfg.res, ijk, eps_voxels=0.5)
    assert contours, "no wall contour was produced at all"
    path = RD.outline_path(contours, cfg.res, ijk)

    ii, jj = np.indices(occ.shape)
    centres = np.stack([(ii + ijk[0] + 0.5) * cfg.res,
                        (jj + ijk[1] + 0.5) * cfg.res], axis=-1).reshape(-1, 2)
    inside = path.contains_points(centres).reshape(occ.shape)

    bad = inside != occ
    if bad.any():
        edge = occ ^ ndimage.binary_erosion(occ, np.ones((3, 3), bool))
        near = ndimage.binary_dilation(edge, np.ones((3, 3), bool))
        assert (bad & ~near).sum() == 0, (
            "%d cells disagree with the outline away from any boundary"
            % int((bad & ~near).sum()))
    # and the disagreement must be a boundary effect, not a wholesale offset
    assert bad.sum() <= 0.10 * occ.sum(), (
        "%d of %d occupied cells disagree -- that is an offset, not a tolerance"
        % (int(bad.sum()), int(occ.sum())))


def test_drawing_does_not_mutate_the_grid():
    import matplotlib.pyplot as plt
    cfg = EntryCfg(close_iter=0)
    g, _ = build_entry_grid(_extrude(_room()), cfg, k_sigma=0.0)
    before = {k: np.array(v, copy=True) for k, v in g.items()}
    routes = RT.plan(g, "walk", cfg)
    fig, ax = plt.subplots()
    RD.draw_band(ax, g, "walk", cfg, routes=routes)
    plt.close(fig)
    for k, v in before.items():
        assert np.array_equal(np.asarray(g[k]), v), "draw_band mutated %r" % k


def test_douglas_peucker_keeps_corners_and_drops_collinear_points():
    line = np.stack([np.arange(21.0), np.zeros(21)], 1)
    assert len(RD.douglas_peucker(line, 0.5)) == 2
    bent = np.vstack([line, np.stack([np.full(10, 20.0), np.arange(1.0, 11)], 1)])
    out = RD.douglas_peucker(bent, 0.5)
    assert len(out) == 3, "an L should simplify to exactly its three vertices"
    assert np.allclose(out[1], [20.0, 0.0])
    spike = line.copy()
    spike[10, 1] = 2.0
    kept = RD.douglas_peucker(spike, 0.5)
    # the apex must survive by VALUE. Its count is not asserted: the two slanted
    # segments a spike creates leave the flat points ~1 unit off them, so DP
    # legitimately keeps several -- an expectation of "exactly 3" was wrong about
    # the algorithm, not about the code.
    assert np.any(np.all(np.isclose(kept, [10.0, 2.0]), axis=1)), \
        "a 2-unit spike was smoothed away at eps = 0.5"
    assert len(kept) > 2
    assert len(RD.douglas_peucker(spike, 3.0)) == 2, "and is dropped at eps = 3"


def test_route_prefers_the_wide_way_when_lambda_says_so():
    """Two ways round a block: short and narrow, or long and open.

    lambda = 0.5 is meant to buy width with length. The test asserts the choice
    flips at lambda = 0 -- so it is the cost term doing the work, not the
    geometry -- and reports both path lengths by value.
    """
    nx, ny = 60, 34
    plan = np.full((nx, ny), FREE, np.uint8)
    plan[0, :] = plan[-1, :] = plan[:, 0] = plan[:, -1] = OCCUPIED
    plan[20:40, 12:14] = OCCUPIED                 # the block to get around
    plan[20:40, 4:5] = OCCUPIED                   # pinches the short way
    lab = _extrude(plan)
    entry = ((5 + 0) * 0.1 + 0.05, (8 + 0) * 0.1 + 0.05)
    cfg_w = EntryCfg(close_iter=0, w=0.30, lambda_route=0.5)
    g, _ = build_entry_grid(lab, cfg_w, entry_xy=entry, k_sigma=0.0)
    goal = (52, 8)
    assert g["walk_passable"][goal]
    start = tuple(int(v) for v in g["walk_ij"]) if "walk_ij" in g else tuple(
        int(v) for v in g["entry_ij"])
    P, C, L = g["walk_passable"], g["walk_clearance"], g["walk_label"]
    wide = RT.astar(P, C, L, start, goal, cfg_w)
    narrow = RT.astar(P, C, L, start, goal, EntryCfg(close_iter=0, w=0.30,
                                                     lambda_route=0.0))
    assert wide is not None and narrow is not None
    yw = np.asarray(wide)[:, 1].mean()
    yn = np.asarray(narrow)[:, 1].mean()
    assert len(wide) >= len(narrow), (
        "the clearance-weighted route (%d cells) should not be shorter than the "
        "pure-distance one (%d)" % (len(wide), len(narrow)))
    assert yw > yn, (
        "lambda = 0.5 should pull the route away from the pinch: mean y %.2f "
        "vs %.2f at lambda = 0" % (yw, yn))


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

#!/usr/bin/env python3
"""Driver: a 3D occupancy grid -> entry_grid.npz (DESIGN_entry_map.md §2-0..§2-3).

    # GT entry map (the positive ceiling of DESIGN §6)
    python scripts/entry/build_entry_map.py --gt results/synth_room909/gt_voxel.npz \
        --out results/synth_room909/entry/entry_grid_gt.npz

    # a built map dumped by scripts/entry/dump_fused_map.py
    python scripts/entry/build_entry_map.py --map <dump>.npz --out <...>.npz

The GT entry map and the map entry map go through THIS function and no other, so
the only difference between them is the 3D grid handed in -- which is what makes
the DESIGN §6 metrics a comparison of maps rather than of pipelines.

Route and render are Part 4 and are deliberately absent: this stops at the grid.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.entry import clean3d, project as PJ, inflate as IN, reach as RE
from covor.entry.config import (EntryCfg, UNKNOWN, FREE, OCCUPIED,
                                BLOCKED, NARROW, WALK, CLS_UNKNOWN, CLS_OCCUPIED,
                                CLS_FREE_BLOCKED, CLS_FREE_CRAWL_ONLY,
                                CLS_FREE_NARROW, CLS_FREE_WALK)

BANDS = ("walk", "crawl")


def select_floors(lab3d, peaks, cfg):
    """Every STOREY's floor row, bottom up. DESIGN §3-6.

    "층이 둘이면 피크가 둘 -> 층별로 §2 반복." The occupied z histogram has many
    more peaks than it has storeys: room909's fused map peaks at rows 6, 9, 14,
    22, 29 and 32, of which exactly one is a floor. Taking every peak reports
    five storeys in a single-storey room.

    Two conditions separate them, and neither is a tuned constant:

      1. a floor has FREE SPACE ABOVE IT, inside the band a body occupies. A
         ceiling has nothing over it -- which is what once let the ceiling be
         selected silently (clean3d.floor_ceiling_rows).
      2. that free space is a volume NO LOWER STOREY ALREADY CLAIMS. This is the
         one that matters. A first attempt used "the peak is not inside a lower
         floor's band", which fails on any room taller than H_walk: room909's
         ceiling is at 2.60 m and the walk band stops at 1.90 m, so a wall
         course at 2.00 m sits above the band, has free air over it, and was
         accepted as a second storey. Connectivity settles it without a
         threshold -- the air over that course is the SAME 3D free component as
         the room below it, while the air over a real inter-storey slab is a
         different one, because the slab is what separates them.

    KNOWN LIMITS, both of them the design's own scope note ("계단 연결은 향후과제"):
      * a stairwell joins two storeys' air into one component, and the upper
        storey is then missed. That is the same assumption as not routing
        between storeys.
      * a spurious peak BELOW the real floor would be accepted first and the
        real floor skipped. It does not occur here (the histogram climbs
        monotonically into the floor) but it is the failure to look for.
    """
    lab3d = np.asarray(lab3d)
    nz = lab3d.shape[2]
    cc, _ = ndimage.label(lab3d == FREE,
                          structure=ndimage.generate_binary_structure(3, 1))
    floors, claimed, tried = [], set(), []
    for p in peaks:
        p = int(p)
        try:
            lo, hi = PJ.rows_for_band(p, cfg.H_walk, cfg, nz)
        except ValueError:
            # the band would fall off the top of the grid: there is no volume
            # above this peak at all, so it cannot be a floor. Reached by the
            # top course of any wall that runs to the grid ceiling.
            tried.append((p, 0, None))
            continue
        ids = cc[:, :, lo:hi]
        ids = ids[ids > 0]
        if ids.size == 0:
            tried.append((p, 0, None))
            continue
        vals, counts = np.unique(ids, return_counts=True)
        main = int(vals[int(np.argmax(counts))])   # the storey's air volume
        tried.append((p, int(ids.size), main))
        if main in claimed:
            continue
        claimed.add(main)
        floors.append(p)
    if not floors:
        raise ValueError(
            "no occupied peak has free space in the walk band above it "
            "(row, free voxels above, air component): %s -- this grid has no "
            "floor to stand on" % tried)
    return floors


def select_floor(lab3d, peaks, cfg):
    """The lowest storey's floor row. See select_floors."""
    return select_floors(lab3d, peaks, cfg)[0]


def merged_class(bands, wc):
    """The DESIGN §1 label set, from the two bands' grades.

    Read top down: a column is described by the best thing a rescuer can do in
    it. Walk if the walk band takes a forward pass, narrow if it takes a sideways
    one, crawl-only if the walk band does not but the crawl band does, and
    free_blocked if the space is free in the walk band yet too tight for either.
    """
    out = np.full(bands["walk"].shape, CLS_UNKNOWN, np.uint8)
    out[bands["walk"] == OCCUPIED] = CLS_OCCUPIED
    free_w = bands["walk"] == FREE
    out[free_w] = CLS_FREE_BLOCKED
    crawl_ok = (bands["crawl"] == FREE) & (wc["crawl"] >= NARROW)
    out[crawl_ok & ~(free_w & (wc["walk"] >= NARROW))] = CLS_FREE_CRAWL_ONLY
    out[free_w & (wc["walk"] == NARROW)] = CLS_FREE_NARROW
    out[free_w & (wc["walk"] == WALK)] = CLS_FREE_WALK
    return out


def snap_entry(passable, ij, radius_cells):
    """Nearest passable cell to ``ij`` within ``radius_cells``, or None.

    DESIGN §8 makes the entry point user-specified. A door is a location a
    person names to within a metre, not a cell index; and the cell under that
    location may be the door FRAME, a step, or the erosion band along the wall
    -- free but narrower than w/2. Snapping to the nearest passable cell is what
    "enter here" means on a grid. The radius bounds how far the map may move
    the door, and the move is reported, so a snap of 1.4 m is visible.
    """
    P = np.asarray(passable, bool)
    if not P.any():
        return None
    idx = np.argwhere(P)
    d = np.hypot(idx[:, 0] - ij[0], idx[:, 1] - ij[1])
    k = int(np.argmin(d))
    return (tuple(int(v) for v in idx[k]), float(d[k])) if d[k] <= radius_cells else None


def build_entry_grid(lab3d, cfg, sigma_xy=None, n_obs=None, entry_xy=None,
                     ijk_min=(0, 0, 0), k_sigma=None, floor_row=None,
                     entry_snap_m=None):
    """3D labels -> the full entry grid, as a dict of arrays + info.

    sigma_xy   (nx, ny) per-column sigma_xy in metres, or None for the k = 0 case.
    n_obs      (nx, ny) frames that wrote each column, or None (GT has no frames).
    entry_xy   (x, y) in metres; None uses cfg.entry_xy, then reach.default_entry.
    floor_row  build this storey instead of the lowest one (see select_floors).
               ``info["floors"]`` always lists every storey that was detected, so
               a caller that ignores this argument can still see there were more.
    """
    ijk_min = np.asarray(ijk_min, np.int64)
    lab3d, info = clean3d.clean(np.asarray(lab3d, np.uint8), cfg)
    floors = select_floors(lab3d, info["floor"]["peaks"], cfg)
    if floor_row is None:
        floor_row = floors[0]
    elif int(floor_row) not in floors:
        raise ValueError("floor row %s is not one of the detected storeys %s"
                         % (floor_row, floors))
    info["floors"] = [int(f) for f in floors]
    info["floor_index"] = floors.index(int(floor_row))
    info["floor_row"] = int(floor_row)
    floor_row = int(floor_row)
    bands, rows = PJ.project(lab3d, floor_row, cfg)

    clear, wc, r, passable = {}, {}, {}, {}
    for b in BANDS:
        clear[b] = IN.clearance_map(bands[b], cfg)
        wc[b] = IN.width_class(clear[b], cfg)
        r[b] = IN.radius_map(bands[b].shape, cfg, sigma_xy, k_sigma)
        passable[b] = IN.passable_mask(bands[b], clear[b], r[b])

    # The entry point is chosen ONCE, on the walk band, and both bands are then
    # scored from the same door. Choosing it per band would let the crawl map
    # enter through an opening the walk map cannot use, and the two reachability
    # figures would no longer be comparable.
    xy = entry_xy if entry_xy is not None else cfg.entry_xy
    info["entry_snapped_m"] = 0.0
    if xy is not None:
        entry_ij = RE.world_to_cell(xy, ijk_min, cfg.res)
        P = passable["walk"]
        inb = 0 <= entry_ij[0] < P.shape[0] and 0 <= entry_ij[1] < P.shape[1]
        if entry_snap_m and not (inb and P[entry_ij]):
            hit = snap_entry(P, entry_ij, entry_snap_m / cfg.res)
            if hit is None:
                raise ValueError("entry point %s has no passable cell within %.2f m"
                                 % (tuple(xy), entry_snap_m))
            entry_ij, dcells = hit
            info["entry_snapped_m"] = float(dcells * cfg.res)
            info["entry_nominal_xy"] = [float(v) for v in xy[:2]]
    else:
        entry_ij = RE.default_entry(passable["walk"])
    reach_m = {b: RE.reachable(passable[b], entry_ij, bands[b]) for b in BANDS}

    out = dict(ijk_min=ijk_min, res=np.array([cfg.res]),
               floor_row=np.array([info["floor_row"]]),
               ceiling_row=np.array([info["ceiling_row"]]),
               entry_ij=np.array(entry_ij, np.int64),
               entry_xy=RE.cell_to_world(entry_ij, ijk_min, cfg.res),
               entry_class=merged_class(bands, wc))
    for b in BANDS:
        out["%s_label" % b] = bands[b]
        out["%s_clearance" % b] = clear[b].astype(np.float32)
        out["%s_width_class" % b] = wc[b]
        out["%s_r" % b] = r[b].astype(np.float32)
        out["%s_passable" % b] = passable[b]
        out["%s_reachable" % b] = reach_m[b]
        out["%s_rows" % b] = np.array(rows[b], np.int64)
    out["n_obs"] = (np.zeros(bands["walk"].shape, np.int32) if n_obs is None
                    else np.asarray(n_obs, np.int32))
    out["has_n_obs"] = np.array([n_obs is not None])
    out["sigma_xy"] = (np.zeros(bands["walk"].shape, np.float32) if sigma_xy is None
                       else np.asarray(sigma_xy, np.float32))
    info["rows"] = {k: list(v) for k, v in rows.items()}
    info["entry_ij"] = list(entry_ij)
    return out, info


def band_table(g):
    """Cells per class, per band -- the table DESIGN §6 requires alongside any
    single number (RESULTS_SUMMARY §9-1: a rate on its own is gameable)."""
    rows = []
    for b in BANDS:
        lab, wc = g["%s_label" % b], g["%s_width_class" % b]
        rows.append(dict(
            band=b,
            unknown=int((lab == UNKNOWN).sum()),
            occupied=int((lab == OCCUPIED).sum()),
            free=int((lab == FREE).sum()),
            free_blocked=int(((lab == FREE) & (wc == BLOCKED)).sum()),
            free_narrow=int(((lab == FREE) & (wc == NARROW)).sum()),
            free_walk=int(((lab == FREE) & (wc == WALK)).sum()),
            passable=int(g["%s_passable" % b].sum()),
            reachable=int(g["%s_reachable" % b].sum()),
        ))
    return rows


def build_entry_grids(lab3d, cfg, **kw):
    """One entry grid PER STOREY, bottom up (DESIGN §3-6).

    Stair connection is out of scope by the design's own words, so these are
    independent maps that happen to share a building: each has its own entry
    point and its own reachable set, and nothing here claims you can get from
    one to the next.
    """
    kw.pop("floor_row", None)
    clean, info = clean3d.clean(np.asarray(lab3d, np.uint8), cfg)
    out = []
    for f in select_floors(clean, info["floor"]["peaks"], cfg):
        out.append(build_entry_grid(lab3d, cfg, floor_row=f, **kw))
    return out


def load_3d(args):
    """(labels, ijk_min, res) from either a gt_voxel.npz or a dumped map.

    The resolution travels with the grid. EntryCfg's default is only a default:
    a 0.05 m grid read with res = 0.10 puts the band at 0.05-0.95 m instead of
    0.10-1.90 m, and nothing raises.
    """
    z = np.load(args.gt or args.map, allow_pickle=False)
    res = float(np.asarray(z["res"]).ravel()[0])
    if "labels" in z.files:
        return z["labels"], z["ijk_min"], res
    return clean3d.labels_from_masks(z["M_occ"], z["M_free"]), z["ijk_min"], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", help="gt_voxel.npz (labels straight from the mesh)")
    ap.add_argument("--map", help="dump from scripts/entry/dump_fused_map.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--k-sigma", type=float, default=0.0,
                    help="0 until PREREG_entry_sigma.md is registered and approved")
    ap.add_argument("--cube-min-voxels", type=int, default=None)
    ap.add_argument("--entry", default=None, help="x,y in metres")
    ap.add_argument("--entry-snap", type=float, default=None,
                    help="snap --entry to the nearest passable cell within this "
                         "many metres (see snap_entry)")
    args = ap.parse_args()
    if bool(args.gt) == bool(args.map):
        ap.error("give exactly one of --gt / --map")

    lab, ijk_min, res = load_3d(args)
    kw = dict(res=res)
    if args.w is not None:
        kw["w"] = args.w
    if args.cube_min_voxels is not None:
        kw["cube_min_voxels"] = args.cube_min_voxels
    cfg = EntryCfg(**kw)
    sigma = n_obs = None
    if args.map:
        z = np.load(args.map, allow_pickle=False)
        if "n_obs" in z.files:
            n_obs = z["n_obs"]
        if "sigma_xy" in z.files:
            sigma = z["sigma_xy"]
    xy = tuple(float(v) for v in args.entry.split(",")) if args.entry else None

    g, info = build_entry_grid(lab, cfg, sigma_xy=sigma, n_obs=n_obs,
                               entry_xy=xy, ijk_min=ijk_min,
                               k_sigma=args.k_sigma, entry_snap_m=args.entry_snap)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, cfg=json.dumps(cfg.__dict__),
                        info=json.dumps(info), **g)
    print("floor row %d (z = %.2f m), ceiling row %d; entry cell %s = (%.2f, %.2f) m"
          % (info["floor_row"], (info["floor_row"] + ijk_min[2]) * cfg.res,
             info["ceiling_row"], info["entry_ij"],
             g["entry_xy"][0], g["entry_xy"][1]))
    print("%-6s %8s %8s %8s | %8s %8s %8s | %8s %9s"
          % ("band", "unknown", "occupied", "free", "blocked", "narrow", "walk",
             "passable", "reachable"))
    for r in band_table(g):
        print("%-6s %8d %8d %8d | %8d %8d %8d | %8d %9d"
              % (r["band"], r["unknown"], r["occupied"], r["free"],
                 r["free_blocked"], r["free_narrow"], r["free_walk"],
                 r["passable"], r["reachable"]))
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()

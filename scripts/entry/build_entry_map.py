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

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.entry import clean3d, project as PJ, inflate as IN, reach as RE
from covor.entry.config import (EntryCfg, UNKNOWN, FREE, OCCUPIED,
                                BLOCKED, NARROW, WALK, CLS_UNKNOWN, CLS_OCCUPIED,
                                CLS_FREE_BLOCKED, CLS_FREE_CRAWL_ONLY,
                                CLS_FREE_NARROW, CLS_FREE_WALK)

BANDS = ("walk", "crawl")


def select_floor(lab3d, peaks, cfg):
    """The lowest occupied peak THAT HAS FREE SPACE ABOVE IT.

    The peak itself is the design's rule (§3-6). This adds the one structural
    condition that tells a floor from a ceiling without a tuned constant: a floor
    has walkable volume over it and a ceiling has nothing. It exists because the
    ceiling once WAS selected, silently, and the map came back empty rather than
    wrong-looking (see clean3d.floor_ceiling_rows).

    Raises rather than guessing when no peak qualifies -- a grid with no free
    space above any horizontal surface is not something to publish a floor plan
    of.
    """
    nz = np.asarray(lab3d).shape[2]
    tried = []
    for p in peaks:
        lo, hi = PJ.rows_for_band(int(p), cfg.H_walk, cfg, nz)
        n_free = int((np.asarray(lab3d)[:, :, lo:hi] == FREE).sum())
        tried.append((int(p), n_free))
        if n_free > 0:
            return int(p)
    raise ValueError(
        "no occupied peak has free space in the walk band above it "
        "(row, free voxels above): %s -- this grid has no floor to stand on"
        % tried)


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


def build_entry_grid(lab3d, cfg, sigma_xy=None, n_obs=None, entry_xy=None,
                     ijk_min=(0, 0, 0), k_sigma=None):
    """3D labels -> the full entry grid, as a dict of arrays + info.

    sigma_xy  (nx, ny) per-column sigma_xy in metres, or None for the k = 0 case.
    n_obs     (nx, ny) frames that wrote each column, or None (GT has no frames).
    entry_xy  (x, y) in metres; None uses cfg.entry_xy, then reach.default_entry.
    """
    ijk_min = np.asarray(ijk_min, np.int64)
    lab3d, info = clean3d.clean(np.asarray(lab3d, np.uint8), cfg)
    floor_row = select_floor(lab3d, info["floor"]["peaks"], cfg)
    info["floor_row"] = floor_row
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
    if xy is not None:
        entry_ij = RE.world_to_cell(xy, ijk_min, cfg.res)
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


def load_3d(args):
    """(labels, ijk_min) from either a gt_voxel.npz or a dumped map."""
    if args.gt:
        z = np.load(args.gt, allow_pickle=False)
        return z["labels"], z["ijk_min"]
    z = np.load(args.map, allow_pickle=False)
    if "labels" in z.files:
        return z["labels"], z["ijk_min"]
    return clean3d.labels_from_masks(z["M_occ"], z["M_free"]), z["ijk_min"]


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
    args = ap.parse_args()
    if bool(args.gt) == bool(args.map):
        ap.error("give exactly one of --gt / --map")

    kw = {}
    if args.w is not None:
        kw["w"] = args.w
    if args.cube_min_voxels is not None:
        kw["cube_min_voxels"] = args.cube_min_voxels
    cfg = EntryCfg(**kw)
    lab, ijk_min = load_3d(args)
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
                               k_sigma=args.k_sigma)
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

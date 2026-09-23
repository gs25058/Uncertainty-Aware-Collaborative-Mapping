#!/usr/bin/env python3
"""PREREG_sgbm_depth.md §7 auxiliaries (no part in the verdict), plus the cause split.

    python scripts/entry/sgbm_aux.py

  1. 3D precision / recall / false-free, ideal vs sgbm, observability domain.
  2. Per-4 m-slab maximum walk-band clearance: GT / ideal / sgbm.
  3. Walk-band unknown difference figure (sgbm - ideal).
  4. Why GT-passable walk cells are not passable on each map: the map's band
     label there (occupied / unknown / free but narrower than w/2), and each
     map's floor-row agreement with the GT's (flatten_floor.py estimates both).
  5. Where the sgbm map puts occupied voxels on GT-free cells, by height above
     the GT's local floor and by corridor slab.
  6. DIAGNOSTIC: both maps flattened with the GT's OWN floor shift instead of
     their own estimate, scored the same way. This removes floor estimation
     from the comparison; it is not a deployable map (a map has no GT floor)
     and it is not the pre-registered comparison.

  7. PREREG_sgbm_rescue.md auxiliaries: every arm of the rescue gets 1, 4, 5
     and 6, and the primary arm (sgbm + A + B) gets the cube_min_voxels sweep
     {1, 3, 10, 30} on its own floor -- reported, never judged.
  8. Distance from each occupied-on-GT-free voxel to the nearest GT-occupied
     voxel: floating debris, or walls grown thicker?

Everything is read from files the verdict run wrote; nothing is re-rendered.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.entry import metrics as EM
from covor.entry.config import EntryCfg, OCCUPIED, UNKNOWN, FREE
from covor.synth import mesh_gt as MG, metrics as ME
from build_entry_map import build_entry_grid
from run_entry_metrics import load_map
from flatten_floor import shift_columns

SEQ = "/src/gs25058/cr_RNE/covor_slam/results/synth_corridor915"
D = SEQ + "/entry"
MAP = D + "/map3d_C_3drone_ii_all_cams_depth_only_%s_s4%s.npz"
ARMS = (("ideal", "clean"), ("sgbm", "clean_sgbm"),
        ("sgbm_A", "clean_sgbm_A"), ("sgbm_B", "clean_sgbm_B035"),
        ("sgbm_AB", "clean_sgbm_A_B035"), ("ideal_B", "clean_B035"))
PRIMARY = "sgbm_AB"
CUBES = (1, 3, 10, 30)
DOOR, SNAP = (-2.35, -19.55), 1.5


def grid(lab, ijk, sig=None, nobs=None, cube=1):
    cfg = EntryCfg(res=0.10, w=0.70, cube_min_voxels=cube)
    try:
        g, _ = build_entry_grid(lab, cfg, sigma_xy=sig, n_obs=nobs, ijk_min=ijk,
                                k_sigma=0.0, entry_xy=DOOR, entry_snap_m=SNAP)
    except ValueError:
        g, _ = build_entry_grid(lab, cfg, sigma_xy=sig, n_obs=nobs, ijk_min=ijk,
                                k_sigma=0.0)
    return g


def main():
    out = {}
    lab, ijk, res, _, _ = MG.load_gt(SEQ + "/gt_voxel.npz")
    obs = np.load(SEQ + "/obs_mask_ii_all_cams.npz")["obs"]

    print("1. 3D, observability domain")
    out["3d"] = {}
    for arm, tag in ARMS:
        z = np.load(MAP % (tag, ""))
        s = ME.score(z["M_occ"], z["M_free"], lab, ijk, res, obs)
        out["3d"][arm] = {k: float(s[k]) for k in
                          ("precision", "recall", "iou_1vox", "false_free_rate")}
        out["3d"][arm].update(n_occ=int(z["M_occ"].sum()), n_free=int(z["M_free"].sum()),
                              cols_observed=int((z["n_obs"] > 0).sum()))
        print("   %-5s %s" % (arm, out["3d"][arm]))

    lab_gf, ijk_gf, _, _, _, _ = load_map(SEQ + "/gt_voxel_flat.npz")
    ggt = grid(lab_gf, ijk_gf)
    G = {"gt": ggt}
    floor = {"gt": np.load(SEQ + "/gt_voxel_flat.npz")["floor_row_map"]}
    for arm, tag in ARMS:
        p = MAP % (tag, "_flat")
        lp, ij, sig, nobs, _, _ = load_map(p)
        G[arm] = grid(lp, ij, sig, nobs)
        floor[arm] = np.load(p)["floor_row_map"]
    y0 = int(ijk_gf[1])

    print("2. max walk clearance per 4 m slab [m]")
    out["clearance"] = {}
    for ya in range(-20, 20, 4):
        j = slice(max(int(round(ya / res)) - y0, 0), int(round((ya + 4) / res)) - y0)
        row = {}
        for k in ["gt"] + [a for a, _ in ARMS]:
            c = np.where(G[k]["walk_label"] == FREE, G[k]["walk_clearance"], 0.0)[:, j]
            row[k] = round(float(c.max()), 2) if c.size else 0.0
        out["clearance"]["%d..%d" % (ya, ya + 4)] = row
        print("   y %+3d..%+3d  %s" % (ya, ya + 4, row))

    print("4. GT-passable walk cells not passable on the map, by map band label")
    gp = ggt["walk_passable"]
    out["miss"] = {}
    for arm, _ in ARMS:
        g = G[arm]
        miss = gp & ~g["walk_passable"]
        L = g["walk_label"][miss]
        obsd = g["n_obs"] > 0
        d = floor[arm] - floor["gt"]
        out["miss"][arm] = dict(
            gt_passable=int(gp.sum()), missed=int(miss.sum()),
            occupied=int((L == OCCUPIED).sum()), unknown=int((L == UNKNOWN).sum()),
            free_narrow=int((L == FREE).sum()),
            floor_agree_le1=float((np.abs(d[obsd]) <= 1).mean()),
            floor_exact=float((d[obsd] == 0).mean()),
            floor_diff_hist={int(a): int(b) for a, b in
                             zip(*np.unique(d[obsd], return_counts=True))})
        print("   %-5s %s" % (arm, out["miss"][arm]))

    print("5. occupied voxels on GT-free cells, by height above the GT floor")
    out["floaters"] = {}
    edges = [-1, 0.1, 0.5, 1.0, 1.9, 9]
    for arm, tag in ARMS:
        z = np.load(MAP % (tag, ""))
        fp = z["M_occ"] & (lab == MG.FREE)
        I, J, K = np.nonzero(fp)
        # floor_row_map rows are in the GT grid's own row index (same origin)
        h = (K - floor["gt"][I, J]) * res
        hc = np.histogram(h, edges)[0]
        f = dict(total=int(fp.sum()),
                 by_height={"%.1f..%.1f" % (a, b): int(c) for a, b, c in
                            zip(edges[:-1], edges[1:], hc)},
                 by_slab={})
        for ya in range(-20, 20, 8):
            jj = (J >= int(round(ya / res)) - int(ijk[1])) & \
                 (J < int(round((ya + 8) / res)) - int(ijk[1]))
            f["by_slab"]["%d..%d" % (ya, ya + 8)] = int(jj.sum())
        out["floaters"][arm] = f
        print("   %-7s %s" % (arm, f))

    print("8. occupied-on-GT-free voxels: distance to the nearest GT-occupied voxel")
    from scipy import ndimage
    dwall = ndimage.distance_transform_edt(lab != MG.OCC) * res
    out["wall_distance"] = {}
    for arm, tag in ARMS:
        z = np.load(MAP % (tag, ""))
        v = dwall[z["M_occ"] & (lab == MG.FREE)]
        if not len(v):
            continue
        out["wall_distance"][arm] = dict(
            n=int(len(v)), le_1vox=float((v <= res + 1e-6).mean()),
            le_2vox=float((v <= 2 * res + 1e-6).mean()),
            le_3vox=float((v <= 3 * res + 1e-6).mean()),
            gt_0p5m=float((v > 0.5).mean()), median=float(np.median(v)))
        print("   %-7s %s" % (arm, out["wall_distance"][arm]))

    print("6. DIAGNOSTIC: maps flattened with the GT floor shift (not the verdict)")
    gshift = np.load(SEQ + "/gt_voxel_flat.npz")["flatten_shift"]
    out["gt_floor_diag"] = {}
    for arm, tag in ARMS:
        lp, ij, sig, nobs, _, _ = load_map(MAP % (tag, ""))
        g = grid(shift_columns(lp, gshift, UNKNOWN), ij, sig, nobs)
        r = EM.score(g, ggt, "walk")
        out["gt_floor_diag"][arm] = {k: (float(r[k]) if isinstance(r[k], (float, np.floating))
                                         else int(r[k])) for k in
                                     ("passable_recall", "n_true_passable",
                                      "false_passable_rate", "n_false_passable",
                                      "n_pred_passable", "n_pred_unknown")}
        print("   %-5s %s" % (arm, out["gt_floor_diag"][arm]))

    print("7. cube_min_voxels sweep on the primary arm (own floor; reported, not judged)")
    out["cube_sweep"] = {}
    lp, ij, sig, nobs, _, _ = load_map(MAP % (dict(ARMS)[PRIMARY], "_flat"))
    for c in CUBES:
        g = grid(lp, ij, sig, nobs, cube=c)
        r = EM.score(g, ggt, "walk")
        out["cube_sweep"][c] = {k: (float(r[k]) if isinstance(r[k], (float, np.floating))
                                    else int(r[k])) for k in
                                ("passable_recall", "n_true_passable",
                                 "false_passable_rate", "n_false_passable",
                                 "n_false_passable_on_gt_occupied",
                                 "n_pred_passable", "n_pred_unknown")}
        print("   cube %2d %s" % (c, out["cube_sweep"][c]))

    print("3. unknown difference figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ui = G["ideal"]["walk_label"] == UNKNOWN
    us = G["sgbm"]["walk_label"] == UNKNOWN
    img = np.full(ui.shape, np.nan)
    img[ui & us] = 0
    img[us & ~ui] = 1
    img[ui & ~us] = 2
    gtin = lab_gf.max(axis=2) > 0
    img[~gtin & ~ui & ~us] = np.nan
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#d9d9d9", "#d62728", "#1f77b4"])
    ext = (ijk_gf[0] * res, (ijk_gf[0] + ui.shape[0]) * res,
           ijk_gf[1] * res, (ijk_gf[1] + ui.shape[1]) * res)
    f, ax = plt.subplots(1, 3, figsize=(12, 13), sharey=True)
    for a, (k, t) in zip(ax, (("ideal", "ideal depth"), ("sgbm", "SGBM depth"))):
        L = G[k]["walk_label"].astype(float)
        a.imshow(np.where(G[k]["walk_passable"], 3, L).T, origin="lower", extent=ext,
                 cmap=ListedColormap(["#eeeeee", "#9ecae1", "#333333", "#31a354"]),
                 vmin=0, vmax=3, interpolation="nearest")
        a.set_title("%s\nwalk band: unknown / free / occupied / passable" % t, fontsize=9)
    ax[2].imshow(img.T, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=2,
                 interpolation="nearest")
    ax[2].set_title("unknown: both (grey) / SGBM only (red) / ideal only (blue)",
                    fontsize=9)
    for a in ax:
        a.set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    f.suptitle("2026-09-15 corridor, walk band, w 0.70, floor-flattened: ideal vs SGBM depth",
               fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.97))
    fig = D + "/sgbm_vs_ideal_walk.png"
    f.savefig(fig, dpi=110)
    print("   -> %s" % fig)
    labels = [int(v) for v in np.unique(G["ideal"]["walk_label"])]
    assert set(labels) <= {UNKNOWN, FREE, OCCUPIED} and (UNKNOWN, FREE, OCCUPIED) == (0, 1, 2), \
        "figure colours assume unknown=0, free=1, occupied=2"

    js = D + "/sgbm_aux.json"
    json.dump(out, open(js, "w"), indent=1)
    print("-> %s" % js)


if __name__ == "__main__":
    main()

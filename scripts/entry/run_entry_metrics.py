#!/usr/bin/env python3
"""Score one or more dumped 3D maps against the GT entry map (DESIGN §6).

    python scripts/entry/run_entry_metrics.py --gt results/synth_room909/gt_voxel.npz \
        --map results/synth_room909/entry/map3d_C_3drone_ii_all_cams_depth_only_clean_s4.npz \
        --w 0.5,0.7,0.9 --k-sigma 0

Both sides run the SAME pipeline with the SAME cfg; the only difference is the
3D grid. Every row carries the whole §6 record, so no rate can be quoted without
the counts that make it readable (RESULTS_SUMMARY §9-1).

ENTRY POINT. The door is chosen once from the GT map and handed to the map under
test, so "reachable" means the same journey on both sides. If the map cannot be
entered at the true door the row says so rather than the run failing -- that is
itself a result, and the remaining metrics are still computed from the map's own
default door.
"""
import argparse
import csv
import fcntl
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.entry import clean3d, metrics as EM
from covor.entry.config import EntryCfg
from build_entry_map import build_entry_grid, BANDS


def append_row(path, row, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def load_map(path):
    z = np.load(path, allow_pickle=False)
    lab = (z["labels"] if "labels" in z.files
           else clean3d.labels_from_masks(z["M_occ"], z["M_free"]))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    sig = z["sigma_xy"] if "sigma_xy" in z.files else None
    nobs = z["n_obs"] if "n_obs" in z.files else None
    return lab, z["ijk_min"], sig, nobs, meta, float(np.asarray(z["res"]).ravel()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--map", action="append", required=True,
                    help="repeatable: path[:tag]")
    ap.add_argument("--w", default="0.5,0.7,0.9")
    ap.add_argument("--k-sigma", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zg = np.load(args.gt, allow_pickle=False)
    lab_gt, ijk_gt = zg["labels"], zg["ijk_min"]
    res_gt = float(np.asarray(zg["res"]).ravel()[0])
    ws = [float(v) for v in args.w.split(",")]
    out_csv = args.out or os.path.join(os.path.dirname(args.map[0].split(":")[0]),
                                       "results_entry.csv")
    fields = None
    for spec in args.map:
        path, _, tag = spec.partition(":")
        tag = tag or os.path.basename(path).replace("map3d_", "").replace(".npz", "")
        lab_p, ijk_p, sig, nobs, meta, res_p = load_map(path)
        if not np.array_equal(np.asarray(ijk_p), np.asarray(ijk_gt)):
            raise ValueError("%s is on a different grid origin than the GT" % path)
        print("\n=== %s ===" % tag)
        if meta:
            print("    %s" % {k: meta[k] for k in
                              ("cond", "coverage", "arm", "stride", "corrupt")
                              if k in meta})
        if abs(res_p - res_gt) > 1e-9:
            raise ValueError("%s is %.3f m and the GT is %.3f m -- "
                             "PREREG_RESOLUTION.md forbids scoring across "
                             "resolutions" % (path, res_p, res_gt))
        for w in ws:
            cfg = EntryCfg(res=res_gt, w=w)
            g_gt, i_gt = build_entry_grid(lab_gt, cfg, ijk_min=ijk_gt, k_sigma=0.0)
            door = tuple(float(v) for v in g_gt["entry_xy"])
            try:
                g_p, i_p = build_entry_grid(lab_p, cfg, sigma_xy=sig, n_obs=nobs,
                                            entry_xy=door, ijk_min=ijk_p,
                                            k_sigma=args.k_sigma)
                entry_from, entry_note = "gt_door", ""
            except ValueError as e:
                g_p, i_p = build_entry_grid(lab_p, cfg, sigma_xy=sig, n_obs=nobs,
                                            ijk_min=ijk_p, k_sigma=args.k_sigma)
                entry_from, entry_note = "map_default", str(e)
                print("    w=%.2f: the GT door is not usable on this map -- %s"
                      % (w, entry_note))
            for band in BANDS:
                rec = EM.score(g_p, g_gt, band)
                rec.update(tag=tag, w=w, k_sigma=args.k_sigma,
                           entry_from=entry_from, entry_note=entry_note,
                           gt_entry_x=door[0], gt_entry_y=door[1],
                           cond=meta.get("cond", ""), coverage=meta.get("coverage", ""),
                           arm=meta.get("arm", ""), corrupt=meta.get("corrupt", ""),
                           stride=meta.get("stride", ""),
                           sigma_xy_median=meta.get("sigma_xy_median", float("nan")),
                           sigma_xy_p90=meta.get("sigma_xy_p90", float("nan")),
                           ate_pooled=float(np.sqrt(np.mean(
                               [v ** 2 for v in meta.get("ate", {}).values()])))
                           if meta.get("ate") else float("nan"))
                if fields is None:
                    fields = (["tag", "cond", "coverage", "arm", "corrupt", "stride"]
                              + list(EM.FIELDS_HEAD)
                              + [k for k in rec if k not in EM.FIELDS_HEAD
                                 and k not in ("tag", "cond", "coverage", "arm",
                                               "corrupt", "stride")])
                append_row(out_csv, rec, fields)
                print("    " + EM.row_text(rec))
        # the confusion matrix is per map at the design's own w
        cfg = EntryCfg(res=res_gt, w=0.70)
        g_gt, _ = build_entry_grid(lab_gt, cfg, ijk_min=ijk_gt, k_sigma=0.0)
        g_p, _ = build_entry_grid(lab_p, cfg, sigma_xy=sig, n_obs=nobs,
                                  ijk_min=ijk_p, k_sigma=args.k_sigma)
        rec = EM.score(g_p, g_gt, "walk")
        print("    width grade confusion, walk band, w=0.70, GT-free cells only "
              "(agreement %.4f):" % rec["grade_agreement_on_gt_free"])
        for line in EM.confusion_table(rec).splitlines():
            print("      " + line)
    print("\n-> %s" % out_csv)


if __name__ == "__main__":
    main()

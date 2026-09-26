#!/usr/bin/env python3
"""Score dumped 3D maps with the human-like traversability model (DESIGN_traversability.md).

    python scripts/entry/run_traverse.py --gt results/synth_corridor915f/gt_voxel.npz \
        --map <map3d>.npz:tag ... --entry=-2.35,-19.55 --out <csv> --fig <png>

GT and maps go through the SAME function with the SAME TravCfg; the only
difference is the 3D grid. No floor flattening: the model has no global floor.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.entry import clean3d, traverse as TV
from covor.entry.config import EntryCfg
from run_entry_metrics import load_map


def run_one(lab, ijk, res, entry, snap):
    lab3, _ = clean3d.clean(np.asarray(lab, np.uint8), EntryCfg(res=res))
    return TV.traverse(lab3, TV.TravCfg(res=res), entry, ijk, snap_m=snap)


def figure(results, ijk, res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    n = len(results)
    f, ax = plt.subplots(1, n, figsize=(3.2 * n, 13), sharey=True)
    ax = np.atleast_1d(ax)
    cm = ListedColormap(["#e8e8e8", "#9a7fb8", "#2fb86b", "#f2c14e", "#e8590c", "#1c7ed6"])
    for a, (tag, t) in zip(ax, results):
        img = np.zeros(t["reach"].shape)
        img[t["feasible"]] = 1                              # can stand, not reached
        R = t["reach"]
        img[R & (t["posture"] == TV.STAND)] = 2
        img[R & (t["posture"] == TV.STOOP)] = 3
        img[R & (t["posture"] == TV.CRAWL)] = 4
        img[R & (t["width"] == TV.SIDEWAYS)] = 5
        ext = (ijk[0] * res, (ijk[0] + img.shape[0]) * res,
               ijk[1] * res, (ijk[1] + img.shape[1]) * res)
        a.imshow(img.T, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=5,
                 interpolation="nearest")
        if t["entry"] is not None:
            e = t["entry"]
            a.plot((e[0] + ijk[0] + .5) * res, (e[1] + ijk[1] + .5) * res, "o",
                   ms=9, mfc="#d6336c", mec="w")
        a.set_title(tag, fontsize=10)
        a.set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    import matplotlib.patches as mp
    f.legend(handles=[mp.Patch(color=c, label=l) for c, l in zip(
        ["#9a7fb8", "#2fb86b", "#f2c14e", "#e8590c", "#1c7ed6"],
        ["can stand, not reached", "reached standing", "reached stooping",
         "reached crawling", "reached sideways"])], loc="lower center", ncol=5)
    f.suptitle(title)
    f.tight_layout(rect=(0, 0.04, 1, 0.97))
    f.savefig(path, dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--entry", required=True)
    ap.add_argument("--snap", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--title", default="human-like traversability")
    args = ap.parse_args()
    entry = tuple(float(v) for v in args.entry.split(","))

    lg, ijk, _, _, _, res = load_map(args.gt)
    t0 = time.time()
    tg = run_one(lg, ijk, res, entry, args.snap)
    print("GT: reach %d cols (stand %d, stoop %d, crawl %d, sideways %d), feasible %d, %.0fs"
          % (tg["reach"].sum(), (tg["reach"] & (tg["posture"] == TV.STAND)).sum(),
             (tg["reach"] & (tg["posture"] == TV.STOOP)).sum(),
             (tg["reach"] & (tg["posture"] == TV.CRAWL)).sum(),
             (tg["reach"] & (tg["width"] == TV.SIDEWAYS)).sum(),
             tg["feasible"].sum(), time.time() - t0))
    rows, figs = [], [("GT", tg)]
    for spec in args.map:
        path, tag = spec.rsplit(":", 1)
        lab, ij, _, _, meta, r = load_map(path)
        assert np.array_equal(ij, ijk) and abs(r - res) < 1e-9
        tm = run_one(lab, ij, r, entry, args.snap)
        s = TV.score(tm, tg)
        s.update(tag=tag, map=os.path.basename(path),
                 entry_snapped_m=tm["entry_snapped_m"],
                 gt_reach=int(tg["reach"].sum()))
        rows.append(s)
        figs.append((tag, tm))
        print("%-14s reach recall %.3f (%d/%d) IoU %.3f false-reach %.4f (%d) | stoop %d crawl %d sideways %d | entry snap %s"
              % (tag, s["reach_recall"], s["n_reach_true"], s["n_gt_reach"], s["reach_iou"],
                 s["false_reach_rate"], s["n_false_reach"], s["n_map_stoop"],
                 s["n_map_crawl"], s["n_map_sideways"], s["entry_snapped_m"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if args.fig:
        figure(figs, ijk, res, args.fig, args.title)
        print("-> %s" % args.fig)
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()

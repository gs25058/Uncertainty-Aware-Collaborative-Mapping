#!/usr/bin/env python3
"""Bake human-like traversability results into the 3D viewer (web/traverse_3d.template.html).

    python scripts/entry/export_traverse_3d.py --gt results/synth_corridor915f/gt_voxel_fs.npz \
        --map <map3d>.npz:key:label:blurb ... --entry=-2.35,-19.55 \
        --meta web/traverse_3d_corridor915f.meta.json --out web/traverse_3d_corridor915f.html

Every class the viewer paints comes out of covor/entry/traverse.py for that
grid; nothing here decides anything. Per variant the payload carries:

  vox    occupied voxels (the cleaned 3D grid the model judged), 16-bit planes
  tiles  one tile per column that a body can stand on, drawn AT ITS SUPPORT
         HEIGHT (stairs and landings are where they are, not on one plane):
         class 1 stand, 2 stoop, 3 crawl, 4 sideways, 5 crossed unseen floor,
         6 can stand but not reached from the entry
  route  entry -> the farthest reached column, by steepest descent of the
         Dijkstra cost, with the posture at every point
  stats  reach areas per posture, and recall / IoU / false-reach against the
         GT run through the SAME model
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.entry import clean3d, traverse as TV
from covor.entry.config import EntryCfg, OCCUPIED
from run_entry_metrics import load_map

STAND_T, STOOP_T, CRAWL_T, SIDE_T, GAP_T, UNREACHED_T = 1, 2, 3, 4, 5, 6


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def u16(a):
    return b64(np.asarray(a).astype("<u2"))


def judge(lab, ijk, res, entry, snap):
    lab3, _ = clean3d.clean(np.asarray(lab, np.uint8), EntryCfg(res=res))
    cfg = TV.TravCfg(res=res)
    t = TV.traverse(lab3, cfg, entry, ijk, snap_m=snap)
    _, P, _ = TV.node_modes(lab3, cfg)
    return lab3, t, P


def tile_classes(t, P):
    """(cls (nx,ny) uint8, row (nx,ny) int16) -- 0 = no tile."""
    R = t["reach"]
    cls = np.zeros(R.shape, np.uint8)
    row = np.full(R.shape, -1, np.int16)
    feas = (P >= 0)
    # can stand, not reached: lowest feasible support
    fr = feas.any(axis=2) & ~R
    kk = np.argmax(feas, axis=2)
    cls[fr] = UNREACHED_T
    row[fr] = kk[fr]
    row[R] = t["level_row"][R]
    cls[R & (t["posture"] == TV.STAND)] = STAND_T
    cls[R & (t["posture"] == TV.STOOP)] = STOOP_T
    cls[R & (t["posture"] == TV.CRAWL)] = CRAWL_T
    cls[R & (t["width"] == TV.SIDEWAYS)] = SIDE_T
    cls[R & t["gap_only"]] = GAP_T
    return cls, row


def route(t):
    """Entry -> farthest reached column by steepest descent of the cost field."""
    if t["entry"] is None or not t["reach"].any():
        return []
    C = np.where(t["reach"], t["cost"], np.inf)
    goal = np.unravel_index(int(np.argmax(np.where(np.isfinite(C), C, -1))), C.shape)
    path = [goal]
    cur = goal
    start = tuple(t["entry"][:2])
    nx, ny = C.shape
    for _ in range(nx * ny):
        if cur == start or C[cur] == 0:
            break
        best, bc = None, C[cur]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = cur[0] + di, cur[1] + dj
                if (di or dj) and 0 <= a < nx and 0 <= b < ny and C[a, b] < bc:
                    best, bc = (a, b), C[a, b]
        if best is None:
            break
        cur = best
        path.append(cur)
    return path[::-1]


def variant(key, label, blurb, lab, ijk, res, entry, snap, tg, is_ref):
    lab3, t, P = judge(lab, ijk, res, entry, snap)
    cls, row = tile_classes(t, P)
    ii, jj = np.nonzero(cls)
    oi, oj, ok = np.nonzero(lab3 == OCCUPIED)
    a = res * res
    R = t["reach"]
    reached_rows = t["level_row"][R]
    floor_row = int(np.median(reached_rows)) if R.any() else int(np.median(row[cls > 0]))
    path = route(t)
    pts = [[round(float((i + ijk[0] + .5) * res), 3), round(float((j + ijk[1] + .5) * res), 3),
            round(float((t["level_row"][i, j] + 1 + ijk[2]) * res), 3), int(cls[i, j])]
           for i, j in path]
    seg = np.array([np.hypot(pts[n + 1][0] - pts[n][0], pts[n + 1][1] - pts[n][1])
                    for n in range(len(pts) - 1)]) if len(pts) > 1 else np.zeros(0)
    by = {c: float(seg[[pts[n + 1][3] == c for n in range(len(seg))]].sum()) if len(seg) else 0.0
          for c in (STAND_T, STOOP_T, CRAWL_T, SIDE_T, GAP_T)}
    out = dict(
        key=key, label=label, blurb=blurb, is_reference=is_ref,
        floor_z=round(float((floor_row + 1 + ijk[2]) * res), 3),
        n_occ=int(len(oi)),
        vox=dict(i=u16(oi), j=u16(oj), k=u16(ok), bits=16),
        tiles=dict(i=u16(ii), j=u16(jj), k=u16(row[ii, jj]), c=b64(cls[ii, jj])),
        n_tiles=int(len(ii)),
        area_reach=float(R.sum() * a),
        area=dict(stand=float((cls == STAND_T).sum() * a), stoop=float((cls == STOOP_T).sum() * a),
                  crawl=float((cls == CRAWL_T).sum() * a), side=float((cls == SIDE_T).sum() * a),
                  gap=float((cls == GAP_T).sum() * a), unreached=float((cls == UNREACHED_T).sum() * a)),
        entry_xy=None if t["entry"] is None else
        [round(float((t["entry"][0] + ijk[0] + .5) * res), 3),
         round(float((t["entry"][1] + ijk[1] + .5) * res), 3),
         round(float((t["entry"][2] + 1 + ijk[2]) * res), 3)],
        entry_snapped_m=t["entry_snapped_m"],
        route=pts, route_len=float(seg.sum()),
        route_by={k: round(v, 2) for k, v in zip(("stand", "stoop", "crawl", "side", "gap"),
                                                 by.values())},
        recall=None, iou=None, false_reach=None, n_false_reach=None,
    )
    if tg is not None:
        s = TV.score(t, tg)
        out.update(recall=round(s["reach_recall"], 4), iou=round(s["reach_iou"], 4),
                   false_reach=round(s["false_reach_rate"], 4), n_false_reach=s["n_false_reach"])
    return out, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-label", default="GT (스캔 메시)")
    ap.add_argument("--gt-blurb", default="")
    ap.add_argument("--map", action="append", default=[], help="path:key:label:blurb")
    ap.add_argument("--entry", required=True)
    ap.add_argument("--snap", type=float, default=1.5)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--template", default="web/traverse_3d.template.html")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    entry = tuple(float(v) for v in args.entry.split(","))

    lg, ijk, _, _, _, res = load_map(args.gt)
    gv, tg = variant("gt", args.gt_label, args.gt_blurb, lg, ijk, res, entry, args.snap,
                     None, True)
    variants = [gv]
    for spec in args.map:
        parts = spec.split(":", 3)
        path, key = parts[0], parts[1]
        label = parts[2] if len(parts) > 2 else key
        blurb = parts[3] if len(parts) > 3 else ""
        lab, ij, _, _, _, r = load_map(path)
        if not np.array_equal(np.asarray(ij), np.asarray(ijk)) or abs(r - res) > 1e-9:
            raise SystemExit("%s is not on the GT grid" % path)
        v, _ = variant(key, label, blurb, lab, ij, r, entry, args.snap, tg, False)
        variants.append(v)

    known = (np.asarray(lg) != 0).any(axis=2)
    ki, kj = np.nonzero(known)
    roi = [float((ki.min() + ijk[0]) * res), float((ki.max() + 1 + ijk[0]) * res),
           float((kj.min() + ijk[1]) * res), float((kj.max() + 1 + ijk[1]) * res)]
    tcfg = TV.TravCfg(res=res)
    data = dict(
        variants=variants, roi=roi,
        grid=dict(nx=int(lg.shape[0]), ny=int(lg.shape[1]), nz=int(lg.shape[2]), res=res,
                  origin=[float(ijk[0] * res), float(ijk[1] * res), float(ijk[2] * res)]),
        cfg=dict(w=tcfg.w, w_side=tcfg.w_side, h_stand=tcfg.h_stand, h_stoop=tcfg.h_stoop,
                 h_crawl=tcfg.h_crawl, step=tcfg.step, gap=tcfg.gap, res=res))
    if args.meta:
        data["meta"] = json.load(open(args.meta, encoding="utf-8"))
    tpl = open(args.template, encoding="utf-8").read()
    if "/*__DATA__*/null" not in tpl:
        raise SystemExit("template has no /*__DATA__*/null placeholder")
    html = tpl.replace("/*__DATA__*/null",
                       json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    for v in variants:
        print("  %-10s reach %6.1f m2 (stand %.1f stoop %.1f crawl %.1f side %.1f gap %.1f) | "
              "route %.1f m | recall %s IoU %s false-reach %s"
              % (v["key"], v["area_reach"], v["area"]["stand"], v["area"]["stoop"],
                 v["area"]["crawl"], v["area"]["side"], v["area"]["gap"], v["route_len"],
                 v["recall"], v["iou"], v["false_reach"]))
    print("-> %s (%.0f kB)" % (args.out, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()

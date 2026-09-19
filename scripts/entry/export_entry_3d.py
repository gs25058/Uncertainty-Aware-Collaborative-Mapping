#!/usr/bin/env python3
"""Bake entry maps into the standalone 3D viewer (web/entry_map_3d.template.html).

    python scripts/entry/export_entry_3d.py --gt results/synth_room909/gt_voxel.npz \
        --map <dump>.npz:C:"blurb" ... --out web/entry_map_3d_room909.html

WHY 3D AT ALL, when DESIGN §5's deliverable is a floor plan. Because the plan
hides the one thing that decides whether it means anything: how far the occupied
evidence is smeared in z. A map that spreads a single storey over metres of
height still projects to a plausible-looking 2D band -- the band rule takes ANY
occupied cell, so vertical smear reads as more wall, not as a broken map. The
viewer draws the voxels inside the body band in one colour and everything above
and below in another, so that smear is visible rather than inferred.

The idea and the template are ported from uacm.zip (uacm/viz3d + scripts/
export_3d.py). The scene is theirs; the payload and the statistics are this
pipeline's, because this map has a real voxel ground truth and uacm's arena
metrics do not apply to it.

NOTHING HERE DECIDES ANYTHING. Every class it paints comes out of
covor/entry/'s grid. The viewer is a drawing of entry_grid.npz, exactly as
render.py is.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.entry import clean3d, inflate as IN, metrics as EM, route as RT
from covor.entry.config import (EntryCfg, UNKNOWN, FREE, OCCUPIED,
                                BLOCKED, NARROW, WALK)
from build_entry_map import build_entry_grid

# the template's PLAN_RGBA indices; class 3 (drone-only) is unused here because
# this project's drone and human radii are both 0.35 m
UNSEEN, WALL, TOO_TIGHT, DRONE_ONLY, SQUEEZE, STRANDED, REACH, WALKABLE = range(8)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def load_map(path):
    z = np.load(path, allow_pickle=False)
    lab = (z["labels"] if "labels" in z.files
           else clean3d.labels_from_masks(z["M_occ"], z["M_free"]))
    return (lab, z["ijk_min"], float(np.asarray(z["res"]).ravel()[0]),
            z["sigma_xy"] if "sigma_xy" in z.files else None,
            z["n_obs"] if "n_obs" in z.files else None,
            json.loads(str(z["meta"])) if "meta" in z.files else {})


def roi_from_gt(lab_gt, ijk, res):
    """The scanned room: the bounding box of columns the GT has an opinion about.

    uacm used the surveyed UWB arena. There is no arena here, but there is
    something better -- the GT knows which columns it observed -- and it plays
    the same role: free floor the map carves OUTSIDE the scanned room looks
    identical to floor inside it unless the boundary is drawn.
    """
    known = (np.asarray(lab_gt) != UNKNOWN).any(axis=2)
    ii, jj = np.nonzero(known)
    return [float((ii.min() + ijk[0]) * res), float((ii.max() + 1 + ijk[0]) * res),
            float((jj.min() + ijk[1]) * res), float((jj.max() + 1 + ijk[1]) * res)]


def plan_image(g, cfg, band):
    """The 8-class plan the viewer colours, all of it read off the grid."""
    lab = np.asarray(g["%s_label" % band])
    wc = np.asarray(g["%s_width_class" % band])
    P = np.asarray(g["%s_passable" % band], bool)
    R = np.asarray(g["%s_reachable" % band], bool)
    free = lab == FREE
    swept = IN.swept_workspace(R, cfg) & free      # clipped: see metrics docstring
    plan = np.full(lab.shape, UNSEEN, np.uint8)
    plan[free & (wc == BLOCKED)] = TOO_TIGHT
    # SQUEEZE is the ADVISORY class -- narrow enough that a body has to turn, and
    # not offered as passable. A narrow-grade cell that IS passable (clearance in
    # [w/2, 0.45)) is somewhere a rescuer may stand, so it belongs to the offered
    # set and is painted as such. The 2D figure colours by width GRADE and this
    # one by OFFER STATUS: two views of the same grid, neither changing a label.
    plan[free & (wc == NARROW) & ~P] = SQUEEZE
    plan[P & ~R] = STRANDED
    plan[swept] = REACH
    plan[R] = WALKABLE
    plan[lab == OCCUPIED] = WALL                   # structure always wins
    return plan, swept


def variant(key, label, blurb, lab3d, ijk, res, sigma, nobs, meta, cfg, band,
            gt_grid, roi, entry_xy=None):
    g, info = build_entry_grid(lab3d, cfg, sigma_xy=sigma, n_obs=nobs,
                               entry_xy=entry_xy, ijk_min=ijk, k_sigma=0.0)
    plan, swept = plan_image(g, cfg, band)
    lab = np.asarray(g["%s_label" % band])
    P = np.asarray(g["%s_passable" % band], bool)
    R = np.asarray(g["%s_reachable" % band], bool)
    clear = np.asarray(g["%s_clearance" % band], float)
    wc = np.asarray(g["%s_width_class" % band])
    a = res ** 2

    occ3 = np.asarray(lab3d) == OCCUPIED
    ii, jj, kk = np.nonzero(occ3)
    ks = np.nonzero(occ3.sum(axis=(0, 1)))[0]
    zspan = [float((ks.min() + ijk[2]) * res), float((ks.max() + 1 + ijk[2]) * res)]

    inroi = np.zeros(lab.shape, bool)
    i0 = int(round(roi[0] / res)) - ijk[0]; i1 = int(round(roi[1] / res)) - ijk[0]
    j0 = int(round(roi[2] / res)) - ijk[1]; j1 = int(round(roi[3] / res)) - ijk[1]
    inroi[max(i0, 0):i1, max(j0, 0):j1] = True

    cc, n_isl = ndimage.label(R, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    isl = np.bincount(cc.ravel())[1:] if n_isl else np.array([0])
    stranded = P & ~R
    sc, n_pockets = ndimage.label(stranded)
    pockets = np.bincount(sc.ravel())[1:] if n_pockets else np.array([0])

    start = tuple(int(v) for v in np.asarray(g["entry_ij"]))
    d = RT.geodesic(P, start)
    depth = float(np.max(d[R])) * res if R.any() else None
    rt = RT.plan(g, band, cfg)
    path = next((r["cells"] for r in rt["routes"] if r.get("reachable")), [])
    route = [[round(float((i + ijk[0] + 0.5) * res), 3),
              round(float((j + ijk[1] + 0.5) * res), 3)] for i, j in path]

    out = dict(
        key=key, label=label, blurb=blurb, is_reference=(gt_grid is None),
        floor_z=round(float((info["floor_row"] + ijk[2]) * res), 2),
        z_span=[round(v, 2) for v in zspan],
        z_span_m=round(zspan[1] - zspan[0], 2),
        n_occ=int(occ3.sum()),
        area_free=float(((lab == FREE).sum()) * a),
        area_passable_reach=float(R.sum() * a),
        area_swept=float(swept.sum() * a),
        area_swept_roi=float((swept & inroi).sum() * a),
        swept_outside_roi=float((swept & ~inroi).sum() * a),
        roi_m2=float(inroi.sum() * a),
        area_unreach=float(stranded.sum() * a),
        unreach_pockets=int(n_pockets),
        unreach_largest=float(pockets.max() * a) if n_pockets else 0.0,
        area_narrow=float(((lab == FREE) & (wc == NARROW)).sum() * a),
        islands=int(n_isl),
        largest=float(isl.max() * a) if n_isl else 0.0,
        reach_depth=(None if depth is None else round(depth, 2)),
        med_clearance=round(float(np.median(clear[lab == FREE])), 2)
        if (lab == FREE).any() else 0.0,
        unknown_frac=round(100.0 * float((lab == UNKNOWN).mean()), 1),
        entry_placed=True,
        entry_xy=[round(float(v), 2) for v in np.asarray(g["entry_xy"])],
        route=route,
        vox=dict(i=b64(ii.astype(np.uint8)), j=b64(jj.astype(np.uint8)),
                 k=b64(kk.astype(np.uint8))),
        plan=b64(plan.T),                          # (ny, nx) for the texture
    )
    if gt_grid is None:
        out.update(iou_cspace=None, iou_workspace=None, recall=None)
    else:
        rec = EM.score(g, gt_grid, band)
        rec.update(EM.workspace_scores(g, gt_grid, band, cfg))
        out.update(iou_cspace=round(rec["reachable_iou"], 3),
                   iou_workspace=round(rec["reachable_iou_workspace"], 3),
                   recall=round(rec["passable_recall"], 3))
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--map", action="append", default=[],
                    help="path:key:label:blurb  (repeatable)")
    ap.add_argument("--out", default="web/entry_map_3d_room909.html")
    ap.add_argument("--template", default="web/entry_map_3d.template.html")
    ap.add_argument("--band", default="walk")
    ap.add_argument("--w", type=float, default=0.70)
    args = ap.parse_args()

    cfg = EntryCfg(w=args.w)
    zg = np.load(args.gt, allow_pickle=False)
    lab_gt, ijk = zg["labels"], zg["ijk_min"]
    res = float(np.asarray(zg["res"]).ravel()[0])
    roi = roi_from_gt(lab_gt, ijk, res)

    gt_v, gt_grid = variant(
        "gt", "GT (mesh voxel)", "메시에서 직접 복셀화한 정답. 지도가 아니라 "
        "기준이다 — 같은 covor/entry 파이프라인을 같은 설정으로 통과시켰고, "
        "다른 조건과의 차이는 입력 3D 격자뿐이다.",
        lab_gt, ijk, res, None, None, {}, cfg, args.band, None, roi)
    variants = [gt_v]
    # the GT's own entry point, so every variant is entered at the same door
    door = tuple(float(v) for v in gt_grid["entry_xy"])

    for spec in args.map:
        parts = spec.split(":")
        path, key = parts[0], (parts[1] if len(parts) > 1 else "m%d" % len(variants))
        label = parts[2] if len(parts) > 2 else key
        blurb = parts[3] if len(parts) > 3 else ""
        lab, ijk_m, res_m, sig, nobs, meta = load_map(path)
        if not np.array_equal(np.asarray(ijk_m), np.asarray(ijk)) or res_m != res:
            raise SystemExit("%s is on a different grid than the GT" % path)
        try:
            v, _ = variant(key, label, blurb, lab, ijk_m, res_m, sig, nobs, meta,
                           cfg, args.band, gt_grid, roi, entry_xy=door)
        except ValueError as e:
            print("  %-10s the GT door is unusable (%s) -- using its own" % (key, e))
            v, _ = variant(key, label, blurb, lab, ijk_m, res_m, sig, nobs, meta,
                           cfg, args.band, gt_grid, roi)
        variants.append(v)

    zs = [v["z_span"] for v in variants]
    data = dict(
        variants=variants, entry_xy=list(door), roi=roi,
        z_scale=[round(min(z[0] for z in zs) - 0.5, 1),
                 round(max(z[1] for z in zs) + 0.5, 1)],
        grid=dict(nx=int(lab_gt.shape[0]), ny=int(lab_gt.shape[1]),
                  nz=int(lab_gt.shape[2]), res=res,
                  origin=[float(ijk[0] * res), float(ijk[1] * res),
                          float(ijk[2] * res)]),
        cfg=dict(body_lo=cfg.z_min, body_hi=cfg.H_walk, w=cfg.w,
                 k_sigma=0.0, res=cfg.res, clear_walk=cfg.clear_walk,
                 clear_narrow=cfg.clear_narrow))

    tpl = open(args.template, encoding="utf-8").read()
    if "/*__DATA__*/null" not in tpl:
        raise SystemExit("template has no /*__DATA__*/null placeholder")
    html = tpl.replace("/*__DATA__*/null",
                       json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    for v in variants:
        print("  %-10s floor %+.2f  z span %.2f m  swept %5.2f m2 "
              "(room %5.2f / %.1f, outside %4.2f)  islands %d  depth %s  "
              "IoU ws %s / cs %s"
              % (v["key"], v["floor_z"], v["z_span_m"], v["area_swept"],
                 v["area_swept_roi"], v["roi_m2"], v["swept_outside_roi"],
                 v["islands"], v["reach_depth"], v["iou_workspace"],
                 v["iou_cspace"]))
    print("-> %s (%.0f kB)" % (args.out, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()

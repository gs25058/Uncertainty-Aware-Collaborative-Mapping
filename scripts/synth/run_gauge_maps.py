#!/usr/bin/env python3
"""Part 3 step 5: map-stage results under the G1 gauge (PREREG_synth_gauge.md).

    python scripts/synth/run_gauge_maps.py --name room909 --seed 0 \
        --conds C_3drone --coverage ii_all_cams --scoring absolute --stride 8

One invocation = one (seed, condition, coverage, scoring) unit = three arms from
a single pass over the frames. Sequential and resumable on purpose: this sandbox
kills every form of detached worker, so a unit has to fit inside one foreground
call, and ``--resume`` skips units already in the csv.

SCORING (§2.3). ``absolute`` scores the map where it lands, which is what a
consumer of the map sees. ``s_align`` first fits ONE 4-DoF (yaw + xyz) transform
per condition, from the fused ifo001 camera track to the GT one, and applies it
to EVERY robot's poses before integration -- removing the map's global gauge
freedom while deliberately leaving inter-robot misregistration in, because that
is the thing the ranges are supposed to fix. Both are reported; neither alone.
"""
import argparse
import csv
import fcntl
import os
import sys
import time

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
from covor import data as D
from covor.occupancy import OccCfg
from covor.synth import dataset as DS, mapping as MP, mesh_gt as MG, metrics as ME
from covor.synth.config import SynthCfg, ROBOTS
from fuse_synth import fuse, CONDITIONS
from run_experiment import ARMS, COVERAGE, FIELDS as BASE_FIELDS
from run_gauge_pose import ARMS as COND_ARMS

FIELDS = BASE_FIELDS + ["gauge_init", "scoring", "uwb_sigma"]


def obs_mask(cfg, scene, gt, lab, ijk_min, res, key, cams, stride, node_stride):
    """Observability domain, built once per (coverage, stride) from GT poses."""
    p = os.path.join(cfg.outdir(), "obs_mask_%s_s%d.npz" % (key, stride))
    if os.path.exists(p):
        return np.load(p)["obs"]
    poses, mates = MP.gt_pose_source(cfg, gt, stride_nodes=node_stride)
    bs, _ = MP.build_map(cfg, scene, cams, poses, {"o": OccCfg(resolution=res)},
                         mode="ideal", stride=stride, record="o",
                         teammate_pos=mates)
    obs = ME._dense(MP.touched_index(bs["o"], res), lab.shape, ijk_min)
    np.savez_compressed(p, obs=obs)
    print("  built obs mask %s stride %d: %d cells" % (key, stride, obs.sum()),
          flush=True)
    return obs


def s_align_transform(poses, gt):
    """One 4-DoF transform per condition, fitted on ifo001's camera track."""
    from covor.fusion import umeyama_yaw
    P = poses["ifo001"]
    Tg, _ = DS.gt_camera_poses("ifo001", P["t"], gt)
    R, t = umeyama_yaw(P["T"][:, :3, 3], Tg[:, :3, 3])
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = t
    return A


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def done_units(path):
    if not os.path.exists(path):
        return set()
    n = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = (int(r["seed"]), r["condition"], r["coverage"], r["scoring"])
            n[k] = n.get(k, 0) + 1
    return {k for k, v in n.items() if v >= len(ARMS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--conds", default="C_3drone")
    ap.add_argument("--coverage", default="ii_all_cams", choices=list(COVERAGE))
    ap.add_argument("--scoring", default="absolute", choices=["absolute", "s_align"])
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--node-stride", type=int, default=2)
    args = ap.parse_args()
    out = os.path.join(SynthCfg(name=args.name).outdir(), "results_synth_gauge.csv")
    have = done_units(out)

    for seed in [int(s) for s in args.seeds.split(",")]:
        cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=seed)
        DS.install_paths(cfg)
        base_root = cfg.dataroot()
        lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
        gt = DS.load_gt_traj(cfg)
        scene = MG.raycasting_scene(DS.world_mesh(cfg))
        cams = COVERAGE[args.coverage]
        obs = obs_mask(cfg, scene, gt, lab, ijk_min, res, args.coverage, cams,
                       args.stride, args.node_stride)
        for name in args.conds.split(","):
            key = (seed, name, args.coverage, args.scoring)
            if key in have:
                print("  skip (done)", key, flush=True)
                continue
            ckey, ginit, noisy = COND_ARMS[name]
            D.DATA = base_root + ("_noise" if noisy else "")
            D._MOCAP_CACHE.clear()
            t0 = time.time()
            poses, gstats = fuse(cfg.seq, CONDITIONS[ckey], gauge_init=ginit)
            if args.scoring == "s_align":
                A = s_align_transform(poses, gt)
                poses = {r: dict(v, T=np.einsum("ij,njk->nik", A, v["T"]),
                                 p_body=v["p_body"] @ A[:3, :3].T + A[:3, 3])
                         for r, v in poses.items()}
            ates, trs = {}, []
            for r in ROBOTS:
                Tg, _ = DS.gt_camera_poses(r, poses[r]["t"], gt)
                ates[r] = ME.ate(poses[r]["T"][:, :3, 3], Tg[:, :3, 3])["ate_rmse"]
                trs.append(np.sqrt(poses[r]["tr"]))
            pooled = float(np.sqrt(np.mean([v ** 2 for v in ates.values()])))
            trs = np.concatenate(trs)
            mates = {r: (poses[r]["t"], poses[r]["p_body"]) for r in ROBOTS}
            arms = {k: OccCfg(resolution=res, **v) for k, v in ARMS.items()}
            bs, st = MP.build_map(cfg, scene, cams, poses, arms, mode="ideal",
                                  stride=args.stride, teammate_pos=mates)
            dt = time.time() - t0
            for arm, b in bs.items():
                M_occ, M_free = ME.map_masks(b, res, lab, ijk_min)
                s = ME.score(M_occ, M_free, lab, ijk_min, res, obs)
                append(out, dict(
                    s, seed=seed, condition=name, coverage=args.coverage,
                    arm=arm, res=res, stride=args.stride, gauge_init=ginit,
                    scoring=args.scoring,
                    uwb_sigma=cfg.uwb_sigma * (10.0 if noisy else 1.0),
                    ate_rmse_pooled=pooled, coverage_frac=s["coverage"],
                    sqrt_tr_sigma_median=float(np.median(trs)),
                    sqrt_tr_sigma_mean=float(trs.mean()),
                    n_frames=int(sum(st["frames"].values())),
                    n_points=int(st["points"]),
                    n_inter_range=int(gstats["n_inter_range"]),
                    ate_ifo001=ates["ifo001"], ate_ifo002=ates["ifo002"],
                    ate_ifo003=ates["ifo003"], build_s=dt))
                print("    %-9s %-13s %-9s %-11s P=%.3f R=%.3f IoU@1=%.3f "
                      "ff=%.3f occ=%d unk=%d"
                      % (name, args.coverage, args.scoring, arm, s["precision"],
                         s["recall"], s["iou_1vox"], s["false_free_rate"],
                         s["n_occupied"], s["n_unknown"]), flush=True)
            print("  done %s seed=%d (%.0fs)" % (name, seed, dt), flush=True)
        D.DATA = base_root
        D._MOCAP_CACHE.clear()


if __name__ == "__main__":
    main()

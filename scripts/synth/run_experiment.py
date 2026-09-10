#!/usr/bin/env python3
"""Part D: the 1 / 2 / 3-drone experiment on the synthetic room.

    python scripts/synth/run_experiment.py --name room909 --seed 0 --cond C_3drone
    python scripts/synth/run_experiment.py --name room909 --all      # every cell

Design (task brief):
  conditions   A_1drone / B_2drone / C_3drone, set with Cfg.inter_pairs
               (NOT --drones -- appendix A-13)
  coverage     (i)  ifo001's camera only, poses swapped per condition -- coverage
                    is held FIXED so any change is the pose, not the viewpoints
               (ii) all three cameras -- includes the collaborative coverage gain
  arms         uniform / depth-only / full
  seeds        one dataset per seed (VIO + UWB noise redrawn; trajectories fixed)

Every metric the brief lists is written for every cell, on one row, so none can
be quoted without the others. Rows are appended and flushed immediately, under
an flock so parallel workers can share the file.
"""
import argparse
import csv
import fcntl
import json
import os
import sys
import time

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
from covor.occupancy import OccCfg
from covor.synth import dataset as DS, mapping as MP, mesh_gt as MG, metrics as ME
from covor.synth.config import SynthCfg, ROBOTS
from fuse_synth import fuse, CONDITIONS

# The three weighting arms. `uniform` is the standard-OctoMap negative control:
# if a weighted arm cannot beat it, the weighting does nothing.
ARMS = {
    "uniform":    dict(weighted=False),
    "depth_only": dict(weighted=True, use_w_pose=False),
    "full":       dict(weighted=True, use_w_pose=True),
}
COVERAGE = {"i_ifo001_only": ["ifo001"], "ii_all_cams": list(ROBOTS)}

FIELDS = [
    "seed", "condition", "coverage", "arm", "res", "stride",
    "precision", "recall", "iou", "iou_1vox", "precision_1vox", "recall_1vox",
    "false_free_rate", "n_free", "n_occupied", "occ_over_gt", "n_unknown",
    "coverage_frac", "ate_rmse_pooled", "sqrt_tr_sigma_median",
    "precision_tol0", "precision_tol1", "precision_tol2",
    "recall_tol0", "recall_tol1", "recall_tol2",
    "false_free_tol0", "false_free_tol1", "false_free_tol2",
    "n_domain", "n_gt_occ_domain", "n_gt_free_domain", "tp", "fp", "fn",
    "n_false_free", "n_frames", "n_points", "n_inter_range", "ate_ifo001",
    "ate_ifo002", "ate_ifo003", "sqrt_tr_sigma_mean", "build_s",
]


def obs_mask_path(cfg, key):
    return os.path.join(cfg.outdir(), "obs_mask_%s.npz" % key)


def observability(cfg, scene, gt, lab, ijk_min, res, key, cams, stride, node_stride):
    """Cells any ideal beam from the GT poses touches, for one coverage mode.

    Built ONCE from ground truth and cached: every condition inside a coverage
    mode is then scored on identical cells, so a difference between conditions
    cannot come from a difference in which cells were scored.
    """
    p = obs_mask_path(cfg, key)
    if os.path.exists(p):
        return np.load(p)["obs"]
    poses, mates = MP.gt_pose_source(cfg, gt, stride_nodes=node_stride)
    bs, _ = MP.build_map(cfg, scene, cams, poses, {"o": OccCfg()}, mode="ideal",
                         stride=stride, record="o", teammate_pos=mates)
    obs = ME._dense(MP.touched_index(bs["o"], res), lab.shape, ijk_min)
    np.savez_compressed(p, obs=obs)
    return obs


def append_row(path, row):
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


def run_cell(cfg, seed, cond, stride, node_stride, csv_path, mode="ideal",
             coverages=None):
    """One (seed, condition): fuse once, then map under every coverage x arm."""
    DS.install_paths(cfg)
    lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
    gt = DS.load_gt_traj(cfg)
    scene = MG.raycasting_scene(DS.world_mesh(cfg))

    t0 = time.time()
    poses, gstats = fuse(cfg.seq, CONDITIONS[cond])
    print("  fusion %s: %d inter ranges, %.0fs" % (cond, gstats["n_inter_range"],
                                                   time.time() - t0))
    # trajectory accuracy + reported uncertainty
    ates, trs = {}, []
    for r in ROBOTS:
        p_est = poses[r]["T"][:, :3, 3]        # camera position
        # compare like with like: GT camera positions at the same instants
        T_gt, _ = DS.gt_camera_poses(r, poses[r]["t"], gt)
        ates[r] = ME.ate(p_est, T_gt[:, :3, 3])["ate_rmse"]
        trs.append(np.sqrt(poses[r]["tr"]))
    pooled = float(np.sqrt(np.mean([v ** 2 for v in ates.values()])))
    trs = np.concatenate(trs)

    mates = {r: (poses[r]["t"], poses[r]["p_body"]) for r in ROBOTS}
    for ckey, cams in COVERAGE.items():
        if coverages and ckey not in coverages:
            continue
        obs = observability(cfg, scene, gt, lab, ijk_min, res, ckey, cams,
                            stride, node_stride)
        t1 = time.time()
        arms = {k: OccCfg(resolution=res, **v) for k, v in ARMS.items()}
        bs, st = MP.build_map(cfg, scene, cams, poses, arms, mode=mode,
                              stride=stride, teammate_pos=mates)
        dt = time.time() - t1
        for arm, b in bs.items():
            M_occ, M_free = ME.map_masks(b, res, lab, ijk_min)
            s = ME.score(M_occ, M_free, lab, ijk_min, res, obs)
            row = dict(s, seed=seed, condition=cond, coverage=ckey, arm=arm,
                       res=res, stride=stride, ate_rmse_pooled=pooled,
                       coverage_frac=s["coverage"],
                       sqrt_tr_sigma_median=float(np.median(trs)),
                       sqrt_tr_sigma_mean=float(trs.mean()),
                       n_frames=int(sum(st["frames"].values())),
                       n_points=int(st["points"]),
                       n_inter_range=int(gstats["n_inter_range"]),
                       ate_ifo001=ates["ifo001"], ate_ifo002=ates["ifo002"],
                       ate_ifo003=ates["ifo003"], build_s=dt)
            append_row(csv_path, row)
            print("    %-14s %-11s P=%.3f R=%.3f IoU@1=%.3f ff=%.3f "
                  "free=%d occ=%d cov=%.3f ATE=%.3f"
                  % (ckey, arm, s["precision"], s["recall"], s["iou_1vox"],
                     s["false_free_rate"], s["n_free"], s["n_occupied"],
                     s["coverage"], pooled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="C_3drone", choices=list(CONDITIONS))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--node-stride", type=int, default=2)
    ap.add_argument("--mode", default="ideal", choices=["ideal", "sgbm"])
    ap.add_argument("--coverage", default=None,
                    help="restrict to one coverage mode (i_ifo001_only | ii_all_cams)")
    ap.add_argument("--regen", action="store_true",
                    help="regenerate the dataset for this seed first")
    args = ap.parse_args()

    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=args.seed)
    if args.regen:
        print("=== regenerating dataset, seed %d ===" % args.seed)
        DS.install_paths(cfg)
        DS.build(cfg)
    csv_path = os.path.join(cfg.outdir(), "results_synth.csv")
    print("=== seed %d | %s | stride %d | -> %s ===" % (args.seed, args.cond,
                                                        args.stride, csv_path))
    run_cell(cfg, args.seed, args.cond, args.stride, args.node_stride, csv_path,
             mode=args.mode,
             coverages=[args.coverage] if args.coverage else None)


if __name__ == "__main__":
    main()

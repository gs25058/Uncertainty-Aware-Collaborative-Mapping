#!/usr/bin/env python3
"""Per-robot fusion accuracy and uncertainty, per (seed, condition).

    python scripts/synth/pose_stats.py --name room909 --seeds 0,1,2

results_synth.csv carries POOLED sqrt(tr Sigma) and pooled ATE, and the pooling
hides the effect: measured on seed 0, ifo001's tr(Sigma) falls 0.165 -> 0.039 ->
0.027 across A -> B -> C while ifo002's RISES 0.160 -> 0.226 in condition B.

That rise is not a collaboration effect, it is the gauge confound
RESULTS_SUMMARY §7.4-2 already documents: with gauge_mode="component", condition
A has three connected components and therefore THREE tight gauge priors -- one
absolute anchor per robot -- while condition B merges robots 0 and 1 into one
component with a single prior, so robot 1 loses its own anchor and is held only
through the ranges. Condition A is therefore not a clean "no collaboration"
baseline for Sigma, and the per-robot table is the only way to see that.

It also separates three different things that "pose error" can mean, because
they disagree here and only one of them is what the map sees:

  ate_rmse     error after the 4-DoF alignment -- the trajectory's SHAPE
  abs_rms      error with NO alignment -- where the robot actually is in the
               world frame. THE MAP IS SCORED IN ABSOLUTE COORDINATES, so this
               is the one that moves cells.
  rel_pos_rms  node-to-node relative error -- local consistency, which is what
               blurs or sharpens a surface.

Writes pose_stats.csv next to results_synth.csv.
"""
import argparse
import csv
import os
import sys
import time

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
from covor.synth import dataset as DS, metrics as ME
from covor.synth.config import SynthCfg, ROBOTS
from fuse_synth import fuse, CONDITIONS

FIELDS = ["seed", "condition", "robot", "ate_rmse", "ate_median", "ate_max",
          "abs_rms", "abs_median", "rel_pos_rms", "rel_rot_rms",
          "tr_sigma_median", "tr_sigma_mean", "sqrt_tr_median", "n_nodes",
          "n_inter_range", "n_gauge_prior", "gauge_on"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=seed)
        DS.install_paths(cfg)
        gt = DS.load_gt_traj(cfg)
        for cond, pairs in CONDITIONS.items():
            t0 = time.time()
            poses, st = fuse(cfg.seq, pairs)
            for r in ROBOTS:
                T = poses[r]["T"]
                T_gt, _ = DS.gt_camera_poses(r, poses[r]["t"], gt)
                a = ME.ate(T[:, :3, 3], T_gt[:, :3, 3])
                e_abs = np.linalg.norm(T[:, :3, 3] - T_gt[:, :3, 3], axis=1)
                dP = np.einsum("nji,nj->ni", T[:-1, :3, :3],
                               T[1:, :3, 3] - T[:-1, :3, 3])
                dPg = np.einsum("nji,nj->ni", T_gt[:-1, :3, :3],
                                T_gt[1:, :3, 3] - T_gt[:-1, :3, 3])
                dR = np.einsum("nji,njk->nik", T[:-1, :3, :3], T[1:, :3, :3])
                dRg = np.einsum("nji,njk->nik", T_gt[:-1, :3, :3], T_gt[1:, :3, :3])
                er = Rot.from_matrix(np.einsum("nji,njk->nik", dRg, dR)).as_rotvec()
                tr = poses[r]["tr"]
                rows.append(dict(
                    seed=seed, condition=cond, robot=r, **a,
                    abs_rms=float(np.sqrt((e_abs ** 2).mean())),
                    abs_median=float(np.median(e_abs)),
                    rel_pos_rms=float(np.sqrt(((dP - dPg) ** 2).sum(1).mean())),
                    rel_rot_rms=float(np.sqrt((er ** 2).sum(1).mean())),
                    tr_sigma_median=float(np.median(tr)),
                    tr_sigma_mean=float(tr.mean()),
                    sqrt_tr_median=float(np.sqrt(np.median(tr))),
                    n_nodes=len(tr), n_inter_range=st["n_inter_range"],
                    n_gauge_prior=st["n_gauge_prior"],
                    gauge_on="|".join(str(g) for g in st["gauge_on"])))
                print("  seed %d %-9s %-7s ATE %.4f  ABS %.4f  rel %.5f  "
                      "tr %.4f  gauge %s"
                      % (seed, cond, r, a["ate_rmse"],
                         np.sqrt((e_abs ** 2).mean()),
                         np.sqrt(((dP - dPg) ** 2).sum(1).mean()),
                         np.median(tr), st["gauge_on"]), flush=True)
            print("  (%.0fs, %d inter)" % (time.time() - t0, st["n_inter_range"]),
                  flush=True)
    p = os.path.join(SynthCfg(name=args.name).outdir(), "pose_stats.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", p)


if __name__ == "__main__":
    main()

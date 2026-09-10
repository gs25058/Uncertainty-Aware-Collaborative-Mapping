#!/usr/bin/env python3
"""Part 3 step 4: pose-stage results for every condition (S1 + negative control).

    python scripts/synth/run_gauge_pose.py --name room909 --seeds 0,1,2

Conditions (PREREG_synth_gauge.md §2.2):
    A_1drone / B_2drone / C_3drone   gauge_init="first_pose" (G1)
    A_ref                            gauge_init="umeyama"  -- reference, NOT a baseline
    C_noise                          G1, UWB sigma x10     -- negative control

``C_noise`` corrupts the OBSERVATION, not just the weight: its uwb_range.csv is
regenerated with sigma = 0.5 m from the same rng stream, so the residual really
is ten times larger and the csv's own ``std`` column carries 0.5 -- which is what
``max(std, range_sigma_floor)`` then hands the range factor. Changing only the
sigma the graph is told would leave the measurements accurate and test nothing.

Writes results/synth_<name>/pose_stats_gauge.csv (append + flush).
"""
import argparse
import csv
import fcntl
import os
import shutil
import sys
import time

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
from covor import data as D
from covor.synth import dataset as DS, metrics as ME, uwb as UWB
from covor.synth.config import SynthCfg, ROBOTS
from fuse_synth import fuse, CONDITIONS

NOISE_FACTOR = 10.0

# name -> (inter_pairs key, gauge_init, noisy_data)
ARMS = {
    "A_1drone": ("A_1drone", "first_pose", False),
    "B_2drone": ("B_2drone", "first_pose", False),
    "C_3drone": ("C_3drone", "first_pose", False),
    "A_ref":    ("A_1drone", "umeyama",    False),
    "C_noise":  ("C_3drone", "first_pose", True),
}

FIELDS = ["seed", "condition", "gauge_init", "uwb_sigma", "robot",
          "abs_rms", "abs_median", "ate_rmse", "rel_pos_rms", "rel_rot_rms",
          "tr_sigma_median", "sqrt_tr_median", "n_nodes", "n_inter_range",
          "n_gauge_prior", "gauge_on"]


def noisy_dataroot(cfg):
    """Sibling data root whose UWB ranges carry sigma x NOISE_FACTOR.

    Everything else -- trajectories, vio.csv, mocap.csv, timeshift -- is the
    file the normal condition uses, so the ONLY difference is the range noise.
    """
    dst = cfg.dataroot() + "_noise"
    if not os.path.exists(dst):
        shutil.copytree(cfg.dataroot(), dst)
    noisy = SynthCfg(name=cfg.name, seq=cfg.seq, seed=cfg.seed,
                     uwb_sigma=cfg.uwb_sigma * NOISE_FACTOR)
    gt = DS.load_gt_traj(cfg)
    df = UWB.generate({r: gt[r] for r in ROBOTS}, noisy)
    for rob in ROBOTS:
        m = (df["_a"] == rob) | (df["_b"] == rob)
        df.loc[m, ["timestamp", "from_id", "to_id", "range", "std",
                   "gt_range", "bias"]].to_csv(
            os.path.join(dst, cfg.seq, rob, "uwb_range.csv"), index=False)
    resid = float(np.sqrt(((df["range"] - df["gt_range"]) ** 2).mean()))
    return dst, resid


def append(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    out = os.path.join(SynthCfg(name=args.name).outdir(), "pose_stats_gauge.csv")
    if os.path.exists(out):
        os.remove(out)
    for seed in [int(s) for s in args.seeds.split(",")]:
        cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=seed)
        DS.install_paths(cfg)
        gt = DS.load_gt_traj(cfg)
        base_root = cfg.dataroot()
        nz_root, nz_resid = noisy_dataroot(cfg)
        print("seed %d: C_noise range residual rms %.4f m (normal 0.05)"
              % (seed, nz_resid), flush=True)
        for name, (ckey, ginit, noisy) in ARMS.items():
            D.DATA = nz_root if noisy else base_root
            D._MOCAP_CACHE.clear()
            t0 = time.time()
            poses, st = fuse(cfg.seq, CONDITIONS[ckey], gauge_init=ginit)
            rows = []
            for r in ROBOTS:
                T = poses[r]["T"]
                Tg, _ = DS.gt_camera_poses(r, poses[r]["t"], gt)
                e = np.linalg.norm(T[:, :3, 3] - Tg[:, :3, 3], axis=1)
                dP = np.einsum("nji,nj->ni", T[:-1, :3, :3],
                               T[1:, :3, 3] - T[:-1, :3, 3])
                dPg = np.einsum("nji,nj->ni", Tg[:-1, :3, :3],
                                Tg[1:, :3, 3] - Tg[:-1, :3, 3])
                dR = np.einsum("nji,njk->nik", T[:-1, :3, :3], T[1:, :3, :3])
                dRg = np.einsum("nji,njk->nik", Tg[:-1, :3, :3], Tg[1:, :3, :3])
                er = Rot.from_matrix(np.einsum("nji,njk->nik", dRg, dR)).as_rotvec()
                tr = poses[r]["tr"]
                rows.append(dict(
                    seed=seed, condition=name, gauge_init=ginit,
                    uwb_sigma=cfg.uwb_sigma * (NOISE_FACTOR if noisy else 1.0),
                    robot=r, abs_rms=float(np.sqrt((e ** 2).mean())),
                    abs_median=float(np.median(e)),
                    ate_rmse=ME.ate(T[:, :3, 3], Tg[:, :3, 3])["ate_rmse"],
                    rel_pos_rms=float(np.sqrt(((dP - dPg) ** 2).sum(1).mean())),
                    rel_rot_rms=float(np.sqrt((er ** 2).sum(1).mean())),
                    tr_sigma_median=float(np.median(tr)),
                    sqrt_tr_median=float(np.sqrt(np.median(tr))),
                    n_nodes=len(tr), n_inter_range=st["n_inter_range"],
                    n_gauge_prior=st["n_gauge_prior"],
                    gauge_on="|".join(str(g) for g in st["gauge_on"])))
            append(out, rows)
            print("  %-9s gauge=%s priors=%d inter=%5d | abs_rms %s | (%.0fs)"
                  % (name, ginit, st["n_gauge_prior"], st["n_inter_range"],
                     " ".join("%.3f" % r["abs_rms"] for r in rows),
                     time.time() - t0), flush=True)
        D.DATA = base_root
        D._MOCAP_CACHE.clear()
    print("wrote", out)


if __name__ == "__main__":
    main()

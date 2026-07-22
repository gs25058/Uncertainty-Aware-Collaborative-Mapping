#!/usr/bin/env python3
"""Loop-closure-OFF comparison (CoVOR-SLAM paper alignment).

The paper's premise: per-robot front-end is pure Visual Odometry (drifting, no
loop closing); UWB ranges -- not inter-agent visual loop closure -- correct the
drift. We re-ran ORB-SLAM3 with `loopClosing: 0`, so the current
`*_kf.txt` are the drifting VO; the `*_kf_LCon.txt` backups are the previous
loop-closed SLAM (much more accurate) for reference.

This script reports, per robot:
  - VO_LCon   : previous loop-closed baseline (Sim3), reference
  - VO_LCoff  : new drifting VO baseline (Sim3) -- what CoVOR must beat
  - fused ... : several UWB fusion variants on the LCoff VO (SE3, metric)

Run:  python scripts/compare_lc.py
Rows -> diagnosis_results.csv
"""
import sys
import time
import numpy as np
import gtsam

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts")
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E, factors as F, data as D
from diaglog import log_rows

SEQ = "default_3_zigzag_0"
VODIR = "/src/gs25058/cr_RNE/covor_slam/vo_output"


def kf_ate(path_suffix):
    """Sim3-aligned ATE of a raw VO keyframe file variant, per robot."""
    out = {}
    for robot in D.ROBOTS:
        path = f"{VODIR}/{SEQ}_{robot}_kf{path_suffix}.txt"
        try:
            df = __import__("pandas").read_csv(path, sep=r"\s+", header=None,
                names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"]).sort_values("t")
        except Exception:
            out[robot] = float("nan"); continue
        mocap = D.load_mocap(SEQ, robot)
        traj = df[["t", "x", "y", "z"]].to_numpy()
        out[robot] = E.ate_rmse(traj, mocap, with_scale=True)[0]
    return out


def fused(tag, cfg):
    cov = CoVOR(SEQ, cfg).build()
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(30); p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    t0 = time.time()
    res = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p).optimize()
    dt = time.time() - t0
    a, z = {}, {}
    for k, rb in enumerate(cov.robots):
        traj = np.array([[rb.t[i], *res.atPose3(F.X(k, i)).translation()]
                         for i in range(rb.n())])
        r, est, gt, _ = E.ate_rmse(traj, rb.mocap, with_scale=False)
        a[rb.name] = r; z[rb.name] = float(np.sqrt(np.mean((est[:, 2] - gt[:, 2]) ** 2)))
    print("%-26s ate[%.3f %.3f %.3f] z[%.3f %.3f %.3f] (%.0fs)" % (
        tag, a["ifo001"], a["ifo002"], a["ifo003"],
        z["ifo001"], z["ifo002"], z["ifo003"], dt), flush=True)
    log_rows([dict(test="LCoff_cmp", config=tag, robot=n, metric="ate",
                   value=round(a[n], 4), time_s=round(dt, 1), note="z=%.3f" % z[n])
              for n in a])
    return a


def main():
    print("=== VO baselines (Sim3-aligned) ===", flush=True)
    lcon = kf_ate("_LCon"); lcoff = kf_ate("")
    for r in D.ROBOTS:
        print("  %-7s  VO_LCon(ref) %.3f   VO_LCoff %.3f   drift x%.1f" % (
            r, lcon[r], lcoff[r],
            lcoff[r] / lcon[r] if lcon[r] else float("nan")), flush=True)
    log_rows([dict(test="LCoff_cmp", config="VO_LCon", robot=r, metric="ate",
                   value=round(lcon[r], 4), note="Sim3 ref") for r in D.ROBOTS]
             + [dict(test="LCoff_cmp", config="VO_LCoff", robot=r, metric="ate",
                     value=round(lcoff[r], 4), note="Sim3 drifting baseline")
                for r in D.ROBOTS])

    print("\n=== UWB fusion on the LCoff (drifting) VO ===", flush=True)
    # paper-aligned: antenna pre-compensated (no moment arm), inter+anchor ranges
    fused("fused_inter_only", Cfg(use_anchor=False, bias_mode="const",
          range_sigma_floor=0.3, huber_k=1.0, prior_every=8))
    fused("fused_full_UWB", Cfg(bias_mode="const", range_sigma_floor=0.3,
          huber_k=1.0, prior_every=8))
    fused("fused_full_UWB+height", Cfg(bias_mode="const", range_sigma_floor=0.3,
          huber_k=1.0, use_height=True, prior_every=8))
    # observation-model-improved (moment arm + height, bias off)
    fused("fused_ma+height", Cfg(use_moment_arm=True, bias_mode="off",
          range_sigma_floor=0.3, huber_k=1.0, use_height=True, prior_every=8))


if __name__ == "__main__":
    main()

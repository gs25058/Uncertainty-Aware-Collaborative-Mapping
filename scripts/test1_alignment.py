#!/usr/bin/env python3
"""Test 1 (H1): unify the evaluation alignment. Compute ATE for BOTH the mono-VO
and the fused trajectory under BOTH SE(3) and Sim(3) alignment, using the *same*
covor.evaluate.align/ate_rmse code for each. Fills a 2x2 (robot x alignment) so
we can see whether the "degradation" is real shape error or an artefact of VO
getting a free Sim(3) scale fit while fused is scored with SE(3).

Run:  python scripts/test1_alignment.py
"""
import sys
import time
import numpy as np
import gtsam

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts")
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E, factors as F
from diaglog import log_rows

SEQ = "default_3_zigzag_0"
# "previous best" fused config: UWB-only (height off), const bias, floor/huber/gauge
CFG = Cfg(bias_mode="const", range_sigma_floor=0.5, huber_k=0.3, prior_every=8)


def vo_traj(rb):
    return np.array([[rb.t[i], *rb.poses_vo[i].translation()] for i in range(rb.n())])


def fused_traj(cov, res, k):
    rb = cov.robots[k]
    return np.array([[rb.t[i], *res.atPose3(F.X(k, i)).translation()]
                     for i in range(rb.n())])


def main():
    cov = CoVOR(SEQ, CFG).build()
    print("fused config: UWB-only, const bias, floor0.5 huber0.3 pe8 | "
          "inter=%d anchor=%d (all-3-robot, post load_ranges fix)"
          % (cov.stats["n_inter_range"], cov.stats["n_anchor_range"]), flush=True)
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(30); p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    t0 = time.time()
    res = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p).optimize()
    dt = time.time() - t0

    rows = []
    print("\n%-8s | %-19s | %-19s" % ("robot", "SE(3) align", "Sim(3) align"), flush=True)
    print("%-8s | %8s %8s | %8s %8s" % ("", "VO", "fused", "VO", "fused"), flush=True)
    print("-" * 54, flush=True)
    for k, rb in enumerate(cov.robots):
        if rb.n() < 5:
            continue
        vt, ft = vo_traj(rb), fused_traj(cov, res, k)
        # identical align function applied to both trajectories
        vo_se3, *_ = E.ate_rmse(vt, rb.mocap, with_scale=False)
        vo_sim3, *_ = E.ate_rmse(vt, rb.mocap, with_scale=True)
        fu_se3, *_ = E.ate_rmse(ft, rb.mocap, with_scale=False)
        fu_sim3, *_ = E.ate_rmse(ft, rb.mocap, with_scale=True)
        print("%-8s | %8.3f %8.3f | %8.3f %8.3f"
              % (rb.name, vo_se3, fu_se3, vo_sim3, fu_sim3), flush=True)
        for metric, val in [("VO_SE3", vo_se3), ("fused_SE3", fu_se3),
                            ("VO_Sim3", vo_sim3), ("fused_Sim3", fu_sim3)]:
            rows.append(dict(test="T1_align", config="UWBonly_f0.5_h0.3_pe8",
                             robot=rb.name, metric=metric, value=round(val, 4),
                             time_s=round(dt, 1),
                             note="same align fn for VO and fused"))
    log_rows(rows)

    # decision helper: compare VO vs fused under the FAIR (Sim3) alignment
    print("\nFair comparison (both Sim(3)-aligned):", flush=True)
    for k, rb in enumerate(cov.robots):
        if rb.n() < 5:
            continue
        vo_sim3, *_ = E.ate_rmse(vo_traj(rb), rb.mocap, with_scale=True)
        fu_sim3, *_ = E.ate_rmse(fused_traj(cov, res, k), rb.mocap, with_scale=True)
        verdict = "fused WORSE" if fu_sim3 > vo_sim3 else "fused >= VO"
        print("  %-8s VO_Sim3 %.3f  fused_Sim3 %.3f  -> %s"
              % (rb.name, vo_sim3, fu_sim3, verdict), flush=True)


if __name__ == "__main__":
    main()

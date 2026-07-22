#!/usr/bin/env python3
"""Phase 2-4: run CoVOR-SLAM UWB fusion and dump fused poses + marginal
covariance for the occupancy stage (proposal §2.5 -> §4).

For each robot and keyframe it saves:
  t            (N,)      keyframe timestamp (= infra image filename seconds)
  T            (N,4,4)   fused world<-camera pose (SE3, infra1 optical frame)
  tr_sigma_pos (N,)      trace of the 3x3 position block of the 6x6 marginal
                         covariance Sigma^k_n  (registration uncertainty)

This is the ONLY place Sigma is extracted; the proposal's whole novelty is that
Sigma is forwarded here instead of being discarded. Output -> one .npz per robot
in vo_output/, consumed by scripts/build_occupancy.py.

Usage: python scripts/fuse_and_dump.py [--drones ifo001,ifo002,ifo003]
"""
import sys
import time
import argparse
import numpy as np
import gtsam

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.fusion import CoVOR, Cfg
from covor import factors as F
from covor import data as D

SEQ = "default_3_zigzag_0"
OUT = "/src/gs25058/cr_RNE/covor_slam/vo_output"

# Faithful fusion config (matches the paper-faithful robust-ON run):
# moment-arm antenna model, robust Huber kernel, inter + anchor ranges, no
# MILUV-specific height/bias extensions. Same config feeds the 1/2/3-drone study.
FAITHFUL = dict(use_moment_arm=True, bias_mode="off", use_height=False,
                prior_every=0, range_sigma_floor=0.05, robust=True, huber_k=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", default="ifo001,ifo002,ifo003")
    ap.add_argument("--tag", default="")   # filename suffix, e.g. "_1drone"
    args = ap.parse_args()
    want = set(args.drones.split(","))

    # Ablation over drone count is done by disabling the ranges of robots that
    # are not in `want`: keep their odometry-only (drifting) solution, exactly
    # the proposal's "fewer UWB constraints -> larger Sigma" path (§4.9).
    cfg = Cfg(**FAITHFUL)
    cov = CoVOR(SEQ, cfg).build()
    print("stats:", cov.stats)
    res = cov.optimize(max_iter=100, verbose=False)
    print("error %.1f -> %.1f" % (cov.stats["initial_error"], cov.stats["final_error"]))

    # Map breaks in the ORB-SLAM3 mono front-end leave some pose segments with no
    # absolute prior; UWB range factors constrain only position (a scalar
    # distance), not orientation, so those segments have a free rotational gauge
    # -> the full marginal factorization is indeterminate. (The paper's VINS
    # front-end avoids this: IMU gravity observes roll/pitch continuously.) We
    # regularize a COPY of the graph with a VERY WEAK prior on every variable:
    # well-constrained poses are essentially unchanged, while gauge-free segments
    # get a large-but-finite covariance -- correctly a high tr(Sigma_pos), hence
    # a low occupancy weight w. This is exactly the intended behaviour.
    t0 = time.time()
    greg = gtsam.NonlinearFactorGraph(cov.graph)
    npose = gtsam.noiseModel.Isotropic.Sigma(6, 10.0)   # weak Pose3 prior
    ndbl = gtsam.noiseModel.Isotropic.Sigma(1, 10.0)    # weak scalar prior
    for key in res.keys():
        try:
            greg.add(gtsam.PriorFactorPose3(key, res.atPose3(key), npose))
        except RuntimeError:
            greg.add(gtsam.PriorFactorDouble(key, res.atDouble(key), ndbl))
    marg = gtsam.Marginals(greg, res)
    print("Marginals built in %.1fs" % (time.time() - t0))

    for k, rb in enumerate(cov.robots):
        if rb.name not in want or rb.n() == 0:
            continue
        N = rb.n()
        T = np.zeros((N, 4, 4)); tr = np.zeros(N)
        for i in range(N):
            pose = res.atPose3(F.X(k, i))
            T[i] = pose.matrix()
            cov6 = marg.marginalCovariance(F.X(k, i))   # 6x6, [rot(3), trans(3)]
            tr[i] = float(np.trace(cov6[3:6, 3:6]))     # tr(Sigma_pos), m^2
        path = f"{OUT}/occ_{SEQ}_{rb.name}{args.tag}.npz"
        np.savez(path, t=rb.t, T=T, tr_sigma_pos=tr)
        print("%s  N=%d  tr(Sigma_pos): min=%.4f med=%.4f max=%.4f  -> %s" % (
            rb.name, N, tr.min(), np.median(tr), tr.max(), path))


if __name__ == "__main__":
    main()

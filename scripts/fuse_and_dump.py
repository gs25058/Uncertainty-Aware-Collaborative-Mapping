#!/usr/bin/env python3
"""Phase 2-4: run CoVOR-SLAM UWB fusion and dump fused poses + marginal
covariance for the occupancy stage (proposal §2.5 -> §4).

For each robot and node it saves:
  t            (N,)      node timestamp (MILUV relative seconds)
  T            (N,4,4)   fused world<-CAMERA pose (SE3, infra1 optical frame)
  tr_sigma_pos (N,)      trace of the 3x3 position block of the 6x6 marginal
                         covariance Sigma^k_n  (registration uncertainty)

This is the ONLY place Sigma is extracted; the proposal's whole novelty is that
Sigma is forwarded here instead of being discarded. Output -> one .npz per robot
in vo_output/, consumed by scripts/build_occupancy.py.

FRAME. The graph state is the BODY (IMU) pose T_wb -- that is what VINS estimates,
what the UWB tag moment arms are referenced to, and what mocap tracks. The
occupancy stage casts rays from the camera, so this script composes
T_wc = T_wb @ body_T_cam0 on the way out. Do not remove that step: the extrinsic is
a 119-121 deg rotation, so feeding body poses to the ray caster aims every depth
ray in the wrong direction (silently -- the map just comes out wrong).

Usage: python scripts/fuse_and_dump.py [--seq SEQ] [--drones ifo001,ifo002,ifo003]
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

OUT = "/src/gs25058/cr_RNE/covor_slam/vo_output"

# Faithful fusion config (matches the paper-faithful robust-ON run):
# moment-arm antenna model, robust Huber kernel, inter + anchor ranges, no
# MILUV-specific height/bias extensions. Same config feeds the 1/2/3-drone study.
FAITHFUL = dict(use_moment_arm=True, bias_mode="off", use_height=False,
                prior_every=0, range_sigma_floor=0.05, robust=True, huber_k=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="default_3_zigzag_0")
    ap.add_argument("--drones", default="ifo001,ifo002,ifo003")
    ap.add_argument("--tag", default="")   # filename suffix, e.g. "_1drone"
    ap.add_argument("--anchor-free", action="store_true",
                    help="anchor_robots=() -- proposal's target system (§4.9 condition C: "
                         "inter UWB only, no fixed anchors). Gauge handled by G1 "
                         "(one prior per ungrounded connected component).")
    args = ap.parse_args()
    want = set(args.drones.split(","))

    # NOTE: --drones selects which robots get WRITTEN OUT. It does not change the
    # graph -- every robot's ranges are always used. An earlier comment here
    # claimed this performed the proposal's §4.9 drone-count ablation; it does not,
    # and no such ablation exists yet. Cfg only has the global use_ranges /
    # use_anchor / use_inter switches, so running the §4.9 ladder needs a per-pair
    # Cfg.inter_pairs and a per-robot Cfg.anchor_robots first (design in
    # OCCUPANCY_PIPELINE.md, "§4.9 collaboration-gain experiment").
    extra = dict(anchor_robots=()) if args.anchor_free else {}
    cfg = Cfg(**FAITHFUL, **extra)
    cov = CoVOR(args.seq, cfg).build()
    print("stats:", cov.stats)
    res = cov.optimize(max_iter=100, verbose=False)
    print("error %.1f -> %.1f" % (cov.stats["initial_error"], cov.stats["final_error"]))

    # Marginals directly on the solved graph -- NO regularizer.
    #
    # The mono pipeline had to add a weak (sigma=10) prior to every variable before
    # this call: ORB-SLAM3 map breaks left whole pose segments with no absolute
    # constraint, and UWB ranges pin position only (a scalar distance), never
    # orientation, so those segments had a free rotational gauge and the
    # factorization was indeterminate. With the VINS front-end that precondition is
    # gone -- the odometry chain is unbroken (0 map breaks) and gravity observes
    # roll/pitch throughout, so every node is reachable from the first pose's prior.
    # Verified rather than assumed, on default_3_zigzag_0: build reports
    # n_weak_priors=0 (no node is left unconstrained) and this call factorizes in
    # 1.8 s giving tr(Sigma_pos) in 0.0012-0.0134 m^2. Those are plausible
    # registration uncertainties (sigma_pos ~ 3-12 cm), NOT the regularizer's
    # fingerprint -- under the old sigma=10 blanket a gauge-free node read ~300 m^2.
    # That is the point: tr(Sigma) now means something, so the occupancy weight
    # w = exp(-tr(Sigma)/alpha) responds to real uncertainty.
    #
    # If this ever throws IndeterminantLinearSystemException, do NOT paper over it
    # by reinstating the blanket prior -- it means some node lost its constraints,
    # and the cause is what needs finding.
    t0 = time.time()
    marg = gtsam.Marginals(cov.graph, res)
    print("Marginals built in %.1fs" % (time.time() - t0))

    for k, rb in enumerate(cov.robots):
        if rb.name not in want or rb.n() == 0:
            continue
        b_T_c = D.load_body_T_cam(rb.name, 0)
        N = rb.n()
        T = np.zeros((N, 4, 4)); tr = np.zeros(N); p_body = np.zeros((N, 3))
        for i in range(N):
            pose = res.atPose3(F.X(k, i))
            p_body[i] = pose.translation()        # airframe centre
            T[i] = pose.matrix() @ b_T_c          # world<-body -> world<-camera
            cov6 = marg.marginalCovariance(F.X(k, i))   # 6x6, [rot(3), trans(3)]
            tr[i] = float(np.trace(cov6[3:6, 3:6]))     # tr(Sigma_pos), m^2
        # tr(Sigma_pos) is reported at the body origin. The camera sits ~0.11 m
        # away on a rigid body, so its position uncertainty also picks up the
        # rotational block; that extra term is small next to the values above and
        # is left out deliberately rather than silently -- revisit if tr(Sigma_rot)
        # ever grows.
        # p_body is saved alongside T because the two are needed for different
        # things: T (camera) is where rays START, p_body is where the AIRFRAME is.
        # Teammate masking compares beam endpoints against the airframe centre
        # within dyn_radius = 0.35 m, so using the camera position there would be
        # 0.11 m (~31 % of that radius) off.
        path = f"{OUT}/occ_{args.seq}_{rb.name}{args.tag}.npz"
        np.savez(path, t=rb.t, T=T, tr_sigma_pos=tr, p_body=p_body)
        print("%s  N=%d  tr(Sigma_pos): min=%.4f med=%.4f max=%.4f  -> %s" % (
            rb.name, N, tr.min(), np.median(tr), tr.max(), path))


if __name__ == "__main__":
    main()

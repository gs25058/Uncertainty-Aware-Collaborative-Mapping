#!/usr/bin/env python3
"""Fusion + marginal covariance on the synthetic sequence (Part D, pose stage).

The synthetic twin of scripts/fuse_and_dump.py, with two differences and no
others:
  * ``covor.data``'s paths point at results/synth_<name>/ (dataset.install_paths);
  * the drone-count ladder is expressed with ``Cfg.inter_pairs``, NOT ``--drones``
    (appendix A-13: --drones only filters the OUTPUT, it does not change the
    graph, so it cannot produce a 1 vs 2 vs 3 drone comparison).

Fusion parameters are appendix B's corrected values verbatim, via FAITHFUL
below; anchor_robots=() because the proposal is anchor-free and the synthetic
UWB set contains no anchor ranges at all.
"""
import numpy as np

# Ladder: C(N,2) = 0 -> 1 -> 3 inter-agent pairs, i.e. 1 -> 2 -> 3 collaborating
# drones. Robot indices follow covor.data.ROBOTS.
CONDITIONS = {
    "A_1drone": (),
    "B_2drone": ((0, 1),),
    "C_3drone": ((0, 1), (0, 2), (1, 2)),
}

# RESULTS_SUMMARY.md appendix B (corrected 2026-09-03) -- the values the §4.9
# and §8 experiments actually used.
FAITHFUL = dict(use_moment_arm=True, bias_mode="off", use_height=False,
                prior_every=0, range_sigma_floor=0.05, robust=True, huber_k=1.0,
                gauge_mode="component", anchor_robots=())


def fuse(seq, inter_pairs, verbose=False):
    """Run the fusion and return {robot: dict(t, T (world<-camera), tr, sigma_pos,
    p_body)} plus the graph stats."""
    import gtsam
    from covor.fusion import CoVOR, Cfg
    from covor import factors as F
    from covor import data as D

    cfg = Cfg(**FAITHFUL, inter_pairs=inter_pairs)
    cov = CoVOR(seq, cfg).build()
    res = cov.optimize(max_iter=100, verbose=verbose)
    marg = gtsam.Marginals(cov.graph, res)
    out = {}
    for k, rb in enumerate(cov.robots):
        if rb.n() == 0:
            continue
        b_T_c = D.load_body_T_cam(rb.name, 0)
        N = rb.n()
        T = np.zeros((N, 4, 4)); tr = np.zeros(N)
        p_body = np.zeros((N, 3)); Sig = np.zeros((N, 3, 3))
        for i in range(N):
            pose = res.atPose3(F.X(k, i))
            p_body[i] = pose.translation()
            T[i] = pose.matrix() @ b_T_c        # world<-body -> world<-camera
            cov6 = marg.marginalCovariance(F.X(k, i))
            Sig[i] = cov6[3:6, 3:6]
            tr[i] = float(np.trace(Sig[i]))
        out[rb.name] = dict(t=rb.t, T=T, tr=tr, sig=Sig, p_body=p_body)
    return out, cov.stats

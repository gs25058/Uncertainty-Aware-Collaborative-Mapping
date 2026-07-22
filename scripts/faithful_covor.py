#!/usr/bin/env python3
"""Paper-faithful CoVOR-SLAM reproduction on default_3_zigzag_0.

Front-end: ORB-SLAM3 mono VO with loop DETECTION + map MERGING kept but loop
CORRECTION disabled (loopClosing:1, loopCorrection:0) -> continuous drifting VO,
matching the paper's "monocular VO" baseline (no intra-agent loop closure).

Faithful fusion config (paper alignment, no MILUV-specific extensions):
  - Gaussian weighted LS, NO robust kernel        (paper: relies on redundancy)
  - moment-arm antenna model                      (paper: antenna pre-compensated)
  - no height factor, no const/online range bias  (paper has neither)
  - prior only at each agent's first pose         (paper: phi^1_pri,k)
  - inter-agent and anchor range factors

We compare, per robot vs mocap GT: VO-only / VO+inter / VO+inter+anchor.
A second block repeats with the robust kernel ON (MILUV deviation) for contrast,
since MILUV UWB has a heavy tail (p99~0.9 m) the paper's outdoor data did not.

Run:  python scripts/faithful_covor.py
Rows -> diagnosis_results.csv
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


def vo_baseline(cov):
    out = {}
    for rb in cov.robots:
        vo = np.array([[rb.t[i], *rb.poses_vo[i].translation()] for i in range(rb.n())])
        out[rb.name] = E.ate_rmse(vo, rb.mocap, with_scale=True)[0] if rb.n() >= 5 else float("nan")
    return out


def run(tag, cfg):
    cov = CoVOR(SEQ, cfg).build()
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(40); p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    t0 = time.time()
    res = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p).optimize()
    dt = time.time() - t0
    a = {}
    for k, rb in enumerate(cov.robots):
        if rb.n() < 5:
            a[rb.name] = float("nan"); continue
        traj = np.array([[rb.t[i], *res.atPose3(F.X(k, i)).translation()]
                         for i in range(rb.n())])
        a[rb.name] = E.ate_rmse(traj, rb.mocap, with_scale=False)[0]
    print("%-30s ate[%.3f %.3f %.3f] (inter=%d anch=%d, %.0fs)" % (
        tag, a["ifo001"], a["ifo002"], a["ifo003"],
        cov.stats["n_inter_range"], cov.stats["n_anchor_range"], dt), flush=True)
    log_rows([dict(test="faithful_covor", config=tag, robot=n, metric="ate",
                   value=round(a[n], 4), time_s=round(dt, 1)) for n in a])
    return a


def block(label, robust):
    # faithful base: moment arm on, no height, no bias, prior at start only,
    # weight = per-measurement csv std (floor tiny), robust per-arg
    base = dict(use_moment_arm=True, bias_mode="off", use_height=False,
                prior_every=0, range_sigma_floor=0.05, robust=robust, huber_k=1.0)
    print("\n=== %s (robust=%s) ===" % (label, robust), flush=True)
    run("%s_VOonly" % label, Cfg(use_ranges=False, prior_every=0))
    run("%s_VO+inter" % label, Cfg(use_anchor=False, **base))
    run("%s_VO+inter+anchor" % label, Cfg(**base))


if __name__ == "__main__":
    cov0 = CoVOR(SEQ, Cfg()).build()
    vo = vo_baseline(cov0)
    print("VO baseline (drifting, Sim3): %.3f / %.3f / %.3f" % (
        vo["ifo001"], vo["ifo002"], vo["ifo003"]), flush=True)
    log_rows([dict(test="faithful_covor", config="VO_baseline_Sim3", robot=n,
                   metric="ate", value=round(vo[n], 4)) for n in vo])
    block("paper", robust=False)          # paper-faithful: no robust kernel
    block("miluv", robust=True)           # MILUV deviation: Huber on

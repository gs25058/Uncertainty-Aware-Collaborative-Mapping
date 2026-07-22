#!/usr/bin/env python3
"""Test 2 (H2) main + Test 3a: isolate observation-model defect from real UWB
quality by swapping the measurement (real range vs mocap gt_range) and the model
(body-centre distance vs moment-arm antenna distance).

  gt_range + body-centre : leaves only the lever-arm mismatch (~0.15 m std)
  gt_range + moment-arm  : perfect measurement + correct model -> should ~= VO
  real     + moment-arm  : correct model, real UWB noise (~0.22 m)
  real     + body-centre : the degraded baseline (const bias)

Reports per-robot 3D ATE (SE(3)) vs the VO baseline and the range-residual std at
the optimum. Rows appended to diagnosis_results.csv.

Run:  python scripts/test2_gtrange.py
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
VO = {"ifo001": 0.091, "ifo002": 0.129, "ifo003": 0.284}


def ate(cov, res):
    out = {}
    for k, rb in enumerate(cov.robots):
        traj = np.array([[rb.t[i], *res.atPose3(F.X(k, i)).translation()]
                         for i in range(rb.n())])
        out[rb.name] = E.ate_rmse(traj, rb.mocap, with_scale=False)[0]
    return out


def resid_std(cov, res):
    """std of range residuals at the optimum (body-centre model, real range)."""
    r = []
    for _, row in cov.ranges.iterrows():
        fk = D.robot_of_tag(int(row.from_id))
        if fk is None:
            continue
        ka = int(fk[-1]) - 1; ia = cov._assoc(ka, float(row.timestamp))
        if ia is None:
            continue
        pa = res.atPose3(F.X(ka, ia)).translation()
        if row["kind"] == "anchor":
            aid = int(row.to_id)
            if aid not in cov.anchors:
                continue
            r.append(np.linalg.norm(pa - cov.anchors[aid]) - float(row["range"]))
        else:
            tk = D.robot_of_tag(int(row.to_id))
            if tk is None:
                continue
            kb = int(tk[-1]) - 1; ib = cov._assoc(kb, float(row.timestamp))
            if ib is None:
                continue
            pb = res.atPose3(F.X(kb, ib)).translation()
            r.append(np.linalg.norm(pa - pb) - float(row["range"]))
    return float(np.std(r))


def run(tag, cfg):
    cov = CoVOR(SEQ, cfg).build()
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(30); p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    t0 = time.time()
    res = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p).optimize()
    dt = time.time() - t0
    a = ate(cov, res); rs = resid_std(cov, res)
    print("%-26s ate[001 %.3f  002 %.3f  003 %.3f]  residStd=%.3f  (%.0fs)"
          % (tag, a["ifo001"], a["ifo002"], a["ifo003"], rs, dt), flush=True)
    rows = [dict(test="T2_T3a", config=tag, robot=r, metric="ate_se3",
                 value=round(a[r], 4), time_s=round(dt, 1),
                 note="VO=%.3f residStd=%.3f" % (VO[r], rs)) for r in a]
    log_rows(rows)
    return a


if __name__ == "__main__":
    print("VO baseline: 0.091 / 0.129 / 0.284\n", flush=True)
    run("real_bodycentre_constbias",
        Cfg(prior_every=8, bias_mode="const", range_sigma_floor=0.5, huber_k=0.3))
    run("gtrange_bodycentre",
        Cfg(prior_every=8, use_gt_range=True, gt_range_sigma=0.05))
    run("gtrange_momentarm",
        Cfg(prior_every=8, use_gt_range=True, gt_range_sigma=0.05, use_moment_arm=True))
    run("real_momentarm_biasoff",
        Cfg(prior_every=8, use_moment_arm=True, bias_mode="off",
            range_sigma_floor=0.3, huber_k=1.0))

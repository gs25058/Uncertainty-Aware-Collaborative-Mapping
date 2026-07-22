#!/usr/bin/env python3
"""End-to-end fusion evaluation for one MILUV sequence, using the recommended
observation model from the zigzag diagnosis (moment arm + height, no const bias).

Reports per-robot 3D ATE (fused SE(3), VO Sim(3)), vertical z-RMSE, and the
anchor-vs-mocap rigid shift Delta (horizontal/vertical) so we can check whether
the ~0.25 m vertical anchor inconsistency seen on zigzag recurs here.

Run:  python scripts/eval_seq.py [sequence]
Rows appended to diagnosis_results.csv.
"""
import sys
import time
import numpy as np
import gtsam
from scipy.optimize import least_squares

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts")
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E, factors as F, data as D
from diaglog import log_rows

SEQ = sys.argv[1] if len(sys.argv) > 1 else "default_3_random_0"


def metrics(cov, res, with_scale):
    a, z = {}, {}
    for k, rb in enumerate(cov.robots):
        if rb.n() < 5:
            a[rb.name] = float("nan"); z[rb.name] = float("nan"); continue
        if with_scale:
            traj = np.array([[rb.t[i], *rb.poses_vo[i].translation()] for i in range(rb.n())])
        else:
            traj = np.array([[rb.t[i], *res.atPose3(F.X(k, i)).translation()]
                             for i in range(rb.n())])
        r, est, gt, _ = E.ate_rmse(traj, rb.mocap, with_scale=with_scale)
        a[rb.name] = r
        z[rb.name] = float(np.sqrt(np.mean((est[:, 2] - gt[:, 2]) ** 2)))
    return a, z


def run(tag, cfg, with_scale=False):
    cov = CoVOR(SEQ, cfg).build()
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(30); p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    t0 = time.time()
    res = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p).optimize()
    dt = time.time() - t0
    a, z = metrics(cov, res, with_scale)
    names = [rb.name for rb in cov.robots if rb.n() >= 5]
    print("%-26s ate[%s] z[%s] (%.0fs)" % (
        tag, " ".join("%.3f" % a[n] for n in names),
        " ".join("%.3f" % z[n] for n in names), dt), flush=True)
    log_rows([dict(test="rand_eval", config=tag, robot=n, metric="ate",
                   value=round(a[n], 4), time_s=round(dt, 1),
                   note="z=%.3f" % z[n]) for n in names])
    return a, cov, res


def rigid_delta(cov):
    print("\n=== anchor-vs-mocap rigid shift (bias-corrected 0.14 removed) ===", flush=True)
    per = {k: [] for k in range(3)}
    for _, r in cov.ranges.iterrows():
        if r["kind"] != "anchor":
            continue
        fk = D.robot_of_tag(int(r.from_id))
        if fk is None:
            continue
        ka = int(fk[-1]) - 1; ia = cov._assoc(ka, float(r.timestamp))
        if ia is None:
            continue
        aid = int(r.to_id)
        if aid not in cov.anchors:
            continue
        pgt = D.mocap_position_at(cov.robots[ka].mocap, cov.robots[ka].t[ia])
        per[ka].append((pgt, cov.anchors[aid], float(r["range"]) - 0.14))
    rows = []
    for k in range(3):
        if len(per[k]) < 5:
            continue
        P = np.array([o[0] for o in per[k]]); A = np.array([o[1] for o in per[k]])
        Z = np.array([o[2] for o in per[k]])
        sol = least_squares(lambda dx: np.linalg.norm(P + dx - A, axis=1) - Z, [0, 0, 0])
        dx = sol.x
        print("  %-7s n=%d Delta=[%+.3f %+.3f %+.3f] horiz=%.3f vert=%.3f" % (
            cov.robots[k].name, len(per[k]), dx[0], dx[1], dx[2],
            np.hypot(dx[0], dx[1]), abs(dx[2])), flush=True)
        rows.append(dict(test="rand_rigidDelta", config="anchors_vs_mocap",
                         robot=cov.robots[k].name, metric="vert_delta",
                         value=round(abs(dx[2]), 4),
                         note="horiz=%.3f" % np.hypot(dx[0], dx[1])))
    log_rows(rows)


if __name__ == "__main__":
    print("=== sequence: %s ===" % SEQ, flush=True)
    va, cov0, _ = run("VO_baseline", Cfg(), with_scale=True)  # VO Sim3
    run("odometry_only", Cfg(use_ranges=False, prior_every=8))
    run("VO+height", Cfg(use_ranges=False, use_height=True, prior_every=8))
    run("VO+UWB(ma,biasoff)",
        Cfg(use_moment_arm=True, bias_mode="off", range_sigma_floor=0.3,
            huber_k=1.0, prior_every=8))
    run("VO+UWB+height(ma)",
        Cfg(use_moment_arm=True, bias_mode="off", range_sigma_floor=0.3,
            huber_k=1.0, use_height=True, prior_every=8))
    rigid_delta(cov0)

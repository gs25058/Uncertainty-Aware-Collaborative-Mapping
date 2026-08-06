#!/usr/bin/env python3
"""LEGACY (Sim(3) era) -- Part C sweep: VO / VO+UWB / VO+height / VO+UWB+height,
reporting per-robot 3D ATE, vertical (z) RMSE, horizontal RMSE and the estimated
per-robot scale.

Reads the per-node scale variable, which the SE(3) transition removed. The height
study's conclusion is already recorded (HANDOFF pitfall 9: the 6 anchors all sit at
~1.7 m so vertical is unobservable from ranges; height factors fix zRMSE 0.30->0.05
but are outside the proposal's scope, hence OFF by default). Re-run at a0a5e07, or
port by dropping the scale column.

Run:  python scripts/sweep_height.py [sequence]
"""
import sys as _sys
_sys.exit(__doc__)
import os
import sys
import csv
import time
import itertools
import numpy as np
import gtsam

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E, factors as F

SEQ = sys.argv[1] if len(sys.argv) > 1 else "default_3_zigzag_0"
OUT = "/src/gs25058/cr_RNE/covor_slam/sweep_height_results.csv"
MAX_ITER = 30
ROBOTS = ["ifo001", "ifo002", "ifo003"]

FIELDS = (["seq", "tag", "use_ranges", "use_anchor", "use_inter", "use_height",
           "bias_mode", "range_sigma_floor", "huber_k", "height_sigma_floor",
           "prior_every", "err_init", "err_final"]
          + [f"ate_{r}" for r in ROBOTS] + ["ate_mean"]
          + [f"z_{r}" for r in ROBOTS] + [f"h_{r}" for r in ROBOTS]
          + [f"scale_{r}" for r in ROBOTS] + ["iters", "time_s", "note"])


def metrics(cov, values, with_scale):
    ate, zr, hr, sc = [], [], [], []
    for k, rb in enumerate(cov.robots):
        if rb.n() < 5:
            ate.append(np.nan); zr.append(np.nan); hr.append(np.nan); sc.append(np.nan)
            continue
        if with_scale:                              # VO baseline: own frame
            traj = np.array([[rb.t[i], *rb.poses_vo[i].translation()]
                             for i in range(rb.n())])
        else:                                       # fused: metric world
            traj = np.array([[rb.t[i], *values.atPose3(F.X(k, i)).translation()]
                             for i in range(rb.n())])
        r, est, gt, _ = E.ate_rmse(traj, rb.mocap, with_scale=with_scale)
        ate.append(r)
        zr.append(float(np.sqrt(np.mean((est[:, 2] - gt[:, 2]) ** 2))))
        hr.append(float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, 1)))))
        if with_scale:
            sc.append(np.nan)
        else:
            sc.append(float(np.mean([values.atDouble(F.Sc(k, i)) for i in range(rb.n())])))
    return ate, zr, hr, sc


def mkrow(tag, cfg, ate, zr, hr, sc, ei, ef, it, dt, note=""):
    d = dict(seq=SEQ, tag=tag, use_ranges=cfg.use_ranges, use_anchor=cfg.use_anchor,
             use_inter=cfg.use_inter, use_height=cfg.use_height, bias_mode=cfg.bias_mode,
             range_sigma_floor=cfg.range_sigma_floor, huber_k=cfg.huber_k,
             height_sigma_floor=cfg.height_sigma_floor, prior_every=cfg.prior_every,
             err_init=ei, err_final=ef, ate_mean=round(float(np.nanmean(ate)), 4),
             iters=it, time_s=round(dt, 1), note=note)
    for i, r in enumerate(ROBOTS):
        d[f"ate_{r}"] = round(ate[i], 4); d[f"z_{r}"] = round(zr[i], 4)
        d[f"h_{r}"] = round(hr[i], 4)
        d[f"scale_{r}"] = "" if np.isnan(sc[i]) else round(sc[i], 4)
    return d


def run(tag, cfg, note=""):
    cov = CoVOR(SEQ, cfg).build()
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(MAX_ITER)
    p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    o = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p)
    t0 = time.time(); res = o.optimize(); dt = time.time() - t0
    ate, zr, hr, sc = metrics(cov, res, with_scale=False)
    return mkrow(tag, cfg, ate, zr, hr, sc,
                 round(cov.graph.error(cov.values), 1), round(cov.graph.error(res), 1),
                 o.iterations(), dt, note), ate


def main():
    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader(); f.flush()

    def emit(row):
        w.writerow(row); f.flush(); os.fsync(f.fileno())

    # VO baseline (Sim3-aligned, own frame)
    cov0 = CoVOR(SEQ, Cfg()).build()
    ate, zr, hr, sc = metrics(cov0, cov0.values, with_scale=True)
    emit(mkrow("mono_vo_baseline", Cfg(), ate, zr, hr, sc, "-", "-", "-", 0.0,
               "Sim3-aligned mono VO"))
    print("VO baseline ate=", [round(x, 3) for x in ate], flush=True)

    # ablation references
    refs = [
        ("odometry_only", Cfg(use_ranges=False, prior_every=8)),
        ("height_only",   Cfg(use_ranges=False, use_height=True, prior_every=8)),
        ("uwb_only",      Cfg(prior_every=8)),
        ("uwb_plus_height", Cfg(use_height=True, prior_every=8)),
    ]
    for tag, cfg in refs:
        row, ate = run(tag, cfg, "ablation")
        emit(row)
        print("%-18s ate=[%.3f %.3f %.3f] mean=%s z=[%.3f %.3f %.3f]" % (
            tag, ate[0], ate[1], ate[2], row["ate_mean"],
            row["z_ifo001"], row["z_ifo002"], row["z_ifo003"]), flush=True)

    # grid: UWB+height, vary anchor/inter, floor, huber, height sigma
    anchors = [(True, True), (False, True), (True, False)]   # (use_anchor, use_inter)
    floors = [0.3, 0.5, 0.8]
    hubers = [0.3, 0.5]
    hsig = [0.08, 0.12]
    combos = list(itertools.product(anchors, floors, hubers, hsig))
    print("running %d grid configs..." % len(combos), flush=True)
    for i, ((ua, ui), fl, hk, hs) in enumerate(combos):
        cfg = Cfg(use_height=True, use_anchor=ua, use_inter=ui, range_sigma_floor=fl,
                  huber_k=hk, height_sigma_floor=hs, prior_every=8)
        tag = "g%02d_%s%s_f%s_h%s_hs%s" % (i, "A" if ua else "-", "I" if ui else "-",
                                           fl, hk, hs)
        try:
            row, ate = run(tag, cfg)
            emit(row)
            print("[%2d/%d] %-22s ate=[%.3f %.3f %.3f] mean=%s %ss" % (
                i + 1, len(combos), tag, ate[0], ate[1], ate[2], row["ate_mean"],
                row["time_s"]), flush=True)
        except Exception as e:
            print("[%2d/%d] %-22s FAILED %s" % (i + 1, len(combos), tag, e), flush=True)
    f.close()
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()

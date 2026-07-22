#!/usr/bin/env python3
"""Parameter sweep for the UWB-VO fusion, scoring every config by ATE RMSE vs
mocap ground truth (the ONLY success criterion -- graph-error reduction is not).

Grid (task spec):  range_sigma_floor x huber_delta x bias-correction on/off.
We additionally sweep ``prior_every`` (gauge anchoring), because the diagnostics
showed the failure mode is a near-rigid trajectory drift that odometry cannot
resist -- so a reference block with prior_every in {0, 8} is included.

Each finished config is appended to sweep_results.csv and flushed+fsync'd
immediately, so a killed/backgrounded run never loses completed rows.

Run:  python scripts/sweep.py [sequence]
"""
import os
import sys
import csv
import time
import itertools
import numpy as np
import gtsam

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E, factors as F, data as D

SEQ = sys.argv[1] if len(sys.argv) > 1 else "default_3_zigzag_0"
OUT = "/src/gs25058/cr_RNE/covor_slam/sweep_results.csv"
MAX_ITER = 30

FIELDS = ["seq", "tag", "bias_mode", "range_sigma_floor", "huber_k", "prior_every",
          "robust", "err_init", "err_final",
          "ate_ifo001", "ate_ifo002", "ate_ifo003", "ate_mean",
          "iters", "time_s", "note"]


def ate_all(cov, values):
    out = []
    for k, rb in enumerate(cov.robots):
        if rb.n() < 5:
            out.append(float("nan")); continue
        traj = np.array([[rb.t[i], *values.atPose3(F.X(k, i)).translation()]
                         for i in range(rb.n())])
        out.append(E.ate_rmse(traj, rb.mocap, with_scale=False)[0])
    return out


def vo_baseline(cov):
    out = []
    for rb in cov.robots:
        if rb.n() < 5:
            out.append(float("nan")); continue
        vo = np.array([[rb.t[i], *rb.poses_vo[i].translation()] for i in range(rb.n())])
        out.append(E.ate_rmse(vo, rb.mocap, with_scale=True)[0])
    return out


def writer_open():
    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader(); f.flush()
    return f, w


def emit(f, w, row):
    w.writerow(row)
    f.flush(); os.fsync(f.fileno())            # never lose a completed config


def run_cfg(tag, cfg, note=""):
    cov = CoVOR(SEQ, cfg).build()
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(MAX_ITER)
    p.setRelativeErrorTol(1e-4); p.setAbsoluteErrorTol(1e-2)
    o = gtsam.LevenbergMarquardtOptimizer(cov.graph, cov.values, p)
    t0 = time.time(); res = o.optimize(); dt = time.time() - t0
    ate = ate_all(cov, res)
    return dict(
        seq=SEQ, tag=tag, bias_mode=cfg.bias_mode,
        range_sigma_floor=cfg.range_sigma_floor, huber_k=cfg.huber_k,
        prior_every=cfg.prior_every, robust=cfg.robust,
        err_init=round(cov.graph.error(cov.values), 2),
        err_final=round(cov.graph.error(res), 2),
        ate_ifo001=round(ate[0], 4), ate_ifo002=round(ate[1], 4),
        ate_ifo003=round(ate[2], 4), ate_mean=round(float(np.nanmean(ate)), 4),
        iters=o.iterations(), time_s=round(dt, 1), note=note), ate


def main():
    f, w = writer_open()
    # reference rows -------------------------------------------------------
    cov0 = CoVOR(SEQ, Cfg()).build()
    vo = vo_baseline(cov0)
    emit(f, w, dict(seq=SEQ, tag="mono_vo_baseline", bias_mode="-",
                    range_sigma_floor="-", huber_k="-", prior_every="-", robust="-",
                    err_init="-", err_final="-",
                    ate_ifo001=round(vo[0], 4), ate_ifo002=round(vo[1], 4),
                    ate_ifo003=round(vo[2], 4), ate_mean=round(float(np.nanmean(vo)), 4),
                    iters="-", time_s="-", note="Sim3-aligned mono VO (target to beat)"))
    print("mono-VO baseline:", [round(x, 3) for x in vo], flush=True)

    for pe in (0, 8):
        row, _ = run_cfg(f"odometry_only_pe{pe}",
                         Cfg(use_ranges=False, prior_every=pe), "no range factors")
        emit(f, w, row)
        print("odometry-only pe%d:" % pe, row["ate_ifo001"], row["ate_ifo002"],
              row["ate_ifo003"], flush=True)

    # main grid ------------------------------------------------------------
    floors = [0.2, 0.3, 0.4, 0.5]
    hubers = [0.3, 0.5, 1.0]
    biases = ["const", "off"]
    priors = [0, 8]                     # gauge anchoring off / on
    combos = list(itertools.product(biases, floors, hubers, priors))
    print("running %d grid configs..." % len(combos), flush=True)
    for i, (bm, fl, hk, pe) in enumerate(combos):
        cfg = Cfg(bias_mode=bm, range_sigma_floor=fl, huber_k=hk, prior_every=pe)
        tag = f"g{i:02d}_{bm}_f{fl}_h{hk}_pe{pe}"
        try:
            row, ate = run_cfg(tag, cfg)
            emit(f, w, row)
            print("[%2d/%d] %-30s ate=[%.3f %.3f %.3f] mean=%.3f %.0fs" % (
                i + 1, len(combos), tag, ate[0], ate[1], ate[2],
                row["ate_mean"], row["time_s"]), flush=True)
        except Exception as e:
            emit(f, w, dict(seq=SEQ, tag=tag, bias_mode=bm, range_sigma_floor=fl,
                            huber_k=hk, prior_every=pe, robust=True, err_init="-",
                            err_final="-", ate_ifo001="-", ate_ifo002="-",
                            ate_ifo003="-", ate_mean="-", iters="-", time_s="-",
                            note="FAILED: %s" % e))
            print("[%2d/%d] %-30s FAILED %s" % (i + 1, len(combos), tag, e), flush=True)
    f.close()
    print("done -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()

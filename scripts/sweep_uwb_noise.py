#!/usr/bin/env python3
"""UWB observation-quality sweep -- where does the fusion gain flip sign?

Pre-registered in PREREG_uwb_noise.md. Injects gt_range + controlled error as the
UWB observation (timestamps and association untouched -- only the value changes)
and sweeps the error size, looking for the point where fused ATE crosses the
VINS-alone ATE, per robot.

One process handles ONE grid cell so the grid can be farmed out across cores;
rows are appended to the CSV with an immediate flush + fsync, and a cell already
present in the CSV is skipped, so the sweep is resumable.

Usage:
  python scripts/sweep_uwb_noise.py --cell '<json>'      # one cell (worker)
  python scripts/sweep_uwb_noise.py --plan               # print the grid as JSON
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.fusion import CoVOR, Cfg, umeyama_yaw
from covor import factors as F, data as D
import gtsam

SEQ = "default_3_zigzag_0"
OUT = "/src/gs25058/cr_RNE/covor_slam/results/results_uwb_noise_sweep.csv"

# Appendix B (corrected): the values every §4.9 / §8 experiment actually used.
BASE = dict(use_moment_arm=True, bias_mode="off", use_height=False, prior_every=0,
            robust=True, huber_k=1.0)

SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50]
SEEDS = [0, 1, 2, 3, 4]
CONDS = {  # anchor-free 3 pairs, and the anchored current pipeline
    "C": dict(inter_pairs=((0, 1), (0, 2), (1, 2)), anchor_robots=()),
    "D": dict(),
}
FLOORS = ["matched", "0.05", "0.3"]
FIELDS = ["condition", "sigma", "bias_mode", "outlier", "sigma_floor", "seed",
          "robot", "ATE", "tr_sigma_med", "align_gap", "n_range_factors", "time_s"]


def grid():
    for cond in CONDS:
        for sg in SIGMAS:
            for bias in (0, 1):
                for out in (0, 1):
                    for fl in FLOORS:
                        for sd in SEEDS:
                            yield dict(condition=cond, sigma=sg, bias_mode=bias,
                                       outlier=out, sigma_floor=fl, seed=sd)


def _floor_value(fl, sigma):
    # matched: the floor tracks the injected sigma. The 0.01 m lower bound only
    # keeps sigma=0 from producing an infinite-weight factor; it is a numerical
    # guard, not a tuned parameter.
    if fl == "matched":
        return max(sigma, 0.01)
    return float(fl)


def run_cell(c):
    t0 = time.time()
    cfg = Cfg(**BASE, **CONDS[c["condition"]],
              range_inject=dict(sigma=c["sigma"], bias=bool(c["bias_mode"]),
                                outlier=bool(c["outlier"]), seed=c["seed"]),
              range_sigma_override=_floor_value(c["sigma_floor"], c["sigma"]),
              range_sigma_floor=0.05)
    cov = CoVOR(SEQ, cfg).build()
    res = cov.optimize(max_iter=100, verbose=False)
    marg = gtsam.Marginals(cov.graph, res)
    nrf = cov.stats["n_inter_range"] + cov.stats["n_anchor_range"]

    P, G, TR, names = [], [], [], []
    for k, rb in enumerate(cov.robots):
        if rb.n() == 0:
            continue
        P.append(np.array([res.atPose3(F.X(k, i)).translation() for i in range(rb.n())]))
        G.append(D.mocap_pose_at(SEQ, rb.name, rb.t)[0])
        TR.append(np.array([np.trace(marg.marginalCovariance(F.X(k, i))[3:6, 3:6])
                            for i in range(rb.n())]))
        names.append(rb.name)

    # Same alignment convention as §7: per-robot yaw+translation (4 DoF) for ATE,
    # and the gap to a single joint 4-DoF fit as the inter-robot registration error.
    def rmse(a, b):
        return float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    per = []
    for p, g in zip(P, G):
        R, t = umeyama_yaw(p, g)
        per.append(rmse((R @ p.T).T + t, g))
    Rj, tj = umeyama_yaw(np.vstack(P), np.vstack(G))
    joint = [rmse((Rj @ p.T).T + tj, g) for p, g in zip(P, G)]

    dt = time.time() - t0
    return [dict(**c, robot=nm, ATE=round(a, 5), tr_sigma_med=round(float(np.median(tr)), 6),
                 align_gap=round(j - a, 5), n_range_factors=nrf, time_s=round(dt, 1))
            for nm, a, j, tr in zip(names, per, joint, TR)]


def append_rows(rows):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())


def done_cells():
    if not os.path.exists(OUT):
        return set()
    out = set()
    with open(OUT) as f:
        for r in csv.DictReader(f):
            out.add((r["condition"], float(r["sigma"]), int(r["bias_mode"]),
                     int(r["outlier"]), r["sigma_floor"], int(r["seed"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if a.plan:
        have = done_cells()
        todo = [c for c in grid()
                if (c["condition"], float(c["sigma"]), c["bias_mode"], c["outlier"],
                    c["sigma_floor"], c["seed"]) not in have]
        print(json.dumps(todo))
        return
    c = json.loads(a.cell)
    append_rows(run_cell(c))


if __name__ == "__main__":
    main()

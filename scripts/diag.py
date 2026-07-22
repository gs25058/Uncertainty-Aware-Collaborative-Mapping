#!/usr/bin/env python3
"""Ablation diagnostic: per-robot ATE RMSE for several factor configurations,
to locate why fusion degrades vs the VO baseline."""
import sys
import numpy as np
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E

seq = sys.argv[1] if len(sys.argv) > 1 else "default_3_zigzag_0"


def rmse_of(cov, values, k, with_scale):
    rb = cov.robots[k]
    if rb.n() < 5:
        return None
    traj = np.array([[rb.t[i], *values.atPose3(cov._X(k, i)).translation()]
                     for i in range(rb.n())]) if hasattr(cov, "_X") else None
    from covor import factors as F
    traj = np.array([[rb.t[i], *values.atPose3(F.X(k, i)).translation()]
                     for i in range(rb.n())])
    r, *_ = E.ate_rmse(traj, rb.mocap, with_scale=with_scale)
    return r


def run(name, cfg, iters=40):
    cov = CoVOR(seq, cfg).build()
    init_rmse = [rmse_of(cov, cov.values, k, with_scale=False) for k in range(3)]
    cov.optimize(max_iter=iters, verbose=False)
    fin_rmse = [rmse_of(cov, cov.result, k, with_scale=False) for k in range(3)]
    def fmt(xs):
        return " ".join(f"{x:.3f}" if x is not None else "  -  " for x in xs)
    print(f"{name:22s} init[{fmt(init_rmse)}]  final[{fmt(fin_rmse)}]  "
          f"(inter={cov.stats['n_inter_range']} anch={cov.stats['n_anchor_range']})")


# VO-only Sim3-aligned baseline
cov0 = CoVOR(seq, Cfg()).build()
from covor import evaluate as E2
base = []
for k in range(3):
    rb = cov0.robots[k]
    if rb.n() < 5:
        base.append(None); continue
    vo = np.array([[rb.t[i], *rb.poses_vo[i].translation()] for i in range(rb.n())])
    r, *_ = E2.ate_rmse(vo, rb.mocap, with_scale=True)
    base.append(r)
print("mono-VO (Sim3-aligned) RMSE:", " ".join(f"{x:.3f}" if x else "-" for x in base))
print("(below: SE3-aligned metric RMSE, init vs final)\n")

run("odometry-only",       Cfg(use_ranges=False))
run("full (current)",      Cfg())
run("anchor-only",         Cfg(use_inter=False))
run("inter-only",          Cfg(use_anchor=False))
run("tight-odo x0.2",      Cfg(sigma_odo_trans=0.01, sigma_odo_rot=0.01))
run("tight-odo+looseRange",Cfg(sigma_odo_trans=0.01, sigma_odo_rot=0.01))

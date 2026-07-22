#!/usr/bin/env python3
"""Run CoVOR-SLAM on a MILUV sequence: build the visual-range factor graph,
optimize, and compare mono-VO-only vs fused trajectories against mocap.

Usage: python run_covor.py [sequence]      (default: default_3_zigzag_0)
"""
import sys
import json
import numpy as np

from covor import data as D
from covor.fusion import CoVOR, Cfg
from covor import evaluate as E

RESULTS = "/src/gs25058/cr_RNE/covor_slam/results"


def main(seq):
    print(f"=== CoVOR-SLAM on MILUV '{seq}' ===")
    cov = CoVOR(seq, Cfg()).build()
    print("Graph stats:")
    print(json.dumps(cov.stats, indent=2))

    cov.optimize(max_iter=100, verbose=True)
    print(f"\nError: initial={cov.stats['initial_error']:.1f} "
          f"-> final={cov.stats['final_error']:.1f}")
    biases = cov.biases(cov.result)
    if biases:
        print(f"Online range-bias estimate (m): {biases}")

    robots, vo_only_al, fused_al, mocaps = [], [], [], []
    vo_err, fused_err = [], []
    print("\n%-8s %12s %12s %10s" % ("robot", "monoVO_RMSE", "CoVOR_RMSE", "improve"))
    print("-" * 46)
    summary = {}
    for rb in cov.robots:
        k = rb.k
        if rb.n() < 5:
            print("%-8s   (insufficient VO keyframes: %d)" % (rb.name, rb.n()))
            robots.append(rb.name); vo_only_al.append(None); fused_al.append(None)
            mocaps.append(None); vo_err.append(None); fused_err.append(None)
            continue
        # mono-VO-only trajectory in its own frame (Sim3-aligned for eval)
        vo_traj = np.array([[rb.t[i], *rb.poses_vo[i].translation()] for i in range(rb.n())])
        vo_rmse, vo_a, gt, ts = E.ate_rmse(vo_traj, rb.mocap, with_scale=True)
        # fused trajectory (metric, SE3-aligned for eval)
        fu_traj = cov.trajectory(cov.result, k)
        fu_rmse, fu_a, gt2, ts2 = E.ate_rmse(fu_traj, rb.mocap, with_scale=False)

        imp = 100 * (vo_rmse - fu_rmse) / vo_rmse
        print("%-8s %12.3f %12.3f %9.1f%%" % (rb.name, vo_rmse, fu_rmse, imp))
        summary[rb.name] = dict(monoVO_rmse=round(vo_rmse, 4),
                                covor_rmse=round(fu_rmse, 4),
                                improve_pct=round(imp, 1),
                                keyframes=rb.n(), scale_init=round(rb.s0, 4))
        robots.append(rb.name)
        vo_only_al.append(np.column_stack([vo_a[:, 0], vo_a[:, 1]]))
        fused_al.append(np.column_stack([fu_a[:, 0], fu_a[:, 1]]))
        mocaps.append(np.column_stack([gt[:, 0], gt[:, 1]]))
        vo_err.append((ts, np.linalg.norm(vo_a - gt, axis=1)))
        fused_err.append((ts2, np.linalg.norm(fu_a - gt2, axis=1)))

    E.plot_trajectories(seq, robots, vo_only_al, fused_al, mocaps,
                        f"{RESULTS}/{seq}_trajectories.png")
    E.plot_errors(seq, robots, vo_err, fused_err, f"{RESULTS}/{seq}_errors.png")

    with open(f"{RESULTS}/{seq}_summary.json", "w") as f:
        json.dump(dict(stats=cov.stats, per_robot=summary), f, indent=2, default=float)
    print(f"\nSaved figures + summary to {RESULTS}/")
    return summary


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "default_3_zigzag_0"
    main(seq)

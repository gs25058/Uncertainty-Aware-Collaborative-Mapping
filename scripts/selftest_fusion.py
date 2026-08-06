#!/usr/bin/env python3
"""LEGACY (Sim(3) era) -- synthetic self-test of the CoVOR-SLAM factor machinery.

Built 2 robots with known metric trajectories, generated UP-TO-SCALE VO and noisy
ranges, and checked that LM recovered both the poses and the per-node scale.

The scale variable no longer exists: the VINS front-end is metric and the graph is
pure SE(3). This test is therefore not just broken but meaningless as written --
its whole subject (scale recovery) is gone. Run it at commit a0a5e07, or write an
SE(3) replacement. Failing loudly beats an AttributeError deep in the build.
"""
import sys
import numpy as np
import gtsam

sys.exit(__doc__)

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor import factors as F

rng = np.random.default_rng(0)


def make_traj(n, radius, phase):
    """Circle in XY, metric."""
    poses = []
    for i in range(n):
        a = phase + 2 * np.pi * i / n * 0.6
        R = gtsam.Rot3.Rz(a)
        t = np.array([radius * np.cos(a), radius * np.sin(a), 0.0])
        poses.append(gtsam.Pose3(R, t))
    return poses


def test():
    n = 15
    gt = [make_traj(n, 2.0, 0.0), make_traj(n, 2.0, np.pi)]
    true_scale = [1.0, 1.0]
    # up-to-scale VO: apply an arbitrary similarity (rot+trans+scale) to metric traj
    vo_scale = [0.5, 1.7]
    vo = []
    for k in range(2):
        Ralign = gtsam.Rot3.Rz(0.4 * (k + 1))
        talign = np.array([1.0 * k, -0.5, 0.0])
        s = vo_scale[k]
        vk = []
        for p in gt[k]:
            # invert: vo = Sim3^{-1}(metric); here just craft a consistent up-to-scale traj
            tw = (1.0 / s) * (Ralign.matrix().T @ (p.translation() - talign))
            Rw = Ralign.inverse().compose(p.rotation())
            vk.append(gtsam.Pose3(Rw, tw))
        vo.append(vk)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    # init world = gt + noise (simulating imperfect Umeyama init); scale init = vo_scale
    for k in range(2):
        for i in range(n):
            noisyR = gt[k][i].rotation()
            noisyt = gt[k][i].translation() + rng.normal(0, 0.2, 3)
            values.insert(F.X(k, i), gtsam.Pose3(noisyR, noisyt))
            values.insert(F.Sc(k, i), vo_scale[k])
        graph.add(F.pose_prior(k, 0, gt[k][0], 0.05, 0.1))
        graph.add(F.scale_prior(k, 0, vo_scale[k], 0.5))
        for i in range(n - 1):
            dpose = vo[k][i].between(vo[k][i + 1])
            graph.add(F.odometry_factor(k, i, dpose, 0.02, 0.05))
            graph.add(F.scale_walk_factor(k, i, 0.05))

    # anchor at origin-ish + inter-agent ranges from GT (noisy)
    anchor = np.array([3.0, 0.0, 0.0])
    for i in range(n):
        za = np.linalg.norm(gt[0][i].translation() - anchor) + rng.normal(0, 0.05)
        graph.add(F.anchor_range_factor(0, i, anchor, za, 0.05, robust=False))
        zi = np.linalg.norm(gt[0][i].translation() - gt[1][i].translation()) + rng.normal(0, 0.05)
        graph.add(F.inter_range_factor(0, i, 1, i, zi, 0.05, robust=False))

    e0 = graph.error(values)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    res = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    e1 = graph.error(res)

    # position recovery error
    errs = []
    for k in range(2):
        for i in range(n):
            est = res.atPose3(F.X(k, i)).translation()
            errs.append(np.linalg.norm(est - gt[k][i].translation()))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    print(f"factors={graph.size()} vars={values.size()}")
    print(f"error: {e0:.2f} -> {e1:.2f}")
    print(f"position RMSE vs GT after fusion: {rmse:.3f} m")
    s_est = [res.atDouble(F.Sc(k, 0)) for k in range(2)]
    print(f"scale init {vo_scale} -> est {[round(x,3) for x in s_est]}")
    assert e1 < e0, "optimization did not reduce error"
    assert rmse < 0.15, f"position RMSE too high: {rmse}"
    print("SELFTEST PASSED")


if __name__ == "__main__":
    test()

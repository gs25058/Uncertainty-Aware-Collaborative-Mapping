#!/usr/bin/env python3
"""Part 1: measure what the gauge prior actually is, before designing around it.

    python scripts/synth/gauge_probe.py --name room909 --seed 0

Two questions, both answered by measurement rather than by reading the code:

  Q1  How many priors, on which nodes, with what sigma, and WHERE DOES THE MEAN
      COME FROM? The suspicion is that the mean is not "the robot's first pose"
      but a whole-trajectory least-squares fit to mocap, which would let the
      prior absorb the front-end's drift.

  Q2  If the mean were the GT FIRST POSE instead, how far would the absolute
      (unaligned) error move? A large move confirms the confound; no move
      refutes the hypothesis and the Part 2 design has to change.

Q2 is run WITHOUT editing covor/, by overriding ``Robot.align_to_world`` at
RUN TIME inside this probe. The input-only route was tried first and does not
work: ``load_mocap`` passes the file through MILUV's csaps cleaner, which at our
20 Hz mocap rate distorts the track by 0.18-0.42 m rms (measured below), so a
substitute mocap does not survive the loader. See ``rate_check`` -- the
distortion is a SAMPLE-RATE artefact and vanishes at 100 Hz (0.0004 m rms),
which is the rate MILUV's own mocap runs at and the regime smooth=0.9999 was
tuned for.
"""
import argparse
import json
import os
import shutil
import sys

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
from covor import data as D
from covor.fusion import CoVOR, Cfg, umeyama_yaw
from covor.synth import dataset as DS, metrics as ME
from covor.synth.config import SynthCfg, ROBOTS
from fuse_synth import FAITHFUL, CONDITIONS


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def q1(cfg, gt):
    """Where the prior sits and where its mean comes from."""
    out = {}
    for cond, pairs in CONDITIONS.items():
        cov = CoVOR(cfg.seq, Cfg(**FAITHFUL, inter_pairs=pairs)).build()
        s = cov.stats
        rec = dict(n_gauge_prior=s["n_gauge_prior"], gauge_on=list(s["gauge_on"]),
                   n_weak_priors=s["n_weak_priors"], n_inter=s["n_inter_range"],
                   robots={})
        for rb in cov.robots:
            p0 = np.asarray(rb.init_world[0].translation(), float)
            R0 = rb.init_world[0].rotation().matrix()
            # the GT body pose at this robot's first node time
            g = gt[rb.name]
            i0 = int(np.abs(g["t"] - rb.t[0]).argmin())
            p_gt0, R_gt0 = g["p"][i0], g["R"][i0]
            # what a FIRST-POSE-ONLY 4-DoF alignment would have produced
            pv0 = np.asarray(rb.poses_vo[0].translation(), float)
            Rv0 = rb.poses_vo[0].rotation().matrix()
            psi = _yaw(R_gt0) - _yaw(Rv0)
            Rz = Rot.from_euler("z", psi).as_matrix()
            p0_first = p_gt0                      # by construction of that fit
            # residual of the whole-trajectory fit that IS used
            src = np.array([p.translation() for p in rb.poses_vo])
            dst = np.array([D.mocap_position_at(rb.mocap, t) for t in rb.t])
            res = np.linalg.norm(src @ rb.align_R.T + rb.align_t - dst, axis=1)
            rec["robots"][rb.name] = dict(
                prior_node=0,
                prior_mean_pos=p0.tolist(),
                gt_first_pos=p_gt0.tolist(),
                d_prior_mean_to_gt_first=float(np.linalg.norm(p0 - p_gt0)),
                d_firstfit_to_gt_first=float(np.linalg.norm(p0_first - p_gt0)),
                align_yaw_deg=float(np.degrees(_yaw(rb.align_R))),
                firstpose_yaw_deg=float(np.degrees(psi)),
                umeyama_residual_rms=float(np.sqrt((res ** 2).mean())),
                umeyama_residual_max=float(res.max()),
                prior_rot_vs_gt_deg=float(np.degrees(
                    Rot.from_matrix(R_gt0.T @ R0).magnitude())))
        out[cond] = rec
    return out


def rate_check(cfg, gt, rates=(20, 50, 100, 200)):
    """How faithful load_mocap's csaps cleaner is at each mocap sample rate.

    This is why the input-only route to Q2 fails, and it is also a defect in the
    synthetic dataset itself: mocap.csv is written at 20 Hz, where the cleaner
    deviates from the exact GT by 0.18-0.42 m rms.
    """
    import tempfile
    from scipy.spatial.transform import Slerp
    base = cfg.dataroot()
    out = {}
    tmp = tempfile.mkdtemp(prefix="mocap_rate_")
    try:
        for rate in rates:
            root = os.path.join(tmp, str(rate))
            shutil.copytree(base, root)
            for r in ROBOTS:
                g = gt[r]
                tq = np.arange(g["t"][0], g["t"][-1], 1.0 / rate)
                p = np.stack([np.interp(tq, g["t"], g["p"][:, i]) for i in range(3)], 1)
                q = Slerp(g["t"], Rot.from_matrix(g["R"]))(tq).as_quat()
                pd.DataFrame({
                    "timestamp": tq,
                    "pose.position.x": p[:, 0], "pose.position.y": p[:, 1],
                    "pose.position.z": p[:, 2],
                    "pose.orientation.x": q[:, 0], "pose.orientation.y": q[:, 1],
                    "pose.orientation.z": q[:, 2], "pose.orientation.w": q[:, 3],
                }).to_csv(os.path.join(root, cfg.seq, r, "mocap.csv"), index=False)
            D.DATA = root
            D._MOCAP_CACHE.clear()
            rec = {}
            for r in ROBOTS:
                m = D.load_mocap(cfg.seq, r)
                t = m["timestamp"].to_numpy()
                g = gt[r]
                exact = np.stack([np.interp(t, g["t"], g["p"][:, i]) for i in range(3)], 1)
                e = np.linalg.norm(m[["x", "y", "z"]].to_numpy() - exact, axis=1)
                rec[r] = dict(rms=float(np.sqrt((e ** 2).mean())), max=float(e.max()),
                              first=float(e[0]))
            out[rate] = rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        D.DATA = base
        D._MOCAP_CACHE.clear()
    return out


def patch_first_pose(gt):
    """Override Robot.align_to_world so the L_k -> G transform is fixed by the
    robot's FIRST pose alone (yaw + position), not by a whole-trajectory fit.

    RUN-TIME override inside this probe only -- covor/fusion.py is untouched.
    This is the behaviour Part 2 would give a `gauge_init="first_pose"` option.
    """
    from covor.fusion import Robot
    orig = Robot.align_to_world

    def patched(self):
        g = gt[self.name]
        i0 = int(np.abs(g["t"] - self.t[0]).argmin())
        pv0 = np.asarray(self.poses_vo[0].translation(), float)
        Rv0 = self.poses_vo[0].rotation().matrix()
        psi = _yaw(g["R"][i0]) - _yaw(Rv0)
        R = Rot.from_euler("z", psi).as_matrix()
        tvec = g["p"][i0] - R @ pv0
        self.align_R, self.align_t = R, tvec
        Ralign = gtsam.Rot3(R)
        self.init_world = [gtsam.Pose3(Ralign.compose(p.rotation()),
                                       R @ p.translation() + tvec)
                           for p in self.poses_vo]
    Robot.align_to_world = patched
    return orig


def fuse_here(seq, pairs):
    cov = CoVOR(seq, Cfg(**FAITHFUL, inter_pairs=pairs)).build()
    res = cov.optimize(max_iter=100, verbose=False)
    from covor import factors as F
    out = {}
    for k, rb in enumerate(cov.robots):
        if rb.n() == 0:
            continue
        b_T_c = D.load_body_T_cam(rb.name, 0)
        T = np.zeros((rb.n(), 4, 4))
        for i in range(rb.n()):
            T[i] = res.atPose3(F.X(k, i)).matrix() @ b_T_c
        out[rb.name] = dict(t=rb.t, T=T,
                            align_R=rb.align_R, align_t=rb.align_t)
    return out, cov.stats


def errors(poses, gt):
    rec = {}
    for r, P in poses.items():
        T_gt, _ = DS.gt_camera_poses(r, P["t"], gt)
        e = np.linalg.norm(P["T"][:, :3, 3] - T_gt[:, :3, 3], axis=1)
        rec[r] = dict(abs_rms=float(np.sqrt((e ** 2).mean())),
                      abs_median=float(np.median(e)),
                      ate_rmse=ME.ate(P["T"][:, :3, 3], T_gt[:, :3, 3])["ate_rmse"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conds", default="A_1drone")
    args = ap.parse_args()
    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=args.seed)
    DS.install_paths(cfg)
    gt = DS.load_gt_traj(cfg)

    print("=== Q1: what the gauge prior is ===")
    facts = q1(cfg, gt)
    for cond, rec in facts.items():
        print("%s: %d gauge prior(s) on robot(s) %s, node 0 only, "
              "sigma_rot=sigma_trans=1e-3, weak priors=%d, inter=%d"
              % (cond, rec["n_gauge_prior"], rec["gauge_on"],
                 rec["n_weak_priors"], rec["n_inter"]))
    print("\n  prior mean vs GT first pose (condition-independent: same "
          "align_to_world for every condition)")
    print("  %-8s %14s %14s %12s %12s %10s" % (
        "robot", "|mean-GTfirst|", "|first-fit-GT|", "align_yaw",
        "firstpose_yaw", "fit_rms"))
    for r, v in facts["A_1drone"]["robots"].items():
        print("  %-8s %14.4f %14.4f %11.3f° %12.3f° %10.4f"
              % (r, v["d_prior_mean_to_gt_first"], v["d_firstfit_to_gt_first"],
                 v["align_yaw_deg"], v["firstpose_yaw_deg"],
                 v["umeyama_residual_rms"]))

    print("\n=== mocap loader fidelity vs sample rate ===")
    rc = rate_check(cfg, gt)
    print("  %6s %-8s %10s %10s %12s" % ("rate", "robot", "rms [m]", "max [m]",
                                         "first [m]"))
    for rate, rec in rc.items():
        for r, v in rec.items():
            print("  %5dHz %-8s %10.5f %10.5f %12.5f"
                  % (rate, r, v["rms"], v["max"], v["first"]))

    print("\n=== Q2: swap the prior mean for the GT first pose ===")
    print("  (run-time override of Robot.align_to_world; covor/ untouched)")
    cond = args.conds.split(",")[0]
    base, _ = fuse_here(cfg.seq, CONDITIONS[cond])
    e_base = errors(base, gt)
    orig = patch_first_pose(gt)
    try:
        alt, _ = fuse_here(cfg.seq, CONDITIONS[cond])
    finally:
        from covor.fusion import Robot
        Robot.align_to_world = orig
    e_alt = errors(alt, gt)
    print("\n  %-8s %-28s %-28s" % ("robot", "umeyama gauge (current)",
                                    "first-pose gauge"))
    print("  %-8s %12s %12s   %12s %12s" % ("", "abs_rms", "ATE", "abs_rms", "ATE"))
    for r in ROBOTS:
        print("  %-8s %12.4f %12.4f   %12.4f %12.4f"
              % (r, e_base[r]["abs_rms"], e_base[r]["ate_rmse"],
                 e_alt[r]["abs_rms"], e_alt[r]["ate_rmse"]))
    pool = lambda e: float(np.mean([v["abs_rms"] for v in e.values()]))
    print("  %-8s %12.4f %12s   %12.4f" % ("POOLED", pool(e_base), "",
                                           pool(e_alt)))

    out = dict(q1=facts, mocap_rate=rc, condition=cond,
               q2=dict(umeyama=e_base, first_pose=e_alt))
    p = os.path.join(cfg.outdir(), "gauge_probe_seed%d.json" % args.seed)
    with open(p, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    print("\nwrote", p)


if __name__ == "__main__":
    main()

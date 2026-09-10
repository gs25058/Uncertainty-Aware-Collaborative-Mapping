"""Assemble the synthetic sequence and hand it to the UNMODIFIED pipeline.

``covor.data`` reads MILUV from module-level path constants. Rather than copy a
fake sequence into the real dataset tree (which would make the two impossible to
tell apart later), ``install_paths`` redirects those constants at run time to
``results/synth_<name>/``. Nothing in covor/ is edited; the same
``covor.fusion.CoVOR`` and ``covor.occupancy.OccupancyBuilder`` run on both.

Layout written:
    results/synth_<name>/data/<seq>/timeshift.yaml
    results/synth_<name>/data/<seq>/<robot>/mocap.csv       (GT body pose)
    results/synth_<name>/data/<seq>/<robot>/uwb_range.csv   (inter-agent only)
    results/synth_<name>/vins/<seq>/<robot>/vio.csv         (raw local frame)
    results/synth_<name>/gt_traj.npz                        (EXACT GT, for metrics)
"""
import json
import os

import numpy as np

from . import mesh_gt as MG
from . import trajectory as TJ
from . import uwb as UWB
from . import vio as VIO
from .config import ROBOTS


def install_paths(cfg):
    """Point covor.data at the synthetic sequence. Returns the old values."""
    from covor import data as D
    old = (D.DATA, D.VINS_DIR)
    D.DATA = cfg.dataroot()
    D.VINS_DIR = cfg.vinsroot()
    D._MOCAP_CACHE.clear()
    return old


def restore_paths(old):
    from covor import data as D
    D.DATA, D.VINS_DIR = old
    D._MOCAP_CACHE.clear()


def world_mesh(cfg):
    mc = json.load(open(cfg.mesh_config()))
    return MG.apply_transform(MG.load(mc["obj"]), np.array(mc["T_mesh_world"]))


def build(cfg, verbose=True):
    """Generate and write the whole synthetic dataset. Returns a report dict."""
    lab, ijk_min, res, T_mw, ginfo = MG.load_gt(cfg.gt_voxel())
    gt = TJ.plan_all(lab, ijk_min, res, cfg)
    rep = dict(seq=cfg.seq, seed=cfg.seed, robots={})

    # --- GT trajectories (exact; every metric uses these, never mocap.csv) ---
    np.savez(os.path.join(cfg.outdir(), "gt_traj.npz"),
             **{f"{r}_{k}": gt[r][k] for r in gt for k in ("t", "p", "R")})

    # --- front-end -----------------------------------------------------------
    for k, r in enumerate(ROBOTS):
        g = gt[r]
        v = VIO.generate(g["t"], g["p"], g["R"], cfg, k)
        gt[r]["vio"] = v
        VIO.to_vins_csv(os.path.join(cfg.vinsroot(), cfg.seq, r, "vio.csv"),
                        v["t"], v["p"], v["R"], v["v"], cfg.timeshift)
        VIO.to_mocap_csv(os.path.join(cfg.dataroot(), cfg.seq, r, "mocap.csv"),
                         g["t"], g["p"], g["R"])
        rep["robots"][r] = dict(
            n_samples=int(len(g["t"])), duration_s=float(g["t"][-1]),
            route_len_m=g["route_len"], flown_len_m=g["path_len"],
            z_range=[float(g["p"][:, 2].min()), float(g["p"][:, 2].max())],
            clearance_min_m=float(g["clearance"].min()),
            n_clearance_violation=int(g["n_violation"]),
            vio=v["stats"])
        if verbose:
            s = v["stats"]
            print("  %s: rel_rot %.4f (target %.4f) rel_trans %.4f (target %.4f) "
                  "yaw drift %+.2f deg/min, raw VIO ATE %.3f m"
                  % (r, s["rel_rot_rms"], s["rel_rot_target"], s["rel_trans_rms"],
                     s["rel_trans_target"], s["yaw_drift_deg_per_min"],
                     s["raw_vio_ate_rmse"]))

    # --- UWB -----------------------------------------------------------------
    scene = MG.raycasting_scene(world_mesh(cfg)) if cfg.uwb_nlos else None
    df = UWB.generate({r: gt[r] for r in ROBOTS}, cfg, scene)
    files = UWB.write(df, cfg)
    rep["uwb"] = dict(n_rows_unique=int(len(df)),
                      per_robot_rows={r: n for r, (_, n) in files.items()},
                      residual_rms=float(np.sqrt(
                          ((df["range"] - df["gt_range"]) ** 2).mean())),
                      residual_mean=float((df["range"] - df["gt_range"]).mean()),
                      nlos=bool(cfg.uwb_nlos))

    # --- clock ---------------------------------------------------------------
    p = os.path.join(cfg.dataroot(), cfg.seq, "timeshift.yaml")
    with open(p, "w") as f:
        f.write("timeshift_ns: %d\ntimeshift_s: %d\n"
                % (cfg.timeshift_ns, cfg.timeshift_s))

    with open(os.path.join(cfg.outdir(),
                           "dataset_report_seed%d.json" % cfg.seed), "w") as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    return rep, gt


def load_gt_traj(cfg):
    z = np.load(os.path.join(cfg.outdir(), "gt_traj.npz"))
    return {r: dict(t=z[f"{r}_t"], p=z[f"{r}_p"], R=z[f"{r}_R"]) for r in ROBOTS}


def gt_camera_poses(robot, t_query, gt):
    """world<-camera SE3 at ``t_query``, from the EXACT GT body trajectory.

    T_wc = T_wb @ body_T_cam0 -- the same composition fuse_and_dump.py applies to
    the fused body poses (appendix A-9: skipping it aims every ray ~120 deg off).
    """
    from scipy.spatial.transform import Rotation as R, Slerp
    from covor import data as D
    g = gt[robot]
    tq = np.clip(np.asarray(t_query, float), g["t"][0], g["t"][-1])
    p = np.stack([np.interp(tq, g["t"], g["p"][:, i]) for i in range(3)], 1)
    Rb = Slerp(g["t"], R.from_matrix(g["R"]))(tq).as_matrix()
    T = np.tile(np.eye(4), (len(tq), 1, 1))
    T[:, :3, :3] = Rb
    T[:, :3, 3] = p
    return T @ D.load_body_T_cam(robot, 0), p

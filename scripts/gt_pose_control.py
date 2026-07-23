#!/usr/bin/env python3
"""Control experiment: build the SAME occupancy map from mocap GT poses.

This is the experiment that separates "the mapping is wrong" from "the poses are
wrong". Everything downstream of the pose -- stereo, back-projection, ray
casting, weighted log-odds -- is identical to build_occupancy.py; only the pose
source changes. GT carries no registration uncertainty, so w_pose is forced to 1
(tr_sigma_pos=0) and only the depth weight w_depth applies.

It also scores both symptoms against ground truth, which MILUV happens to
provide for free:
  - object recall: config/apriltags/apriltags.yaml gives the 3D positions of the
    13 elevated tag stands, i.e. the only real objects in the room. We count the
    occupied voxels that land on each.
  - floating cubes: occupied voxels strictly inside the room and above the floor,
    plus how many of them coincide with a teammate's position (a moving drone).

Frames: mocap gives the px4-IMU body pose in the mocap world frame, and
config/realsense/<robot>/extrinsics_px4imu.yaml gives T_cam_imu (cam0 <- body),
so T_wc = T_wb @ inv(T_cam_imu). This chain was validated by reprojecting the
known AprilTag positions into the images -- every tag landed on its marker.

Usage: python scripts/gt_pose_control.py --robot ifo001 --stride 2
"""
import os
import sys
import ast
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts")
from covor.occupancy import (load_stereo_calib, StereoDepth, DepthCfg,
                             OccupancyBuilder, OccCfg)
from build_occupancy import FrameIndex, _fname_for, SEQ, DATA, VO, OUTDIR

MILUV = "/src/gs25058/cr_RNE/miluv"
ROBOTS = ["ifo001", "ifo002", "ifo003"]


def gt_poses(robot, ts):
    """world <- cam0 (infra1) SE3 at times ts, from mocap + the px4imu extrinsic."""
    ex = yaml.safe_load(open(f"{MILUV}/config/realsense/{robot}/extrinsics_px4imu.yaml"))
    T_cb = np.array(ex["cam0"]["T_cam_imu"], float)          # cam0 <- body
    m = pd.read_csv(f"{MILUV}/data/{SEQ}/{robot}/mocap.csv")
    j = np.abs(m.timestamp.values[None, :] - ts[:, None]).argmin(1)
    R = Rot.from_quat(m[["pose.orientation.x", "pose.orientation.y",
                         "pose.orientation.z", "pose.orientation.w"]].values[j]).as_matrix()
    p = m[["pose.position.x", "pose.position.y", "pose.position.z"]].values[j]
    T_wb = np.tile(np.eye(4), (len(ts), 1, 1))
    T_wb[:, :3, :3] = R
    T_wb[:, :3, 3] = p
    return T_wb @ np.linalg.inv(T_cb)


def gt_objects():
    """(13,3) positions of the elevated AprilTag stands -- the room's only objects."""
    d = yaml.safe_load(open(f"{MILUV}/config/apriltags/apriltags.yaml"))["0"]
    A = np.array([ast.literal_eval(v) for v in d.values()])
    return A[A[:, 2] > 0.3]


def build(robot, stride):
    npz = np.load(f"{VO}/occ_{SEQ}_{robot}.npz")
    t = npz["t"]
    T = gt_poses(robot, t)
    mates = {o: gt_poses(o, t)[:, :3, 3] for o in ROBOTS if o != robot}
    b = OccupancyBuilder(OccCfg(weighted=True))
    sd = StereoDepth(load_stereo_calib(robot), DepthCfg())
    idx = FrameIndex(robot)
    used, traj = 0, []
    for i in range(0, len(t), stride):
        tv = idx.pair(t[i])
        if tv is None:
            continue
        f1 = _fname_for(robot, "infra1", tv)
        f2 = _fname_for(robot, "infra2", tv)
        if not f1 or not f2:
            continue
        il = cv2.imread(os.path.join(DATA, SEQ, robot, "infra1", f1), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(os.path.join(DATA, SEQ, robot, "infra2", f2), cv2.IMREAD_GRAYSCALE)
        if il is None or ir is None:
            continue
        Z, sZ, valid = sd.depth(il, ir)
        if valid.sum() < 100:
            continue
        P_cam, sZp = sd.backproject(Z, sZ, valid, downsample=4)
        b.integrate_frame(T[i], 0.0, P_cam, sZp,               # GT -> w_pose = 1
                          teammates=np.array([v[i] for v in mates.values()]))
        traj.append(T[i][:3, 3])
        used += 1
    b.finalize()
    return b, used, np.array(traj), {o: v[::stride] for o, v in mates.items()}


def score(occ, objects, mates):
    """(object voxels, per-stand counts, interior floating voxels, teammate-coincident)."""
    per = [int(((np.linalg.norm(occ[:, :2] - g[:2], axis=1) < 0.4) &
                (occ[:, 2] > 0.2) & (occ[:, 2] < 1.5)).sum()) for g in objects]
    inner = occ[(np.abs(occ[:, 0]) < 3) & (np.abs(occ[:, 1]) < 3) &
                (occ[:, 2] > 0.35) & (occ[:, 2] < 2.0)]
    M = np.vstack(list(mates.values()))
    near = int((np.linalg.norm(inner[:, None, :] - M[None, :, :], axis=2).min(1) < 0.4).sum())
    return sum(per), np.array(per), len(inner), near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="ifo001")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    b, n, traj, mates = build(args.robot, args.stride)
    occ, free = b.classify_points()          # before write_bt, which is destructive
    G = gt_objects()
    ov, per, inner, near = score(occ, G, mates)
    print("GT-pose control | %s | %d keyframes | %.0fs" % (args.robot, n, time.time() - t0))
    print("  occupied=%d  free=%d" % (len(occ), len(free)))
    print("  objects: %d/%d stands with >=3 voxels, %d voxels total (median %d/stand)"
          % ((per >= 3).sum(), len(per), ov, np.median(per)))
    print("  floating inside the room: %d, of which within 0.4 m of a teammate: %d"
          % (inner, near))

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.6))
    m = (occ[:, 2] > -0.2) & (occ[:, 2] < 2.5)
    sc = ax[0].scatter(occ[m, 0], occ[m, 1], c=occ[m, 2], s=3, cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="height z [m]", shrink=.8)
    ax[0].plot(traj[:, 0], traj[:, 1], "r.-", ms=2, lw=.6)
    ax[0].scatter(G[:, 0], G[:, 1], marker="*", s=260, c="red", ec="k", zorder=5,
                  label="GT object (tag stand)")
    ax[0].set_title("occupied, top-down"); ax[0].legend(); ax[0].axis("equal"); ax[0].grid(alpha=.3)
    zc, band = 0.8, 0.15
    mf = np.abs(free[:, 2] - zc) < band
    mo = np.abs(occ[:, 2] - zc) < band
    ax[1].scatter(free[mf, 0], free[mf, 1], c="tab:blue", s=4, alpha=.3, label="free")
    ax[1].scatter(occ[mo, 0], occ[mo, 1], c="tab:red", s=8, label="occupied")
    ax[1].scatter(G[:, 0], G[:, 1], marker="*", s=260, c="lime", ec="k", zorder=5)
    ax[1].set_title("slice z=%.1f+-%.2f m" % (zc, band))
    ax[1].axis("equal"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.suptitle("GT-pose control (w_pose=1) | %s, %d KF | occ=%d free=%d | "
                 "%d/%d objects, %d floating" % (args.robot, n, len(occ), len(free),
                                                 (per >= 3).sum(), len(per), inner))
    plt.tight_layout()
    out = f"{OUTDIR}/gt_pose_control_{args.robot}.png"
    plt.savefig(out, dpi=95); print("  saved", out)
    b.write_bt(f"{OUTDIR}/gt_pose_control_{args.robot}.bt")


if __name__ == "__main__":
    main()

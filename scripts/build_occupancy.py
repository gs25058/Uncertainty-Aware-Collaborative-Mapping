#!/usr/bin/env python3
"""Phase 1 (single drone) / Phase 3 (multi-drone) occupancy driver.

Ties the pipeline together: fused poses + tr(Sigma_pos) (from fuse_and_dump.py)
+ MILUV stereo -> uncertainty-aware OctoMap. The same OccupancyBuilder is reused
across robots (proposal Phase 3-3: 1 vs 2 vs 3 drones = which .npz files are fed
into the same tree). The uncertainty weighting toggles with --uniform, giving the
Phase 3-7 ablation (standard OctoMap vs uncertainty-weighted).

Usage:
  python scripts/build_occupancy.py --drones ifo001 --weighted   [--stride 2]
  python scripts/build_occupancy.py --drones ifo001 --uniform
  python scripts/build_occupancy.py --drones ifo001,ifo002,ifo003 --weighted
"""
import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.occupancy import (load_stereo_calib, StereoDepth, DepthCfg,
                             OccupancyBuilder, OccCfg)

SEQ = "default_3_zigzag_0"
DATA = "/src/gs25058/cr_RNE/miluv/data"
VO = "/src/gs25058/cr_RNE/covor_slam/vo_output"
# Figures and .bt trees land in the repo's results/ so they survive the session.
# This used to be a hard-coded per-session /tmp scratchpad, which meant the
# ablation artefacts we cite disappeared with the session that made them.
OUTDIR = os.environ.get(
    "COVOR_OUTDIR", "/src/gs25058/cr_RNE/covor_slam/results")


class FrameIndex:
    """Nearest infra1/infra2 stereo pair for a keyframe timestamp."""
    def __init__(self, robot):
        d = os.path.join(DATA, SEQ, robot)
        self.d = d
        f1 = set(os.listdir(os.path.join(d, "infra1")))
        f2 = set(os.listdir(os.path.join(d, "infra2")))
        common = sorted((float(f[:-5]) for f in (f1 & f2)))  # shared timestamps
        self.ts = np.array(common)

    def pair(self, t, tol=0.05):
        """Nearest shared timestamp value to keyframe time t (or None)."""
        i = int(np.abs(self.ts - t).argmin())
        if abs(self.ts[i] - t) > tol:
            return None
        return self.ts[i]


def _fname_for(robot, cam, tval, cache={}):
    key = (robot, cam)
    if key not in cache:
        d = os.path.join(DATA, SEQ, robot, cam)
        cache[key] = {float(f[:-5]): f for f in os.listdir(d)}
    return cache[key].get(tval)


class Teammates:
    """Where the other robots were, in the fused frame, at any instant.

    Only knowable because CoVOR fuses every robot into one frame (§4.7); used to
    keep observations of flying teammates out of the static map (OccCfg.dyn_radius).
    """
    def __init__(self, robot, others, in_tag=""):
        self.tracks = []
        for o in others:
            if o == robot:
                continue
            npz = np.load(f"{VO}/occ_{SEQ}_{o}{in_tag}.npz")
            # Mask against the AIRFRAME centre, not the camera: T is world<-camera
            # and the camera sits ~0.11 m off the body origin, which is 31 % of
            # dyn_radius. Older .npz files have no p_body; fall back to T's
            # translation so they still load.
            p = npz["p_body"] if "p_body" in npz.files else npz["T"][:, :3, 3]
            self.tracks.append((npz["t"], p))

    def at(self, t, tol=0.5):
        """(K,3) teammate positions at time t; a track is skipped if its nearest
        pose is more than tol away in time (we then simply do not mask it)."""
        out = []
        for ts, ps in self.tracks:
            i = int(np.abs(ts - t).argmin())
            if abs(ts[i] - t) <= tol:
                out.append(ps[i])
        return np.array(out) if out else None


def process_robot(builder, robot, depth_cfg, stride, max_frames=None, mates=None, in_tag=""):
    npz = np.load(f"{VO}/occ_{SEQ}_{robot}{in_tag}.npz")
    t, T, tr = npz["t"], npz["T"], npz["tr_sigma_pos"]
    calib = load_stereo_calib(robot)
    sd = StereoDepth(calib, depth_cfg)
    idx = FrameIndex(robot)
    used = 0
    traj = []
    for i in range(0, len(t), stride):
        tval = idx.pair(t[i])
        if tval is None:
            continue
        f1 = _fname_for(robot, "infra1", tval)
        f2 = _fname_for(robot, "infra2", tval)
        if f1 is None or f2 is None:
            continue
        il = cv2.imread(os.path.join(DATA, SEQ, robot, "infra1", f1), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(os.path.join(DATA, SEQ, robot, "infra2", f2), cv2.IMREAD_GRAYSCALE)
        if il is None or ir is None:
            continue
        Z, sZ, valid = sd.depth(il, ir)
        if valid.sum() < 100:
            continue
        P_cam, sZp = sd.backproject(Z, sZ, valid, downsample=4)
        builder.integrate_frame(T[i], float(tr[i]), P_cam, sZp,
                                teammates=mates.at(t[i]) if mates else None)
        traj.append(T[i][:3, 3])
        used += 1
        if max_frames and used >= max_frames:
            break
    return used, np.array(traj)


def visualize(builder, trajs, title, path, z_lo=-0.2, z_hi=2.5):
    occ, free = builder.classify_points()
    print("  occupied voxels: %d | free voxels: %d" % (len(occ), len(free)))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    # (a) top-down: occupied colored by height, trajectories overlaid
    if len(occ):
        m = (occ[:, 2] > z_lo) & (occ[:, 2] < z_hi)
        sc = ax[0].scatter(occ[m, 0], occ[m, 1], c=occ[m, 2], s=3, cmap="viridis")
        plt.colorbar(sc, ax=ax[0], label="height z [m]", shrink=0.8)
    for tr in trajs:
        if len(tr):
            ax[0].plot(tr[:, 0], tr[:, 1], "r.-", ms=2, lw=0.6, alpha=0.8)
    ax[0].set_title("occupied voxels (top-down) + trajectory")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3)
    # (b) horizontal slice at mid height: free (light) vs occupied (dark)
    zc = 0.8
    band = 0.15
    ax[1].set_title("horizontal slice z=%.1f+-%.2f m  (free=blue, occ=red)" % (zc, band))
    if len(free):
        mf = np.abs(free[:, 2] - zc) < band
        ax[1].scatter(free[mf, 0], free[mf, 1], c="tab:blue", s=4, alpha=0.35, label="free")
    if len(occ):
        mo = np.abs(occ[:, 2] - zc) < band
        ax[1].scatter(occ[mo, 0], occ[mo, 1], c="tab:red", s=8, label="occupied")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    ax[1].axis("equal"); ax[1].grid(alpha=0.3); ax[1].legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=95)
    print("  saved", path)
    return len(occ), len(free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", default="ifo001")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--in-tag", default="",
                    help="suffix of the input vo_output/occ_<seq>_<robot><in-tag>.npz "
                         "to read, e.g. _af for fuse_and_dump.py --anchor-free --tag _af")
    ap.add_argument("--no-mask-dynamic", action="store_true",
                    help="keep observations of flying teammates in the static map")
    args = ap.parse_args()
    weighted = not args.uniform          # default weighted unless --uniform
    drones = args.drones.split(",")

    occ_cfg = OccCfg(resolution=args.res, weighted=weighted,
                     dyn_radius=0.0 if args.no_mask_dynamic else OccCfg.dyn_radius)
    builder = OccupancyBuilder(occ_cfg)
    depth_cfg = DepthCfg()
    print("=== build_occupancy drones=%s weighted=%s stride=%d res=%.2f ===" % (
        drones, weighted, args.stride, args.res))
    t0 = time.time()
    trajs = []
    ALL = ["ifo001", "ifo002", "ifo003"]
    for rob in drones:
        mates = None if args.no_mask_dynamic else Teammates(rob, ALL, in_tag=args.in_tag)
        n, traj = process_robot(builder, rob, depth_cfg, args.stride, mates=mates,
                                in_tag=args.in_tag)
        trajs.append(traj)
        print("  %s: %d keyframes integrated" % (rob, n))
    builder.finalize()
    mode = "weighted" if weighted else "uniform"
    tag = args.tag or ("_".join(drones) + "_" + mode)
    # classify/plot BEFORE exporting: write_bt collapses the tree to
    # max-likelihood and prunes it, which destroys log-odds and cell counts
    visualize(builder, trajs,
              "CoVOR->occupancy | %s | %s | %.1fs" % (",".join(drones), mode, time.time() - t0),
              f"{OUTDIR}/occ_{tag}.png")
    bt = f"{OUTDIR}/occ_{tag}.bt"
    builder.write_bt(bt); print("  wrote", bt)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation: uncertainty-weighted vs uniform (standard OctoMap) occupancy.

Demonstrates the proposal's core novelty (§4.5-4.6, Phase 3-7): forwarding the
registration covariance Sigma and depth uncertainty sigma_Z into a per-cell
weight w actually changes the map, and does so in the safety-conservative
direction (fewer confidently-free cells behind uncertain observations = fewer
false-free). Both maps are built from the SAME keyframes and depth; only w
differs (weighted: w=exp(-trSig/a)exp(-sZ^2/b); uniform: w=1).

Output: side-by-side slice maps + a difference map highlighting cells that
uniform declares FREE but the weighted map leaves UNKNOWN/OCCUPIED.

Usage: python scripts/compare_weighting.py --drones ifo001 --stride 2
"""
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.occupancy import (load_stereo_calib, StereoDepth, DepthCfg,
                             OccupancyBuilder, OccCfg)
from build_occupancy import FrameIndex, _fname_for, SEQ, DATA, VO, OUTDIR


def build_pair(drones, stride):
    """Integrate identical (pose, Sigma, depth) into a weighted and a uniform
    tree in one pass (depth computed once per frame)."""
    wb = OccupancyBuilder(OccCfg(weighted=True))
    ub = OccupancyBuilder(OccCfg(weighted=False))
    depth_cfg = DepthCfg()
    trajs = []
    for rob in drones:
        npz = np.load(f"{VO}/occ_{SEQ}_{rob}.npz")
        t, T, tr = npz["t"], npz["T"], npz["tr_sigma_pos"]
        sd = StereoDepth(load_stereo_calib(rob), depth_cfg)
        idx = FrameIndex(rob)
        traj = []
        for i in range(0, len(t), stride):
            tval = idx.pair(t[i])
            if tval is None:
                continue
            f1 = _fname_for(rob, "infra1", tval); f2 = _fname_for(rob, "infra2", tval)
            if not f1 or not f2:
                continue
            il = cv2.imread(os.path.join(DATA, SEQ, rob, "infra1", f1), cv2.IMREAD_GRAYSCALE)
            ir = cv2.imread(os.path.join(DATA, SEQ, rob, "infra2", f2), cv2.IMREAD_GRAYSCALE)
            if il is None or ir is None:
                continue
            Z, sZ, valid = sd.depth(il, ir)
            if valid.sum() < 100:
                continue
            P_cam, sZp = sd.backproject(Z, sZ, valid, downsample=4)
            wb.integrate_frame(T[i], float(tr[i]), P_cam, sZp)
            ub.integrate_frame(T[i], float(tr[i]), P_cam, sZp)
            traj.append(T[i][:3, 3])
        trajs.append(np.array(traj))
    wb.finalize(); ub.finalize()
    return wb, ub, trajs


def raster(builder, zc, band, res, bounds):
    """Rasterize a horizontal slice into a label grid: 1=occ, -1=free, 0=unknown."""
    x0, x1, y0, y1 = bounds
    nx = int((x1 - x0) / res) + 1; ny = int((y1 - y0) / res) + 1
    g = np.zeros((ny, nx), np.int8)
    occ, free = builder.classify_points()
    for pts, lab in ((free, -1), (occ, 1)):   # occ overrides free
        if not len(pts):
            continue
        m = np.abs(pts[:, 2] - zc) < band
        for p in pts[m]:
            ix = int((p[0] - x0) / res); iy = int((p[1] - y0) / res)
            if 0 <= ix < nx and 0 <= iy < ny:
                g[iy, ix] = lab
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", default="ifo001")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--zc", type=float, default=0.8)
    args = ap.parse_args()
    drones = args.drones.split(",")
    print("building weighted + uniform for", drones, "stride", args.stride)
    wb, ub, trajs = build_pair(drones, args.stride)

    wo, wf = wb.classify_points(); uo, uf = ub.classify_points()
    print("weighted: occ=%d free=%d | uniform: occ=%d free=%d" % (
        len(wo), len(wf), len(uo), len(uf)))

    res = 0.10; band = 0.15; zc = args.zc
    bounds = (-6, 8, -6, 8)
    gw = raster(wb, zc, band, res, bounds)
    gu = raster(ub, zc, band, res, bounds)
    # false-free suppressed: uniform says FREE, weighted says NOT free (unknown/occ)
    supp = (gu == -1) & (gw != -1)
    n_supp = int(supp.sum()); n_ufree = int((gu == -1).sum())
    print("slice z=%.1f: uniform-free cells=%d | suppressed-by-weighting=%d (%.1f%%)" % (
        zc, n_ufree, n_supp, 100 * n_supp / max(n_ufree, 1)))

    ext = [bounds[0], bounds[1], bounds[2], bounds[3]]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#2b6cb0", "#f5f5f5", "#c0392b"])  # free, unknown, occ
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.4))
    for a, g, ttl in ((ax[0], gw, "weighted (w=exp(-trS/a)exp(-sZ^2/b))"),
                      (ax[1], gu, "uniform (standard OctoMap, w=1)")):
        a.imshow(g + 1, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, aspect="equal")
        for tr in trajs:
            if len(tr):
                a.plot(tr[:, 0], tr[:, 1], "k.-", ms=1.5, lw=0.5, alpha=0.6)
        a.set_title(ttl); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    d = np.zeros_like(gw)
    d[supp] = 1
    ax[2].imshow(gu + 1, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2,
                 aspect="equal", alpha=0.35)
    ys, xs = np.where(supp)
    ax[2].scatter(bounds[0] + xs * res, bounds[2] + ys * res, c="#e67e22", s=6,
                  label="uniform=free but weighted!=free\n(false-free suppressed)")
    ax[2].set_title("suppressed by uncertainty weighting: %d cells (%.1f%% of uniform-free)"
                    % (n_supp, 100 * n_supp / max(n_ufree, 1)))
    ax[2].set_xlabel("x [m]"); ax[2].set_ylabel("y [m]"); ax[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Uncertainty-aware vs standard occupancy | %s | slice z=%.1f m" %
                 (",".join(drones), zc))
    plt.tight_layout()
    out = f"{OUTDIR}/compare_weighting_{'_'.join(drones)}.png"
    plt.savefig(out, dpi=95); print("saved", out)


if __name__ == "__main__":
    main()

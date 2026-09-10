#!/usr/bin/env python3
"""Part A: inspect a scanned OBJ, freeze its world transform, voxelise the GT.

    python scripts/synth/build_gt_voxel.py --obj <path> --name room909 [--res 0.10]

Writes to results/synth_<name>/:
    mesh_config.json   units / up-axis / floor / T_mesh_world  (the frozen decision)
    gt_voxel.npz       dense GT labels on the mapper's grid  (see mesh_gt.build_gt_grid)
    gt_voxel.png       3D view + z-slices, for the eyeball check
    mesh_report.txt    the inspection dump

Nothing downstream may run before the numbers in mesh_config.json are confirmed:
if the up-axis or the unit is wrong, every later metric is meaningless (Part A
gate in the task brief).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.synth import mesh_gt as MG

ROOT = "/src/gs25058/cr_RNE/covor_slam"


def figure(lab, ijk_min, res, out, name, seed):
    occ_c, _ = MG.centers_of(lab, ijk_min, res, MG.OCC)
    fig = plt.figure(figsize=(16, 9.5))
    # (a) 3D scatter of occupied, subsampled
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    s = occ_c[::max(1, len(occ_c) // 40000)]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 2], s=0.5, cmap="viridis")
    ax.set_title("GT occupied voxels (3D)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    try:
        ax.set_box_aspect(np.ptp(occ_c, axis=0))
    except Exception:
        pass
    # (b) top-down occupied density
    ax = fig.add_subplot(2, 3, 2)
    m = (occ_c[:, 2] > 0.05) & (occ_c[:, 2] < 2.0)
    ax.scatter(occ_c[m, 0], occ_c[m, 1], c=occ_c[m, 2], s=1, cmap="viridis")
    ax.set_title("occupied, 0.05 < z < 2.0 m (top-down)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal"); ax.grid(alpha=.3)
    # (c)-(f) horizontal slices: free (blue) vs occupied (red)
    for n, zc in enumerate([0.5, 1.0, 1.25, 1.5]):
        ax = fig.add_subplot(2, 3, 3 + n) if n < 1 else fig.add_subplot(2, 3, 3 + n)
        k = int(np.floor(zc / res)) - ijk_min[2]
        if not (0 <= k < lab.shape[2]):
            continue
        sl = lab[:, :, k]
        ii, jj = np.nonzero(sl == MG.FREE)
        ax.scatter((ii + ijk_min[0] + .5) * res, (jj + ijk_min[1] + .5) * res,
                   c="tab:blue", s=1.5, alpha=.35, label="GT free")
        ii, jj = np.nonzero(sl == MG.OCC)
        ax.scatter((ii + ijk_min[0] + .5) * res, (jj + ijk_min[1] + .5) * res,
                   c="tab:red", s=3, label="GT occupied")
        if n == 0:
            ax.plot([seed[0]], [seed[1]], "k*", ms=12, label="fill seed")
        ax.set_title("z = %.2f m slice" % zc)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.axis("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("%s | GT voxel grid, res = %.2f m | occ=%d free=%d unknown=%d"
                 % (name, res, (lab == MG.OCC).sum(), (lab == MG.FREE).sum(),
                    (lab == MG.UNKNOWN).sum()))
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print("  saved", out)


def figure_enclosure(encl, lab, ijk_min, res, out, name, thresh):
    """The one free parameter of the GT domain, shown rather than asserted."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, zc in zip(axes.ravel(), [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]):
        k = int(np.floor(zc / res)) - ijk_min[2]
        if not (0 <= k < lab.shape[2]):
            continue
        sl = encl[:, :, k]
        ii, jj = np.mgrid[0:sl.shape[0], 0:sl.shape[1]]
        X = (ii + ijk_min[0] + .5) * res
        Y = (jj + ijk_min[1] + .5) * res
        im = ax.pcolormesh(X, Y, np.where(sl < 0, np.nan, sl), vmin=0, vmax=1,
                           cmap="RdYlGn")
        ax.contour(X, Y, np.where(sl < 0, 0, sl), levels=[thresh], colors="k",
                   linewidths=0.8)
        o = lab[:, :, k] == MG.OCC
        ax.scatter(X[o], Y[o], c="k", s=0.7)
        ax.set_title("z = %.1f m" % zc); ax.axis("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6,
                 label="enclosure fraction (black contour = %.2f threshold)" % thresh)
    fig.suptitle("%s | enclosure fraction: how the GT free domain is defined" % name)
    plt.savefig(out, dpi=100)
    print("  saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--name", required=True, help="space name -> results/synth_<name>/")
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--up-axis", default=None, help="override the detected up-axis")
    ap.add_argument("--up-sign", type=int, default=None)
    ap.add_argument("--encl-thresh", type=float, default=0.90,
                    help="enclosure fraction above which a cell counts as room interior")
    ap.add_argument("--encl-rays", type=int, default=128)
    ap.add_argument("--close-iter", type=int, default=2,
                    help="binary closing radius (voxels) applied to the enclosure mask")
    args = ap.parse_args()

    outdir = os.path.join(ROOT, "results", "synth_" + args.name)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    mesh = MG.load(args.obj)
    info = MG.inspect(mesh)
    print("=== mesh inspection (%.1fs) ===" % (time.time() - t0))
    for k in ("n_vertices", "n_faces", "extents", "bounds", "area", "watertight",
              "winding_consistent", "euler_number", "up_axis_guess", "up_sign",
              "floor_plane", "ceiling_plane", "room_height"):
        print("  %-20s %s" % (k, info.get(k)))
    print("  dominant planes per axis:")
    for a, pl in info["planes"].items():
        print("    %s: %s" % (a, ["%.2f m / %.1f m2" % (c, v) for c, v in pl[:3]]))

    T = MG.mesh_to_world(info, args.up_axis, args.up_sign)
    mw = MG.apply_transform(mesh, T)
    b = mw.bounds
    print("=== world frame (z up, floor z=0, footprint centred) ===")
    print("  T_mesh_world =\n", np.array2string(T, precision=4))
    print("  world bounds: x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]"
          % (b[0, 0], b[1, 0], b[0, 1], b[1, 1], b[0, 2], b[1, 2]))

    t0 = time.time()
    occ = MG.voxelize_surface(mw, args.res)
    print("=== voxelisation (%.1fs) === occupied cells: %d" % (time.time() - t0, len(occ)))

    # Seed the interior fill at the room centroid, at 1.2 m -- inside the
    # flight band (1.0-1.5 m) the trajectories will use.
    seed = np.array([0.0, 0.0, 1.2])
    scene = MG.raycasting_scene(mw)
    t0 = time.time()
    lab, ijk_min, ginfo, encl = MG.build_gt_grid(
        occ, args.res, seed, scene, thresh=args.encl_thresh,
        n_dirs=args.encl_rays, z_band=(0.0, info["room_height"]),
        close_iter=args.close_iter)
    print("=== GT labelling (%.1fs) ===" % (time.time() - t0))
    print("  grid shape %s  ijk_min %s" % (lab.shape, ijk_min.tolist()))
    print("  GT occupied %d | GT free (interior) %d | unknown %d"
          % (ginfo["n_occ"], ginfo["n_free"], ginfo["n_unknown"]))
    print("  interior touches domain boundary: %s (must be False)"
          % ginfo["touches_boundary"])
    print("  FREE cell count vs enclosure threshold: %s" % ginfo["free_vs_thresh"])
    vox = args.res ** 3
    print("  free volume %.1f m3, occupied volume %.1f m3"
          % (ginfo["n_free"] * vox, ginfo["n_occ"] * vox))

    cfg = dict(obj=os.path.abspath(args.obj), name=args.name, res=args.res,
               T_mesh_world=T.tolist(), seed=seed.tolist(),
               world_bounds=b.tolist(), inspect=info, grid=ginfo)
    MG.write_config(os.path.join(outdir, "mesh_config.json"), cfg)
    MG.save_gt(os.path.join(outdir, "gt_voxel.npz"), lab, ijk_min, args.res, T,
               dict(cfg, inspect={k: v for k, v in info.items() if k != "planes"}),
               enclosure=encl)
    with open(os.path.join(outdir, "mesh_report.txt"), "w") as f:
        f.write(json.dumps(cfg, indent=2, sort_keys=True))
    figure(lab, ijk_min, args.res, os.path.join(outdir, "gt_voxel.png"),
           args.name, seed)
    figure_enclosure(encl, lab, ijk_min, args.res,
                     os.path.join(outdir, "gt_enclosure.png"), args.name,
                     args.encl_thresh)
    print("  wrote", outdir)


if __name__ == "__main__":
    main()

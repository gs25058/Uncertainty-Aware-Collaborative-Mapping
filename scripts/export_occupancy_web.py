#!/usr/bin/env python3
"""Export an occupancy map to the self-contained WebGL2 viewer (web/*.html).

Builds the map exactly like build_occupancy.py (same OccupancyBuilder, same
fixes: DDA traversal, tau_occ = l_occ, teammate masking) and then serializes the
CLASSIFIED voxels -- not the .bt export, which is destructive -- into
web/occupancy_viewer.template.html.

Payload (window.__OCC__), all little-endian:
  res              voxel size (m)
  occ_idx_b64      base64 of int16 [i,j,k]*N voxel indices (occupied)
  occ_col_b64      base64 of uint8 [r,g,b]*N   (viridis by height)
  free_idx_b64     base64 of int16 [i,j,k]*M   (free, subsampled)
  zmin/zmax        colour-scale bounds (robust percentiles)
  center/radius    camera framing
  trajs            [{name, color, pts}]        drone trajectories

Usage:
  python scripts/export_occupancy_web.py --drones ifo001,ifo002,ifo003 --stride 3
  python scripts/export_occupancy_web.py --drones ifo001 --uniform --out /tmp/uniform.html
"""
import os
import sys
import json
import time
import base64
import argparse
import numpy as np

ROOT = "/src/gs25058/cr_RNE/covor_slam"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from covor.occupancy import OccupancyBuilder, OccCfg, DepthCfg          # noqa: E402
from build_occupancy import process_robot, Teammates, SEQ              # noqa: E402

TEMPLATE = os.path.join(ROOT, "web", "occupancy_viewer.template.html")
ALL = ["ifo001", "ifo002", "ifo003"]
DRONE_COLORS = {"ifo001": [0.96, 0.65, 0.14],   # amber
                "ifo002": [0.18, 0.83, 0.75],   # teal
                "ifo003": [0.88, 0.36, 0.78]}   # magenta

# matplotlib's viridis, 9 stops -- enough for a smooth 0-1 ramp by interpolation
VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float64)


def viridis(t):
    """t in [0,1] (N,) -> uint8 RGB (N,3), linear interpolation over the stops."""
    t = np.clip(t, 0.0, 1.0) * (len(VIRIDIS) - 1)
    i = np.floor(t).astype(int)
    i = np.minimum(i, len(VIRIDIS) - 2)
    f = (t - i)[:, None]
    return np.round(VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f).astype(np.uint8)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def to_idx(points, res):
    """Voxel centers (N,3) in m -> int16 voxel indices (the viewer re-adds +0.5)."""
    idx = np.floor(points / res + 1e-6).astype(np.int64)
    keep = (np.abs(idx) < 32767).all(axis=1)
    return idx[keep].astype(np.int16), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", default="ifo001,ifo002,ifo003")
    ap.add_argument("--uniform", action="store_true",
                    help="w=1 (standard OctoMap) instead of uncertainty-weighted")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--no-mask-dynamic", action="store_true")
    ap.add_argument("--max-free", type=int, default=60000,
                    help="cap on free voxels sent to the browser (subsampled)")
    ap.add_argument("--title", default="CoVOR Occupancy Map — 3D Viewer")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "occupancy_viewer.html"))
    args = ap.parse_args()

    weighted = not args.uniform
    drones = args.drones.split(",")
    cfg = OccCfg(resolution=args.res, weighted=weighted,
                 dyn_radius=0.0 if args.no_mask_dynamic else OccCfg.dyn_radius)
    builder = OccupancyBuilder(cfg)

    print("=== export_occupancy_web drones=%s weighted=%s stride=%d res=%.2f ===" % (
        drones, weighted, args.stride, args.res))
    t0 = time.time()
    trajs = []
    for rob in drones:
        mates = None if args.no_mask_dynamic else Teammates(rob, ALL)
        n, traj = process_robot(builder, rob, DepthCfg(), args.stride, mates=mates)
        print("  %s: %d keyframes integrated" % (rob, n))
        if len(traj):
            trajs.append({"name": rob,
                          "color": DRONE_COLORS.get(rob, [0.7, 0.7, 0.7]),
                          "pts": [[round(float(v), 3) for v in p] for p in traj]})
    builder.finalize()

    # classify BEFORE any .bt export (write_bt collapses the tree)
    occ, free = builder.classify_points()
    print("  occupied voxels: %d | free voxels: %d  (%.1fs)" % (
        len(occ), len(free), time.time() - t0))

    occ_idx, keep = to_idx(occ, args.res)
    occ = occ[keep]
    # robust colour bounds: ignore the few stray voxels far above/below the room
    z = occ[:, 2] if len(occ) else np.zeros(1)
    zlo, zhi = (float(np.percentile(z, 1)), float(np.percentile(z, 99))) if len(occ) else (0.0, 1.0)
    if zhi - zlo < 1e-3:
        zhi = zlo + 1.0
    col = viridis((z - zlo) / (zhi - zlo))

    if len(free) > args.max_free:                       # deterministic thinning
        sel = np.linspace(0, len(free) - 1, args.max_free).astype(int)
        free = free[sel]
    free_idx, _ = to_idx(free, args.res)

    # camera framing from the trajectory + the bulk of the occupied cloud
    pts = np.vstack([np.array(t["pts"]) for t in trajs]) if trajs else occ
    if len(occ):
        core = occ[(occ[:, 2] > zlo) & (occ[:, 2] < zhi)]
        pts = np.vstack([pts, core]) if len(core) else pts
    center = np.percentile(pts, 50, axis=0)
    radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 97))

    payload = {
        "res": args.res,
        "occ_idx_b64": b64(occ_idx),
        "occ_col_b64": b64(col),
        "free_idx_b64": b64(free_idx),
        "zmin": round(float(z.min()) if len(occ) else 0.0, 2),
        "zmax": round(float(z.max()) if len(occ) else 1.0, 2),
        "czmin": round(zlo, 2), "czmax": round(zhi, 2),
        "center": [round(float(v), 2) for v in center],
        "radius": round(radius, 2),
        "trajs": trajs,
    }
    mode = "uncertainty-weighted" if weighted else "uniform (standard OctoMap)"
    subtitle = args.subtitle or (
        "%d-drone collaborative map on MILUV <code>%s</code>, built from UWB-fused "
        "poses with %s log-odds." % (len(drones), SEQ, mode))

    html = (open(TEMPLATE).read()
            .replace("/*__OCC_DATA__*/null", json.dumps(payload, separators=(",", ":")))
            .replace("__TITLE__", args.title)
            .replace("__SUBTITLE__", subtitle))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print("  wrote %s (%.2f MB)" % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()

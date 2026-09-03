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
  occ_ao_b64       base64 of uint8 [ao]*N      (baked AO: occupied ratio over the
                                                 26-cell neighborhood, 0=isolated,
                                                 255=fully enclosed; viewer applies
                                                 it as an ambient-only darkening
                                                 multiplier, computed once here so
                                                 there is no runtime cost)
  occ_mask_b64     base64 of uint8 [mask]*N    (per-voxel 6-bit face-visibility
                                                 mask, bit order +x,-x,+y,-y,+z,-z
                                                 matching cube()'s face order in
                                                 the vertex shader; bit=1 means a
                                                 same-size occupied neighbor sits
                                                 against that face, so the viewer
                                                 degenerates it out of the clip
                                                 volume instead of rasterizing it)
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

# matplotlib stops, hardcoded so the exporter never has to import matplotlib.
VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float64)
CIVIDIS = np.array([
    [0, 34, 78], [18, 53, 112], [59, 73, 108], [87, 93, 109], [112, 113, 115],
    [138, 134, 120], [165, 156, 116], [195, 179, 105], [225, 204, 85], [254, 232, 56],
], dtype=np.float64)


def colormap(t, stops):
    """t in [0,1] (N,) -> uint8 RGB (N,3), linear interpolation over `stops`."""
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.floor(t).astype(int)
    i = np.minimum(i, len(stops) - 2)
    f = (t - i)[:, None]
    return np.round(stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def viridis(t):
    return colormap(t, VIRIDIS)


def cividis(t):
    return colormap(t, CIVIDIS)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def to_idx(points, res):
    """Voxel centers (N,3) in m -> int16 voxel indices (the viewer re-adds +0.5)."""
    idx = np.floor(points / res + 1e-6).astype(np.int64)
    keep = (np.abs(idx) < 32767).all(axis=1)
    return idx[keep].astype(np.int16), keep


_NEI26 = np.array([[dx, dy, dz]
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)

# must match the face order of cube() in occupancy_viewer.template.html: +x,-x,+y,-y,+z,-z
_FACE6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)


def bake_ao_mask(occ_idx):
    """occ_idx (N,3) int -> (ao uint8 (N,), mask uint8 (N,)).
    ao: fraction of the 26-cell neighborhood that is also occupied, scaled to
    0..255. mask: bit i set (face order above) when the axis-neighbor on that
    side is occupied, i.e. that face is permanently hidden. Pure lookups over
    the already-classified grid, no interpolation of occupancy values."""
    n = len(occ_idx)
    if n == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)
    idx64 = occ_idx.astype(np.int64)
    keys = set(map(tuple, idx64.tolist()))
    cnt = np.zeros(n, dtype=np.int32)
    for dx, dy, dz in _NEI26.tolist():
        shifted = idx64 + (dx, dy, dz)
        cnt += np.fromiter((t in keys for t in map(tuple, shifted.tolist())), dtype=bool, count=n)
    ao = np.round(cnt.astype(np.float64) / 26.0 * 255.0).astype(np.uint8)

    mask = np.zeros(n, dtype=np.uint8)
    for bit, (dx, dy, dz) in enumerate(_FACE6.tolist()):
        shifted = idx64 + (dx, dy, dz)
        hit = np.fromiter((t in keys for t in map(tuple, shifted.tolist())), dtype=bool, count=n)
        mask |= (hit.astype(np.uint8) << bit)
    return ao, mask


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
    ap.add_argument("--in-tag", default="",
                    help="suffix of the input vo_output/occ_<seq>_<robot><in-tag>.npz "
                         "to read, e.g. _af for fuse_and_dump.py --anchor-free --tag _af")
    ap.add_argument("--color-mode", choices=["height", "uncertainty", "class"], default="height",
                    help="voxel colour source. 'uncertainty' (w or tr(Sigma), cividis) and "
                         "'class' are hooks for future per-voxel fields this exporter does not "
                         "currently produce -- they fall back to 'height' with a warning.")
    args = ap.parse_args()

    color_mode = args.color_mode
    if color_mode != "height":
        # hook only: no per-voxel w / tr(Sigma) or semantic-class field is
        # threaded through classify_points() today. Wire one through (e.g. via
        # OccCfg.track_sigma_attribution + sigma_attribution()) before enabling.
        print("  WARNING: --color-mode=%s requested but no such per-voxel field "
              "is exported yet; falling back to height." % color_mode)
        color_mode = "height"

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
        mates = None if args.no_mask_dynamic else Teammates(rob, ALL, in_tag=args.in_tag)
        n, traj = process_robot(builder, rob, DepthCfg(), args.stride, mates=mates,
                                in_tag=args.in_tag)
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
    # color_mode is always "height" today (see the fallback warning above);
    # this branch is the hook uncertainty/class colouring plugs into.
    if color_mode == "height":
        col = viridis((z - zlo) / (zhi - zlo))
        color_label = "Voxel height (viridis)"
    else:
        raise AssertionError("unreachable: color_mode falls back to height above")
    t_ao = time.time()
    ao, face_mask = bake_ao_mask(occ_idx)
    hidden_faces = int(sum(bin(m).count("1") for m in face_mask.tolist()))
    print("  baked AO + face mask for %d voxels, %d/%d faces hidden (%.1fs)" % (
        len(occ_idx), hidden_faces, len(occ_idx) * 6, time.time() - t_ao))

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
        "occ_ao_b64": b64(ao),
        "occ_mask_b64": b64(face_mask),
        "free_idx_b64": b64(free_idx),
        "zmin": round(float(z.min()) if len(occ) else 0.0, 2),
        "zmax": round(float(z.max()) if len(occ) else 1.0, 2),
        "czmin": round(zlo, 2), "czmax": round(zhi, 2),
        "center": [round(float(v), 2) for v in center],
        "radius": round(radius, 2),
        "trajs": trajs,
        "color_mode": color_mode,
        "color_label": color_label,
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

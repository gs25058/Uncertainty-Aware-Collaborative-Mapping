#!/usr/bin/env python3
"""Repair a Polycam scan's floor: remove the slow drift and the doubled floor sheets.

    python scripts/scan/repair_floor.py \
        --obj "meshes/2026-09-15/2026. 9. 15.obj" \
        --mesh-config results/synth_corridor915/mesh_config.json \
        --out "meshes/2026-09-15/2026. 9. 15.floorfix.obj"

WHY. The entry map reads the floor as a single plane (DESIGN §3-6) and starts
the body band one voxel above it. The scan's floor is not that. Measured on the
2026-09-15 corridor (RESULTS_sgbm.md §14):

  - it drifts slowly: the dominant floor sits near 0.00 m in the south and near
    -0.10 m over much of the rest, a Polycam-style global drift, while the
    ceiling stays 2.50-2.53 m above it everywhere (locally metric, globally
    bent);
  - where two scan passes overlap it is DOUBLED: a second floor-like sheet
    0.11-0.35 m below the first (median gap 0.16 m) over y -7..-4 in the
    corridor, in the lobby and in room 201, with holes in the upper sheet
    through which the lower one shows;
  - mesh_to_world puts the dominant floor plane exactly on z = 0, a voxel
    boundary, so +-1 cm of floor noise flips the floor between two voxel rows.

Together these produce steps that do not exist: the corridor is cut twice
(y -8.25..-6.85 and -5.35..-4.45) by rows of "obstacles" 0.1-0.2 m tall, and
the entry map reports 62 % of the passable floor as unreachable. A real
walker sees a flat corridor (the textured renders show it).

WHAT. Three steps, all vertical, so no vertex moves in x or y:

  1. FLOOR FIELD D(x, y). Floor-like faces are horizontal (|n_z| > 0.9; the
     scan's chunks disagree on winding, so both orientations count). The
     topmost of them at or below EST_TOP = +0.08 m is sampled from above on a
     5 cm lattice -- mesh_to_world puts the DOMINANT floor plane at z = 0, so a
     floor never sits 8 cm above it, while stair treads (+0.15 m and up) and
     furniture tops do: sampling below that line is what keeps the stairs out
     of the floor estimate (measured: without it the landing's tiles read
     +0.11 m and the first tread was flattened). Each 0.5 m tile takes the
     MODE of the samples (1 cm bins); a tile more than OUTLIER = 6 cm from the
     median of its 5x5-tile (2.5 m) neighbourhood is dropped (a patch where the
     lower sheet of a doubled floor dominates); a 3x3 median smooths the rest;
     empty tiles take the nearest tile. Bilinear between tile centres.
  2. SNAP. Every vertex of a floor-like face whose height is within
     [D - 0.45, D + 0.10] is set to D. That puts both sheets of a doubled floor
     on one surface and flattens mm-cm floor noise. Low horizontal tops within
     0.10 m of the floor (a doormat, a threshold) are flattened with it; the
     band starts at 0.10 m, so they never blocked a walker's band before
     either. Anything taller (the stair treads at 0.15 m, the 0.26 m object in
     the lobby) keeps its height.
  3. DE-DRIFT. Every vertex drops by D(x, y), so the floor lands on one plane
     and walls, stairs, furniture and ceiling keep their height ABOVE the
     local floor.

The output is the input OBJ with only the up-coordinate of `v` lines
rewritten; faces, UVs, normals, materials and texture references are byte-for-
byte the source's. Nothing about the scan's appearance changes except where the
floor sits.

WHAT THIS IS NOT. It does not fix the model. The entry pipeline keeps z_min =
0.10 (a design value the user chose to keep on 2026-09-15; changing it after
seeing results would be §9-4's threshold-by-result). This repairs the GT
scene the model is evaluated in, the way a scan would be cleaned before use.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")

NZ_MIN = 0.9          # |n_z| of a floor-like face
BAND = (-0.6, 0.35)   # world z window of faces that may be SNAPPED to the floor
EST_TOP = 0.08        # floor-field samples are taken at or below this height
EST_BOTTOM = -0.35    # ... and above this one
OUTLIER = 0.06        # m, tile vs its 5x5-tile median
LATTICE = 0.05        # m, top-surface sampling step
TILE = 0.5            # m, floor-field tile
MIN_HITS = 20         # lattice samples a tile needs to estimate its own floor
SNAP_BELOW = 0.45     # floor-like vertices this far below D are a doubled sheet
SNAP_ABOVE = 0.10     # ... and this far above D are floor noise / flat covers


def read_obj(path):
    """-> (lines, vertex line indices, V (n,3) mesh coords, faces (m,3) 0-based)."""
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        lines = f.readlines()
    vidx, V, F = [], [], []
    for n, s in enumerate(lines):
        if s.startswith("v "):
            vidx.append(n)
            V.append([float(t) for t in s.split()[1:4]])
        elif s.startswith("f "):
            ids = [int(t.split("/")[0]) for t in s.split()[1:]]
            ids = [i - 1 if i > 0 else len(V) + i for i in ids]
            for k in range(1, len(ids) - 1):              # fan-triangulate
                F.append([ids[0], ids[k], ids[k + 1]])
    return lines, np.array(vidx), np.array(V, float), np.array(F, np.int64)


def up_axis(T):
    """(mesh axis that maps to world z, its sign) -- T only permutes/negates."""
    a = int(np.argmax(np.abs(T[2, :3])))
    return a, float(np.sign(T[2, a]))


def floor_field(Vw, F, lattice=LATTICE, tile=TILE):
    """Tile-wise floor height from the topmost floor-like surface.

    Returns (D(x, y) callable, report dict).
    """
    import open3d as o3d
    tri = Vw[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 0
    nz = np.zeros(len(F))
    nz[ok] = np.abs(n[ok, 2]) / nn[ok]
    cz = tri[:, :, 2].mean(1)
    cand = (nz > NZ_MIN) & (cz > BAND[0]) & (cz < BAND[1])
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(Vw), o3d.utility.Vector3iVector(F[cand].astype(np.int32)))))
    lo, hi = Vw[:, :2].min(0), Vw[:, :2].max(0)
    xs = np.arange(lo[0], hi[0], lattice) + lattice / 2
    ys = np.arange(lo[1], hi[1], lattice) + lattice / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, EST_TOP)], 1)
    rays = np.concatenate([o, np.tile([0.0, 0.0, -1.0], (X.size, 1))], 1).astype(np.float32)
    t = rs.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    z = EST_TOP - t
    hit = np.isfinite(z) & (z > EST_BOTTOM)
    # tiles
    tx = np.arange(lo[0], hi[0] + tile, tile)
    ty = np.arange(lo[1], hi[1] + tile, tile)
    ti = np.clip(((X.ravel() - lo[0]) // tile).astype(int), 0, len(tx) - 2)
    tj = np.clip(((Y.ravel() - lo[1]) // tile).astype(int), 0, len(ty) - 2)
    G = np.full((len(tx) - 1, len(ty) - 1), np.nan)
    bins = np.arange(EST_BOTTOM, EST_TOP + 0.01, 0.01)
    for a in range(G.shape[0]):
        for b in range(G.shape[1]):
            m = hit & (ti == a) & (tj == b)
            if m.sum() >= MIN_HITS:
                h, e = np.histogram(z[m], bins=bins)
                k = int(np.argmax(h))
                sel = z[m][(z[m] >= e[k]) & (z[m] < e[k + 1])]
                G[a, b] = float(np.median(sel))
    have = np.isfinite(G)
    # drop tiles that disagree with their 2.5 m neighbourhood
    n_out = 0
    for a in range(G.shape[0]):
        for b in range(G.shape[1]):
            if have[a, b]:
                w = G[max(a - 2, 0):a + 3, max(b - 2, 0):b + 3]
                if abs(G[a, b] - np.nanmedian(w)) > OUTLIER:
                    n_out += 1
    if n_out:
        keep = have.copy()
        for a in range(G.shape[0]):
            for b in range(G.shape[1]):
                if have[a, b]:
                    w = G[max(a - 2, 0):a + 3, max(b - 2, 0):b + 3]
                    keep[a, b] = abs(G[a, b] - np.nanmedian(w)) <= OUTLIER
        G = np.where(keep, G, np.nan)
        have = keep
    # 3x3 median over tiles that have their own estimate
    Gm = G.copy()
    for a in range(G.shape[0]):
        for b in range(G.shape[1]):
            if have[a, b]:
                w = G[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2]
                Gm[a, b] = float(np.nanmedian(w))
    # empty tiles take the nearest tile's value
    _, (ii, jj) = ndimage.distance_transform_edt(~have, return_indices=True)
    Gf = Gm[ii, jj]
    cx = tx[:-1] + tile / 2
    cy = ty[:-1] + tile / 2

    def D(x, y):
        fx = np.clip((np.asarray(x) - cx[0]) / tile, 0, len(cx) - 1)
        fy = np.clip((np.asarray(y) - cy[0]) / tile, 0, len(cy) - 1)
        i0 = np.minimum(np.floor(fx).astype(int), len(cx) - 2) if len(cx) > 1 else np.zeros_like(fx, int)
        j0 = np.minimum(np.floor(fy).astype(int), len(cy) - 2) if len(cy) > 1 else np.zeros_like(fy, int)
        u = fx - i0
        v = fy - j0
        i1 = np.minimum(i0 + 1, len(cx) - 1)
        j1 = np.minimum(j0 + 1, len(cy) - 1)
        return ((1 - u) * (1 - v) * Gf[i0, j0] + u * (1 - v) * Gf[i1, j0]
                + (1 - u) * v * Gf[i0, j1] + u * v * Gf[i1, j1])

    rep = dict(n_floor_like_faces=int(cand.sum()), n_lattice_hits=int(hit.sum()),
               n_tiles=int(G.size), n_tiles_estimated=int(have.sum()),
               n_tiles_dropped_as_outliers=int(n_out),
               field_min=float(np.nanmin(Gf)), field_max=float(np.nanmax(Gf)),
               field_p5=float(np.nanpercentile(Gm[have], 5)),
               field_p95=float(np.nanpercentile(Gm[have], 95)))
    return D, cand, rep


def repair(Vw, F):
    """-> (new world z per vertex, report)."""
    D, cand, rep = floor_field(Vw, F)
    d = D(Vw[:, 0], Vw[:, 1])
    floor_v = np.zeros(len(Vw), bool)
    floor_v[np.unique(F[cand])] = True
    rel = Vw[:, 2] - d
    snap = floor_v & (rel >= -SNAP_BELOW) & (rel <= SNAP_ABOVE)
    z = Vw[:, 2] - d                     # de-drift: floor field -> 0
    z[snap] = 0.0                        # both sheets of a doubled floor on one plane
    dz = z - Vw[:, 2]
    rep.update(n_vertices=int(len(Vw)), n_floor_vertices_snapped=int(snap.sum()),
               n_snapped_from_below_2cm=int((snap & (rel < -0.02)).sum()),
               max_abs_dz=float(np.abs(dz).max()),
               n_moved_over_1cm=int((np.abs(dz) > 0.01).sum()))
    return z, rep


def write_obj(lines, vidx, Vm, out):
    new = list(lines)
    for n, v in zip(vidx, Vm):
        head = lines[n].split()
        tail = head[4:]                  # keep vertex colours etc. if present
        new[n] = "v %.6f %.6f %.6f%s\n" % (v[0], v[1], v[2],
                                           (" " + " ".join(tail)) if tail else "")
    with open(out, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.writelines(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--mesh-config", required=True,
                    help="mesh_config.json whose T_mesh_world defines the world z")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    T = np.array(json.load(open(args.mesh_config))["T_mesh_world"], float)
    lines, vidx, Vm, F = read_obj(args.obj)
    Vw = Vm @ T[:3, :3].T + T[:3, 3]
    z, rep = repair(Vw, F)
    a, sgn = up_axis(T)
    Vm2 = Vm.copy()
    Vm2[:, a] += (z - Vw[:, 2]) / sgn
    write_obj(lines, vidx, Vm2, args.out)
    rep.update(source=os.path.abspath(args.obj), out=os.path.abspath(args.out),
               mesh_config=os.path.abspath(args.mesh_config),
               params=dict(NZ_MIN=NZ_MIN, BAND=BAND, EST_TOP=EST_TOP,
                           EST_BOTTOM=EST_BOTTOM, OUTLIER=OUTLIER,
                           LATTICE=LATTICE, TILE=TILE,
                           MIN_HITS=MIN_HITS, SNAP_BELOW=SNAP_BELOW,
                           SNAP_ABOVE=SNAP_ABOVE))
    rp = os.path.splitext(args.out)[0] + ".report.json"
    json.dump(rep, open(rp, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()

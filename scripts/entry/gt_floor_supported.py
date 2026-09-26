#!/usr/bin/env python3
"""Extend a GT grid with the air a standing person would occupy over scanned floor.

    python scripts/entry/gt_floor_supported.py --name corridor915f

DESIGN_traversability.md §8. The GT's free domain is "enclosure >= 0.9"
(covor/synth/mesh_gt.build_gt_grid). A stair landing whose floor is scanned but
whose upper stairwell and far walls are not falls short of that (measured
0.76-0.83) and stays unknown, although a person plainly stands there.

Rule: in every column where the mesh has a horizontal floor surface
(|n_z| > 0.9, z in FLOOR_BAND), UNKNOWN cells from that surface up to standing
height (1.90 m) become FREE if they are 6-connected to the existing FREE region.
Mesh surfaces are already OCCUPIED voxels, so the fill cannot cross a wall.
Columns without scanned floor stay as they were. Writes gt_voxel_fs.npz next to
gt_voxel.npz; the original is not touched.
"""
import argparse
import json
import os
import sys

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np
from scipy import ndimage

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.synth import dataset as DS, mesh_gt as MG
from covor.synth.config import SynthCfg

FLOOR_BAND = (-0.3, 0.6)   # world z window for floor-like faces (floor at 0.05)
H_STAND = 1.90


def floor_height(mesh, xs, ys):
    """Topmost horizontal surface in FLOOR_BAND under each (x, y), NaN if none."""
    import open3d as o3d
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    n = mesh.face_normals
    cz = mesh.triangles_center[:, 2]
    cand = (np.abs(n[:, 2]) > 0.9) & (cz > FLOOR_BAND[0]) & (cz < FLOOR_BAND[1])
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F[cand].astype(np.int32)))))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, FLOOR_BAND[1])], 1)
    r = np.concatenate([o, np.tile([0.0, 0.0, -1.0], (X.size, 1))], 1).astype(np.float32)
    t = rs.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy().reshape(X.shape)
    h = FLOOR_BAND[1] - t
    h[~np.isfinite(h)] = np.nan
    return h


def extend(lab, ijk_min, res, h):
    lab = np.asarray(lab).copy()
    nx, ny, nz = lab.shape
    zc = (np.arange(nz) + ijk_min[2] + 0.5) * res
    cand = np.zeros(lab.shape, bool)
    ok = np.isfinite(h)
    for i, j in zip(*np.nonzero(ok)):
        m = (zc > h[i, j]) & (zc < h[i, j] + H_STAND)
        cand[i, j, m] = lab[i, j, m] == MG.UNKNOWN
    region = (lab == MG.FREE) | cand
    cc, _ = ndimage.label(region, structure=ndimage.generate_binary_structure(3, 1))
    keep = np.unique(cc[lab == MG.FREE])
    keep = keep[keep > 0]
    add = cand & np.isin(cc, keep)
    lab[add] = MG.FREE
    return lab, int(add.sum()), int(cand.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=0)
    z = np.load(cfg.gt_voxel(), allow_pickle=False)
    lab, ijk, res = z["labels"], z["ijk_min"], float(np.asarray(z["res"]).ravel()[0])
    mesh = DS.world_mesh(cfg)
    xs = (np.arange(lab.shape[0]) + ijk[0] + 0.5) * res
    ys = (np.arange(lab.shape[1]) + ijk[1] + 0.5) * res
    h = floor_height(mesh, xs, ys)
    lab2, n_add, n_cand = extend(lab, ijk, res, h)
    out = {k: z[k] for k in z.files}
    out["labels"] = lab2
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    info["floor_supported"] = dict(rule="DESIGN_traversability.md §8", n_added=n_add,
                                   n_candidates=n_cand, floor_band=FLOOR_BAND,
                                   h_stand=H_STAND, n_floor_columns=int(np.isfinite(h).sum()))
    out["info"] = json.dumps(info)
    p = os.path.join(cfg.outdir(), "gt_voxel_fs.npz")
    np.savez_compressed(p, **out)
    print("floor columns %d | candidate unknown cells %d | added (connected) %d -> %s"
          % (np.isfinite(h).sum(), n_cand, n_add, p))


if __name__ == "__main__":
    main()

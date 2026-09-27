#!/usr/bin/env python3
"""Render the textured Polycam scan without WebGL: a top-down cutaway and eye-level views.

    python scripts/scan/render_scan_views.py \
        --mesh-config results/synth_corridor915/mesh_config.json --out paper/figures

This host cannot run WebGL headless, so web/scan_3d_*.html cannot be screenshotted
here. This renders the same mesh, the same textures and the same T_mesh_world on
the CPU (open3d ray casting), drawn unlit like Polycam's own viewer
(scripts/scan/mesh_payload.py's material note), with a mild |n.v| shade for
depth cues. It made paper/figures/fig_scan_views.png and the top-down layer of
fig_scan_vs_band_model.png.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
import gtsam  # noqa: F401  -- must precede open3d
import numpy as np
import open3d as o3d
import trimesh


class Scan:
    def __init__(self, mesh_config, obj=None):
        mc = json.load(open(mesh_config))
        T = np.array(mc["T_mesh_world"], float)
        scn = trimesh.load(obj or mc["obj"], process=False)
        self.tex = []
        self.rs = o3d.t.geometry.RaycastingScene()
        for name in sorted(scn.geometry):
            g = scn.geometry[name].copy()
            g.apply_transform(T)
            uv = np.asarray(g.visual.uv, float)[g.faces]
            img = np.asarray(g.visual.material.image.convert("RGB"), np.uint8)
            self.tex.append((uv, img))
            tm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(g.vertices, float)),
                                           o3d.utility.Vector3iVector(np.asarray(g.faces, np.int32)))
            self.rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tm))

    def cast(self, o, d):
        rays = np.concatenate([o, d], 1).astype(np.float32)
        a = self.rs.cast_rays(o3d.core.Tensor(rays))
        return {k: a[k].numpy() for k in ("t_hit", "geometry_ids", "primitive_ids",
                                          "primitive_uvs", "primitive_normals")}

    def color(self, a, d, shade=0.3, bg=(18, 22, 26)):
        gid, pid, b = a["geometry_ids"], a["primitive_ids"], a["primitive_uvs"]
        out = np.empty((len(gid), 3), np.float32)
        out[:] = bg
        hit = np.isfinite(a["t_hit"])
        for g, (uv, img) in enumerate(self.tex):
            m = hit & (gid == g)
            if not m.any():
                continue
            p, bb = pid[m], b[m]
            t = uv[p, 0] * (1 - bb[:, :1] - bb[:, 1:]) + uv[p, 1] * bb[:, :1] + uv[p, 2] * bb[:, 1:]
            H, W = img.shape[:2]
            c = np.clip((t[:, 0] * W).astype(int), 0, W - 1)
            r = np.clip(((1 - t[:, 1]) * H).astype(int), 0, H - 1)   # OBJ uv origin bottom-left
            out[m] = img[r, c]
        if shade:
            n = a["primitive_normals"]
            k = np.abs((n * d).sum(1)) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            out[hit] *= ((1 - shade) + shade * k[hit])[:, None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def topdown(self, x0, x1, y0, y1, z0, px=0.02):
        """Rays straight down from z0 (below the ceiling): a cutaway plan."""
        xs = np.arange(x0, x1, px) + px / 2
        ys = np.arange(y1, y0, -px) - px / 2
        X, Y = np.meshgrid(xs, ys)
        o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z0)], 1)
        d = np.tile([0, 0, -1.0], (X.size, 1))
        a = self.cast(o, d)
        return self.color(a, d, 0.25).reshape(X.shape + (3,)), (x0, x1, y0, y1)

    def persp(self, pos, yaw_deg, pitch_deg=0.0, hfov=95.0, W=800, H=500):
        yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
        fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        up = np.cross(right, fwd)
        f = (W / 2) / np.tan(np.radians(hfov) / 2)
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = (fwd[None] + ((u.ravel() - W / 2) / f)[:, None] * right[None]
             - ((v.ravel() - H / 2) / f)[:, None] * up[None])
        o = np.tile(np.asarray(pos, float), (len(d), 1))
        return self.color(self.cast(o, d), d).reshape(H, W, 3)


# the six views of paper/figures/fig_scan_views.png (world frame of corridor915)
VIEWS = [("A  lobby, looking along the corridor", (-2.0, -18.5, 1.5), 90, -5),
         ("B  stairwell doorway (2.26 m wide)", (-2.0, -14.3, 1.5), 0, -8),
         ("C  corridor at the floor seam (y -8..-5)", (-2.0, -11.0, 1.5), 90, -25),
         ("D  corridor past room 201", (-2.0, -2.0, 1.5), -90, -10),
         ("E  wall of room 201", (-2.0, -6.5, 1.5), 0, -5),
         ("F  north corridor, doors of 202-204", (-2.0, 3.0, 1.5), 90, -5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-config", required=True)
    ap.add_argument("--obj", default=None, help="override the OBJ (e.g. the floor-repaired one)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    S = Scan(args.mesh_config, args.obj)
    f, ax = plt.subplots(3, 2, figsize=(16, 15))
    for a, (t, p, yaw, pitch) in zip(ax.ravel(), VIEWS):
        a.imshow(S.persp(p, yaw, pitch))
        a.set_title(t, fontsize=11)
        a.axis("off")
    f.tight_layout()
    f.savefig(os.path.join(args.out, "scan_views.png"), dpi=70)
    img, ext = S.topdown(-5.8, 5.8, -21.0, 21.0, 1.75)
    np.savez_compressed(os.path.join(args.out, "scan_topdown.npz"), col=img, ext=np.array(ext))
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()

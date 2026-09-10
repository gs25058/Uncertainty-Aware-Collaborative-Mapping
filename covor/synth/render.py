"""Depth (and optionally stereo image) rendering from the room mesh (Part B-2/5).

Two modes, both ending in EXACTLY the tensor pair ``OccupancyBuilder`` already
consumes -- ``(P_cam (N,3), sigma_Z (N,))`` in the raw infra1 optical frame:

  ideal  the mesh's own depth, with sigma_Z from the SAME formula the stereo
         path uses (sigma_Z = Z^2 / (f B) * Delta_d, proposal §4.2). This
         isolates mapping error from stereo-matching error.
  sgbm   render a textured left/right pair with the real MILUV baseline and push
         it through the UNMODIFIED ``covor.occupancy.StereoDepth``, so the
         synthetic run carries the same disparity quantisation, matching holes
         and speckle the MILUV run does.

Both reuse ``StereoDepth`` for the intrinsics, the rectification and the
back-projection: nothing about the camera model is re-implemented here, so the
synthetic and the real path cannot drift apart. The renderer only supplies Z.

FRAME. Rays are built in the RECTIFIED left frame (the frame ``StereoDepth``
returns depth in) and rotated to world by R_wc @ R1^T, because
``StereoDepth.backproject`` maps rectified points back to the raw infra1 frame
with P_cam = P_rect @ R1. Depth is read straight off ``t_hit``: the ray
directions are built with z-component exactly 1, and open3d measures t in units
of the direction vector, so t_hit IS the z-coordinate of the hit in the
rectified frame -- the same quantity SGBM's f*B/disparity produces.
"""
import numpy as np

from covor.occupancy import load_stereo_calib, StereoDepth, DepthCfg


class DepthRenderer:
    def __init__(self, scene, robot, depth_cfg=None, textures=None):
        self.scene = scene
        self.calib = load_stereo_calib(robot)
        self.sd = StereoDepth(self.calib, depth_cfg or DepthCfg())
        self.textures = textures            # for the sgbm mode
        w, h = self.calib.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        sd = self.sd
        # unnormalised rays with d_z == 1 -> t_hit == depth Z
        self.dir_rect = np.stack([(u - sd.cx) / sd.fx, (v - sd.cy) / sd.fy,
                                  np.ones_like(u)], -1).astype(np.float32)
        self.R_cam_rect = sd.R1.T.astype(np.float32)   # P_cam = R1^T P_rect
        self.fB = float(sd.fx * self.calib.baseline)

    # -- raw depth ---------------------------------------------------------
    def _cast(self, T_wc, dx=0.0, extra=False):
        """Cast the pixel ray bundle from a world<-camera pose.

        dx: baseline offset along the rectified x axis (the right camera sits at
        +baseline in the rectified frame), so the stereo pair is rendered from
        the two real optical centres.
        """
        import open3d as o3d
        R = (T_wc[:3, :3] @ self.R_cam_rect).astype(np.float32)
        o = (T_wc[:3, 3] + T_wc[:3, :3] @ self.R_cam_rect @
             np.array([dx, 0.0, 0.0])).astype(np.float32)
        d = self.dir_rect.reshape(-1, 3) @ R.T
        rays = np.empty((len(d), 6), np.float32)
        rays[:, :3] = o
        rays[:, 3:] = d
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        h, w = self.dir_rect.shape[:2]
        t = ans["t_hit"].numpy().reshape(h, w)
        if not extra:
            return t
        return t, {k: ans[k].numpy() for k in
                   ("geometry_ids", "primitive_ids", "primitive_uvs")}

    def depth_ideal(self, T_wc):
        """(Z, sigma_Z, valid) in the rectified-left frame -- the same triple
        ``StereoDepth.depth`` returns, so the two are interchangeable."""
        c = self.sd.cfg
        Z = self._cast(T_wc)
        valid = np.isfinite(Z) & (Z >= c.z_min) & (Z <= c.z_max)
        Z = np.where(valid, Z, np.nan).astype(np.float32)
        # A ray that hits nothing (the scan's open side) is INVALID, not
        # max-range: the real stereo path also drops Z > z_max, so neither path
        # ever carves free space it did not measure.
        sigma_Z = (Z ** 2 / self.fB * c.disp_sigma_px).astype(np.float32)
        return Z, sigma_Z, valid

    # -- textured stereo pair ---------------------------------------------
    def images(self, T_wc):
        """(left, right) uint8 grayscale renders of the scan's own texture."""
        if self.textures is None:
            raise RuntimeError("sgbm mode needs textures=load_textures(...)")
        out = []
        for dx in (0.0, self.calib.baseline):
            t, e = self._cast(T_wc, dx=dx, extra=True)
            out.append(_shade(e, self.textures, t.shape))
        return out

    def depth_sgbm(self, T_wc):
        il, ir = self.images(T_wc)
        return self.sd.depth(il, ir)

    # -- the adapter the occupancy stage consumes -------------------------
    def frame(self, T_wc, mode="ideal", downsample=4):
        """-> (P_cam (N,3) raw infra1 frame, sigma_Z (N,)), or (None, None).

        Identical in type, frame and units to
        ``StereoDepth.backproject(*StereoDepth.depth(il, ir))`` in
        scripts/build_occupancy.py, which is what makes the synthetic and the
        MILUV pipelines the same pipeline.
        """
        Z, sZ, valid = (self.depth_ideal(T_wc) if mode == "ideal"
                        else self.depth_sgbm(T_wc))
        if valid.sum() < 100:
            return None, None
        return self.sd.backproject(Z, sZ, valid, downsample=downsample)


# ---------------------------------------------------------------------------
# texture handling for the sgbm mode
# ---------------------------------------------------------------------------
def load_textures(scene_trimesh, T_mesh_world):
    """Per-geometry (vertices, faces, uv, image) for a multi-material OBJ.

    Returns (list of trimesh submeshes in world coordinates, list of
    (uv (F,3,2), gray image (H,W) uint8)) in the order they must be added to the
    RaycastingScene, so open3d's ``geometry_ids`` indexes this list.
    """
    subs, tex = [], []
    for name in sorted(scene_trimesh.geometry):
        g = scene_trimesh.geometry[name].copy()
        g.apply_transform(T_mesh_world)
        uv = np.asarray(g.visual.uv, float)[g.faces]          # (F,3,2)
        img = np.asarray(g.visual.material.image.convert("L"), np.uint8)
        subs.append(g)
        tex.append((uv, img))
    return subs, tex


def _shade(e, tex, shape):
    """Nearest-texel lookup for every hit pixel; misses render black."""
    gid = e["geometry_ids"].reshape(-1)
    pid = e["primitive_ids"].reshape(-1)
    bary = e["primitive_uvs"].reshape(-1, 2)
    out = np.zeros(gid.shape, np.uint8)
    INVALID = np.iinfo(gid.dtype).max if np.issubdtype(gid.dtype, np.integer) else -1
    for g, (uv, img) in enumerate(tex):
        m = gid == g
        if not m.any():
            continue
        p = pid[m]
        ok = p < len(uv)
        if not ok.all():
            m[np.nonzero(m)[0][~ok]] = False
            p = p[ok]
        b = bary[m]
        t = (uv[p, 0] * (1 - b[:, :1] - b[:, 1:]) + uv[p, 1] * b[:, :1]
             + uv[p, 2] * b[:, 1:])
        H, W = img.shape
        # OBJ uv origin is bottom-left; image row 0 is the top
        c = np.clip((t[:, 0] * W).astype(int), 0, W - 1)
        r = np.clip(((1 - t[:, 1]) * H).astype(int), 0, H - 1)
        out[m] = img[r, c]
    out[gid == INVALID] = 0
    return out.reshape(shape)

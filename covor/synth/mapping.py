"""Drive the UNMODIFIED OccupancyBuilder from rendered depth (Part B-5).

``build_map`` is the synthetic twin of scripts/build_occupancy.py: same builder,
same OccCfg, same ``integrate_frame(T_wc, tr_sigma_pos, P_cam, sigma_Z,
teammates=...)`` call, same teammate masking against the airframe centre. Only
the depth source changes -- rendered instead of SGBM-on-MILUV-images.

It also records WHICH CELLS received evidence, by wrapping (not editing) the
OcTree with a proxy that notes every ``updateNode`` coordinate. That set is the
observability domain: a cell no beam ever touched cannot be scored for or
against any condition, and every condition inside one coverage mode is scored on
the same domain (built once, from GT poses).
"""
import numpy as np

from covor.occupancy import OccupancyBuilder, OccCfg
from . import mesh_gt as MG
from .config import ROBOTS
from .render import DepthRenderer


class _RecordingTree:
    """Proxy that records WHICH CELLS were written, as voxel indices.

    The index is taken at write time and kept in a set, not as a list of
    coordinates: updateNode is called tens of thousands of times per frame, so
    keeping one small array per call means tens of millions of objects for a
    three-camera pass. The set collapses that to the ~10^5 distinct cells that
    are the only thing the observability mask needs.
    """

    def __init__(self, tree, sink, res):
        object.__setattr__(self, "_t", tree)
        object.__setattr__(self, "_sink", sink)
        object.__setattr__(self, "_r", float(res))

    def updateNode(self, ctr, val, lazy):
        r = self._r
        self._sink.add((int(np.floor(ctr[0] / r)), int(np.floor(ctr[1] / r)),
                        int(np.floor(ctr[2] / r))))
        return self._t.updateNode(ctr, val, lazy)

    def __getattr__(self, k):
        return getattr(self._t, k)


def make_builder(occ_cfg, record=False):
    b = OccupancyBuilder(occ_cfg)
    if record:
        b._touched = set()
        b.tree = _RecordingTree(b.tree, b._touched, occ_cfg.resolution)
    return b


def touched_index(builder, res):
    t = getattr(builder, "_touched", None)
    if not t:
        return np.empty((0, 3), np.int64)
    return np.array(sorted(t), dtype=np.int64)


def build_map(cfg, scene, cams, poses, arms, mode="ideal", stride=2,
              record=None, teammate_pos=None, textures=None, render_T=None):
    """Accumulate one map PER ARM from one pass over the frames.

    cams      robots whose cameras contribute (the coverage mode).
    poses     {robot: dict(t (N,), T (N,4,4) world<-camera, tr (N,),
                           sig (N,3,3) or None)}
    arms      {tag: OccCfg} -- every arm sees the SAME rendered frames, so the
              arms differ only in the weighting, never in the observations.
    teammate_pos {robot: (t, p_body)} for dynamic masking; None disables.
    render_T  {robot: (N,4,4)} to RENDER from while still INTEGRATING at
              poses[robot]["T"]. Only a negative control uses this. It exists
              because rendering and integrating from the same pose is
              self-consistent BY CONSTRUCTION: the endpoints land on the mesh
              whatever the pose is, so corrupting the pose leaves precision at
              1.000 (measured: even a 120 deg body-vs-camera error did). The
              bug class that actually threatens the pipeline is a MISMATCH
              between the frame the depth is rendered in and the frame the rays
              are cast in -- which is what this reproduces.
    Returns ({tag: builder}, stats).
    """
    bs = {k: make_builder(v, record=(record == k or record is True))
          for k, v in arms.items()}
    stats = dict(frames={}, points=0)
    for rob in cams:
        P = poses[rob]
        rend = DepthRenderer(scene, rob, textures=textures)
        n = 0
        RT = (render_T or {}).get(rob)
        for i in range(0, len(P["t"]), stride):
            Pc, sZ = rend.frame(RT[i] if RT is not None else P["T"][i], mode=mode)
            if Pc is None:
                continue
            mates = _mates_at(teammate_pos, rob, P["t"][i]) if teammate_pos else None
            for k, b in bs.items():
                kw = {}
                if arms[k].pose_mode != "off" and P.get("sig") is not None:
                    kw["sigma_pos"] = P["sig"][i]
                b.integrate_frame(P["T"][i], float(P["tr"][i]), Pc, sZ,
                                  teammates=mates, **kw)
            stats["points"] += len(Pc)
            n += 1
        stats["frames"][rob] = n
    for b in bs.values():
        b.finalize()
    return bs, stats


def _mates_at(tp, rob, t, tol=0.5):
    out = []
    for o, (ts, ps) in tp.items():
        if o == rob:
            continue
        i = int(np.abs(ts - t).argmin())
        if abs(ts[i] - t) <= tol:
            out.append(ps[i])
    return np.array(out) if out else None


def gt_pose_source(cfg, gt, stride_nodes=1):
    """Pose source built from the EXACT GT trajectory: T_wc, tr_sigma_pos = 0.

    tr = 0 forces w_pose = 1 (there is no registration uncertainty in ground
    truth), exactly as scripts/gt_pose_control.py does on MILUV.
    """
    from .dataset import gt_camera_poses
    out, mates = {}, {}
    for r in ROBOTS:
        t = gt[r]["t"][::stride_nodes]
        T, p_body = gt_camera_poses(r, t, gt)
        out[r] = dict(t=t, T=T, tr=np.zeros(len(t)), sig=None)
        mates[r] = (t, p_body)
    return out, mates


def npz_pose_source(paths):
    """Pose source from fuse_and_dump-style .npz (fused poses + Sigma)."""
    out, mates = {}, {}
    for r, p in paths.items():
        z = np.load(p)
        out[r] = dict(t=z["t"], T=z["T"], tr=z["tr_sigma_pos"],
                      sig=z["sigma_pos"] if "sigma_pos" in z.files else None)
        mates[r] = (z["t"], z["p_body"])
    return out, mates

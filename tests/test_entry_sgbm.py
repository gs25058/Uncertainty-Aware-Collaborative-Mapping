#!/usr/bin/env python3
"""The sgbm wiring (PREREG_sgbm_depth.md §4), pinned by value.

  1. textured_scene: the multi-geometry scene is the same surface as the
     single-mesh scene every ideal map was built with, so its ideal depth must be
     equal pixel for pixel, and each geometry id must index its own texture.
  2. dump_fused_map.CachedFrames: a cache rendered from other poses must be
     refused, never integrated.

Needs the corridor915 mesh and dataset; skips cleanly without them.
Run: python tests/test_entry_sgbm.py   (or: pytest tests/test_entry_sgbm.py)
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "synth"))
import gtsam                       # noqa: F401,E402  -- must precede open3d
from covor.synth import dataset as DS, mesh_gt as MG    # noqa: E402
from covor.synth.config import SynthCfg                 # noqa: E402
from covor.synth.render import DepthRenderer            # noqa: E402

CFG = SynthCfg(name="corridor915", seq="synth_corridor915_0", seed=0)


def _have():
    import json
    try:
        return os.path.exists(json.load(open(CFG.mesh_config()))["obj"])
    except (OSError, KeyError, ValueError):
        return False


def _poses():
    """A handful of GT camera poses spread along the corridor."""
    DS.install_paths(CFG)
    gt = DS.load_gt_traj(CFG)
    out = []
    for r in ("ifo001", "ifo002", "ifo003"):
        t = gt[r]["t"]
        T, _ = DS.gt_camera_poses(r, t[:: max(len(t) // 3, 1)][:3], gt)
        out += [(r, T[k]) for k in range(len(T))]
    return out


def test_multi_geometry_ideal_is_bit_identical():
    if not _have():
        print("skip: corridor915 mesh not present")
        return
    from textured_scene import textured_scene, ideal_bitcheck
    DS.install_paths(CFG)
    single = MG.raycasting_scene(DS.world_mesh(CFG))
    multi, tex = textured_scene(CFG.mesh_config())
    assert len(tex) == 6
    n = 0
    for r, T in _poses():
        d = ideal_bitcheck(DepthRenderer(single, r), DepthRenderer(multi, r), T)
        assert d == 0, "%s: %d pixels differ" % (r, d)
        n += 1
    assert n >= 6


def test_geometry_ids_index_their_own_texture():
    """Every hit's geometry id is one of the six, and its primitive id is a
    face of THAT submesh -- the pairing render._shade relies on."""
    if not _have():
        print("skip: corridor915 mesh not present")
        return
    from textured_scene import textured_scene
    multi, tex = textured_scene(CFG.mesh_config())
    r, T = _poses()[0]
    _, e = DepthRenderer(multi, r)._cast(T, extra=True)
    gid, pid = e["geometry_ids"].ravel(), e["primitive_ids"].ravel()
    hit = gid != np.iinfo(gid.dtype).max
    assert hit.mean() > 0.5
    for g in np.unique(gid[hit]):
        assert 0 <= g < len(tex)
        assert pid[hit & (gid == g)].max() < len(tex[g][0])


def test_cached_frames_refuse_a_stale_pose():
    from dump_fused_map import CachedFrames
    T = np.stack([np.eye(4), np.eye(4)])
    T[1, 0, 3] = 1.0
    P = np.arange(12, dtype=float).reshape(4, 3)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.npz")
        np.savez(p, idx=np.array([0, 4]), T=T, offs=np.array([0, 4, 4]), P=P,
                 sZ=np.ones(4))
        C = CachedFrames(p)
        Pc, sZ = C.frame(0, T[0])
        assert np.array_equal(Pc, P) and np.array_equal(sZ, np.ones(4))
        assert C.frame(4, T[1]) == (None, None)          # empty frame
        Tbad = T[1].copy()
        Tbad[0, 3] += 1e-12
        try:
            C.frame(4, Tbad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a stale cached pose was accepted")


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

#!/usr/bin/env python3
"""The two duplications Phase 4 introduces, pinned by value.

Phase 4 may not edit covor/occupancy.py or covor/synth/, so two pieces of logic
that already exist there had to be rewritten in new files:

  1. scripts/entry/dump_fused_map.build_and_record re-implements the frame loop
     of covor.synth.mapping.build_map, because build_map constructs its builders
     internally and offers no hook for per-frame recording.
  2. covor.entry.sigma_map.occupied_endpoint_voxels re-implements the endpoint
     half of OccupancyBuilder.integrate_frame, because neither ``_touched`` nor
     ``_attr`` can say which frames wrote occupied evidence into a column.

A duplicated rule drifts silently when the original changes. These tests are the
only thing that makes the duplication safe, so they compare VALUES: per-cell
log-odds, and the exact set of endpoint cells.

Needs the seed-0 synthetic dataset; skips cleanly without it.
Run: python tests/test_entry_dump.py   (or: pytest tests/test_entry_dump.py)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "synth"))
import gtsam                       # noqa: F401,E402  -- must precede open3d
from covor.entry import sigma_map as SM                                # noqa: E402
from covor.occupancy import OccCfg, OccupancyBuilder                   # noqa: E402
from covor.synth import dataset as DS, mapping as MP, mesh_gt as MG    # noqa: E402
from covor.synth import metrics as ME                                  # noqa: E402
from covor.synth.config import SynthCfg, ROBOTS                        # noqa: E402
from covor.synth.render import DepthRenderer                           # noqa: E402
from dump_fused_map import build_and_record                            # noqa: E402

STRIDE = 64          # coarse on purpose: this is an identity check, not coverage


def _ready(cfg):
    return os.path.exists(cfg.gt_voxel()) and os.path.exists(
        os.path.join(cfg.vinsroot(), cfg.seq, "ifo001", "vio.csv"))


def _setup():
    cfg = SynthCfg(seed=0)
    if not _ready(cfg):
        return None
    DS.install_paths(cfg)
    lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
    gt = DS.load_gt_traj(cfg)
    scene = MG.raycasting_scene(DS.world_mesh(cfg))
    poses, mates = MP.gt_pose_source(cfg, gt, stride_nodes=2)
    for r in ROBOTS:                       # a Sigma the sigma path can chew on
        poses[r]["sig"] = np.tile(np.diag([0.04, 0.09, 0.01]),
                                  (len(poses[r]["t"]), 1, 1))
    return cfg, lab, ijk_min, res, scene, poses, mates


def test_recording_loop_matches_build_map_logodds_cell_by_cell():
    """The dump driver's frame loop IS build_map's, to the last log-odds."""
    s = _setup()
    if s is None:
        return print("SKIP: seed-0 dataset not built")
    cfg, lab, ijk_min, res, scene, poses, mates = s
    occ = OccCfg(resolution=res, weighted=True, use_w_pose=False)
    cams = list(ROBOTS)

    bs, st_ref = MP.build_map(cfg, scene, cams, poses, {"a": occ}, mode="ideal",
                              stride=STRIDE, teammate_pos=mates)
    want = ME.logodds_grid(bs["a"], res, lab.shape, ijk_min)

    b, ev, st = build_and_record(cfg, scene, cams, poses, occ, STRIDE,
                                 teammate_pos=mates)
    got = ME.logodds_grid(b, res, lab.shape, ijk_min)

    assert st["frames"] == st_ref["frames"], (st["frames"], st_ref["frames"])
    assert st["points"] == st_ref["points"], (st["points"], st_ref["points"])
    same = np.isfinite(got) == np.isfinite(want)
    assert same.all(), ("%d cells changed between written and unwritten"
                        % int((~same).sum()))
    m = np.isfinite(want)
    d = np.abs(got[m] - want[m]).max() if m.any() else 0.0
    assert d == 0.0, ("%d written cells, max log-odds difference %.3e -- the "
                      "recording loop is not build_map's loop" % (int(m.sum()), d))
    assert (ev.dense(lab.shape[:2], ijk_min)[0] > 0).any(), "nothing was recorded"


def test_occupied_endpoint_replica_matches_the_builder_exactly():
    """With l_free = 0 the builder's positive writes ARE the occupied endpoints.

    Free evidence then contributes w * 0 = 0 to every cell it crosses, so a cell
    whose accumulated update is > 0 got occupied evidence and nothing else can
    have made it positive. That gives ground truth for the replica without
    touching covor/occupancy.py.
    """
    s = _setup()
    if s is None:
        return print("SKIP: seed-0 dataset not built")
    cfg, lab, ijk_min, res, scene, poses, mates = s
    occ = OccCfg(resolution=res, l_free=0.0, weighted=True, use_w_pose=False)
    b = OccupancyBuilder(occ)
    seen = []

    class _Rec:
        def __init__(self, t, r):
            object.__setattr__(self, "_t", t)
            object.__setattr__(self, "_r", r)

        def updateNode(self, ctr, val, lazy):
            r = self._r
            if val > 0:
                seen.append((int(np.floor(ctr[0] / r)), int(np.floor(ctr[1] / r)),
                             int(np.floor(ctr[2] / r))))
            return self._t.updateNode(ctr, val, lazy)

        def __getattr__(self, k):
            return getattr(self._t, k)

    b.tree = _Rec(b.tree, res)
    n = 0
    for rob in ROBOTS:
        rend = DepthRenderer(scene, rob)
        P = poses[rob]
        for i in range(0, len(P["t"]), 200):
            Pc, sZ = rend.frame(P["T"][i], mode="ideal")
            if Pc is None:
                continue
            mts = MP._mates_at(mates, rob, P["t"][i])
            del seen[:]
            b.integrate_frame(P["T"][i], 0.0, Pc, sZ, teammates=mts)
            from_builder = set(seen)
            replica = set(map(tuple, SM.occupied_endpoint_voxels(
                P["T"][i], Pc, mts, occ).tolist()))
            assert from_builder == replica, (
                "%s frame %d: builder wrote %d occupied cells, replica says %d "
                "(%d only in builder, %d only in replica)"
                % (rob, i, len(from_builder), len(replica),
                   len(from_builder - replica), len(replica - from_builder)))
            assert len(replica) > 0
            n += 1
    assert n >= 9, "only %d frames compared" % n


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

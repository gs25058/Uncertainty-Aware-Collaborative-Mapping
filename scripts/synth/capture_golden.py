#!/usr/bin/env python3
"""Freeze the current fusion+mapping output, to prove a later change is a no-op.

    python scripts/synth/capture_golden.py --name room909 --seed 0

Run this BEFORE editing covor/fusion.py. It stores, for each condition:
  * every node's 4x4 fused pose, and
  * the per-cell log-odds of a small map built from those poses,
so tests/test_synth_gauge.py can assert the default path is unchanged BY VALUE
rather than by factor counts (RESULTS_SUMMARY §9-2 / A-4: a count-only check
once let a free-evidence double-counting bug through).

The map uses a coarse frame stride on purpose -- the point is a value-identical
comparison, not coverage.
"""
import argparse
import os
import sys

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
from covor.occupancy import OccCfg
from covor.synth import dataset as DS, mapping as MP, mesh_gt as MG, metrics as ME
from covor.synth.config import SynthCfg, ROBOTS
from fuse_synth import fuse, CONDITIONS

MAP_STRIDE = 32          # ~28 frames: enough cells to be a real comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=args.seed)
    DS.install_paths(cfg)
    lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
    gt = DS.load_gt_traj(cfg)
    scene = MG.raycasting_scene(DS.world_mesh(cfg))

    out = {}
    for cond, pairs in CONDITIONS.items():
        poses, st = fuse(cfg.seq, pairs)
        for r in ROBOTS:
            out["%s_%s_T" % (cond, r)] = poses[r]["T"]
            out["%s_%s_tr" % (cond, r)] = poses[r]["tr"]
        out["%s_ninter" % cond] = np.array([st["n_inter_range"]])
        out["%s_ngauge" % cond] = np.array([st["n_gauge_prior"]])
        mates = {r: (poses[r]["t"], poses[r]["p_body"]) for r in ROBOTS}
        bs, _ = MP.build_map(cfg, scene, ["ifo001"], poses,
                             {"full": OccCfg(resolution=res, weighted=True)},
                             mode="ideal", stride=MAP_STRIDE, teammate_pos=mates)
        L = ME.logodds_grid(bs["full"], res, lab.shape, ijk_min)
        out["%s_logodds" % cond] = L
        print("  %s: %d inter, %d gauge, %d written cells"
              % (cond, st["n_inter_range"], st["n_gauge_prior"],
                 int(np.isfinite(L).sum())), flush=True)

    p = args.out or os.path.join("/src/gs25058/cr_RNE/covor_slam", "tests", "data",
                                 "synth_gauge_golden.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, map_stride=np.array([MAP_STRIDE]),
                        seed=np.array([args.seed]), **out)
    print("wrote %s (%.1f MB)" % (p, os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render one robot's sgbm depth for a whole run and cache it, with the P0 record.

    python scripts/entry/render_depth_cache.py --name corridor915 --robot ifo001

PREREG_sgbm_depth.md §4-5. Splitting the render from the integration is only a
timeout measure (one foreground call per job): the frames, poses and stride are
dump_fused_map.py's, and dump_fused_map.py --depth-cache checks every cached
pose against its own fused pose bit for bit before integrating a frame.

Per frame (the frames dump_fused_map integrates: stride over the FUSED poses,
render pose == integration pose, as in covor.synth.mapping.build_map):

  1. BIT CHECK: ideal depth from the multi-geometry textured scene vs the
     single-mesh scene every ideal map was built with. Any differing pixel is
     recorded; the prereg says anything but 0 invalidates the run.
  2. sgbm depth through the UNMODIFIED DepthRenderer.depth_sgbm, backprojected
     exactly as DepthRenderer.frame does (asserted equal on the first frame).
  3. P0: |Z_sgbm - Z_ideal| over pixels valid in BOTH, pooled into a 1 mm
     histogram, so the pooled median over the full run is exact to 1 mm.
     Also the signed sum, valid-pixel counts, and how many of the pixels SGBM
     loses lie in the left num_disp-column search margin.
  4. Per cached point, ``miss``: the left camera's ray through that pixel hits
     no mesh at all (a scan hole). PREREG_sgbm_rescue.md treatment A drops
     these; the backprojected points themselves are unchanged.
"""
import argparse
import json
import os
import sys
import time

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.synth import dataset as DS, mesh_gt as MG
from covor.synth.config import SynthCfg
from covor.synth.render import DepthRenderer
from fuse_synth import fuse, CONDITIONS
from textured_scene import textured_scene

HIST_BIN = 0.001      # m
HIST_MAX = 10.0       # m; z_max is 5, so |dZ| < 5 always


def cache_path(cfg, cond, robot, stride):
    return os.path.join(cfg.outdir(), "entry", "sgbm_cache",
                        "depth_%s_%s_s%d.npz" % (cond, robot, stride))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="corridor915")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="C_3drone", choices=list(CONDITIONS))
    ap.add_argument("--robot", required=True)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=args.seed)
    DS.install_paths(cfg)
    single = MG.raycasting_scene(DS.world_mesh(cfg))
    multi, tex = textured_scene(cfg.mesh_config())
    poses, _ = fuse(cfg.seq, CONDITIONS[args.cond])
    P = poses[args.robot]

    r_single = DepthRenderer(single, args.robot)
    r_multi = DepthRenderer(multi, args.robot, textures=tex)
    margin = r_multi.sd.cfg.min_disp + r_multi.sd.cfg.num_disp

    nb = int(HIST_MAX / HIST_BIN)
    hist = np.zeros(nb, np.int64)
    idx, Ts, pts, szs, misses, offs = [], [], [], [], [], [0]
    rec = dict(bitdiff_px=0, bitdiff_frames=0, n_both=0, sum_signed=0.0,
               n_valid_ideal=0, n_valid_sgbm=0, n_lost=0, n_lost_margin=0,
               n_frames=0, n_none=0, n_px=0)
    t0 = time.time()
    for i in range(0, len(P["t"]), args.stride):
        T = P["T"][i]
        Zi, _, vi = r_single.depth_ideal(T)
        Zm, _, vm = r_multi.depth_ideal(T)
        d = int((~((vi == vm) & ((Zi == Zm) | (np.isnan(Zi) & np.isnan(Zm))))).sum())
        rec["bitdiff_px"] += d
        rec["bitdiff_frames"] += int(d > 0)

        Zs, sZs, vs = r_multi.depth_sgbm(T)
        both = vi & vs
        dz = np.abs(Zs[both] - Zi[both]).astype(np.float64)
        hist += np.bincount(np.minimum((dz / HIST_BIN).astype(np.int64), nb - 1),
                            minlength=nb)
        rec["n_both"] += int(both.sum())
        rec["sum_signed"] += float((Zs[both] - Zi[both]).astype(np.float64).sum())
        rec["n_valid_ideal"] += int(vi.sum())
        rec["n_valid_sgbm"] += int(vs.sum())
        lost = vi & ~vs
        rec["n_lost"] += int(lost.sum())
        rec["n_lost_margin"] += int(lost[:, :margin].sum())
        rec["n_px"] += vi.size

        # DepthRenderer.frame, written out so the sgbm render is done once
        if vs.sum() < 100:
            Pc, sZ = None, None
            rec["n_none"] += 1
        else:
            Pc, sZ = r_multi.sd.backproject(Zs, sZs, vs, downsample=4)
            # backproject keeps the [::4, ::4] valid pixels in row-major order
            miss = ~np.isfinite(r_single._cast(T))[::4, ::4][vs[::4, ::4]]
            assert len(miss) == len(Pc)
        if not idx:
            Pf, sf = r_multi.frame(T, mode="sgbm")
            assert (Pf is None and Pc is None) or (
                np.array_equal(Pf, Pc) and np.array_equal(sf, sZ)), \
                "inlined frame() differs from DepthRenderer.frame"
        idx.append(i)
        Ts.append(T)
        if Pc is not None:
            pts.append(Pc.astype(np.float64))
            szs.append(sZ.astype(np.float64))
            misses.append(miss)
            offs.append(offs[-1] + len(Pc))
        else:
            offs.append(offs[-1])
        rec["n_frames"] += 1
    rec["seconds"] = round(time.time() - t0, 1)

    out = cache_path(cfg, args.cond, args.robot, args.stride)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, idx=np.array(idx), T=np.array(Ts), offs=np.array(offs),
             P=np.concatenate(pts) if pts else np.zeros((0, 3)),
             sZ=np.concatenate(szs) if szs else np.zeros(0),
             miss=np.concatenate(misses) if misses else np.zeros(0, bool),
             hist=hist, hist_bin=np.array([HIST_BIN]), margin=np.array([margin]),
             rec=json.dumps(rec))
    c = np.cumsum(hist)
    med = (np.searchsorted(c, (c[-1] + 1) // 2) + 0.5) * HIST_BIN if c[-1] else np.nan
    print("%s: %d frames (%d empty), %.0fs | bit-diff px %d in %d frames | "
          "median |dZ| %.3f m, mean signed %+.3f m | valid ideal %.3f sgbm %.3f | "
          "lost %d, %.1f%% in the left %d-col margin | -> %s"
          % (args.robot, rec["n_frames"], rec["n_none"], rec["seconds"],
             rec["bitdiff_px"], rec["bitdiff_frames"], med,
             rec["sum_signed"] / max(rec["n_both"], 1),
             rec["n_valid_ideal"] / rec["n_px"], rec["n_valid_sgbm"] / rec["n_px"],
             rec["n_lost"], 100 * rec["n_lost_margin"] / max(rec["n_lost"], 1),
             margin, out))


if __name__ == "__main__":
    main()

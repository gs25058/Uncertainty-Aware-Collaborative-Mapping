#!/usr/bin/env python3
"""Rebuild one synthetic condition's fused 3D map and dump it for Phase 4.

    python scripts/entry/dump_fused_map.py --cond C_3drone --coverage ii_all_cams \
        --arm depth_only --stride 4

run_experiment.py builds these maps and writes only SCALARS; the grids
themselves are never saved, so the entry stage has nothing to read. This script
re-runs the same fusion and the same OccupancyBuilder and saves the grid:
3D labels, per-cell log-odds, per-column observation counts, and per-column
sigma_xy.

NOTHING EXISTING IS EDITED. covor/fusion.py, covor/occupancy.py and covor/synth/
are imported and called. The frame loop is written out here rather than calling
covor.synth.mapping.build_map because the per-frame recording needs a hook
build_map does not offer (it constructs its builders internally). That is a
duplication risk, so tests/test_entry_dump.py asserts that this loop and
build_map produce IDENTICAL per-cell log-odds -- by value, not by cell count.

NEGATIVE CONTROL. --corrupt displaces the RENDER pose from the INTEGRATION pose.
Corrupting both is a no-op: rendering and integrating from the same pose is
self-consistent by construction (gate.py's header records that a 120 deg error
left precision at 1.000). "jitter<sigma>" is defined here (per-frame independent
displacement, PREREG_entry_control.md); every other kind is gate.py's own.

DEPTH MODE (PREREG_sgbm_depth.md). --mode sgbm renders a textured stereo pair
through the unmodified DepthRenderer.depth_sgbm, from the multi-geometry scene
in textured_scene.py. --depth-cache reads the frames render_depth_cache.py
rendered instead (a timeout split only); every cached pose must equal this
run's fused pose bit for bit, or the run stops. The ideal path is unchanged.

POINT FILTERS (PREREG_sgbm_rescue.md), both off by default:
  --drop-mesh-miss   treatment A: drop cached points whose ray hits no mesh
                     (a scan hole). Needs --depth-cache.
  --max-sigma-z S    treatment B: drop points with sigma_Z > S (any mode).
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
from covor.entry import clean3d, sigma_map as SM
from covor.occupancy import OccCfg, OccupancyBuilder
from covor.synth import dataset as DS, mapping as MP, mesh_gt as MG, metrics as ME
from covor.synth.config import SynthCfg, ROBOTS
from covor.synth.render import DepthRenderer
from fuse_synth import fuse, CONDITIONS
from run_experiment import ARMS, COVERAGE
import gate as GATE


def corrupt_poses(poses, how, gt, seed=0):
    """Break the pose chain on purpose. PREREG_entry_control.md §2.

    "jitter<sigma>" is the kind this file adds: an INDEPENDENT isotropic
    Gaussian displacement per frame, so two frames that see the same wall put it
    in different cells. That is the corruption an entry map should be able to
    feel, and the one a rigid shift is not -- a rigid transform moves a map
    without making it inconsistent with itself, and clearance is invariant to
    that (RESULTS_entry.md §4).

    Every other kind is scripts/synth/gate.py's ``corrupt``, called unmodified.
    The draw is seeded from (seed, robot index, sigma), so a rerun reproduces the
    same map. sigma = 0 returns the input poses bit for bit, which is the
    control's own control (PREREG §3 P0).
    """
    if not how.startswith("jitter"):
        return GATE.corrupt(poses, how, gt)
    sigma = float(how[6:] or 0.0)
    out = {}
    for k, (r, P) in enumerate(sorted(poses.items())):
        T = np.array(P["T"], copy=True)
        if sigma > 0:
            # an explicit integer seed sequence, not hash(): hash() of a tuple
            # is stable for ints but that is an implementation detail, and the
            # pre-registration promises a rerun reproduces the same map.
            rng = np.random.default_rng([int(seed), k, int(round(sigma * 1e6))])
            T[:, :3, 3] += rng.normal(0.0, sigma, size=(len(T), 3))
        out[r] = dict(P, T=T)
    return out


class CachedFrames:
    """Frames from render_depth_cache.py, keyed like DepthRenderer.frame."""

    def __init__(self, path, drop_mesh_miss=False):
        z = np.load(path)
        self.idx = {int(i): k for k, i in enumerate(z["idx"])}
        self.T, self.offs, self.P, self.sZ = z["T"], z["offs"], z["P"], z["sZ"]
        self.keep = None
        if drop_mesh_miss:
            if "miss" not in z.files:
                raise RuntimeError("%s has no per-point miss flag; re-render it"
                                   % path)
            self.keep = ~z["miss"]

    def frame(self, i, T_wc):
        k = self.idx[i]
        if not np.array_equal(self.T[k], T_wc):
            raise RuntimeError("cached pose of frame %d differs from this run's "
                               "fused pose: the cache is stale" % i)
        a, b = self.offs[k], self.offs[k + 1]
        if a == b:
            return None, None
        if self.keep is None:
            return self.P[a:b], self.sZ[a:b]
        m = self.keep[a:b]
        if not m.any():
            return None, None
        return self.P[a:b][m], self.sZ[a:b][m]


def filter_points(Pc, sZ, max_sigma_z=None):
    """Treatment B: drop points whose depth sigma exceeds ``max_sigma_z``.
    Only a frame left with NO points is dropped: a filter must remove points,
    not add a second frame-level threshold of its own."""
    if Pc is None or max_sigma_z is None:
        return Pc, sZ
    m = sZ <= max_sigma_z
    if not m.any():
        return None, None
    return Pc[m], sZ[m]


def build_and_record(cfg, scene, cams, poses, occ_cfg, stride, mode="ideal",
                     teammate_pos=None, render_T=None, textures=None,
                     cached=None, max_sigma_z=None):
    """One pass over the frames: build the map AND record per-column evidence.

    The integrate_frame call, the teammate masking and the frame stride are
    exactly covor.synth.mapping.build_map's; only the recording is added.
    """
    b = OccupancyBuilder(occ_cfg)
    rec = SM.ColumnRecorder(b.tree, occ_cfg.resolution)
    b.tree = rec
    ev = SM.ColumnEvidence()
    stats = dict(frames={}, points=0)
    for rob in cams:
        P = poses[rob]
        rend = DepthRenderer(scene, rob, textures=textures)
        RT = (render_T or {}).get(rob)
        C = (cached or {}).get(rob)
        n = 0
        for i in range(0, len(P["t"]), stride):
            if C is not None:
                Pc, sZ = C.frame(i, RT[i] if RT is not None else P["T"][i])
            else:
                Pc, sZ = rend.frame(RT[i] if RT is not None else P["T"][i],
                                    mode=mode)
            Pc, sZ = filter_points(Pc, sZ, max_sigma_z)
            if Pc is None:
                continue
            mates = (MP._mates_at(teammate_pos, rob, P["t"][i])
                     if teammate_pos else None)
            rec.cols = set()
            b.integrate_frame(P["T"][i], float(P["tr"][i]), Pc, sZ, teammates=mates)
            # a pose source without Sigma (GT poses) records n_obs but no margin
            sig = 0.0 if P.get("sig") is None else SM.sigma_xy_from_cov(P["sig"][i])
            ev.add_frame(rec.cols,
                         SM.occupied_endpoint_voxels(P["T"][i], Pc, mates, occ_cfg),
                         sig)
            stats["points"] += len(Pc)
            n += 1
        stats["frames"][rob] = n
    b.finalize()
    return b, ev, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="C_3drone", choices=list(CONDITIONS))
    ap.add_argument("--coverage", default="ii_all_cams", choices=list(COVERAGE))
    ap.add_argument("--arm", default="depth_only", choices=list(ARMS))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--corrupt", default="none",
                    help="negative control: none | jitter<m> (per-frame, "
                         "PREREG_entry_control.md) | shift<m> | yaw<deg> | nobodycam")
    ap.add_argument("--gt", default=None,
                    help="gt_voxel.npz to take the grid (and hence the "
                         "RESOLUTION) from. Default: the sequence's own 0.10 m "
                         "grid. DESIGN §7 sanctions a 0.05 m re-integration as "
                         "the 'final deliverable version'; PREREG_RESOLUTION.md "
                         "forbids comparing absolute metrics ACROSS resolutions, "
                         "so a 0.05 dump is for figures and for differences "
                         "measured inside 0.05.")
    ap.add_argument("--mode", default="ideal", choices=("ideal", "sgbm"),
                    help="depth source (PREREG_sgbm_depth.md)")
    ap.add_argument("--depth-cache", action="store_true",
                    help="sgbm only: read render_depth_cache.py's frames")
    ap.add_argument("--render-truth", action="store_true",
                    help="render each frame at the TRUE camera pose and integrate it at "
                         "the fused pose -- what a real camera does (RESULTS_"
                         "traversability.md §7). Default: render at the fused pose.")
    ap.add_argument("--drop-mesh-miss", action="store_true",
                    help="PREREG_sgbm_rescue.md treatment A (needs --depth-cache)")
    ap.add_argument("--max-sigma-z", type=float, default=None,
                    help="PREREG_sgbm_rescue.md treatment B: drop sigma_Z > this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.drop_mesh_miss and not args.depth_cache:
        ap.error("--drop-mesh-miss needs --depth-cache")
    if args.depth_cache and args.mode != "sgbm":
        ap.error("--depth-cache is for --mode sgbm")

    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=args.seed)
    DS.install_paths(cfg)
    lab_gt, ijk_min, res, _, _ = MG.load_gt(args.gt or cfg.gt_voxel())
    gt = DS.load_gt_traj(cfg)
    textures = cached = None
    if args.mode == "sgbm" and args.depth_cache:
        from render_depth_cache import cache_path
        scene = None
        cached = {r: CachedFrames(cache_path(cfg, args.cond, r, args.stride,
                                             truth=args.render_truth),
                                  drop_mesh_miss=args.drop_mesh_miss)
                  for r in COVERAGE[args.coverage]}
    elif args.mode == "sgbm":
        from textured_scene import textured_scene
        scene, textures = textured_scene(cfg.mesh_config())
    else:
        scene = MG.raycasting_scene(DS.world_mesh(cfg))

    t0 = time.time()
    poses, gstats = fuse(cfg.seq, CONDITIONS[args.cond])
    ates = {}
    for r in ROBOTS:
        T_gt, _ = DS.gt_camera_poses(r, poses[r]["t"], gt)
        ates[r] = ME.ate(poses[r]["T"][:, :3, 3], T_gt[:, :3, 3])["ate_rmse"]
    trs = np.concatenate([np.sqrt(poses[r]["tr"]) for r in ROBOTS])
    sxy = np.concatenate([[SM.sigma_xy_from_cov(S) for S in poses[r]["sig"]]
                          for r in ROBOTS])
    print("fusion %s: %d inter ranges, %.0fs | ATE %s | sqrt(trSigma) med %.3f | "
          "sigma_xy med %.3f p90 %.3f"
          % (args.cond, gstats["n_inter_range"], time.time() - t0,
             {k: round(v, 3) for k, v in ates.items()}, np.median(trs),
             np.median(sxy), np.percentile(sxy, 90)))

    render_T = None
    if args.render_truth:
        if args.corrupt != "none":
            ap.error("--render-truth and --corrupt both set the render pose")
        render_T = {r: DS.gt_camera_poses(r, poses[r]["t"], gt)[0] for r in ROBOTS}
        print("RENDER AT TRUE POSE, integrate at fused pose")
    if args.corrupt != "none":
        render_T = {r: v["T"] for r, v in
                    corrupt_poses(poses, args.corrupt, gt, args.seed).items()}
        print("NEGATIVE CONTROL: render pose corrupted with %r" % args.corrupt)

    mates = {r: (poses[r]["t"], poses[r]["p_body"]) for r in ROBOTS}
    occ_cfg = OccCfg(resolution=res, **ARMS[args.arm])
    t1 = time.time()
    b, ev, st = build_and_record(cfg, scene, COVERAGE[args.coverage], poses, occ_cfg,
                                 args.stride, mode=args.mode, teammate_pos=mates,
                                 render_T=render_T, textures=textures, cached=cached,
                                 max_sigma_z=args.max_sigma_z)
    print("map: %s frames, %d points, %.0fs" % (st["frames"], st["points"],
                                                time.time() - t1))

    M_occ, M_free = ME.map_masks(b, res, lab_gt, ijk_min)
    lab = clean3d.labels_from_masks(M_occ, M_free)
    n_obs, sigma_xy = ev.dense(lab.shape[:2], ijk_min)
    tag = args.corrupt if args.corrupt != "none" else "clean"
    if args.mode != "ideal":
        tag += "_" + args.mode
    if args.render_truth:
        tag += "_rt"
    if args.drop_mesh_miss:
        tag += "_A"
    if args.max_sigma_z is not None:
        tag += "_B%03d" % round(args.max_sigma_z * 100)
    out = args.out or os.path.join(
        cfg.outdir(), "entry", "map3d_%s_%s_%s_%s_s%d%s.npz"
        % (args.cond, args.coverage, args.arm, tag, args.stride,
           "" if abs(res - 0.10) < 1e-9 else "_r%03d" % round(res * 1000)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, labels=lab, M_occ=M_occ, M_free=M_free, ijk_min=ijk_min,
        res=np.array([res]), n_obs=n_obs, sigma_xy=sigma_xy,
        logodds=ME.logodds_grid(b, res, lab_gt.shape, ijk_min),
        meta=json.dumps(dict(
            cond=args.cond, coverage=args.coverage, arm=args.arm,
            stride=args.stride, corrupt=args.corrupt, seed=args.seed,
            mode=args.mode, depth_cache=bool(args.depth_cache),
            render_truth=bool(args.render_truth),
            drop_mesh_miss=bool(args.drop_mesh_miss), max_sigma_z=args.max_sigma_z,
            res=res, occ_cfg={k: v for k, v in ARMS[args.arm].items()},
            n_inter_range=int(gstats["n_inter_range"]), ate=ates,
            sqrt_tr_sigma_median=float(np.median(trs)),
            sigma_xy_median=float(np.median(sxy)),
            sigma_xy_p90=float(np.percentile(sxy, 90)),
            n_frames=int(sum(st["frames"].values())), n_points=int(st["points"]))))
    print("occupied %d, free %d, unknown %d | columns observed %d, with occupied "
          "evidence %d | -> %s"
          % (int(M_occ.sum()), int(M_free.sum()),
             int(lab.size - M_occ.sum() - M_free.sum()), int((n_obs > 0).sum()),
             int((sigma_xy > 0).sum()), out))


if __name__ == "__main__":
    main()

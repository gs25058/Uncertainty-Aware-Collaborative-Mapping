#!/usr/bin/env python3
"""Fixed-pose MILUV IR stereo smoke experiment; isolated output directory."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np

import build_occupancy as original
from covor.occupancy import load_stereo_calib, StereoDepth, DepthCfg
from covor.synth.render import _depth_rectified
from covor.smoke import stereo_smoke, clean_right_depth
from common import (conditions, write_json, write_csv, make_builder, sparse_map,
                    dense_map, map_metrics, depth_metrics, frame_points,
                    save_example, map_figure, digest, MAPPING_ARMS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=32, help="per robot")
    ap.add_argument("--stride", type=int, default=6, help="fused-pose index stride")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated smoke-field seeds")
    ap.add_argument("--drones", default="ifo001,ifo002,ifo003")
    ap.add_argument("--levels", default="all")
    ap.add_argument("--betas", default="0.15,0.4,0.8")
    ap.add_argument("--baseline-check", action="store_true")
    ap.add_argument("--mapping-arm", choices=MAPPING_ARMS, default="depth_only")
    args = ap.parse_args()
    if args.frames<1 or args.stride<1 or args.start_index<0:
        ap.error("positive frames/stride and nonnegative start index required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    configs = {}
    for seed in seeds:
        for name, cfg in conditions(seed, tuple(map(float, args.betas.split(",")))).items():
            if name == "clean" and "clean" in configs:
                continue
            key = name if len(seeds) == 1 or name == "clean" else f"{name}_seed{seed}"
            if args.levels == "all" or name in args.levels.split(","):
                configs[key] = cfg
    configs = {"clean": conditions(args.seed)["clean"], **configs}
    builders = {k: make_builder(args.mapping_arm) for k in configs}
    manifest, rows, hashes = [], [], {}
    started = time.time()
    for robot in args.drones.split(","):
        sd = StereoDepth(load_stereo_calib(robot), DepthCfg())
        pose_path = Path(original.VO) / f"occ_{original.SEQ}_{robot}_af.npz"
        poses = np.load(pose_path)
        hashes[str(pose_path)] = digest(pose_path)
        idx = original.FrameIndex(robot)
        mates = original.Teammates(robot, original.ALL if hasattr(original, "ALL")
                                   else ["ifo001", "ifo002", "ifo003"], in_tag="_af")
        used = 0
        for i in range(args.start_index, len(poses["t"]), args.stride):
            tval = idx.pair(poses["t"][i])
            if tval is None:
                continue
            paths = [Path(original.DATA) / original.SEQ / robot / cam /
                     original._fname_for(robot, cam, tval) for cam in ("infra1", "infra2")]
            raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
            if any(p is None for p in raw):
                raise RuntimeError(f"missing stereo image {paths}")
            left, right = sd.rectify_pair(*raw)
            clean_z, clean_sz, clean_v = _depth_rectified(sd, left, right)
            if used == 0:
                old_z, old_sz, old_v = sd.depth(*raw)
                assert np.array_equal(clean_z, old_z, equal_nan=True)
                assert np.array_equal(clean_sz, old_sz, equal_nan=True)
                assert np.array_equal(clean_v, old_v)
            zr = clean_right_depth(sd, left, right)
            T, tr, t = poses["T"][i], poses["tr_sigma_pos"][i], float(poses["t"][i])
            manifest.append(dict(robot=robot, pose_index=i, pose_t=t, image_t=tval,
                                 left=str(paths[0]), right=str(paths[1]),
                                 left_sha256=digest(paths[0]), right_sha256=digest(paths[1])))
            for name, cfg in configs.items():
                il, ir, syn = stereo_smoke(sd, left, right, clean_z, zr, T, t, cfg)
                z, sz, valid = _depth_rectified(sd, il, ir)
                if name == "clean":
                    assert np.array_equal(z, clean_z, equal_nan=True)
                pc, sp = frame_points(sd, z, sz, valid)
                if pc is not None:
                    builders[name].integrate_frame(T, float(tr), pc, sp, teammates=mates.at(t))
                rows.append(dict(condition=name, seed=cfg.seed, beta=cfg.beta, field=cfg.field,
                                 robot=robot, pose_index=i, timestamp=t,
                                 **depth_metrics(z, valid, clean_z, clean_v), **syn))
                if used == 0 and robot == args.drones.split(",")[0]:
                    save_example(out, name, left, right, il, ir, clean_z, z)
            used += 1
            if used % 8 == 0:
                print(robot, used, "frames", round(time.time()-started, 1), "s", flush=True)
            if used >= args.frames:
                break
        if used < args.frames:
            raise RuntimeError(f"only {used}/{args.frames} frames for {robot}")
    if args.baseline_check:
        if args.start_index != 0 or args.frames > 8:
            raise ValueError("baseline check uses <=8 frames from index zero")
        baseline = make_builder(args.mapping_arm)
        for robot in args.drones.split(","):
            original.process_robot(baseline, robot, DepthCfg(), args.stride,
                                   max_frames=args.frames,
                                   mates=original.Teammates(robot, ["ifo001", "ifo002", "ifo003"], "_af"),
                                   in_tag="_af")
        baseline.finalize()
        builders["clean"].finalize()
        # Verify exact leaf log-odds, not only classified cell counts.
        def leaves(b):
            return {tuple(np.rint(np.array([n.getX(), n.getY(), n.getZ()])/.05).astype(int)):
                    float(n.getValue()) for n in b.tree.begin_leafs()}
        a, b = leaves(baseline), leaves(builders["clean"])
        assert a == b, "new clean path differs from existing process_robot"
        write_json(out / "baseline_check.json", dict(passed=True, exact_leaf_logodds=True,
                                                     leaves=len(a)))
    sparse = {}
    for name, builder in builders.items():
        builder.finalize()
        sparse[name] = sparse_map(builder)
    allp = np.concatenate([p for v in sparse.values() for p in v if len(p)])
    lo, hi = allp.min(axis=0), allp.max(axis=0) + 1
    maps = {k: dense_map(v, lo, hi-lo) for k, v in sparse.items()}
    summary = []
    for name, lab in maps.items():
        np.savez_compressed(out / f"map_{name}.npz", labels=lab, ijk_min=lo, res=.1)
        rr = [r for r in rows if r["condition"] == name]
        summary.append(dict(condition=name, mapping_arm=args.mapping_arm, **configs[name].metadata(), n_frames=len(rr),
                            valid_fraction=np.mean([r["valid_fraction"] for r in rr]),
                            depth_absdiff_mean_m=np.average(
                                [np.nan_to_num(r["depth_absdiff_mean_m"]) for r in rr],
                                weights=[max(r["n_joint"], 1) for r in rr]),
                            joint_fraction=np.mean([r["joint_fraction"] for r in rr]),
                            **map_metrics(lab, maps["clean"])))
    write_csv(out / "frames.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "manifest.json", dict(args=vars(args), frames=manifest, pose_hashes=hashes,
               map_domain=dict(ijk_min=lo, shape=hi-lo), occ_cfg=builders["clean"].cfg.__dict__,
               depth_cfg=DepthCfg().__dict__, elapsed_s=time.time()-started,
               protocol="fixed fused anchor-free poses; clean map is a comparison, not occupancy GT"))
    shown = {k:v for k,v in maps.items() if k in ("clean", "uniform_medium", "heterogeneous_medium", "heterogeneous_strong")}
    if len(shown) == 1 and len(maps)>1:
        shown = dict(list(maps.items())[:4])
    map_figure(out / "map_comparison.png", shown, lo)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

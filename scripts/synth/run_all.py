#!/usr/bin/env python3
"""Run every Part D cell (seeds x conditions) in one process tree.

    python scripts/synth/run_all.py --name room909 --seeds 0,1,2 --stride 4

SEQUENTIAL AND RESUMABLE, on purpose. This sandbox kills every form of detached
worker that was tried -- shell `&` children of a background task, setsid/nohup
processes, and multiprocessing.Pool (which hangs and is then SIGTERMed) -- so
the only thing that survives is one process doing one thing at a time. Instead
of fighting that, each cell's rows are appended to results_synth.csv the moment
they exist and ``--resume`` skips any cell already complete there, so a killed
run loses at most the cell it was in the middle of.

The datasets and the observability masks must exist first (--prep builds them);
they are shared read-only by every worker, so building them inside a worker
would race.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/synth")


ROWS_PER_UNIT = 3          # 3 arms per (seed, condition, coverage)


def done_units(csv_path):
    """{(seed, condition, coverage)} that already have all three arms."""
    if not os.path.exists(csv_path):
        return set()
    n = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            k = (int(r["seed"]), r["condition"], r["coverage"])
            n[k] = n.get(k, 0) + 1
    return {k for k, v in n.items() if v >= ROWS_PER_UNIT}


def prep(name, seeds, stride, node_stride):
    """Datasets (one per seed) and the observability masks (shared)."""
    import gtsam                       # noqa: F401
    from covor.synth import dataset as DS, mesh_gt as MG
    from covor.synth.config import SynthCfg
    from run_experiment import observability, COVERAGE
    for seed in seeds:
        cfg = SynthCfg(name=name, seq="synth_%s_0" % name, seed=seed)
        DS.install_paths(cfg)
        rep, _ = DS.build(cfg, verbose=False)
        print("  seed %d: uwb %d rows, residual rms %.4f m"
              % (seed, rep["uwb"]["n_rows_unique"], rep["uwb"]["residual_rms"]),
              flush=True)
    cfg = SynthCfg(name=name, seq="synth_%s_0" % name, seed=seeds[0])
    DS.install_paths(cfg)
    lab, ijk, res, _, _ = MG.load_gt(cfg.gt_voxel())
    gt = DS.load_gt_traj(cfg)
    scene = MG.raycasting_scene(DS.world_mesh(cfg))
    for key, cams in COVERAGE.items():
        t0 = time.time()
        obs = observability(cfg, scene, gt, lab, ijk, res, key, cams, stride,
                            node_stride)
        print("  obs mask %s: %d cells (%.0fs)" % (key, obs.sum(), time.time() - t0),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conds", default="A_1drone,B_2drone,C_3drone")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--node-stride", type=int, default=2)
    ap.add_argument("--mode", default="ideal")
    ap.add_argument("--coverages", default=None,
                    help="restrict to some coverage modes (comma separated)")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip cells already complete in results_synth.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    conds = args.conds.split(",")

    if args.prep:
        print("=== prep ===", flush=True)
        prep(args.name, seeds, args.stride, args.node_stride)

    import gtsam                       # noqa: F401  -- must precede open3d
    from covor.synth.config import SynthCfg
    from run_experiment import run_cell
    cfg0 = SynthCfg(name=args.name, seq="synth_%s_0" % args.name)
    csv_path = os.path.join(cfg0.outdir(), "results_synth.csv")
    from run_experiment import COVERAGE
    covs = args.coverages.split(",") if args.coverages else list(COVERAGE)
    have = done_units(csv_path) if args.resume else set()
    units = [(s, c, v) for s in seeds for c in conds for v in covs
             if (s, c, v) not in have]
    print("=== %d units to run (%d already complete) ==="
          % (len(units), len(have)), flush=True)
    t0 = time.time()
    for seed, cond, cov in units:
        cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=seed)
        t1 = time.time()
        try:
            run_cell(cfg, seed, cond, args.stride, args.node_stride, csv_path,
                     mode=args.mode, coverages=[cov])
            print("  done seed=%d %-9s %-14s (%.0fs)"
                  % (seed, cond, cov, time.time() - t1), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  FAILED seed=%d %s %s: %s" % (seed, cond, cov, e), flush=True)
    print("=== finished in %.0fs ===" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

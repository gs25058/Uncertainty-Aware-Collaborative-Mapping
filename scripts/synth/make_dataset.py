#!/usr/bin/env python3
"""Part B: generate the synthetic observations for the scanned room.

    python scripts/synth/make_dataset.py --name room909 [--seed 0] [--nlos]

Writes results/synth_<name>/{data,vins}/... (see covor.synth.dataset) plus a
trajectory figure and dataset_report.json. Import order matters: gtsam before
open3d (see covor.synth.__init__).
"""
import argparse
import json
import os
import sys

import gtsam                       # noqa: F401  -- must precede open3d
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.synth import mesh_gt as MG, dataset as DS
from covor.synth.config import SynthCfg, ROBOTS


def figure(cfg, gt, lab, ijk_min, res, out):
    occ, _ = MG.centers_of(lab, ijk_min, res, MG.OCC)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.6))
    m = (occ[:, 2] > 0.9) & (occ[:, 2] < 1.6)
    ax[0].scatter(occ[m, 0], occ[m, 1], c="0.75", s=1, label="GT occupied (flight band)")
    for c, r in zip("rgb", ROBOTS):
        p = gt[r]["p"]
        ax[0].plot(p[:, 0], p[:, 1], c + "-", lw=.8, label="%s GT (z=%.2f m)"
                   % (r, p[:, 2].mean()))
        ax[0].plot(p[0, 0], p[0, 1], c + "o", ms=7)
    ax[0].set_title("GT trajectories in their zones")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].axis("equal"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    for c, r in zip("rgb", ROBOTS):
        v = gt[r]["vio"]
        ax[1].plot(v["p"][:, 0], v["p"][:, 1], c + "-", lw=.8,
                   label="%s raw VIO (own local frame)" % r)
    ax[1].set_title("raw local-frame VIO handed to the fusion\n"
                    "(different origin and yaw per robot -- anchor-free)")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    ax[1].axis("equal"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=105)
    print("  saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nlos", action="store_true",
                    help="add a positive bias to ranges the mesh blocks")
    ap.add_argument("--duration", type=float, default=90.0)
    args = ap.parse_args()

    cfg = SynthCfg(name=args.name, seq="synth_%s_0" % args.name, seed=args.seed,
                   uwb_nlos=args.nlos, duration_s=args.duration)
    print("=== Part B: synthetic observations | %s | seed %d ===" % (cfg.seq, cfg.seed))
    rep, gt = DS.build(cfg)
    u = rep["uwb"]
    print("  UWB: %d unique rows, residual mean %+.4f m rms %.4f m (nlos=%s)"
          % (u["n_rows_unique"], u["residual_mean"], u["residual_rms"], u["nlos"]))
    lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
    figure(cfg, gt, lab, ijk_min, res,
           os.path.join(cfg.outdir(), "trajectories_seed%d.png" % cfg.seed))
    print(json.dumps(rep["robots"], indent=2, sort_keys=True)[:1200])
    print("  wrote", cfg.dataroot(), "and", cfg.vinsroot())


if __name__ == "__main__":
    main()

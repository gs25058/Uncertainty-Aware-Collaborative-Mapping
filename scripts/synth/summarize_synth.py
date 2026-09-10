#!/usr/bin/env python3
"""Tables and figures for Part D.

    python scripts/synth/summarize_synth.py --name room909

Prints markdown tables (mean +- sd over seeds) and writes the 1 vs 2 vs 3-drone
figure. Every metric the brief lists appears in the per-condition table: none of
them is quotable on its own, least of all false-free rate (RESULTS_SUMMARY §9-1).
Comparisons are Deltas WITHIN one coverage mode and one resolution.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.synth.config import SynthCfg

COND = ["A_1drone", "B_2drone", "C_3drone"]
ARMS = ["uniform", "depth_only", "full"]
COV = ["i_ifo001_only", "ii_all_cams"]
# Every metric the brief asks for, in its order.
# Tolerance 0 is the strict metric; 1 and 2 voxels absorb the discretisation
# gap the Part C control measured. The whole curve is reported, never one point
# (RESULTS_SUMMARY §9-4), and free/occupied counts travel with them so
# false-free can never be read alone (§9-1).
METRICS = [
    ("precision_tol0", "precision @0vox", 4),
    ("precision_tol1", "precision @1vox", 4),
    ("precision_tol2", "precision @2vox", 4),
    ("recall_tol0", "recall @0vox", 4),
    ("recall_tol1", "recall @1vox", 4),
    ("recall_tol2", "recall @2vox", 4),
    ("false_free_tol0", "false-free @0vox", 4),
    ("false_free_tol1", "false-free @1vox", 4),
    ("false_free_tol2", "false-free @2vox", 4),
    ("iou", "IoU @0vox", 4), ("iou_1vox", "IoU@1vox", 4),
    ("n_free", "free cells", 0), ("n_occupied", "occupied cells", 0),
    ("occ_over_gt", "occ/GT", 3), ("n_unknown", "unknown cells", 0),
    ("coverage_frac", "coverage", 4), ("ate_rmse_pooled", "ATE [m]", 4),
    ("sqrt_tr_sigma_median", "sqrt(trSigma) [m]", 4),
]


def agg(d, keys):
    g = d.groupby(keys)
    out = g.agg(["mean", "std", "count"])
    return out


def fmt(m, s, n, dec):
    if dec == 0:
        return "%d ± %d" % (round(m), 0 if np.isnan(s) else round(s))
    return ("%." + str(dec) + "f ± %." + str(dec) + "f") % (m, 0 if np.isnan(s) else s)


def table(d, cov, out):
    sub = d[d.coverage == cov]
    if sub.empty:
        return
    print("\n### coverage %s  (n seeds = %d)\n" % (cov, sub.seed.nunique()), file=out)
    hdr = "| metric | arm | " + " | ".join(COND) + " |"
    print(hdr, file=out)
    print("|" + "---|" * (2 + len(COND)), file=out)
    for key, label, dec in METRICS:
        for arm in ARMS:
            row = ["", arm] if arm != ARMS[0] else [label, arm]
            for c in COND:
                s = sub[(sub.condition == c) & (sub.arm == arm)][key]
                row.append(fmt(s.mean(), s.std(), len(s), dec) if len(s) else "-")
            print("| " + " | ".join(row) + " |", file=out)
    # Deltas against the 1-drone condition, within this coverage mode
    print("\n**Δ vs A_1drone** (same coverage, same resolution)\n", file=out)
    print("| metric | arm | B−A | C−A | C−B |", file=out)
    print("|---|---|---|---|---|", file=out)
    for key, label, dec in METRICS:
        for arm in ARMS:
            v = {c: sub[(sub.condition == c) & (sub.arm == arm)][key].mean()
                 for c in COND}
            row = [label if arm == ARMS[0] else "", arm]
            for a, b in (("B_2drone", "A_1drone"), ("C_3drone", "A_1drone"),
                         ("C_3drone", "B_2drone")):
                dv = v[a] - v[b]
                row.append(("%+.*f" % (max(dec, 1), dv)) if dec else "%+d" % round(dv))
            print("| " + " | ".join(row) + " |", file=out)


def figure(d, path):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    x = np.arange(len(COND))
    panels = [("ate_rmse_pooled", "ATE RMSE [m]  (lower better)"),
              ("sqrt_tr_sigma_median", "median sqrt(tr Sigma) [m]"),
              ("recall", "occupied recall (strict)"),
              ("false_free_rate", "false-free rate"),
              ("n_free", "free cells"),
              ("n_occupied", "occupied cells"),
              ("coverage_frac", "coverage (GT occupied observed)"),
              ("iou_1vox", "IoU@1vox")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for ci, cov in enumerate(COV):
            sub = d[d.coverage == cov]
            if sub.empty:
                continue
            for ai, arm in enumerate(ARMS):
                m = [sub[(sub.condition == c) & (sub.arm == arm)][key].mean()
                     for c in COND]
                e = [sub[(sub.condition == c) & (sub.arm == arm)][key].std()
                     for c in COND]
                ax.errorbar(x + (ci * 0.12 + ai * 0.03 - 0.08), m, yerr=e,
                            marker="os"[ci], ls="-" if ci == 0 else "--",
                            capsize=2, ms=5, lw=1.1,
                            color=plt.cm.tab10(ai), alpha=1.0 if ci == 0 else 0.55,
                            label="%s / %s" % (cov.split("_")[0], arm))
        ax.set_xticks(x); ax.set_xticklabels([c[0] for c in COND])
        ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Synthetic room909 | 1 vs 2 vs 3 drones | solid = coverage (i) "
                 "ifo001 camera only, dashed = coverage (ii) all cameras | "
                 "mean ± sd over 3 seeds")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    args = ap.parse_args()
    cfg = SynthCfg(name=args.name)
    p = os.path.join(cfg.outdir(), "results_synth.csv")
    d = pd.read_csv(p)
    d = d.drop_duplicates(subset=["seed", "condition", "coverage", "arm"],
                          keep="last")
    print("rows %d | seeds %s | strides %s"
          % (len(d), sorted(d.seed.unique()), sorted(d.stride.unique())))
    out = open(os.path.join(cfg.outdir(), "tables_synth.md"), "w")
    for cov in COV:
        table(d, cov, out)
        table(d, cov, sys.stdout)
    out.close()
    figure(d, os.path.join(cfg.outdir(), "fig_drone_count.png"))


if __name__ == "__main__":
    main()

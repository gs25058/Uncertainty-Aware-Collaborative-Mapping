#!/usr/bin/env python3
"""Summarize sweep_results.csv: rank configs by mean ATE vs the mono-VO baseline
and report per-robot improvement of the best few."""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "/src/gs25058/cr_RNE/covor_slam/sweep_results.csv"
df = pd.read_csv(CSV)

base = df[df.tag == "mono_vo_baseline"].iloc[0]
b = np.array([base.ate_ifo001, base.ate_ifo002, base.ate_ifo003], dtype=float)
print("mono-VO baseline ATE [ifo001 ifo002 ifo003] = [%.3f %.3f %.3f]  mean=%.3f\n"
      % (*b, b.mean()))

g = df[~df.tag.isin(["mono_vo_baseline"])].copy()
for c in ["ate_ifo001", "ate_ifo002", "ate_ifo003", "ate_mean"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")
g = g.dropna(subset=["ate_mean"]).sort_values("ate_mean")

g["impr_mean_%"] = (100 * (b.mean() - g["ate_mean"]) / b.mean()).round(1)
g["impr_003_%"] = (100 * (b[2] - g["ate_ifo003"]) / b[2]).round(1)

cols = ["tag", "bias_mode", "range_sigma_floor", "huber_k", "prior_every",
        "ate_ifo001", "ate_ifo002", "ate_ifo003", "ate_mean",
        "impr_mean_%", "impr_003_%", "iters", "time_s"]
pd.set_option("display.width", 200, "display.max_columns", 30)
print("=== top 12 configs by mean ATE (lower is better) ===")
print(g[cols].head(12).to_string(index=False))
print("\n=== worst 3 ===")
print(g[cols].tail(3).to_string(index=False))

best = g.iloc[0]
print("\nBEST: %s" % best.tag)
print("  per-robot ATE vs VO:")
for name, bb, ff in zip(["ifo001", "ifo002", "ifo003"], b,
                        [best.ate_ifo001, best.ate_ifo002, best.ate_ifo003]):
    print("    %-7s VO %.3f -> fused %.3f  (%+.1f%%)" % (name, bb, ff, 100 * (bb - ff) / bb))

#!/usr/bin/env python3
"""Rank sweep_height_results.csv by mean 3D ATE and show the ablation + best grid
configs with per-robot 3D / vertical / horizontal error vs the mono-VO baseline."""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "/src/gs25058/cr_RNE/covor_slam/sweep_height_results.csv"
df = pd.read_csv(CSV)
R = ["ifo001", "ifo002", "ifo003"]

base = df[df.tag == "mono_vo_baseline"].iloc[0]
b = np.array([base[f"ate_{r}"] for r in R], float)
print("mono-VO baseline  ATE=[%.3f %.3f %.3f] mean=%.3f  z=[%.3f %.3f %.3f]\n"
      % (*b, b.mean(), *[base[f"z_{r}"] for r in R]))

g = df[df.tag != "mono_vo_baseline"].copy()
for c in g.columns:
    if c.startswith(("ate_", "z_", "h_")):
        g[c] = pd.to_numeric(g[c], errors="coerce")
g["impr_%"] = (100 * (b.mean() - g["ate_mean"]) / b.mean()).round(1)

show = ["tag", "ate_ifo001", "ate_ifo002", "ate_ifo003", "ate_mean", "impr_%",
        "z_ifo001", "z_ifo002", "z_ifo003", "time_s"]
pd.set_option("display.width", 240, "display.max_columns", 40)

abl = g[g.note == "ablation"] if "note" in g else g.iloc[:0]
print("=== ablation ===")
print(abl[show].to_string(index=False))

grid = g[g.note != "ablation"].sort_values("ate_mean")
print("\n=== top 10 grid configs (by mean 3D ATE) ===")
print(grid[show].head(10).to_string(index=False))

best = g.sort_values("ate_mean").iloc[0]
print("\nBEST overall: %s  (mean ATE %.3f, %+.1f%% vs VO)"
      % (best.tag, best.ate_mean, 100 * (b.mean() - best.ate_mean) / b.mean()))
for i, r in enumerate(R):
    print("   %-7s VO %.3f -> %.3f  (%+.1f%%)   z %.3f->%.3f"
          % (r, b[i], best[f"ate_{r}"], 100 * (b[i] - best[f"ate_{r}"]) / b[i],
             base[f"z_{r}"], best[f"z_{r}"]))

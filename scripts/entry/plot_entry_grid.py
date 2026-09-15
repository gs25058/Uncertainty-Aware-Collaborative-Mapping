#!/usr/bin/env python3
"""Draw an entry grid AS A GRID -- no vector layer (DESIGN §5 is Part 4).

    python scripts/entry/plot_entry_grid.py --in <entry_grid.npz> --out <png>

This is the judgement layer shown unretouched: one pixel per 0.10 m cell, the
labels exactly as entry_grid.npz holds them. When render.py exists, anything it
draws that disagrees with this picture is a bug in render.py.
"""
import argparse
import json

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm   # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
# Imported, never restated. An earlier version of this file redeclared them as
# "UNKNOWN, OCCUPIED, FREE = 0, 1, 2" -- the right names in the wrong order --
# and drew every occupied cell as free and every free cell as occupied. Nothing
# failed; the picture was simply a lie, which is the exact failure mode
# RESULTS_SUMMARY appendix A is a list of.
from covor.entry.config import (UNKNOWN, FREE, OCCUPIED,      # noqa: E402
                                BLOCKED, NARROW, WALK)

# unknown, occupied, free-blocked, free-narrow, free-walk
COLORS = ["#c8c8c8", "#101010", "#8a8a8a", "#f0c419", "#ffffff"]
NAMES = ["unknown", "occupied", "free / blocked", "free / narrow", "free / walk"]


def panel(ax, g, band, res, ijk_min):
    lab = g["%s_label" % band]
    wc = g["%s_width_class" % band]
    img = np.full(lab.shape, 0, np.uint8)
    img[lab == UNKNOWN] = 0
    img[lab == OCCUPIED] = 1
    free = lab == FREE
    img[free & (wc == BLOCKED)] = 2
    img[free & (wc == NARROW)] = 3
    img[free & (wc == WALK)] = 4
    x0 = (ijk_min[0]) * res
    y0 = (ijk_min[1]) * res
    ext = [x0, x0 + lab.shape[0] * res, y0, y0 + lab.shape[1] * res]
    cmap = ListedColormap(COLORS)
    ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, 5.5), cmap.N), interpolation="nearest")
    # passable but cut off from the entry point: the map's own warning
    lost = np.argwhere(g["%s_passable" % band] & ~g["%s_reachable" % band])
    if len(lost):
        w = (lost + np.asarray(ijk_min[:2]) + 0.5) * res
        ax.plot(w[:, 0], w[:, 1], ".", ms=2.5, color="tab:red", lw=0,
                label="passable, not reachable (%d)" % len(lost))
    e = g["entry_xy"]
    ax.plot(e[0], e[1], "o", ms=9, mfc="tab:blue", mec="k", mew=1.2, label="entry")
    rows = g["%s_rows" % band]
    zlo = (rows[0] + ijk_min[2]) * res
    zhi = (rows[1] + ijk_min[2]) * res
    ax.set_title("%s band  z = %.2f - %.2f m  (rows %d-%d)"
                 % (band, zlo, zhi, rows[0], rows[1] - 1))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal"); ax.grid(alpha=.15, lw=.4)
    ax.legend(loc="upper right", fontsize=7, framealpha=.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    g = np.load(args.inp, allow_pickle=False)
    res = float(g["res"][0]); ijk_min = g["ijk_min"]
    cfg = json.loads(str(g["cfg"]))

    fig, ax = plt.subplots(1, 2, figsize=(16.5, 6.4))
    for a, band in zip(ax, ("walk", "crawl")):
        panel(a, g, band, res, ijk_min)
    handles = [Patch(fc=c, ec="0.3", lw=.4, label=n) for c, n in zip(COLORS, NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("%s  |  res %.2f m, w %.2f m, k_sigma %s, close %s x%d, "
                 "cube_min %d  |  grid as judged, no render layer"
                 % (args.title or args.inp, res, cfg["w"], cfg["k_sigma"],
                    cfg["close_structure"], cfg["close_iter"], cfg["cube_min_voxels"]),
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()

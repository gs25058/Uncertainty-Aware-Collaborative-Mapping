#!/usr/bin/env python3
"""The finished entry map: grid -> routes -> SVG + PNG (DESIGN_entry_map.md §5).

    python scripts/entry/make_entry_figure.py --gt results/synth_room909/gt_voxel.npz \
        --map <dump>.npz:C --map <dump_A>.npz:A --out results/.../entry_map

One column per map, one row per band. Every figure carries the conditions it was
made under, and a map built from a corrupted run says so on its own face -- a
picture that circulates without its caption should still not be able to mislead.

Also writes routes.json and, when a GT map is given, the route metrics of
DESIGN §6 (validity, length ratio). The GT is only ever a scorer here: it never
touches what is drawn.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts/entry")
from covor.entry import clean3d, metrics as EM, render as RD, route as RT
from covor.entry.config import EntryCfg
from build_entry_map import build_entry_grid, band_table, BANDS


def load(path):
    z = np.load(path, allow_pickle=False)
    lab = (z["labels"] if "labels" in z.files
           else clean3d.labels_from_masks(z["M_occ"], z["M_free"]))
    return (lab, z["ijk_min"], float(np.asarray(z["res"]).ravel()[0]),
            z["sigma_xy"] if "sigma_xy" in z.files else None,
            z["n_obs"] if "n_obs" in z.files else None,
            json.loads(str(z["meta"])) if "meta" in z.files else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=None, help="gt_voxel.npz, for scoring only")
    ap.add_argument("--map", action="append", required=True, help="path[:label]")
    ap.add_argument("--out", required=True, help="output stem (no extension)")
    ap.add_argument("--w", type=float, default=0.70)
    ap.add_argument("--k-sigma", type=float, default=0.0)
    ap.add_argument("--bands", default=",".join(BANDS))
    ap.add_argument("--title", default="")
    ap.add_argument("--sigma-panel", action="store_true",
                    help="add a row showing the measured sigma_xy field. It "
                         "enters no decision at k_sigma = 0 and the panel says so.")
    args = ap.parse_args()

    bands = args.bands.split(",")
    # the grid resolution comes from the DATA, never from EntryCfg's default.
    # Handing a 0.05 m grid to a cfg that still says 0.10 puts the band at
    # 0.05-0.95 m instead of 0.10-1.90 m and every number after it is quietly
    # wrong -- nothing raises, the map just describes a different slab.
    res_seen = {load(spec.partition(":")[0])[2] for spec in args.map}
    if len(res_seen) > 1:
        ap.error("maps are on different resolutions: %s" % sorted(res_seen))
    cfg = EntryCfg(res=res_seen.pop(), w=args.w, k_sigma=args.k_sigma)
    maps = []
    for spec in args.map:
        path, _, lb = spec.partition(":")
        lab, ijk, res_m, sig, nobs, meta = load(path)
        g, info = build_entry_grid(lab, cfg, sigma_xy=sig, n_obs=nobs,
                                   ijk_min=ijk, k_sigma=args.k_sigma)
        maps.append(dict(label=lb or os.path.basename(path), grid=g, info=info,
                         meta=meta, path=path))

    gt_grid = None
    if args.gt:
        zg = np.load(args.gt, allow_pickle=False)
        if abs(float(np.asarray(zg["res"]).ravel()[0]) - cfg.res) > 1e-9:
            ap.error("the GT grid is %.3f m and the maps are %.3f m -- "
                     "PREREG_RESOLUTION.md forbids scoring across resolutions"
                     % (float(np.asarray(zg["res"]).ravel()[0]), cfg.res))
        gt_grid, _ = build_entry_grid(zg["labels"], cfg, ijk_min=zg["ijk_min"],
                                      k_sigma=0.0)

    routes_all = {}
    # One map -> lay the bands out side by side; a single column two panels tall
    # is the wrong shape for a paper page. Several maps -> maps across, bands
    # down, so a reader compares like with like by looking along a row.
    side_by_side = len(maps) == 1
    nr, nc = (1, len(bands)) if side_by_side else (len(bands), len(maps))
    if args.sigma_panel:
        nr += 1
    fig, axes = plt.subplots(nr, nc, figsize=(9.0 * nc, 6.4 * nr), squeeze=False)
    for c, m in enumerate(maps):
        for r, band in enumerate(bands):
            ax = axes[0][r] if side_by_side else axes[r][c]
            rt = RT.plan(m["grid"], band, cfg)
            routes_all["%s/%s" % (m["label"], band)] = rt
            rows = np.asarray(m["grid"]["%s_rows" % band])
            ijk = np.asarray(m["grid"]["ijk_min"])
            zlo = (rows[0] + ijk[2]) * cfg.res
            zhi = (rows[1] + ijk[2]) * cfg.res
            ttl = "%s band   ($z$ = %.2f - %.2f m)" % (band, zlo, zhi)
            # the conditions ride above the panels, never on an xlabel: a caption
            # under the last row gets clipped by bbox_inches="tight" and
            # collides with its neighbour's. With one map they belong to the
            # whole figure, so they go under the suptitle instead of on a panel.
            if not side_by_side and r == 0:
                ttl = "%s\n%s\n%s" % (m["label"], ttl,
                                       RD.caption(m["grid"], cfg, m["meta"]))
            RD.draw_band(ax, m["grid"], band, cfg, routes=rt, title=ttl)
            if r == 0:
                ax.title.set_fontsize(9.5 if not side_by_side else 11)

    if args.sigma_panel:
        vmax = max(float(np.nanpercentile(np.where(m["grid"]["sigma_xy"] > 0,
                                                   m["grid"]["sigma_xy"], np.nan), 98))
                   for m in maps)
        for c, m in enumerate(maps):
            ax = axes[-1][c if not side_by_side else 0]
            im = RD.draw_sigma(ax, m["grid"], cfg, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=.82, pad=.02,
                         label="$\\sigma_{xy}$ [m]")
        for c in range(len(maps), nc):
            axes[-1][c].axis("off")

    fig.legend(handles=RD.legend_handles(cfg), loc="lower center",
               ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.002))
    top = 0.975
    if args.title:
        fig.suptitle(args.title, fontsize=13.5, y=0.995)
    if side_by_side:
        fig.text(0.5, 0.905 if args.title else 0.965,
                 RD.caption(maps[0]["grid"], cfg, maps[0]["meta"]).replace("\n", "   |   "),
                 ha="center", va="top", fontsize=8.5)
        top = 0.875 if args.title else 0.93
    bottom = 0.155 / nr + 0.02
    fig.tight_layout(rect=[0, bottom, 1, top], h_pad=2.6, w_pad=1.6)
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (args.out, ext), dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print("-> %s.%s" % (args.out, ext))
    plt.close(fig)

    # routes.json, plus the §6 route metrics when a GT is available
    payload = {}
    for key, rt in routes_all.items():
        band = key.split("/")[-1]
        recs = []
        for rec in rt["routes"]:
            out = {k: v for k, v in rec.items() if k != "cells"}
            if gt_grid is not None:
                out.update(EM.score_route(rec, gt_grid, band, cfg))
            recs.append(out)
        payload[key] = dict(entry_ij=rt["entry_ij"], routes=recs)
    with open(args.out + "_routes.json", "w") as f:
        json.dump(dict(w=cfg.w, k_sigma=cfg.k_sigma, lambda_route=cfg.lambda_route,
                       unknown_penalty_steps=cfg.unknown_penalty_steps,
                       routes=payload), f, indent=2)
    print("-> %s_routes.json" % args.out)

    for m in maps:
        print("\n%s  %s" % (m["label"], {k: m["meta"].get(k) for k in
                                         ("cond", "coverage", "arm", "corrupt")}))
        for row in band_table(m["grid"]):
            if row["band"] in bands:
                print("   %-6s free %5d (blocked %4d / narrow %3d / walk %4d)  "
                      "passable %5d  reachable %5d  unknown %5d"
                      % (row["band"], row["free"], row["free_blocked"],
                         row["free_narrow"], row["free_walk"], row["passable"],
                         row["reachable"], row["unknown"]))
        for band in bands:
            for rec in payload["%s/%s" % (m["label"], band)]["routes"]:
                if not rec.get("reachable"):
                    print("   %-6s route to %s: NOT REACHABLE" % (band, rec["goal_ij"]))
                    continue
                print("   %-6s route %.2f m, min width %.2f m at (%.2f, %.2f), "
                      "%d cells beside unknown%s"
                      % (band, rec["length_m"], rec["min_width_m"],
                         rec["min_width_at"][0], rec["min_width_at"][1],
                         rec["n_unknown_adjacent"],
                         ("  | GT-valid %.3f, length ratio %.3f"
                          % (rec["route_validity"], rec["route_length_ratio"]))
                         if gt_grid is not None else ""))


if __name__ == "__main__":
    main()

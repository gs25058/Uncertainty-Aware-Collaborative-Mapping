#!/usr/bin/env python3
"""Why the Part C-1 control map misses the recall / false-free thresholds.

Reads results/synth_<name>/gate_masks.npz (dumped by gate.py) and attributes
every failing cell to a mechanism, by measurement:

  * how far a false-free cell is from the nearest cell the map DID call
    occupied (1 voxel => the surface straddles a voxel boundary; more => a real
    punch-through);
  * what log-odds the missed GT cells actually carry (0 => never hit at all;
    0 < l <= tau_occ => hit but below the single-observation ceiling that
    tau_occ = l_occ defines, appendix A-6);
  * how thick the GT surface shell is, since a beam can only terminate in one
    cell of it.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.occupancy import OccCfg
from covor.synth import mesh_gt as MG, metrics as ME
from covor.synth.config import SynthCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="room909")
    args = ap.parse_args()
    cfg = SynthCfg(name=args.name)
    lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
    z = np.load(os.path.join(cfg.outdir(), "gate_masks.npz"))
    M_occ, M_free, obs, L = z["M_occ"], z["M_free"], z["obs"], z["logodds"]
    occ_cfg = OccCfg()

    G = lab == MG.OCC
    dom = ((lab == MG.OCC) | (lab == MG.FREE)) & obs
    g = G & dom
    ff = M_free & g                       # false-free cells
    fn = g & ~M_occ                       # missed GT cells (free OR unknown)
    print("domain %d | GT occ in domain %d | map occ %d | false-free %d | missed %d"
          % (dom.sum(), g.sum(), (M_occ & dom).sum(), ff.sum(), fn.sum()))

    # --- 1. how far is a false-free cell from a cell the map called occupied? -
    d = ndimage.distance_transform_edt(~M_occ)          # in voxels
    dd = d[ff]
    print("\n[1] false-free cell -> nearest MAP-occupied cell (voxels)")
    for lim in (1.0, 1.5, 2.0, 3.0):
        print("     <= %.1f vox : %6d (%.1f%%)" % (lim, (dd <= lim).sum(),
                                                   100 * (dd <= lim).mean()))
    print("     max %.2f vox, median %.2f" % (dd.max(), np.median(dd)))

    # --- 2. what log-odds do the missed cells carry? -------------------------
    lv = L[fn]
    never = np.isnan(lv)
    seen = lv[~never]
    print("\n[2] log-odds on the %d missed GT cells (tau_occ = %.2f)"
          % (fn.sum(), occ_cfg.tau_occ))
    print("     never written        : %6d (%.1f%%)" % (never.sum(), 100 * never.mean()))
    if seen.size:
        band = (seen > 0) & (seen <= occ_cfg.tau_occ)
        print("     0 < l <= tau_occ     : %6d (%.1f%%)  <- hit, but one view can"
              " never exceed tau_occ" % (band.sum(), 100 * band.mean()))
        print("     l <= 0 (net free)    : %6d (%.1f%%)" % ((seen <= 0).sum(),
                                                            100 * (seen <= 0).mean()))
        print("     median l = %+.3f, p10 %+.3f, p90 %+.3f"
              % (np.median(seen), np.percentile(seen, 10), np.percentile(seen, 90)))

    # --- 3. GT shell thickness ----------------------------------------------
    # a beam terminates in ONE cell; a surface that touches two cells therefore
    # leaves the other one to be carved free by the same beam
    lbl = G.astype(np.uint8)
    thick = ndimage.distance_transform_edt(G)
    print("\n[3] GT surface shell: %d occupied cells, %.1f%% have an occupied "
          "6-neighbour on the SAME surface"
          % (G.sum(), 100 * (ndimage.convolve(G.astype(np.int8),
                                              ndimage.generate_binary_structure(3, 1)
                                              .astype(np.int8), mode="constant")[G] > 1).mean()))
    print("     cells with GT-occupied neighbours in >=2 opposite directions "
          "(a 2-cell-thick wall): %.1f%%"
          % (100 * _straddle(G)[G].mean()))

    # --- 4. tolerant false-free ---------------------------------------------
    dm = ME.dilate1(M_occ)
    ff_t = ff & ~dm
    print("\n[4] false-free rate")
    print("     strict            %.4f  (%d / %d)" % (ff.sum() / g.sum(), ff.sum(), g.sum()))
    print("     @1vox tolerance   %.4f  (%d / %d)  <- the map called SOME cell "
          "within 1 voxel occupied" % (ff_t.sum() / g.sum(), ff_t.sum(), g.sum()))

    out = dict(n_domain=int(dom.sum()), n_gt_occ=int(g.sum()),
               n_false_free=int(ff.sum()), n_missed=int(fn.sum()),
               ff_within_1vox=float((dd <= 1.0).mean()),
               ff_strict=float(ff.sum() / g.sum()),
               ff_tolerant=float(ff_t.sum() / g.sum()),
               missed_never_written=float(never.mean()),
               missed_below_tau=float(((seen > 0) & (seen <= occ_cfg.tau_occ)).mean())
               if seen.size else 0.0)
    p = os.path.join(cfg.outdir(), "gate_diagnosis.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("\nwrote", p)


def _straddle(G):
    """Cells whose GT-occupied neighbourhood spans opposite faces on some axis
    -- i.e. the surface is at least two cells thick there."""
    out = np.zeros(G.shape, bool)
    for ax in range(3):
        a = np.roll(G, 1, ax)
        b = np.roll(G, -1, ax)
        out |= (a & G) | (b & G)
    return out


if __name__ == "__main__":
    main()

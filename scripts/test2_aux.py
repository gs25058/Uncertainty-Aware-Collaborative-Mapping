#!/usr/bin/env python3
"""Test 2 auxiliary diagnostics (data only, no optimization).

(1) gt_range - ||mocap body-centre distance||, split by TAG. gt_range is the true
    antenna-to-antenna / antenna-to-anchor distance (mocap-based); the body-centre
    distance is what the current factor predicts. Their difference is exactly the
    lever-arm effect. Per-tag mean/std shows its size and whether it is systematic.
(2) Per-anchor decomposition of that difference: one anchor standing out => bad
    anchor coordinate; uniform across anchors => geometric (lever arm) effect.

Run:  python scripts/test2_aux.py
"""
import sys
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam/scripts")
from covor.fusion import CoVOR, Cfg
from covor import data as D
from diaglog import log_rows

SEQ = "default_3_zigzag_0"


def main():
    cov = CoVOR(SEQ, Cfg()).build()
    anchors = cov.anchors
    rows = []

    # per-tag lever-arm effect: gt_range - body-centre distance -----------------
    per_tag = {}          # tag -> list of (gt_range - d_body)
    per_anchor = {}       # anchor id -> list
    for _, r in cov.ranges.iterrows():
        t = float(r.timestamp)
        fk = D.robot_of_tag(int(r.from_id))
        if fk is None:
            continue
        ka = int(fk[-1]) - 1
        ia = cov._assoc(ka, t)
        if ia is None:
            continue
        pa = D.mocap_position_at(cov.robots[ka].mocap, cov.robots[ka].t[ia])
        gt = float(r["gt_range"])
        if r["kind"] == "anchor":
            aid = int(r.to_id)
            if aid not in anchors:
                continue
            d_body = np.linalg.norm(pa - anchors[aid])
            per_tag.setdefault(int(r.from_id), []).append(gt - d_body)
            per_anchor.setdefault(aid, []).append(gt - d_body)
        else:
            tk = D.robot_of_tag(int(r.to_id))
            if tk is None:
                continue
            kb = int(tk[-1]) - 1
            ib = cov._assoc(kb, t)
            if ib is None:
                continue
            pb = D.mocap_position_at(cov.robots[kb].mocap, cov.robots[kb].t[ib])
            d_body = np.linalg.norm(pa - pb)
            per_tag.setdefault((int(r.from_id), int(r.to_id)), []).append(gt - d_body)

    print("=== (1) gt_range - body-centre distance, by TAG (lever-arm effect) ===",
          flush=True)
    for tag in sorted(per_tag, key=str):
        a = np.array(per_tag[tag])
        print("  tag %-10s n=%4d  mean=%+.3f  std=%.3f  |mean|-> lever arm size"
              % (str(tag), len(a), a.mean(), a.std()), flush=True)
        rows.append(dict(test="T2aux_leverarm", config="mocap_gt", robot=str(tag),
                         metric="gt_minus_bodydist_mean", value=round(a.mean(), 4),
                         note="std=%.3f n=%d" % (a.std(), len(a))))

    print("\n=== (2) same difference by ANCHOR id (coord error vs geometry) ===",
          flush=True)
    for aid in sorted(per_anchor):
        a = np.array(per_anchor[aid])
        print("  anchor %d  pos=%s  n=%4d  mean=%+.3f  std=%.3f"
              % (aid, np.round(anchors[aid], 2), len(a), a.mean(), a.std()), flush=True)
        rows.append(dict(test="T2aux_peranchor", config="mocap_gt", robot="anchor%d" % aid,
                         metric="gt_minus_bodydist_mean", value=round(a.mean(), 4),
                         note="std=%.3f n=%d" % (a.std(), len(a))))

    # also: real range vs gt_range (measurement quality) ----------------------
    rr = cov.ranges
    err = (rr["range"] - rr["gt_range"]).to_numpy()
    print("\n=== (3) measured range - gt_range (real UWB error) mean=%+.3f std=%.3f "
          "p99|.|=%.3f ===" % (err.mean(), err.std(), np.percentile(np.abs(err), 99)),
          flush=True)
    rows.append(dict(test="T2aux_uwberr", config="mocap_gt", robot="all",
                     metric="range_minus_gtrange", value=round(float(err.mean()), 4),
                     note="std=%.3f" % err.std()))
    log_rows(rows)


if __name__ == "__main__":
    main()

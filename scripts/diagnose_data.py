#!/usr/bin/env python3
"""Data-only auxiliary diagnostics (no optimization) for the fusion regression.

(a) Lever-arm size: gt_range vs distance between mocap BODY CENTERS (ignoring the
    tag moment arms). The gap is exactly the rotating-antenna (moment-arm) effect
    the current body-center range factor omits. Reported per from-tag.
(b) Antenna-aware check: does gt_range match ||antenna_a - antenna_b|| when we DO
    apply the tag moment arms via mocap orientation? If yes, gt_range is the true
    antenna distance and the factor's body-center model is provably mis-specified.
(c) Per-anchor residual decomposition at ground-truth poses: for robot-1 anchor
    ranges, mean/std of (||body-anchor|| - gt_range) and (||body-anchor|| - range)
    per anchor id. A constant per-anchor offset -> anchor coord error; a spread
    that tracks orientation -> geometry/lever-arm.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor import data as D

SEQ = sys.argv[1] if len(sys.argv) > 1 else "default_3_zigzag_0"

ranges = D.load_ranges(SEQ)
anchors = D.load_anchors(SEQ)
arms = D.load_tag_arms()
mocaps = {r: D.load_mocap(SEQ, r) for r in D.ROBOTS}


def body_pose_at(robot, t):
    """Return (position[3], Rotation) of robot body at nearest mocap time."""
    m = mocaps[robot]
    i = int(np.abs(m.timestamp.values - t).argmin())
    row = m.iloc[i]
    p = np.array([row.x, row.y, row.z], float)
    q = np.array([row.qx, row.qy, row.qz, row.qw], float)
    return p, Rot.from_quat(q)


def antenna_pos(robot, tag_id, t):
    p, R = body_pose_at(robot, t)
    return p + R.apply(arms[int(tag_id)])


print(f"=== DATA DIAGNOSTICS  seq={SEQ} ===")
print(f"total range rows (deduped, std<=0.5): {len(ranges)}  "
      f"anchor={int((ranges.kind=='anchor').sum())} inter={int((ranges.kind=='inter').sum())}")

# ---------------------------------------------------------------------------
# (a)+(b) lever-arm effect, per from-tag
# ---------------------------------------------------------------------------
print("\n-- (a/b) lever-arm: gt_range vs body-center dist, and vs antenna dist --")
print(f"{'from_tag':>8s} {'kind':>7s} {'n':>5s} "
      f"{'|gtR-bodyD| mean':>16s} {'std':>7s} {'|gtR-antD| mean':>15s} {'std':>7s}")
rowsum = []
for tag, g in ranges.groupby("from_id"):
    frobot = D.robot_of_tag(int(tag))
    if frobot is None:
        continue
    d_body, d_ant = [], []
    for _, r in g.iterrows():
        t = float(r.timestamp)
        pa = antenna_pos(frobot, int(r.from_id), t)
        ba, _ = body_pose_at(frobot, t)
        if r.kind == "anchor":
            aid = int(r.to_id)
            if aid not in anchors:
                continue
            tgt_body = tgt_ant = anchors[aid]
        else:
            trobot = D.robot_of_tag(int(r.to_id))
            if trobot is None:
                continue
            tgt_ant = antenna_pos(trobot, int(r.to_id), t)
            tgt_body, _ = body_pose_at(trobot, t)
        gtr = float(r.gt_range)
        d_body.append(abs(gtr - np.linalg.norm(ba - tgt_body)))
        d_ant.append(abs(gtr - np.linalg.norm(pa - tgt_ant)))
    d_body, d_ant = np.array(d_body), np.array(d_ant)
    kind = g.kind.iloc[0]
    print(f"{int(tag):>8d} {kind:>7s} {len(d_body):>5d} "
          f"{d_body.mean():>16.4f} {d_body.std():>7.4f} "
          f"{d_ant.mean():>15.4f} {d_ant.std():>7.4f}")

# ---------------------------------------------------------------------------
# (c) per-anchor residual decomposition at GT (robot-1 anchor ranges)
# ---------------------------------------------------------------------------
print("\n-- (c) per-anchor residuals at GT poses (body-center model) --")
print(f"  resid_gt  = ||body - anchor|| - gt_range   (pure geometry/lever-arm)")
print(f"  resid_meas= ||body - anchor|| - range      (adds UWB meas error+bias)")
print(f"{'anchor':>6s} {'from_tag':>8s} {'n':>5s} "
      f"{'resid_gt mean':>14s} {'std':>7s} {'resid_meas mean':>16s} {'std':>7s}")
anc = ranges[ranges.kind == "anchor"]
for (aid, tag), g in anc.groupby(["to_id", "from_id"]):
    frobot = D.robot_of_tag(int(tag))
    if frobot is None or int(aid) not in anchors:
        continue
    rg, rm = [], []
    for _, r in g.iterrows():
        t = float(r.timestamp)
        b, _ = body_pose_at(frobot, t)
        d = np.linalg.norm(b - anchors[int(aid)])
        rg.append(d - float(r.gt_range))
        rm.append(d - float(r["range"]))
    rg, rm = np.array(rg), np.array(rm)
    print(f"{int(aid):>6d} {int(tag):>8d} {len(rg):>5d} "
          f"{rg.mean():>14.4f} {rg.std():>7.4f} {rm.mean():>16.4f} {rm.std():>7.4f}")

# overall const bias implied by measurement vs gt
print("\n-- overall (range - gt_range) stats (the systematic UWB bias) --")
for kind, g in ranges.groupby("kind"):
    d = (g["range"] - g["gt_range"]).to_numpy()
    print(f"  {kind:>7s}: n={len(d):5d} mean={d.mean():+.4f} std={d.std():.4f} "
          f"median={np.median(d):+.4f}")

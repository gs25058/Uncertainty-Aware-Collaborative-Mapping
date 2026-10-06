#!/usr/bin/env python3
"""Slide figure 1: MILUV default_3_zigzag_0, three drones, top-down x-y.

    python scripts/slides/s1_traj.py            # covor env python

(a) VIO alone: each drone's RAW VINS trajectory (vio.csv via covor.data.load_vins --
    MILUV clock = timeshift_s + timeshift_ns/1e9, quaternion w-first; never a
    mocap-aligned file, HANDOFF pitfall 2/12/15), each in its OWN local frame,
    not aligned to anything. The point is that the three frames do not agree.
(b) UWB collaboration, anchor-free (condition C): the fused body positions
    (vo_output/occ_<seq>_<robot>_af.npz, written by scripts/fuse_and_dump.py
    --anchor-free) over GT. GT = covor.data._mocap_splines (cleaned mocap,
    pitfall 20). Anchor-free fixes no absolute frame (the fused frame sits
    0.71-0.82 m from mocap), so ONE yaw+translation transform common to all three
    drones is fitted for display; per-drone alignment would hide exactly the
    inter-drone registration the panel is about.
The ATE printed in (b) is the per-drone 4-DoF ATE (recomputed here and asserted
equal to results/results_49_collaboration.csv, condition C).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from covor import data as D

SEQ = "default_3_zigzag_0"
ROBOTS = ("ifo001", "ifo002", "ifo003")
COL = {"ifo001": "#1f77b4", "ifo002": "#d62728", "ifo003": "#2ca02c"}
NAME = {"ifo001": "드론 1", "ifo002": "드론 2", "ifo003": "드론 3"}
CSV_ATE_C = {"ifo001": 0.1517, "ifo002": 0.2218, "ifo003": 0.3010}   # results_49_collaboration.csv
OUT = "/src/gs25058/cr_RNE/covor_slam/paper/figures/slides/s1_traj.png"
VO = "/src/gs25058/cr_RNE/covor_slam/vo_output/occ_%s_%s_af.npz"

plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False,
                     "font.size": 15, "mathtext.fontset": "dejavusans"})


def fit4(P, Q):
    """yaw + translation (4-DoF) least squares mapping P onto Q -> (R, t)."""
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp)[:, :2].T @ (Q - mq)[:, :2]
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return R, mq - R @ mp


def main():
    fused, gtp, ate = {}, {}, {}
    for r in ROBOTS:
        z = np.load(VO % (SEQ, r))
        ps, _, (t0, t1), _, _ = D._mocap_splines(SEQ, r)
        m = (z["t"] >= t0) & (z["t"] <= t1)
        p, g = z["p_body"][m], ps(z["t"][m]).T
        R, t = fit4(p, g)
        ate[r] = float(np.sqrt(np.mean(np.sum(((R @ p.T).T + t - g) ** 2, 1))))
        assert abs(ate[r] - CSV_ATE_C[r]) < 5e-4, (r, ate[r], CSV_ATE_C[r])
        fused[r], gtp[r] = p, g
    R, t = fit4(np.concatenate([fused[r] for r in ROBOTS]),
                np.concatenate([gtp[r] for r in ROBOTS]))

    f, ax = plt.subplots(1, 2, figsize=(10, 5.3))
    for r in ROBOTS:
        v = D.load_vins(SEQ, r, stride=1)
        ax[0].plot(v["x"], v["y"], color=COL[r], lw=1.4, label=NAME[r])
        ps, _, (t0, t1), _, _ = D._mocap_splines(SEQ, r)
        tt = np.arange(t0, t1, 0.05)
        g = ps(tt).T
        ax[1].plot(g[:, 0], g[:, 1], "--", color="k", lw=1.0, dashes=(4, 3))
        q = (R @ fused[r].T).T + t
        ax[1].plot(q[:, 0], q[:, 1], color=COL[r], lw=1.4)
    ax[0].set_title("(a) VIO 단독 — 각자의 좌표계", fontsize=15)
    ax[1].set_title("(b) UWB 협업 정합 (앵커 없음)", fontsize=15)
    for a in ax:
        a.set_aspect("equal", adjustable="datalim")
        a.set_xlabel("x [m]")
        a.grid(color="#dddddd", lw=0.6)
        a.tick_params(labelsize=14)
        for s in a.spines.values():
            s.set_color("#444444")
    ax[0].set_ylabel("y [m]")
    handles = [Line2D([], [], color=COL[r], lw=2.2, label=NAME[r]) for r in ROBOTS]
    handles.append(Line2D([], [], color="k", ls="--", dashes=(4, 3), lw=1.2, label="GT (mocap)"))
    f.legend(handles=handles, loc="lower center", ncol=4, fontsize=14, frameon=False,
             handlelength=1.8, columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))
    ax[1].set_xlabel("x [m]\nATE: " + " · ".join("%s %.2f m" % (NAME[r], ate[r]) for r in ROBOTS),
                     fontsize=14)
    f.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    print("ATE (4-DoF per drone):", {r: round(ate[r], 4) for r in ROBOTS}, "-> %s" % OUT)


if __name__ == "__main__":
    main()

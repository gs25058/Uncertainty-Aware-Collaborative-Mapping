#!/usr/bin/env python3
"""Root-cause diagnosis for the UWB-VO fusion regression on MILUV.

Two decisive tests:
  Test 1 (H1: evaluation-protocol mismatch) -- score VO and fused under the SAME
          alignment (both SE3 and both Sim3), completing a 2x2 ATE table.
  Test 2 (H2: range observation-model mismatch) -- inject gt_range and compare;
          plus data-only auxiliary diagnostics (lever-arm size, per-anchor bias).

All results are appended (flushed immediately) to diagnosis_results.csv.
"""
import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor.fusion import CoVOR, Cfg, umeyama_sim3
from covor import data as D
from covor import factors as F

RESCSV = "/src/gs25058/cr_RNE/covor_slam/diagnosis_results.csv"
SEQ = sys.argv[1] if len(sys.argv) > 1 else "default_3_zigzag_0"

# previous best fused config from sweep_results.csv (g17): const bias, floor 0.4,
# huber 1.0, gauge prior every 8 KFs; all range factors on.
BEST = dict(bias_mode="const", range_sigma_floor=0.4, huber_k=1.0, prior_every=8)


# ---- one shared alignment function, used identically for VO and fused --------
def align(est_xyz, gt_xyz, with_scale):
    """Umeyama alignment of est onto gt. with_scale=True -> Sim3 (7dof, scale
    absorbed); False -> SE3 (rotation+translation only, metric)."""
    s, R, t = umeyama_sim3(est_xyz, gt_xyz)
    if not with_scale:
        s = 1.0
        mu_s, mu_d = est_xyz.mean(0), gt_xyz.mean(0)
        H = (est_xyz - mu_s).T @ (gt_xyz - mu_d) / len(est_xyz)
        U, _, Vt = np.linalg.svd(H)
        Sgn = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            Sgn[2, 2] = -1
        R = Vt.T @ Sgn @ U.T
        t = mu_d - R @ mu_s
    return s * (R @ est_xyz.T).T + t


def rmse(est_xyz, gt_xyz, with_scale):
    a = align(est_xyz, gt_xyz, with_scale)
    return float(np.sqrt((np.linalg.norm(a - gt_xyz, axis=1) ** 2).mean()))


# ---- results logging --------------------------------------------------------
FIELDS = ["test", "robot", "variant", "align", "z_source", "config",
          "value", "extra", "time_s"]


def log_rows(rows):
    new = not os.path.exists(RESCSV)
    with open(RESCSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()


def gt_at(mocap, ts):
    idx = [int(np.abs(mocap.timestamp.values - t).argmin()) for t in ts]
    return mocap.iloc[idx][["x", "y", "z"]].to_numpy(dtype=float)


# ============================================================================
# TEST 1 -- evaluation-alignment 2x2 table (H1)
# ============================================================================
def test1():
    t0 = time.time()
    cfg = Cfg(**BEST)
    cov = CoVOR(SEQ, cfg).build()
    cov.optimize(max_iter=100, verbose=False)
    dt = round(time.time() - t0, 1)

    print(f"\n=== TEST 1: evaluation-alignment 2x2 (config={BEST}) ===")
    print(f"{'robot':8s} {'VO/SE3':>9s} {'VO/Sim3':>9s} "
          f"{'fused/SE3':>10s} {'fused/Sim3':>11s}")
    rows = []
    for rb in cov.robots:
        if rb.n() < 5:
            continue
        ts = rb.t
        gt = gt_at(rb.mocap, ts)
        vo = np.array([p.translation() for p in rb.poses_vo])
        fu = cov.trajectory(cov.result, rb.k)[:, 1:4]
        vals = {
            ("VO", "SE3"):    rmse(vo, gt, False),
            ("VO", "Sim3"):   rmse(vo, gt, True),
            ("fused", "SE3"): rmse(fu, gt, False),
            ("fused", "Sim3"):rmse(fu, gt, True),
        }
        print(f"{rb.name:8s} {vals[('VO','SE3')]:9.3f} {vals[('VO','Sim3')]:9.3f} "
              f"{vals[('fused','SE3')]:10.3f} {vals[('fused','Sim3')]:11.3f}")
        for (variant, al), v in vals.items():
            rows.append(dict(test="test1_align2x2", robot=rb.name, variant=variant,
                             align=al, z_source="range", config=str(BEST),
                             value=round(v, 4), extra="", time_s=dt))
    log_rows(rows)
    print("(VO/SE3 is meaningless -- VO is up-to-scale, not metric; shown for completeness.)")
    print("Interpretation: compare fused vs VO under the SAME alignment.")
    return rows


# ============================================================================
# TEST 2 -- gt_range injection (H2). If the observation MODEL is the defect, even
# perfect (mocap-derived) ranges cannot restore robot ATE to the VO level, because
# the body-center factor is geometrically inconsistent with the true antenna
# distance by the (rotating) moment arm.  If ATE recovers, the model is fine and
# real UWB quality was the cause.
# ============================================================================
def test2():
    t0 = time.time()
    cfg = Cfg(use_gt_range=True, gt_range_sigma=0.05, bias_mode="off",
              prior_every=BEST["prior_every"])
    cov = CoVOR(SEQ, cfg).build()
    cov.optimize(max_iter=100, verbose=False)
    dt = round(time.time() - t0, 1)

    print(f"\n=== TEST 2: gt_range injection (bias off, sigma=0.05, pe=8) ===")
    print(f"{'robot':8s} {'VO/Sim3':>9s} {'fused/Sim3':>11s} {'fused/SE3':>10s} {'verdict'}")
    rows = []
    for rb in cov.robots:
        if rb.n() < 5:
            continue
        ts = rb.t
        gt = gt_at(rb.mocap, ts)
        vo = np.array([p.translation() for p in rb.poses_vo])
        fu = cov.trajectory(cov.result, rb.k)[:, 1:4]
        vo_s = rmse(vo, gt, True)
        fu_s = rmse(fu, gt, True)
        fu_e = rmse(fu, gt, False)
        verdict = "RECOVERED" if fu_s <= vo_s * 1.3 else "still degraded"
        print(f"{rb.name:8s} {vo_s:9.3f} {fu_s:11.3f} {fu_e:10.3f}  {verdict}")
        for variant, al, v in [("fused", "Sim3", fu_s), ("fused", "SE3", fu_e),
                               ("VO", "Sim3", vo_s)]:
            rows.append(dict(test="test2_gtrange", robot=rb.name, variant=variant,
                             align=al, z_source="gt_range", config=str(cfg.__dict__),
                             value=round(v, 4), extra=verdict, time_s=dt))
    log_rows(rows)
    return rows


# ============================================================================
# TEST 3a -- observation-model fix (moment arm). Range residual uses the true
# antenna position q = p + R*l (tags.yaml). Two runs:
#   (i)  gt_range + moment arm  -> should nearly perfectly track GT if the model
#        fix is correct (perfect measurement + correct model).
#   (ii) real range + moment arm -> does ATE recover toward the VO level on real,
#        biased/noisy UWB?
# ============================================================================
def _run_and_report(label, cfg, z_source):
    t0 = time.time()
    cov = CoVOR(SEQ, cfg).build()
    cov.optimize(max_iter=100, verbose=False)
    dt = round(time.time() - t0, 1)
    print(f"\n--- {label} ---")
    print(f"{'robot':8s} {'VO/Sim3':>9s} {'fused/Sim3':>11s} {'fused/SE3':>10s} {'verdict'}")
    rows = []
    for rb in cov.robots:
        if rb.n() < 5:
            continue
        gt = gt_at(rb.mocap, rb.t)
        vo = np.array([p.translation() for p in rb.poses_vo])
        fu = cov.trajectory(cov.result, rb.k)[:, 1:4]
        vo_s = rmse(vo, gt, True); fu_s = rmse(fu, gt, True); fu_e = rmse(fu, gt, False)
        verdict = "RECOVERED" if fu_s <= vo_s * 1.3 else ("improved" if fu_s < 0.42 else "still degraded")
        print(f"{rb.name:8s} {vo_s:9.3f} {fu_s:11.3f} {fu_e:10.3f}  {verdict}")
        for variant, al, v in [("fused", "Sim3", fu_s), ("fused", "SE3", fu_e),
                               ("VO", "Sim3", vo_s)]:
            rows.append(dict(test=label, robot=rb.name, variant=variant, align=al,
                             z_source=z_source, config=str(cfg.__dict__),
                             value=round(v, 4), extra=verdict, time_s=dt))
    log_rows(rows)
    return rows


def test3():
    print("\n=== TEST 3a: observation-model fix (moment arm) ===")
    # (i) perfect measurement + corrected model
    _run_and_report("test3a_gtrange_ma",
                    Cfg(use_gt_range=True, gt_range_sigma=0.05, bias_mode="off",
                        use_moment_arm=True, prior_every=BEST["prior_every"]),
                    "gt_range")
    # (ii) real UWB + corrected model (best prior config, const bias)
    _run_and_report("test3a_realrange_ma",
                    Cfg(use_moment_arm=True, bias_mode="const",
                        range_sigma_floor=BEST["range_sigma_floor"],
                        huber_k=BEST["huber_k"], prior_every=BEST["prior_every"]),
                    "range")


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "test1"
    if which == "test1":
        test1()
    elif which == "test2":
        test2()
    elif which == "test3":
        test3()

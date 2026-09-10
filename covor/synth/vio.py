"""Synthetic VIO front-end (Part B-3).

Produces the file ``covor.data.load_vins`` reads: a RAW LOCAL-FRAME ``vio.csv``
per robot, each in its OWN gravity-aligned frame with its own yaw and position
offset. Appendix A-2: a mocap-aligned trajectory must never reach the fusion,
because ground truth would then silently do the UWB's job and the result would
look better.

THE ERROR PROCESS, and why it is not just "accumulate N(0, sigma_odo)".
Accumulating white relative-rotation noise at the appendix-B value gives a yaw
random walk of sigma_odo_rot * sqrt(N): over 90 s of 10 Hz nodes that is
0.0124*sqrt(900) = 21 deg. The real VINS front-end drifts -4.5 .. +0.2 deg/min
(RESULTS_SUMMARY §10), i.e. about 6 deg over the same 90 s. A pure random walk
would therefore hand the UWB fusion a front-end three times worse than the one
it actually helps -- inflating the measured benefit of our own method. So the
rotation error is split into the two pieces the real system has, with the split
fixed by measured numbers and not by any result:

  yaw        random walk, step sigma_walk, calibrated to the MEASURED VINS drift
             rate (cfg.yaw_drift_deg_per_min, per robot). Yaw is the one DoF
             gravity does not observe, so it is the one that drifts.
  roll/pitch Ornstein-Uhlenbeck, stationary sigma = Cfg.sigma_tilt (0.0114 rad,
             measured), correlation time cfg.tilt_tau_s. Bounded, because
             gravity observes it -- which is exactly the assumption
             factors.gravity_prior encodes.
  white      per-node body-frame rotation noise sigma_delta, sized so the TOTAL
             node-to-node relative rotation residual equals Cfg.sigma_odo_rot.
  position   accumulated white body-frame relative-translation noise at
             Cfg.sigma_odo_trans per node -- a genuine random walk, kept as-is
             because its drift (0.0041*sqrt(900) = 0.12 m over 90 s) is already
             the right order.

``generate`` returns the achieved statistics (relative residual per axis, yaw
drift rate, raw-VIO ATE) so the model is CHECKED against its targets in the
report instead of asserted.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from covor.fusion import Cfg


def _ou(n, dt, tau, sigma, rng):
    """Ornstein-Uhlenbeck samples with stationary std ``sigma``."""
    a = np.exp(-dt / tau)
    s = sigma * np.sqrt(1 - a ** 2)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = a * x[i - 1] + rng.normal(0, s)
    return x


def _yaw_of(Rm):
    return np.arctan2(Rm[..., 1, 0], Rm[..., 0, 0])


def generate(t, p_gt, R_gt, cfg, k, fusion_cfg=None, rng=None):
    """GT body trajectory -> noisy local-frame VIO trajectory.

    Returns dict(t, p, R, v, T_G_L (4x4), stats).
    """
    fc = fusion_cfg or Cfg()
    rng = rng or np.random.default_rng(cfg.seed * 100 + k)
    n = len(t)
    dt = float(np.median(np.diff(t)))
    stride = fc.vins_stride                      # raw samples per graph node

    # --- the robot's own local frame: origin and yaw of its first GT pose ----
    psi0 = float(_yaw_of(R_gt[0]))
    Rz0 = R.from_euler("z", -psi0).as_matrix()
    p_loc = (p_gt - p_gt[0]) @ Rz0.T
    R_loc = Rz0 @ R_gt                            # gravity-aligned, yaw zeroed
    T_G_L = np.eye(4)
    T_G_L[:3, :3] = R.from_euler("z", psi0).as_matrix()
    T_G_L[:3, 3] = p_gt[0]

    # --- error process ------------------------------------------------------
    # Yaw: a systematic drift at the measured per-robot rate, plus a random walk
    # a third that size. (The real front-end's yaw error is dominated by a rate,
    # not by a walk -- §10 reports it in deg/min.)
    dur_min = (t[-1] - t[0]) / 60.0
    target = np.radians(cfg.yaw_drift_deg_per_min[k]) * dur_min
    sig_walk = 0.3 * abs(target) / np.sqrt(max(n - 1, 1))
    psi_e = (target * (t - t[0]) / max(t[-1] - t[0], 1e-9)
             + np.concatenate([[0.0], np.cumsum(rng.normal(0, sig_walk, n - 1))]))

    tilt = np.stack([_ou(n, dt, cfg.tilt_tau_s, fc.sigma_tilt, rng),
                     _ou(n, dt, cfg.tilt_tau_s, fc.sigma_tilt, rng)], 1)
    Rw = R.from_euler("z", psi_e).as_matrix()
    Rt = R.from_euler("xy", tilt).as_matrix()

    # position: every GT increment is rotated by the orientation error live at
    # that instant (that is how a drifting front-end accumulates), plus white
    # per-node translation noise scaled to the raw rate.
    sig_tr = fc.sigma_odo_trans / np.sqrt(stride)
    dp = np.diff(p_loc, axis=0)
    dp_n = np.einsum("nij,nj->ni", (Rw @ Rt)[1:], dp) + rng.normal(0, sig_tr, (n - 1, 3))
    p_vio = np.concatenate([np.zeros((1, 3)), np.cumsum(dp_n, axis=0)])

    # White body-frame rotation noise, sized by MEASUREMENT rather than by an
    # analytic guess: the node-to-node relative-rotation variance is
    # (what the drift and the OU tilt already produce) + 2*sigma_delta^2, so
    # measure the first term with delta = 0 and solve for the second.
    R0 = Rw @ Rt @ R_loc
    v0 = _stats(t, p_loc, R_loc, p_vio, R0, stride, fc, 0.0, sig_walk)["rel_rot_rms"]
    sig_delta = np.sqrt(max(fc.sigma_odo_rot ** 2 - v0 ** 2, 0.0) / 2.0)
    delta = rng.normal(0, sig_delta, (n, 3))
    R_vio = R0 @ R.from_rotvec(delta).as_matrix()

    v = np.zeros((n, 3))
    v[1:] = np.diff(p_vio, axis=0) / dt
    v[0] = v[1]

    stats = _stats(t, p_loc, R_loc, p_vio, R_vio, stride, fc, sig_delta, sig_walk)
    return dict(t=t, p=p_vio, R=R_vio, v=v, T_G_L=T_G_L, stats=stats)


def _stats(t, p_gt, R_gt, p_vio, R_vio, stride, fc, sig_delta, sig_walk):
    """What the process ACHIEVED, against what the graph will assume."""
    a, b = slice(0, -stride, stride), slice(stride, None, stride)
    dR_gt = np.einsum("nji,njk->nik", R_gt[a], R_gt[b])
    dR_vi = np.einsum("nji,njk->nik", R_vio[a], R_vio[b])
    err = R.from_matrix(np.einsum("nji,njk->nik", dR_gt, dR_vi)).as_rotvec()
    dp_gt = np.einsum("nji,nj->ni", R_gt[a], p_gt[b] - p_gt[a])
    dp_vi = np.einsum("nji,nj->ni", R_vio[a], p_vio[b] - p_vio[a])
    dperr = dp_vi - dp_gt
    yaw_err = np.unwrap(_yaw_of(R_vio) - _yaw_of(R_gt))
    dur_min = (t[-1] - t[0]) / 60.0
    # raw-VIO ATE after the 4-DoF alignment evaluation uses
    from covor.fusion import umeyama_yaw
    Ra, ta = umeyama_yaw(p_vio, p_gt)
    ate = np.linalg.norm((p_vio @ Ra.T + ta) - p_gt, axis=1)
    return dict(
        rel_rot_rms=float(np.sqrt((err ** 2).sum(1).mean() / 3)),
        rel_rot_target=float(fc.sigma_odo_rot),
        rel_trans_rms=float(np.sqrt((dperr ** 2).sum(1).mean() / 3)),
        rel_trans_target=float(fc.sigma_odo_trans),
        tilt_rms_rad=float(np.sqrt((err[:, :2] ** 2).mean())),
        yaw_drift_deg_per_min=float(np.degrees(yaw_err[-1] - yaw_err[0]) / dur_min),
        yaw_err_final_deg=float(np.degrees(yaw_err[-1])),
        raw_vio_ate_rmse=float(np.sqrt((ate ** 2).mean())),
        raw_vio_ate_max=float(ate.max()),
        sigma_delta=float(sig_delta), sigma_yaw_walk=float(sig_walk))


def to_vins_csv(path, t, p, R_mat, v, timeshift):
    """Write VINS-Fusion's own visualization.cpp column order.

        t(ns), x, y, z, qw, qx, qy, qz, vx, vy, vz     <- quaternion W FIRST

    The timestamp is absolute nanoseconds, so ``load_vins`` recovers relative
    seconds only if it applies timeshift_s + timeshift_ns/1e9 (appendix A-8).
    A zero offset here would hide that bug class instead of exercising it.
    """
    import os
    q = R.from_matrix(R_mat).as_quat()                # xyzw
    ns = np.rint((np.asarray(t, float) + timeshift) * 1e9).astype(np.int64)
    M = np.column_stack([ns, p, q[:, 3], q[:, 0], q[:, 1], q[:, 2], v])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in M:
            f.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n"
                    % (int(r[0]), *r[1:]))


def to_mocap_csv(path, t, p, R_mat):
    """MILUV mocap.csv: the GT body pose in the world frame.

    Only ``fusion.Robot.align_to_world`` consumes it (the initial guess), and
    it is passed through MILUV's spline cleaner on the way in. The EXACT GT is
    kept separately in gt_traj_<robot>.npz -- every metric uses that, never this.
    """
    import os
    import pandas as pd
    q = R.from_matrix(R_mat).as_quat()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "timestamp": t,
        "pose.position.x": p[:, 0], "pose.position.y": p[:, 1],
        "pose.position.z": p[:, 2],
        "pose.orientation.x": q[:, 0], "pose.orientation.y": q[:, 1],
        "pose.orientation.z": q[:, 2], "pose.orientation.w": q[:, 3],
    }).to_csv(path, index=False)

"""Evaluation and plotting for CoVOR-SLAM: ATE RMSE vs mocap ground truth, and
trajectory / error-over-time figures in the style of the paper (Figs 9, 10, 15)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data as D
from .fusion import umeyama_sim3


def mocap_at(mocap, ts):
    idx = [int(np.abs(mocap.timestamp.values - t).argmin()) for t in ts]
    return mocap.iloc[idx][["x", "y", "z"]].to_numpy(dtype=float)


def align(est_xyz, gt_xyz, with_scale):
    """Align est to gt (Umeyama). with_scale=True for up-to-scale mono baseline."""
    if with_scale:
        s, R, t = umeyama_sim3(est_xyz, gt_xyz)
    else:
        s, R, t = umeyama_sim3(est_xyz, gt_xyz)
        s = 1.0
        # recompute R,t with fixed unit scale
        mu_s, mu_d = est_xyz.mean(0), gt_xyz.mean(0)
        H = (est_xyz - mu_s).T @ (gt_xyz - mu_d) / len(est_xyz)
        U, _, Vt = np.linalg.svd(H)
        Sgn = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            Sgn[2, 2] = -1
        R = Vt.T @ Sgn @ U.T
        t = mu_d - R @ mu_s
    return (s * (R @ est_xyz.T).T + t)


def ate_rmse(traj, mocap, with_scale):
    """traj: (N,4) [t,x,y,z]. Returns rmse [m], aligned est (N,3), gt (N,3)."""
    ts = traj[:, 0]
    est = traj[:, 1:4]
    gt = mocap_at(mocap, ts)
    est_a = align(est, gt, with_scale)
    err = np.linalg.norm(est_a - gt, axis=1)
    return float(np.sqrt((err ** 2).mean())), est_a, gt, ts


def plot_trajectories(seq, robots, vo_only, fused, mocaps, out):
    fig, axs = plt.subplots(1, len(robots), figsize=(5 * len(robots), 4.5))
    if len(robots) == 1:
        axs = [axs]
    for ax, name, vo, fu, mc in zip(axs, robots, vo_only, fused, mocaps):
        if mc is not None:
            ax.plot(mc[:, 0], mc[:, 1], color="0.5", lw=2, label="Ground truth")
        if vo is not None:
            ax.plot(vo[:, 0], vo[:, 1], color="tab:orange", lw=1.2, ls="--",
                    label="Mono-VO (Sim3-aligned)")
        if fu is not None:
            ax.plot(fu[:, 0], fu[:, 1], color="tab:red", lw=1.5, label="CoVOR-SLAM")
        ax.set_title(name); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"MILUV {seq}: horizontal trajectories")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_errors(seq, robots, vo_err, fused_err, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, ve, fe in zip(robots, vo_err, fused_err):
        if ve is not None:
            ax.plot(ve[0], ve[1], ls="--", alpha=0.7, label=f"{name} mono-VO")
        if fe is not None:
            ax.plot(fe[0], fe[1], label=f"{name} CoVOR")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position error [m]")
    ax.set_title(f"MILUV {seq}: position error vs ground truth")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

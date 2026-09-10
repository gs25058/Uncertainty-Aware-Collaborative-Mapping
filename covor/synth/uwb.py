"""Synthetic UWB inter-agent ranges (Part B-4).

Writes MILUV's own ``uwb_range.csv`` schema so ``covor.data.load_ranges`` reads
it unchanged: ``timestamp, from_id, to_id, range, std, gt_range, bias``, tag ids
following MILUV's ``id // 10 -> ifo00k`` rule, and every inter-agent row written
into BOTH endpoints' files exactly as MILUV records them -- so the loader's
de-duplication on (timestamp, from_id, to_id) is exercised rather than bypassed.

No anchor rows are generated. The proposal is anchor-free (§1, §2.4) and Part D
runs with ``anchor_robots=()``; a to_id < 10 row would be silently ignored there
anyway, and generating one would only invite it back in by accident.

Observation model, all three terms measured on MILUV rather than chosen:
    range = |q_a - q_b| + bias + N(0, sigma^2) [+ NLOS]
    q = p_body + R_body @ moment_arm     (config/uwb/tags.yaml, appendix B
                                          use_moment_arm=True)
    bias  = +0.004 m   (MILUV inter-agent bias is +0.0035)
    sigma =  0.05 m    (= Cfg.range_sigma_floor)
The ``std`` column is written as the TRUE sigma. On real MILUV that column is
optimistic (appendix A-7: mean 0.17 vs actual 0.22); here the simulator knows
the truth, so the graph's ``max(std, range_sigma_floor)`` gets the right number
and any residual mis-weighting cannot be blamed on the csv.
"""
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

from covor import data as D
from .config import TAGS


def _interp_pose(t_src, p_src, R_src, t_q):
    p = np.stack([np.interp(t_q, t_src, p_src[:, i]) for i in range(3)], 1)
    sl = Slerp(t_src, R.from_matrix(R_src))
    return p, sl(np.clip(t_q, t_src[0], t_src[-1])).as_matrix()


def antenna(p, Rb, arm, use_arm):
    return p + (Rb @ arm if use_arm else 0.0)


def generate(gt, cfg, scene=None, rng=None):
    """gt: {robot: dict(t, p, R)} GT body trajectories -> one DataFrame of rows.

    ``scene`` is only needed when cfg.uwb_nlos is on (line-of-sight test against
    the room mesh).
    """
    rng = rng or np.random.default_rng(cfg.seed * 977 + 13)
    arms = D.load_tag_arms()
    robots = list(gt)
    rows = []
    for ia in range(len(robots)):
        for ib in range(ia + 1, len(robots)):
            A, B = robots[ia], robots[ib]
            t0 = max(gt[A]["t"][0], gt[B]["t"][0])
            t1 = min(gt[A]["t"][-1], gt[B]["t"][-1])
            for ta in TAGS[A]:
                for tb in TAGS[B]:
                    n = int((t1 - t0) * cfg.uwb_rate_hz)
                    if n < 2:
                        continue
                    # even spacing with a random phase per tag pair, as the four
                    # tag pairs of a robot pair are independent radios
                    ts = t0 + (np.arange(n) + rng.random()) / cfg.uwb_rate_hz
                    ts = ts[(ts >= t0) & (ts <= t1)]
                    pa, Ra = _interp_pose(gt[A]["t"], gt[A]["p"], gt[A]["R"], ts)
                    pb, Rb = _interp_pose(gt[B]["t"], gt[B]["p"], gt[B]["R"], ts)
                    qa = pa + (np.einsum("nij,j->ni", Ra, arms[ta])
                               if cfg.use_moment_arm else 0.0)
                    qb = pb + (np.einsum("nij,j->ni", Rb, arms[tb])
                               if cfg.use_moment_arm else 0.0)
                    gtr = np.linalg.norm(qb - qa, axis=1)
                    bias = np.full(len(ts), cfg.uwb_bias)
                    if cfg.uwb_nlos:
                        blocked = _blocked(scene, qa, qb)
                        bias = bias + blocked * (
                            cfg.uwb_nlos_bias +
                            np.abs(rng.normal(0, cfg.uwb_nlos_sigma, len(ts))))
                    z = gtr + bias + rng.normal(0, cfg.uwb_sigma, len(ts))
                    rows.append(pd.DataFrame(dict(
                        timestamp=ts, from_id=ta, to_id=tb, range=z,
                        std=cfg.uwb_sigma, gt_range=gtr, bias=bias,
                        _a=A, _b=B)))
    df = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return df.reset_index(drop=True)


def _blocked(scene, qa, qb):
    """True where the mesh intersects the segment qa->qb (NLOS)."""
    import open3d as o3d
    d = qb - qa
    L = np.linalg.norm(d, axis=1)
    rays = np.concatenate([qa, d], 1).astype(np.float32)   # t in units of |d|
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.isfinite(t) & (t < 1.0 - 1e-3)


def write(df, cfg):
    """One csv per robot, each holding the rows in which it is an endpoint --
    MILUV's layout, duplicates included."""
    out = {}
    for rob in df["_a"].unique().tolist() + df["_b"].unique().tolist():
        m = (df["_a"] == rob) | (df["_b"] == rob)
        d = df.loc[m, ["timestamp", "from_id", "to_id", "range", "std",
                       "gt_range", "bias"]]
        p = os.path.join(cfg.dataroot(), cfg.seq, rob, "uwb_range.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        d.to_csv(p, index=False)
        out[rob] = (p, len(d))
    return out

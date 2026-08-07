"""MILUV data loaders for CoVOR-SLAM.

All timestamps within a sequence share one relative-seconds clock (mocap, UWB and
camera frames), so no cross-clock alignment is needed. UWB ``to_id < 10`` denotes
an anchor, ``>= 10`` a robot tag; tag ``id // 10`` maps to ifo001/2/3.
"""
import os
import glob
import numpy as np
import pandas as pd
import yaml

DATA = "/src/gs25058/cr_RNE/miluv/data"
UWB_CFG = "/src/gs25058/cr_RNE/miluv/config/uwb"
HEIGHT_CFG = "/src/gs25058/cr_RNE/miluv/config/height"
VO_DIR = "/src/gs25058/cr_RNE/covor_slam/vo_output"

# VINS-Fusion stereo+IMU front-end output (OUTSIDE this repo, git-untracked).
VINS_DIR = "/src/gs25058/ws/vins_ws/run/out"
# Camera/IMU extrinsics come from MILUV's tracked per-robot config (see
# load_body_T_cam), not from the git-untracked per-sequence run copies.
VINS_CFG = "/src/gs25058/cr_RNE/miluv/config/vins"

# Graph node density. VINS publishes at 14.98 Hz; we keep every VINS_STRIDE-th
# pose -> 7.49 Hz. Chosen where two independent criteria meet (measured, not tuned):
#   (1) Independence: the odometry residual RMS follows the white-noise sqrt(s) law
#       up to s=2 (1.42/1.44/1.56 vs sqrt(2)=1.41) and breaks super-sqrt(s) beyond,
#       so consecutive relative poses are still effectively uncorrelated here.
#   (2) Association: half the node spacing is 0.067 s, over which the p95 robot
#       speed (0.804 m/s) moves 0.054 m ~= the UWB noise floor (0.05 m). Sparser
#       nodes would let the timing error dominate the range measurement itself.
VINS_STRIDE = 2

ROBOTS = ["ifo001", "ifo002", "ifo003"]


def robot_of_tag(tag_id: int) -> str | None:
    """Map a UWB tag id to a robot ('ifo00k'); None if it is an anchor."""
    if tag_id < 10:
        return None
    return f"ifo00{tag_id // 10}"


def _mocap_splines(seq: str, robot: str):
    """Position/quaternion smoothing splines over CLEANED mocap.

    Port of MILUV's own ``miluv/utils.py:get_mocap_splines`` (same gap rule, same
    outlier rule, same csaps smooth=0.9999), reimplemented here only to avoid
    pulling in that module's unrelated ``pymlg`` dependency. Keeping the rules
    identical matters: raw mocap is NOT clean, and every metric in this project is
    measured against it.

    What it removes:
      * gaps   -- all-zero position or quaternion rows (tracker dropout);
      * outliers -- any sample whose rotation differs from the last good one by
        more than 1 rad; BOTH that sample and its predecessor are dropped.
    Raw mocap on this sequence carries 173-180 deg rotation jumps between samples
    0.13 s apart in ~0.5 % of ifo001/ifo002 rows, and position jumps up to 1.08 m.
    Sampling it nearest-in-time (what this file used to do) feeds those straight
    into ATE, NEES and the GT-pose control.
    """
    from csaps import csaps
    from scipy.spatial.transform import Rotation
    df = pd.read_csv(os.path.join(DATA, seq, robot, "mocap.csv"))
    t = df["timestamp"].values
    pos = df[["pose.position.x", "pose.position.y", "pose.position.z"]].values
    quat = df[["pose.orientation.x", "pose.orientation.y",
               "pose.orientation.z", "pose.orientation.w"]].values

    gaps = (np.linalg.norm(pos, axis=1) < 1e-6) | (np.linalg.norm(quat, axis=1) < 1e-6)
    t, pos, quat = t[~gaps], pos[~gaps], quat[~gaps]

    outliers = np.zeros(len(t), dtype=bool)
    last_good = Rotation.from_quat(quat[0]).as_matrix()
    for i in range(1, len(quat)):
        R_now = Rotation.from_quat(quat[i]).as_matrix()
        if Rotation.from_matrix(last_good.T @ R_now).magnitude() > 1.0:
            outliers[i - 1] = True
            outliers[i] = True
        else:
            last_good = R_now
    t, pos, quat = t[~outliers], pos[~outliers], quat[~outliers]

    quat = quat / np.linalg.norm(quat, axis=1)[:, None]
    for i in range(1, len(quat)):            # keep the quaternion path continuous
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1

    n_drop = int(gaps.sum() + outliers.sum())
    return (csaps(t, pos.T, smooth=0.9999).spline,
            csaps(t, quat.T, smooth=0.9999).spline,
            (t[0], t[-1]), n_drop, len(df))


_MOCAP_CACHE = {}


def load_mocap(seq: str, robot: str, clean: bool = True) -> pd.DataFrame:
    """Ground-truth body pose in the mocap/world frame G.

    clean=True (default) returns the spline-smoothed, outlier-rejected track
    resampled on the original timestamps -- the MILUV convention. clean=False
    returns the raw csv, kept only for before/after comparisons.
    """
    raw = pd.read_csv(os.path.join(DATA, seq, robot, "mocap.csv"))
    ren = {"pose.position.x": "x", "pose.position.y": "y", "pose.position.z": "z",
           "pose.orientation.x": "qx", "pose.orientation.y": "qy",
           "pose.orientation.z": "qz", "pose.orientation.w": "qw"}
    if not clean:
        return raw.rename(columns=ren)
    key = (seq, robot)
    if key not in _MOCAP_CACHE:
        _MOCAP_CACHE[key] = _mocap_splines(seq, robot)
    ps, qs, (t0, t1), _, _ = _MOCAP_CACHE[key]
    t = np.clip(raw["timestamp"].values, t0, t1)
    p = np.asarray(ps(t)).T
    q = np.asarray(qs(t)).T
    q = q / np.linalg.norm(q, axis=1)[:, None]     # splines break unit norm
    return pd.DataFrame({"timestamp": t, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2],
                         "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3]})


def mocap_pose_at(seq: str, robot: str, ts) -> tuple:
    """(positions (N,3), quaternions xyzw (N,4)) evaluated ON the spline at ts.

    Preferred over nearest-neighbour lookup: it is continuous, and it cannot land
    on a dropped sample.
    """
    key = (seq, robot)
    if key not in _MOCAP_CACHE:
        _MOCAP_CACHE[key] = _mocap_splines(seq, robot)
    ps, qs, (t0, t1), _, _ = _MOCAP_CACHE[key]
    ts = np.clip(np.asarray(ts, dtype=float), t0, t1)
    p = np.asarray(ps(ts)).T
    q = np.asarray(qs(ts)).T
    return p, q / np.linalg.norm(q, axis=1)[:, None]


def mocap_position_at(mocap: pd.DataFrame, t: float) -> np.ndarray:
    """Nearest-in-time mocap position (x,y,z) at time t, from an already-loaded
    (cleaned) frame. Prefer mocap_pose_at when the sequence/robot are to hand."""
    i = int(np.abs(mocap.timestamp.values - t).argmin())
    return mocap.iloc[i][["x", "y", "z"]].to_numpy(dtype=float)


def load_ranges(seq: str, max_std: float = 0.5) -> pd.DataFrame:
    """Canonical de-duplicated UWB range set for the whole swarm.

    IMPORTANT: each robot's own ``uwb_range.csv`` holds *its own* anchor ranges
    (from_id = that robot's tags -> to_id < 10). ifo001's csv contains ifo001's
    anchor ranges plus the inter-agent ranges, but NOT ifo002/ifo003's anchor
    ranges -- so reading only ifo001 (as this code used to) drops 2/3 of the
    anchor measurements and leaves only robot 1 anchored. We therefore union all
    robots' csvs and de-duplicate on (timestamp, from_id, to_id): inter-agent
    ranges recorded by both endpoints collapse to one row, and every robot keeps
    its anchor ranges. Adds a ``kind`` column and drops very noisy rows.
    """
    cols = ["timestamp", "from_id", "to_id", "range", "std", "gt_range", "bias"]
    frames = []
    for robot in ROBOTS:
        path = os.path.join(DATA, seq, robot, "uwb_range.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path)[cols])
    if not frames:
        raise FileNotFoundError(f"no uwb_range.csv found under {os.path.join(DATA, seq)}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp", "from_id", "to_id"])
    df = df[df["std"] <= max_std]
    df["kind"] = np.where(df["to_id"] < 10, "anchor", "inter")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_anchors(seq: str) -> dict[int, np.ndarray]:
    """Anchor positions [x,y,z] in the world frame for this sequence's constellation.

    The constellation key is the trailing token of the sequence name
    (default_3_zigzag_0 -> '0', obstacles_1_random3_0b -> '0b').
    """
    key = seq.rsplit("_", 1)[-1]
    cfg = yaml.safe_load(open(os.path.join(UWB_CFG, "anchors.yaml")))
    const = cfg[key]
    out = {}
    for aid, vec in const.items():
        out[int(aid)] = np.array(
            [float(v) for v in vec.strip("[]").replace(",", " ").split()], dtype=float)
    return out


def load_tag_arms() -> dict[int, np.ndarray]:
    """UWB tag moment arms in the robot body frame (antenna offsets)."""
    cfg = yaml.safe_load(open(os.path.join(UWB_CFG, "tags.yaml")))
    out = {}
    for robot, tags in cfg.items():
        for tid, vec in tags.items():
            out[int(tid)] = np.array(
                [float(v) for v in vec.strip("[]").replace(",", " ").split()], dtype=float)
    return out


def _height_bias(robot: str) -> float:
    """Ground-to-mocap-origin offset (m) to subtract from the laser height so it
    reads in the mocap world frame (config/height/bias.yaml)."""
    cfg = yaml.safe_load(open(os.path.join(HEIGHT_CFG, "bias.yaml")))
    v = cfg[robot]
    if isinstance(v, dict):
        v = list(v.keys())[0]
    if isinstance(v, (list, tuple)):
        v = v[0]
    return float(v)


def load_height(seq: str, robot: str) -> pd.DataFrame | None:
    """Downward laser altimeter height in the mocap world frame (bias removed).

    Columns: ``timestamp``, ``h`` (metric z, ground offset already subtracted).
    Returns None if the height csv is missing.
    """
    path = os.path.join(DATA, seq, robot, "height.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={"range": "h"})
    df["h"] = df["h"] - _height_bias(robot)
    return df[["timestamp", "h"]].sort_values("timestamp").reset_index(drop=True)


def height_at(height: pd.DataFrame, t: float, tol: float = 0.1):
    """Nearest-in-time bias-corrected height at time t, or None outside tol."""
    i = int(np.abs(height.timestamp.values - t).argmin())
    if abs(height.timestamp.values[i] - t) > tol:
        return None
    return float(height.h.values[i])


def load_vo(seq: str, robot: str) -> pd.DataFrame | None:
    """ORB-SLAM3 keyframe trajectory (TUM): t, position and quaternion in the
    robot's own up-to-scale VO frame Lk. Returns None if VO output is missing.

    LEGACY: the mono front-end is up-to-scale, so this needs the Sim(3) scale
    variables that were removed in the SE(3) transition. Kept as a reader for the
    preserved comparison group; the Sim(3) graph itself lives at commit a0a5e07.
    """
    path = os.path.join(VO_DIR, f"{seq}_{robot}_kf.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
    return df.sort_values("t").reset_index(drop=True)


def load_body_T_cam(robot: str, cam: int = 0) -> np.ndarray:
    """4x4 body(IMU) <- camera extrinsic for a robot.

    VINS state (and hence every fused pose) is the BODY pose T_wb, but the
    occupancy stage casts rays from the CAMERA: it needs T_wc = T_wb @ body_T_cam.
    Skipping this is not a small error -- body_T_cam0 is a 119-121 deg rotation
    (the optical z axis lies roughly along the body x axis) plus a ~0.11 m offset,
    so using the body pose as a camera pose points every depth ray the wrong way.

    Read from MILUV's own per-robot config, not from the per-sequence copies under
    ws/vins_ws/run/cfg: the extrinsic is a physical property of the airframe, the
    same in every sequence, and the MILUV tree is the tracked original. (The run/cfg
    copies exist only to redirect output_path, and are incomplete -- there is no
    default_3_zigzag_0/ifo001 directory.)

    Parsed by hand: these are OpenCV FileStorage files ("%YAML:1.0" plus
    "!!opencv-matrix" tags), which PyYAML's safe_load will not read.
    """
    path = os.path.join(VINS_CFG, robot, "vins.yaml")
    key = f"body_T_cam{cam}"
    txt = open(path).read()
    if key not in txt:
        raise KeyError(f"{key} not found in {path}")
    body = txt.split(key, 1)[1]
    body = body.split("data:", 1)[1]
    body = body[body.index("[") + 1:body.index("]")]
    vals = [float(v) for v in body.replace("\n", " ").split(",") if v.strip()]
    if len(vals) != 16:
        raise ValueError(f"{key} in {path} has {len(vals)} entries, expected 16")
    return np.array(vals, dtype=float).reshape(4, 4)


def timeshift(seq: str) -> float:
    """Absolute epoch offset (s) of this sequence's relative-seconds clock.

    MILUV's own convention (miluv/utils.py:316) is seconds PLUS nanoseconds:
    dropping the ns term leaves a silent systematic offset of 0.256 s on
    default_3_zigzag_0 and 0.067 s on obstacles_1_random3_0b -- five times the
    0.05 s range-association tolerance. Verified: with this offset the VINS pose
    timestamps land bit-exactly (max |dt| = 0.000000 s) on the MILUV stereo image
    filename times, on all three trajectories.
    """
    ts = yaml.safe_load(open(os.path.join(DATA, seq, "timeshift.yaml")))
    return float(ts["timeshift_s"]) + float(ts["timeshift_ns"]) / 1e9


def load_vins(seq: str, robot: str, stride: int = VINS_STRIDE) -> pd.DataFrame | None:
    """VINS-Fusion VIO trajectory, downsampled to the graph node rate.

    Reads the raw local-frame ``vio.csv`` (never a mocap-aligned variant -- that
    would let ground truth do the UWB's job and fail silently, looking *better*).

    Column order is fixed by VINS's visualization.cpp:
        t(ns), x, y, z, qw, qx, qy, qz, vx, vy, vz    <- quaternion is W-FIRST.

    Returns t (relative seconds, MILUV clock), position + quaternion in XYZW order
    to match load_vo, and ``v`` = body speed magnitude.

    FRAME: these are body(IMU) poses T_wb in VINS's own gravity-aligned init frame
    -- NOT camera poses. The UWB tag moment arms (tags.yaml) and the mocap markers
    are body-referenced too, so the graph state stays in body frame; the camera
    extrinsic body_T_cam0 is applied only where camera rays are needed (occupancy).

    ``v`` is VINS's own velocity estimate (no ground truth), used to inflate the
    range sigma by the distance the robot moves within the association window:
    sigma_eff = sqrt(sigma_uwb^2 + (v*dt)^2).
    """
    path = os.path.join(VINS_DIR, seq, robot, "vio.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    raw = pd.read_csv(path, header=None).iloc[:, :11]
    raw.columns = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"]
    df = pd.DataFrame({
        "t": raw["t"].to_numpy(float) / 1e9 - timeshift(seq),
        "x": raw["x"], "y": raw["y"], "z": raw["z"],
        "qx": raw["qx"], "qy": raw["qy"], "qz": raw["qz"], "qw": raw["qw"],
        "v": np.linalg.norm(raw[["vx", "vy", "vz"]].to_numpy(float), axis=1),
    })
    df = df.sort_values("t").reset_index(drop=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    return df

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

ROBOTS = ["ifo001", "ifo002", "ifo003"]


def robot_of_tag(tag_id: int) -> str | None:
    """Map a UWB tag id to a robot ('ifo00k'); None if it is an anchor."""
    if tag_id < 10:
        return None
    return f"ifo00{tag_id // 10}"


def load_mocap(seq: str, robot: str) -> pd.DataFrame:
    """Ground-truth body pose in the mocap/world frame G."""
    df = pd.read_csv(os.path.join(DATA, seq, robot, "mocap.csv"))
    return df.rename(columns={
        "pose.position.x": "x", "pose.position.y": "y", "pose.position.z": "z",
        "pose.orientation.x": "qx", "pose.orientation.y": "qy",
        "pose.orientation.z": "qz", "pose.orientation.w": "qw",
    })


def mocap_position_at(mocap: pd.DataFrame, t: float) -> np.ndarray:
    """Nearest-in-time mocap position (x,y,z) at time t."""
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
    robot's own up-to-scale VO frame Lk. Returns None if VO output is missing."""
    path = os.path.join(VO_DIR, f"{seq}_{robot}_kf.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
    return df.sort_values("t").reset_index(drop=True)

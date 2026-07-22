#!/usr/bin/env python3
"""Generate ORB-SLAM3 monocular settings + a sorted image-list file for a MILUV
robot/sequence, from the dataset's per-robot intrinsics.yaml.

Usage: prep_vo.py <sequence> <robot>   e.g.  prep_vo.py default_3_zigzag_0 ifo001
"""
import os, sys, glob, yaml

DATA = "/src/gs25058/cr_RNE/miluv/data"
CALIB = "/src/gs25058/cr_RNE/miluv/config/realsense"
PROJ = "/src/gs25058/cr_RNE/covor_slam"
CAM = "infra1"   # cam0 = left IR, used as the monocular stream


def main(seq, robot):
    intr_path = os.path.join(CALIB, robot, "intrinsics.yaml")
    with open(intr_path) as f:
        intr = yaml.safe_load(f)
    cam0 = intr["cam0"]
    fx, fy, cx, cy = cam0["intrinsics"]
    k1, k2, p1, p2 = cam0["distortion_coeffs"]  # radtan
    W, H = cam0["resolution"]

    cfg = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}

Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.p1: {p1}
Camera1.p2: {p2}

Camera.width: {W}
Camera.height: {H}
Camera.fps: 30
Camera.RGB: 1

# Pure visual odometry to match CoVOR-SLAM (Lee 2023): loop closing + map merging
# + place recognition disabled, so each robot's front-end drifts as odometry and
# the UWB ranges (not visual loop closure) provide the drift correction. Local
# bundle adjustment in LocalMapping is kept (that is part of the VO front-end).
loopClosing: 0

ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    cfg_path = os.path.join(PROJ, "configs", f"miluv_{robot}.yaml")
    with open(cfg_path, "w") as f:
        f.write(cfg)

    img_dir = os.path.join(DATA, seq, robot, CAM)
    imgs = glob.glob(os.path.join(img_dir, "*.jpeg"))
    imgs.sort(key=lambda p: float(os.path.basename(p)[:-5]))
    list_path = os.path.join(PROJ, "vo_output", f"{seq}_{robot}_list.txt")
    with open(list_path, "w") as f:
        for p in imgs:
            t = float(os.path.basename(p)[:-5])
            f.write(f"{t:.9f} {p}\n")

    print(f"[{robot}] config={cfg_path}")
    print(f"[{robot}] list={list_path}  ({len(imgs)} frames)  intr fx={fx:.1f} {W}x{H}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

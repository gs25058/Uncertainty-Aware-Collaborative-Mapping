"""Shared smoke experiment I/O. Existing mapping and verdicts remain unchanged."""
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from covor.occupancy import OccCfg, OccupancyBuilder
from covor.smoke import SmokeCfg


def conditions(seed, betas=(0.15, 0.4, 0.8), fields=("uniform", "heterogeneous")):
    if len(betas)!=3 or any(not np.isfinite(b) or b<=0 for b in betas) or list(betas)!=sorted(betas):
        raise ValueError("provide three finite, positive, increasing attenuation settings")
    out = {"clean": SmokeCfg(seed=seed)}
    for field in fields:
        for strength, beta in zip(("weak", "medium", "strong"), betas):
            out[f"{field}_{strength}"] = SmokeCfg(beta=beta, field=field, seed=seed)
    return out


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()
                                    if isinstance(x, np.ndarray) else str(x)) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MAPPING_ARMS = ('uniform', 'depth_only', 'pose_depth')


def make_builder(arm='depth_only'):
    """Explicit proposal ablations; retain the archived depth-only default.

    Pose+depth is a hypothesis to test, not an assumed safer configuration.
    Every arm must be compared using identical poses, measurements and domain.
    """
    if arm not in MAPPING_ARMS:raise ValueError(f'unknown mapping arm: {arm}')
    return OccupancyBuilder(OccCfg(weighted=arm!='uniform',use_w_pose=arm=='pose_depth'))


def sparse_map(builder):
    occ, free = builder.classify_points()
    res = builder.cfg.resolution
    return tuple(np.floor(p / res + 1e-5).astype(np.int32) for p in (occ, free))


def dense_map(sparse, ijk_min, shape):
    lab = np.zeros(shape, np.uint8)
    for label, points in ((2, sparse[0]), (1, sparse[1])):
        ind = points - ijk_min
        ind = ind[((ind >= 0) & (ind < np.array(shape))).all(axis=1)]
        if len(ind):
            lab[tuple(ind.T)] = label
    return lab


def map_metrics(lab, clean):
    def iou(a, b):
        return int((a & b).sum()) / max(int((a | b).sum()), 1)
    return dict(n_occupied=int((lab == 2).sum()), n_free=int((lab == 1).sum()),
                n_unknown=int((lab == 0).sum()), domain_voxels=int(lab.size),
                occupied_iou_clean=iou(lab == 2, clean == 2),
                free_iou_clean=iou(lab == 1, clean == 1),
                known_iou_clean=iou(lab != 0, clean != 0))


def depth_metrics(z, valid, clean_z, clean_valid):
    joint = valid & clean_valid
    diff = np.abs(z[joint] - clean_z[joint])
    return dict(valid_fraction=float(valid.mean()), joint_fraction=float(joint.mean()),
                depth_absdiff_mean_m=float(diff.mean()) if len(diff) else float("nan"),
                depth_absdiff_median_m=float(np.median(diff)) if len(diff) else float("nan"),
                depth_absdiff_p90_m=float(np.percentile(diff, 90)) if len(diff) else float("nan"),
                n_joint=int(joint.sum()))


def frame_points(sd, z, sz, valid):
    # Same threshold and pixel stride as build_occupancy/DepthRenderer.frame.
    if valid.sum() < 100:
        return None, None
    return sd.backproject(z, sz, valid, downsample=4)


def save_example(out, tag, left, right, fog_left, fog_right, clean_z, fog_z):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = Path(out) / "examples"
    p.mkdir(exist_ok=True)
    for name, arr in (("clean_left", left), ("clean_right", right),
                      ("smoke_left", fog_left), ("smoke_right", fog_right)):
        cv2.imwrite(str(p / f"{tag}_{name}.png"), arr)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, arr, title in zip(axes.flat, (left, fog_left, clean_z, fog_z),
                               ("Clean IR / grayscale", "Simulated smoke",
                                "Clean SGBM Z", "SGBM re-run on smoke")):
        ax.imshow(arr, cmap="gray" if "SGBM" not in title else "viridis",
                  **({} if "SGBM" not in title else dict(vmin=.3, vmax=5)))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(p / f"{tag}.png", dpi=120)
    plt.close(fig)


def map_figure(path, maps, ijk_min, res=.1, z=.8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, len(maps), figsize=(3.6 * len(maps), 8))
    for ax, (tag, lab) in zip(np.atleast_1d(axes), maps.items()):
        k = int(np.floor(z / res)) - int(ijk_min[2])
        k = np.clip(k, 0, lab.shape[2] - 1)
        ax.imshow(lab[:, :, k].T, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["#dddddd", "#b9def0", "#c93131"]),
                  vmin=0, vmax=2, extent=(ijk_min[0]*res,
                  (ijk_min[0]+lab.shape[0])*res, ijk_min[1]*res,
                  (ijk_min[1]+lab.shape[1])*res))
        ax.set_title(tag)
        ax.set_xlabel("x [m]")
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("y [m]; gray unknown, blue free, red occupied")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

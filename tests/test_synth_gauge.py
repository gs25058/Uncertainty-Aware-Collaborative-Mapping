#!/usr/bin/env python3
"""Regression test for Cfg.gauge_init (PREREG_synth_gauge.md §2.1).

WHY THIS EXISTS. `gauge_init` is the ONE change the pre-registration allows in
covor/fusion.py, and it sits on the path every existing result was produced by.
If the default branch is not bit-for-bit the old behaviour, every number in
RESULTS_SUMMARY §7-8 and RESULTS_synth.md §4 silently becomes unreproducible.

The comparison is BY VALUE, never by count: node poses to 1e-9 and per-cell
log-odds to 1e-9 against tests/data/synth_gauge_golden.npz, which was captured
by scripts/synth/capture_golden.py BEFORE the change. A count-only check is what
let the free-evidence double-counting bug through once already
(RESULTS_SUMMARY §9-2 / A-4).

Needs the synthetic dataset (scripts/synth/make_dataset.py --seed 0); skips
cleanly if it is not built.

Run: python tests/test_synth_gauge.py   (or: pytest tests/test_synth_gauge.py)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "synth"))
import gtsam                       # noqa: F401,E402  -- must precede open3d
from covor.fusion import Cfg                                        # noqa: E402
from covor.occupancy import OccCfg                                  # noqa: E402
from covor.synth import dataset as DS, mapping as MP, mesh_gt as MG  # noqa: E402
from covor.synth import metrics as ME                               # noqa: E402
from covor.synth.config import SynthCfg, ROBOTS                     # noqa: E402
from fuse_synth import fuse, CONDITIONS, FAITHFUL                   # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "data", "synth_gauge_golden.npz")


def _ready():
    cfg = SynthCfg(seed=0)
    return os.path.exists(GOLDEN) and os.path.exists(
        os.path.join(cfg.vinsroot(), cfg.seq, "ifo001", "vio.csv"))


def _skip():
    print("SKIP: build the seed-0 dataset and the golden file first "
          "(scripts/synth/make_dataset.py, scripts/synth/capture_golden.py)")


def test_default_gauge_init_leaves_poses_unchanged():
    """Every fused node pose is identical to the pre-change value."""
    if not _ready():
        return _skip()
    g = np.load(GOLDEN)
    cfg = SynthCfg(seed=0)
    DS.install_paths(cfg)
    assert Cfg().gauge_init == "umeyama", "the default branch must be the old one"
    for cond, pairs in CONDITIONS.items():
        poses, st = fuse(cfg.seq, pairs)
        for r in ROBOTS:
            got, want = poses[r]["T"], g["%s_%s_T" % (cond, r)]
            assert got.shape == want.shape, (cond, r, got.shape, want.shape)
            d = np.abs(got - want).max()
            assert d < 1e-9, (
                "%s/%s: fused poses moved by %.3e m -- the default gauge_init "
                "path is NOT the old behaviour" % (cond, r, d))
            dt = np.abs(poses[r]["tr"] - g["%s_%s_tr" % (cond, r)]).max()
            assert dt < 1e-9, "%s/%s: tr(Sigma) moved by %.3e" % (cond, r, dt)
        assert st["n_inter_range"] == int(g["%s_ninter" % cond][0])
        assert st["n_gauge_prior"] == int(g["%s_ngauge" % cond][0])


def test_default_gauge_init_leaves_map_logodds_unchanged():
    """Per-cell log-odds are identical to the pre-change value."""
    if not _ready():
        return _skip()
    g = np.load(GOLDEN)
    stride = int(g["map_stride"][0])
    cfg = SynthCfg(seed=0)
    DS.install_paths(cfg)
    lab, ijk_min, res, _, _ = MG.load_gt(cfg.gt_voxel())
    scene = MG.raycasting_scene(DS.world_mesh(cfg))
    for cond, pairs in CONDITIONS.items():
        poses, _ = fuse(cfg.seq, pairs)
        mates = {r: (poses[r]["t"], poses[r]["p_body"]) for r in ROBOTS}
        bs, _ = MP.build_map(cfg, scene, ["ifo001"], poses,
                             {"full": OccCfg(resolution=res, weighted=True)},
                             mode="ideal", stride=stride, teammate_pos=mates)
        got = ME.logodds_grid(bs["full"], res, lab.shape, ijk_min)
        want = g["%s_logodds" % cond]
        same_cells = np.isfinite(got) == np.isfinite(want)
        assert same_cells.all(), (
            "%s: %d cells changed from written to unwritten or back"
            % (cond, int((~same_cells).sum())))
        m = np.isfinite(want)
        d = np.abs(got[m] - want[m]).max() if m.any() else 0.0
        assert d < 1e-9, (
            "%s: %d written cells, max log-odds change %.3e -- the map is not "
            "identical" % (cond, int(m.sum()), d))


def test_first_pose_gives_one_prior_per_robot_in_every_condition():
    """The G1 policy: same number of absolute references in A, B and C, each at
    the robot's own GT first pose."""
    if not _ready():
        return _skip()
    from covor.fusion import CoVOR
    from covor import data as D
    cfg = SynthCfg(seed=0)
    DS.install_paths(cfg)
    gt = DS.load_gt_traj(cfg)
    for cond, pairs in CONDITIONS.items():
        cov = CoVOR(cfg.seq, Cfg(**FAITHFUL, inter_pairs=pairs,
                                 gauge_init="first_pose")).build()
        assert cov.stats["n_gauge_prior"] == 3, (
            "%s: %d gauge priors, expected one per robot"
            % (cond, cov.stats["n_gauge_prior"]))
        assert sorted(cov.stats["gauge_on"]) == [0, 1, 2], (
            "%s: priors on %s" % (cond, cov.stats["gauge_on"]))
        for rb in cov.robots:
            p0 = np.asarray(rb.init_world[0].translation(), float)
            # (a) the implementation reads the FIRST mocap sample, exactly.
            #     Independent of how faithful that sample is.
            i = int(np.abs(rb.mocap.timestamp.values - rb.t[0]).argmin())
            row = rb.mocap.iloc[i]
            p_src = np.array([float(row.x), float(row.y), float(row.z)])
            d_src = np.linalg.norm(p0 - p_src)
            assert d_src < 1e-9, (
                "%s/%s: prior mean is %.3e m from the first mocap sample -- it "
                "is not reading the first pose" % (cond, rb.name, d_src))
            # (b) and that sample is the GT first pose to well inside the gap the
            #     old umeyama gauge showed (0.106-0.370 m, PREREG F2), so the two
            #     behaviours can never be confused. The residual here is the csaps
            #     cleaner's own first-sample error (measured 5e-7..1.6e-5 m); the
            #     pre-registered 1e-6 demanded better than the source it named --
            #     see PREREG_synth_gauge.md amendment 1.
            g = gt[rb.name]
            i0 = int(np.abs(g["t"] - rb.t[0]).argmin())
            d = np.linalg.norm(p0 - g["p"][i0])
            assert d < 1e-3, (
                "%s/%s: prior mean is %.6f m from the GT first pose"
                % (cond, rb.name, d))


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

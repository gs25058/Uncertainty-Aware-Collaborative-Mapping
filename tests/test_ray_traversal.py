#!/usr/bin/env python3
"""Regression test for the occupancy ray traversal.

WHY THIS EXISTS. The first vectorized ``integrate_frame`` sampled each beam at
res/2 in arc length instead of doing a real DDA. That dropped 2-4x the intended
free evidence into every cell containing several samples, and skipped cells the
beam only clipped. Effective l_free became -0.8..-1.6 against l_occ=+0.85, i.e.
the safety asymmetry |l_free| < |l_occ| of proposal §4.6 was silently INVERTED,
eroding thin structure out of the map. The unit check at the time only compared
the NUMBER of free cells, so it passed. These tests compare per-cell log-odds
VALUES against the single-ray reference DDA ``_voxel_traverse``.

Run: python tests/test_ray_traversal.py   (or: pytest tests/test_ray_traversal.py)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covor.occupancy import OccupancyBuilder, OccCfg, _voxel_traverse, _dda_batch

RES = 0.10
DIRS = np.array([
    [3.0, 0.0, 0.0],          # axis aligned
    [0.0, -2.4, 0.0],
    [0.0, 0.0, 1.7],
    [2.0, 2.0, 0.0],          # 45 deg in-plane (corner-clipping case)
    [1.8, 1.8, 1.8],          # body diagonal (worst case for arc-length sampling)
    [2.31, -1.07, 0.63],      # generic
    [-0.94, 2.85, -1.42],
    [4.5, 0.31, -2.02],
])


def _reference_logodds(origin, pts, cfg, w=None):
    """Per-cell log-odds from the single-ray reference DDA (ground truth)."""
    acc = {}
    w = np.ones(len(pts)) if w is None else w
    for k, e in enumerate(pts):
        free_idx, end_idx = _voxel_traverse(origin, e, cfg.resolution)
        for v in free_idx:
            acc[tuple(v)] = acc.get(tuple(v), 0.0) + w[k] * cfg.l_free
        acc[tuple(end_idx)] = acc.get(tuple(end_idx), 0.0) + w[k] * cfg.l_occ
    return acc


def _builder_logodds(origin, pts, cfg, tr_sigma=0.0, sigma_Z=None):
    """Per-cell log-odds actually written into the tree by integrate_frame."""
    b = OccupancyBuilder(cfg)
    T = np.eye(4); T[:3, 3] = origin
    sZ = np.zeros(len(pts)) if sigma_Z is None else sigma_Z
    b.integrate_frame(T, tr_sigma, pts - origin, sZ)   # P_cam == world pts (R=I)
    b.finalize()
    out = {}
    for it in b.tree.begin_leafs():
        v = tuple(np.floor(np.array(it.getCoordinate()) / cfg.resolution).astype(np.int64))
        out[v] = it.getValue()
    return out


def _compare(ref, got, tol=1e-4, ctx=""):
    """Assert the two cell->log-odds maps agree on keys AND on values."""
    kr, kg = set(ref), set(got)
    assert kr == kg, ("%s cell sets differ: %d missing, %d spurious"
                      % (ctx, len(kr - kg), len(kg - kr)))
    bad = [(k, ref[k], got[k]) for k in kr if abs(ref[k] - got[k]) > tol]
    assert not bad, ("%s %d cells differ in VALUE, e.g. %s: reference %.4f vs got %.4f"
                     % (ctx, len(bad), bad[0][0], bad[0][1], bad[0][2]))


def test_single_ray_matches_reference_dda():
    """One beam at a time: cells and per-cell log-odds must match the DDA exactly."""
    cfg = OccCfg(resolution=RES, weighted=False, clamp_min=-1e3, clamp_max=1e3)
    for e in DIRS:
        pts = e[None, :]
        _compare(_reference_logodds(np.zeros(3), pts, cfg),
                 _builder_logodds(np.zeros(3), pts, cfg),
                 ctx="dir %s:" % np.round(e, 2))


def test_single_ray_free_evidence_is_exactly_l_free():
    """The bug that motivated this file: each pass-through cell of a LONE beam
    must hold exactly l_free, never a multiple of it."""
    cfg = OccCfg(resolution=RES, weighted=False, clamp_min=-1e3, clamp_max=1e3)
    for e in DIRS:
        got = _builder_logodds(np.zeros(3), e[None, :], cfg)
        free = np.array([v for v in got.values() if v < 0])
        mult = free / cfg.l_free
        assert np.allclose(mult, 1.0, atol=1e-4), (
            "dir %s: free evidence is %.2fx..%.2fx l_free (must be exactly 1x)"
            % (np.round(e, 2), mult.min(), mult.max()))


def test_safety_asymmetry_holds_in_the_tree():
    """Proposal §4.6: a single observation must move a cell toward FREE less
    than it moves a cell toward OCCUPIED."""
    cfg = OccCfg(resolution=RES, weighted=False, clamp_min=-1e3, clamp_max=1e3)
    for e in DIRS:
        got = _builder_logodds(np.zeros(3), e[None, :], cfg)
        v = np.array(list(got.values()))
        assert abs(v[v < 0].min()) < v[v > 0].max(), (
            "dir %s: |l_free|_eff=%.2f >= l_occ=%.2f -- asymmetry inverted"
            % (np.round(e, 2), abs(v[v < 0].min()), v[v > 0].max()))


def test_many_rays_accumulate_additively():
    """Multiple beams into one tree: log-odds additivity (§4.7). Cells shared by
    several beams DO accumulate; each beam still contributes once per cell."""
    cfg = OccCfg(resolution=RES, weighted=False, clamp_min=-1e3, clamp_max=1e3)
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=1.5, size=(60, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.5]
    origin = np.array([0.13, -0.07, 0.21])
    _compare(_reference_logodds(origin, pts, cfg),
             _builder_logodds(origin, pts, cfg), ctx="60 random rays:")


def test_weighted_update_matches_reference():
    """Same, with the §4.5 weight active: per-cell values must be sum(w*l_meas)."""
    cfg = OccCfg(resolution=RES, weighted=True, alpha=0.3, beta=0.5,
                 clamp_min=-1e3, clamp_max=1e3)
    rng = np.random.default_rng(7)
    pts = rng.normal(scale=1.2, size=(40, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.5]
    origin = np.zeros(3)
    sZ = rng.uniform(0.05, 0.9, len(pts))
    tr = 0.06
    b = OccupancyBuilder(cfg)
    w = b.weight(tr, sZ)
    _compare(_reference_logodds(origin, pts, cfg, w=w),
             _builder_logodds(origin, pts, cfg, tr_sigma=tr, sigma_Z=sZ),
             ctx="weighted:")


def test_dda_batch_matches_reference_cells():
    """_dda_batch vs _voxel_traverse, ray by ray, as sets of pass-through cells."""
    origin = np.array([0.02, 0.31, -0.11])
    rng = np.random.default_rng(3)
    pts = origin + rng.normal(scale=1.4, size=(50, 3))
    pts = pts[np.linalg.norm(pts - origin, axis=1) > RES]
    ray, vox = _dda_batch(origin, pts, RES)
    for k in range(len(pts)):
        ref = {tuple(v) for v in _voxel_traverse(origin, pts[k], RES)[0]}
        got = {tuple(v) for v in vox[ray == k]}
        assert ref == got, "ray %d: %d missing, %d spurious" % (
            k, len(ref - got), len(got - ref))
        assert len(vox[ray == k]) == len(got), "ray %d: duplicate cells emitted" % k


# ---------------------------------------------------------------------------
# free-side pose-uncertainty encodings (PREREG_free_side.md §7)
# POST-HOC DERIVED AFTER §8. These check the two properties the pre-registration
# leans on: the sub-voxel guard is an EXACT no-op, and the truncation drops
# exactly floor(k*sigma_u/res) cells. Values, never counts, for the no-op checks
# (appendix A-4: a count-only test once let a real traversal bug through).
# ---------------------------------------------------------------------------
def _free_builder(origin, pts, cfg, S):
    b = OccupancyBuilder(cfg)
    T = np.eye(4); T[:3, 3] = origin
    b.integrate_frame(T, float(np.trace(S)), pts - origin, np.zeros(len(pts)),
                      sigma_pos=S)
    b.finalize()
    out = {}
    for it in b.tree.begin_leafs():
        v = tuple(np.floor(np.array(it.getCoordinate()) / cfg.resolution).astype(np.int64))
        out[v] = it.getValue()
    return out


def test_free_trunc_subvoxel_is_exact_noop():
    """arm_F2 with sigma_u < res/2 must reproduce the unweighted map cell-for-cell."""
    o = np.array([0.05, 0.05, 0.05])
    pts = o + np.array([[0.93, 0.11, 0.0], [0.12, 0.81, 0.07], [-0.7, 0.3, 0.25]])
    base = _builder_logodds(o, pts, OccCfg(resolution=RES, weighted=False))
    sig = (RES / 2 * 0.8) ** 2                       # sigma_u = 0.04 < res/2 = 0.05
    got = _free_builder(o, pts, OccCfg(resolution=RES, weighted=False,
                                       pose_mode="free_trunc"), np.eye(3) * sig)
    _compare(base, got, ctx="free_trunc sub-voxel:")


def test_free_scale_subvoxel_noop_and_occupied_never_changes():
    """arm_F1: sub-voxel is an exact no-op, and l_occ is invariant to sigma_u."""
    o = np.array([0.05, 0.05, 0.05])
    pts = o + np.array([[0.93, 0.11, 0.0], [0.12, 0.81, 0.07]])
    cfg0 = OccCfg(resolution=RES, weighted=False)
    cfgF = OccCfg(resolution=RES, weighted=False, pose_mode="free_scale")
    base = _builder_logodds(o, pts, cfg0)
    _compare(base, _free_builder(o, pts, cfgF, np.eye(3) * (RES / 2 * 0.8) ** 2),
             ctx="free_scale sub-voxel:")
    # occupied endpoints must be identical for ANY sigma_u
    ends = {tuple(np.floor(p / RES).astype(np.int64)) for p in pts}
    for s in (0.0, 0.04, 0.2, 1.0):
        got = _free_builder(o, pts, cfgF, np.eye(3) * s ** 2)
        for e in ends:
            assert abs(got[e] - base[e]) < 1e-9, (
                "free_scale changed an OCCUPIED cell at sigma_u=%.2f: %.6f vs %.6f"
                % (s, got[e], base[e]))


def test_free_trunc_drops_exactly_floor_k_sigma_over_res():
    """The truncated band is exactly floor(k*sigma_u/res) cells per ray."""
    o = np.zeros(3)
    pts = np.array([[1.55, 0.0, 0.0], [0.0, 1.25, 0.0], [0.0, 0.0, 0.95]])
    cfg0 = OccCfg(resolution=RES, weighted=False)
    for s in (0.12, 0.27, 0.44):
        n_exp = int(np.floor(1.0 * s / RES))         # trunc_k = 1
        base = _builder_logodds(o, pts, cfg0)
        got = _free_builder(o, pts, OccCfg(resolution=RES, weighted=False,
                                           pose_mode="free_trunc"), np.eye(3) * s ** 2)
        # every dropped cell is a FREE cell that vanished or lost one ray's evidence
        dropped = sum(1 for k in base
                      if k not in got or abs(base[k] - got[k]) > 1e-9)
        assert dropped == n_exp * len(pts), (
            "sigma_u=%.2f: expected %d dropped free cells (%d rays x %d), got %d"
            % (s, n_exp * len(pts), len(pts), n_exp, dropped))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS  %s" % f.__name__)
    print("\nall %d ray-traversal regression tests passed" % len(fns))

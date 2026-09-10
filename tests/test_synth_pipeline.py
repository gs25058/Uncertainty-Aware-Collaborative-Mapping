#!/usr/bin/env python3
"""Regression tests for the synthetic (mesh -> observation -> map) pipeline.

WHY THIS EXISTS. Everything Part D reports rests on two claims that fail
SILENTLY when they are wrong -- the map still builds, the numbers still print:

  1. the GT voxeliser and the mapper index space on the SAME grid, and
  2. the renderer hands back depth in the frame ``StereoDepth.backproject``
     expects, so a rendered point lands on the surface it was rendered from.

A half-voxel grid offset or a rectification rotation applied the wrong way round
costs a few points of precision -- easy to mistake for "the estimator is a bit
off" and impossible to find later. Both are asserted here by VALUE, following
the rule the ray-traversal tests established (RESULTS_SUMMARY §9-2 / A-4):
compare per-cell log-odds and metric distances, never cell counts.

Run: python tests/test_synth_pipeline.py  (or: pytest tests/test_synth_pipeline.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gtsam                       # noqa: F401  -- must precede open3d
from covor.occupancy import OccupancyBuilder, OccCfg
from covor.synth import mesh_gt as MG

RES = 0.10


def _box(sx=2.0, sy=1.6, sz=1.2, ctr=(0.0, 0.0, 0.6)):
    """A closed axis-aligned box, as an interior room would be."""
    import trimesh
    m = trimesh.creation.box(extents=(sx, sy, sz))
    m.apply_translation(ctr)
    return m


# ---------------------------------------------------------------------------
# 1. grid convention
# ---------------------------------------------------------------------------
def test_octomap_cells_match_mesh_gt_indices():
    """The mapper's cell centres ARE (floor(p/res)+0.5)*res, at every sign."""
    b = OccupancyBuilder(OccCfg(resolution=RES))
    pts = np.array([[0.37, -1.24, 0.61], [-0.05, 0.05, 0.05], [4.99, -3.01, 1.5],
                    [-2.0, -2.0, -0.35], [0.0, 0.0, 0.0], [-0.001, 0.001, 2.999]])
    for p in pts:
        c = np.asarray(b.tree.keyToCoord(b.tree.coordToKey(p)), float)
        want = (MG.index_of(p, RES) + 0.5) * RES
        assert np.allclose(c, want, atol=1e-6), (
            "OctoMap puts %s in a cell centred %s; mesh_gt.index_of says %s. "
            "The GT voxels and the map would be a half voxel apart." % (p, c, want))


def test_voxelize_surface_matches_an_analytic_box():
    """Exact SAT voxelisation of a box, checked cell-by-cell against the
    analytic shell (not against a count)."""
    m = _box(2.0, 1.6, 1.2, (0.0, 0.0, 0.6))
    got = set(map(tuple, MG.voxelize_surface(m, RES).tolist()))
    lo, hi = m.bounds
    # a cell is on the surface iff its cube meets the box boundary: it overlaps
    # the box AND is not strictly inside it
    ii = np.arange(int(np.floor(lo[0] / RES)) - 1, int(np.floor(hi[0] / RES)) + 2)
    jj = np.arange(int(np.floor(lo[1] / RES)) - 1, int(np.floor(hi[1] / RES)) + 2)
    kk = np.arange(int(np.floor(lo[2] / RES)) - 1, int(np.floor(hi[2] / RES)) + 2)
    want = set()
    eps = 1e-9
    for i in ii:
        for j in jj:
            for k in kk:
                a = np.array([i, j, k]) * RES
                b = a + RES
                if np.any(b < lo - eps) or np.any(a > hi + eps):
                    continue                        # disjoint from the box
                inside = np.all(a > lo + eps) and np.all(b < hi - eps)
                if not inside:
                    want.add((int(i), int(j), int(k)))
    assert got == want, ("surface voxels differ: %d missing, %d spurious"
                         % (len(want - got), len(got - want)))


# ---------------------------------------------------------------------------
# 2. renderer frame
# ---------------------------------------------------------------------------
def _render_points(T_wc, mesh, robot="ifo001", downsample=16):
    from covor.synth.render import DepthRenderer
    sc = MG.raycasting_scene(mesh)
    r = DepthRenderer(sc, robot)
    P_cam, sZ = r.frame(T_wc, mode="ideal", downsample=downsample)
    return P_cam, sZ, r


def test_rendered_points_land_on_the_surface_they_came_from():
    """The whole frame chain, end to end, in metres.

    Render depth from a pose inside a box, back-project with the UNMODIFIED
    StereoDepth.backproject, push to world with T_wc, and require every point to
    lie on the box surface. Getting R1 (rectification) transposed, or treating
    the body pose as the camera pose, moves points off the surface by decimetres
    to metres -- this catches both.
    """
    import open3d as o3d
    m = _box(4.0, 3.0, 2.4, (0.0, 0.0, 1.2))
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.2]
    # camera looking along +x: optical z -> world x, optical y -> world -z
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    P_cam, sZ, _ = _render_points(T, m)
    assert len(P_cam) > 500, "renderer produced almost nothing (%d)" % len(P_cam)
    P_w = P_cam @ T[:3, :3].T + T[:3, 3]
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                  o3d.utility.Vector3iVector(m.faces))))
    d = sc.compute_distance(o3d.core.Tensor(P_w.astype(np.float32))).numpy()
    assert d.max() < 1e-3, (
        "rendered points are up to %.3f m off the surface they were rendered "
        "from -- the camera frame chain is wrong" % d.max())


def test_sigma_Z_follows_the_stereo_formula():
    """sigma_Z must be the SAME function of Z the stereo path uses; a renderer
    with its own noise model would make the depth weight incomparable."""
    m = _box(4.0, 3.0, 2.4, (0.0, 0.0, 1.2))
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.2]
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    P_cam, sZ, r = _render_points(T, m)
    # backproject returns points in the RAW infra1 frame (P_cam = P_rect @ R1),
    # while sigma_Z is a function of the RECTIFIED depth, so undo R1 first.
    # Skipping this step is a 0.3 deg error -- invisible in a plot, 0.5 % in Z.
    Z = (P_cam @ r.sd.R1.T)[:, 2]
    want = Z ** 2 / r.fB * r.sd.cfg.disp_sigma_px
    assert np.allclose(sZ, want, rtol=1e-5, atol=1e-9), (
        "sigma_Z deviates from Z_rect^2/(fB)*dd by up to %.6f m"
        % np.abs(sZ - want).max())


# ---------------------------------------------------------------------------
# 3. the control map, by value
# ---------------------------------------------------------------------------
def test_noise_free_control_recovers_the_box_walls():
    """A miniature Part C-1: exact poses + ideal depth inside a known box.

    Asserted on the SIGNED LOG-ODDS of specific cells, not on aggregate scores:
    a cell on the wall the camera faces must end up positive (occupied
    evidence), and a cell in mid-air on the beam must end up negative -- with
    |free| < |occ| per the §4.6 asymmetry.
    """
    from covor.synth.render import DepthRenderer
    m = _box(4.0, 3.0, 2.4, (0.0, 0.0, 1.2))
    sc = MG.raycasting_scene(m)
    T = np.eye(4)
    T[:3, 3] = [-1.5, 0.0, 1.2]
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    b = OccupancyBuilder(OccCfg(resolution=RES, weighted=True, dyn_radius=0.0))
    r = DepthRenderer(sc, "ifo001")
    for dx in (0.0, 0.05, 0.10):                 # a few nearby views
        Tk = T.copy(); Tk[0, 3] += dx
        P_cam, sZ = r.frame(Tk, mode="ideal", downsample=8)
        b.integrate_frame(Tk, 0.0, P_cam, sZ)
    b.finalize()
    # The wall plane sits at x = 2.0, EXACTLY on a voxel boundary, so the beam's
    # endpoint falls in whichever of the two straddling cells the hit point
    # floors into; the other one is carved free by the same beam. Both cells are
    # GT-occupied (the surface touches both), which is the discretisation effect
    # the map's 1-voxel tolerance exists for -- so the assertion is on the PAIR.
    cells = [b.tree.search(np.array([x, 0.0, 1.2])) for x in (1.95, 2.05)]
    air = b.tree.search(np.array([0.0, 0.0, 1.2]))       # mid-air on the beam
    vals = [c.getValue() for c in cells if c is not None]
    assert vals and air is not None, "control map missed the cells"
    assert max(vals) > 0, (
        "neither cell straddling the wall carries positive log-odds (%s) -- "
        "occupied evidence did not land on the surface" % vals)
    assert air.getValue() < 0, (
        "the mid-air cell carries log-odds %.3f -- free carving did not happen"
        % air.getValue())
    assert abs(air.getValue()) < 12 * max(vals), (
        "free evidence (%.3f) overwhelms occupied (%.3f): the §4.6 asymmetry is "
        "inverted" % (air.getValue(), max(vals)))


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

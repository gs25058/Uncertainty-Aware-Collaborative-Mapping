#!/usr/bin/env python3
"""The 3D viewer must draw the grid, and the template must get what it reads.

Two failure modes, both silent in a browser:

  * a field the template calls .toFixed() on is missing -> the panel throws and
    the page shows a half-built scene with no error a reader would notice;
  * the plan classes drift from the grid's labels -> the picture says something
    the judgement never said, which is the one thing DESIGN §5 forbids.

So the template is parsed for every field it dereferences and the payload is
checked against that list, and every painted class is checked back against
entry_grid.npz cell by cell.

Run: python tests/test_entry_3d.py   (or: pytest tests/test_entry_3d.py)
"""
import base64
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
from covor.entry.config import EntryCfg, UNKNOWN, FREE, OCCUPIED, NARROW  # noqa: E402
from build_entry_map import build_entry_grid                             # noqa: E402
import export_entry_3d as EX                                             # noqa: E402

GT = os.path.join(ROOT, "results", "synth_room909", "gt_voxel.npz")
TPL = os.path.join(ROOT, "web", "entry_map_3d.template.html")


def _payload():
    z = np.load(GT, allow_pickle=False)
    lab, ijk = z["labels"], z["ijk_min"]
    res = float(np.asarray(z["res"]).ravel()[0])
    cfg = EntryCfg(w=0.70)
    roi = EX.roi_from_gt(lab, ijk, res)
    v, g = EX.variant("gt", "GT", "", lab, ijk, res, None, None, {}, cfg,
                      "walk", None, roi)
    return v, g, cfg, ijk, res, lab


def test_template_contract_is_satisfied_by_the_payload():
    if not (os.path.exists(GT) and os.path.exists(TPL)):
        return print("SKIP: gt_voxel.npz or the template is missing")
    v, _, cfg, ijk, res, lab = _payload()
    body = open(TPL, encoding="utf-8").read()
    body = body[body.index("const DATA ="):]
    for name in sorted(set(re.findall(r"\bv\.([a-z_0-9]+)", body))):
        assert name in v, (
            "the template dereferences v.%s and the exporter never sets it -- "
            "the stats panel would throw in the browser" % name)
    data_keys = set(re.findall(r"\bDATA\.([a-z_0-9]+)", body))
    for k in sorted(data_keys):
        assert k in ("variants", "cfg", "grid", "roi", "z_scale", "entry_xy"), (
            "template reads DATA.%s, which the exporter does not emit" % k)
    for k in sorted(set(re.findall(r"DATA\.cfg\.([a-z_0-9]+)", body))):
        assert k in ("body_lo", "body_hi", "w", "k_sigma", "res", "clear_walk",
                     "clear_narrow"), "template reads DATA.cfg.%s" % k


def _vox(v, axis):
    """Decode one index plane the way the template's unvox does."""
    dt = np.dtype("<u2") if v["vox"].get("bits") == 16 else np.dtype(np.uint8)
    return np.frombuffer(base64.b64decode(v["vox"][axis]), dt)


def test_voxel_indices_fit_in_the_uint8_planes():
    """A grid under 256 cells stays one byte per index and carries no bits
    field -- the format every room909 viewer was baked in."""
    if not os.path.exists(GT):
        return print("SKIP: gt_voxel.npz is missing")
    v, _, _, _, _, lab = _payload()
    assert max(lab.shape) < 256
    assert "bits" not in v["vox"], "a small grid must keep the 8-bit format"
    for a in "ijk":
        arr = _vox(v, a)
        assert len(arr) == v["n_occ"]
        assert arr.max() < lab.shape["ijk".index(a)]


def test_a_long_grid_uses_16_bit_planes_and_round_trips():
    """The corridor is 422 cells in y. Its indices must come back exactly --
    by value, against np.nonzero of the grid -- not wrapped at 256."""
    gt = os.path.join(ROOT, "results", "synth_corridor915", "gt_voxel_flat.npz")
    if not os.path.exists(gt):
        return print("SKIP: corridor915 GT is missing")
    z = np.load(gt, allow_pickle=False)
    lab, ijk = z["labels"], z["ijk_min"]
    res = float(np.asarray(z["res"]).ravel()[0])
    assert max(lab.shape) > 255
    cfg = EntryCfg(res=res, w=0.70)
    v, _ = EX.variant("gt", "GT", "", lab, ijk, res, None, None, {}, cfg, "walk",
                      None, EX.roi_from_gt(lab, ijk, res),
                      entry_xy=(-2.35, -19.55), entry_snap_m=1.5)
    assert v["vox"]["bits"] == 16
    ii, jj, kk = np.nonzero(lab == OCCUPIED)
    for a, ref in zip("ijk", (ii, jj, kk)):
        assert np.array_equal(_vox(v, a), ref), "vox.%s does not round-trip" % a
    assert int(jj.max()) > 255, "the test must exercise an index past one byte"
    try:
        EX.index_bits((70000, 2, 2))
    except SystemExit:
        pass
    else:
        raise AssertionError("a grid past 16 bits was accepted")


def test_painted_classes_agree_with_the_grid_cell_by_cell():
    """DESIGN §5 in the 3D layer: the drawing may not say anything the grid did
    not. Each class is checked back against entry_grid.npz."""
    if not os.path.exists(GT):
        return print("SKIP: gt_voxel.npz is missing")
    v, g, cfg, _, _, _ = _payload()
    gr = np.load(GT, allow_pickle=False)
    plan = np.frombuffer(base64.b64decode(v["plan"]), np.uint8).reshape(
        gr["labels"].shape[1], gr["labels"].shape[0]).T
    lab = np.asarray(g["walk_label"])
    P = np.asarray(g["walk_passable"], bool)
    R = np.asarray(g["walk_reachable"], bool)
    wc = np.asarray(g["walk_width_class"])

    assert (lab[plan == EX.WALL] == OCCUPIED).all(), "WALL painted off an obstacle"
    assert (lab[plan == EX.UNSEEN] == UNKNOWN).all(), "UNSEEN painted off unknown"
    assert R[plan == EX.WALKABLE].all(), "WALKABLE painted where the grid says unreachable"
    assert (lab[plan == EX.REACH] == FREE).all(), "the swept floor left the free set"
    assert (P & ~R)[plan == EX.STRANDED].all(), "STRANDED is not passable-and-cut-off"
    sq = plan == EX.SQUEEZE
    if sq.any():
        assert ((lab[sq] == FREE) & (wc[sq] == NARROW) & ~P[sq]).all(), (
            "SQUEEZE must be free, narrow-grade and NOT offered as passable")
    assert not (plan == EX.DRONE_ONLY).any(), (
        "the drone-only class is unused in this project and must stay empty")
    # every reachable cell must be drawn as walkable: the offer cannot shrink
    # between the grid and the picture
    assert (plan[R] == EX.WALKABLE).all(), "a reachable cell was painted as something else"


def test_every_baked_viewer_is_loadable_at_its_own_resolution():
    """Validate the files that actually ship, not just a freshly built payload.

    The contract test above builds its payload from the 0.10 m GT, so it never
    exercises a 0.05 m grid -- where the voxel index planes are 222 wide against
    a 255 ceiling, which is the number that would wrap silently and scatter the
    voxels if the grid ever grew. This walks the baked HTML instead.
    """
    files = sorted(glob.glob(os.path.join(ROOT, "web", "entry_map_3d_*.html")))
    if not files or not os.path.exists(TPL):
        return print("SKIP: no baked viewer to check")
    body = open(TPL, encoding="utf-8").read()
    body = body[body.index("const DATA ="):]
    need = sorted(set(re.findall(r"\bv\.([a-z_0-9]+)", body)))
    for f in files:
        h = open(f, encoding="utf-8").read()
        m = re.search(r"const DATA = (\{.*?\});\n", h, re.S)
        assert m, "%s has no baked DATA -- the placeholder was never filled" % f
        data = json.loads(m.group(1))
        g = data["grid"]
        assert data["variants"], "%s carries no variants" % f
        for v in data["variants"]:
            for k in need:
                assert k in v, "%s / %s is missing v.%s" % (f, v["key"], k)
            for axis, n in zip("ijk", (g["nx"], g["ny"], g["nz"])):
                bits = v["vox"].get("bits", 8)
                idx = _vox(v, axis)
                assert len(idx) == v["n_occ"], (
                    "%s / %s: vox.%s has %d entries for %d occupied cells"
                    % (f, v["key"], axis, len(idx), v["n_occ"]))
                assert n < (1 << bits), (
                    "%s: the grid is %d cells on %s and the payload stores "
                    "%d bits per index" % (f, n, axis, bits))
                if len(idx):
                    assert int(idx.max()) < n, (
                        "%s / %s: vox.%s reaches %d on a %d-cell axis -- the "
                        "index plane wrapped" % (f, v["key"], axis, idx.max(), n))
            plan = np.frombuffer(base64.b64decode(v["plan"]), np.uint8)
            assert len(plan) == g["nx"] * g["ny"], (
                "%s / %s: plan is %d bytes for a %dx%d grid"
                % (f, v["key"], len(plan), g["nx"], g["ny"]))
            seen = set(np.unique(plan).tolist())
            assert seen <= {0, 1, 2, 4, 5, 6, 7}, (
                "%s / %s paints classes %s; 3 is the unused drone class"
                % (f, v["key"], sorted(seen)))


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

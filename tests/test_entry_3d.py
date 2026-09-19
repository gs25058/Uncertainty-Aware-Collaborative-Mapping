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


def test_voxel_indices_fit_in_the_uint8_planes():
    """The payload stores i/j/k as one byte each. A grid over 255 cells on any
    axis would wrap silently and scatter the voxels."""
    if not os.path.exists(GT):
        return print("SKIP: gt_voxel.npz is missing")
    v, _, _, _, _, lab = _payload()
    assert max(lab.shape) < 256, (
        "grid %s exceeds the uint8 payload; the exporter needs wider planes"
        % (lab.shape,))
    for a in "ijk":
        arr = np.frombuffer(base64.b64decode(v["vox"][a]), np.uint8)
        assert len(arr) == v["n_occ"]
        assert arr.max() < lab.shape["ijk".index(a)]


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


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

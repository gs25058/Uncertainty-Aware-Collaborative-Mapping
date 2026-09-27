#!/usr/bin/env python3
"""The traversability viewer must draw traverse.py's verdict and nothing else.

  1. every field the template dereferences on a variant is in the payload;
  2. each painted tile class equals the verdict of covor/entry/traverse.py for
     that column, cell by cell, and sits at that column's support row;
  3. the route is a chain of 8-neighbours from the entry to the farthest
     reached column, every point reached;
  4. the voxel planes round-trip to the occupied cells by value.

Run: python tests/test_traverse_3d.py   (or: pytest tests/test_traverse_3d.py)
"""
import base64
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "entry"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from covor.entry import traverse as TV                               # noqa: E402
from covor.entry.config import OCCUPIED                              # noqa: E402
import export_traverse_3d as EX                                      # noqa: E402
from test_entry_traverse import corridor, FLOOR                      # noqa: E402

TPL = os.path.join(ROOT, "web", "traverse_3d.template.html")


def dec16(s):
    return np.frombuffer(base64.b64decode(s), "<u2")


def scene():
    lab = corridor(nx=40)
    lab[18:21, 1:-1, FLOOR + 17:FLOOR + 19] = OCCUPIED        # beam -> stoop
    lab[26, 1:-1, FLOOR + 1:28] = OCCUPIED                    # wall with a 0.5 m door
    lab[26, 3:8, FLOOR + 1:28] = 1                            # FREE
    return lab


def build():
    lab = scene()
    v, t = EX.variant("m", "m", "", lab, np.zeros(3, int), 0.1, (0.45, 0.55), 1.0,
                      None, True)
    lab3, t2, P = EX.judge(lab, np.zeros(3, int), 0.1, (0.45, 0.55), 1.0)
    return v, t2, P, lab3


def test_template_contract():
    v, *_ = build()
    body = open(TPL, encoding="utf-8").read()
    body = body[body.index("const DATA ="):]
    for name in sorted(set(re.findall(r"\bv\.([a-z_0-9]+)", body))):
        assert name in v, "template reads v.%s, the exporter never sets it" % name


def test_tiles_are_the_verdict_cell_by_cell():
    v, t, P, _ = build()
    cls, row = EX.tile_classes(t, P)
    ti, tj, tk = dec16(v["tiles"]["i"]), dec16(v["tiles"]["j"]), dec16(v["tiles"]["k"])
    tc = np.frombuffer(base64.b64decode(v["tiles"]["c"]), np.uint8)
    assert len(tc) == int((cls > 0).sum())
    assert np.array_equal(tc, cls[ti, tj]) and np.array_equal(tk, row[ti, tj])
    R = t["reach"]
    for i, j, c in zip(ti, tj, tc):
        if c == EX.UNREACHED_T:
            assert not R[i, j] and (P[i, j] >= 0).any()
        else:
            assert R[i, j]
    assert (tc == EX.STOOP_T).any(), "the beam must show a stoop tile"
    assert (tc == EX.SIDE_T).any(), "the 0.5 m door must show a sideways tile"


def test_route_is_a_chain_from_the_entry():
    v, t, *_ = build()
    pts = v["route"]
    assert len(pts) > 10
    ij = [(int(round(p[0] / 0.1 - 0.5)), int(round(p[1] / 0.1 - 0.5))) for p in pts]
    assert ij[0] == tuple(t["entry"][:2])
    for a, b in zip(ij, ij[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, (a, b)
        assert t["reach"][b]
    C = np.where(t["reach"], t["cost"], -1)
    assert C[ij[-1]] == C.max()


def test_voxels_round_trip():
    v, _, _, lab3 = build()
    oi, oj, ok = np.nonzero(lab3 == OCCUPIED)
    for a, ref in zip("ijk", (oi, oj, ok)):
        assert np.array_equal(dec16(v["vox"][a]), ref)


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

#!/usr/bin/env python3
"""scripts/scan/repair_floor.py, pinned by value on a scene with a known answer.

The synthetic scan has what the 2026-09-15 corridor has: a floor that drifts
(z = 0.02 * y - 0.08, -0.08..+0.08 m over 8 m around the dominant plane at 0), a
doubled floor sheet 0.20 m below the real one, a 0.30 m box standing on the
floor, a 2.5 m wall, and a flight of 0.15 m stair treads -- written in a y-up
mesh frame with the same kind of permuting T_mesh_world the corridor uses. The
stairs are there because the first version of the repair read the corridor's
stair landing as floor at +0.11 m and flattened the first tread.

After the repair: every floor vertex (both sheets) is at world z = 0, the box
top is 0.30 m and the wall top 2.50 m above it, no vertex moved in x or y, and
every line of the OBJ except the `v` lines is byte-identical.

Run: python tests/test_scan_repair.py   (or: pytest tests/test_scan_repair.py)
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "scan"))
import gtsam                       # noqa: F401,E402  -- must precede open3d
import repair_floor as RF          # noqa: E402

# world = T @ mesh: mesh x -> world x, mesh -z -> world y, mesh y -> world z
T = np.array([[1.0, 0, 0, 0.3], [0, 0, -1.0, -0.2], [0, 1.0, 0, 1.5], [0, 0, 0, 1]])
DRIFT = 0.02


def floor_z(y):
    return DRIFT * y - 0.08


def build_scene():
    V, F, groups = [], [], []

    def quad(p0, p1, p2, p3):
        b = len(V)
        V.extend([p0, p1, p2, p3])
        F.append([b, b + 1, b + 2])
        F.append([b, b + 2, b + 3])

    # real floor: 0.25 m quads over x 0..4, y 0..8, following the drift
    for x in np.arange(0, 4, 0.25):
        for y in np.arange(0, 8, 0.25):
            quad([x, y, floor_z(y)], [x + .25, y, floor_z(y)],
                 [x + .25, y + .25, floor_z(y + .25)], [x, y + .25, floor_z(y + .25)])
    n_floor = len(V)
    # doubled sheet 0.20 m below, x 1..3, y 3..5
    for x in np.arange(1, 3, 0.25):
        for y in np.arange(3, 5, 0.25):
            quad([x, y, floor_z(y) - .2], [x + .25, y, floor_z(y) - .2],
                 [x + .25, y + .25, floor_z(y + .25) - .2], [x, y + .25, floor_z(y + .25) - .2])
    n_sheet = len(V)
    # 0.5 x 0.5 x 0.3 box at (2, 6): top and four sides
    z0 = floor_z(6.0)
    quad([1.75, 5.75, z0 + .3], [2.25, 5.75, z0 + .3], [2.25, 6.25, z0 + .3], [1.75, 6.25, z0 + .3])
    n_top = len(V)
    for (a, b) in (((1.75, 5.75), (2.25, 5.75)), ((2.25, 5.75), (2.25, 6.25)),
                   ((2.25, 6.25), (1.75, 6.25)), ((1.75, 6.25), (1.75, 5.75))):
        quad([a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z0 + .3], [a[0], a[1], z0 + .3])
    n_box = len(V)
    # stairs: x 2.5..4, y 0.5..2, four 0.15 m treads rising in +x, 0.375 m deep
    n_pre = len(V)
    for k in range(4):
        x0 = 2.5 + 0.375 * k
        zt = floor_z(1.25) + 0.15 * (k + 1)
        quad([x0, 0.5, zt], [x0 + .375, 0.5, zt], [x0 + .375, 2.0, zt], [x0, 2.0, zt])
    n_treads = len(V)
    # wall at x = 4, y 0..8, floor to floor + 2.5
    for y in np.arange(0, 8, 1.0):
        quad([4, y, floor_z(y)], [4, y + 1, floor_z(y + 1)],
             [4, y + 1, floor_z(y + 1) + 2.5], [4, y, floor_z(y) + 2.5])
    Vw = np.array(V, float)
    Tinv = np.linalg.inv(T)
    Vm = Vw @ Tinv[:3, :3].T + Tinv[:3, 3]
    return Vm, np.array(F), dict(floor=(0, n_floor), sheet=(n_floor, n_sheet),
                                 top=(n_sheet, n_top), box=(n_top, n_box),
                                 treads=(n_pre, n_treads), wall=(n_treads, len(V)))


def write_scene(path, Vm, F):
    with open(path, "w") as f:
        f.write("# synthetic scan\nmtllib scene.mtl\n")
        for v in Vm:
            f.write("v %.6f %.6f %.6f\n" % tuple(v))
        f.write("vt 0.5 0.5\nusemtl Material_1\n")
        for t in F:
            f.write("f %d/1 %d/1 %d/1\n" % tuple(t + 1))


def run_repair(tmp):
    Vm, F, g = build_scene()
    src = os.path.join(tmp, "scene.obj")
    write_scene(src, Vm, F)
    lines, vidx, Vm_r, F_r = RF.read_obj(src)
    Vw = Vm_r @ T[:3, :3].T + T[:3, 3]
    z, rep = RF.repair(Vw, F_r)
    a, sgn = RF.up_axis(T)
    Vm2 = Vm_r.copy()
    Vm2[:, a] += (z - Vw[:, 2]) / sgn
    out = os.path.join(tmp, "scene.fixed.obj")
    RF.write_obj(lines, vidx, Vm2, out)
    lines2, _, Vm3, F3 = RF.read_obj(out)
    Vw3 = Vm3 @ T[:3, :3].T + T[:3, 3]
    return lines, lines2, Vw, Vw3, F_r, F3, g, rep


def test_floor_is_flat_and_heights_above_it_survive():
    with tempfile.TemporaryDirectory() as tmp:
        lines, lines2, Vw, Vw3, F, F3, g, rep = run_repair(tmp)
    s = slice(*g["floor"])
    assert np.abs(Vw3[s, 2]).max() < 1e-5, "real floor not on z = 0"
    s = slice(*g["sheet"])
    assert np.abs(Vw3[s, 2]).max() < 1e-5, "doubled sheet not merged onto the floor"
    s = slice(*g["top"])
    assert np.abs(Vw3[s, 2] - 0.30).max() < 0.011, (
        "box top should stay 0.30 m above the floor, got %s" % Vw3[s, 2])
    s = slice(*g["wall"])
    top = Vw3[s][np.argsort(Vw3[s, 2])[-4:], 2]
    assert np.abs(top - 2.50).max() < 0.02, "wall top should stay 2.50 m up, got %s" % top
    assert rep["n_snapped_from_below_2cm"] >= 1
    # each tread is level while the floor under it drifts 0.02 m/m, so its
    # height ABOVE THE LOCAL FLOOR is 0.15 (k+1) +- 0.015 m across its 1.5 m
    # width by construction; the repair must reproduce exactly that
    a0, a1 = g["treads"]
    for k in range(4):
        idx = np.arange(a0 + 4 * k, a0 + 4 * k + 4)
        want = 0.15 * (k + 1) + (floor_z(1.25) - floor_z(Vw[idx, 1]))
        assert np.abs(Vw3[idx, 2] - want).max() < 0.011, (
            "tread %d: %s, want %s" % (k, Vw3[idx, 2], want))


def test_nothing_moves_sideways_and_only_v_lines_change():
    with tempfile.TemporaryDirectory() as tmp:
        lines, lines2, Vw, Vw3, F, F3, g, rep = run_repair(tmp)
    assert np.abs(Vw3[:, :2] - Vw[:, :2]).max() < 2e-6, "a vertex moved in x or y"
    assert np.array_equal(F, F3)
    assert len(lines) == len(lines2)
    for a, b in zip(lines, lines2):
        if not a.startswith("v "):
            assert a == b, "a non-vertex line changed: %r -> %r" % (a, b)


def test_up_axis_is_read_from_the_transform():
    assert RF.up_axis(T) == (1, 1.0)
    T2 = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1]])
    assert RF.up_axis(T2) == (2, -1.0)


if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok  ", f.__name__)
    print("%d passed" % len(fs))

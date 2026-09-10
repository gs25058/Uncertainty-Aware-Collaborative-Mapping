"""Ground-truth flight paths inside the scanned room (Part B-1).

A trajectory here is COLLISION-FREE BY CONSTRUCTION, not by inspection: every
waypoint-to-waypoint leg is a shortest path found by BFS over the cells that
already satisfy the clearance test, and the sampled poses are then re-checked
against the GT grid. ``plan_all`` raises if a single sample fails.

The three robots get disjoint zones (equal-area split of the flyable region
along x) and different cruise heights, so "3 drones" differs from "1 drone" in
coverage as well as in graph connectivity -- which is the whole point of the
map (ii) / map (i) pair in Part D.
"""
import numpy as np
from scipy import ndimage

from . import mesh_gt as MG
from .config import ROBOTS


# ---------------------------------------------------------------------------
# flyable volume
# ---------------------------------------------------------------------------
def clearance_field(lab, res):
    """Distance (m) from every cell to the nearest cell that is not GT FREE.

    Not-free means occupied OR unknown: an unknown cell is somewhere we have no
    ground truth for, and flying into it would put observations where nothing
    can be scored. Both are obstacles for planning.
    """
    return ndimage.distance_transform_edt(lab == MG.FREE) * res


def flyable(lab, ijk_min, res, clearance, z_lo, z_hi):
    """(mask, world z of each k) of cells at least ``clearance`` from anything
    that is not GT free, within the height band."""
    d = clearance_field(lab, res)
    zc = (np.arange(lab.shape[2]) + ijk_min[2] + 0.5) * res
    m = d >= clearance
    m[:, :, ~((zc >= z_lo) & (zc <= z_hi))] = False
    return m, zc


def zone_split(mask2d, n):
    """Split a 2-D flyable footprint into ``n`` equal-cell zones along x."""
    ii = np.nonzero(mask2d.any(1))[0]
    cnt = mask2d.sum(1)[ii]
    edges = np.searchsorted(np.cumsum(cnt), np.linspace(0, cnt.sum(), n + 1)[1:-1])
    bnds = [0] + [int(ii[min(e, len(ii) - 1)]) for e in edges] + [mask2d.shape[0]]
    out = []
    for k in range(n):
        z = np.zeros_like(mask2d)
        z[bnds[k]:bnds[k + 1]] = mask2d[bnds[k]:bnds[k + 1]]
        out.append(z)
    return out


# ---------------------------------------------------------------------------
# path planning on the flyable footprint
# ---------------------------------------------------------------------------
def _bfs(mask, start, goal):
    """Shortest 8-connected path through ``mask`` (True = passable), as cells."""
    if not (mask[start] and mask[goal]):
        return None
    H, W = mask.shape
    prev = -np.ones((H, W, 2), np.int32)
    seen = np.zeros((H, W), bool)
    seen[start] = True
    q = [start]
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    head = 0
    while head < len(q):
        c = q[head]; head += 1
        if c == goal:
            break
        for di, dj in nb:
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < H and 0 <= n[1] < W and mask[n] and not seen[n]:
                seen[n] = True
                prev[n] = c
                q.append(n)
    if not seen[goal]:
        return None
    path, c = [goal], goal
    while c != start:
        c = tuple(prev[c])
        path.append(c)
    return path[::-1]


def _visible(mask, a, b):
    """Is the straight segment a->b entirely inside ``mask``? (dense sampling)"""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 3) + 1
    ii = np.round(np.linspace(a[0], b[0], n)).astype(int)
    jj = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return bool(mask[ii, jj].all())


def _shortcut(mask, path):
    """Greedy line-of-sight simplification -- keeps the path inside ``mask``."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _visible(mask, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def lawnmower(mask2d, res, row_step):
    """Boustrophedon waypoints (cell indices) covering ``mask2d``.

    Rows run along x; consecutive rows alternate direction. Rows are connected
    by BFS through the mask itself, so a row pair separated by an obstacle is
    joined by a legal detour instead of a straight line through the obstacle.
    """
    step = max(1, int(round(row_step / res)))
    rows = [j for j in range(mask2d.shape[1]) if mask2d[:, j].any()]
    if not rows:
        return []
    sel = rows[::step]
    if sel[-1] != rows[-1]:
        sel.append(rows[-1])
    ends = []
    for n, j in enumerate(sel):
        ii = np.nonzero(mask2d[:, j])[0]
        # longest contiguous run in this row: a row split by a pillar should not
        # produce a leg that flies through the pillar
        br = np.split(ii, np.nonzero(np.diff(ii) > 1)[0] + 1)
        run = max(br, key=len)
        a, b = (int(run[0]), int(run[-1]))
        ends.append(((a, j), (b, j)) if n % 2 == 0 else ((b, j), (a, j)))
    way = []
    for n, (p0, p1) in enumerate(ends):
        if n and way:
            seg = _bfs(mask2d, way[-1], p0)
            if seg is None:
                continue
            way += _shortcut(mask2d, seg)[1:]
        else:
            way.append(p0)
        way.append(p1)
    return way


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def resample(xy, speed, rate, duration):
    """Constant-speed samples along a polyline, looping back and forth until
    ``duration`` is filled. Returns (t (N,), p (N,2))."""
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    n = int(round(duration * rate)) + 1
    t = np.arange(n) / rate
    d = speed * t
    # ping-pong along the path so the drone stays in its zone for the whole run
    if total <= 0:
        return t, np.repeat(xy[:1], n, axis=0)
    ph = np.mod(d, 2 * total)
    ph = np.where(ph > total, 2 * total - ph, ph)
    p = np.stack([np.interp(ph, s, xy[:, 0]), np.interp(ph, s, xy[:, 1])], 1)
    return t, p


def _yaw_from(p, smooth=9):
    d = np.gradient(p, axis=0)
    k = np.ones(smooth) / smooth
    d = np.stack([np.convolve(d[:, 0], k, "same"), np.convolve(d[:, 1], k, "same")], 1)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def poses_from_path(t, p_xy, z, cfg, phase=0.0):
    """(N,) t, (N,3) position, (N,3,3) body rotation.

    Body frame: x forward, z up (MILUV/px4 convention; body_T_cam0 puts the
    camera's optical z along body x). Yaw follows the heading with a slow sweep;
    roll/pitch carry a small oscillation so the gravity prior is not acting on a
    perfectly level trajectory.
    """
    from scipy.spatial.transform import Rotation as R
    psi = _yaw_from(p_xy) + np.radians(cfg.yaw_amp_deg) * \
        np.sin(2 * np.pi * t / cfg.yaw_period_s + phase)
    a = np.radians(cfg.tilt_amp_deg)
    roll = a * np.sin(2 * np.pi * t / cfg.tilt_period_s + phase)
    pitch = a * np.cos(2 * np.pi * t / (cfg.tilt_period_s * 1.3) + phase)
    Rb = R.from_euler("zyx", np.stack([psi, pitch, roll], 1)).as_matrix()
    P = np.column_stack([p_xy, z])
    return P, Rb


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def check_clear(P, lab, ijk_min, res, clearance):
    """Per-sample clearance (m) to the nearest non-free cell; and how many
    samples violate the requirement. Nothing downstream may run on a violation:
    a pose inside a wall renders depth from inside geometry."""
    d = clearance_field(lab, res)
    idx = MG.index_of(P, res) - ijk_min
    ok = np.all((idx >= 0) & (idx < np.array(lab.shape)), axis=1)
    out = np.full(len(P), -1.0)
    i = idx[ok]
    out[ok] = d[i[:, 0], i[:, 1], i[:, 2]]
    return out, int((out < clearance).sum())


def plan_all(lab, ijk_min, res, cfg):
    """Plan every robot's GT trajectory. Returns {robot: dict(t, p, R, ...)}."""
    z_lo = min(cfg.heights) - cfg.bob_amp
    z_hi = max(cfg.heights) + cfg.bob_amp
    m3, zc = flyable(lab, ijk_min, res, cfg.clearance, z_lo, z_hi)
    foot = m3.any(2)                      # any height in the band is flyable
    # keep the largest connected footprint: an isolated pocket cannot be reached
    cc, _ = ndimage.label(foot)
    if cc.max():
        foot = cc == (1 + np.argmax(np.bincount(cc.ravel())[1:]))
    zones = zone_split(foot, len(cfg.heights))

    out = {}
    for k, (rob, h) in enumerate(zip(ROBOTS, cfg.heights)):
        # Plan on the footprint flyable over the WHOLE height band this robot
        # actually occupies (cruise +- bob), not just at its cruise level: the
        # z oscillation would otherwise take it into cells the plan never
        # checked (measured: 28 samples below clearance when planning at the
        # cruise level alone).
        # every CELL the bob can put the drone in -- a cell whose extent
        # [zc-res/2, zc+res/2) overlaps [h-bob, h+bob], not just the cell
        # centres inside that interval (that off-by-one left 28 samples in
        # unchecked cells)
        kband = ((zc + res / 2 > h - cfg.bob_amp) &
                 (zc - res / 2 < h + cfg.bob_amp))
        mk = zones[k] & np.all(m3[:, :, kband], axis=2)
        cc, _ = ndimage.label(mk)
        if cc.max():
            mk = cc == (1 + np.argmax(np.bincount(cc.ravel())[1:]))
        way = lawnmower(mk, res, cfg.row_step)
        if len(way) < 2:
            raise RuntimeError("no lawnmower path for %s" % rob)
        xy = (np.array(way, float) + ijk_min[:2] + 0.5) * res
        t, p = resample(xy, cfg.speed, cfg.rate_hz, cfg.duration_s)
        z = h + cfg.bob_amp * np.sin(2 * np.pi * t / cfg.bob_period_s + k)
        P, Rb = poses_from_path(t, p, z, cfg, phase=k * 2.0)
        clr, bad = check_clear(P, lab, ijk_min, res, cfg.clearance)
        if bad:
            raise RuntimeError(
                "%s: %d of %d GT poses are closer than %.2f m to a non-free cell "
                "(min %.3f m). A pose inside geometry renders depth from inside "
                "the mesh, so nothing downstream may run." %
                (rob, bad, len(P), cfg.clearance, clr.min()))
        out[rob] = dict(t=t, p=P, R=Rb, zone=mk, waypoints=xy,
                        clearance=clr, n_violation=bad,
                        route_len=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                        path_len=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()))
    return out

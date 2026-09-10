"""Scanned-mesh -> occupancy ground truth (Part A).

Three jobs, in order:

  1. ``inspect``   -- units, up-axis, watertightness, room size. A photogrammetry
     scan arrives in an arbitrary frame; every downstream number is invalid if
     this is wrong, so it is reported and frozen into a config file rather than
     guessed per script.
  2. ``mesh_to_world`` -- the 4x4 that takes mesh coordinates to the experiment
     world frame: up-axis -> +z, floor plane -> z = 0, footprint centred on
     (0, 0). Saved to the config so the renderer, the trajectories and the GT
     voxels all use one frame.
  3. ``voxelize``  -- surface voxels by an EXACT triangle/box overlap test
     (Akenine-Moller 13-axis SAT), on the SAME grid convention the mapper uses:

         index  i = floor(p / res)            (grid anchored at world origin)
         centre   = (i + 0.5) * res

     This is the convention of ``covor.occupancy._dda_batch`` /
     ``OccupancyBuilder.integrate_frame``; it is re-asserted in
     tests/synth/test_grid_convention.py so the two cannot drift apart.

Why SAT and not ``trimesh.voxelized`` / open3d's ``create_from_triangle_mesh``:
those subdivide or sample and each anchors its own grid. Both would leave the GT
half a voxel off the map's grid -- exactly the failure Part C is meant to catch,
so the GT must not share it.
"""
import json
import os

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# 1. inspection
# ---------------------------------------------------------------------------
def load(path, process=False):
    """Load an OBJ as ONE Trimesh. process=False keeps the vertices as authored
    (merging them would silently change the surface a photogrammetry scan has)."""
    return trimesh.load(path, process=process, force="mesh")


def _plane_modes(mesh, axis, tol_normal=0.9, bins=200):
    """(centre, area) of the dominant planes perpendicular to ``axis``.

    Area-weighted histogram of the face centres whose normal is within
    ``tol_normal`` of the axis -- the floor and the ceiling are the two biggest.
    """
    N, A, C = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    m = np.abs(N[:, axis]) > tol_normal
    if not m.any():
        return np.empty((0, 2))
    h, edges = np.histogram(C[m, axis], bins=bins, weights=A[m])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    o = np.argsort(h)[::-1]
    return np.stack([ctr[o], h[o]], axis=1)


def inspect(mesh):
    """Everything Part A has to decide before anything else may run."""
    N, A = mesh.face_normals, mesh.area_faces
    ext = mesh.extents
    up = int(np.argmin(ext))          # a room is wider than it is tall
    out = dict(
        n_vertices=int(len(mesh.vertices)), n_faces=int(len(mesh.faces)),
        bounds=mesh.bounds.tolist(), extents=ext.tolist(),
        area=float(A.sum()),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        up_axis_guess="xyz"[up],
        axis_normal_area={
            "xyz"[a]: dict(pos=float(A[N[:, a] > 0.95].sum()),
                           neg=float(A[N[:, a] < -0.95].sum()))
            for a in range(3)},
        planes={"xyz"[a]: _plane_modes(mesh, a)[:5].tolist() for a in range(3)},
    )
    # floor / ceiling along the presumed up-axis: the two largest horizontal
    # planes, and which of them the surface normals face away from.
    pl = _plane_modes(mesh, up)
    if len(pl) >= 2:
        cand = pl[:12]
        lo = cand[cand[:, 0] < np.median(mesh.vertices[:, up])]
        hi = cand[cand[:, 0] >= np.median(mesh.vertices[:, up])]
        floor = float(lo[0, 0]) if len(lo) else float(mesh.bounds[0, up])
        ceil = float(hi[0, 0]) if len(hi) else float(mesh.bounds[1, up])
        out["floor_plane"] = floor
        out["ceiling_plane"] = ceil
        out["room_height"] = ceil - floor
        # normals of the floor plane must point AWAY from the floor (into the
        # room) for the scan to be an interior; that also fixes the sign of "up".
        C = mesh.triangles_center
        m = (np.abs(C[:, up] - floor) < 0.08) & (np.abs(N[:, up]) > 0.9)
        out["floor_normal_positive_area"] = float(A[m & (N[:, up] > 0)].sum())
        out["floor_normal_negative_area"] = float(A[m & (N[:, up] < 0)].sum())
        out["up_sign"] = 1 if out["floor_normal_positive_area"] >= \
            out["floor_normal_negative_area"] else -1
    return out


# ---------------------------------------------------------------------------
# 2. mesh -> world transform
# ---------------------------------------------------------------------------
def mesh_to_world(info, up_axis=None, up_sign=None):
    """4x4 taking mesh coordinates to the experiment world frame.

    z is up, the floor plane sits at z = 0, and the footprint bounding box is
    centred on (x, y) = (0, 0) -- the MILUV convention (room around the origin,
    floor near z = 0), so heights and z-slices mean the same thing in both.

    Rotation only permutes/negates axes, so it is exact: no interpolation, and
    the scan's own metric scale is preserved (scale = 1; the scan is metric --
    verified in Part A by the room height and footprint being metres, not
    centimetres or inches).
    """
    up = "xyz".index(up_axis or info["up_axis_guess"])
    sgn = float(up_sign if up_sign is not None else info.get("up_sign", 1))
    other = [a for a in range(3) if a != up]
    R = np.zeros((3, 3))
    R[2, up] = sgn                       # mesh up-axis -> world +z
    R[0, other[0]] = 1.0                 # first remaining axis -> world x
    R[1, other[1]] = 1.0                 # second -> world y
    if np.linalg.det(R) < 0:             # keep it a right-handed rotation
        R[1, other[1]] = -1.0
    T = np.eye(4)
    T[:3, :3] = R
    # floor plane -> z = 0
    T[2, 3] = -sgn * float(info["floor_plane"])
    # footprint bbox centred on (0,0): transform the mesh bbox corners
    b = np.array(info["bounds"], float)
    corners = np.array([[b[i, 0], b[j, 1], b[k, 2]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    w = corners @ R.T
    T[0, 3] = -0.5 * (w[:, 0].min() + w[:, 0].max())
    T[1, 3] = -0.5 * (w[:, 1].min() + w[:, 1].max())
    return T


def apply_transform(mesh, T):
    out = mesh.copy()
    out.apply_transform(T)
    return out


# ---------------------------------------------------------------------------
# 3. voxelisation (exact triangle/box SAT on the mapper's grid)
# ---------------------------------------------------------------------------
def _tri_box_overlap(V, centers, half):
    """Akenine-Moller separating-axis test, vectorised over N (triangle, box).

    V: (N,3,3) triangle vertices, centers: (N,3) box centres, half: scalar box
    half-size. Returns (N,) bool. All 13 axes: 3 box faces, 1 triangle normal,
    9 edge-cross-axis.
    """
    u = V - centers[:, None, :]                     # (N,3,3) box-centred
    h = half
    # Every test is INCLUSIVE to within a relative epsilon. Exact tangency --
    # a surface lying exactly on a voxel boundary plane -- is a knife edge in
    # floating point: |d| and r are then the same number computed two ways, and
    # a strict comparison drops the cell. Measured on an axis-aligned box placed
    # on the grid: 800 of the 3024 correct cells survived, whole faces missing,
    # while the same box shifted to a generic position voxelised correctly. A
    # scan is always in a generic position, so this never fires on the room --
    # but a CAD-like mesh would lose its walls silently.
    # (tests/test_synth_pipeline.py::test_voxelize_surface_matches_an_analytic_box)
    EPS = 1e-9
    # 3 box-face axes: triangle AABB vs box AABB
    sep = (u.min(1) > h + EPS) | (u.max(1) < -h - EPS)          # (N,3)
    out = ~np.any(sep, axis=1)
    idx = np.nonzero(out)[0]
    if idx.size == 0:
        return out
    u = u[idx]
    e = np.stack([u[:, 1] - u[:, 0], u[:, 2] - u[:, 1], u[:, 0] - u[:, 2]], 1)
    # 1 triangle-normal axis
    n = np.cross(e[:, 0], e[:, 1])
    d = np.einsum("ni,ni->n", n, u[:, 0])
    r = h * np.abs(n).sum(1)
    ok = np.abs(d) <= r * (1 + EPS) + EPS
    # 9 edge-cross-axis tests
    for i in range(3):
        for j in range(3):
            b = np.zeros(3)
            b[j] = 1.0
            a = np.cross(e[:, i], b)
            p = np.einsum("nij,nj->ni", u, a)        # (N,3) projections
            rr = h * np.abs(a).sum(1) * (1 + EPS) + EPS
            ok &= ~((p.min(1) > rr) | (p.max(1) < -rr))
    out[idx] = ok
    return out


def voxelize_surface(mesh, res, chunk=40000):
    """Occupied voxel indices (M,3) int64: every cell the surface passes through.

    Grid convention i = floor(p/res), centre (i+0.5)*res -- identical to
    covor.occupancy. Exact (SAT), so a cell is occupied iff a triangle really
    intersects that cube: no dilation, no sampling gaps.
    """
    Vt = mesh.triangles                                 # (F,3,3)
    keys = set()
    for s in range(0, len(Vt), chunk):
        V = Vt[s:s + chunk]
        # The candidate box is widened by an epsilon before flooring, for the
        # same exact-tangency reason as the SAT epsilon: a triangle lying ON a
        # voxel boundary plane has min == max == that plane, so flooring picks
        # only the cell on one side and the cell on the other is never even
        # offered to the overlap test.
        lo = np.floor((V.min(1) - 1e-9) / res).astype(np.int64)  # (n,3)
        hi = np.floor((V.max(1) + 1e-9) / res).astype(np.int64)
        span = hi - lo + 1                              # cells per axis
        cnt = span.prod(1)
        tri = np.repeat(np.arange(len(V)), cnt)
        # per-candidate offset within each triangle's cell box
        off = np.concatenate([np.arange(c) for c in cnt]) if len(cnt) else \
            np.zeros(0, np.int64)
        sx, sy, sz = span[tri, 0], span[tri, 1], span[tri, 2]
        ix = lo[tri, 0] + off // (sy * sz)
        iy = lo[tri, 1] + (off // sz) % sy
        iz = lo[tri, 2] + off % sz
        idx = np.stack([ix, iy, iz], 1)
        ctr = (idx + 0.5) * res
        hit = _tri_box_overlap(V[tri], ctr, res / 2.0)
        k = idx[hit]
        keys.update(map(tuple, k.tolist()))
    if not keys:
        return np.empty((0, 3), np.int64)
    return np.array(sorted(keys), dtype=np.int64)


# ---------------------------------------------------------------------------
# dense grid + interior (free-space) mask
# ---------------------------------------------------------------------------
UNKNOWN, FREE, OCC = 0, 1, 2


def grid_bounds(occ_idx, pad=2):
    lo = occ_idx.min(0) - pad
    hi = occ_idx.max(0) + pad
    return lo, tuple((hi - lo + 1).tolist())


def _fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], 1).astype(np.float32)


def raycasting_scene(mesh):
    """open3d RaycastingScene for a trimesh mesh.

    NOTE the import trap in this package's docstring: open3d must not be the
    first of {open3d, gtsam} imported in a process.
    """
    import open3d as o3d
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tm))
    return sc


def enclosure_fraction(scene, points, n_dirs=128, chunk=20000):
    """Fraction of ``n_dirs`` uniform rays from each point that hit the mesh.

    This replaces an inside/outside test. A photogrammetry scan is NOT
    watertight (this one: 2005 surface components, euler 1790), so winding
    number, ray parity and signed distance are all undefined on it -- and a plain
    flood fill leaks through the first hole in a wall (measured here: it filled
    313,374 of 313,423 non-surface cells, i.e. inside and outside became one
    component). Enclosure degrades gracefully instead: a hole subtends a small
    solid angle, so a cell inside the room still sees geometry in ~all
    directions, while a cell behind a wall or above the ceiling sees open sky in
    half of them.
    """
    import open3d as o3d
    D = _fibonacci_dirs(n_dirs)
    P = np.asarray(points, np.float32)
    out = np.zeros(len(P), np.float32)
    for s in range(0, len(P), chunk):
        q = P[s:s + chunk]
        rays = np.empty((len(q), n_dirs, 6), np.float32)
        rays[:, :, :3] = q[:, None, :]
        rays[:, :, 3:] = D[None, :, :]
        t = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy()
        out[s:s + chunk] = np.isfinite(t.reshape(len(q), n_dirs)).mean(1)
    return out


def build_gt_grid(occ_idx, res, seed_xyz, scene, thresh=0.90, n_dirs=128, pad=2,
                  z_band=None, close_iter=2,
                  sweep=(0.80, 0.85, 0.90, 0.95)):
    """Dense GT label grid.

    Returns (labels (nx,ny,nz) uint8, ijk_min (3,), info, enclosure), with
        world index = ijk_min + array index,  centre = (index + 0.5) * res.

    OCC   = a surface cell (from voxelize_surface), wherever it is.
    FREE  = a non-surface cell that is
            (a) enclosed by the scan: enclosure_fraction >= thresh;
            (b) inside the floor/ceiling band ``z_band`` (both planes are
                measured, so the vertical bounds are certain in a way the
                lateral ones are not);
            (c) after a binary closing of (a) by ``close_iter`` voxels, which
                fills the pinholes the scan's own floor gaps punch in the
                enclosure field (a cell under a desk loses its downward rays
                through a hole in the floor, not because it is outside the
                room). Closing cannot bleed through a wall: the wall cells are
                removed by (d) and the outside shell then falls off the seed
                component;
            (d) 6-connected to ``seed_xyz`` through cells satisfying (a)-(c).
                This removes sealed pockets -- the inside of a cabinet is
                "enclosed" but is not the room.
    UNKNOWN = everything else: behind the walls, above the ceiling, the volume
            past the scan's open side, and any pocket the fill cannot reach.
            EXCLUDED from every free/occupied metric rather than counted as
            free: only cells whose truth we know may score the map.

    THE OPEN SIDE. This scan is not a closed room -- it is open around
    x < -4 m -- so "where the room ends" laterally is a decision, not a
    measurement, and ``thresh`` is where it is made. ``sweep`` records what FREE
    would contain at other thresholds so that dependence is reported and not
    hidden (RESULTS_SUMMARY.md §9-4), and the trajectories are designed to stay
    well inside the domain so no result rests on the boundary.
    """
    from scipy import ndimage
    lo, shape = grid_bounds(occ_idx, pad)
    lab = np.zeros(shape, np.uint8)
    o = occ_idx - lo
    lab[o[:, 0], o[:, 1], o[:, 2]] = OCC
    cand = np.argwhere(lab != OCC)
    ctr = (cand + lo + 0.5) * res
    frac = enclosure_fraction(scene, ctr, n_dirs)
    E = np.full(shape, -1.0, np.float32)
    E[cand[:, 0], cand[:, 1], cand[:, 2]] = frac

    s = np.floor(np.asarray(seed_xyz, float) / res).astype(np.int64) - lo
    if np.any(s < 0) or np.any(s >= np.array(shape)):
        raise ValueError("seed %s outside the grid" % (seed_xyz,))
    if lab[tuple(s)] == OCC:
        raise ValueError("seed lands in an occupied cell")

    zc = (np.arange(shape[2]) + lo[2] + 0.5) * res
    zmask = np.ones(shape, bool)
    if z_band is not None:
        zmask[:, :, ~((zc > z_band[0]) & (zc < z_band[1]))] = False
    conn = ndimage.generate_binary_structure(3, 1)

    def component(th):
        m = E >= th
        if close_iter:
            m = ndimage.binary_closing(m, structure=conn, iterations=close_iter)
        m &= zmask & (lab != OCC)
        cc, n = ndimage.label(m, structure=conn)
        cid = int(cc[tuple(s)])
        return ((cc == cid) if cid else np.zeros(shape, bool)), n

    interior, ncomp = component(thresh)
    lab[interior] = FREE
    b = np.zeros(shape, bool)
    b[0] = b[-1] = True
    b[:, 0] = b[:, -1] = True
    b[:, :, 0] = b[:, :, -1] = True
    info = dict(
        thresh=float(thresh), n_dirs=int(n_dirs), close_iter=int(close_iter),
        z_band=(list(z_band) if z_band is not None else None),
        n_components=int(ncomp), touches_boundary=bool((interior & b).any()),
        n_occ=int((lab == OCC).sum()), n_free=int(interior.sum()),
        n_unknown=int((lab == UNKNOWN).sum()),
        free_vs_thresh={str(t): int(component(t)[0].sum()) for t in sweep})
    return lab, lo, info, E


def save_gt(path, lab, ijk_min, res, T_mesh_world, info, enclosure=None):
    d = dict(labels=lab, ijk_min=ijk_min, res=res, T_mesh_world=T_mesh_world,
             info=json.dumps(info))
    if enclosure is not None:
        d["enclosure"] = enclosure.astype(np.float32)
    np.savez_compressed(path, **d)


def load_gt(path, with_enclosure=False):
    z = np.load(path, allow_pickle=False)
    out = (z["labels"], z["ijk_min"], float(z["res"]), z["T_mesh_world"],
           json.loads(str(z["info"])))
    return out + (z["enclosure"],) if with_enclosure else out


def centers_of(lab, ijk_min, res, value):
    idx = np.argwhere(lab == value) + ijk_min
    return (idx + 0.5) * res, idx


def index_of(points, res):
    """World points -> voxel indices, the mapper's convention."""
    return np.floor(np.asarray(points, float) / res).astype(np.int64)


def write_config(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

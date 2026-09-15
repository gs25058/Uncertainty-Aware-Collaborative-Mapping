"""Per-column pose uncertainty and observation count (DESIGN_entry_map.md §3-3).

sigma_xy(x, y) is the MEDIAN, over the frames that put OCCUPIED evidence into
column (x, y), of sqrt(lambda_max(Sigma_xy)) for that frame's fused pose. It is
what turns DESIGN §3-3's "widen the body by how badly we know where we are" into
a number, and it is the only new quantity Phase 4 needs that the existing
pipeline does not already record.

WHY THE ENDPOINT RULE IS REPLICATED HERE. Neither existing record can answer
"which frames wrote occupied evidence into this column":

  * covor/synth/mapping.py's ``_touched`` is a SET of voxel indices, pooled over
    every frame and every robot, and it does not distinguish free evidence from
    occupied.
  * covor/occupancy.py's ``_attr`` keeps one evidence-weighted mean tr(Sigma) per
    cell -- a scalar, already averaged over frames, mixing free and occupied
    evidence, and tr(Sigma) rather than Sigma_xy.

Extending either would mean editing covor/occupancy.py or covor/synth/, which is
out of bounds. So ``occupied_endpoint_voxels`` recomputes the endpoint rule from
the same public inputs integrate_frame is given. That is a duplicated rule and
therefore a drift risk, which is why tests/test_entry_dump.py pins it against the
builder itself: run the builder with l_free = 0, so free evidence contributes
exactly 0, and every cell it writes with a positive value is an occupied
endpoint. Measured across 3 robots and 15 frames: identical, cell for cell.
"""
import numpy as np


def sigma_xy_from_cov(Sigma):
    """sqrt(lambda_max(Sigma_xy)) in metres, from a 3x3 position covariance.

    The largest principal standard deviation of the horizontal block, i.e. the
    worst direction. Taking the trace instead would mix the vertical error in,
    and taking the mean of the two horizontal axes would under-report an error
    that is large along one of them -- which is the anchor-free case.
    """
    S = np.asarray(Sigma, float).reshape(3, 3)[:2, :2]
    lam = np.linalg.eigvalsh(0.5 * (S + S.T))
    return float(np.sqrt(max(lam[-1], 0.0)))


def occupied_endpoint_voxels(T_wc, P_cam, teammates, occ_cfg):
    """Voxels that receive OCCUPIED evidence from this frame.

    A line-for-line replica of the endpoint half of
    covor.occupancy.OccupancyBuilder.integrate_frame: transform into world, drop
    beams shorter than a voxel or longer than max_ray, drop endpoints that landed
    on a teammate's airframe, floor-divide by the resolution. Nothing here
    depends on the weights, so sigma_Z is not needed.
    """
    res = occ_cfg.resolution
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    P = np.asarray(P_cam, float) @ R.T + t
    L = np.linalg.norm(P - t, axis=1)
    P = P[(L > res) & (L <= occ_cfg.max_ray)]
    if len(P) == 0:
        return np.empty((0, 3), np.int64)
    if teammates is not None and len(teammates) and occ_cfg.dyn_radius > 0:
        near = np.linalg.norm(P[:, None, :] - np.atleast_2d(teammates)[None, :, :],
                              axis=2).min(axis=1) < occ_cfg.dyn_radius
        P = P[~near]
    return np.floor(P / res).astype(np.int64)


class ColumnRecorder:
    """Tree proxy counting, per frame, which COLUMNS received any evidence.

    Wraps the OcTree the way covor/synth/mapping._RecordingTree does -- the
    builder is never edited, and it cannot tell the difference. Columns rather
    than cells because the entry map's "observation count" is per column, and
    because one set of ~10^3 columns per frame is a thousandth of the memory that
    keeping the cells would cost.
    """

    def __init__(self, tree, res):
        object.__setattr__(self, "_t", tree)
        object.__setattr__(self, "_r", float(res))
        object.__setattr__(self, "cols", set())

    def updateNode(self, ctr, val, lazy):
        r = self._r
        self.cols.add((int(np.floor(ctr[0] / r)), int(np.floor(ctr[1] / r))))
        return self._t.updateNode(ctr, val, lazy)

    def __getattr__(self, k):
        return getattr(self._t, k)


class ColumnEvidence:
    """Accumulates n_obs and the per-column sigma_xy samples over a whole run."""

    def __init__(self):
        self.n_obs = {}          # (ix, iy) -> frames that wrote anything here
        self.sigma = {}          # (ix, iy) -> [sigma_xy of each occupied frame]

    def add_frame(self, columns, endpoint_voxels, sigma_xy):
        for c in columns:
            self.n_obs[c] = self.n_obs.get(c, 0) + 1
        if len(endpoint_voxels) == 0:
            return
        # one sample per (frame, column): a frame that hits a column with 400
        # rays is still one observation of it, and counting rays would make the
        # median a function of viewing angle rather than of pose uncertainty.
        for c in set(map(tuple, endpoint_voxels[:, :2].tolist())):
            self.sigma.setdefault(c, []).append(sigma_xy)

    def dense(self, shape, ijk_min):
        """(n_obs int32, sigma_xy float32) on the (nx, ny) grid of ``shape``.

        Columns with no occupied evidence get sigma_xy = 0, i.e. no extra margin.
        That is the honest default: sigma_xy is a statement about an obstacle
        whose position is uncertain, and where there is no obstacle there is
        nothing to widen. inflate.radius_map is what decides how it is used.
        """
        n = np.zeros(shape, np.int32)
        s = np.zeros(shape, np.float32)
        lo = np.asarray(ijk_min[:2], np.int64)
        for (ix, iy), v in self.n_obs.items():
            i, j = ix - lo[0], iy - lo[1]
            if 0 <= i < shape[0] and 0 <= j < shape[1]:
                n[i, j] = v
        for (ix, iy), vs in self.sigma.items():
            i, j = ix - lo[0], iy - lo[1]
            if 0 <= i < shape[0] and 0 <= j < shape[1]:
                s[i, j] = float(np.median(vs))
        return n, s
